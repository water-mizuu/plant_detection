import "dart:async";
import "dart:io";
import "dart:isolate";
import "dart:math";

import "package:flutter/foundation.dart";
import "package:flutter/material.dart";
import "package:image/image.dart" as img;
import "package:image_picker/image_picker.dart";
import "package:permission_handler/permission_handler.dart";
import "package:plant_recognition/parallelism.dart";
import "package:tflite_flutter/tflite_flutter.dart";

const classLabels = {
  0: "Acacia mangium",
  1: "Broussonetia papyrifera",
  2: "Chromolaena Odarata",
  3: "Clidemia hirta",
  4: "Eichhornia crassipes",
  5: "Hiptage Benghalensis",
  6: "Imperata cylindrical",
  7: "Lantana camara",
  8: "Leucaena leucocephala",
  9: "Mesophaerum suaveolens",
  10: "Mikania micrantha Kunth",
  11: "Mimosa Pigra",
  12: "Panicum Maximum",
  13: "Piper Aduncum",
  14: "Prosopis juliflora",
  15: "Pueraria montana",
  16: "Salvinia molesta",
  17: "Senna Obtusifolia",
  18: "Solanum torvum",
  19: "Spathodea campanulata Beauverd",
  20: "Tithonia diversifolia",
  21: "Urochloa mutica",
  22: "Wedelia trilobata",
  23: "Not a plant",
};

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "TFLite Image Processor",
      theme: ThemeData(
        primarySwatch: Colors.blue,
        brightness: Brightness.light,
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        primarySwatch: Colors.blue,
        brightness: Brightness.dark,
        useMaterial3: true,
      ),
      themeMode: ThemeMode.system,
      home: const SplashScreen(),
    );
  }
}

class IsolateChannel {
  final SendPort sendPort;
  final ListenedReceivePort receivePort;

  const IsolateChannel({required this.sendPort, required this.receivePort});

  Future<Object?> invoke(Object? parameters) async {
    sendPort.send(parameters);

    return await receivePort.next();
  }
}

class ImageProcessIsolate {
  final Completer<void> _setupCompleter;
  Future<void> get setupFuture => _setupCompleter.future;

  Isolate? isolate;
  ListenedReceivePort? receivePort;
  SendPort? sendPort;
  IsolateChannel? _channel;
  IsolateChannel get channel {
    if (_channel case var channel?) {
      return channel;
    }
    throw UnsupportedError("Isolate not loaded yet!");
  }

  ImageProcessIsolate() : _setupCompleter = Completer<void>.sync() {
    _initialize();
  }

  void dispose() {
    sendPort?.send("close");
  }

  Future<void> _initialize() async {
    /// Load the interpreter from the main isolate.
    ///   This doesn't work in the isolate, as it throws an error
    ///   relating to
    final (interpreter, error) = await _loadInterpreter();
    if (error case ["exception", final error?, StackTrace stackTrace?]) {
      _setupCompleter.completeError(error, stackTrace);
      return;
    }

    /// Create a receive port to communicate with the created isolate.
    final receivePort = this.receivePort = ReceivePort().hostListener();

    /// Spawn the isolate, and pass the send port and the reciever.
    ///   Establish two-way communication between the main isolate and the spawned isolate.
    isolate = await Isolate.spawn(_imageClassificationIsolate, receivePort.sendPort);
    final sendPort = this.sendPort = await receivePort.next<SendPort>();

    /// Send the interpreter to the isolate.
    ///   As a response, we expect a confirmation message.
    sendPort.send(interpreter);
    var object = await receivePort.next<Object?>();
    if (interpreter case ["exception", [var error?, StackTrace stackTrace]]) {
      _setupCompleter.completeError(error, stackTrace);
      return;
    } else if (interpreter case ["exception", ...]) {
      if (kDebugMode) {
        print("Unexpected error format: $interpreter");
      }
      _setupCompleter.completeError("Unexpected error format: $interpreter");
      return;
    } else {
      if (kDebugMode) {
        print(object);
      }
    }

    _channel = IsolateChannel(receivePort: receivePort, sendPort: sendPort);

    return _setupCompleter.complete();
  }

  Future<Uint8List> cropToSquare(Uint8List imageData) async {
    final result = await channel.invoke(["cropToSquare", imageData]);
    if (result case Uint8List cropped) {
      return cropped;
    } else if (result case ["exception", final error]) {
      throw Exception("Error in isolate: $error");
    } else {
      throw Exception("Unexpected result format in cropToSquare: $result");
    }
  }

  Future<List<double>> classify(Uint8List imageData) async {
    final result = await channel.invoke(["classify", imageData]);
    if (result case [List<double> classification]) {
      return classification;
    } else if (result case ["exception", final error]) {
      throw Exception("Error in isolate: $error");
    } else {
      throw Exception("Unexpected result format in classify: $result");
    }
  }

  static Future<(Interpreter?, Object?)> _loadInterpreter() async {
    // Load the model
    try {
      final interpreter = await Interpreter.fromAsset("assets/model.tflite")
        ..allocateTensors();

      return (interpreter, null);
    } catch (e, stackTrace) {
      if (kDebugMode) {
        print("Error loading interpreter: $e");
        print(stackTrace);
      }
      return (null, ["exception", e, stackTrace]);
    }
  }

  static Future<List<List<double>>> _classifyImage(
    Interpreter interpreter,
    Uint8List imageData,
  ) async {
    final image = img.decodeImage(imageData);

    if (image == null) {
      throw Exception("Failed to decode image");
    }

    final size = min(image.width, image.height);
    final croppedImage = switch (true) {
      _ when size == image.width && size == image.height => image,
      _ when size == image.width => img.copyCrop(image, 0, (image.height - size) ~/ 2, size, size),
      _ when size == image.height => img.copyCrop(image, (image.width - size) ~/ 2, 0, size, size),
      _ => throw Error(),
    };

    // Resize the image to 224x224
    final resizedImage = img.copyResize(
      croppedImage,
      width: 224,
      height: 224,
      interpolation: img.Interpolation.average,
    );

    // Convert to buffer
    final inputBuffer = Float32List(1 * 224 * 224 * 3);
    var pixelIndex = 0;

    for (var y = 0; y < 224; y++) {
      for (var x = 0; x < 224; x++) {
        final pixel = resizedImage.getPixel(x, y);

        /// BGR, as apparently that"s what tensorflow does.
        inputBuffer[pixelIndex++] = img.getBlue(pixel).toDouble();
        inputBuffer[pixelIndex++] = img.getGreen(pixel).toDouble();
        inputBuffer[pixelIndex++] = img.getRed(pixel).toDouble();
      }
    }

    // Reshape the buffer to match model input shape [1, 224, 224, 3]
    final input = inputBuffer.reshape([1, 224, 224, 3]);

    // Prepare output buffer (adjust size based on your model"s output)
    // For demonstration, assuming output is [1, 10] (10 classes)
    final output = [List<double>.filled(24, 0)];

    // Run inference
    interpreter.run(input, output);

    return output;
  }

  static Future<Uint8List> _cropImage(Uint8List imageData) async {
    img.Image? image = img.decodeImage(imageData);

    if (image == null) {
      throw Exception("Failed to decode image");
    }

    int size = min(image.width, image.height);
    img.Image croppedImage = switch (true) {
      _ when size == image.width && size == image.height => image,
      _ when size == image.width => img.copyCrop(image, 0, (image.height - size) ~/ 2, size, size),
      _ when size == image.height => img.copyCrop(image, (image.width - size) ~/ 2, 0, size, size),
      _ => throw Error(),
    };

    return Uint8List.fromList(img.encodeJpg(croppedImage));
  }

  static void _imageClassificationIsolate(SendPort sendPort) async {
    /// INITIALIZATION

    final receivePort = ReceivePort().hostListener();
    sendPort.send(receivePort.sendPort);
    if (kDebugMode) {
      print("Succesfully loaded isolate");
    }

    final interpreter = await receivePort.next<Interpreter>();
    sendPort.send(1);

    /// RUNNING LOOP
    outer:
    while (true) {
      var message = await receivePort.next<Object?>();

      switch (message) {
        case ["classify", Uint8List imageData]:
          // Process the image data
          var classification = await _classifyImage(interpreter, imageData);
          sendPort.send(classification);
          break;
        case ["cropToSquare", Uint8List imageData]:
          // Process the image data
          var cropped = await _cropImage(imageData);
          sendPort.send(cropped);
          break;
        case [null]:
          break outer;
      }
    }

    /// DISPOSE

    receivePort.close();
  }
}

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  bool _modelLoaded = false;
  bool _errorLoading = false;
  String _errorMessage = "";

  @override
  void initState() {
    super.initState();

    _requestPermissions();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      var isolate = ImageProcessIsolate();
      await isolate.setupFuture;
      if (!mounted) return;

      setState(() {
        _modelLoaded = true;
      });
      // Wait for 1 second to show the splash screen
      await Future.delayed(const Duration(seconds: 1));
      if (!mounted) return;

      // Navigate to the main screen
      var navigator = Navigator.of(context);
      navigator.pushReplacement(MaterialPageRoute(builder: (_) => MainScreen(isolate: isolate)));
    } catch (e) {
      setState(() {
        _errorLoading = true;
        _errorMessage = e.toString();
      });
    }
  }

  Future<void> _requestPermissions() async {
    // Request camera permission
    var cameraStatus = await Permission.camera.request();
    if (cameraStatus.isDenied) {
      throw Exception("Camera permission is required");
    }

    // Request storage permission for Android
    if (Platform.isAndroid) {
      var storageStatus = await Permission.manageExternalStorage.request();
      if (storageStatus.isDenied) {
        throw Exception("Storage permission is required");
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.blue.shade300, Colors.blue.shade700],
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.camera_alt, size: 80, color: Colors.white),
              const SizedBox(height: 20),
              const Text(
                "Herbalyzer 2.0",
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white),
              ),
              const SizedBox(height: 30),
              if (_errorLoading)
                Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Text(
                    "Error loading model: $_errorMessage",
                    style: const TextStyle(color: Colors.red),
                    textAlign: TextAlign.center,
                  ),
                )
              else
                const CircularProgressIndicator(color: Colors.white),
              const SizedBox(height: 20),
              Text(
                _modelLoaded ? "Model loaded successfully!" : "Loading model...",
                style: const TextStyle(color: Colors.white),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class MainScreen extends StatefulWidget {
  final ImageProcessIsolate isolate;

  const MainScreen({super.key, required this.isolate});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  late final ValueNotifier<String?> _loadingMessage;

  Uint8List? _imageData;
  bool _isProcessingData = false;
  List<double>? _prediction;
  final ImagePicker _picker = ImagePicker();

  Future<void> _getImage(ImageSource source) async {
    final context = this.context;

    try {
      _loadingMessage.value = "Picking image...";
      final pickedFile = await _picker.pickImage(source: source, imageQuality: 85);
      if (!mounted) return;

      /// When the user cancels
      if (pickedFile == null) {
        _loadingMessage.value = null;
        return;
      }

      _loadingMessage.value = "Reading image from storage...";
      final bytes = await File(pickedFile.path).readAsBytes();
      if (!mounted) return;

      _loadingMessage.value = "Cropping image to bytes...";
      final cropped = await widget.isolate.cropToSquare(bytes);
      if (!mounted) return;

      setState(() {
        _loadingMessage.value = null;
        _imageData = cropped;
        _prediction = null;
        _isProcessingData = true;
      });

      await _processImage(cropped);
    } catch (e) {
      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Error: ${e.toString()}")));
    }
  }

  Future<void> _processImage(Uint8List imageData) async {
    final context = this.context;
    try {
      final output = await widget.isolate.classify(imageData);

      setState(() {
        _prediction = output;
        _isProcessingData = false;
      });
    } catch (e, stackTrace) {
      if (kDebugMode) {
        print(e);
        print(stackTrace);
      }
      setState(() {
        _isProcessingData = false;
      });

      if (!context.mounted) return;

      ScaffoldMessenger.of(context) //
      .showSnackBar(SnackBar(content: Text("Processing error: ${e.toString()}")));
    }
  }

  @override
  void initState() {
    super.initState();

    _loadingMessage = ValueNotifier(null);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Herbalyzer 2.0"),
        centerTitle: true,
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.blue.shade50, Colors.white],
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Expanded(
                  child: SingleChildScrollView(
                    child: Column(
                      children: [
                        if (_imageData == null)
                          Container(
                            decoration: BoxDecoration(
                              color: Colors.grey.shade200,
                              borderRadius: BorderRadius.circular(16),
                            ),
                            child: const Center(
                              child: Text(
                                "No image selected",
                                style: TextStyle(fontSize: 18, color: Colors.grey),
                              ),
                            ),
                          )
                        else
                          ClipRRect(
                            borderRadius: BorderRadius.circular(16),
                            child: Image.memory(_imageData!, fit: BoxFit.cover),
                          ),
                        const SizedBox(height: 20),
                        if (_prediction?.cast<double>() case var prediction?)
                          Builder(
                            builder: (context) {
                              var topFive = (prediction.indexed.toList()
                                    ..sort((a, b) => b.$2.compareTo(a.$2)))
                                  .sublist(0, 5);
                              var (index, confidence) = topFive[0];
                              Widget widget;
                              if (confidence < 0.75) {
                                widget = Text(
                                  "WARNING: Top confidence from the image is low.",
                                  style: TextStyle(color: Colors.redAccent, fontSize: 16),
                                  textAlign: TextAlign.center,
                                );
                              } else {
                                widget = Text(
                                  "Top prediction: ${classLabels[index]}",
                                  style: const TextStyle(
                                    color: Colors.black,
                                    fontSize: 18,
                                    fontWeight: FontWeight.bold,
                                  ),
                                );
                              }

                              return Padding(padding: EdgeInsets.all(16.0), child: widget);
                            },
                          ),
                        ListenableBuilder(
                          listenable: _loadingMessage,
                          builder: (context, _) {
                            final value = _loadingMessage.value;
                            if (value == null) {
                              return const SizedBox.shrink();
                            }

                            return Center(
                              child: Column(
                                children: [
                                  CircularProgressIndicator(),
                                  Text(value, style: TextStyle(color: Colors.black)),
                                ],
                              ),
                            );
                          },
                        ),
                        if (_isProcessingData)
                          const Center(
                            child: Column(
                              children: [
                                CircularProgressIndicator(),
                                SizedBox(height: 10),
                                Text("Processing image...", style: TextStyle(color: Colors.black)),
                              ],
                            ),
                          )
                        else if (_prediction?.cast<double>() case var prediction?)
                          Card(
                            elevation: 4,
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                            child: Padding(
                              padding: const EdgeInsets.all(16.0),
                              child: Builder(
                                builder: (context) {
                                  var topFive = (prediction.indexed.toList()
                                        ..sort((a, b) => b.$2.compareTo(a.$2)))
                                      .sublist(0, 5);

                                  return Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      const Text(
                                        "Prediction Results:",
                                        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                                      ),
                                      const SizedBox(height: 10),
                                      ...List.generate(
                                        topFive.length,
                                        (index) => Padding(
                                          padding: const EdgeInsets.symmetric(vertical: 2),
                                          child: Row(
                                            children: [
                                              Expanded(
                                                child: Text(
                                                  "${classLabels[topFive[index].$1]}",
                                                  style: const TextStyle(
                                                    fontWeight: FontWeight.w500,
                                                  ),
                                                  overflow: TextOverflow.ellipsis,
                                                ),
                                              ),
                                              const SizedBox(width: 8),
                                              Expanded(
                                                child: LinearProgressIndicator(
                                                  value: (topFive[index].$2).clamp(0.0, 1.0),
                                                  backgroundColor: Colors.grey.shade200,
                                                  valueColor: AlwaysStoppedAnimation<Color>(
                                                    Colors.blue.shade700,
                                                  ),
                                                ),
                                              ),
                                              const SizedBox(width: 8),
                                              Text(
                                                "${(topFive[index].$2 * 100).toStringAsFixed(2)}%",
                                                style: const TextStyle(fontWeight: FontWeight.w500),
                                              ),
                                            ],
                                          ),
                                        ),
                                      ),
                                    ],
                                  );
                                },
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton.icon(
                        icon: const Icon(Icons.camera_alt),
                        label: const Text("Camera"),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.blue,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(vertical: 12),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                        onPressed: () => _getImage(ImageSource.camera),
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: ElevatedButton.icon(
                        icon: const Icon(Icons.photo_library),
                        label: const Text("Gallery"),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(vertical: 12),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                        onPressed: () => _getImage(ImageSource.gallery),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
