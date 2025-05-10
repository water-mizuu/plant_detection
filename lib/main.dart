import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

const classLabels = {
  0: 'Acacia mangium',
  1: 'Broussonetia papyrifera',
  2: 'Chromolaena Odarata',
  3: 'Clidemia hirta',
  4: 'Eichhornia crassipes',
  5: 'Hiptage Benghalensis',
  6: 'Imperata cylindrical',
  7: 'Lantana camara',
  8: 'Leucaena leucocephala',
  9: 'Mesophaerum suaveolens',
  10: 'Mikania micrantha Kunth',
  11: 'Mimosa Pigra',
  12: 'Panicum Maximum',
  13: 'Piper Aduncum',
  14: 'Prosopis juliflora',
  15: 'Pueraria montana',
  16: 'Salvinia molesta',
  17: 'Senna Obtusifolia',
  18: 'Solanum torvum',
  19: 'Spathodea campanulata Beauverd',
  20: 'Tithonia diversifolia',
  21: 'Urochloa mutica',
  22: 'Wedelia trilobata',
  23: 'Not a plant',
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
      title: 'TFLite Image Processor',
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

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  bool _modelLoaded = false;
  bool _errorLoading = false;
  String _errorMessage = "";
  late Interpreter _interpreter;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      // Request permissions first
      await _requestPermissions();

      // Load the model
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
      _interpreter.allocateTensors();
      setState(() {
        _modelLoaded = true;
      });

      // Wait for 1 second to show the splash screen
      await Future.delayed(const Duration(seconds: 1));

      // Navigate to the main screen
      if (mounted) {
        Navigator.of(context).pushReplacement(
          MaterialPageRoute(builder: (context) => MainScreen(interpreter: _interpreter)),
        );
      }
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
      throw Exception('Camera permission is required');
    }

    // Request storage permission for Android
    if (Platform.isAndroid) {
      var storageStatus = await Permission.manageExternalStorage.request();
      if (storageStatus.isDenied) {
        throw Exception('Storage permission is required');
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
                'TFLite Image Processor',
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white),
              ),
              const SizedBox(height: 30),
              if (_errorLoading)
                Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Text(
                    'Error loading model: $_errorMessage',
                    style: const TextStyle(color: Colors.red),
                    textAlign: TextAlign.center,
                  ),
                )
              else
                const CircularProgressIndicator(color: Colors.white),
              const SizedBox(height: 20),
              Text(
                _modelLoaded ? 'Model loaded successfully!' : 'Loading model...',
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
  final Interpreter interpreter;

  const MainScreen({super.key, required this.interpreter});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  File? _image;
  bool _isProcessing = false;
  List<double>? _prediction;
  final ImagePicker _picker = ImagePicker();

  Future<void> _getImage(ImageSource source) async {
    var context = this.context;

    try {
      final XFile? pickedFile = await _picker.pickImage(source: source, imageQuality: 85);

      if (pickedFile != null) {
        setState(() {
          _image = File(pickedFile.path);
          _prediction = null;
          _isProcessing = true;
        });

        await _processImage();
      }
    } catch (e) {
      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: ${e.toString()}')));
    }
  }

  Future<void> _processImage() async {
    var context = this.context;
    try {
      if (_image == null) return;

      // Read the image file
      final imageData = await _image!.readAsBytes();
      // final imageData = Uint8List.sublistView(await rootBundle.load("assets/lantana_camara.jpg"));
      img.Image? image = img.decodeImage(imageData);

      if (image == null) {
        throw Exception('Failed to decode image');
      }

      // Resize the image to 224x224
      img.Image resizedImage = img.copyResize(
        image,
        width: 224,
        height: 224,
        interpolation: img.Interpolation.average,
      );

      // Convert to buffer
      Float32List inputBuffer = Float32List(1 * 224 * 224 * 3);
      int pixelIndex = 0;

      for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
          final pixel = resizedImage.getPixel(x, y);

          /// BGR, as apparently that's what tensorflow does.
          inputBuffer[pixelIndex++] = img.getBlue(pixel).toDouble();
          inputBuffer[pixelIndex++] = img.getGreen(pixel).toDouble();
          inputBuffer[pixelIndex++] = img.getRed(pixel).toDouble();
        }
      }

      // Reshape the buffer to match model input shape [1, 224, 224, 3]
      final input = inputBuffer.reshape([1, 224, 224, 3]);

      // Prepare output buffer (adjust size based on your model's output)
      // For demonstration, assuming output is [1, 10] (10 classes)
      final output = [List<double>.filled(24, 0)];

      // Run inference
      widget.interpreter.run(input, output);

      setState(() {
        _prediction = output[0];
        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
      });

      if (!context.mounted) return;

      if (kDebugMode) {
        print(e);
      }
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Processing error: ${e.toString()}')));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('TFLite Image Processor'),
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
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: SingleChildScrollView(
                  child: Column(
                    children: [
                      if (_image == null)
                        Container(
                          decoration: BoxDecoration(
                            color: Colors.grey.shade200,
                            borderRadius: BorderRadius.circular(16),
                          ),
                          child: const Center(
                            child: Text(
                              'No image selected',
                              style: TextStyle(fontSize: 18, color: Colors.grey),
                            ),
                          ),
                        )
                      else
                        ClipRRect(
                          borderRadius: BorderRadius.circular(16),
                          child: Image.file(_image!, fit: BoxFit.cover),
                        ),
                      const SizedBox(height: 20),
                      if (_prediction?.cast<double>() case var prediction?)
                        Builder(
                          builder: (context) {
                            var topFive = (prediction.indexed.toList()
                                  ..sort((a, b) => b.$2.compareTo(a.$2)))
                                .sublist(0, 5);

                            if (topFive[0].$2 < 0.75) {
                              return const Padding(
                                padding: EdgeInsets.all(16.0),
                                child: Text(
                                  'WARNING: Top confidence from the image is low.',
                                  style: TextStyle(color: Colors.redAccent, fontSize: 16),
                                  textAlign: TextAlign.center,
                                ),
                              );
                            }
                            return const SizedBox.shrink();
                          },
                        ),
                      if (_isProcessing)
                        const Center(
                          child: Column(
                            children: [
                              CircularProgressIndicator(),
                              SizedBox(height: 10),
                              Text('Processing image...'),
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
                                      'Prediction Results:',
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
                                                '${classLabels[topFive[index].$1]}',
                                                style: const TextStyle(fontWeight: FontWeight.w500),
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
                                              (topFive[index].$2).toStringAsFixed(4),
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
                      label: const Text('Camera'),
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
                      label: const Text('Gallery'),
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
    );
  }
}
