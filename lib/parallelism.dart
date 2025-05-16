/// A helper library that makes it easier to work with ReceivePorts and SendPorts.
///   It lets receive ports be used declaratively.
///   - For example, you can use the [next] method to get the next message from the receive port.
///   - You can also use the [call] method to get the next message from the receive port.
library;

import "dart:async";
import "dart:collection";
import "dart:isolate";

extension type ListenedReceivePort._(ReceivePort _port) {
  ListenedReceivePort(this._port, FutureOr<void> Function(Object? message)? fallbackListener) {
    _hosts[_port] = true;
    _fallbackListener = fallbackListener;
    _completer = Queue();
    _queue = Queue();

    assert(!_port.isBroadcast, "The receive port must not be a broadcast stream.");

    _port.listen((message) async {
      /// We add the message to the queue.
      _queue.add(message);

      /// If there is a completer waiting for a message, we complete it.
      if (_completers[_port]?.firstOrNull case var completer?) {
        completer.complete(message);
        _queue.removeLast();

        return;
      }

      if (_fallbackListener case var fallbackListener?) {
        _completers[_port] = null;
        await fallbackListener(message);
        _queue.removeLast();

        return;
      }
    });
  }

  /// This has the value of true for all ReceivePorts that can be listened to.
  ///   Since the ReceivePort is listened to, it can be used with the [next] extension method.
  ///   However, they cannot be listened to again.
  static final Expando<bool> _hosts = Expando();

  /// A map of [ReceivePort]s with their fallback listeners.
  static final Expando<void Function(Object? message)> _fallbackListeners = Expando();
  static final Expando<Queue<Completer<Object?>>> _completers = Expando();
  static final Expando<Queue<Object?>> _queues = Expando();

  // Pseudo-fields. These are used to store values specific to each [ReceivePort] instance.
  FutureOr<void> Function(Object?)? get _fallbackListener => _fallbackListeners[_port];
  set _fallbackListener(FutureOr<void> Function(Object?)? listener) =>
      _fallbackListeners[_port] = listener;

  Queue<Completer<Object?>> get _completer => _completers[_port]!;
  set _completer(Queue<Completer<Object?>> completer) => _completers[_port] = completer;

  Queue<Object?> get _queue => _queues[_port]!;
  set _queue(Queue<Object?> queue) => _queues[_port] = queue;

  Future<T> next<T>() async {
    if (_queue.isNotEmpty) {
      return _queue.removeFirst() as T;
    }

    var completer = Completer<void>();
    _completer.addLast(completer);
    var rawValue = await completer.future as Object?;
    assert(
      rawValue is T,
      "The value received from the [ReceivePort] must be of type $T. "
      "Got ${rawValue.runtimeType} instead",
    );
    var value = rawValue as T;
    _completer.removeFirst();

    return value;
  }

  Future<T> call<T>() => next<T>();

  /// Redirects all the messages received by the [ReceivePort] to the [sendPort].
  void redirectMessagesTo(SendPort sendPort) {
    _fallbackListeners[_port] = (message) {
      sendPort.send(message);
    };
  }

  /// Closes the [ReceivePort] and removes all the listeners.
  void close() {
    _hosts[_port] = null;
    _fallbackListeners[_port] = null;
    _completers[_port] = null;
    _port.close();
  }

  /// A [SendPort] which sends messages to this receive port.
  SendPort get sendPort => _port.sendPort;
}

extension ReceivePortExtension on ReceivePort {
  ListenedReceivePort hostListener([FutureOr<void> Function(Object?)? fallbackListener]) =>
      ListenedReceivePort(this, fallbackListener);
}
