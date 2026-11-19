import 'dart:isolate';
import 'dart:math' as math;

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// One class and how sure the CNN is about it (0.0 - 1.0).
class Prediction {
  const Prediction(this.label, this.confidence);

  final String label;
  final double confidence;
}

/// Loads the trained CNN (TensorFlow Lite) and classifies leaf photos on the phone.
class BeanLeafClassifier {
  BeanLeafClassifier._(this._interpreter, this.labels, this.inputSize);

  static const _modelAsset = 'assets/model/bean_cnn.tflite';
  static const _labelsAsset = 'assets/model/labels.txt';

  final Interpreter _interpreter;
  final List<String> labels;
  final int inputSize;

  static Future<BeanLeafClassifier> load() async {
    final data = await rootBundle.load(_modelAsset);
    final interpreter = Interpreter.fromBuffer(
      data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
    );
    final labelText = await rootBundle.loadString(_labelsAsset);
    final labels = labelText
        .split('\n')
        .map((line) => line.trim())
        .where((line) => line.isNotEmpty)
        .toList();
    final shape = interpreter.getInputTensor(0).shape; // [1, height, width, 3]
    return BeanLeafClassifier._(interpreter, labels, shape[1]);
  }

  /// Returns every class sorted from most to least likely.
  Future<List<Prediction>> classify(Uint8List imageBytes) async {
    final input = await _preprocessInBackground(imageBytes, inputSize);
    final output = [List<double>.filled(labels.length, 0.0)];
    _interpreter.run(input, output);

    final scores = output.first;
    final results = [
      for (var i = 0; i < labels.length; i++) Prediction(labels[i], scores[i]),
    ];
    results.sort((a, b) => b.confidence.compareTo(a.confidence));
    return results;
  }

  void close() => _interpreter.close();

  static Future<List<List<List<List<double>>>>> _preprocessInBackground(
    Uint8List bytes,
    int size,
  ) {
    return Isolate.run(() => preprocessImage(bytes, size));
  }
}

/// Same preprocessing as training: square image resized to [size] x [size],
/// RGB values kept in 0-255 (the model's first layer divides by 255).
List<List<List<List<double>>>> preprocessImage(Uint8List bytes, int size) {
  final decoded = img.decodeImage(bytes);
  if (decoded == null) {
    throw const FormatException('Hindi mabasa ang larawan.');
  }
  final oriented = img.bakeOrientation(decoded);
  final side = math.min(oriented.width, oriented.height);
  final square = img.copyCrop(
    oriented,
    x: (oriented.width - side) ~/ 2,
    y: (oriented.height - side) ~/ 2,
    width: side,
    height: side,
  );
  final resized = img.copyResize(
    square,
    width: size,
    height: size,
    interpolation: img.Interpolation.linear,
  );

  return [
    List.generate(
      size,
      (y) => List.generate(size, (x) {
        final pixel = resized.getPixel(x, y);
        return [pixel.r.toDouble(), pixel.g.toDouble(), pixel.b.toDouble()];
      }),
    ),
  ];
}
