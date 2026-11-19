import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'classifier.dart';
import 'leaf_info.dart';

void main() => runApp(const BeanLeafApp());

class BeanLeafApp extends StatelessWidget {
  const BeanLeafApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Bean Leaf CNN',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF2E7D32),
        useMaterial3: true,
      ),
      home: const ClassifierPage(),
    );
  }
}

class ClassifierPage extends StatefulWidget {
  const ClassifierPage({super.key});

  @override
  State<ClassifierPage> createState() => _ClassifierPageState();
}

class _ClassifierPageState extends State<ClassifierPage> {
  final _picker = ImagePicker();
  BeanLeafClassifier? _classifier;
  Uint8List? _imageBytes;
  List<Prediction>? _results;
  bool _busy = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  @override
  void dispose() {
    _classifier?.close();
    super.dispose();
  }

  Future<void> _loadModel() async {
    try {
      final classifier = await BeanLeafClassifier.load();
      if (!mounted) {
        classifier.close();
        return;
      }
      setState(() => _classifier = classifier);
    } catch (e) {
      if (mounted) setState(() => _error = 'Hindi ma-load ang CNN model: $e');
    }
  }

  Future<void> _pickAndClassify(ImageSource source) async {
    final classifier = _classifier;
    if (classifier == null || _busy) return;
    try {
      final file = await _picker.pickImage(
        source: source,
        maxWidth: 640,
        maxHeight: 640,
        imageQuality: 95,
      );
      if (file == null) return;
      final bytes = await file.readAsBytes();
      setState(() {
        _busy = true;
        _imageBytes = bytes;
        _results = null;
        _error = null;
      });
      final results = await classifier.classify(bytes);
      if (mounted) setState(() => _results = results);
    } catch (e) {
      if (mounted) setState(() => _error = 'May error: $e');
    } finally {
      if (mounted && _busy) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final ready = _classifier != null && !_busy;
    final results = _results;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Bean Leaf CNN Classifier'),
        centerTitle: true,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            _ImagePreview(bytes: _imageBytes, busy: _busy),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: ready ? () => _pickAndClassify(ImageSource.camera) : null,
                    icon: const Icon(Icons.photo_camera),
                    label: const Text('Camera'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.tonalIcon(
                    onPressed: ready ? () => _pickAndClassify(ImageSource.gallery) : null,
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                  ),
                ),
              ],
            ),
            if (_classifier == null && _error == null) ...[
              const SizedBox(height: 16),
              const Center(child: Text('Nilo-load ang CNN model...')),
            ],
            if (_error != null) ...[
              const SizedBox(height: 16),
              Card(
                color: theme.colorScheme.errorContainer,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Text(
                    _error!,
                    style: TextStyle(color: theme.colorScheme.onErrorContainer),
                  ),
                ),
              ),
            ],
            if (results != null) ...[
              const SizedBox(height: 16),
              _ResultCard(results: results),
            ],
            const SizedBox(height: 24),
            Text(
              'CNN model na na-train sa 1,000 larawan ng dahon ng bean '
              '(Healthy, Angular Leaf Spot, Bean Rust). Tumatakbo offline sa phone.',
              style: theme.textTheme.bodySmall,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}

class _ImagePreview extends StatelessWidget {
  const _ImagePreview({required this.bytes, required this.busy});

  final Uint8List? bytes;
  final bool busy;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final image = bytes;
    return AspectRatio(
      aspectRatio: 1,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: Stack(
          fit: StackFit.expand,
          children: [
            if (image == null)
              ColoredBox(
                color: scheme.surfaceContainerHighest,
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.eco, size: 72, color: scheme.primary),
                    const SizedBox(height: 12),
                    const Padding(
                      padding: EdgeInsets.symmetric(horizontal: 24),
                      child: Text(
                        'Kumuha o pumili ng malinaw na larawan ng isang dahon ng bean',
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ],
                ),
              )
            else
              Image.memory(image, fit: BoxFit.cover),
            if (busy)
              const ColoredBox(
                color: Color(0x66000000),
                child: Center(child: CircularProgressIndicator()),
              ),
          ],
        ),
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.results});

  final List<Prediction> results;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final top = results.first;
    final info = infoFor(top.label);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(info.icon, color: info.color, size: 36),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        info.displayName,
                        style: theme.textTheme.headlineSmall?.copyWith(
                          color: info.color,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text('${(top.confidence * 100).toStringAsFixed(1)}% confidence'),
                    ],
                  ),
                ),
              ],
            ),
            if (top.confidence < 0.6) ...[
              const SizedBox(height: 8),
              Text(
                'Hindi masyadong sigurado ang model. Subukan ang mas malinaw na larawan '
                'ng isang dahon, na may maliwanag na ilaw.',
                style: TextStyle(color: theme.colorScheme.error),
              ),
            ],
            const Divider(height: 28),
            for (final r in results) _ScoreBar(prediction: r),
            const Divider(height: 28),
            if (info.description.isNotEmpty) Text(info.description),
            if (info.advice.isNotEmpty) ...[
              const SizedBox(height: 8),
              Text(
                'Payo: ${info.advice}',
                style: const TextStyle(fontWeight: FontWeight.w600),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _ScoreBar extends StatelessWidget {
  const _ScoreBar({required this.prediction});

  final Prediction prediction;

  @override
  Widget build(BuildContext context) {
    final info = infoFor(prediction.label);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(child: Text(info.displayName)),
              Text('${(prediction.confidence * 100).toStringAsFixed(1)}%'),
            ],
          ),
          const SizedBox(height: 4),
          LinearProgressIndicator(
            value: prediction.confidence,
            minHeight: 8,
            color: info.color,
            borderRadius: BorderRadius.circular(4),
          ),
        ],
      ),
    );
  }
}
