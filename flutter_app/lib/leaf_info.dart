import 'package:flutter/material.dart';

/// Display text for each class the CNN can predict.
class LeafInfo {
  const LeafInfo({
    required this.displayName,
    required this.description,
    required this.advice,
    required this.color,
    required this.icon,
  });

  final String displayName;
  final String description;
  final String advice;
  final Color color;
  final IconData icon;
}

const Map<String, LeafInfo> leafInfo = {
  'angular_leaf_spot': LeafInfo(
    displayName: 'Angular Leaf Spot',
    description:
        'Sakit na dulot ng fungus (Pseudocercospora griseola). May mga kayumanggi o '
        'abong batik na hugis-anggulo dahil nililimitahan ito ng mga ugat ng dahon.',
    advice:
        'Alisin ang mga apektadong dahon, iwasang basain ang dahon kapag nagdidilig, '
        'at gumamit ng angkop na fungicide.',
    color: Color(0xFFE65100),
    icon: Icons.blur_on,
  ),
  'bean_rust': LeafInfo(
    displayName: 'Bean Rust',
    description:
        'Sakit na dulot ng fungus (Uromyces appendiculatus). May maliliit at bilog na '
        'pustules na kulay-kalawang (reddish-brown) sa dahon.',
    advice:
        'Mag-crop rotation, magtanim ng resistant variety, alisin ang may sakit na '
        'halaman, at gumamit ng fungicide kung kailangan.',
    color: Color(0xFF8D3B13),
    icon: Icons.coronavirus,
  ),
  'healthy': LeafInfo(
    displayName: 'Healthy',
    description: 'Walang nakitang sakit sa dahon.',
    advice:
        'Ipagpatuloy ang tamang pagdidilig at pag-aabono, at regular na i-check ang mga dahon.',
    color: Color(0xFF2E7D32),
    icon: Icons.check_circle,
  ),
};

LeafInfo infoFor(String label) =>
    leafInfo[label] ??
    LeafInfo(
      displayName: label,
      description: '',
      advice: '',
      color: Colors.grey,
      icon: Icons.help_outline,
    );
