import 'package:flutter_test/flutter_test.dart';

import 'package:bean_leaf_classifier/main.dart';

void main() {
  testWidgets('Home screen shows title and both buttons', (tester) async {
    await tester.pumpWidget(const BeanLeafApp());
    expect(find.text('Bean Leaf CNN Classifier'), findsOneWidget);
    expect(find.text('Camera'), findsOneWidget);
    expect(find.text('Gallery'), findsOneWidget);
  });
}
