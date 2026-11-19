# Bean Leaf CNN Classifier (Flutter + CNN)

Isang **Flutter image classification app** na gumagamit ng **CNN (Convolutional Neural Network)** para
malaman kung ang dahon ng bean (sitaw/habichuelas) ay **Healthy**, may **Angular Leaf Spot**, o may
**Bean Rust**. Na-train ang model sa **1,000 larawan** at tumatakbo ito **offline** sa phone (TensorFlow Lite).

## I-install sa phone (Android)

1. Sa phone, buksan ang link na ito (o i-scan ang QR code mula sa PC):
   **https://github.com/ZPGalagnao/bean-leaf-cnn-classifier/releases/latest/download/BeanLeafCNN.apk**
2. I-tap ang na-download na `BeanLeafCNN.apk`.
3. Kapag lumabas ang *"Install unknown apps"*, i-allow ito para sa browser o Files app mo, tapos **Install**.
   Kung lumabas ang Play Protect warning, piliin ang **Install anyway** (dahil hindi galing sa Play Store ang app).
4. Buksan ang **Bean Leaf CNN**, pindutin ang **Camera** o **Gallery**, at pumili ng larawan ng dahon.

Kailangan ng Android 8.0 (API 26) o mas bago.

## Dataset

| | |
|---|---|
| Pangalan | **iBeans** (Makerere AI Lab, Uganda): mga larawan ng dahon ng bean mula sa mga taniman |
| Source | https://huggingface.co/datasets/AI-Lab-Makerere/beans (orihinal: github.com/AI-Lab-Makerere/ibean) |
| Orihinal na laki | 1,295 larawan, 500x500 px |
| **Ginamit natin** | **Eksaktong 1,000 larawan** (random, seed = 42) |
| Classes | `angular_leaf_spot` (334), `bean_rust` (333), `healthy` (333) |
| Hati | **700 train / 150 validation / 150 test** (70/15/15, stratified) |

Bakit ito ang napili: unique ito (hindi ang karaniwang cats vs dogs), may totoong gamit sa agrikultura
(maagang matukoy ang sakit ng halaman), at madaling i-demo gamit ang camera ng phone.

## CNN model (from scratch, walang pre-trained model)

```
Input 128x128x3 (RGB, 0-255)
Rescaling 1/255
[Conv2D 32  3x3 -> BatchNorm -> ReLU -> MaxPool 2x2]
[Conv2D 64  3x3 -> BatchNorm -> ReLU -> MaxPool 2x2]
[Conv2D 128 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2]
[Conv2D 256 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2]
GlobalAveragePooling2D
Dropout 0.4 -> Dense 128 ReLU -> Dropout 0.3
Dense 3 Softmax  ->  [angular_leaf_spot, bean_rust, healthy]
```

- **Data augmentation** (para hindi mag-overfit sa 700 training images lang): random flip, rotation, zoom, contrast
- **Optimizer:** Adam (lr 0.001) + ReduceLROnPlateau. **Loss:** sparse categorical cross-entropy
- **EarlyStopping** (patience 12, ibinabalik ang pinakamahusay na weights), hanggang 60 epochs
- Na-convert sa **TensorFlow Lite** (`bean_cnn.tflite`, mga 1.7 MB) para tumakbo sa phone

## Resulta

Makikita ang eksaktong numbers at graphs sa
[Releases page](https://github.com/ZPGalagnao/bean-leaf-cnn-classifier/releases/latest)
(`metrics.json`, `training_curves.png`, `confusion_matrix.png`, `sample_predictions.png`) at sa
`training/outputs/` kapag na-train mo nang local.

## Paano gumagana ang app

1. Kukuha ng larawan (Camera) o pipili sa Gallery (`image_picker`).
2. I-crop sa gitna para maging square, at i-resize sa 128x128 (`image` package), katulad ng training.
3. Ipapasok sa CNN gamit ang `tflite_flutter`; lalabas ang probability ng bawat class.
4. Ipapakita ang pinaka-malamang na class, ang confidence ng lahat ng class, paliwanag, at payo.

## Project structure

```
training/prepare_dataset.py   pumipili ng 1,000 larawan at naghahati sa train/val/test
training/train_cnn.py         CNN model, training, evaluation, graphs, TFLite export
training/requirements.txt     Python packages (TensorFlow 2.21, Keras 3)
flutter_app/lib/main.dart     UI ng app (camera/gallery, resulta)
flutter_app/lib/classifier.dart  pag-load ng TFLite model + preprocessing + inference
flutter_app/lib/leaf_info.dart   paliwanag at payo para sa bawat class
.github/workflows/build-apk.yml  sa cloud: dataset -> training -> APK -> Release
```

## Pag-train at pag-build nang local (opsyonal)

```
pip install -r training/requirements.txt
python training/prepare_dataset.py --raw <folder na may train/ validation/ test/ ng iBeans>
python training/train_cnn.py
```
Kokopyahin ng `train_cnn.py` ang `bean_cnn.tflite` at `labels.txt` sa `flutter_app/assets/model/`. Pagkatapos:
```
cd flutter_app
flutter create --platforms=android .
flutter build apk --release
```
(Itakda ang `minSdk = 26` sa `android/app/build.gradle.kts`.)

Sa bawat `git push` sa `main`, kusang uulitin ng GitHub Actions ang buong proseso at gagawa ng bagong APK sa Releases.
