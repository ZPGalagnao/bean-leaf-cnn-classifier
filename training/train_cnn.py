"""
Step 2: I-train ang CNN (Convolutional Neural Network) mula sa simula (from scratch)
at i-export ito sa TensorFlow Lite para magamit sa Flutter app.

Input  : 128 x 128 RGB na larawan ng dahon (0-255; ang model na mismo ang nagno-normalize)
Output : probability ng 3 classes -> angular_leaf_spot, bean_rust, healthy

Gamit:
    python training/train_cnn.py
"""
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras import layers  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "training" / "outputs"
APP_MODEL_DIR = ROOT / "flutter_app" / "assets" / "model"

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
DISPLAY = {"angular_leaf_spot": "Angular Leaf Spot", "bean_rust": "Bean Rust", "healthy": "Healthy"}
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
SEED = 42


def load_split(name: str, shuffle: bool) -> tf.data.Dataset:
    return keras.utils.image_dataset_from_directory(
        DATA_DIR / name,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


def build_cnn_plain() -> keras.Model:
    """CNN A (classic): 4 x [Conv2D -> ReLU -> MaxPool] -> Flatten -> Dense."""
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="normalize")(inputs)
    for i, filters in enumerate((32, 64, 128, 128), start=1):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"conv{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(0.5, name="dropout1")(x)
    x = layers.Dense(128, activation="relu", name="dense")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="cnn_plain")


def build_cnn_batchnorm() -> keras.Model:
    """CNN B: 4 x [Conv2D -> BatchNorm -> ReLU -> MaxPool] -> GlobalAvgPool -> Dense."""
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="normalize")(inputs)
    for i, filters in enumerate((32, 64, 128, 256), start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(momentum=0.9, name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="dropout1")(x)
    x = layers.Dense(128, activation="relu", name="dense")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="cnn_batchnorm")


CANDIDATES = {
    "CNN A (plain)": build_cnn_plain,
    "CNN B (batchnorm)": build_cnn_batchnorm,
}


def train_candidate(builder, train_aug, val_ds):
    keras.utils.set_random_seed(SEED)
    model = builder()
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(
        train_aug,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=15,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5,
                                              patience=6, min_lr=1e-5),
        ],
        verbose=2,
    )
    return model, history


def plot_history(history: keras.callbacks.History) -> None:
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h["accuracy"], label="train")
    axes[0].plot(h["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[1].plot(h["loss"], label="train")
    axes[1].plot(h["val_loss"], label="validation")
    axes[1].set_title("Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=[DISPLAY[c] for c in CLASSES]).plot(
        ax=ax, cmap="Greens", colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_samples(test_ds: tf.data.Dataset, y_prob: np.ndarray) -> None:
    images = np.concatenate([x.numpy() for x, _ in test_ds])
    labels = np.concatenate([y.numpy() for _, y in test_ds])
    idx = np.random.default_rng(SEED).choice(len(images), size=12, replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.5))
    for ax, i in zip(axes.ravel(), idx):
        pred = int(np.argmax(y_prob[i]))
        ok = pred == labels[i]
        ax.imshow(images[i].astype("uint8"))
        ax.set_title(f"Pred: {DISPLAY[CLASSES[pred]]} ({y_prob[i][pred]:.0%})\n"
                     f"True: {DISPLAY[CLASSES[labels[i]]]}",
                     color="green" if ok else "red", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sample_predictions.png", dpi=120)
    plt.close(fig)


def export_tflite(model: keras.Model) -> bytes:
    saved_dir = OUT_DIR / "saved_model"
    if saved_dir.exists():
        shutil.rmtree(saved_dir)
    model.export(str(saved_dir))
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
    return converter.convert()


def tflite_accuracy(tflite_model: bytes, test_ds: tf.data.Dataset) -> float:
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    inp = interpreter.get_input_details()[0]
    interpreter.resize_tensor_input(inp["index"], [1, IMG_SIZE, IMG_SIZE, 3])
    interpreter.allocate_tensors()
    out = interpreter.get_output_details()[0]
    correct = total = 0
    for images, labels in test_ds:
        for img, label in zip(images.numpy(), labels.numpy()):
            interpreter.set_tensor(inp["index"], img[None].astype(np.float32))
            interpreter.invoke()
            correct += int(np.argmax(interpreter.get_tensor(out["index"])[0]) == label)
            total += 1
    return correct / total


def main() -> None:
    keras.utils.set_random_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = load_split("train", shuffle=True)
    val_ds = load_split("val", shuffle=False)
    test_ds = load_split("test", shuffle=False)

    # Data augmentation: para hindi mag-overfit sa 700 training images lang.
    augment = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ], name="augmentation")
    autotune = tf.data.AUTOTUNE
    train_aug = train_ds.map(lambda x, y: (augment(x, training=True), y),
                             num_parallel_calls=autotune).prefetch(autotune)

    # ---- I-train ang 2 CNN variants, piliin ang may pinakamataas na validation accuracy ----
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    comparison = {}
    best = None
    for name, builder in CANDIDATES.items():
        print(f"\n===== Training {name} =====")
        cand, hist = train_candidate(builder, train_aug, val_ds.prefetch(autotune))
        val_acc = float(max(hist.history["val_accuracy"]))
        cand_test = float((cand.predict(test_ds, verbose=0).argmax(axis=1) == y_true).mean())
        comparison[name] = {
            "parameters": int(cand.count_params()),
            "epochs_trained": len(hist.history["loss"]),
            "best_val_accuracy": round(val_acc, 4),
            "test_accuracy": round(cand_test, 4),
        }
        print(f"{name}: best val acc {val_acc:.4f}, test acc {cand_test:.4f}")
        if best is None or val_acc > best[2]:
            best = (name, cand, val_acc, hist)
    chosen_name, model, _, history = best
    print(f"\nNapiling model (pinakamataas na validation accuracy): {chosen_name}")
    (OUT_DIR / "model_comparison.json").write_text(
        json.dumps({"chosen": chosen_name, "models": comparison}, indent=2))

    model.summary()
    with open(OUT_DIR / "model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line, **_: f.write(line + "\n"))
    plot_history(history)

    # ---- Evaluation sa test set (150 larawan na hindi nakita ng model) ----
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    test_acc = float((y_pred == y_true).mean())
    report = classification_report(y_true, y_pred, target_names=[DISPLAY[c] for c in CLASSES],
                                   output_dict=True)
    print(classification_report(y_true, y_pred, target_names=[DISPLAY[c] for c in CLASSES]))
    plot_confusion(y_true, y_pred)
    plot_samples(test_ds, y_prob)

    # ---- Save + export sa TFLite para sa Flutter ----
    model.save(OUT_DIR / "bean_cnn.keras")
    tflite_model = export_tflite(model)
    (OUT_DIR / "bean_cnn.tflite").write_bytes(tflite_model)
    (OUT_DIR / "labels.txt").write_text("\n".join(CLASSES) + "\n")
    lite_acc = tflite_accuracy(tflite_model, test_ds)

    APP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_DIR / "bean_cnn.tflite", APP_MODEL_DIR / "bean_cnn.tflite")
    shutil.copy2(OUT_DIR / "labels.txt", APP_MODEL_DIR / "labels.txt")

    metrics = {
        "chosen_model": chosen_name,
        "model_comparison": comparison,
        "image_size": IMG_SIZE,
        "epochs_trained": len(history.history["loss"]),
        "parameters": int(model.count_params()),
        "test_accuracy_keras": round(test_acc, 4),
        "test_accuracy_tflite": round(lite_acc, 4),
        "best_val_accuracy": round(float(max(history.history["val_accuracy"])), 4),
        "tflite_size_kb": round(len(tflite_model) / 1024, 1),
        "classification_report": report,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    rows = "\n".join(
        f"| {DISPLAY[c]} | {report[DISPLAY[c]]['precision']:.2f} | {report[DISPLAY[c]]['recall']:.2f} "
        f"| {report[DISPLAY[c]]['f1-score']:.2f} |" for c in CLASSES)
    cmp_rows = "\n".join(
        f"| {n} | {m['parameters']:,} | {m['epochs_trained']} | {m['best_val_accuracy']:.1%} "
        f"| {m['test_accuracy']:.1%} |" for n, m in comparison.items())
    (OUT_DIR / "release_notes.md").write_text(
        "## Bean Leaf CNN Classifier\n\n"
        "**I-download sa phone:** i-tap ang `BeanLeafCNN.apk` sa ibaba, tapos i-install.\n\n"
        f"- Dataset: 1,000 bean leaf images (700 train / 150 validation / 150 test)\n"
        f"- Model sa app: **{chosen_name}**, CNN trained from scratch, "
        f"{metrics['parameters']:,} parameters, input {IMG_SIZE}x{IMG_SIZE}\n"
        f"- **Test accuracy: {test_acc:.1%}** (TFLite on phone: {lite_acc:.1%})\n\n"
        "| Class | Precision | Recall | F1 |\n|---|---|---|---|\n" + rows + "\n\n"
        "### Model comparison\n\n"
        "| Model | Parameters | Epochs | Best val acc | Test acc |\n|---|---|---|---|---|\n"
        + cmp_rows + "\n")

    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
