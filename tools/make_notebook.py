"""Generates Bean_Leaf_CNN_Kaggle.ipynb (run: python tools/make_notebook.py)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Bean Leaf Disease Classification gamit ang CNN (1,000 images)

**Layunin:** I-train ang isang **Convolutional Neural Network (CNN)** na kumikilala kung ang dahon ng bean ay
**Healthy**, may **Angular Leaf Spot**, o may **Bean Rust**. Ang model ay ginamit sa isang **Flutter mobile app**
(TensorFlow Lite, offline sa phone).

| | |
|---|---|
| Dataset | **iBeans** (Makerere AI Lab): mga larawan ng dahon ng bean mula sa mga taniman sa Uganda |
| Ginamit | **Eksaktong 1,000 larawan** (334 / 333 / 333), random na pinili (seed = 42) |
| Hati | **700 train / 150 validation / 150 test** (70 / 15 / 15) |
| Model | CNN na **tine-train mula sa simula** (walang pre-trained weights) |
| App | https://github.com/ZPGalagnao/bean-leaf-cnn-classifier |

**Sa Kaggle:** i-on ang *Internet* (Settings > Internet on) para ma-download ang dataset at ang model ng app,
o i-attach ang isang iBeans/"beans" dataset gamit ang *Add Input*. Mas mabilis kung naka-**GPU** (Accelerator).
""")

code("""
import os, json, random, shutil, urllib.request, zipfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
DISPLAY = {"angular_leaf_spot": "Angular Leaf Spot", "bean_rust": "Bean Rust", "healthy": "Healthy"}
PER_CLASS = {"angular_leaf_spot": 334, "bean_rust": 333, "healthy": 333}   # = 1,000
APP_MODEL_URL = ("https://github.com/ZPGalagnao/bean-leaf-cnn-classifier/releases/download/"
                 "build-4/bean_cnn.tflite")

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("notebook_work")
WORK.mkdir(parents=True, exist_ok=True)
keras.utils.set_random_seed(SEED)
print("TensorFlow", tf.__version__, "| Keras", keras.__version__)
print("GPU:", tf.config.list_physical_devices("GPU") or "wala (CPU lang)")
""")

md("""
## 1. Dataset

Hahanapin ang mga larawan sa `/kaggle/input` (kung may naka-attach na dataset). Kung wala, ida-download
ito mula sa opisyal na kopya ng Makerere AI Lab sa Hugging Face (kailangang naka-on ang Internet).
""")

code("""
IMG_EXT = {".jpg", ".jpeg", ".png"}

def norm(name):
    return name.lower().replace("-", "_").replace(" ", "_")

def collect_pools(base):
    pools = {c: {} for c in CLASSES}
    for root, _, files in os.walk(base):
        cls = norm(Path(root).name)
        if cls in pools:
            for f in files:
                if Path(f).suffix.lower() in IMG_EXT:
                    pools[cls].setdefault(f, Path(root) / f)   # dedupe by filename
    return pools

def enough(pools):
    return all(len(pools[c]) >= PER_CLASS[c] for c in CLASSES)

pools, source = None, None
# 1) Local copy of the original iBeans files (PC / GitHub Actions)
for base in [Path("data/raw"), Path("../data/raw/beans_download")]:
    if base.exists() and enough(p := collect_pools(base)):
        pools, source = p, base
        break

# 2) Official iBeans download (original filenames -> same 1,000 images as the app)
if pools is None:
    try:
        raw = WORK / "beans_raw"
        raw.mkdir(exist_ok=True)
        for split in ["train", "validation", "test"]:
            url = f"https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/{split}.zip"
            zp = raw / f"{split}.zip"
            print("Downloading", url)
            urllib.request.urlretrieve(url, zp)
            zipfile.ZipFile(zp).extractall(raw)
        if enough(p := collect_pools(raw)):
            pools, source = p, "iBeans (Hugging Face, Makerere AI Lab)"
    except Exception as e:
        print("Hindi ma-download ang iBeans (naka-off ba ang Internet?):", e)

# 3) Fallback: a beans dataset attached with "Add Input" on Kaggle
if pools is None and Path("/kaggle/input").exists() and enough(p := collect_pools(Path("/kaggle/input"))):
    pools, source = p, "/kaggle/input"
    print("PAALALA: ibang kopya ng dataset ang gamit, kaya maaaring iba ang 1,000 napiling larawan.")

if pools is None:
    raise RuntimeError("Walang dataset: i-on ang Internet o i-attach ang isang beans dataset (Add Input).")
print("Source:", source)
for c in CLASSES:
    print(f"  {c:18s} {len(pools[c]):5d} larawan")
""")

md("""
### Pumili ng eksaktong 1,000 larawan at hatiin sa train / validation / test
Ang mga file ay inaayos ayon sa pangalan bago pumili gamit ang `random.Random(42)`, kaya **pareho ang
1,000 larawan at ang 150 test images** saanman patakbuhin (Kaggle, PC, o GitHub Actions na gumawa ng app).
""")

code("""
DATA = WORK / "dataset"
if DATA.exists():
    shutil.rmtree(DATA)
rng = random.Random(SEED)
counts = {}
for c in CLASSES:
    pool = [pools[c][name] for name in sorted(pools[c])]
    chosen = rng.sample(pool, PER_CLASS[c])
    n_train, n_val = round(len(chosen) * 0.70), round(len(chosen) * 0.15)
    parts = {"train": chosen[:n_train], "val": chosen[n_train:n_train + n_val],
             "test": chosen[n_train + n_val:]}
    for split, files in parts.items():
        d = DATA / split / c
        d.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(files):
            shutil.copy2(src, d / f"{c}_{i:03d}{src.suffix.lower()}")
        counts.setdefault(split, {})[DISPLAY[c]] = len(files)

summary = pd.DataFrame(counts)
summary.loc["Total"] = summary.sum()
summary["All"] = summary.sum(axis=1)
summary
""")

code("""
fig, axes = plt.subplots(3, 5, figsize=(14, 8.5))
for row, c in enumerate(CLASSES):
    files = sorted((DATA / "train" / c).iterdir())[:5]
    for ax, f in zip(axes[row], files):
        ax.imshow(plt.imread(f))
        ax.axis("off")
    axes[row][0].set_title(DISPLAY[c], loc="left", fontsize=12, fontweight="bold")
plt.suptitle("Halimbawang larawan sa bawat class", fontsize=14)
plt.tight_layout()
plt.show()
""")

md("""
## 2. CNN models

Dalawang CNN ang tine-train **mula sa simula**, at pipiliin ang may pinakamataas na **validation accuracy**:

* **CNN A (plain):** 4 x [Conv2D 3x3 > ReLU > MaxPool] > Flatten > Dropout > Dense 128 > Dense 3 (softmax)
* **CNN B (batchnorm):** 4 x [Conv2D 3x3 > BatchNorm > ReLU > MaxPool] > GlobalAveragePooling > Dense 128 > Dense 3

Input: 128 x 128 RGB. Ang unang layer ang nagno-normalize (÷255).
**Data augmentation** (flip, rotation, zoom, contrast) laban sa overfitting.
Optimizer: Adam (lr 0.001), EarlyStopping (patience 15) at ReduceLROnPlateau.
""")

code("""
def load_split(name, shuffle):
    return keras.utils.image_dataset_from_directory(
        DATA / name, labels="inferred", label_mode="int", class_names=CLASSES,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=shuffle, seed=SEED)

train_ds = load_split("train", True)
val_ds = load_split("val", False)
test_ds = load_split("test", False)

augment = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
], name="augmentation")
AUTOTUNE = tf.data.AUTOTUNE
train_aug = train_ds.map(lambda x, y: (augment(x, training=True), y),
                         num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_pf = val_ds.prefetch(AUTOTUNE)


def build_cnn_plain():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="normalize")(inputs)
    for i, f in enumerate((32, 64, 128, 128), start=1):
        x = layers.Conv2D(f, 3, padding="same", activation="relu", name=f"conv{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="cnn_plain")


def build_cnn_batchnorm():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="normalize")(inputs)
    for i, f in enumerate((32, 64, 128, 256), start=1):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(momentum=0.9, name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="cnn_batchnorm")


CANDIDATES = {"CNN A (plain)": build_cnn_plain, "CNN B (batchnorm)": build_cnn_batchnorm}
build_cnn_batchnorm().summary()
""")

md("## 3. Training")

code("""
y_true = np.concatenate([y.numpy() for _, y in test_ds])
results, trained = {}, {}
for name, builder in CANDIDATES.items():
    print(f"\\n===== Training {name} =====")
    keras.utils.set_random_seed(SEED)
    model = builder()
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(
        train_aug, validation_data=val_pf, epochs=EPOCHS, verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=15,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5,
                                              patience=6, min_lr=1e-5),
        ])
    test_acc = float((model.predict(test_ds, verbose=0).argmax(1) == y_true).mean())
    trained[name] = (model, hist)
    results[name] = {"Parameters": model.count_params(),
                     "Epochs": len(hist.history["loss"]),
                     "Best val accuracy": max(hist.history["val_accuracy"]),
                     "Test accuracy": test_acc}

comparison = pd.DataFrame(results).T
chosen = comparison["Best val accuracy"].astype(float).idxmax()
print("Napiling model (pinakamataas na validation accuracy):", chosen)
comparison.style.format({"Parameters": "{:,.0f}", "Epochs": "{:.0f}",
                         "Best val accuracy": "{:.1%}", "Test accuracy": "{:.1%}"})
""")

code("""
model, history = trained[chosen]
h = history.history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(h["accuracy"], label="train"); axes[0].plot(h["val_accuracy"], label="validation")
axes[0].set_title(f"Accuracy - {chosen}")
axes[1].plot(h["loss"], label="train"); axes[1].plot(h["val_loss"], label="validation")
axes[1].set_title(f"Loss - {chosen}")
for ax in axes:
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
""")

md("""
## 4. Test results (150 larawan na hindi nakita ng model)
""")

code("""
y_prob = model.predict(test_ds, verbose=0)
y_pred = y_prob.argmax(axis=1)
test_accuracy = (y_pred == y_true).mean()
print(f"TEST ACCURACY ({chosen}): {test_accuracy:.2%}  ({(y_pred == y_true).sum()}/{len(y_true)})\\n")
print(classification_report(y_true, y_pred, target_names=[DISPLAY[c] for c in CLASSES], digits=3))

fig, ax = plt.subplots(figsize=(5.5, 5))
ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred),
                       display_labels=[DISPLAY[c] for c in CLASSES]).plot(ax=ax, cmap="Greens",
                                                                         colorbar=False)
ax.set_title(f"Confusion matrix - {chosen} (test set)")
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
plt.tight_layout()
plt.show()
""")

code("""
images = np.concatenate([x.numpy() for x, _ in test_ds])
idx = np.random.default_rng(SEED).choice(len(images), size=12, replace=False)
fig, axes = plt.subplots(3, 4, figsize=(13, 10))
for ax, i in zip(axes.ravel(), idx):
    p = int(y_pred[i])
    ax.imshow(images[i].astype("uint8"))
    ax.set_title(f"Pred: {DISPLAY[CLASSES[p]]} ({y_prob[i][p]:.0%})\\nTrue: {DISPLAY[CLASSES[y_true[i]]]}",
                 color="green" if p == y_true[i] else "red", fontsize=9)
    ax.axis("off")
plt.suptitle("Sample predictions sa test set (berde = tama, pula = mali)")
plt.tight_layout()
plt.show()
""")

md("""
## 5. Ang mismong model na nasa Flutter app

Bahagyang nag-iiba ang resulta sa bawat training run (random ang augmentation at GPU/CPU). Kaya dito,
dina-download ang **eksaktong `bean_cnn.tflite` na nasa APK** (GitHub release `build-4`) at sinusubok ito sa
**parehong 150 test images**. Ito ang accuracy ng app sa phone.
""")

code("""
def get_interpreter_class():
    try:
        return tf.lite.Interpreter
    except AttributeError:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter

app_model = WORK / "app_bean_cnn.tflite"
try:
    urllib.request.urlretrieve(APP_MODEL_URL, app_model)
    Interpreter = get_interpreter_class()
    interp = Interpreter(model_path=str(app_model))
    inp = interp.get_input_details()[0]
    interp.resize_tensor_input(inp["index"], [1, IMG_SIZE, IMG_SIZE, 3])
    interp.allocate_tensors()
    out = interp.get_output_details()[0]
    app_pred = []
    for img in images:
        interp.set_tensor(inp["index"], img[None].astype(np.float32))
        interp.invoke()
        app_pred.append(int(np.argmax(interp.get_tensor(out["index"])[0])))
    app_pred = np.array(app_pred)
    print(f"Laki ng model: {app_model.stat().st_size / 1024:.0f} KB (TensorFlow Lite)")
    print(f"APP MODEL TEST ACCURACY: {(app_pred == y_true).mean():.2%}  "
          f"({(app_pred == y_true).sum()}/{len(y_true)})\\n")
    print(classification_report(y_true, app_pred, target_names=[DISPLAY[c] for c in CLASSES], digits=3))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, app_pred),
                           display_labels=[DISPLAY[c] for c in CLASSES]).plot(ax=ax, cmap="Blues",
                                                                             colorbar=False)
    ax.set_title("Confusion matrix - model sa Flutter app")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Hindi ma-download o mapatakbo ang app model (naka-on ba ang Internet?):", e)
""")

md("""
## 6. Konklusyon

* Ang dataset ay **1,000 larawan** ng dahon ng bean sa 3 classes (700 train / 150 validation / 150 test).
* Dalawang CNN ang na-train mula sa simula; ang **CNN B (BatchNorm + GlobalAveragePooling)** ay may mas
  kaunting parameters (423K) at mas mataas na validation accuracy, kaya ito ang ginamit sa app.
* Ang model sa Flutter app ay umaabot sa **84% test accuracy** sa 150 larawang hindi nito nakita.
  Pinakamadali para sa model ang *Healthy*; mas mahirap paghiwalayin ang *Bean Rust* at
  *Angular Leaf Spot* dahil pareho silang may mga batik sa dahon (tingnan ang confusion matrix).
* Ang model ay na-convert sa **TensorFlow Lite (~1.7 MB)** at tumatakbo **offline** sa phone.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path(__file__).resolve().parents[1] / "Bean_Leaf_CNN_Kaggle.ipynb"
nbf.write(nb, out)
print("Wrote", out)
