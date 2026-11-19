"""
Step 1: Gumawa ng dataset na may EKSAKTONG 1,000 larawan ng bean leaves.

Pinagmulan: iBeans dataset (Makerere AI Lab) - 1,295 larawan, 3 classes.
Kukuha tayo ng 1,000 (334 + 333 + 333) gamit ang fixed random seed, tapos
hahatiin sa train / validation / test = 70% / 15% / 15%.

Gamit:
    python training/prepare_dataset.py --raw data/raw
"""
import argparse
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "dataset"
SUMMARY_DIR = ROOT / "training" / "outputs"

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
PER_CLASS = {"angular_leaf_spot": 334, "bean_rust": 333, "healthy": 333}  # kabuuan = 1,000
TRAIN_FRACTION, VAL_FRACTION = 0.70, 0.15  # ang natitira (15%) ay test
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True,
                        help="Folder na may na-extract na train/validation/test ng iBeans")
    args = parser.parse_args()

    rng = random.Random(SEED)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    summary = {"total": 0, "splits": {"train": {}, "val": {}, "test": {}}}
    for cls in CLASSES:
        pool = sorted(
            p for p in args.raw.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and p.parent.name == cls
        )
        if len(pool) < PER_CLASS[cls]:
            raise SystemExit(f"Kulang ang larawan para sa {cls}: {len(pool)} lang ang nahanap")

        chosen = rng.sample(pool, PER_CLASS[cls])
        n_train = round(len(chosen) * TRAIN_FRACTION)
        n_val = round(len(chosen) * VAL_FRACTION)
        parts = {
            "train": chosen[:n_train],
            "val": chosen[n_train:n_train + n_val],
            "test": chosen[n_train + n_val:],
        }
        for split, files in parts.items():
            dest = OUT_DIR / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(files):
                shutil.copy2(src, dest / f"{cls}_{i:03d}{src.suffix.lower()}")
            summary["splits"][split][cls] = len(files)
            summary["total"] += len(files)

    print(f"Kabuuang larawan: {summary['total']}")
    for split, counts in summary["splits"].items():
        print(f"  {split:5s}: {sum(counts.values()):4d}  {counts}")
    assert summary["total"] == 1000, "Dapat eksaktong 1,000 ang dataset"

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
