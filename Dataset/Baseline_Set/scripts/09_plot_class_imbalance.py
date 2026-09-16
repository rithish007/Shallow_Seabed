"""Plot per-class box-count imbalance (train vs. val) for the dataset
currently referenced by data.yaml.

Reads data.yaml the same way 00_preflight.py does, so this automatically
reflects whichever image/label set is currently active (e.g. dr_trans_imgs).
Class balance depends only on the label .txt files, not which image render
they're paired with -- the dr_trans_imgs labels are an exact copy of the
original input_images/labels/ set (see README), so this plot is the same
regardless of which of the two image variants data.yaml currently points at.

Saves a grouped bar chart to output_images/class_imbalance.png and prints
the underlying per-class counts plus an imbalance ratio (max class / min
class, per split).
"""
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from utils.label_paths import images_to_labels_rel_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "data.yaml")
OUT_PATH = os.path.join(ROOT, "output_images", "class_imbalance.png")


def count_boxes_per_class(labels_dir, class_names):
    counts = Counter({name: 0 for name in class_names})
    for fname in os.listdir(labels_dir):
        if not fname.lower().endswith(".txt"):
            continue
        with open(os.path.join(labels_dir, fname), "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                cls_id = int(ln.split()[0])
                counts[class_names[cls_id]] += 1
    return counts


def main():
    with open(DATA_YAML, "r") as f:
        cfg = yaml.safe_load(f)

    class_names = cfg["names"]
    base = cfg["path"]
    train_labels = os.path.join(base, images_to_labels_rel_path(cfg["train"]))
    val_labels = os.path.join(base, images_to_labels_rel_path(cfg["val"]))

    train_counts = count_boxes_per_class(train_labels, class_names)
    val_counts = count_boxes_per_class(val_labels, class_names)

    print(f"=== Class imbalance (labels: {train_labels}, {val_labels}) ===")
    print(f"{'class':<10} {'train':>8} {'val':>8} {'total':>8}")
    for name in class_names:
        t, v = train_counts[name], val_counts[name]
        print(f"{name:<10} {t:>8} {v:>8} {t + v:>8}")

    for split_name, counts in [("train", train_counts), ("val", val_counts)]:
        nonzero = [c for c in counts.values() if c > 0]
        if nonzero:
            ratio = max(nonzero) / min(nonzero)
            print(f"{split_name} imbalance ratio (max class / min class): {ratio:.1f}x")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    x = range(len(class_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], [train_counts[n] for n in class_names], width, label="train")
    ax.bar([i + width / 2 for i in x], [val_counts[n] for n in class_names], width, label="val")
    ax.set_xticks(list(x))
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Annotated box count")
    ax.set_title("Class imbalance: box count per class (train vs. val)")
    ax.legend()
    for i, name in enumerate(class_names):
        for offset, counts in [(-width / 2, train_counts), (width / 2, val_counts)]:
            v = counts[name]
            ax.text(i + offset, v, str(v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
