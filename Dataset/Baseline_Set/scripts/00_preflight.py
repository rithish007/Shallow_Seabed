"""Pre-flight sanity check for the Baseline_Set YOLO dataset."""
import os
import random

import torch
import yaml
from PIL import Image, ImageDraw
from utils.label_paths import images_to_labels_rel_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "data.yaml")
OUT_DIR = os.path.join(ROOT, "output_images", "preflight_out")


def check_gpu():
    print("=== GPU ===")
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device name:", torch.cuda.get_device_name(0))
        print("total VRAM (MiB):", torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))


def load_data_yaml():
    with open(DATA_YAML, "r") as f:
        cfg = yaml.safe_load(f)
    print("=== data.yaml ===")
    print(cfg)
    return cfg


def check_split(split_name, images_dir, labels_dir, num_classes):
    print(f"=== split: {split_name} ===")
    image_files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(".png"))
    label_files = sorted(f for f in os.listdir(labels_dir) if f.lower().endswith(".txt"))
    print(f"images: {len(image_files)}  labels: {len(label_files)}")

    image_stems = {os.path.splitext(f)[0] for f in image_files}
    label_stems = {os.path.splitext(f)[0] for f in label_files}
    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems
    if missing_labels:
        print(f"WARNING: {len(missing_labels)} images with no label file (assumed empty/background if intentional): {sorted(missing_labels)[:5]}...")
    if missing_images:
        print(f"WARNING: {len(missing_images)} label files with no matching image: {sorted(missing_images)[:5]}...")

    bad_class_id = []
    bad_coords = []
    empty_files = []
    total_boxes = 0
    box_counts = []
    for lf in label_files:
        path = os.path.join(labels_dir, lf)
        with open(path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            empty_files.append(lf)
        box_counts.append(len(lines))
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                bad_coords.append((lf, ln))
                continue
            cls_id = int(parts[0])
            coords = [float(p) for p in parts[1:]]
            total_boxes += 1
            if not (0 <= cls_id < num_classes):
                bad_class_id.append((lf, cls_id))
            if not all(0.0 <= c <= 1.0 for c in coords):
                bad_coords.append((lf, ln))

    print(f"total boxes: {total_boxes}")
    print(f"empty label files (background frames): {len(empty_files)}")
    print(f"bad class ids: {len(bad_class_id)}  (sample: {bad_class_id[:5]})")
    print(f"bad coords: {len(bad_coords)}  (sample: {bad_coords[:5]})")
    if box_counts:
        print(f"boxes per image: min={min(box_counts)} max={max(box_counts)} avg={sum(box_counts)/len(box_counts):.1f}")
        dense = sum(1 for c in box_counts if c > 10)
        print(f"images with >10 boxes: {dense}/{len(box_counts)} ({100*dense/len(box_counts):.1f}%)")

    return image_files, labels_dir, images_dir


def render_samples(images_dir, labels_dir, class_names, n=3, seed=0):
    os.makedirs(OUT_DIR, exist_ok=True)
    image_files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(".png"))
    random.seed(seed)
    sample = random.sample(image_files, min(n, len(image_files)))
    colors = ["red", "lime", "cyan", "yellow", "magenta", "orange"]
    for fname in sample:
        img_path = os.path.join(images_dir, fname)
        label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + ".txt")
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    cls_id, cx, cy, bw, bh = ln.split()
                    cls_id = int(cls_id)
                    cx, cy, bw, bh = (float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h)
                    x0, y0 = cx - bw / 2, cy - bh / 2
                    x1, y1 = cx + bw / 2, cy + bh / 2
                    color = colors[cls_id % len(colors)]
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                    label_text = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                    draw.text((x0, max(0, y0 - 12)), label_text, fill=color)
        out_path = os.path.join(OUT_DIR, f"overlay_{fname}")
        img.save(out_path)
        print(f"saved overlay: {out_path}")


def main():
    check_gpu()
    cfg = load_data_yaml()
    class_names = cfg["names"]
    num_classes = cfg["nc"]
    base = cfg["path"]

    train_images = os.path.join(base, cfg["train"])
    train_labels = os.path.join(base, images_to_labels_rel_path(cfg["train"]))
    val_images = os.path.join(base, cfg["val"])
    val_labels = os.path.join(base, images_to_labels_rel_path(cfg["val"]))

    check_split("train", train_images, train_labels, num_classes)
    check_split("val", val_images, val_labels, num_classes)

    render_samples(train_images, train_labels, class_names, n=3, seed=0)
    render_samples(val_images, val_labels, class_names, n=2, seed=0)

    print("=== DONE ===")


if __name__ == "__main__":
    main()
