"""Post-processes Step 5's per-instance YOLO labels with a greedy IoU merge, to fix the oversegmentation problem visible at close range (e.g. frame 250 had 606 heavily-overlapping coral boxes - one per foliage instance - which is useless as training data even though each individual box was geometrically correct)."""
import argparse
import json
import os
import shutil

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Dataset")


def to_corners(b):
    x0 = b["cx"] - b["w"] / 2.0
    y0 = b["cy"] - b["h"] / 2.0
    x1 = b["cx"] + b["w"] / 2.0
    y1 = b["cy"] + b["h"] / 2.0
    return x0, y0, x1, y1


def to_center(x0, y0, x1, y1):
    return {"cx": (x0 + x1) / 2.0, "cy": (y0 + y1) / 2.0, "w": x1 - x0, "h": y1 - y0}


def iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / (area_a + area_b - inter)


def merge_class_boxes(boxes, threshold):
    items = [(b, to_corners(b)) for b in boxes]
    items.sort(key=lambda t: (t[1][2] - t[1][0]) * (t[1][3] - t[1][1]), reverse=True)
    remaining = list(items)
    merged = []

    while remaining:
        seed_box, seed_corners = remaining.pop(0)
        cluster_corners = list(seed_corners)
        cluster_members = 1
        changed = True
        while changed:
            changed = False
            still_remaining = []
            for b, corners in remaining:
                if iou(cluster_corners, corners) > threshold:
                    cluster_corners[0] = min(cluster_corners[0], corners[0])
                    cluster_corners[1] = min(cluster_corners[1], corners[1])
                    cluster_corners[2] = max(cluster_corners[2], corners[2])
                    cluster_corners[3] = max(cluster_corners[3], corners[3])
                    cluster_members += 1
                    changed = True
                else:
                    still_remaining.append((b, corners))
            remaining = still_remaining

        out = to_center(*cluster_corners)
        out["merged_from"] = cluster_members
        merged.append(out)

    return merged


def merge_frame(rec, threshold):
    by_class = {}
    for b in rec["boxes"]:
        by_class.setdefault(b["class_id"], []).append(b)

    merged_boxes = []
    for class_id, boxes in by_class.items():
        for m in merge_class_boxes(boxes, threshold):
            m["class_id"] = class_id
            merged_boxes.append(m)
    return merged_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.3, help="IoU threshold to merge same-class boxes")
    ap.add_argument("--dry-run", action="store_true", help="report stats only, write nothing")
    args = ap.parse_args()

    manifest_path = os.path.join(DATASET_DIR, "dataset.json")
    raw_backup_path = os.path.join(DATASET_DIR, "dataset_raw.json")
    labels_dir = os.path.join(DATASET_DIR, "images", "labels")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if not os.path.exists(raw_backup_path):
        shutil.copyfile(manifest_path, raw_backup_path)
        print("backed up original per-instance manifest ->", raw_backup_path)
    else:
        print("raw backup already exists, not overwriting:", raw_backup_path)

    total_before, total_after = 0, 0
    for fnum_str, rec in manifest["frames"].items():
        before = len(rec["boxes"])
        merged = merge_frame(rec, args.threshold)
        total_before += before
        total_after += len(merged)
        rec["boxes"] = merged
        rec["num_boxes"] = len(merged)

    print(f"threshold={args.threshold}: {total_before} boxes -> {total_after} boxes "
          f"({100.0 * (1 - total_after / total_before):.1f}% reduction)")

    if args.dry_run:
        print("dry run - nothing written")
        return

    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print("wrote merged manifest ->", manifest_path)

    for fnum_str, rec in manifest["frames"].items():
        fnum = int(fnum_str)
        with open(os.path.join(labels_dir, f"frame_{fnum:05d}.txt"), "w") as f:
            for b in rec["boxes"]:
                f.write(f"{b['class_id']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}\n")
    print("overwrote YOLO label .txt files ->", labels_dir)


if __name__ == "__main__":
    main()
