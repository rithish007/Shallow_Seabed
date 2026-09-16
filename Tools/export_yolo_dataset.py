"""
Step 6 - export the merged labels (Dataset/dataset.json + images/rgb) into a
standard Ultralytics-YOLO training layout with a train/val split.

SPLIT METHOD: block-based, not per-frame-random. Frames are 500 evenly-spaced
samples along one continuous spline path (~8m apart) with a ~12.4m visibility
range, so consecutive frames see overlapping geometry - a naive per-frame
random split would leak near-duplicate content across train/val and inflate
validation metrics. Instead this chunks the path into contiguous blocks
(BLOCK_SIZE frames each) and randomly assigns whole blocks to train/val, so
adjacent frames stay together and both splits still sample the whole path
(not e.g. "first 80% of the path = train").

CLASS ID REMAP: the raw labels use class_id 1-4 (matching
tag_stencil_classes.CLASS_STENCIL_MAP's stencil values). YOLO/Ultralytics
`data.yaml` `names` lists are 0-indexed - exporting the raw 1-4 ids verbatim
against a 4-entry names list would silently shift every class by one. This
script remaps 1->0, 2->1, 3->2, 4->3 while copying.

STANDALONE - run with regular system Python (no third-party deps needed).

Usage:
    python Tools/export_yolo_dataset.py --out "Dataset/Baseline_Set" --val-fraction 0.2
"""
import argparse
import json
import os
import random
import shutil

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_DIR, "Dataset")

BLOCK_SIZE = 20
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="output directory for the exported YOLO dataset")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    args = ap.parse_args()

    with open(os.path.join(DATASET_DIR, "dataset.json"), "r") as f:
        manifest = json.load(f)

    class_names_raw = manifest["class_names"]  # {"1": "coral", "2": "kelp", "3": "rock", "4": "sponge"}
    old_ids_sorted = sorted(int(k) for k in class_names_raw.keys())
    remap = {old_id: new_id for new_id, old_id in enumerate(old_ids_sorted)}
    names_0indexed = [class_names_raw[str(old_id)] for old_id in old_ids_sorted]
    print("class remap (old -> new):", remap)
    print("names (0-indexed):", names_0indexed)

    frame_nums = sorted(int(k) for k in manifest["frames"].keys())
    blocks = [frame_nums[i:i + args.block_size] for i in range(0, len(frame_nums), args.block_size)]
    rng = random.Random(SEED)
    rng.shuffle(blocks)

    n_val_blocks = max(1, round(len(blocks) * args.val_fraction))
    val_blocks = blocks[:n_val_blocks]
    train_blocks = blocks[n_val_blocks:]
    val_frames = sorted(f for block in val_blocks for f in block)
    train_frames = sorted(f for block in train_blocks for f in block)
    print(f"{len(blocks)} blocks of <= {args.block_size} frames -> "
          f"{len(train_blocks)} train blocks ({len(train_frames)} frames), "
          f"{len(val_blocks)} val blocks ({len(val_frames)} frames)")

    out_dir = args.out
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    class_totals = {"train": {}, "val": {}}

    def export_split(frames, split):
        for fnum in frames:
            rec = manifest["frames"][str(fnum)]
            src_img = os.path.join(DATASET_DIR, "images", "rgb", f"frame_{fnum:05d}.png")
            dst_img = os.path.join(out_dir, "images", split, f"frame_{fnum:05d}.png")
            shutil.copyfile(src_img, dst_img)

            dst_lbl = os.path.join(out_dir, "labels", split, f"frame_{fnum:05d}.txt")
            with open(dst_lbl, "w") as f:
                for b in rec["boxes"]:
                    new_id = remap[b["class_id"]]
                    class_totals[split][new_id] = class_totals[split].get(new_id, 0) + 1
                    f.write(f"{new_id} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}\n")

    export_split(train_frames, "train")
    export_split(val_frames, "val")

    for split in ("train", "val"):
        named = {names_0indexed[cid]: cnt for cid, cnt in sorted(class_totals[split].items())}
        print(f"{split}: {len(named) and sum(named.values())} boxes -> {named}")

    yaml_path = os.path.join(out_dir, "data.yaml")
    yaml_content = (
        f"path: {out_dir.replace(os.sep, '/')}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(names_0indexed)}\n"
        f"names: {json.dumps(names_0indexed)}\n"
    )
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print("wrote", yaml_path)
    print("\n" + yaml_content)


if __name__ == "__main__":
    main()
