"""Draws the YOLO label boxes from Dataset/dataset.json onto the RGB frames, so you can actually look at annotated images instead of trusting the raw .txt numbers."""
import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Dataset")

CLASS_COLORS = {
    1: (230, 60, 60),
    2: (60, 200, 90),
    3: (70, 120, 230),
    4: (230, 210, 60),
}


def load_manifest():
    with open(os.path.join(DATASET_DIR, "dataset.json"), "r") as f:
        return json.load(f)


def pick_frames(manifest, frames_arg, count, all_frames):
    all_nums = sorted(int(k) for k in manifest["frames"].keys())
    if all_frames:
        return all_nums
    if frames_arg:
        wanted = [int(x) for x in frames_arg.split(",")]
        return [f for f in wanted if f in set(all_nums)]
    if count >= len(all_nums):
        return all_nums
    step = len(all_nums) / count
    return sorted({all_nums[int(i * step)] for i in range(count)})


def draw_frame(manifest, frame_num, out_dir, class_names):
    rec = manifest["frames"][str(frame_num)]
    img_path = os.path.join(DATASET_DIR, "images", "rgb", f"frame_{frame_num:05d}.png")
    img = Image.open(img_path).convert("RGB")
    w_px, h_px = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for b in rec["boxes"]:
        color = CLASS_COLORS.get(b["class_id"], (255, 255, 255))
        cx, cy, bw, bh = b["cx"] * w_px, b["cy"] * h_px, b["w"] * w_px, b["h"] * h_px
        x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

    label = f"frame {frame_num} - {rec['num_boxes']} boxes"
    draw.rectangle([0, 0, 8 + 7 * len(label), 18], fill=(0, 0, 0))
    draw.text((4, 2), label, fill=(255, 255, 255), font=font)

    ly = 24
    for cid, name in sorted(class_names.items(), key=lambda kv: int(kv[0])):
        color = CLASS_COLORS.get(int(cid), (255, 255, 255))
        draw.rectangle([4, ly, 20, ly + 12], fill=color)
        draw.text((24, ly), name, fill=(255, 255, 255), font=font)
        ly += 16

    out_path = os.path.join(out_dir, f"frame_{frame_num:05d}.png")
    img.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=str, default=None, help="comma-separated frame numbers")
    ap.add_argument("--count", type=int, default=20, help="how many frames to spread across the dataset")
    ap.add_argument("--all", action="store_true", help="render every frame")
    args = ap.parse_args()

    manifest = load_manifest()
    class_names = manifest["class_names"]
    out_dir = os.path.join(DATASET_DIR, "debug_overlays")
    os.makedirs(out_dir, exist_ok=True)

    frame_nums = pick_frames(manifest, args.frames, args.count, args.all)
    print(f"rendering {len(frame_nums)} frames -> {out_dir}")
    for fn in frame_nums:
        path = draw_frame(manifest, fn, out_dir, class_names)
        print("  ", path)


if __name__ == "__main__":
    main()
