"""Plot Ground Truth vs. no_aug_dr_trans vs. with_aug_dr_trans predictions side-by-side for every val image."""
import os

from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "data.yaml")
VAL_IMAGES = os.path.join(ROOT, "input_images", "dr_trans_imgs", "images", "val")
VAL_LABELS = os.path.join(ROOT, "input_images", "dr_trans_imgs", "labels", "val")
OUT_DIR = os.path.join(ROOT, "output_images", "testtheval_set_plots_dr_trans")

CLASS_NAMES = ["coral", "kelp", "rock", "sponge"]
CLASS_COLORS = ["red", "lime", "cyan", "yellow"]
CONF_THRESHOLD = 0.25
TITLE_BAR_H = 36

MODELS = {
    "no_aug_dr_trans": os.path.join(ROOT, "runs", "no_aug_dr_trans", "weights", "best.pt"),
    "with_aug_dr_trans": os.path.join(ROOT, "runs", "with_aug_dr_trans", "weights", "best.pt"),
}


def draw_ground_truth(image_path, label_path):
    img = Image.open(image_path).convert("RGB")
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
                x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                draw.text((x0, max(0, y0 - 12)), CLASS_NAMES[cls_id], fill=color)
    return img


def add_title(img, title):
    w, h = img.size
    canvas = Image.new("RGB", (w, h + TITLE_BAR_H), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="white")
    canvas.paste(img, (0, TITLE_BAR_H))
    return canvas


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    models = {name: YOLO(path) for name, path in MODELS.items()}

    image_files = sorted(f for f in os.listdir(VAL_IMAGES) if f.lower().endswith(".png"))
    print(f"Plotting {len(image_files)} val images x {len(models) + 1} panels (GT + {len(models)} models)")

    for i, fname in enumerate(image_files):
        img_path = os.path.join(VAL_IMAGES, fname)
        label_path = os.path.join(VAL_LABELS, os.path.splitext(fname)[0] + ".txt")

        gt_panel = add_title(draw_ground_truth(img_path, label_path), "Ground Truth")

        panels = [gt_panel]
        for name, model in models.items():
            result = model.predict(img_path, imgsz=1024, conf=CONF_THRESHOLD, verbose=False)[0]
            pred_img = Image.fromarray(result.plot()[..., ::-1])
            panels.append(add_title(pred_img, f"{name} (conf>={CONF_THRESHOLD})"))

        w, h = panels[0].size
        composite = Image.new("RGB", (w * len(panels), h), "black")
        for j, panel in enumerate(panels):
            composite.paste(panel, (w * j, 0))

        out_path = os.path.join(OUT_DIR, fname)
        composite.save(out_path)
        if (i + 1) % 20 == 0 or (i + 1) == len(image_files):
            print(f"  {i + 1}/{len(image_files)} saved")

    print(f"Done. Composites saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
