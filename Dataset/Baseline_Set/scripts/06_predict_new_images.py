"""Run the trained model(s) on a folder of brand-new images (e.g. freshly captured from the UE simulation) that have no ground-truth labels."""
import os

from PIL import Image
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(ROOT, "input_images", "new_test_images")
OUT_DIR = os.path.join(ROOT, "output_images", "new_test_predictions")

CONF_THRESHOLD = 0.25
RUN_BOTH_MODELS = False
TITLE_BAR_H = 36

MODELS = {
    "with_aug_dr_trans": os.path.join(ROOT, "runs", "with_aug_dr_trans", "weights", "best.pt"),
}
if RUN_BOTH_MODELS:
    MODELS["no_aug_dr_trans"] = os.path.join(ROOT, "runs", "no_aug_dr_trans", "weights", "best.pt")


def add_title(img, title):
    from PIL import ImageDraw
    w, h = img.size
    canvas = Image.new("RGB", (w, h + TITLE_BAR_H), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="white")
    canvas.paste(img, (0, TITLE_BAR_H))
    return canvas


def main():
    if not os.path.isdir(SOURCE_DIR):
        print(f"Create {SOURCE_DIR} and put your new images in it, then re-run.")
        return

    image_files = sorted(
        f for f in os.listdir(SOURCE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not image_files:
        print(f"No images found in {SOURCE_DIR}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    models = {name: YOLO(path) for name, path in MODELS.items()}
    print(f"Running {len(models)} model(s) on {len(image_files)} new image(s)")

    for fname in image_files:
        img_path = os.path.join(SOURCE_DIR, fname)
        panels = []
        summary_parts = []

        for name, model in models.items():
            result = model.predict(img_path, imgsz=1024, conf=CONF_THRESHOLD, verbose=False)[0]
            pred_img = Image.fromarray(result.plot()[..., ::-1])
            panels.append(add_title(pred_img, f"{name} (conf>={CONF_THRESHOLD})"))

            counts = {}
            for cls_id in result.boxes.cls.tolist():
                cls_name = result.names[int(cls_id)]
                counts[cls_name] = counts.get(cls_name, 0) + 1
            summary_parts.append(f"{name}: {counts if counts else 'no detections'}")

        w, h = panels[0].size
        composite = Image.new("RGB", (w * len(panels), h), "black")
        for j, panel in enumerate(panels):
            composite.paste(panel, (w * j, 0))
        composite.save(os.path.join(OUT_DIR, fname))

        print(f"{fname}: " + " | ".join(summary_parts))

    print(f"\nDone. Annotated images saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
