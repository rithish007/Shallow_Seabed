"""Run all trained models on real (non-simulated) underwater test photos -- this is the actual held-out test set for this project (as opposed to the val split used in 04_evaluate.py / 05_plot_test_predictions.py, which is used for early-stopping/checkpoint selection during training and isn't held out)."""
import os

from PIL import Image
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(ROOT, "input_images", "real_test_images")
OUT_DIR = os.path.join(ROOT, "output_images", "real_test_preds")

CONF_THRESHOLD = 0.25
PROGRESS_EVERY = 100

MODELS = {
    "no_aug": os.path.join(ROOT, "runs", "no_aug", "weights", "best.pt"),
    "with_aug": os.path.join(ROOT, "runs", "with_aug", "weights", "best.pt"),
    "no_aug_dr_trans": os.path.join(ROOT, "runs", "no_aug_dr_trans", "weights", "best.pt"),
    "with_aug_dr_trans": os.path.join(ROOT, "runs", "with_aug_dr_trans", "weights", "best.pt"),
}


def main():
    if not os.path.isdir(SOURCE_DIR):
        print(f"{SOURCE_DIR} not found.")
        return

    image_files = sorted(
        f for f in os.listdir(SOURCE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not image_files:
        print(f"No images found in {SOURCE_DIR}")
        return

    models = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"SKIP {name}: {path} not found (has training finished?)")
            continue
        models[name] = YOLO(path)
        os.makedirs(os.path.join(OUT_DIR, name), exist_ok=True)

    if not models:
        print("No trained models found -- nothing to run.")
        return

    print(f"Running {len(models)} model(s) on {len(image_files)} real test image(s)", flush=True)

    detected_counts = {name: 0 for name in models}
    for i, fname in enumerate(image_files):
        img_path = os.path.join(SOURCE_DIR, fname)
        out_name = os.path.splitext(fname)[0] + ".png"

        for name, model in models.items():
            result = model.predict(img_path, imgsz=1024, conf=CONF_THRESHOLD, verbose=False)[0]
            if len(result.boxes):
                detected_counts[name] += 1
            pred_img = Image.fromarray(result.plot()[..., ::-1])
            pred_img.save(os.path.join(OUT_DIR, name, out_name))

        if (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == len(image_files):
            print(f"  {i + 1}/{len(image_files)} images done", flush=True)

    print(f"\n=== SUMMARY (images with >=1 detection, conf >= {CONF_THRESHOLD}) ===")
    for name in models:
        pct = 100 * detected_counts[name] / len(image_files)
        print(f"  {name}: {detected_counts[name]}/{len(image_files)} ({pct:.1f}%)")

    print(f"\nDone. Annotated images saved to: {OUT_DIR}\\<model_name>\\")


if __name__ == "__main__":
    main()
