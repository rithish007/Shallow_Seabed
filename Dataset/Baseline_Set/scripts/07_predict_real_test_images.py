"""Run all trained models on real (non-simulated) underwater test photos --
this is the actual held-out test set for this project (as opposed to the val
split used in 04_evaluate.py / 05_plot_test_predictions.py, which is used for
early-stopping/checkpoint selection during training and isn't held out).

These images (input_images/real_test_images/) are actual camera captures, not
frames from the Unreal simulation the models were trained on -- so this is a
genuine sim-to-real domain-gap check, not another in-distribution val split.
There are no ground-truth labels for these, so results are qualitative only
(no mAP can be computed) -- judge by eye, or use the aggregate detection-rate
summary printed at the end.

Runs every trained comparison model (no_aug, with_aug, no_aug_dr_trans,
with_aug_dr_trans -- smoke_test is excluded, it's a 3-epoch yolo26n pipeline
sanity check, not one of the real comparison models) and saves each model's
own annotated predictions to its own subfolder:
    output_images/real_test_preds/<model_name>/<image>.png
One folder per model rather than side-by-side composites, since
input_images/real_test_images/ now holds ~2000 photos (composites across 4
models would be both slow to build and unwieldy to browse at that scale).

output_images/real_test_predictions/ and output_images/real_test_dr_preds/
(2-panel composites from the original 10-photo spot-check) are untouched --
left as a historical record of that smaller run.

Console output is intentionally sparse (progress every 100 images, not one
line per image) since this now processes ~2000 photos x N models.
"""
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
