"""Evaluate both trained models (no_aug_dr_trans vs with_aug_dr_trans) on the val split and compare per-class AP plus a handful of side-by-side prediction images."""
import os

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "data.yaml")
RUNS = os.path.join(ROOT, "runs")

MODELS = {
    "no_aug_dr_trans": os.path.join(RUNS, "no_aug_dr_trans", "weights", "best.pt"),
    "with_aug_dr_trans": os.path.join(RUNS, "with_aug_dr_trans", "weights", "best.pt"),
}


def main():
    summary = {}
    for label, weights_path in MODELS.items():
        if not os.path.exists(weights_path):
            print(f"SKIP {label}: {weights_path} not found (has training finished?)")
            continue

        print(f"\n=== Validating: {label} ({weights_path}) ===")
        model = YOLO(weights_path)
        metrics = model.val(data=DATA_YAML, imgsz=1024, split="val")
        summary[label] = {
            "map50-95": float(metrics.box.map),
            "map50": float(metrics.box.map50),
            "per_class_map50-95": {
                name: float(ap) for name, ap in zip(metrics.names.values(), metrics.box.maps)
            },
        }

        pred_dir = os.path.join(ROOT, "output_images", "eval_preds", label)
        os.makedirs(pred_dir, exist_ok=True)
        val_images_dir = os.path.join(ROOT, "input_images", "dr_trans_imgs", "images", "val")
        sample_images = sorted(
            f for f in os.listdir(val_images_dir) if f.lower().endswith(".png")
        )[:10]
        for fname in sample_images:
            model.predict(
                source=os.path.join(val_images_dir, fname),
                imgsz=1024,
                save=True,
                project=os.path.join(ROOT, "output_images", "eval_preds"),
                name=label,
                exist_ok=True,
                conf=0.25,
            )

    print("\n=== SUMMARY ===")
    for label, m in summary.items():
        print(f"{label}: mAP50-95={m['map50-95']:.4f}  mAP50={m['map50']:.4f}")
        for cls_name, ap in m["per_class_map50-95"].items():
            print(f"    {cls_name}: {ap:.4f}")

    if len(summary) == 2:
        no_aug_map = summary["no_aug_dr_trans"]["map50-95"]
        with_aug_map = summary["with_aug_dr_trans"]["map50-95"]
        winner = "with_aug_dr_trans" if with_aug_map > no_aug_map else "no_aug_dr_trans"
        print(f"\nBetter generalizing model on val: {winner} "
              f"(no_aug_dr_trans={no_aug_map:.4f} vs with_aug_dr_trans={with_aug_map:.4f})")


if __name__ == "__main__":
    main()
