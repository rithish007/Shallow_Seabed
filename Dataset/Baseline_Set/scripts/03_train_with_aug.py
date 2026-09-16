"""Full training run: YOLO26l, 1024px, light augmentation (small-dataset recipe)."""
import os

from ultralytics import YOLO
from utils.epoch_logger import attach_epoch_logger

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "data.yaml")


def main():
    model = YOLO("yolo26l.pt")
    attach_epoch_logger(model, every=2)
    results = model.train(
        data=DATA_YAML,
        epochs=150,
        patience=25,
        imgsz=1024,
        batch=4,
        device=0,
        amp=True,
        cache=True,
        lr0=0.001,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        project=os.path.join(ROOT, "runs"),
        name="with_aug_dr_trans",
        exist_ok=True,
        verbose=True,
    )
    print("=== WITH-AUG (dr_trans) TRAINING DONE ===")
    print("results dir:", results.save_dir)


if __name__ == "__main__":
    main()
