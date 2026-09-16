"""Full training run: YOLO26l, 1024px, light augmentation (small-dataset recipe).

Uses Ultralytics' recommended settings for <1000-image datasets: reduced
(not zero) mosaic, mixup/copy_paste off, lower lr0, early stopping. Kept
deliberately lighter than stock large-dataset defaults (mosaic=1.0, mixup=0.3)
since 400 images is still small enough that heavy augmentation could distort
the tight, dense object layouts in these scenes.

Trains on the dr_trans_imgs image set (data.yaml now points train/val at
input_images/dr_trans_imgs/images/{train,val}, reusing the same labels via
input_images/dr_trans_imgs/labels/{train,val}). Output goes to
runs/with_aug_dr_trans -- the original runs/with_aug (trained on the original
input_images/images/{train,val} set, see TRAINING_REPORT.md) is untouched.

Console output is intentionally terse: Ultralytics' own per-batch/per-epoch
chatter is silenced and replaced with one summary line every 2 epochs (see
utils/epoch_logger.py) instead of thousands of progress-bar redraws.
"""
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
        # light augmentation (small-dataset recipe)
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
