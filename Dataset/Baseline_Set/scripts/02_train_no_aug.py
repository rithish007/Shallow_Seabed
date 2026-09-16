"""Full training run: YOLO26l, 1024px, ALL augmentation disabled.

Answers "does this dataset give good accuracy without augmentation?" directly.
With only 400 training images, expect faster convergence but a larger
train/val gap (overfitting) than the augmented run in 03_train_with_aug.py.

Trains on the dr_trans_imgs image set (data.yaml now points train/val at
input_images/dr_trans_imgs/images/{train,val}, reusing the same labels via
input_images/dr_trans_imgs/labels/{train,val}). Output goes to
runs/no_aug_dr_trans -- the original runs/no_aug (trained on the original
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
        # augmentation fully disabled
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        erasing=0.0,
        project=os.path.join(ROOT, "runs"),
        name="no_aug_dr_trans",
        exist_ok=True,
        verbose=True,
    )
    print("=== NO-AUG (dr_trans) TRAINING DONE ===")
    print("results dir:", results.save_dir)


if __name__ == "__main__":
    main()
