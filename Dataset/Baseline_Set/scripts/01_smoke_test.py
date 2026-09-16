"""Smoke test: confirm the training pipeline runs end-to-end on this GPU/data."""
import os

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "data.yaml")


def main():
    model = YOLO("yolo26n.pt")
    results = model.train(
        data=DATA_YAML,
        epochs=3,
        imgsz=640,
        batch=16,
        device=0,
        amp=True,
        project=os.path.join(ROOT, "runs"),
        name="smoke_test",
        exist_ok=True,
        verbose=True,
    )
    print("=== SMOKE TEST DONE ===")
    print("results dir:", results.save_dir)


if __name__ == "__main__":
    main()
