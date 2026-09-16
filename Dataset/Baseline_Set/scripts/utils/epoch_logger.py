"""Concise, fixed-cadence epoch logging for Ultralytics training runs."""
import logging

logging.getLogger("ultralytics").setLevel(logging.WARNING)


def _metrics_str(trainer):
    m = trainer.metrics
    return (
        f"P={m.get('metrics/precision(B)', 0.0):.4f} "
        f"R={m.get('metrics/recall(B)', 0.0):.4f} "
        f"mAP50={m.get('metrics/mAP50(B)', 0.0):.4f} "
        f"mAP50-95={m.get('metrics/mAP50-95(B)', 0.0):.4f}"
    )


def attach_epoch_logger(model, every=2):

    def _on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1
        if epoch % every != 0 and epoch != trainer.epochs:
            return
        losses = " ".join(f"{k}={float(v):.4f}" for k, v in trainer.tloss.items())
        print(f"[epoch {epoch}/{trainer.epochs}] {losses} {_metrics_str(trainer)}", flush=True)

    def _on_train_end(trainer):
        print(f"[final best.pt validation] {_metrics_str(trainer)}", flush=True)

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    model.add_callback("on_train_end", _on_train_end)
