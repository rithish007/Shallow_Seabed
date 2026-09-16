"""Concise, fixed-cadence epoch logging for Ultralytics training runs.

By default Ultralytics prints a live-updating progress bar via '\\r' redraws
(many per epoch) plus a full per-class validation table every epoch. That is
fine in an interactive terminal, but once captured to a log file (or viewed in
a tool that treats '\\r' as a line break) it turns into thousands of lines for
a single training run.

`attach_epoch_logger(model, every=2)` silences that default per-batch/per-
validation output and replaces it with one plain, real-newline-terminated
summary line every `every` epochs (always including the final epoch), cutting
a 150-epoch run down to ~75 printed lines instead of thousands.
"""
import logging

# Silence ultralytics' own INFO-level chatter (per-batch progress bars, the
# per-class validation table, model summary, etc.) without touching WARNING/
# ERROR messages -- this also disables Ultralytics' TQDM bars, since TQDM
# auto-disables when this logger's effective level is above INFO.
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
    """Register callbacks that print one summary line every `every` epochs,
    plus one final line for Ultralytics' post-training best.pt re-validation."""

    def _on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1
        if epoch % every != 0 and epoch != trainer.epochs:
            return
        losses = " ".join(f"{k}={float(v):.4f}" for k, v in trainer.tloss.items())
        print(f"[epoch {epoch}/{trainer.epochs}] {losses} {_metrics_str(trainer)}", flush=True)

    def _on_train_end(trainer):
        # Fires once, right after BaseTrainer.final_eval() re-validates the
        # saved best.pt -- unlike inferring this from epoch arithmetic, this
        # fires correctly whether training ran to completion or stopped early
        # via patience (an epoch-count-based check would miss the early-stop
        # case, since Ultralytics' internal epoch bump during final_eval never
        # exceeds trainer.epochs when patience triggers first).
        print(f"[final best.pt validation] {_metrics_str(trainer)}", flush=True)

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    model.add_callback("on_train_end", _on_train_end)
