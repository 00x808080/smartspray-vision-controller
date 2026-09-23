#!/usr/bin/env python3
"""Execute one bounded YOLO11n experiment with native train/val operations."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import subprocess
import time

try:
    from .ml_runtime import BaselineTrainer
    from .yolo_data import POLICY, sha256_file
except ImportError:
    from ml_runtime import BaselineTrainer
    from yolo_data import POLICY, sha256_file

import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import Metric


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str)+"\n")


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: refusing CPU fallback")
    output = args.output.resolve()
    if output.exists():
        raise ValueError("experiment output already exists; inspect its process/checkpoints before resuming")
    config = yaml.safe_load(args.config.read_text())
    config.update(data=str(args.data.resolve()), model=str(args.weights.resolve()),
                  project=str(output.parent), name=output.name, exist_ok=False)
    if args.smoke:
        config.update(epochs=2, patience=10)
    if args.batch is not None:
        config["batch"] = args.batch
    if args.workers is not None:
        config["workers"] = args.workers
    if config["epochs"] > 50 or config["time"] is not None or config["device"] != 0:
        raise ValueError("run must have <=50 epochs, separate time budget, and GPU 0")
    # Freeze and verify actual installed best.pt selection semantics.
    metric = Metric()
    metric.p = __import__("numpy").array([0.2])
    metric.r = __import__("numpy").array([0.3])
    metric.all_ap = __import__("numpy").array([[0.9]+[0.1]*9])
    if abs(metric.fitness()-metric.map) > 1e-12:
        raise RuntimeError("installed fitness differs from mAP50-95")
    initial = {"requested": config, "policy": POLICY,
               "selection": "maximum validation mAP50-95; native best.pt overwrites on equal fitness",
               "weight_sha256": sha256_file(args.weights),
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "source_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
               "source_files_sha256": {str(p): sha256_file(p) for folder in ("tools", "configs") for p in sorted(Path(folder).glob("*")) if p.is_file()},
               "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "ultralytics", "numpy", "pillow")},
               "time_budget_seconds": args.max_seconds, "gpu": torch.cuda.get_device_name(0)}
    model = YOLO(str(args.weights.resolve()))
    started = time.monotonic()
    requested_stop = []
    epoch_records = []
    state = {}

    def request_stop(signum, frame):
        requested_stop.append(f"signal_{signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)

    def on_start(trainer):
        state["train_start"] = time.monotonic()
        state["initial_parameters_sha256"] = state_hash(trainer.model)
        write_json(trainer.save_dir/"run-provenance.json", initial)
        write_json(trainer.save_dir/"effective-config.json", vars(trainer.args))

    def batch_end(trainer):
        if not torch.isfinite(trainer.loss).all():
            raise RuntimeError("nonfinite training loss")

    def epoch_end(trainer):
        if not all(torch.isfinite(p).all() for p in trainer.model.parameters()):
            raise RuntimeError("nonfinite trained parameters")

    def fit_end(trainer):
        if getattr(trainer, "final_validation_started", False):
            return
        record = {"epoch": trainer.epoch+1, "elapsed_seconds": time.monotonic()-state["train_start"],
                  "metrics": {k: float(v) for k, v in trainer.metrics.items()},
                  "losses": {k: float(v) for k, v in trainer.tloss.items()},
                  "fitness": float(trainer.fitness), "best_fitness": float(trainer.best_fitness),
                  "batch": trainer.batch_size, "accumulate": trainer.accumulate,
                  "optimizer_groups": [{"lr": g["lr"], "weight_decay": g["weight_decay"]} for g in trainer.optimizer.param_groups]}
        epoch_records.append(record)
        write_json(trainer.save_dir/"epochs.json", epoch_records)
        write_json(trainer.save_dir/"effective-config.json", vars(trainer.args))
        if requested_stop or record["elapsed_seconds"] >= args.max_seconds:
            trainer.stop = True
            state["stop_reason"] = requested_stop[-1] if requested_stop else "time_budget"

    model.add_callback("on_train_start", on_start)
    model.add_callback("on_train_batch_end", batch_end)
    model.add_callback("on_train_epoch_end", epoch_end)
    model.add_callback("on_fit_epoch_end", fit_end)
    model.train(trainer=BaselineTrainer, **config)
    trainer = model.trainer
    final_hash = state_hash(trainer.model)
    if state["initial_parameters_sha256"] == final_hash:
        raise RuntimeError("training did not update model parameters")
    checkpoints = {p.name: {"path": str(p), "sha256": sha256_file(p)} for p in trainer.wdir.glob("*.pt")}
    report = {**state, "epochs_recorded": len(epoch_records),
              "elapsed_seconds_including_setup_final_validation": time.monotonic()-started,
              "training_seconds": epoch_records[-1]["elapsed_seconds"], "parameters_updated": True,
              "final_parameters_sha256": final_hash, "checkpoints": checkpoints,
              "stop_reason": state.get("stop_reason", "epoch_limit" if len(epoch_records) >= config["epochs"] else "early_stopping")}
    write_json(output/"completion.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/yolo11n-baseline.yaml"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-seconds", type=float, default=10800)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 10800 or (args.workers is not None and not 0 <= args.workers <= 4):
        parser.error("time budget must be in (0, 10800], workers in [0, 4]")
    train(args)


if __name__ == "__main__":
    main()
