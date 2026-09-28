#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Load v0.1.0 tensor-only weights with the pinned YOLO11n architecture.

CLI: python tools/load_release_weights.py --weights detector-weights.pt
     --architecture detector-architecture.json --image sample.png --output predictions.json
The weights use torch.load(weights_only=True), without additional safe globals.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path

try:
    from . import onnx_deploy as ref
except ImportError:
    import onnx_deploy as ref


def read_payload(path):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if type(payload) is not dict or set(payload) != {"format_version", "architecture_sha256", "names", "license", "upstream", "state_dict"}:
        raise ValueError("unexpected release weights structure")
    if (payload["format_version"] != 1 or payload["names"] != ref.NAMES
            or payload["license"] != "AGPL-3.0 (https://ultralytics.com/license)"
            or payload["upstream"] != "Ultralytics YOLO11n 8.4.163"):
        raise ValueError("unexpected release weights metadata")
    state = payload["state_dict"]
    if type(state) is not dict or not state or not all(type(k) is str and type(v) is torch.Tensor for k, v in state.items()):
        raise ValueError("state_dict must contain only named tensors")
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("nonfinite model tensor")
    return payload


ARCHITECTURE_SHA256 = "6c32e11f48c3d3156c5f35eaad2cd6d90c7bc5f8d0c475de284e30510050116d"


def load_model(weights, architecture, *, fuse=True):
    """Return a CPU FP32 eval model; fuse=False retains editable Conv/BN layers."""
    import torch
    if importlib.metadata.version("ultralytics") != "8.4.163":
        raise RuntimeError("this architecture is verified with ultralytics==8.4.163")
    from ultralytics.nn.tasks import DetectionModel
    payload = read_payload(weights)
    raw = Path(architecture).read_bytes()
    if (hashlib.sha256(raw).hexdigest() != ARCHITECTURE_SHA256
            or payload["architecture_sha256"] != ARCHITECTURE_SHA256):
        raise ValueError("architecture differs from the frozen two-class YOLO11n descriptor")
    cfg = json.loads(raw)
    # The independent fixed digest above binds the exact audited descriptor.
    # Never resolve a YAML filename from CWD or trust a payload-supplied digest.
    torch.set_num_threads(1)
    # Constructor initializes temporary random parameters before strict loading;
    # restore the caller's RNG state and never train or export a model here.
    with torch.random.fork_rng(devices=[]):
        model = DetectionModel(deepcopy(cfg), ch=3, nc=2, verbose=False).cpu().float()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.names = dict(ref.NAMES)
    model.eval()
    if model.model[-1].nc != 2 or model.end2end:
        raise ValueError("unexpected reconstructed detection head")
    if fuse:
        model.fuse(verbose=False, imgsz=ref.SIZE)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--architecture", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must be a new file")
    model = load_model(args.weights, args.architecture)
    tensor, geometry = ref.preprocess(ref.decode(args.image))
    predictions = ref.postprocess(ref.infer_torch(model, tensor), geometry)
    # Exclusive creation preserves existing artifacts, including input aliases.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(predictions, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Loaded tensor-only weights; wrote {len(predictions['detections'])} predictions")


if __name__ == "__main__":
    main()