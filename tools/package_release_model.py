#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Package the frozen v0.1.0 models without executing checkpoint pickle code.

Requires the original SHA256-verified checkpoint and ONNX, the frozen demo image,
and a NEW output directory. Produces detector.onnx, detector-weights.pt,
detector-architecture.json and local inspection/verification reports. Only the
three detector files are release assets. No training or ONNX export is performed.

Example: python tools/package_release_model.py --checkpoint ORIGINAL_BEST_PT
  --onnx ORIGINAL_ONNX --image FROZEN_DEMO_PNG --output NEW_DIRECTORY
An optional --reference-raw path checks the preserved M4 PyTorch FP32 output.
"""
import argparse
from dataclasses import dataclass
import hashlib
import io
import importlib.metadata
import json
from pathlib import Path
import pickletools
import sys
import zipfile

try:
    from . import onnx_deploy as ref
    from .load_release_weights import load_model, read_payload
    from .verify_inference import raw_comparison, compare_detections
except ImportError:
    import onnx_deploy as ref
    from load_release_weights import load_model, read_payload
    from verify_inference import raw_comparison, compare_detections

ONNX_SHA256 = "1897b7c32c91b73665f9faa179739ae85e664530380971f3699c684ef6f82c35"
IMAGE_SHA256 = "43aa736cfa133817c93ea33269e7835a274577ef3a35c863c27546aa9e8bd9ba"
DESCRIPTION = "Ultralytics YOLO11n crop/weed model fine-tuned for SmartSpray Vision Controller on PhenoBench v1.1.0"


@dataclass(frozen=True)
class GlobalName:
    """An inert name from GLOBAL; never resolved, imported or invoked."""
    name: str


@dataclass
class Record:
    """An inert NEWOBJ/REDUCE result; BUILD only assigns an inert state."""
    kind: object
    args: object
    state: object = None


def inspect_pickle(data):
    """Interpret only the opcodes present in the frozen protocol-2 checkpoint.

    This is not pickle.loads/Unpickler: GLOBAL, REDUCE, NEWOBJ and BUILD cannot
    invoke code. They create plain records. Any unsupported opcode fails closed.
    The caller must verify the complete frozen checkpoint hash first.
    """
    stack, memo, mark = [], {}, object()

    def marked():
        index = len(stack) - 1
        while index >= 0 and stack[index] is not mark:
            index -= 1
        if index < 0:
            raise ValueError("missing pickle MARK")
        values = stack[index + 1:]
        del stack[index:]
        return values

    for opcode, arg, position in pickletools.genops(data):
        op = opcode.name
        if op == "PROTO":
            if arg != 2:
                raise ValueError("only the frozen protocol-2 checkpoint is supported")
        elif op == "MARK":
            stack.append(mark)
        elif op in {"BINPUT", "LONG_BINPUT"}:
            memo[arg] = stack[-1]
        elif op in {"BINGET", "LONG_BINGET"}:
            stack.append(memo[arg])
        elif op in {"BINUNICODE", "BININT", "BININT1", "BININT2", "BINFLOAT"}:
            stack.append(arg)
        elif op in {"NONE", "NEWTRUE", "NEWFALSE"}:
            stack.append({"NONE": None, "NEWTRUE": True, "NEWFALSE": False}[op])
        elif op == "EMPTY_DICT":
            stack.append({})
        elif op == "EMPTY_LIST":
            stack.append([])
        elif op == "EMPTY_TUPLE":
            stack.append(())
        elif op == "GLOBAL":
            stack.append(GlobalName(arg))
        elif op == "TUPLE":
            stack.append(tuple(marked()))
        elif op in {"TUPLE1", "TUPLE2", "TUPLE3"}:
            count = int(op[-1])
            values = tuple(stack[-count:])
            del stack[-count:]
            stack.append(values)
        elif op == "SETITEMS":
            values = marked()
            if len(values) % 2:
                raise ValueError("odd pickle mapping items")
            stack[-1].update(zip(values[::2], values[1::2]))
        elif op == "SETITEM":
            value, key = stack.pop(), stack.pop()
            stack[-1][key] = value
        elif op == "APPENDS":
            values = marked()
            stack[-1].extend(values)
        elif op == "APPEND":
            value = stack.pop()
            stack[-1].append(value)
        elif op in {"REDUCE", "NEWOBJ"}:
            args, kind = stack.pop(), stack.pop()
            if kind == GlobalName("collections OrderedDict"):
                stack.append(dict(*args))
            elif kind == GlobalName("__builtin__ set"):
                stack.append(set(*args))
            else:
                stack.append(Record(kind, args))
        elif op == "BUILD":
            state = stack.pop()
            if not isinstance(stack[-1], Record):
                raise ValueError("BUILD target is not an inert record")
            stack[-1].state = state
        elif op == "BINPERSID":
            stack.append(Record("storage", stack.pop()))
        elif op == "STOP":
            if position + 1 != len(data) or len(stack) != 1:
                raise ValueError("unexpected pickle tail/stack")
            return stack[0]
        else:
            raise ValueError(f"unsupported non-executing pickle opcode: {op}")
    raise ValueError("missing pickle STOP")


def extract_checkpoint(path):
    """Read registered parameters/buffers directly from trusted ZIP storages."""
    import torch
    if ref.sha256(path) != ref.CHECKPOINT_SHA256:
        raise ValueError("original checkpoint SHA256 mismatch; no inspection")
    if sys.byteorder != "little":
        raise RuntimeError("the frozen storage format requires a little-endian host")
    with zipfile.ZipFile(path) as archive:
        if archive.read("best/byteorder") != b"little":
            raise ValueError("unexpected checkpoint byte order")
        checkpoint = inspect_pickle(archive.read("best/data.pkl"))
        model = checkpoint["model"]
        if model.kind != GlobalName("ultralytics.nn.tasks DetectionModel") or model.state["names"] != ref.NAMES:
            raise ValueError("unexpected frozen checkpoint model")
        dtypes = {"torch HalfStorage": torch.float16, "torch FloatStorage": torch.float32,
                  "torch LongStorage": torch.int64}
        storages = {}

        def tensor(record):
            if record.kind == GlobalName("torch._utils _rebuild_parameter"):
                record = record.args[0]
            if record.kind != GlobalName("torch._utils _rebuild_tensor_v2"):
                raise ValueError("unexpected tensor reconstruction descriptor")
            storage, offset, shape, stride, _, _ = record.args
            tag, dtype_name, key, device, count = storage.args
            if storage.kind != "storage" or tag != "storage" or device != "cpu" or not key.isdecimal():
                raise ValueError("unexpected persistent storage descriptor")
            dtype = dtypes[dtype_name.name]
            if key not in storages:
                raw = archive.read("best/data/" + key)
                if len(raw) != count * torch.empty((), dtype=dtype).element_size():
                    raise ValueError("tensor storage size mismatch")
                storages[key] = torch.frombuffer(bytearray(raw), dtype=dtype)
            view = torch.as_strided(storages[key], shape, stride, offset)
            return view.clone(memory_format=torch.contiguous_format)

        state, roles = {}, {}

        def visit(module, prefix=""):
            data = module.state
            for group, role in [("_parameters", "parameter"), ("_buffers", "buffer")]:
                for name, value in data[group].items():
                    if value is not None and (role != "buffer" or name not in data["_non_persistent_buffers_set"]):
                        key = prefix + name
                        if key in state:
                            raise ValueError("duplicate registered tensor key")
                        state[key], roles[key] = tensor(value), role
            for name, child in data["_modules"].items():
                if child is not None:
                    visit(child, prefix + name + ".")

        visit(model)
        return checkpoint, state, roles


def digest_tensor(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def package(args):
    import numpy as np
    import onnx
    import torch
    expected_versions = {"torch": "2.13.0+cu126", "ultralytics": "8.4.163", "onnx": "1.18.0", "onnxruntime": "1.22.0"}
    versions = {name: importlib.metadata.version(name) for name in expected_versions}
    if versions != expected_versions:
        raise RuntimeError("packaging requires the recorded pinned model environment")
    for path, expected in [(args.checkpoint, ref.CHECKPOINT_SHA256), (args.onnx, ONNX_SHA256),
                           (args.image, IMAGE_SHA256)]:
        if ref.sha256(path) != expected:
            raise ValueError("frozen input SHA256 mismatch")
    if args.output.exists():
        raise ValueError("output directory must be new")
    original, state, roles = extract_checkpoint(args.checkpoint)
    # JSON serialization accepts only this plain architecture dictionary; model
    # instances, training history, local arguments and git context are excluded.
    cfg = original["model"].state["yaml"]
    architecture_bytes = (json.dumps(cfg, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    payload = {"format_version": 1, "architecture_sha256": hashlib.sha256(architecture_bytes).hexdigest(),
               "names": dict(ref.NAMES), "license": "AGPL-3.0 (https://ultralytics.com/license)", "upstream": "Ultralytics YOLO11n 8.4.163",
               "state_dict": state}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    weights_bytes = buffer.getvalue()
    model = onnx.load(args.onnx, load_external_data=False)
    before_graph = model.graph.SerializeToString(deterministic=True)
    description = [item for item in model.metadata_props if item.key == "description"]
    if len(description) != 1:
        raise ValueError("expected exactly one ONNX description")
    description[0].value = DESCRIPTION
    onnx.checker.check_model(model, full_check=True)
    args.output.mkdir(parents=True)
    (args.output / "detector-architecture.json").write_bytes(architecture_bytes)
    (args.output / "detector-weights.pt").write_bytes(weights_bytes)
    onnx.save(model, args.output / "detector.onnx")
    public = read_payload(args.output / "detector-weights.pt")
    if list(state) != list(public["state_dict"]) or not all(
            value.dtype == public["state_dict"][key].dtype and value.shape == public["state_dict"][key].shape
            and torch.equal(value, public["state_dict"][key]) for key, value in state.items()):
        raise RuntimeError("restricted reload changed a registered tensor")
    after = onnx.load(args.output / "detector.onnx")
    original_graph = onnx.load(args.onnx)
    next(item for item in original_graph.metadata_props if item.key == "description").value = DESCRIPTION
    if after.SerializeToString(deterministic=True) != original_graph.SerializeToString(deterministic=True):
        raise RuntimeError("ONNX changed beyond description metadata")
    if after.graph.SerializeToString(deterministic=True) != before_graph:
        raise RuntimeError("ONNX graph changed")
    # Inspect the public serialized string literals; no pickle code executes.
    with zipfile.ZipFile(io.BytesIO(weights_bytes)) as archive:
        data = archive.read(next(n for n in archive.namelist() if n.endswith("/data.pkl")))
    for opcode, value, _ in pickletools.genops(data):
        if isinstance(value, str) and any(x in value for x in ("/home/", "/mnt/", "\\Users\\", ".local/")):
            raise RuntimeError("private metadata remains in public weights")
    rebuilt = load_model(args.output / "detector-weights.pt", args.output / "detector-architecture.json", fuse=False)
    rebuilt_state = rebuilt.state_dict()
    if list(state) != list(rebuilt_state) or not all(torch.equal(value.float() if value.is_floating_point() else value,
                                                               rebuilt_state[key]) for key, value in state.items()):
        raise RuntimeError("strict architecture reconstruction changed tensor values")
    parameter_count = sum(p.numel() for p in rebuilt.parameters())
    rebuilt.fuse(verbose=False, imgsz=ref.SIZE)
    tensor, geometry = ref.preprocess(ref.decode(args.image))
    raw = ref.infer_torch(rebuilt, tensor)
    ort_original = ref.session(args.onnx).run(None, {"images": tensor})[0]
    ort_public = ref.session(args.output / "detector.onnx").run(None, {"images": tensor})[0]
    if not np.array_equal(ort_original, ort_public):
        raise RuntimeError("public ONNX inference changed")
    raw_report = raw_comparison(raw, ort_public)
    detections = compare_detections(ref.postprocess(raw, geometry), ref.postprocess(ort_public, geometry))
    if not all(row["pass"] for row in raw_report.values()) or not detections["pass"]:
        raise RuntimeError("PyTorch/ONNX unchanged numerical budget exceeded")
    reference_report = None
    if args.reference_raw is not None:
        reference = np.fromfile(args.reference_raw, dtype=np.float32).reshape(raw.shape)
        reference_report = {"sha256": ref.sha256(args.reference_raw), "raw_exact": bool(np.array_equal(reference, raw)),
                            "max_absolute_error": float(np.max(np.abs(reference - raw)))}
        if not reference_report["raw_exact"]:
            raise RuntimeError("reconstructed model differs from preserved PyTorch output")
    if ref.sha256(args.checkpoint) != ref.CHECKPOINT_SHA256 or ref.sha256(args.onnx) != ONNX_SHA256:
        raise RuntimeError("original model artifact changed")
    tensors = {key: {"role": roles[key], "dtype": str(value.dtype), "shape": list(value.shape),
                     "sha256": digest_tensor(value)} for key, value in state.items()}
    report = {"schema_version": 1, "result": "PASS", "method": "static checkpoint storage extraction; no pickle execution",
              "versions": versions,
              "original_checkpoint_sha256": ref.CHECKPOINT_SHA256, "original_onnx_sha256": ONNX_SHA256,
              "assets": {name: {"sha256": ref.sha256(args.output / name), "bytes": (args.output / name).stat().st_size}
                         for name in ["detector.onnx", "detector-weights.pt", "detector-architecture.json"]},
              "graph_sha256_unchanged": hashlib.sha256(before_graph).hexdigest(),
              "restricted_weights_only_reload": True, "all_registered_tensors_exact": True,
              "strict_architecture_load": True, "parameter_count": parameter_count, "tensors": tensors,
              "image_sha256": IMAGE_SHA256, "public_original_onnx_raw_exact": True,
              "preserved_pytorch_reference": reference_report, "pytorch_onnx_raw": raw_report,
              "pytorch_onnx_detections": detections, "original_artifacts_unchanged": True,
              "no_training_or_reexport": True}
    inspection = {"checkpoint_hash_verified": True, "onnx_hash_verified": True,
                  "checkpoint_top_level_fields": list(original),
                  "checkpoint_top_level_types": {key: type(value).__name__ for key, value in original.items()},
                  "original_checkpoint_bytes": args.checkpoint.stat().st_size,
                  "original_upstream_version": original["version"],
                  "original_license": original["license"],
                  "original_docs": original["docs"],
                  "onnx_node_count": len(model.graph.node),
                  "onnx_initializer_count": len(model.graph.initializer),
                  "onnx_external_data_count": sum(bool(value.external_data) for value in model.graph.initializer),
                  "onnx_original_private_string_fields": ["metadata.description"],
                  "checkpoint_private_metadata_categories": ["pretrained weight path", "derived dataset path", "project and save directories", "git root directory"],
                  "checkpoint_secret_like_key_literals_found": False,
                  "optimizer_scaler_ema_updates_modelopt_originally_null": all(original[key] is None for key in ["optimizer", "scaler", "ema", "updates", "modelopt"]),
                  "discarded_checkpoint_metadata": ["model instance and runtime attributes", "model.pt_path",
                      "model.args", "train_args", "git", "date", "epoch", "best_fitness", "train_metrics",
                      "train_results", "ema", "updates", "modelopt", "optimizer", "scaler"],
                  "preserved": ["all registered parameter/buffer values, shapes and dtypes", "exact architecture",
                                "crop=0, weed=1", "Ultralytics attribution and AGPL-3.0 notice"],
                  "onnx_change": "description metadata only; private derived-dataset path replaced by generic provenance",
                  "no_private_path_values_in_report": True,
                  "unrestricted_checkpoint_deserialization": "not used"}
    write_json(args.output / "inspection.json", inspection)
    write_json(args.output / "verification.json", report)
    print(f"PASS: {len(state)} tensors, {parameter_count} parameters; exact public/original ONNX output")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-raw", type=Path)
    package(parser.parse_args())


if __name__ == "__main__":
    main()