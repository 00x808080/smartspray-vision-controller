# SmartSpray crop/weed detector — v0.1.0

This model supports a single-image portfolio demonstration, with simulated geometry and eight-channel actuation. Predictions are not ground truth; no calibration, field safety, crop protection, real-time or unseen-field generalization claim is made.

## Source, architecture and training

The corresponding application and packaging source is the annotated [v0.1.0 tag](https://github.com/00x808080/smartspray-vision-controller/tree/v0.1.0). The release asset `SOURCE_REVISION.txt` records its exact commit and annotated-tag object. [Training configuration](https://github.com/00x808080/smartspray-vision-controller/blob/v0.1.0/configs/yolo11n-baseline.yaml), [data preparation](https://github.com/00x808080/smartspray-vision-controller/blob/v0.1.0/docs/DATA.md#reproduction), [training/evaluation](https://github.com/00x808080/smartspray-vision-controller/blob/v0.1.0/docs/BASELINE.md#reproduction), [export implementation](https://github.com/00x808080/smartspray-vision-controller/blob/v0.1.0/tools/onnx_deploy.py) and [export instructions](https://github.com/00x808080/smartspray-vision-controller/blob/v0.1.0/docs/INFERENCE.md#verification-and-reproduction) are available with exact dependency pins.

- **Architecture:** Ultralytics YOLO11n detection, 2,590,230 parameters before fusion; three input channels; crop=0, weed=1, no background class. `detector-architecture.json` preserves the exact embedded YOLO11 configuration (scale `n`, `nc=2`), loaded with Ultralytics 8.4.163.
- **Initialization:** official [COCO-pretrained yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt), SHA256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- **Data:** PhenoBench v1.1.0 official train/validation split: 1,407/772 images, 20,016/10,408 valid object appearances. All valid partial and tiny objects are retained, including one-pixel boxes; raw crop labels 1/3 map to 0 and weed labels 2/4 to 1. Project AP does not apply the official PhenoBench ignore filter.
- **One main run:** imgsz=1024, batch=8, workers=4, seed=42, AdamW lr0=.001, weight decay=.0005, AMP, maximum 50 epochs, patience=10. It completed 31 epochs; epoch 21 was selected by validation mAP50–95. Training-time selection fitness .63635 is distinct from final FP32 evaluation. No architecture search or retraining was done for this release.
- **Exact historical sources:** `training-source.zip` contains only the seven original code/configuration files verified against pre-run fingerprints, their manifest and notices. The run started at Git base `7bb70fb95b15e401a74b857d4552e4aa17c07969` with uncommitted changes, so that base commit alone is not the executed source. Later maintained changes concern precision/evaluation API, local settings and interruption/resume handling. The archived evaluator predates the final FP32 evaluator fixes; use the linked maintained evaluator for metric reproduction. Identical retraining bytes are not promised.

## Files and hashes

| Artifact | SHA256 |
|---|---|
| Original local `best.pt` (provenance; not uploaded) | `ea861ff9ba54a768dffd62814944d157e50037a67fd268ea65fe0d7773196fd6` |
| Original local ONNX (provenance; not uploaded) | `1897b7c32c91b73665f9faa179739ae85e664530380971f3699c684ef6f82c35` |
| Public `detector.onnx` | `13c03f6f8c189f7fd3ef440ec8cc823ac0edaed7390b40ddbaf5d2f683fab348` |
| Public `detector-weights.pt` | `722b9a43ad167ac2bd27f9b1241f0aa276068e8abcdb144e92c074d89a1fb944` |
| Public `detector-architecture.json` | `6c32e11f48c3d3156c5f35eaad2cd6d90c7bc5f8d0c475de284e30510050116d` |
| `training-source.zip` | `4b1e702b0a17c5c2bcbd1581d1b1cb135edfccdab721ae95c5e1a19a43dbbb21` |

Verify downloaded files against the versioned release's `SHA256SUMS.txt`. Model/data binaries are outside Git. The original model, checkpoint, evidence and dataset remain unchanged.

## Metadata-only publication packaging

The [packager](https://github.com/00x808080/smartspray-vision-controller/blob/v0.1.0/tools/package_release_model.py) accepts only the hash-verified original artifacts and creates a new directory. ONNX changes **only** `metadata_props.description`, replacing a private derived-dataset path with a generic, truthful provenance description. Upstream author, license, docs, version, export date, I/O and graph remain unchanged. The graph SHA256 is `e429bed0769e6a39694bb841ff73b6e7fcf54aeb84449ccfb0c00716853aaca1`.

The checkpoint is parsed without executing pickle code. All 499 registered parameters/buffers retain exact values, shapes and stored dtypes. The publication weights contain only tensors and primitive metadata, including the original upstream license notice. Exact architecture and names are supplied separately. Removed objects/metadata: original model instance/runtime attributes, `pt_path`, model/train argument dictionaries, Git metadata, date/epoch/fitness and training-result histories, and null EMA/update/modelopt/optimizer/scaler fields. No optimizer or private environment object is distributed.

The pinned environment reloads publication weights with `torch.load(weights_only=True)` and no extra safe globals, strictly reconstructs the architecture, and verifies every tensor. On the fixed showcase sample, reconstructed fused FP32 PyTorch output is bit-exact to the preserved pre-packaging PyTorch output; original/public ONNX outputs are bit-exact. Existing PyTorch/ONNX raw-output and final-detection budgets pass. Native CSV and PNG outputs match the frozen reference; the only JSON difference is `model.sha256`, reflecting the changed metadata bytes. No graph export, retraining or new full validation was performed for packaging.

## Input/output and runtime

ONNX FP32, opset 18, IR 8; fixed input `images [1,3,1024,1024]`, output `output0 [1,6,21504]`. Output channels are decoded center x/y, width/height in input pixels, then crop/weed probabilities; no objectness, extra sigmoid or embedded NMS. CPU ONNX Runtime 1.22.0 uses sequential execution, graph optimization ALL and one intra/inter-op thread.

Native input is an 8-bit three-channel PNG/BMP. Preprocessing: BGR→RGB, aspect-preserving centered 1024 letterbox, scale-up enabled, padding 114, FP32 division by 255 and NCHW. Deployment keeps score >.25, one class per candidate, deterministic class-aware NMS at IoU >.7, max 300 detections, then restores original-image xyxy. Exact rounding, ordering and runtime contracts are in [INFERENCE](https://github.com/00x808080/smartspray-vision-controller/blob/v0.1.0/docs/INFERENCE.md).

## Loading and modifying the weights

Use the [pinned Python environment](https://github.com/00x808080/smartspray-vision-controller/blob/v0.1.0/docs/REPRODUCIBILITY.md#existing-python-suites), then run from the v0.1.0 source root. Set `PYTHON` to that environment's executable; `MODEL_DIR` to the downloaded assets directory; `IMAGE` to a supported original input; and `PREDICTIONS` to a new output file. Paths with spaces are supported.

```bash
"$PYTHON" tools/load_release_weights.py \
  --weights "$MODEL_DIR/detector-weights.pt" \
  --architecture "$MODEL_DIR/detector-architecture.json" \
  --image "$IMAGE" --output "$PREDICTIONS"
```

For parameter inspection/modification, `tools.load_release_weights.load_model(weights, architecture, fuse=False)` returns the strictly reconstructed CPU FP32 module with separate Conv/BatchNorm layers. The weights preserve checkpoint FP16 values exactly and are converted to FP32 on loading. Calling the default `fuse=True` follows the frozen deployment path. The weights file is intentionally not an Ultralytics object checkpoint and is not passed directly to `YOLO(path)` or the historical exporter. The linked original export implementation documents FP32 fusion, fixed batch=1, imgsz=1024, opset=18, nms=False, dynamic=False, simplify=False and quantize=None. The released ONNX is the original exported graph with description metadata sanitized; no new export is needed to run it.

## Measured results and limitations

Historical **M2.2**, selected checkpoint, fresh-process PyTorch FP32/GPU/batch8 evaluation on the official 772-image validation split: all/crop/weed mAP50–95 **.635192/.800301/.470084**. Historical **M3**, exported ONNX FP32/ORT1.22.0 CPU/batch1 on that same split: **.635255/.800328/.470182**. See the correct experiments in [BASELINE](https://github.com/00x808080/smartspray-vision-controller/blob/v0.1.0/docs/BASELINE.md#measured-result) and [INFERENCE](https://github.com/00x808080/smartspray-vision-controller/blob/v0.1.0/docs/INFERENCE.md#validation-regression-and-timing). These are prior measured results, not a release-time 772-image rerun or official leaderboard score.

Validation selected the checkpoint; 537 crop IDs overlap train and validation, so results do not establish independent unseen-field generalization. Partial/small weeds, inaccurate localization, crop/weed confusion and crowded/border boxes are known limitations. The showcase is training image `05-15_00028_P0030852.png`; it is illustrative, not validation evidence. Predicted weed-box centers are simulation anchors, not verified stems.

Tested native environment: Ubuntu 24.04.5 / WSL 2 x86_64, GCC 13.3.0, CMake 3.28.3, system OpenCV 4.6.0, ORT 1.22.0. Packaging: Python 3.12.3, torch 2.13.0+cu126, Ultralytics 8.4.163, ONNX 1.18.0. The release is not a clean-OS/second-machine test or portability guarantee.

## License and data conditions

Own project source: Copyright (C) 2026 Sergey Gonchar, **AGPL-3.0-only**; [license scope](https://github.com/00x808080/smartspray-vision-controller/blob/v0.1.0/NOTICE.md). The YOLO11n architecture, official pretrained weights and derived models retain [Ultralytics AGPL-3.0 terms](https://www.ultralytics.com/license); original attribution/license metadata is preserved. The project's source-code notice does not replace upstream conditions.

PhenoBench authors: Jan Weyler, Federico Magistri, Elias Marks, Yue Linn Chong, Matteo Sodano, Gianmarco Roggiolani, Nived Chebrolu, Cyrill Stachniss and Jens Behley; *PhenoBench — A Large Dataset and Benchmarks for Semantic Image Interpretation in the Agricultural Domain*, IEEE TPAMI 46(12), 9583–9594, 2024. Data and annotated media remain separate: the [official dataset page](https://www.phenobench.org/dataset.html) states [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); the original archive README states CC BY-NC-SA 4.0. Publication of the single annotated illustration relies on the official page's BY-SA notice accepted by the owner; the discrepancy is retained in [attribution](https://github.com/00x808080/smartspray-vision-controller/blob/v0.1.0/docs/assets/ATTRIBUTION.md). No endorsement is claimed. The release does not distribute the dataset or assert that one code license replaces model/data/dependency conditions.
