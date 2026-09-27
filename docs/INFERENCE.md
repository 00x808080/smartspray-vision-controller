# Native ONNX inference and consistency (M3)

M3 exports the existing two-class YOLO11n and runs image inference in C++17. The controller source and its 40 scenarios are unchanged. The controller bridge is a numerical consistency harness; M4 presentation has not started.

## Export and dependencies

| Item | Verified value |
|---|---|
| Source checkpoint | `.local/m2.2/main/weights/best.pt` |
| Checkpoint SHA256, before/after | `ea861ff9ba54a768dffd62814944d157e50037a67fd268ea65fe0d7773196fd6` |
| ONNX artifact | `.local/m3/model/detector.onnx` |
| ONNX SHA256 | `1897b7c32c91b73665f9faa179739ae85e664530380971f3699c684ef6f82c35` |
| Export stack | Python 3.12.3, PyTorch 2.13.0+cu126, Ultralytics 8.4.163, ONNX 1.18.0 |
| Runtime | Official ONNX Runtime **1.22.0**, Python CPU wheel and Linux x64 C/C++ release |
| Image libraries | Python OpenCV 5.0.0.93; Ubuntu C++ OpenCV 4.6.0+dfsg-13.1ubuntu1 |
| Serializer / compiler | nlohmann-json 3.11.3; GCC 13.3.0, C++17, CMake 3.28.3 |
| Graph | opset **18**, actual IR **8**; ONNX full checker and runtime load passed |

The installed exporter was inspected, including its legacy `torch.onnx.export(dynamo=False)` call. An independent FP32 inference copy is fused as in native prediction. Actual export arguments are `format=onnx, imgsz=1024, batch=1, device=cpu, quantize=None, nms=False, dynamic=False, simplify=False, opset=18`. YOLO11 has no one-to-one head: the pinned exporter warns and retains its one-to-many decoded output. Graph inspection confirms no embedded NMS. No manual IR/opset rewrite, quantization, graph slimming or training is performed.

The graph has one FP32 input `images [1,3,1024,1024]` and one FP32 output `output0 [1,6,21504]`. Channels are decoded center-x, center-y, width, height in input pixels, then crop/weed probabilities. There is no objectness column and no additional sigmoid or box decoding. Metadata declares exactly `0=crop, 1=weed`, detection, and `end2end=False`. C++ validates types, shapes, positive candidate count and metadata; it reads N from the actual graph.

Both runtimes explicitly use CPUExecutionProvider, sequential execution, intra/inter-op threads **1/1**, and graph optimization **ALL**. OpenCV C++ uses one thread. ONNX Runtime 1.22.0 was selected because its official release provides the Linux x64 C/C++ archive and matching Python wheel; 1.22.1 had no Linux x64 release asset.

Only six pinned packages were added to the existing ML environment, using uv with `--no-deps`; all original 60 package versions remain unchanged. `YOLO_AUTOINSTALL=false` prevents exporter-driven upgrades. The original data environment is retained. See [pins](../requirements-inference.txt), [official installation](https://onnxruntime.ai/docs/install/) and [compatibility](https://onnxruntime.ai/docs/reference/compatibility.html).

| Official artifact | SHA256 |
|---|---|
| [ONNX 1.18.0](https://pypi.org/project/onnx/1.18.0/) CPython 3.12 Linux wheel | `99afac90b4cdb1471432203c3c1f74e16549c526df27056d39f41a9a47cfb4af` |
| [ORT 1.22.0](https://pypi.org/project/onnxruntime/1.22.0/) CPython 3.12 Linux CPU wheel | `6964a975731afc19dc3418fad8d4e08c48920144ff590149429a5ebe0d15fb3c` |
| [ORT Linux x64 1.22.0 archive](https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz) | `8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3` |
| Protobuf 6.32.1 cp39-abi3 Linux wheel | `b1864818300c297265c83a4982fd3169f97122c299f56a56e2445c3698d34710` |
| Flatbuffers 25.2.10 wheel | `ebba5f4d5ea615af3f7fd70fc310636fbb2bbd1f566ac0a23d98dd412de50051` |
| Coloredlogs 15.0.1 wheel | `612ee75c546f53e92e70049c9dbfcc18c935a2b9a53b66085ce9ef6a6e5c0934` |
| Humanfriendly 10.0 wheel | `1697e1a8a8f550fd43c2865cd84542fc175a61dcb779b6fee18cf6b6ccba1477` |

Exact PyPI artifact URLs, installed wheel payload verification, Ubuntu package versions/archive digests and source hashes are retained in the local review bundle. Third-party binaries remain under `.local/m3/deps`, outside Git. The existing dataset and Ultralytics licensing findings in [DATA](DATA.md) and [BASELINE](BASELINE.md) are unchanged; no new redistribution rights or project license are asserted.

## Image and output contract

Supported files are **8-bit, three-channel PNG/BMP**, verified by encoding signature and decoded type. OpenCV decodes BGR with `IMREAD_UNCHANGED`, so EXIF orientation is ignored. Grayscale, alpha, 16-bit, corrupt and unsupported files fail explicitly.

Letterbox is fixed 1024 square, aspect-preserving, **scale-up enabled**, centered, no stride-auto padding, padding value 114, `INTER_LINEAR`. Compute `r=min(1024/width,1024/height)`, resize dimensions with Python ties-to-even rounding, and split padding with `round(d/2-0.1)` / `round(d/2+0.1)`. Retain the actual r and integer left/top padding. Convert BGR→RGB, divide uint8 values by 255 in FP32, and store contiguous NCHW. Synthetic odd-padding and exact rounding-tie cases verify the C++ implementation against the pinned native LetterBox.

Postprocessing selects one class per candidate (a score tie chooses crop), accepts **score > 0.25**, sorts by descending score then ascending original graph candidate index, and suppresses same-class boxes only for **IoU > 0.7**. At the boundaries, score exactly 0.25 is excluded and IoU exactly 0.7 is retained. Stop at 300 detections. NMS occurs in input coordinates before clipping. Restore floating xyxy using actual padding/r and clip to original image bounds, without integer coordinate rounding.

Deterministic JSON contains `schema_version=1`, `original_width`, `original_height`, class names and detections with `class_id`, `class_name`, `score`, original-image `xyxy`, and `candidate_index`. Candidate indices are diagnostic identifiers, not plant identities. A valid empty prediction succeeds with an empty detections array. Model/schema errors, nonfinite tensors and inference/write failures exit nonzero. Input/output aliases, including hard links, are rejected.

## Frozen comparisons and measured results

The manifest was frozen before comparison: **32 real validation images** (existing M2.2 examples, smallest class/partial/edge objects and deterministic sorted selection), plus **6 synthetic color/geometry fixtures**. Synthetic predictions are numerical fixtures, not detector-quality evidence. Manifest SHA256: `5fa30b7991a3bed1c470bd818b7dc5255ea61147acaeb9af7e67d26e4e3a4c97`. No images were removed or tolerances widened.

| Layer | Frozen gate | Actual result across 38 samples |
|---|---|---|
| A: decode/geometry | Exact lossless pixels and geometry | Exact; native LetterBox also exact |
| A: normalized input | No resize ≤1e-6; resized ≤1/255+1e-6 | **0** max error; **0** changed pixels, including resized fixtures |
| B: PyTorch→Python ORT boxes | Normalize coordinates by 1024; atol=rtol=1e-4 | Max **6.52671e-6**, mean **2.93449e-8** |
| B: raw class probabilities | atol=rtol=1e-4 | Max **9.95398e-6**, mean **2.68513e-8** |
| C: Python→C++ ORT, identical tensors | Same separate raw budgets | **Bit-exact** boxes and scores |
| D: final matched boxes/scores | ≤0.5 original pixels / ≤1e-4 absolute | PyTorch→ORT max **0.000244141 px / 2.74181e-6**; Python ORT→C++ exact |

All gates pass. Matching uses maximum-cardinality one-to-one same-class geometry overlap (IoU≥0.5), never array position. **447 real + 20 synthetic detections** match; zero unmatched detections, class changes or retained-candidate changes. Native pinned Ultralytics NMS independently agrees with custom postprocessing on both PyTorch/ORT raw predictions. Reports retain maximum/mean errors, candidate indices and every pairing. No confidence crossings, active class-argmax crossings or candidates/pairs within 1e-4 of confidence/NMS thresholds occurred on this sample; synthetic tests separately exercise these boundaries.

### Same-controller effect

Only weed box centers become illustrative targets:
`x_m=2*u/width`, `forward_m=2*(1-v/height)`.
Frozen pitch=0.25 m, speed=2 m/s, capture=1,000,000 us, now=1,100,000 us, actuator delay=50,000 us, pulse=100,000 us. This mapping is not camera calibration; inference latency is never virtual time.

All three paths pass their floating coordinates through the **same unchanged C++ `plan_batch` and `VirtualExecutor`**. The 32 real images contain **173 matched weed targets**, zero unmatched targets, zero channel/rejection changes, **0 us** maximum arrival/ON/OFF difference, and identical merged intervals and executed event structures/timestamps. Every final channel state is OFF. Exact commands are established only for this fixed sample.

Synthetic nextafter tests expose an internal channel change, swath rejection, accepted-versus-TOO_LATE transition, and a **1 us** travel/command difference. At the frozen lateness geometry, the actual ceil transition is near `forward_m=0.299998`; neighboring representable doubles are checked directly. No coordinate rounding/epsilon conceals sensitivity.

## Validation regression and timing

The ONNX graph was evaluated **once** on all **772 official validation images / 10,408 annotated objects** with the unchanged PreservingValidator and native AP code. All partial/tiny objects remain included. AP uses confidence floor=0.001, class-aware IoU=0.7, **multi_label=True**, max_det=300, FP32, no augmentation, LetterBox/Format. Deployment filtering is not used for AP. Required runtime differences are CPU and batch=1 instead of the M2.2 GPU/batch=8; native CPU workers=0 and plots disabled do not alter the metric protocol. Original M2.2 reports remain intact.

| Class | ONNX mAP50–95 | Signed Δ | ONNX AP50 | Signed Δ | ONNX AP75 | Signed Δ |
|---|---:|---:|---:|---:|---:|---:|
| All | 0.63525501 | +0.00006268 | 0.83816223 | +0.00003927 | 0.67486900 | +0.00001359 |
| Crop | 0.80032770 | +0.00002658 | 0.94115386 | −0.00001238 | 0.85623251 | +0.00006797 |
| Weed | 0.47018232 | +0.00009877 | 0.73517061 | +0.00009092 | 0.49350548 | −0.00004080 |

Absolute deltas are the magnitudes above and are also stored explicitly in the report. All are below the predeclared 0.005 investigation threshold. This is the project all-annotated-objects protocol, not PhenoBench leaderboard equivalence or unseen-field testing.

C++ Release timing: AMD Ryzen 7 7800X3D, Ubuntu 24.04 / WSL2 x86_64, 16 logical CPUs, one ORT/OpenCV thread, no explicit CPU affinity. Image `05-15_00052_P0030859.png`, one reused session, five timed-loop warmups after initial output generation, **30 measured runs**. Other project tests/evaluation had finished. Model/session loading was **59.336 ms** (one observation).

| Stage | Median ms | p95 ms |
|---|---:|---:|
| Decode + letterbox + normalization | 19.170 | 19.846 |
| Inference, including input validation/output copy and finite checks | 228.826 | 234.757 |
| Postprocessing | 0.116 | 0.157 |

Percentiles use linear interpolation; JSON/file output is outside these stages. This is a basic local CPU measurement, with no optimization campaign or real-time claim.

## Verification and reproduction

Controller-only Debug/Release: original **40/40 scenarios** each. Vision Debug/Release: **15 offline C++ scenarios**, **6 controller-boundary cases**, and a separately registered real-model determinism smoke pass. Without artifact paths the real-model test explicitly reports **NOT RUN / CTest SKIP 77**. ASan + UBSan + float-cast-overflow pass the host-side controller/vision/harness tests with leak detection and halt-on-error enabled; third-party libraries themselves are prebuilt and uninstrumented.

The full Python suite passes **88 tests**, including the original 52, 20 reference/matching tests, 6 AP-summary tests, and 10 compiled-CLI tests using tiny synthetic ONNX graphs. The original data environment separately passes 36/36. Unit tests require neither private checkpoint nor full dataset. Compiled-CLI tests explicitly skip without `SMARTSPRAY_INFER`. Input/output alias regression tests preserve image/model/harness inputs, including hard links.

Executed command forms from the Linux repository root follow. Create new export/manifest/comparison/evaluation outputs; existing evidence is deliberately not overwritten.

```bash
.local/m2.1/uv/uv pip install --python .venv-ml/bin/python --no-deps --only-binary :all: -r requirements-inference.txt --index-url https://pypi.org/simple
.local/m2.1/uv/uv pip check --python .venv-ml/bin/python

.venv-ml/bin/python tools/onnx_deploy.py --checkpoint .local/m2.2/main/weights/best.pt --output .local/m3/model/detector.onnx
.venv-ml/bin/python tools/verify_inference.py freeze --derived .local/m2.2/dataset --baseline-examples .local/m2.2/evaluation/inspected-predictions.json --output .local/m3/sample-manifest.json

cmake -S . -B build-vision-release -DCMAKE_BUILD_TYPE=Release -DSMARTSPRAY_BUILD_VISION=ON -DONNXRUNTIME_ROOT="$PWD/.local/m3/deps/onnxruntime-linux-x64-1.22.0" -DSMARTSPRAY_REAL_MODEL="$PWD/.local/m3/model/detector.onnx" -DSMARTSPRAY_REAL_IMAGE="$PWD/.local/m2.2/dataset/images/val/05-15_00052_P0030859.png"
cmake --build build-vision-release --parallel 2
ctest --test-dir build-vision-release --output-on-failure --no-tests=error --verbose

./build-vision-release/smartspray_infer --model .local/m3/model/detector.onnx --image .local/m2.2/dataset/images/val/05-15_00052_P0030859.png --output .local/m3/timing-prediction.json --benchmark 30 --timing-output .local/m3/timing.json
.venv-ml/bin/python tools/verify_inference.py compare --manifest .local/m3/sample-manifest.json --checkpoint .local/m2.2/main/weights/best.pt --model .local/m3/model/detector.onnx --executable build-vision-release/smartspray_infer --controller build-vision-release/controller_parity --output .local/m3/parity
.venv-ml/bin/python tools/evaluate_onnx.py --model .local/m3/model/detector.onnx --derived .local/m2.2/dataset --baseline .local/m2.2/evaluation/metrics.json --output .local/m3/evaluation

SMARTSPRAY_INFER="$PWD/build-vision-release/smartspray_infer" .venv-ml/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python -m unittest discover -s tests -p test_phenobench_data.py -v
```

The minimal Ubuntu packages are `libopencv-core-dev libopencv-imgproc-dev libopencv-imgcodecs-dev nlohmann-json3-dev`; installed versions/digests are recorded locally. Unpack the linked official ORT archive under `.local/m3/deps` and verify its SHA256 above. No runtime compilation is required. Controller-only builds omit `SMARTSPRAY_BUILD_VISION` (default OFF) and need none of these vision dependencies. Debug uses the same command with its build directory and `CMAKE_BUILD_TYPE=Debug`.

Sanitizer build used `-fsanitize=address,undefined,float-cast-overflow -fno-omit-frame-pointer` in `CMAKE_CXX_FLAGS` and the sanitizer flags in `CMAKE_EXE_LINKER_FLAGS`; CTest ran with `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`, `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`, excluding the separately reported real-model test.

Local evidence is under `.local/m3`: export metadata, immutable sample manifest, per-layer comparisons and predictions, controller input/output, validation deltas, timing, dependency sources and test logs. Large tensor dumps remain under `.local/m3/parity` and are excluded from the compact review archive along with dataset images and model binaries. Source archives contain tracked source/technical docs only. No retraining, hardware integration or M4 presentation is included.
