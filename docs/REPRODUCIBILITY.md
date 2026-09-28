# Reproducibility and verified environment

The v0.1.0 quickstart uses versioned public release artifacts and ordinary explicit paths. The historical verification below used fresh GitHub checkouts, native builds and Python environments on the **existing Ubuntu/WSL host**; it is not a clean OS, second-machine test or promise of bit-identical retraining. The functional scope and numerical budgets are unchanged.

## Inputs and provenance

| Category | Contents |
|---|---|
| Tracked | C++/Python sources, tests, CMake, pinned requirements, configuration, technical documents, attributed annotated image, command chart and saved JSON/CSV example |
| Required for native demo | v0.1.0 ONNX release asset and a user-supplied image; official ORT CPU archive; declared Ubuntu libraries |
| Optional training/evaluation | Original checkpoint, pretrained weights, full PhenoBench train/val, audit/derived labels and ML stack |
| Generated/historical | Builds, environments, full evidence bundles, other JSON/CSV/PNG outputs and local session records; absent from Git |

Download `detector.onnx` and `SHA256SUMS.txt` from the versioned [v0.1.0 release](https://github.com/00x808080/smartspray-vision-controller/releases/tag/v0.1.0), then supply an input image separately. No prior workflow notes or evidence tree are required. The [model card](../MODEL_CARD.md) distinguishes original hashes from publication copies with private metadata removed. The native model graph and weights are unchanged. The frozen demonstration image identity and hash were read from the original manifest. It is the first pre-existing M2.2 smoke training ID, frozen before M4 inference, not a sample selected for favorable predictions.

| Artifact | SHA256 |
|---|---|
| Public v0.1.0 `detector.onnx` | `13c03f6f8c189f7fd3ef440ec8cc823ac0edaed7390b40ddbaf5d2f683fab348` |
| Original pre-publication `detector.onnx` (provenance only) | `1897b7c32c91b73665f9faa179739ae85e664530380971f3699c684ef6f82c35` |
| `05-15_00028_P0030852.png` | `43aa736cfa133817c93ea33269e7835a274577ef3a35c863c27546aa9e8bd9ba` |
| Tracked `configs/demo.json` | `39256507b0e8cb7c2d8dedb3d8c40e35a623ea6a6185351c3010574f7335328c` |
| Optional original `best.pt` | `ea861ff9ba54a768dffd62814944d157e50037a67fd268ea65fe0d7773196fd6` |
| Official `onnxruntime-linux-x64-1.22.0.tgz` | `8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3` |

The release-model download must match `SHA256SUMS.txt`, not the original pre-publication hash above. The original input is `PhenoBench/train/images/05-15_00028_P0030852.png` within the official [v1.1.0 archive](https://www.phenobench.org/data/PhenoBench-v110.zip); obtain it through the [dataset page](https://www.phenobench.org/dataset.html) and follow its data terms. The image is not a release asset. `docs/assets/annotated.png` is output and must not be used as inference input.

ORT comes from the [official Microsoft release asset](https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz). M5 reused that verified distribution archive and extracted it anew outside the checkout; it did not copy an installed runtime/build tree. The graph is FP32, opset 18 / IR 8, input `[1,3,1024,1024]`, output `[1,6,21504]`. The existing [training](BASELINE.md#reproduction) and [export](INFERENCE.md#verification-and-reproduction) route remains available; neither is rerun for the native demo.

## Tested host and dependencies

| Component | Actual tested version/source |
|---|---|
| OS / kernel | Ubuntu 24.04.5 LTS, WSL 2, Linux 6.18.33.2-microsoft-standard-WSL2, x86_64 |
| C++ / build tools | GCC/G++ 13.3.0; CMake/CTest 3.28.3; Make 4.3; Unix Makefiles; C++17 |
| Git / Python / uv | Git 2.43.0; system CPython 3.12.3; existing uv 0.12.17 |
| Native inference | Official ONNX Runtime 1.22.0 CPU, one intra/inter-op thread |
| Ubuntu image libraries | `libopencv-core-dev`, `libopencv-imgproc-dev`, `libopencv-imgcodecs-dev`: 4.6.0+dfsg-13.1ubuntu1 |
| Ubuntu JSON | `nlohmann-json3-dev`: 3.11.3-1 |
| Data-test environment | [requirements-data.txt](../requirements-data.txt): NumPy 2.5.3, Pillow 12.3.0 |
| Full test environment | [requirements-ml.txt](../requirements-ml.txt) plus [requirements-inference.txt](../requirements-inference.txt), all exact pins |

The declared system packages were already installed and reused; M5 did not upgrade drivers, toolchains or system packages. Python tests use OpenCV 5.0.0.93 and NumPy 2.5.2 from the ML pins; native rendering uses Ubuntu OpenCV 4.6.0. The separate data environment deliberately retains NumPy 2.5.3.

## Checkout and native build

The native dependencies above must be installed; no Python environment is needed for this section. Choose a new absolute working directory and supply your own supported image. For an exact comparison, use the original PhenoBench training image identified above. Paths with spaces are supported.

```bash
read -r -p 'New absolute working directory: ' WORK
read -r -p 'Original input image file: ' SOURCE_IMAGE
mkdir "$WORK"
git clone --branch v0.1.0 --depth 1 https://github.com/00x808080/smartspray-vision-controller.git "$WORK/source"
mkdir -p "$WORK/inputs" "$WORK/deps" "$WORK/outputs" "$WORK/envs"
RELEASE_URL=https://github.com/00x808080/smartspray-vision-controller/releases/download/v0.1.0
curl -fL "$RELEASE_URL/detector.onnx" -o "$WORK/inputs/detector.onnx"
curl -fL "$RELEASE_URL/SHA256SUMS.txt" -o "$WORK/inputs/SHA256SUMS.txt"
(cd "$WORK/inputs" && grep '  detector.onnx$' SHA256SUMS.txt | sha256sum --check --strict)
curl -fL https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz \
  -o "$WORK/deps/onnxruntime-linux-x64-1.22.0.tgz"
printf '%s  %s\n' 8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3 \
  "$WORK/deps/onnxruntime-linux-x64-1.22.0.tgz" | sha256sum --check --strict
tar -xzf "$WORK/deps/onnxruntime-linux-x64-1.22.0.tgz" -C "$WORK/deps"
cd "$WORK/source"
export ORT_ROOT="$WORK/deps/onnxruntime-linux-x64-1.22.0"
export MODEL="$WORK/inputs/detector.onnx"
export IMAGE="$SOURCE_IMAGE"
export OUTPUT="$WORK/outputs/run-a"
cmake -S . -B build-vision-release -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DSMARTSPRAY_BUILD_VISION=ON -DONNXRUNTIME_ROOT="$ORT_ROOT" \
  -DSMARTSPRAY_REAL_MODEL="$MODEL" -DSMARTSPRAY_REAL_IMAGE="$IMAGE"
cmake --build build-vision-release --parallel 2
ctest --test-dir build-vision-release --output-on-failure --no-tests=error --verbose
./build-vision-release/smartspray_vision_demo \
  --model "$MODEL" --image "$IMAGE" --config configs/demo.json --output "$OUTPUT"
```

Use a supported 8-bit, three-channel PNG/BMP. To verify the exact reference image, compare its SHA256 with the table; an arbitrary user image produces a different result. No source image is downloaded by these commands.

The [README](../README.md#getting-started) also provides the model-free controller build. Repeat the demo into a **new** output directory; the executable refuses an existing directory. `ldd build-vision-release/smartspray_vision_demo` should resolve ORT from the newly extracted distribution and OpenCV from system libraries. No old build or model path is required.

For exact comparison with the frozen JSON/CSV and two PNG outputs, the release model's metadata change necessarily changes only `run.json`'s model-file SHA256 field. Compare that field to the downloaded model, and compare every remaining deterministic field plus CSV bytes and PNG pixels to the original reference. Keep the original reference unchanged; do not describe the complete `run.json` files as byte-identical.

## Modifiable model and training/export

The same versioned release provides `detector-weights.pt`, architecture/provenance material, `MODEL_CARD.md` and notices. Download only the intended assets and verify their entries in `SHA256SUMS.txt`. The weights are a minimal tensor representation, loaded with `weights_only=True` in the pinned Python environment; see the [model card](../MODEL_CARD.md) for the tested loading command and exact packaging changes. Native inference uses only the ONNX file.

The [training configuration](../configs/yolo11n-baseline.yaml), [data preparation](DATA.md#reproduction), [training instructions](BASELINE.md#reproduction) and [export instructions](INFERENCE.md#verification-and-reproduction) retain the original experiment route. Those historical procedures may create their own artifact directories; native users need no prior `.local` contents. Retraining is optional and is not claimed to reproduce identical weights.

## Existing Python suites

Set `UV` to the absolute path of an existing uv 0.12.17 executable (for example, enter it with `read -r -p 'uv executable: ' UV`). These environments are for tests, not the native program. The full suite includes actual Ultralytics transforms and a synthetic CPU native-validator check, so it requires the pinned ML stack. Create only these two environments; do not copy site-packages or an installed environment.

```bash
"$UV" venv --python /usr/bin/python3 --no-python-downloads "$WORK/envs/data"
"$UV" pip install --python "$WORK/envs/data/bin/python" --only-binary :all: \
  --index-url https://pypi.org/simple -r requirements-data.txt

"$UV" venv --python /usr/bin/python3 --no-python-downloads "$WORK/envs/tests"
"$UV" pip install --python "$WORK/envs/tests/bin/python" --no-deps --only-binary :all: \
  torch==2.13.0+cu126 torchvision==0.28.0+cu126 --index-url https://download.pytorch.org/whl/cu126
"$UV" pip install --python "$WORK/envs/tests/bin/python" --no-deps --only-binary :all: \
  --index-url https://pypi.org/simple -r requirements-ml.txt
"$UV" pip install --python "$WORK/envs/tests/bin/python" --no-deps --only-binary :all: \
  --index-url https://pypi.org/simple -r requirements-inference.txt
"$UV" pip check --python "$WORK/envs/tests/bin/python"

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" \
SMARTSPRAY_INFER="$PWD/build-vision-release/smartspray_infer" \
SMARTSPRAY_VISION_DEMO="$PWD/build-vision-release/smartspray_vision_demo" \
"$WORK/envs/tests/bin/python" -m unittest discover -s tests -p 'test_*.py' -v
"$WORK/envs/data/bin/python" -m unittest discover -s tests -p test_phenobench_data.py -v

SMARTSPRAY_VISION_DEMO="$PWD/build-vision-release/smartspray_vision_demo" \
"$WORK/envs/tests/bin/python" -m unittest discover -s tests -p test_demo_cli.py -v
```

Explicit source ordering fixes the old combined-index installation: uv's first-index policy found `certifi` on the PyTorch index without the required version. Installing the two exact PyTorch wheels first, then the complete remaining pins from PyPI, avoids that conflict. `--no-deps` preserves the committed complete pin set; `pip check` verifies its dependency consistency. No version was substituted. Cached distribution files and downloads are allowed. The legacy ML hook generates fresh ignored settings/font links under the new checkout's `.local`; it requires no copied prior settings, artifacts or predictions.

Missing compiled-executable environment variables explicitly skip CLI checks. Treat skips as unavailable checks. The original 36 data tests are a subset of the full 103, and the 15 demo CLI tests also appear in the full suite; do not sum these as unique tests.

## Host sanitizer check

Use a fresh build, with the same declared ORT and system dependencies:

```bash
cmake -S . -B build-vision-sanitize -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug \
  -DSMARTSPRAY_BUILD_VISION=ON -DONNXRUNTIME_ROOT="$ORT_ROOT" \
  '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined,float-cast-overflow -fno-omit-frame-pointer' \
  '-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined,float-cast-overflow'
cmake --build build-vision-sanitize --parallel 2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build-vision-sanitize --output-on-failure --no-tests=error --verbose -LE real_model
```

The four host-side CTest registrations run; the real-model smoke is deliberately excluded here and run in normal Debug/Release. Prebuilt third-party libraries are not instrumented.

## Verification result and limits

The fresh clone started at reviewed M4 revision `bbc1703ba01b0a1e423e2b16c61fa5e88e251022`. All application sources, test sources, requirement pins and demo configuration remain unchanged. The M5 changes correct reproduction instructions and consolidate the handoff.

| Fresh-checkout check | Actual result |
|---|---|
| Controller Debug / Release | 1 registered CTest entry with 40 internal scenarios in each |
| Vision Debug / Release | 5/5 registered CTest entries each, including actual real-model smoke; other entries contain 40 controller, 15 vision, 6 parity-boundary and 15 demo scenarios |
| Full Python suite, new pinned environment | 103 passed, no skips |
| Original data suite, separate new environment | 36 passed, no skips; overlaps the full suite |
| Demo CLI regressions | 15 passed in Debug; Release included in the full 103 |
| ASan / UBSan / float-cast-overflow | 4/4 offline CTest entries; no diagnostics; real-model test excluded here |
| Two fresh native processes | Complete JSON/CSV and PNG files byte-identical to M4; decoded pixels identical |
| Recorded result | 17 predictions, 8 crops not selected, 9 weeds/accepted targets, 0 rejections, 8 intervals, 16 events, eight OFF at 1,974,044 us |

Malformed JSON with trailing content, missing inputs and existing-output refusal pass through the existing native and CLI regression registrations. Both native runs used an explicit clean environment without `VIRTUAL_ENV`, `PYTHONPATH`, `CMAKE_PREFIX_PATH` or library-path overrides. Loader diagnostics resolve ORT from the newly extracted archive and OpenCV from system libraries; no library resolves through the primary workspace. The native entry point loads the model and image and calls inference directly; it has no Python or prediction-replay invocation.

The accompanying local review package records commands, versions, hashes and test/comparison summaries, plus the final-revision Release rebuild/tests/demo after publication. It identifies the tested final commit without embedding this document's own SHA in another commit.

The complete `run.json` (including scores, floating coordinates, IDs, classes, selection, accepted/rejected plans, all integer timestamps, merged source IDs, events and final states), `events.csv` and decoded PNG pixels are compared against unchanged M4 outputs. No deterministic field is discarded. Only console `output_directory` and `wall_clock_ms` describe location or nondeterministic timing; both original console logs are retained.

Historical M3 numerical budgets, 32-image parity and 772-image validation remain prior evidence, not M5 reruns. No inference/planning implementation changes, retraining, re-export, full dataset audit or full validation run are required for this documentation-only reproduction fix.

Data/weight/model/runtime/project-code rights remain separate. Preserve [PhenoBench attribution and license discrepancy](DATA.md), [model/upstream notices](BASELINE.md#third-party-notices) and [ORT provenance](INFERENCE.md#export-and-dependencies). The source archive includes only tracked content, including the approved annotated illustration with [attribution and publication basis](assets/ATTRIBUTION.md). Models are separate release assets; the full dataset, original image collection, optimizer dumps and private evidence remain excluded. Reproduction does not validate calibration, real-time behavior, crop-damage prevention or hardware safety.
