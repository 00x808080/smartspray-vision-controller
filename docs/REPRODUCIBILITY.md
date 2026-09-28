# Reproducibility and verified environment

This verifies a fresh GitHub checkout, fresh native builds and new Python environments on the **existing Ubuntu/WSL host**. It is not a clean OS, second-machine test, public release or promise of bit-identical retraining. The functional scope and M3 numerical budgets are unchanged.

## Inputs and provenance

| Category | Contents |
|---|---|
| Tracked | C++/Python sources, tests, CMake, pinned requirements, configuration, technical documents, curated command chart and saved JSON/CSV example |
| Required for native demo | Owner-supplied ONNX and one image; official ORT CPU archive; declared Ubuntu libraries |
| Optional training/evaluation | Original checkpoint, pretrained weights, full PhenoBench train/val, audit/derived labels and ML stack |
| Generated/historical | Builds, environments, full evidence bundles, other JSON/CSV/PNG outputs and local session records; absent from Git |

Supply the already exported `detector.onnx` and an input image as separate files; no prior workflow notes or evidence tree are required. The exact trained model is currently an externally supplied artifact, with no project release/download link. The frozen demonstration image identity and hash were read from the original manifest. It is the first pre-existing M2.2 smoke training ID, frozen before M4 inference, not a sample selected for favorable predictions.

| Artifact | SHA256 |
|---|---|
| `detector.onnx` | `1897b7c32c91b73665f9faa179739ae85e664530380971f3699c684ef6f82c35` |
| `05-15_00028_P0030852.png` | `43aa736cfa133817c93ea33269e7835a274577ef3a35c863c27546aa9e8bd9ba` |
| Tracked `configs/demo.json` | `39256507b0e8cb7c2d8dedb3d8c40e35a623ea6a6185351c3010574f7335328c` |
| Optional original `best.pt` | `ea861ff9ba54a768dffd62814944d157e50037a67fd268ea65fe0d7773196fd6` |
| Official `onnxruntime-linux-x64-1.22.0.tgz` | `8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3` |

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

Use the owner's existing authentication for the private remote. Choose a new absolute `WORK` directory outside the primary repository and an exact reviewed `REVISION`. Run as an ordinary user. Start a shell without inherited project environment settings:

```bash
env -i HOME="$HOME" USER="$USER" PATH=/usr/bin:/bin LANG=C.UTF-8 \
  bash --noprofile --norc
```

Set `WORK` to a new absolute directory whose parent exists; `REVISION` to the exact reviewed commit; `UV` to the existing uv executable; and `OWNER_MODEL`, `OWNER_IMAGE`, `ORT_ARCHIVE` to the verified local source files. These are user-supplied paths, not repository defaults. `UV` may be any existing executable of the recorded version; no personal installation directory is required.

Authenticate Git for the private repository using your existing credential setup before starting. Authentication is external to the project; no Windows-specific helper is required by these commands. Create a fresh remote clone, not a local clone, shared object store or working-directory copy:

```bash
mkdir "$WORK"
git clone https://github.com/00x808080/smartspray-vision-controller.git "$WORK/source"
git -C "$WORK/source" checkout --detach "$REVISION"
mkdir -p "$WORK/inputs" "$WORK/deps" "$WORK/outputs" "$WORK/envs"
cp "$OWNER_MODEL" "$WORK/inputs/detector.onnx"
cp "$OWNER_IMAGE" "$WORK/inputs/05-15_00028_P0030852.png"
cp "$ORT_ARCHIVE" "$WORK/deps/onnxruntime-linux-x64-1.22.0.tgz"
sha256sum "$WORK/inputs/"* "$WORK/deps/onnxruntime-linux-x64-1.22.0.tgz"
tar -xzf "$WORK/deps/onnxruntime-linux-x64-1.22.0.tgz" -C "$WORK/deps"
cd "$WORK/source"
sha256sum configs/demo.json
export ORT_ROOT="$WORK/deps/onnxruntime-linux-x64-1.22.0"
export MODEL="$WORK/inputs/detector.onnx"
export IMAGE="$WORK/inputs/05-15_00028_P0030852.png"
export OUTPUT="$WORK/outputs/run-a"
```

Require every hash to match the table before running. Use the [README quickstart](../README.md#getting-started) for controller and vision Release builds/tests and the native demo. Repeat Debug into fresh build directories. Repeat the demo from a fresh process with `OUTPUT="$WORK/outputs/run-b"`; never reuse an output directory. Neither native invocation needs an activated Python environment. `ldd build-vision-release/smartspray_vision_demo` must resolve ORT from the declared extracted release and OpenCV from system libraries, never from an old project build. CMake supplies the ORT runtime path; no global library-path setting is required.

## Existing Python suites

These environments are for tests, not the native program. The full suite includes actual Ultralytics transforms and a synthetic CPU native-validator check, so it requires the pinned ML stack. Create only these two environments; do not copy site-packages or an installed environment.

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

SMARTSPRAY_VISION_DEMO="$PWD/build-vision-debug/smartspray_vision_demo" \
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

Data/weight/model/runtime/project-code rights remain separate. Preserve [PhenoBench attribution and license discrepancy](DATA.md), [model/upstream notices](BASELINE.md#third-party-notices) and [ORT provenance](INFERENCE.md#export-and-dependencies). The source archive includes only tracked content: no model, checkpoint, original dataset image or annotated field-image derivative. The local review package may retain the withheld annotated image for review, with [attribution and its exact distribution decision](assets/ATTRIBUTION.md). Reproduction does not validate calibration, real-time behavior, crop-damage prevention or hardware safety.
