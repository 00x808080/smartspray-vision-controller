# Demonstration asset provenance

## Included command visualization

`timeline.png` is the unchanged native SmartSpray rendering of the command events in the [saved example](../../examples/frozen-demo/README.md), reproduced at source revision `c861dd25096df614311b0239bbcf24a898599e3b`. It contains the project's simulated command chart and text, with no source-image pixels or annotation masks. No intervals, labels, target IDs or outcomes were redrawn or removed. Dimensions: 1640 × 880 pixels; source/copy SHA256: `7036a98546370b5d25b0c7939b0f75df1224b9c05baa2951d3488c34d4c2c2f3`. No resizing or other transformation was applied.

The JSON/CSV example is also copied unchanged. Its hashes and privacy inspection are in the example README. These are model predictions and simulated commands, not biological ground truth or field measurements. Project-authored source code is separately licensed under [AGPL-3.0-only](../../NOTICE.md).

## Dataset and paper

The demonstration input is **PhenoBench v1.1.0**, training split, image **`05-15_00028_P0030852.png`** (1024 × 1024). Original image SHA256: `43aa736cfa133817c93ea33269e7835a274577ef3a35c863c27546aa9e8bd9ba`. The original image is not included in this repository.

Dataset authors: **Jan Weyler, Federico Magistri, Elias Marks, Yue Linn Chong, Matteo Sodano, Gianmarco Roggiolani, Nived Chebrolu, Cyrill Stachniss and Jens Behley**.

Please credit their work, *PhenoBench — A Large Dataset and Benchmarks for Semantic Image Interpretation in the Agricultural Domain*: [dataset and citation](https://www.phenobench.org/dataset.html), [paper](https://arxiv.org/abs/2306.04557), [original v1.1.0 archive](https://www.phenobench.org/data/PhenoBench-v110.zip).

## Published annotated image

The frozen [annotated.png](annotated.png) is included unchanged under **CC BY-SA 4.0**. Publication relies on the BY-SA notice on the [official dataset page](https://www.phenobench.org/dataset.html), as accepted by the project owner. It combines the original training-image pixels with all predicted boxes, scores and IDs, weed-center anchors, eight illustrative zones, a status ledger and simulation labels. The dataset image itself was unchanged; none of the predictions were retouched, hidden or moved. Dimensions: 1720 × 1235 pixels; source SHA256: `eadaedb79dcb4e7cbdb5c6cbb0e749196c51db11620d69335d7b2e4445cfadf7`.

The recorded publisher notices disagree: the dataset website and [pinned official FAQ](https://github.com/PRBonn/phenobench/blob/0edc128ef7f67c8c6577554c7d1a2e382e2ea81f/README.md#frequently-asked-questions) state [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), while the downloaded archive's `PhenoBench/README.MD` states [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The archive requires attribution, noncommercial use and share-alike treatment of modifications. Both notices remain recorded; the devkit's separate MIT license does not settle image rights.

Modifications by Sergey Gonchar (2026): prediction overlays, labels, weed-center markers, illustrative zones, controller outcomes and simulation annotations. These are genuine saved predictions, not ground truth; the source is a training image. No predictions, boxes or scores were hidden, moved or retouched. No endorsement by the dataset authors or their institutions is claimed. The geometry and actuation are simulated; no field-safety, calibration or real-time claim is made. The annotated image is output, not a substitute inference input.

Citation: Weyler et al., *PhenoBench — A Large Dataset and Benchmarks for Semantic Image Interpretation in the Agricultural Domain*, IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12), 9583–9594, 2024. The website/archive discrepancy remains a factual provenance note; the code license does not replace either dataset notice. [DATA](../DATA.md#provenance-and-rights) retains the original provenance and limitations.
