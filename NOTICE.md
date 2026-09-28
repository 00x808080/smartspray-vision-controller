# License scope and notices

Copyright (C) 2026 Sergey Gonchar.

Project-authored source code is licensed under **AGPL-3.0-only** (SPDX identifier), the GNU Affero General Public License version 3 only. The complete, unmodified official text is in [LICENSE](LICENSE), obtained from [GNU](https://www.gnu.org/licenses/agpl-3.0.txt). This notice selects version 3 only; the explanatory template in the official license text does not change that choice. The code license adds no commercial-use prohibition.

- Third-party components retain their respective licenses and notices; see [upstream notices](docs/BASELINE.md#third-party-notices) and [runtime/export dependencies](docs/INFERENCE.md#export-and-dependencies). Dependency distributions are obtained separately, with their bundled notices.
- The PhenoBench-derived `docs/assets/annotated.png` is distributed under **CC BY-SA 4.0**, relying on the official dataset page's BY-SA notice. Attribution, exact image identity, modifications and the conflicting archive notice remain in [asset attribution](docs/assets/ATTRIBUTION.md). This media is separate from the source-code license.
- Release model artifacts derive from Ultralytics YOLO11n and its official COCO-pretrained weights, fine-tuned on PhenoBench. Preserve the upstream AGPL terms and data provenance; see [MODEL_CARD](MODEL_CARD.md), [Ultralytics licensing](https://www.ultralytics.com/license), and the release notices. The project's own-code license does not replace upstream model, data or dependency terms.

The project does not claim endorsement by Ultralytics, the PhenoBench authors or their institutions. Simulation results do not establish calibration, field safety or real-time performance.
