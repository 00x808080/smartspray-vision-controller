# Saved demonstration trace

These are saved outputs from the verified native demo at source revision `c861dd25096df614311b0239bbcf24a898599e3b`, not a fresh execution on the reader's machine. The run uses PhenoBench training image `05-15_00028_P0030852`; its model/image/configuration identity is in [run.json](run.json).

- [run.json](run.json): every prediction, score, floating-point coordinate, selected target, plan, merged interval, timestamp and final channel state.
- [events.csv](events.csv): the same 16 executed ON/OFF events in a compact table.
- [Timeline](../../docs/assets/timeline.png): the unchanged visualization of those commands.

No fields were redacted: inspection found no absolute paths, host/user metadata or private references in either source file. Both copies are byte-identical to the frozen artifacts. The source hashes below allow checking that identity; original local evidence remains unchanged.

| Artifact | Source SHA256 |
|---|---|
| `run.json` | `3a944eb649c59e33a43ba864d56a88d2764aec1134d32d8717db159495586dd2` |
| `events.csv` | `1e14792e75a4ce54cf67376f01e628a14e557b090beafe6dbf849f745bbaded4` |

Predictions are not ground truth, and command geometry/timing are simulated. This training-image example is not the validation benchmark. See [output semantics](../../docs/DEMO.md#outputs-and-actual-result) and [provenance and asset status](../../docs/assets/ATTRIBUTION.md).
