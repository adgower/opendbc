# Expedition CAN-FD recorded measurement evidence

Start with [implementation and validation](IMPLEMENTATION.md), then the [recorded-signal report](REPORT.md) and [paired controller/safety replay](replay/REPORT.md). The replay includes timing regressions; this remains a draft candidate. [Replay reproduction commands](replay/REPRODUCE.md) are separate from the signal-analysis commands below.

`REPORT.md` contains findings, limitations, exact route/segment timestamps and 10-second-before/after windows. `evidence.json` contains raw compressed-log SHA-256 hashes, recorded fixed model parameters, schema/model hashes, full numerical results and window maps. Raw logs, schemas, extracted pickles and aligned arrays are private inputs/intermediates and are intentionally excluded from this directory.

This measures recorded pinion/yaw/Comma agreement. It is separate from actual-controller paired replay; changing the measurement changes commands, so fixed historical commands cannot demonstrate candidate performance.

## Dependencies and inputs

Verified with Python 3.12.10, NumPy 2.5.3, pycapnp 2.2.4, and the `zstd` 1.5.7 CLI. Install the two Python packages in a local virtual environment and provide `zstd` on PATH. No running device or network service is required.

Provide these private roots (their directory names are arbitrary):

```text
POSTFIX_ROOT/
  logs/00000007--ebffff8226--0/rlog.zst
  ... through --19/rlog.zst
  schema/log.capnp
  schema/car.capnp, custom.capnp, deprecated.capnp, include/...
PRIOR_ROOT/
  logs/00000004--e3907c22d6--0/rlog.zst
  ... through --14/rlog.zst
  schema/log.capnp and its complete matching imports
BASELINE_REPO/
  opendbc/car/vehicle_model.py
```

Use the model source matching `vehicle_model_sha256` in `evidence.json`; this module's mathematics were not changed by the pinion patch. Both recorded route schemas have identical content; analysis checks all `.capnp` hashes before loading one shared schema. A mismatch fails explicitly. Extraction runs in separate processes with each route's own schema.

Only load pickles produced locally by `extract.py` from trusted recordings. The extractor contains the subset of the original route extractor needed here, with the same field decoding and timestamps.

## Reproduce

Run from this evidence directory, substituting real private paths. All generated files go into the specified output directories, preserving the raw data roots.

```bash
python extract.py --data-root "$POSTFIX_ROOT" --output "$WORK/postfix"
python extract.py --data-root "$PRIOR_ROOT" --output "$WORK/prior"
python analyze.py --baseline-repo "$BASELINE_REPO" \
  --postfix-root "$POSTFIX_ROOT" --prior-root "$PRIOR_ROOT" \
  --postfix-extracted "$WORK/postfix/extracted.pkl" \
  --prior-extracted "$WORK/prior/extracted.pkl" \
  --output "$WORK/results"
python report.py --output "$WORK/results"
```

If each data root already contains its original `extracted.pkl`, the two `--*-extracted` arguments may be omitted. No `comparison.json` is required: active rejected targets come directly from recorded CAN address `0x3d6`, bus `192`, nonzero mode; brake and driver flags use the last carState timestamp at or before each rejected frame.

The default dongle identifier is `0e09a1daaf2c4fd2`; `--dongle` can override the label. Viewer permissions are unknown; no viewer URLs are guessed. Stored log paths are portable logical paths (`postfix/logs/...`, `prior/logs/...`).

Extraction and analysis each decompress only one raw segment at a time. Analysis retains selected scalar signals from both full routes, emits private aligned NPZ files, and report generation adds curvature summaries and nearest-motion samples to the JSON. Do not commit the generated NPZ/pickle files or raw recordings.

## Verification of this publication copy

The configurable scripts were rerun from all 35 raw segments using newly generated minimal extractions. All route statistics, raw hashes/timing bounds, eight rejected-event windows, eight selected comparison windows and nearest-motion values exactly matched the earlier report. Six rejected events are non-brake and two are brake-associated. Each target maps to one segment and each window spans exactly 20 seconds. Differences are metadata only: portable logical paths, script hashes, and schema hashes replace machine-specific source/input paths. See `reproducibility.json`.

Segment-local zero is the first raw CAN event in each segment, not repeated initialization metadata or an assumed 60-second boundary. Route zero is initial initData.logMonoTime. These explicit conventions preserve cross-boundary windows and avoid implying sample-time precision from CAN receive times.

The repository-formatted scripts were rerun after replacing NumPy mean with sum/size; every route result and window remained exactly equal. The clean-turn report table also renders full cross-segment window coverage. Repository Ruff check and format pass.
