# Reproduce the paired replay

Prerequisites: Python 3.11/3.12 with the selected opendbc dependencies, `cffi`, `pycapnp`, NumPy, `zstd`, and a C compiler. Supply the original route `00000007--ebffff8226` under `DATA/logs/<segment>/rlog.zst`, and the original schema files under `DATA/schema`. Raw logs are not bundled.

Use a baseline checkout at `eb70bef4f56d2e9e91cb5a0a8374890a6acfaf51` and candidate checkout at `131f73e98bd20dede2744906f1be97421ebdeb16`. Run each command as a separate process: loading both log and opendbc schemas in one process can abort on duplicate schema IDs. Use a fresh output directory to avoid stale caches.

```sh
python extract_controls.py --data "$DATA" --output "$OUT"
python extract_raw_windows.py --data "$DATA" --targets targets.json --output "$OUT/raw_windows.pkl"
python controller_replay.py --repo "$BASELINE" --data "$OUT" --variant baseline --output "$OUT/baseline.pkl"
python controller_replay.py --repo "$CANDIDATE" --data "$OUT" --variant candidate --output "$OUT/candidate.pkl"
python paired_native.py --baseline "$BASELINE" --candidate "$CANDIDATE" --raw-windows "$OUT/raw_windows.pkl" --output "$OUT"
python summarize.py --output "$OUT" --raw-windows "$OUT/raw_windows.pkl" --targets targets.json
```

Check `summary.json` and read `REPORT.md` before interpreting counts. Baseline LMC2 and LKA matches must equal their full batch counts, with zero schedule mismatches. The paired harness asserts every baseline replacement equals the original bytes and retains original non-lateral TX and physical RX ordering.

The checked-in compact summary describes this specific dataset and source pair. Different input hashes or source revisions are different experiments. Full per-record decisions are generated in `paired-native.json`; generated `.so` files and cached logs should remain outside the source repository.
