from pathlib import Path
import argparse
import json
import hashlib
import numpy as np

parser = argparse.ArgumentParser(description="Render evidence.json and aligned NPZ files to a reviewable report.")
parser.add_argument("--output", type=Path, required=True, help="Directory containing analysis outputs; REPORT.md is written here.")
args = parser.parse_args()
O = args.output
x = json.loads((O / "evidence.json").read_text())


def fmt(s):
  return f"{s['median']:.6f} / {s['rmse']:.6f}"


def stats(a):
  return dict(
    n=len(a),
    median=float(np.median(a)),
    rmse=float(np.sqrt(np.sum(a * a) / a.size)),
    p90_abs=float(np.percentile(abs(a), 90)),
    abs_exceed_0_002_count=int(np.sum(abs(a) > 0.002)),
  )


lines = [
  "# Expedition CAN-FD pinion measurement evidence",
  "",
  (
    "Observed signals from two complete recorded routes support evaluating pinion-derived curvature as a "
    + "narrowly scoped measurement source. This evidence does not diagnose a failed yaw sensor and does not"
    + " establish candidate driving performance."
  ),
  "",
  "## Reproduction and conventions",
  "",
  (
    "Run the configurable extraction, analysis and report commands in `README.md`. Requires private raw l"
    + "ogs and matching local schemas; SHA-256 hashes, exact paths, raw CAN segment bounds, CP geometry and"
    + " model source hash are in `evidence.json`. Parsing decompresses one segment at a time."
  ),
  "",
  (
    "Pinion: bus 0, 0x7e, StePinComp_An_Est = (((byte2 & 127) << 8) | byte3) × 0.1 − 1600 degrees; qualit"
    + "y bits byte5[3:2] = 3. Ford yaw: bus 0, 0x91, bytes2:3 big endian × 0.0002 − 6.5 rad/s; quality byte"
    + "6[5:4] = 3. Pinion yaw = VehicleModel(CP).yaw_rate(radians(pinion), vEgoRaw, 0). Recorded fixed CP: "
    + "19.6000003815 steering ratio, 3.1115000248 m wheelbase; stiffness factor 1, no learned offset, ratio"
    + ", stiffness or roll. The copied model source is executed unchanged mathematically; only dependency/t"
    + "ype imports are replaced to avoid incompatible Cap’n Proto schemas."
  ),
  "",
  (
    "Comma: project angularVelocityDevice onto calibrated vertical using extrinsics rpyCalib at publicati"
    + "on time, then negate to Ford steering/yaw sign convention. Compare at deviceMotion.timestamp, using "
    + "last CAN sample ≤ measurement time, both CAN ages <50 ms, quality=3, and valid/calibrated deviceMoti"
    + "on. No fitted slope, bias, sign optimization, lag shift or smoothing. Coordinate sign is fixed; raw "
    + "unnegated Comma is opposite Ford."
  ),
  "",
  "## Whole-route numerical comparison",
  "",
  "Yaw residuals are source − sign-aligned Comma, rad/s. Moving means vEgoRaw >5 m/s.",
  "",
  "| Route | samples | pinion correlation | pinion residual median / RMSE | raw yaw correlation | raw yaw residual median / RMSE |",
  "|---|---:|---:|---:|---:|---:|",
]
for label, r in x["routes"].items():
  m = r["metrics"]["moving_gt5mps"]
  a = m["pinion_vs_comma"]
  b = m["raw_ford_vs_comma"]
  lines.append(f"| {r['route']} | {a['residual']['n']} | {a['correlation']:.6f} | {fmt(a['residual'])} | {b['correlation']:.6f} | {fmt(b['residual'])} |")
  z = np.load(O / (label + "-aligned.npz"))
  mask = z["ok"] & (z["v"] > 5)
  r["curvature_metrics_moving"] = {
    k: stats(((z[key] - z["comma"]) / np.maximum(z["v"], 0.01))[mask]) for k, key in [("pinion", "pinion_yaw"), ("raw_yaw", "raw_yaw")]
  }
  for t in r["reject_windows"] + r["comparable_clean_turns"]:
    i = np.argmin(abs(z["t"] - t["mono_time"]))
    t["nearest_motion_sample"] = dict(
      mono_time=float(z["t"][i]),
      time_delta_to_event=float(z["t"][i] - t["mono_time"]),
      valid=bool(z["ok"][i]),
      v_mps=float(z["v"][i]),
      pinion_yaw=float(z["pinion_yaw"][i]),
      raw_yaw=float(z["raw_yaw"][i]),
      comma_yaw=float(z["comma"][i]),
    )
lines += [
  "",
  (
    "The raw yaw tracks turn dynamics more tightly but has a persistent negative level difference relativ"
    + "e to Comma; pinion has lower bias and larger dynamic/model error. Neither correlation alone establis"
    + "hes which source is safe to substitute. Fixed geometry, steering compliance, tire slip, road bank, s"
    + "ensor calibration and timestamp age can affect residuals."
  ),
  "",
  "| Route | source | curvature median / RMSE (1/m) | p90 absolute | samples >0.002 |",
  "|---|---|---:|---:|---:|",
]
for r in x["routes"].values():
  for k, m in r["curvature_metrics_moving"].items():
    lines.append(f"| {r['route']} | {k} | {m['median']:.7f} / {m['rmse']:.7f} | {m['p90_abs']:.7f} | {m['abs_exceed_0_002_count']} / {m['n']} |")
lines += [
  "",
  (
    "These curvature residuals divide yaw-rate difference by measured speed. The 0.002 comparison is desc"
    + "riptive, not the Panda command-deviation check; Panda uses its own speed, quantization, sample range"
    + " and timing. No limit was relaxed."
  ),
  "",
  "## Exact rejected-command windows",
  "",
  (
    "Dongle `0e09a1daaf2c4fd2`; route `00000007--ebffff8226`. Route seconds use initial initData.logMonoT"
    + "ime (38.510349203 s). Segment-local zero is the first raw CAN logMonoTime in that file. Repeated old"
    + " initData in later segments is excluded from segment timing. Each window is exactly 10 s before/afte"
    + "r the logged rejected frame. Cross-boundary coverage is calculated from raw CAN ranges, never segmen"
    + "t number ×60. Viewer access was not checked, so these are explicit identifiers, not guessed viewer U"
    + "RLs."
  ),
  "",
  "| Event | brake / driver | logMonoTime | route s | segment : local s | ±10 s window coverage (segment : local start–end) |",
  "|---|---|---:|---:|---|---|",
]
for t in x["routes"]["postfix"]["reject_windows"]:
  loc = ", ".join(f"{s['segment']} : {s['segment_local_seconds']:.9f}" for s in t["target_segments"])
  win = "; ".join(f"{s['segment']} : {s['local_start']:.9f}–{s['local_end']:.9f}" for s in t["window_segments"])
  lines.append(f"| {t['id']} | {t['brake']} / {t['driver']} | {t['mono_time']:.9f} | {t['route_seconds']:.9f} | {loc} | {win} |")
lines += ["", "| Event ±10 s | pinion residual median / RMSE (rad/s) | raw yaw residual median / RMSE (rad/s) |", "|---|---:|---:|"]
for t in x["routes"]["postfix"]["reject_windows"]:
  lines.append(f"| {t['id']} | {fmt(t['pinion_vs_comma']['residual'])} | {fmt(t['raw_ford_vs_comma']['residual'])} |")
lines += [
  "",
  "## Comparable turns without recorded rejection",
  "",
  (
    "Deterministic selection scans one-second centers with full ±10 s windows: calibrated valid measureme"
    + "nts, speed >5 m/s, latActive throughout, no brake, center |Comma yaw| >0.03 rad/s, and no recorded a"
    + "ctive 0x3d6 bus192 frame within the window. Driver steering is allowed and not treated as a sensor f"
    + "ault. Centers are separated by >25 s. Postfix choices rank closeness of speed and absolute Comma yaw"
    + " to non-brake rejected events; prior-route choices use evenly spaced reference times. This is observ"
    + "ational comparison, not a randomized matched study. Full selection scores and per-window distributio"
    + "ns are in JSON."
  ),
  "",
  (
    "| Route | turn | center route s | segment : local s | pinion residual median / RMSE | raw yaw residu"
    + "al median / RMSE | ±10 s coverage (segment : local start–end) |"
  ),
  "|---|---|---:|---|---:|---:|---|",
]
for r in x["routes"].values():
  for t in r["comparable_clean_turns"]:
    loc = ", ".join(f"{s['segment']} : {s['segment_local_seconds']:.6f}" for s in t["target_segments"])
    win = "; ".join(f"{s['segment']} : {s['local_start']:.9f}–{s['local_end']:.9f}" for s in t["window_segments"])
    lines.append(
      "| "
      + f"{r['route']}"
      + " | "
      + f"{t['id']}"
      + " | "
      + f"{t['route_seconds']:.6f}"
      + " | "
      + f"{loc}"
      + " | "
      + f"{fmt(t['pinion_vs_comma']['residual'])}"
      + " | "
      + f"{fmt(t['raw_ford_vs_comma']['residual'])}"
      + " | "
      + f"{win}"
      + " |"
    )
lines += ["", "## Timing and limits", ""]
for r in x["routes"].values():
  t = r["timing"]
  lines.append(
    "- "
    + f"{r['route']}"
    + ": device publication minus motion timestamp median "
    + f"{t['device_publish_minus_motion']['median'] * 1000:.3f}"
    + " ms, p10/p90 "
    + f"{t['device_publish_minus_motion']['p10'] * 1000:.3f}"
    + "/"
    + f"{t['device_publish_minus_motion']['p90'] * 1000:.3f}"
    + " ms; accepted pinion/yaw sample-age medians "
    + f"{t['pinion_age_at_motion']['median'] * 1000:.3f}"
    + "/"
    + f"{t['yaw_age_at_motion']['median'] * 1000:.3f}"
    + " ms."
  )
lines += [
  "",
  (
    "CAN log timestamps are receive/publication times, not ECU sensor sample times. Same-frame state and "
    + "exact firmware RX-before-TX order remain uncertain. The nearest motion sample at each rejection is r"
    + "etained with its signed time delta, not asserted simultaneous. The six non-brake and two brake-assoc"
    + "iated frames remain separately identifiable; brake-associated rejection can reflect controls permiss"
    + "ion timing."
  ),
  "",
  (
    "Changing measurement also changes future controller commands. This report compares recorded measurem"
    + "ents only and does not replace CAN feedback while preserving old commands to claim candidate success"
    + ". A separately identified paired replay must regenerate commands with the actual controller and comp"
    + "are identical inputs/state. Even that remains an open-loop replay, without counterfactual vehicle mo"
    + "tion or road validation. The production scope under evaluation is Expedition CAN-FD only; this datas"
    + "et does not justify other Ford platforms."
  ),
]
x["analysis_script_sha256"] = {
  name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ["analyze.py", "report.py", "extract.py"]
}
(O / "evidence.json").write_text(json.dumps(x, indent=2))
(O / "REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
