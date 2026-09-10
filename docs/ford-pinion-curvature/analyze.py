from pathlib import Path
import argparse
import json
import pickle
import hashlib
import subprocess
import types
import collections
import capnp
import numpy as np

parser = argparse.ArgumentParser(description="Compare raw Expedition CAN-FD pinion/yaw against calibrated deviceMotion.")
parser.add_argument("--baseline-repo", type=Path, required=True)
parser.add_argument("--postfix-root", type=Path, required=True)
parser.add_argument("--prior-root", type=Path, required=True)
parser.add_argument("--postfix-extracted", type=Path)
parser.add_argument("--prior-extracted", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--dongle", default="0e09a1daaf2c4fd2")
args = parser.parse_args()
O = args.output
O.mkdir(parents=True, exist_ok=True)
BASE = args.baseline_repo
route_inputs = [("postfix", args.postfix_root, args.postfix_extracted), ("prior", args.prior_root, args.prior_extracted)]
vmfile = BASE / "opendbc/car/vehicle_model.py"
src = vmfile.read_text()
ns = {}
exec(
  compile(
    src.replace("from opendbc.car.structs import CarParams", "CarParams = object").replace(
      "from opendbc.car import ACCELERATION_DUE_TO_GRAVITY", "ACCELERATION_DUE_TO_GRAVITY = 9.81"
    ),
    str(vmfile),
    "exec",
  ),
  ns,
)
# Both recordings use the same schema IDs. Verify complete schema equality before sharing
# one schema loader, avoiding duplicate Cap'n Proto ID registration in this process.
schema_dir = args.postfix_root / "schema"


def schema_hashes(d):
  return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in d.rglob("*.capnp")}


assert schema_hashes(schema_dir) == schema_hashes(args.prior_root / "schema"), "Route schemas differ; analyze separately with matching schemas."
capnp.remove_import_hook()
schema = capnp.load(str(schema_dir / "log.capnp"), imports=[str(schema_dir)])


def stat(a):
  a = np.asarray(a)
  return (
    dict(n=len(a), median=float(np.median(a)), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)), rmse=float(np.sqrt(np.sum(a * a) / a.size)))
    if len(a)
    else None
  )


def compare(a, b, m):
  return dict(correlation=float(np.corrcoef(a[m], b[m])[0, 1]) if np.sum(m) > 2 else None, residual=stat((a - b)[m]))


results = {}
for label, path, extracted in route_inputs:
  extracted = extracted or path / "extracted.pkl"
  D = pickle.load(open(extracted, "rb"))
  raw = []
  segments = []
  CP = None
  bus = collections.Counter()
  for f in sorted((path / "logs").glob("*/rlog.zst"), key=lambda p: int(p.parent.name.rsplit("--", 1)[1])):
    blob = f.read_bytes()
    tmin = None
    tmax = None
    for e in schema.Event.read_multiple_bytes(subprocess.check_output(["zstd", "-dc", str(f)])):
      t = e.logMonoTime / 1e9
      if e.which() == "can":
        tmin = t if tmin is None else min(tmin, t)
        tmax = t if tmax is None else max(tmax, t)
      if e.which() == "carParams" and CP is None:
        keys = ["mass", "rotationalInertia", "wheelbase", "centerToFront", "steerRatioRear", "tireStiffnessFront", "tireStiffnessRear", "steerRatio"]
        CP = {k: getattr(e.carParams, k) for k in keys}
      if e.which() != "can":
        continue
      for c in e.can:
        if c.address not in (0x7E, 0x91):
          continue
        bus[(c.address, c.src)] += 1
        if c.src != 0 or len(c.dat) != 8:
          continue
        d = bytes(c.dat)
        if c.address == 0x7E:
          raw.append((t, c.address, (((d[2] & 127) << 8) | d[3]) * 0.1 - 1600, (d[5] >> 2) & 3))
        else:
          raw.append((t, c.address, int.from_bytes(d[2:4], "big") * 0.0002 - 6.5, (d[6] >> 4) & 3))
    segments.append(
      dict(
        id=f.parent.name,
        segment=int(f.parent.name.rsplit("--", 1)[1]),
        path=str(Path(label) / "logs" / f.parent.name / f.name),
        sha256=hashlib.sha256(blob).hexdigest(),
        bytes=len(blob),
        mono_start=tmin,
        mono_end=tmax,
      )
    )
    print(label, f.parent.name, flush=True)
  origin = min(q["t"] for q in D["initData"])
  for s in segments:
    s.update(route_start=s["mono_start"] - origin, route_end=s["mono_end"] - origin)

  def samp(topic, key, t, D=D):
    rows = sorted(D[topic], key=lambda q: q["t"])
    ts = np.array([q["t"] for q in rows])
    vs = np.array([q[key] for q in rows])
    idx = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)
    return vs[idx]

  raw = np.array(sorted(raw))
  pin = raw[raw[:, 1] == 0x7E]
  yaw = raw[raw[:, 1] == 0x91]
  rows = sorted(D["deviceMotion"], key=lambda q: q["t"])
  pub = np.array([q["t"] for q in rows])
  mt = np.array([q["motionTime"] for q in rows])
  w = np.array([q["omega"] for q in rows])
  r, p, y = samp("extrinsicsCalibration", "rpy", pub).T
  axis = np.stack(
    [np.cos(y) * np.sin(p) * np.cos(r) + np.sin(y) * np.sin(r), np.sin(y) * np.sin(p) * np.cos(r) - np.cos(y) * np.sin(r), np.cos(p) * np.cos(r)], axis=1
  )
  comma = -(axis * w).sum(axis=1)
  pi = np.clip(np.searchsorted(pin[:, 0], mt, side="right") - 1, 0, len(pin) - 1)
  yi = np.clip(np.searchsorted(yaw[:, 0], mt, side="right") - 1, 0, len(yaw) - 1)
  angle = pin[pi, 2]
  ford = yaw[yi, 2]
  v = samp("carState", "vEgoRaw", mt)
  vm = ns["VehicleModel"](types.SimpleNamespace(**CP))
  pin_yaw = vm.yaw_rate(np.deg2rad(angle), v, 0)
  ok = (
    np.array([q["valid"] and q["omegaValid"] and q["sensorsOK"] and q["inputsOK"] for q in rows])
    & (samp("extrinsicsCalibration", "status", pub) == "calibrated")
    & (mt >= pin[0, 0])
    & (mt >= yaw[0, 0])
    & (mt - pin[pi, 0] < 0.05)
    & (mt - yaw[yi, 0] < 0.05)
    & (pin[pi, 3] == 3)
    & (yaw[yi, 3] == 3)
  )
  moving = ok & (v > 5)
  stopped = ok & (v < 0.1)
  active = samp("carControl", "latActive", mt)
  brake = samp("carState", "brakePressed", mt)
  driver = samp("carState", "steeringPressed", mt)
  masks = {
    "moving_gt5mps": moving,
    "active_moving": moving & active,
    "straight_moving": moving & (np.abs(comma) < 0.01),
    "turning_moving": moving & (np.abs(comma) > 0.03),
    "stopped": stopped,
  }
  metrics = {
    k: dict(pinion_vs_comma=compare(pin_yaw, comma, m), raw_ford_vs_comma=compare(ford, comma, m), pinion_vs_raw_ford=compare(pin_yaw, ford, m))
    for k, m in masks.items()
  }

  def window(t, segments=segments, moving=moving, mt=mt, origin=origin, pin_yaw=pin_yaw, comma=comma, ford=ford, v=v):
    ss = [s for s in segments if s["mono_end"] >= t - 10 and s["mono_start"] <= t + 10]
    target = [s for s in segments if s["mono_start"] <= t <= s["mono_end"]]
    m = moving & (mt >= t - 10) & (mt <= t + 10)
    return dict(
      mono_time=t,
      route_seconds=t - origin,
      target_segments=[dict(segment=s["segment"], segment_local_seconds=t - s["mono_start"]) for s in target],
      window_route=[t - 10 - origin, t + 10 - origin],
      window_segments=[
        dict(segment=s["segment"], local_start=max(0, t - 10 - s["mono_start"]), local_end=min(s["mono_end"], t + 10) - s["mono_start"]) for s in ss
      ],
      pinion_vs_comma=compare(pin_yaw, comma, m),
      raw_ford_vs_comma=compare(ford, comma, m),
      speed_mps=stat(v[m]),
      comma_yaw_rad_s=stat(comma[m]),
    )

  rejects = [q for q in D["can"] if q["addr"] == 0x3D6 and q["bus"] == 192 and q["mode"] != 0]
  targets = []
  if label == "postfix":
    for i, q in enumerate(sorted(rejects, key=lambda q: q["t"])):
      targets.append(
        dict(
          id="reject-" + str(i + 1),
          brake=bool(samp("carState", "brakePressed", q["t"])),
          driver=bool(samp("carState", "steeringPressed", q["t"])),
          **window(q["t"]),
        )
      )
  # Comparable turn centers: active, no brake; driver steering is allowed; each full window contains no recorded active blocked CAN, all-sample active
  # and no brake, v>5; rank similarity to non-brake rejection-center speed and absolute gyro yaw.
  candidates = []
  for t in np.arange(mt[0] + 10, mt[-1] - 10, 1):
    m = (mt >= t - 10) & (mt <= t + 10)
    if np.sum(m) < 100 or not np.all(ok[m] & active[m] & ~brake[m] & (v[m] > 5)):
      continue
    if any(abs(q["t"] - t) <= 10 for q in rejects):
      continue
    j = np.argmin(abs(mt - t))
    if abs(comma[j]) < 0.03:
      continue
    candidates.append((t, j))
  selected = []
  references = targets if targets else [{"mono_time": t} for t in np.arange(mt[0] + 100, mt[-1], 150)]
  for ref in references:
    if ref.get("brake"):
      continue
    j = np.argmin(abs(mt - ref["mono_time"]))
    choices = [
      (abs(v[k] - v[j]) / 5 + abs(abs(comma[k]) - abs(comma[j])) / 0.1, t, k) for t, k in candidates if all(abs(t - q["mono_time"]) > 25 for q in selected)
    ]
    if choices:
      score, t, k = min(choices)
      selected.append(dict(id="clean-turn-" + str(len(selected) + 1), matched_to=ref.get("id"), match_score=score, **window(float(t))))
    if len(selected) == 4:
      break
  out = dict(
    route=path.joinpath("logs").iterdir().__next__().name.rsplit("--", 1)[0],
    dongle=args.dongle,
    origin_logMonoTime=origin,
    origin_definition="initial initData.logMonoTime; segment local zero is first raw CAN Event.logMonoTime, excludes repeated metadata",
    segments=segments,
    carParams=CP,
    bus_counts={str(k): v for k, v in bus.items()},
    pinion_quality=dict(collections.Counter(pin[:, 3].astype(int).tolist())),
    yaw_quality=dict(collections.Counter(yaw[:, 3].astype(int).tolist())),
    timing=dict(device_publish_minus_motion=stat(pub - mt), pinion_age_at_motion=stat((mt - pin[pi, 0])[ok]), yaw_age_at_motion=stat((mt - yaw[yi, 0])[ok])),
    metrics=metrics,
    reject_windows=targets,
    comparable_clean_turns=selected,
  )
  results[label] = out
  np.savez_compressed(O / (label + "-aligned.npz"), t=mt, v=v, comma=comma, pinion_yaw=pin_yaw, raw_yaw=ford, ok=ok, active=active, angle=angle)
(O / "evidence.json").write_text(
  json.dumps(
    dict(
      schema_sha256=schema_hashes(schema_dir),
      vehicle_model_source="opendbc/car/vehicle_model.py",
      vehicle_model_sha256=hashlib.sha256(src.encode()).hexdigest(),
      method=(
        "Raw bus 0 0x7e pinion, 0x91 yaw; causal last-value alignment to deviceMotion.timestamp; extrinsics a"
        + "t deviceMotion publication; fixed recorded CP VehicleModel, stiffness 1, roll 0, offset 0; no fittin"
        + "g or lag correction; Comma projected vertical rate negated to Ford steering/yaw convention"
      ),
      routes=results,
    ),
    indent=2,
  )
)
