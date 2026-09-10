from pathlib import Path
import argparse
import sys
import json
import pickle
from types import SimpleNamespace
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--data", required=True)
p.add_argument("--variant", choices=["baseline", "candidate"], required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
sys.path.insert(0, a.repo)
from opendbc.car import structs
from opendbc.car.ford.carcontroller import CarController
from opendbc.car.ford.values import DBC, CAR

D = Path(a.data)
controller = None
cs = None
cc = None
started = False
rows = []
batches = []
counts = defaultdict(int)
hashes = []
for f in sorted(D.glob("00000007*.pkl"), key=lambda p: int(p.stem.rsplit("--", 1)[1])):
  content = pickle.loads(f.read_bytes())
  hashes.append(content["hash"])
  events = content["events"]
  for t, k, v in sorted(events, key=lambda x: x[0]):
    if k == "carParams" and controller is None:
      cp = structs.CarParams(**v)
      if a.variant == "candidate":
        cp.flags |= 2
        for sc in cp.safetyConfigs:
          if str(sc.safetyModel) == "ford":
            sc.safetyParam |= 8
      controller = CarController(DBC[CAR.FORD_EXPEDITION_MK4], cp)
      continue
    if k == "carState":
      cs = SimpleNamespace(
        out=structs.CarState(**v),
        buttons_stock_values=defaultdict(int),
        lkas_status_stock_values=defaultdict(int),
        acc_tja_status_stock_values=defaultdict(int),
      )
    if k == "carControl":
      cc = structs.CarControl(**v)
    if k != "sendcan" or controller is None or cs is None or cc is None:
      continue
    actual = [m for m in v if m[0] == 0x3D6]
    if not started:
      if not actual:
        continue
      # First observed counter supplies the steering frame phase; no fitting thereafter.
      controller.frame = ((actual[0][1][7] >> 1) & 15) * 5
      started = True
    frame = controller.frame
    _, msgs = controller.update(cc.as_reader(), cs, t)
    generated = [m for m in msgs if m[0] in (0x3CA, 0x3D6)]
    expected = [m for m in v if m[0] in (0x3CA, 0x3D6)]
    counts["updates"] += 1
    counts["expected_steering"] += len(actual)
    gensteer = [m for m in generated if m[0] == 0x3D6]
    if gensteer and actual:
      counts["steering_pairs"] += 1
      counts["steering_byte_matches"] += gensteer == actual
    elif gensteer or actual:
      counts["steering_schedule_mismatch"] += 1
    batches.append({"t": t / 1e9, "frame": frame, "generated": generated, "recorded": expected})
    if actual:
      rows.append(
        {
          "t": t / 1e9,
          "frame": frame,
          "recorded": actual[0][1].hex(),
          "generated": gensteer[0][1].hex() if gensteer else None,
          "latActive": bool(cc.latActive),
          "speed": float(cs.out.vEgoRaw),
          "steeringAngleDeg": float(cs.out.steeringAngleDeg),
          "yawRate": float(cs.out.yawRate),
          "curvature": float(cc.actuators.curvature),
          "shadow": float(controller.shadow_curvature),
        }
      )
  print(a.variant, f.stem, dict(counts), flush=True)
result = {
  "variant": a.variant,
  "repo": a.repo,
  "counts": dict(counts),
  "rows": rows,
  "batches": batches,
  "logs": hashes,
  "limitations": [
    "Nearest preceding logged carState/carControl are reconstructed inputs, not a proven exact controller snapshot.",
    "Stock UI/button inputs are zeroed for this lateral-only replay; compare steering byte agreement before interpreting candidate results.",
  ],
}
Path(a.output).write_bytes(pickle.dumps(result))
print(json.dumps(dict(counts)))
