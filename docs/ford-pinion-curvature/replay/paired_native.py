from pathlib import Path
from collections import Counter
import argparse
import json
import pickle
import subprocess
import sys
import shutil
import time
import hashlib

p = argparse.ArgumentParser()
p.add_argument("--baseline", required=True)
p.add_argument("--candidate", required=True)
p.add_argument("--raw-windows", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
O = Path(a.output)
W = O / f"native-{time.time_ns()}"
W.mkdir()
B = Path(a.baseline)
C = Path(a.candidate)
sys.path.insert(0, str(C))
from opendbc.safety.tests.libsafety import libsafety_py as l

l.ffi.cdef("void audit_state(int *out);")
source = """
void audit_state(int *out) {
 int v[]={controls_allowed,relay_malfunction,brake_pressed_prev,gas_pressed_prev,vehicle_moving,
 desired_path_angle_last,ford_shadow_curvature_raw,ford_angle_mode_engaged,
 curvature_state.meas.min,curvature_state.meas.max,vehicle_speed.values[0],curvature_state.desired_last};
 for(int i=0;i<12;i++){out[i]=v[i];}
}
"""
controller = {v: pickle.loads((O / f"{v}.pkl").read_bytes()) for v in ["baseline", "candidate"]}
generated = {v: {(b["t"], m[0]): m for b in controller[v]["batches"] for m in b["generated"]} for v in controller}
for label, root in [("baseline", B), ("candidate", C)]:
  p = W / f"{label}.c"
  p.write_text('#include "' + str(root / "opendbc/safety/tests/libsafety/safety.c") + '"\n' + source)
  subprocess.run(["cc", "-shared", "-fPIC", "-DALLOW_DEBUG", "-O2", "-std=gnu11", "-I", str(root), str(p), "-o", str(W / f"{label}.so")], check=True)
data = pickle.loads(Path(a.raw_windows).read_bytes())
events = data["events"]
windows = data["windows"]
results = []
images = []
records = []
for offset in [-10, 0, 5, 10, 20]:
  for wi, (lo, hi) in enumerate(windows):
    stream = [
      dict(e, rt=e["t"] + (offset / 1000 if e["k"] == "sendcan" else 0)) for e in events if lo <= e["t"] <= hi and (e["k"] == "sendcan" or e["bus"] < 128)
    ]
    stream.sort(key=lambda e: (e["rt"], e["seq"]))
    libs = {}
    states = {}
    counts = {v: Counter() for v in controller}
    for v in controller:
      path = W / f"{v}_{offset}_{wi}.so"
      shutil.copyfile(W / f"{v}.so", path)
      lib = l.ffi.dlopen(str(path))
      images.append(lib)
      lib.set_safety_hooks(6, 3 if v == "baseline" else 11)
      lib.init_tests()
      libs[v] = lib
      states[v] = l.ffi.new("int[12]")
    for e in stream:
      outcomes = {}
      for v, lib in libs.items():
        lib.set_timer(int(e["rt"] * 1e6) % 2**32)
        dat = e["d"]
        if e["k"] == "sendcan" and e["addr"] in (0x3CA, 0x3D6):
          addr, dat, bus = generated[v][(e["t"], e["addr"])]
          assert addr == e["addr"] and bus == e["bus"]
          if v == "baseline":
            assert dat == e["d"]
        packet = l.make_CANPacket(e["addr"], e["bus"], dat)
        lib.audit_state(states[v])
        before = list(states[v])
        if e["k"] == "can":
          lib.safety_fwd_hook(e["bus"], e["addr"])
          ok = bool(lib.safety_rx_hook(packet))
          counts[v]["rx"] += 1
          if not ok:
            counts[v]["rx_invalid"] += 1
        else:
          ok = bool(lib.safety_tx_hook(packet))
          counts[v]["tx"] += 1
          if e["addr"] == 0x3D6:
            active = ((dat[0] >> 4) & 7) != 0
            counts[v]["steering"] += 1
            counts[v]["steering_rejected"] += not ok
            counts[v]["active_steering"] += active
            counts[v]["active_steering_rejected"] += active and not ok
            counts[v]["rejected_with_controls_allowed"] += not ok and bool(before[0])
            counts[v]["rejected_without_controls_allowed"] += not ok and not bool(before[0])
            outcomes[v] = {"accepted": ok, "active": active, "before": before, "data": dat.hex()}
      if outcomes:
        records.append({"offset_ms": offset, "window": wi, "t": e["t"], "outcomes": outcomes})
    row = {"offset_ms": offset, "window": wi, "interval": [lo, hi], "variants": {v: dict(c) for v, c in counts.items()}}
    results.append(row)
    print(row, flush=True)
match = {}
for v, p in controller.items():
  c = Counter()
  for b in p["batches"]:
    for addr in (0x3CA, 0x3D6):
      r = [m for m in b["recorded"] if m[0] == addr]
      g = [m for m in b["generated"] if m[0] == addr]
      if r or g:
        c[f"{addr:x}_batches"] += 1
        c[f"{addr:x}_byte_matches"] += r == g
        c[f"{addr:x}_schedule_mismatch"] += bool(r) != bool(g)
  match[v] = dict(c)
result = {
  "reconstruction": match,
  "cases": results,
  "records": records,
  "identity": data["identity"],
  "source": {
    v: {
      "commit": (
        subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if (repo / ".git").exists()
        else "immutable extracted snapshot; no git metadata"
      ),
      "ford_safety_sha256": hashlib.sha256((repo / "opendbc/safety/modes/ford.h").read_bytes()).hexdigest(),
    }
    for v, repo in [("baseline", B), ("candidate", C)]
  },
  "limitations": [
    "Controller reconstruction uses nearest preceding full logged CS and CC; zero-filled stock UI/button messages are excluded from comparison.",
    "All baseline LMC2 and LKA payloads and schedule exactly match their recorded counterparts, validating the lateral reconstruction on this route.",
    "Native safety initializes separately at each seven-second window; it has no exact pre-window MCU state. No controls_allowed override is used.",
    "MCU periodic ticks, interrupts, SPI transport and vehicle feedback are unmodeled. RX integrity hooks and per-TX pinion freshness do run.",
    "The five offsets perturb TX against fixed RX log timestamps; none is asserted to equal MCU ordering.",
    "Regenerated candidate commands use recorded baseline feedback. This counterfactual replay does not predict closed-loop vehicle outcomes.",
  ],
}
(O / "paired-native.json").write_text(json.dumps(result, indent=2))
print("DONE", flush=True)
