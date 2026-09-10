"""Extract only analysis-required fields; one decompressed raw segment at a time."""

import argparse
import collections
import hashlib
import json
import pickle
import subprocess
from pathlib import Path
import capnp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, required=True, help="Contains logs/<route>--<segment>/rlog.zst and schema/log.capnp.")
parser.add_argument("--output", type=Path, required=True, help="Directory for extracted.pkl and extraction-manifest.json; no raw logs are copied.")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
capnp.remove_import_hook()
schema = capnp.load(str(args.data_root / "schema/log.capnp"), imports=[str(args.data_root / "schema")])
D = collections.defaultdict(list)
manifest = []
for f in sorted((args.data_root / "logs").glob("*/rlog.zst"), key=lambda p: int(p.parent.name.rsplit("--", 1)[1])):
  blob = f.read_bytes()
  manifest.append(dict(segment=f.parent.name, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()))
  for e in schema.Event.read_multiple_bytes(subprocess.check_output(["zstd", "-dc", str(f)])):
    k = e.which()
    row = dict(t=e.logMonoTime / 1e9, valid=e.valid)
    if k == "carState":
      s = e.carState
      row.update({a: getattr(s, a) for a in ["vEgoRaw", "brakePressed", "steeringPressed"]})
    elif k == "carControl":
      row.update(latActive=e.carControl.latActive)
    elif k == "deviceMotion":
      s = e.deviceMotion
      row.update(
        motionTime=s.timestamp / 1e9,
        omega=[s.angularVelocityDevice.x, s.angularVelocityDevice.y, s.angularVelocityDevice.z],
        omegaValid=s.angularVelocityDevice.valid,
        sensorsOK=s.sensorsOK,
        inputsOK=s.inputsOK,
      )
    elif k == "extrinsicsCalibration":
      row.update(rpy=list(e.extrinsicsCalibration.rpyCalib), status=str(e.extrinsicsCalibration.calStatus))
    elif k == "initData":
      pass
    elif k == "can":
      for c in e.can:
        if c.address == 0x3D6 and len(c.dat) == 8:
          D[k].append(dict(row, addr=c.address, bus=c.src, mode=(bytes(c.dat)[0] >> 4) & 7))
      continue
    else:
      continue
    D[k].append(row)
  print(f.parent.name, flush=True)
with (args.output / "extracted.pkl").open("wb") as f:
  pickle.dump(dict(D), f)
(args.output / "extraction-manifest.json").write_text(json.dumps(dict(files=manifest, selected_counts={k: len(v) for k, v in D.items()}), indent=2))
