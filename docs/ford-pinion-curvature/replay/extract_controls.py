from pathlib import Path
import argparse
import capnp
import pickle
import subprocess
import hashlib

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
D = Path(a.data)
O = Path(a.output)
O.mkdir(parents=True, exist_ok=True)
capnp.remove_import_hook()
schema = capnp.load(str(D / "schema/log.capnp"), imports=[str(D / "schema")])
for f in sorted((D / "logs").glob("*/rlog.zst"), key=lambda p: int(p.parent.name.rsplit("--", 1)[1])):
  out = O / (f.parent.name + ".pkl")
  if out.exists():
    continue
  events = []
  for e in schema.Event.read_multiple_bytes(subprocess.check_output(["zstd", "-dc", str(f)])):
    k = e.which()
    if k not in ("carParams", "carControl", "carState", "sendcan"):
      continue
    if k == "sendcan":
      v = [(m.address, bytes(m.dat), m.src) for m in e.sendcan]
    else:
      v = getattr(e, k).to_dict()
    events.append((e.logMonoTime, k, v))
  out.write_bytes(pickle.dumps(dict(events=events, hash={"segment": f.parent.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})))
  print(f.parent.name, len(events), flush=True)
