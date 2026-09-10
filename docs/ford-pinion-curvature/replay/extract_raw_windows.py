"""Standalone raw CAN extraction; do not import opendbc in this schema process."""

import argparse
import capnp
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--targets", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
D = Path(a.data)
targets = json.loads(Path(a.targets).read_text())["targets"]
windows = []
for c in targets:
  lo, hi = c["t"] - 5, c["t"] + 2
  if windows and lo <= windows[-1][1]:
    windows[-1][1] = max(hi, windows[-1][1])
  else:
    windows.append([lo, hi])
capnp.remove_import_hook()
schema = capnp.load(str(D / "schema/log.capnp"), imports=[str(D / "schema")])
events = []
seq = 0
batch = 0
files = []
for p in sorted((D / "logs").glob("*/rlog.zst"), key=lambda p: int(p.parent.name.rsplit("--", 1)[1])):
  files.append(dict(path=str(p.resolve()), size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  for e in schema.Event.read_multiple_bytes(subprocess.check_output(["zstd", "-dc", str(p)])):
    t = e.logMonoTime / 1e9
    k = e.which()
    if k not in ("can", "sendcan") or not any(lo <= t <= hi for lo, hi in windows):
      continue
    for m in getattr(e, k):
      events.append(dict(t=t, k=k, addr=m.address, bus=m.src, d=bytes(m.dat), seq=seq, batch=batch))
      seq += 1
    batch += 1
  print(p.parent.name, flush=True)
events.sort(key=lambda e: (e["t"], e["seq"]))
identity = dict(data_path=str(D.resolve()), raw_files=files, schema_sha256=hashlib.sha256((D / "schema/log.capnp").read_bytes()).hexdigest())
Path(a.output).write_bytes(pickle.dumps(dict(windows=windows, events=events, identity=identity)))
