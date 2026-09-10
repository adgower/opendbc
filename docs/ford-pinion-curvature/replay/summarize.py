import argparse
import json
import pickle
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--raw-windows", required=True)
p.add_argument("--targets", required=True)
a = p.parse_args()
O = Path(a.output)
paired = json.loads((O / "paired-native.json").read_text())
targets = json.loads(Path(a.targets).read_text())
raw = pickle.loads(Path(a.raw_windows).read_bytes())
origin = targets["origin"]
steers = [e for e in raw["events"] if e["k"] == "sendcan" and e["addr"] == 0x3D6]
returns = [e for e in raw["events"] if e["k"] == "can" and e["addr"] == 0x3D6 and e["bus"] in (128, 192)]
acks = {}
for e in steers:
  hits = [r for r in returns if r["d"] == e["d"] and 0 <= r["t"] - e["t"] <= 0.04]
  acks[e["t"]] = {"status": ("accepted" if hits[0]["bus"] == 128 else "rejected") if len(hits) == 1 else "ambiguous", "matches": len(hits)}
summary = {
  "reconstruction": paired["reconstruction"],
  "cases": paired["cases"],
  "source": paired["source"],
  "limitations": paired["limitations"],
  "targets": [],
  "new_rejections": [],
  "offsets": [],
}
for c in targets["targets"]:
  matches = [e for e in steers if e["d"].hex() == c["hex"] and 0 <= c["t"] - e["t"] <= 0.04]
  assert len(matches) == 1
  t = matches[0]["t"]
  rows = [r for r in paired["records"] if r["t"] == t]
  summary["targets"].append(
    {
      "seconds": c["seconds"],
      "send_t": t,
      "brake": c["brake"],
      "recorded_ack": acks[t],
      "offsets": [
        {
          "offset_ms": r["offset_ms"],
          **{
            v: {"accepted": x["accepted"], "controls_allowed": bool(x["before"][0]), "brake_pressed_prev": bool(x["before"][2])}
            for v, x in r["outcomes"].items()
          },
        }
        for r in rows
      ],
    }
  )
for r in paired["records"]:
  b, c = r["outcomes"]["baseline"], r["outcomes"]["candidate"]
  if b["accepted"] and not c["accepted"]:
    summary["new_rejections"].append(
      {
        "offset_ms": r["offset_ms"],
        "window": r["window"],
        "t": r["t"],
        "seconds": r["t"] - origin,
        "warmup_seconds": r["t"] - raw["windows"][r["window"]][0],
        "recorded_ack": acks[r["t"]],
        "baseline": b,
        "candidate": c,
      }
    )
for off in [-10, 0, 5, 10, 20]:
  rows = [r for r in paired["records"] if r["offset_ms"] == off]
  warmed = [r for r in rows if r["t"] - raw["windows"][r["window"]][0] >= 2 and acks[r["t"]]["status"] == "accepted" and r["outcomes"]["baseline"]["active"]]
  summary["offsets"].append(
    {
      "offset_ms": off,
      "all_steering": len(rows),
      "warmed_acknowledged_active_neighbors": len(warmed),
      "total_rejected": {v: sum(not r["outcomes"][v]["accepted"] for r in rows) for v in ["baseline", "candidate"]},
      "warmed_acknowledged_active_rejected": {v: sum(not r["outcomes"][v]["accepted"] for r in warmed) for v in ["baseline", "candidate"]},
    }
  )
(O / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({"targets": summary["targets"], "offsets": summary["offsets"], "new_rejections": summary["new_rejections"]}, indent=2))
