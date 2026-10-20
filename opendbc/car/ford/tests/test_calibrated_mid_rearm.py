from pathlib import Path
import json
from opendbc.car.ford.tests.test_calibrated_mid_unwind import make, step
from opendbc.car.ford.lateral_angle_calibrated_mid_unwind import MidEpisode
from opendbc.car.ford.lateral_angle_sustained_unwind import SustainedTracking
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR, FordFlags
import pytest


def saved_observer():
  cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)
  cp.flags |= int(FordFlags.PINION_CURVATURE | FordFlags.LOW_SPEED_GAIN)
  return SustainedTracking(cp, enabled=True, extra_enabled=True)


def replay_saved(mid):
  o = saved_observer()
  rows = json.loads((Path(__file__).parent / "fixtures/calibrated_mid_rearm_26mph.json").read_text())
  out = []
  for r in rows:
    if not 146 <= r["t"] < 167:
      continue
    o.feedback_valid = r["low"]["feedback_valid"]
    o.set_evidence(
      dict(
        active=r["active"],
        state_valid=r["high"]["input_state_valid"],
        driver_or_fault=bool(r["pressed"] or r["brake"] or r["blinkers"]),
        context=r["context"],
        calibration=r["calibration"],
        motion=r["motion"],
        generation=r["low"]["_sustained_generation"],
        target=r["target"],
        actual=r["actual"],
        model_v=r["model_speed"],
      )
    )
    term = mid.update(o, r["k"], r["mph"] * 0.44704, r["new"])
    out.append(dict(t=r["t"], term=term, area=mid.area, aborted=mid.aborted, rearms=getattr(mid, "rearm_count", 0), reason=mid.reason))
  return out


def test_saved_26mph_turn_rearms_after_actual_valid_quiet_and_receives_bounded_effort():
  m = MidEpisode()
  r = replay_saved(m)
  assert any(x["term"] for x in r if 147 <= x["t"] < 150)
  rearmed = [x for x in r if 150 < x["t"] < 163 and x["rearms"] > 0]
  assert rearmed, "Actual quiet road interval must qualify before the second turn"
  later = [x for x in r if 163 <= x["t"] < 165.49 and abs(x["term"]) > 0]
  assert later, "Second turn must receive finite effort before native clipping binds"
  assert max(abs(x["term"]) for x in r) <= 0.0025 + 1e-12
  assert all(x["area"] <= 0.0015 + 1e-12 for x in r)


@pytest.mark.parametrize("kind", ["quantized", "body_noise"])
def test_bounded_neutral_noise_allows_one_rearm(kind):
  c = make()
  for i in range(30):
    step(c, i, mph=27, k=0.008, wheel_ratio=0.8, body_ratio=0.8)
  assert c.mid.area > 0
  for i in range(30, 70):
    # Independent observed changes remain inside absolute neutral bounds.
    kw = dict(mph=27, k=0.0001, excess=0, body_ratio=1.0)
    if kind == "quantized":
      kw["excess"] = 0.1 if i % 2 else 0.0
    else:
      kw["body_ratio"] = 2.0 if i % 2 else 0.0
    step(c, i, **kw)
  assert c.mid.area == 0
  assert c.mid.rearm_count == 1


@pytest.mark.parametrize("kind", ["driver", "body_invalid", "duplicate_body", "slow_drift", "large_body_steps", "request", "body_turn"])
def test_noise_tolerance_cannot_refill_without_continuously_settled_evidence(kind):
  c = make()
  for i in range(30):
    step(c, i, mph=27, k=0.008, wheel_ratio=0.8, body_ratio=0.8)
  before = c.mid.area
  for i in range(30, 70):
    kw = dict(mph=27, k=0.0001, excess=0, body_ratio=1.0)
    if kind == "driver":
      kw["driver"] = True
    if kind == "body_invalid":
      kw["body_valid"] = False
    if kind == "duplicate_body":
      kw["body_age"] = 0.05 if i % 2 else 0.0
    if kind == "slow_drift":
      kw["body_ratio"] = (i - 30) * 0.15
    if kind == "large_body_steps":
      kw["body_ratio"] = 4.0 if i % 2 else 0.0
    if kind == "request":
      kw["k"] = 0.001
    if kind == "body_turn":
      kw["body_ratio"] = 8.0
    step(c, i, **kw)
  assert c.mid.area == before
  assert getattr(c.mid, "rearm_count", 0) == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_demand_cannot_count_as_neutral_or_refill(bad):
  o = saved_observer()
  m = MidEpisode()
  m.area = 0.0015
  m.start = 1
  m.aborted = True
  for i in range(30):
    now = 1_000_000_000 + i * 50_000_000
    o.set_evidence(
      dict(
        active=True,
        state_valid=True,
        driver_or_fault=False,
        context=(now, now, now, now, False, True),
        calibration=(now, now, True, 0.0, 22.5, 1.0, -1.5),
        motion=(now, now, now, True, 0.0, 0.013),
        generation=0,
        target=-1.5,
        actual=-1.5,
        model_v=27 * 0.44704,
      )
    )
    m.update(o, bad, 27 * 0.44704, 0.01)
  assert m.area == 0.0015 and m.aborted and m.rearm_count == 0
