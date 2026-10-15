import pytest
from opendbc.car.ford.lateral_angle_tracking_budget import BudgetedTrackingPrototype as RampTrackingPrototype
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR, FordFlags
from opendbc.car.ford.lateral_angle_tracking_release import ReleaseTrackingPrototype


def make(cls=RampTrackingPrototype, enabled=True):
  cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)
  cp.flags |= int(FordFlags.PINION_CURVATURE | FordFlags.LOW_SPEED_GAIN)
  return cls(cp, enabled=enabled)


def step(t, i, k, m, **overrides):
  now = 1_000_000_000 + i * 50_000_000
  ctx = dict(now_ns=now, state_ns=now, control_ns=now, model_ns=now, model_valid=True, lane_change=False)
  ctx.update(overrides)
  t.set_context(**ctx)
  v = 15 * 0.44704
  return t.correction(v, k, m, k * v * t.get_curvature_factor(v, k), False)


@pytest.mark.parametrize("sign", [-1, 1])
def test_rising_fractional_shortfall_acts_sooner(sign):
  t = make()
  b = make(ReleaseTrackingPrototype)
  first = None
  old = None
  for i in range(40):
    k = sign * (0.002 + i * 0.0002)
    if step(t, i, k, 0.4 * k) != 0 and first is None:
      first = i
    if step(b, i, k, 0.4 * k) != 0 and old is None:
      old = i
  assert first is not None and first < 25
  assert old is None or first < old


def test_rising_response_that_catches_up_not_boosted():
  t = make()
  for i in range(40):
    k = 0.002 + i * 0.0002
    m = k * min(0.95, 0.15 + i * 0.04)
    assert step(t, i, k, m) == 0


def test_silent_actuator_not_boosted():
  t = make()
  for i in range(100):
    assert step(t, i, 0.006, 0) == 0


def test_rising_admission_uses_smaller_bound():
  t = make()
  seen = False
  for i in range(40):
    k = 0.002 + i * 0.0002
    x = step(t, i, k, 0.4 * k)
    if t.admission_kind == "rising" and x:
      seen = True
      assert abs(x) <= min(0.00375, 0.0625 * abs(k * 15 * 0.44704 * t.get_curvature_factor(15 * 0.44704, k))) + 1e-12
  assert seen


def test_single_large_jump_not_treated_as_ramp():
  t = make()
  for i in range(35):
    step(t, i, 0.004, 0.0016)
  assert t.tracking_correction > 0
  for i in range(35, 44):
    assert step(t, i, 0.008, 0.0032) == 0


@pytest.mark.parametrize("kind", ["flag", "platform", "driver", "fault", "nan"])
def test_ramp_specific_exclusions(kind):
  t = make()
  for i in range(17):
    step(t, i, 0.002 + i * 0.0002, 0.4 * (0.002 + i * 0.0002))
  assert t.tracking_correction > 0
  if kind == "flag":
    t.CP.flags &= ~int(FordFlags.LOW_SPEED_GAIN)
  if kind == "platform":
    t.CP.carFingerprint = CAR.FORD_F_150_MK14
  if kind == "fault":
    t.feedback_valid = False
  now = 1_850_000_000
  t.set_context(now_ns=now, state_ns=now, control_ns=now, model_ns=now, model_valid=True, lane_change=False)
  x = t.correction(15 * 0.44704, 0.008, float("nan") if kind == "nan" else 0.0032, 0.08, kind == "driver")
  assert x == 0 and not t.history


def test_fade_cutoff_keeps_slew_history_and_converges():
  import numpy as np
  from types import SimpleNamespace as NS
  from opendbc.car.ford.lateral_angle import _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES

  t = make()
  b = make(ReleaseTrackingPrototype)
  cs = NS(out=NS(vEgoRaw=15 * 0.44704, yawRate=0.0, steeringAngleDeg=0.0, steeringPressed=False, canValid=True))
  # Supply explicit measured curvature while exercising the real update/output bounds.
  measured = [0.001]
  t.get_current_curvature = lambda c: measured[0]
  b.get_current_curvature = lambda c: measured[0]
  for i in range(40):
    k = 0.002 + i * 0.0002
    measured[0] = k * 0.4
    now = 1_000_000_000 + i * 50_000_000
    for c in [t, b]:
      c.set_context(now_ns=now, state_ns=now, control_ns=now, model_ns=now, model_valid=True, lane_change=False)
    t.update(NS(latActive=True), cs, NS(curvature=k))
    b.update(NS(latActive=True), cs, NS(curvature=k))
  cs.out.vEgoRaw = 45 * 0.44704
  measured[0] = 0.0
  previous = t.path_angle_last
  for i in range(40, 80):
    now = 1_000_000_000 + i * 50_000_000
    for c in [t, b]:
      c.set_context(now_ns=now, state_ns=now, control_ns=now, model_ns=now, model_valid=True, lane_change=False)
    x = t.update(NS(latActive=True), cs, NS(curvature=-0.006))
    y = b.update(NS(latActive=True), cs, NS(curvature=-0.006))
    assert t.tracking_correction == 0
    assert abs(x.path_angle - previous) <= np.interp(cs.out.vEgoRaw, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES) + 1e-12
    previous = x.path_angle
  assert x == y


def test_saved_turn_admits_before_installed_strategy():
  import json
  from pathlib import Path

  rows = json.loads((Path(__file__).parent / "fixtures/navigator_tracking_turn.json").read_text())["rows"]
  first = []
  for cls in [RampTrackingPrototype, ReleaseTrackingPrototype]:
    t = make(cls)
    when = None
    for at, v, k, m, ff, driver, active, sa, ca, ma, lane, mv in rows:
      now = 1_000_000_000 + round(at * 1e9)
      t.set_context(
        now_ns=now,
        state_ns=now - round(sa * 1e9),
        control_ns=now - round(ca * 1e9),
        model_ns=now - round(ma * 1e9),
        model_valid=bool(mv),
        lane_change=bool(lane),
      )
      if not active:
        t.reset_tracking()
        continue
      term = t.correction(v, k, m, ff, bool(driver))
      if at > 885 and term and when is None:
        when = at
    first.append(when)
  assert all(x is not None for x in first)
  assert first[0] <= first[1] - 0.5
  assert first[0] < 889
