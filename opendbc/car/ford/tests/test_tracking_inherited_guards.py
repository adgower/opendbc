from types import SimpleNamespace as NS
from opendbc.car.ford.lateral_angle_tracking_budget import BudgetedTrackingPrototype as ReleaseTrackingPrototype
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR, FordFlags


def make(enabled=True):
  cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)
  cp.flags |= int(FordFlags.PINION_CURVATURE | FordFlags.LOW_SPEED_GAIN)
  return ReleaseTrackingPrototype(cp, enabled=enabled)


def context(t, now, **kw):
  d = dict(now_ns=now, state_ns=now, control_ns=now, model_ns=now, lane_change=False, model_valid=True)
  d.update(kw)
  t.set_context(**d)


def step(t, i, measured=0.003, **kw):
  context(t, 1000000000 + i * 50000000, **kw)
  return t.correction(15 * 0.44704, 0.006, measured, 0.0725, False)


def test_missing_context_fails_closed():
  t = make()
  for i in range(80):
    assert t.correction(6.7, 0.006, 0.0, 0.08, False) == 0


def test_context_guards():
  for kw in [dict(state_ns=0), dict(control_ns=0), dict(model_ns=0), dict(lane_change=True), dict(model_valid=False), dict(state_ns=99000000000)]:
    t = make()
    for i in range(80):
      assert step(t, i, **kw) == 0


def test_catchup_veto_and_stable_deficit():
  t = make()
  for i in range(70):
    x = step(t, i)
  assert x > 0
  for i in range(11):
    x = step(t, 70 + i, 0.0003 * i)
  assert x == 0


def test_discontinuous_clock_resets():
  t = make()
  for i in range(70):
    step(t, i)
  assert t.tracking_correction > 0
  assert step(t, 1) == 0
  assert t.hold_frames == 0


def test_silent_response_never_gets_blind_boost():
  t = make()
  for i in range(100):
    assert step(t, i, 0.0) == 0
  for i in range(70):
    x = step(t, 100 + i, 0.003)
  assert x > 0


import math
import pytest
from opendbc.car.ford.lateral_angle_very_low_speed_prototype import VeryLowSpeedGainPrototype


@pytest.mark.parametrize("enabled", [False, True])
def test_baseline_parity(enabled):
  t = make(enabled)
  b = VeryLowSpeedGainPrototype(t.CP, enabled=True)
  state = NS(out=NS(vEgoRaw=0.0, yawRate=0.0, steeringAngleDeg=0.0, steeringPressed=False, canValid=True))
  cc = NS(latActive=True)
  a = NS(curvature=0.006)
  for i in range(200):
    state.out.vEgoRaw = ([15, 20, 24, 25, 45][i // 40] if not enabled else [25, 45][i // 100]) * 0.44704
    context(t, 1000000000 + i * 50000000)
    assert t.update(cc, state, a) == b.update(cc, state, a)


@pytest.mark.parametrize("kind", ["driver", "inactive", "invalid", "fault", "speed", "reversal"])
def test_live_update_reset(kind):
  t = make()
  v = 15 * 0.44704
  wheel = -math.degrees(t.pinion_model.get_steer_from_curvature(0.003, v, 0.0))
  s = NS(out=NS(vEgoRaw=v, yawRate=0.0, steeringAngleDeg=wheel, steeringPressed=False, canValid=True))
  cc = NS(latActive=True)
  a = NS(curvature=0.006)
  for i in range(70):
    context(t, 1000000000 + i * 50000000)
    t.update(cc, s, a)
  assert t.tracking_correction > 0
  if kind == "driver":
    s.out.steeringPressed = True
  if kind == "inactive":
    cc.latActive = False
  if kind == "invalid":
    s.out.canValid = False
  if kind == "fault":
    s.out.steerFaultPermanent = True
  if kind == "speed":
    s.out.vEgoRaw = 25 * 0.44704
  if kind == "reversal":
    a.curvature = -0.006
  context(t, 4500000000)
  t.update(cc, s, a)
  assert t.tracking_correction == 0


@pytest.mark.parametrize("mph,w", [(5, 1), (15, 1), (20, 0.5), (25, 0), (45, 0)])
def test_fade(mph, w):
  assert ReleaseTrackingPrototype.speed_weight(mph * 0.44704) == pytest.approx(w)


def test_caps_range_rate_and_factors():
  from opendbc.car.ford.lateral_angle import _FORD_LOW_SPEED_FACTOR_ANG, _FORD_HIGH_SPEED_FACTOR_ANG

  assert (_FORD_LOW_SPEED_FACTOR_ANG, _FORD_HIGH_SPEED_FACTOR_ANG) == (1.2, 0.97)
  t = make()
  last = 0.0
  s = NS(out=NS(vEgoRaw=6.7, yawRate=0.0, steeringAngleDeg=0.0, steeringPressed=False, canValid=True))
  for i in range(1000):
    k = (0.006, -0.006, 0.5, -0.5)[i // 100 % 4]
    s.out.steeringAngleDeg = -math.degrees(t.pinion_model.get_steer_from_curvature(k * 0.5, 6.7, 0.0))
    context(t, 1000000000 + i * 50000000)
    x = t.update(NS(latActive=True), s, NS(curvature=k))
    ff = k * 6.7 * t.get_curvature_factor(6.7, k)
    assert abs(t.tracking_correction) <= min(0.0075, 0.125 * abs(ff)) + 1e-12
    assert -0.5 <= x.path_angle <= 0.5235 and abs(x.path_angle - last) <= 0.055 + 1e-12
    last = x.path_angle


def test_fast_cadence_fails_closed():
  t = make()
  for i in range(70):
    step(t, i)
  assert t.tracking_correction > 0
  context(t, 4460000000)
  assert t.correction(6.7, 0.006, 0.003, 0.0725, False) == 0
  assert not t.context_ok
