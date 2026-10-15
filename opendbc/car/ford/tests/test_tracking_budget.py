import pytest
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR, FordFlags
from opendbc.car.ford.lateral_angle_tracking_budget import BudgetedTrackingPrototype


def make():
  cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)
  cp.flags |= int(FordFlags.PINION_CURVATURE | FordFlags.LOW_SPEED_GAIN)
  return BudgetedTrackingPrototype(cp, enabled=True)


def step(t, i, k, m, *, valid=True, lane=False, driver=False, dt=0.05, state_age=0.0):
  now = 1_000_000_000 + round(i * dt * 1e9)
  t.set_context(now_ns=now, state_ns=now - round(state_age * 1e9), control_ns=now, model_ns=now, model_valid=valid, lane_change=lane)
  v = 15 * 0.44704
  return t.correction(v, k, m, k * v * t.get_curvature_factor(v, k), driver)


def exhaust(t):
  area = 0.0
  for i in range(80):
    k = min(0.012, 0.002 + i * 0.0002)
    area += abs(step(t, i, k, 0.4 * k)) * 0.05
  assert 0 < area <= t.RAMP_EXTRA_AREA_LIMIT + 1e-12
  assert abs(area-t.RAMP_EXTRA_AREA_LIMIT) < 1e-12
  return area


@pytest.mark.parametrize("valid", [False, True])
def test_isolated_quiet_sample_cannot_rearm(valid):
  t = make()
  area = exhaust(t)
  step(t, 80, 0.0, 0.0, valid=valid)
  for i in range(81, 150):
    k = min(0.012, 0.002 + (i - 81) * 0.0002)
    area += abs(step(t, i, k, 0.4 * k)) * 0.05
  assert area <= t.RAMP_EXTRA_AREA_LIMIT + 1e-12


def test_reversal_cannot_rearm_without_settling():
  t = make()
  area = exhaust(t)
  for i in range(80, 150):
    k = -min(0.012, 0.002 + (i - 80) * 0.0002)
    area += abs(step(t, i, k, 0.4 * k)) * 0.05
  assert area <= t.RAMP_EXTRA_AREA_LIMIT + 1e-12


@pytest.mark.parametrize("bad", ["invalid", "lane", "driver", "measured", "cadence", "stale"])
def test_interrupted_quiet_requires_a_complete_new_dwell(bad):
  t = make()
  exhaust(t)
  for i in range(80, 99):
    step(t, i, 0.0, 0.0)
  step(
    t,
    99,
    0.0,
    0.002 if bad == "measured" else 0.0,
    valid=bad != "invalid",
    lane=bad == "lane",
    driver=bad == "driver",
    dt=0.1 if bad == "cadence" else 0.05,
    state_age=0.2 if bad == "stale" else 0.0,
  )
  for i in range(100, 118):
    step(t, i, 0.0, 0.0)
  assert t.rearm_count == 0


def test_settled_quiet_rearms_once_and_new_ramp_can_spend():
  t = make()
  exhaust(t)
  for i in range(80, 125):
    step(t, i, 0.0, 0.0)
  assert t.rearm_count == 1 and t.ramp_extra_area == 0
  area = 0.0
  for i in range(125, 180):
    k = min(0.012, 0.002 + (i - 125) * 0.0002)
    area += abs(step(t, i, k, 0.4 * k)) * 0.05
  assert 0 < area <= t.RAMP_EXTRA_AREA_LIMIT + 1e-12


def test_quiet_demand_with_unsettled_feedback_does_not_rearm():
  t = make()
  exhaust(t)
  for i in range(80, 160):
    step(t, i, 0.0, 0.0004 if i % 2 else -0.0004)
  assert t.rearm_count == 0 and t.ramp_extra_area > 0


@pytest.mark.parametrize("sign", [-1, 1])
def test_budget_bound_with_variable_valid_cadence(sign):
  t = make()
  now = 1_000_000_000
  area = 0.0
  for i in range(160):
    dt = 0.04 if i % 2 else 0.06
    now += round(dt * 1e9)
    t.set_context(now_ns=now, state_ns=now, control_ns=now, model_ns=now, model_valid=True, lane_change=False)
    k = sign * min(0.012, 0.002 + i * 0.0002)
    v = 15 * 0.44704
    term = t.correction(v, k, 0.4 * k, k * v * t.get_curvature_factor(v, k), False)
    area += abs(term) * dt
  assert 0 < area <= t.RAMP_EXTRA_AREA_LIMIT + 1e-12
  assert t.tracking_correction == 0


@pytest.mark.parametrize("human_turn", [False, True])
def test_quiet_dwell_does_not_bridge_inactive_or_human_override(human_turn):
  from types import SimpleNamespace as NS

  t = make()
  exhaust(t)
  for i in range(80, 99):
    step(t, i, 0.0, 0.0)
  t.get_current_curvature = lambda _: 0.0
  t.human_turn_detector = NS(reset=lambda: None, update=lambda *args: human_turn)
  cs = NS(out=NS(vEgoRaw=15 * 0.44704, canValid=True, steeringPressed=False, steeringAngleDeg=0.0))
  t.update(NS(latActive=human_turn), cs, NS(curvature=0.0))
  for i in range(100, 118):
    step(t, i, 0.0, 0.0)
  assert t.rearm_count == 0
