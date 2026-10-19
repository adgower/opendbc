"""Behavioral probes for fresh wheel/body-confirmed high-speed unwind trim."""

import math
import pytest
from opendbc.car.ford.lateral_angle_sustained_unwind import SustainedErrorUnwind as HighSpeedReduction
from opendbc.car.ford.tests.test_high_speed import make as make_reference
from types import SimpleNamespace as NS
from opendbc.car.vehicle_model import VehicleModel


def step(
  c,
  i,
  *,
  mph=83.0,
  k=0.00065,
  excess=2.4,
  valid=True,
  active=True,
  driver=False,
  lane=False,
  age=0.0,
  mismatch=0.0,
  roll=0.0,
  state_valid=True,
  fault=False,
  body_ratio=1.35,
  body_valid=True,
  body_age=0.0,
  body_std=0.013,
  state_lag_ns=0,
  generation=0,
  stepdt_ns=50_000_000,
):
  v = mph * 0.44704
  now = 1_000_000_000 + i * stepdt_ns
  vm = VehicleModel(c.CP)
  vm.update_params(1.0, 22.5)
  angle = math.degrees(vm.get_steer_from_curvature(-k, v, roll if math.isfinite(roll) else 0.0)) - 1.5
  state = NS(
    vEgoRaw=v,
    vEgo=v,
    yawRate=0.0,
    steeringAngleDeg=angle + math.copysign(1.0, angle + 1.5) * excess,
    steeringPressed=driver,
    canValid=state_valid,
    steerFaultTemporary=fault,
    steerFaultPermanent=False,
    leftBlinker=False,
    rightBlinker=False,
  )
  c.set_context(now_ns=now, state_ns=now - state_lag_ns, control_ns=now, model_ns=now, model_valid=True, lane_change=lane)
  c.set_calibration(now_ns=now, sample_ns=now - round(age * 1e9), valid=valid, roll=roll, ratio=22.5, stiffness=1.0, offset_deg=-1.5)
  if hasattr(c, "set_motion"):
    c.set_motion(
      now_ns=now, sample_ns=now - round(body_age * 1e9), received_ns=now, valid=body_valid, yaw_rate=k * v * body_ratio, yaw_std=body_std, generation=generation
    )
  return c.update(NS(latActive=active), NS(out=state), NS(curvature=k, steeringAngleDeg=angle + mismatch))


def make(enabled=True):
  c = HighSpeedReduction(make_reference().CP, enabled=enabled)
  c.input_state_valid = True
  return c


@pytest.mark.parametrize("sign", [-1, 1])
def test_sustained_real_excess_during_unwind_can_reduce(sign):
  # Catches an unconditional old-peak latch that ignores fresh wheel/body excess.
  c = make()
  for i in range(60):
    step(c, i, k=sign * 0.00065)
  for j in range(30):
    step(c, 60 + j, k=sign * 0.00065 * 0.99**j)
  assert c.actual_reduction > 0
  assert abs(c.actual_final_residual) > 0
  assert abs(c.actual_final_residual) <= min(0.01, 0.15 * abs(c.actual_reference_path)) + 1e-12


@pytest.mark.parametrize("sign", [-1, 1])
def test_closing_wheel_error_releases_before_it_reaches_target(sign):
  # Catches delayed release that waits for measured wheel to cross the deadband.
  c = make()
  for i in range(60):
    step(c, i, k=sign * 0.00065)
  for j in range(9):
    step(c, 60 + j, k=sign * 0.00065, excess=2.4 - 0.2 * j)
  # Current excess is .8deg (> .5deg deadband), but closes by .6deg in150ms.
  assert c.actual_reduction == 0 and abs(c.actual_final_residual) < 1e-12


@pytest.mark.parametrize("mph", [15, 25, 45, 55, 60])
def test_lower_speed_whole_strategy_parity(mph):
  c = make()
  b = make_reference()
  for i in range(160):
    kw = dict(mph=mph, k=0.00065 * (1 if i < 60 else 0.99 ** (i - 60)), driver=100 <= i < 104)
    assert step(c, i, **kw) == step(b, i, **kw)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize(
  "kwargs",
  [{"driver": True}, {"body_ratio": 0.8}, {"body_age": 0.2}, {"state_valid": False}, {"mismatch": 1.0}, {"fault": True}, {"lane": True}, {"active": False}],
)
def test_invalid_body_driver_or_context_releases_existing_trim(sign, kwargs):
  c = make()
  for i in range(60):
    step(c, i, k=sign * 0.00065)
  assert c.actual_reduction > 0
  step(c, 60, k=sign * 0.00065, **kwargs)
  assert c.actual_reduction == 0 and abs(c.actual_final_residual) < 1e-12


@pytest.mark.parametrize("sign", [-1, 1])
def test_already_under_target_exit_never_gets_reduction(sign):
  c = make()
  for i in range(60):
    step(c, i, k=sign * 0.00065)
  for j in range(25):
    a = step(c, 60 + j, k=sign * 0.00065 * 0.98**j, excess=-0.3, body_ratio=1.35)
    assert c.actual_reduction == 0 and abs(c.actual_final_residual) < 1e-12
    assert a.path_angle == c.actual_reference_path


def test_default_off_preserves_installed_reference():
  c = make(False)
  b = make_reference(False)
  for i in range(240):
    kw = dict(mph=55 + i * 0.125, k=0.00065 * math.sin(i / 30), driver=160 <= i < 166)
    assert step(c, i, **kw) == step(b, i, **kw)


def test_duplicate_wheel_source_cannot_earn_fresh_history():
  c = make()
  for i in range(40):
    step(c, i, state_lag_ns=40_000_000 if i % 2 else 0, stepdt_ns=40_000_000)
    assert c.actual_reduction == 0 and abs(c.actual_final_residual) < 1e-12


def test_new_motion_generation_restarts_wheel_evidence():
  c = make()
  for i in range(60):
    step(c, i)
  assert c.actual_reduction > 0
  for i in range(60, 64):
    step(c, i, generation=1)
    assert c.actual_reduction == 0 and abs(c.actual_final_residual) < 1e-12


@pytest.mark.parametrize("sign", [-1, 1])
def test_transition_abort_sweep_preserves_native_envelopes(sign):
  c = make()
  previous = 0.0
  for i in range(6000):
    mph = 50 + 35 * (0.5 + 0.5 * math.sin(i / 137))
    k = sign * 0.0012 * math.sin(i / 53)
    a = step(c, i, mph=mph, k=k, excess=1.5 + 1.4 * math.sin(i / 17), driver=i % 211 < 3, body_ratio=1.25 + 0.3 * math.sin(i / 37))
    b = c.actual_reference_path
    rate = (
      0.055
      if mph * 0.44704 <= 10
      else (
        0.055 + (mph * 0.44704 - 10) * (0.0425 - 0.055) / 5
        if mph * 0.44704 < 15
        else (0.0425 + (mph * 0.44704 - 15) * (0.009 - 0.0425) / 10 if mph * 0.44704 < 25 else 0.009)
      )
    )
    assert abs(a.path_angle - previous) <= rate + 1e-10
    previous = a.path_angle
    assert a.path_angle * b >= -1e-12 and abs(a.path_angle) <= abs(b) + 1e-12
    assert abs(a.path_angle - b) <= min(0.01, 0.15 * abs(b)) + 1e-12
  assert c.infeasible_intersections == 0


@pytest.mark.parametrize("sign", [-1, 1])
def test_highspeed_warmup_does_not_delay_installed_correction_below60(sign):
  c = make()
  b = make_reference()
  for i in range(7):
    step(c, i, k=sign * 0.00065, mph=83)
    step(b, i, k=sign * 0.00065, mph=83)
  for i in range(7, 20):
    kw = dict(k=sign * 0.00065, mph=55)
    assert step(c, i, **kw) == step(b, i, **kw)


def test_explicit_disable_cannot_be_overwritten_by_baseline_state():
  c = make()
  for i in range(60):
    step(c, i)
  assert c.actual_reduction > 0
  c.enabled = False
  a = step(c, 60)
  assert not c.enabled
  assert c.actual_reduction == 0 and abs(c.actual_final_residual) < 1e-12
  assert a.path_angle == c.actual_reference_path
