import math
from types import SimpleNamespace as NS
import pytest
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR, FordFlags
from opendbc.car.vehicle_model import VehicleModel
from opendbc.car.ford.lateral_angle_tracking_budget import BudgetedTrackingPrototype
from opendbc.car.ford.lateral_angle_roll_mapping import RollAwareMapping


def make(mapping=True):
  cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)
  cp.flags |= int(FordFlags.PINION_CURVATURE | FordFlags.LOW_SPEED_GAIN)
  return RollAwareMapping(cp, enabled=True, mapping_enabled=mapping), BudgetedTrackingPrototype(cp, enabled=True)


def step(
  c,
  i,
  k=0.003,
  roll=0.04,
  mph=60.0,
  valid=True,
  age=0.0,
  angle_error=0.0,
  active=True,
  driver=False,
  packet_roll=None,
  context_overrides=None,
  state_overrides=None,
  **overrides,
):
  v = mph * 0.44704
  now = 1_000_000_000 + i * 50_000_000
  vm = VehicleModel(c.CP)
  vm.update_params(1.0, 22.5)
  offset = -1.5
  angle = math.degrees(vm.get_steer_from_curvature(-k, v, roll)) + offset
  state = NS(
    vEgoRaw=v,
    vEgo=v,
    yawRate=0.0,
    steeringAngleDeg=offset,
    steeringPressed=driver,
    canValid=True,
    steerFaultTemporary=False,
    steerFaultPermanent=False,
    leftBlinker=False,
    rightBlinker=False,
  )
  if state_overrides:
    state.__dict__.update(state_overrides)
  context = dict(now_ns=now, state_ns=now, control_ns=now, model_ns=now, model_valid=True, lane_change=False)
  if context_overrides:
    context.update(context_overrides)
  c.set_context(**context)
  if hasattr(c, "set_calibration"):
    params = dict(now_ns=now, sample_ns=now - round(age * 1e9), valid=valid, roll=roll, ratio=22.5, stiffness=1.0, offset_deg=offset)
    params.update(overrides)
    if packet_roll is not None:
      params["roll"] = packet_roll
    c.set_calibration(**params)
  a = NS(curvature=k, steeringAngleDeg=angle + angle_error)
  return c.update(NS(latActive=active), NS(out=state), a)


@pytest.mark.parametrize("sign", [-1, 1])
def test_roll_assisting_turn_reduces_command(sign):
  c, b = make()
  # Determine the assist sign under the actual vehicle model, not assumed axes.
  vm = VehicleModel(c.CP)
  vm.update_params(1.0, 22.5)
  roll = 0.04 * sign
  k = math.copysign(0.003, -vm.roll_compensation(roll, 60 * 0.44704))
  for i in range(120):
    a = step(c, i, k, roll)
    z = step(b, i, k, roll)
  assert 0 < abs(a.path_angle) < abs(z.path_angle)
  assert abs(c.map_delta) <= min(0.01, 0.15 * abs(z.path_angle)) + 1e-12
  assert a.shadow_curvature == z.shadow_curvature


def test_opposing_roll_can_increase_command():
  c, b = make()
  vm = VehicleModel(c.CP)
  roll = 0.04
  k = math.copysign(0.003, vm.roll_compensation(roll, 60 * 0.44704))
  for i in range(120):
    a = step(c, i, k, roll)
    z = step(b, i, k, roll)
  assert abs(a.path_angle) > abs(z.path_angle)


@pytest.mark.parametrize("mph", [15, 45, 49, 50, 70, 75, 85])
def test_outside_mapping_scope_parity(mph):
  c, b = make()
  for i in range(80):
    assert step(c, i, mph=mph) == step(b, i, mph=mph)


@pytest.mark.parametrize(
  "case", ["disabled", "invalid", "stale", "future", "nan_roll", "bad_ratio", "bad_stiffness", "bad_offset", "wrong_target", "zero", "driver", "inactive"]
)
def test_invalid_or_neutral_falls_back_from_fresh_state(case):
  c, b = make(mapping=case != "disabled")
  kw = {}
  if case == "invalid":
    kw["valid"] = False
  if case == "stale":
    kw["age"] = 0.3
  if case == "future":
    kw["age"] = -0.1
  if case == "nan_roll":
    kw["roll"] = float("nan")
  if case == "bad_ratio":
    kw["ratio"] = 0
  if case == "bad_stiffness":
    kw["stiffness"] = 0
  if case == "bad_offset":
    kw["offset_deg"] = 100
  if case == "wrong_target":
    kw["angle_error"] = 5
  if case == "zero":
    kw["k"] = 0
  if case == "driver":
    kw["driver"] = True
  if case == "inactive":
    kw["active"] = False
  if case == "nan_roll":
    kw = {"packet_roll": float("nan")}
  for i in range(40):
    a = step(c, i, **kw)
    z = step(b, i, **{k: v for k, v in kw.items() if k not in ["ratio", "stiffness", "offset_deg"]})
    assert a == z
  assert c.map_delta == 0


def test_application_model_does_not_mutate_fixed_pinion():
  c, b = make()
  original = c.pinion_model.sR
  for i in range(40):
    step(c, i)
  assert c.pinion_model.sR == original == c.CP.steerRatio
  assert c.map_delta != 0


def test_reversal_does_not_carry_old_mapping_adjustment():
  c, b = make()
  for i in range(100):
    step(c, i, k=0.003, roll=0.04)
  assert abs(c.map_delta) > 0.001
  step(c, 100, k=-0.003, roll=-0.04)
  assert abs(c.map_delta) <= c.DELTA_SLEW_PER_FRAME + 1e-12


def test_invalid_during_mapping_clears_term_not_final_slew_state():
  c, b = make()
  for i in range(100):
    step(c, i)
  previous = c.path_angle_last
  result = step(c, 100, valid=False)
  assert c.map_delta == 0
  assert abs(result.path_angle - previous) <= 0.009 + 1e-12


def test_requested_reversal_clears_even_when_deviation_clip_keeps_old_sign():
  c, b = make()
  c.get_current_curvature = lambda CS: 0.006
  for i in range(100):
    step(c, i, k=0.003, roll=0.04)
  assert abs(c.map_delta) > 0.001
  step(c, 100, k=-0.003, roll=-0.04)
  assert c.map_base > 0
  assert abs(c.map_delta) <= c.DELTA_SLEW_PER_FRAME + 1e-12


@pytest.mark.parametrize("gate", ["cadence", "state_age", "control_age", "model_invalid", "state_invalid", "permanent_fault"])
def test_context_abort_gates_clear_established_mapping(gate):
  c, b = make()
  for i in range(100):
    step(c, i)
  assert c.map_delta != 0
  now = 1_000_000_000 + 100 * 50_000_000
  context = {}
  state = {}
  if gate == "cadence":
    context["now_ns"] = now + 200_000_000
  if gate == "state_age":
    context["state_ns"] = now - 90_000_000
  if gate == "control_age":
    context["control_ns"] = now - 110_000_000
  if gate == "model_invalid":
    context["model_valid"] = False
  if gate == "state_invalid":
    state["canValid"] = False
  if gate == "permanent_fault":
    state["steerFaultPermanent"] = True
  previous = c.path_angle_last
  output = step(c, 100, context_overrides=context, state_overrides=state)
  assert c.map_delta == 0
  assert abs(output.path_angle - previous) <= 0.009 + 1e-12


@pytest.mark.parametrize("ratio,stiffness,offset", [(12.0, 0.5, -4.0), (19.6, 1.0, 0.0), (30.0, 2.0, 3.0)])
def test_matching_zero_roll_target_does_not_invent_calibration_gain(ratio, stiffness, offset):
  c, b = make()
  vm = VehicleModel(c.CP)
  vm.update_params(stiffness, ratio)
  v = 60 * 0.44704
  k = 0.003
  for i in range(40):
    now = 1_000_000_000 + i * 50_000_000
    state = NS(
      vEgoRaw=v,
      vEgo=v,
      yawRate=0.0,
      steeringAngleDeg=0.0,
      steeringPressed=False,
      canValid=True,
      steerFaultTemporary=False,
      steerFaultPermanent=False,
      leftBlinker=False,
      rightBlinker=False,
    )
    a = NS(curvature=k, steeringAngleDeg=math.degrees(vm.get_steer_from_curvature(-k, v, 0.0)) + offset)
    for ctrl in [c, b]:
      ctrl.set_context(now_ns=now, state_ns=now, control_ns=now, model_ns=now, model_valid=True, lane_change=False)
    c.set_calibration(now_ns=now, sample_ns=now, valid=True, roll=0.0, ratio=ratio, stiffness=stiffness, offset_deg=offset)
    result = c.update(NS(latActive=True), NS(out=state), a)
    reference = b.update(NS(latActive=True), NS(out=state), a)
    assert c.map_reason == "mapped"
    assert abs(c.map_delta) < 1e-12
    assert abs(result.path_angle - reference.path_angle) < 1e-12
