import math
from types import SimpleNamespace as NS
import pytest
from opendbc.car.ford.lateral_angle_high_speed import HighSpeedReduction
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR, FordFlags
from opendbc.car.ford.lateral_angle_roll_mapping import RollAwareMapping
from opendbc.car.vehicle_model import VehicleModel


def make(enabled=True):
  cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)
  cp.flags |= int(FordFlags.PINION_CURVATURE | FordFlags.LOW_SPEED_GAIN)
  c = HighSpeedReduction(cp, enabled=enabled)
  c.input_state_valid = True
  return c


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
):
  v = mph * 0.44704
  now = 1_000_000_000 + i * 50_000_000
  vm = VehicleModel(c.CP)
  vm.update_params(1.0, 22.5)
  angle = math.degrees(vm.get_steer_from_curvature(-k, v, roll if math.isfinite(roll) else 0.0)) - 1.5
  state = NS(
    vEgoRaw=v,
    vEgo=v,
    yawRate=0.0,
    steeringAngleDeg=angle + math.copysign(excess, angle + 1.5),
    steeringPressed=driver,
    canValid=state_valid,
    steerFaultTemporary=fault,
    steerFaultPermanent=False,
    leftBlinker=False,
    rightBlinker=False,
  )
  c.set_context(now_ns=now, state_ns=now, control_ns=now, model_ns=now, model_valid=True, lane_change=lane)
  c.set_calibration(now_ns=now, sample_ns=now - round(age * 1e9), valid=valid, roll=roll, ratio=22.5, stiffness=1.0, offset_deg=-1.5)
  if hasattr(c, "set_motion"):
    c.set_motion(now_ns=now, sample_ns=now - round(body_age * 1e9), received_ns=now, valid=body_valid, yaw_rate=k * v * body_ratio, yaw_std=body_std)
  return c.update(NS(latActive=active), NS(out=state), NS(curvature=k, steeringAngleDeg=angle + mismatch))


@pytest.mark.parametrize("sign", [-1, 1])
def test_83mph_persistent_gentle_wheel_excess_reduces(sign):
  c = make()
  for i in range(100):
    a = step(c, i, k=sign * 0.00065)
  b = c.reference.path_angle_last
  assert 0 < abs(a.path_angle) < abs(b)
  assert abs(a.path_angle - b) <= min(0.01, 0.15 * abs(b)) + 1e-12
  assert a.shadow_curvature == c.reference.kappa_cmd


@pytest.mark.parametrize("mph", [15, 25, 45, 55, 59.9])
def test_lower_speeds_unchanged(mph):
  c = make()
  b = RollAwareMapping(c.CP, enabled=True)
  for i in range(120):
    assert step(c, i, mph=mph) == step(b, i, mph=mph)


@pytest.mark.parametrize(
  "kwargs",
  [
    {"excess": 0.0},
    {"excess": -0.5},
    {"valid": False},
    {"driver": True},
    {"active": False},
    {"lane": True},
    {"age": 0.3},
    {"age": -0.1},
    {"mismatch": 5.0},
    {"k": 0.002},
    {"k": 0.0},
    {"roll": float("nan")},
    {"state_valid": False},
    {"fault": True},
  ],
)
def test_ineligible_never_starts(kwargs):
  c = make()
  b = RollAwareMapping(c.CP, enabled=True)
  for i in range(80):
    assert step(c, i, **kwargs) == step(b, i, **kwargs)
  assert c.reduction == 0


def test_default_off_exact():
  c = make(False)
  b = RollAwareMapping(c.CP, enabled=True)
  for i in range(180):
    kwargs = dict(mph=40 + i * 0.25, k=0.00065 * math.sin(i / 25), roll=0.02)
    assert step(c, i, **kwargs) == step(b, i, **kwargs)


def test_reversal_release_and_bounded_residual():
  c = make()
  for i in range(100):
    step(c, i)
  assert c.reduction > 0
  for i in range(100, 170):
    previous = c.path_angle_last
    a = step(c, i, k=-0.00065)
    b = c.reference.path_angle_last
    assert abs(a.path_angle - previous) <= 0.009 + 1e-12
    assert a.path_angle * b >= -1e-12 and abs(a.path_angle) <= abs(b) + 1e-12
    assert abs(a.path_angle - b) <= min(0.01, 0.15 * abs(b)) + 1e-12


def test_driver_release_is_immediate_reference():
  c = make()
  for i in range(100):
    step(c, i)
  a = step(c, 100, driver=True)
  assert a.path_angle == c.reference.path_angle_last
  assert c.reduction == 0


@pytest.mark.parametrize("abort", [{"valid": False}, {"age": 0.3}, {"lane": True}, {"fault": True}, {"state_valid": False}, {"k": 0.0003}, {"k": -0.00065}])
def test_withdrawal_after_activation_rejoins_reference(abort):
  c = make()
  for i in range(100):
    step(c, i)
  assert c.reduction > 0
  for i in range(100, 130):
    previous = c.path_angle_last
    a = step(c, i, **abort)
    if i == 100 or "k" not in abort:
      assert c.reduction == 0
    assert abs(a.path_angle - previous) <= 0.009 + 1e-12
  if "k" not in abort:
    assert c.final_residual == 0
  assert not c.release_pending


def test_invalid_state_envelope_after_activation():
  c = make()
  for i in range(100):
    step(c, i)
  c.input_state_valid = False
  step(c, 100)
  assert c.reduction == 0 and c.final_residual == 0


def test_small_demand_jitter_is_not_an_abort():
  c = make()
  for i in range(120):
    step(c, i, k=0.00065 * (1 + 0.01 * math.sin(i)))
  assert c.reduction > 0


def test_persistence_resets_until_release_finishes():
  c = make()
  for i in range(100):
    step(c, i)
  # With stationary high-speed reference, bounded residual can fully release in
  # one frame (<.009rad). Explicitly emulate a pending max-slew baseline tail.
  c.release_pending = True
  c.final_residual = -0.004
  c._persistent = 0
  step(c, 100)
  assert c.reduction == 0 and c._persistent == 0
  for i in range(101, 105):
    step(c, i)
  assert c.reduction == 0
  step(c, 105)
  assert c.reduction > 0


@pytest.mark.parametrize("phase", [0.0, 1.0, 2.0])
def test_smooth_speed_curve_transition_and_abort_sweep(phase):
  c = make()
  for i in range(4000):
    mph = 60 + 25 * math.sin(i / 300 + phase)
    k = 0.0008 * math.sin(i / 80) + 0.00015 * math.sin(i / 11)
    old = c.path_angle_last
    a = step(c, i, mph=mph, k=k, excess=2.4 if i % 120 < 100 else -0.2, valid=i % 311 != 0, lane=i % 277 == 0, driver=i % 137 == 0)
    b = c.reference.path_angle_last
    from numpy import interp
    from opendbc.car.ford.lateral_angle import _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES

    assert abs(a.path_angle - old) <= float(interp(mph * 0.44704, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES)) + 1e-12
    assert a.path_angle * b >= -1e-12 and abs(a.path_angle) <= abs(b) + 1e-12
    assert abs(a.path_angle - b) <= min(0.01, 0.15 * abs(b)) + 1e-12
  assert c.infeasible_intersections == 0


def test_slow_unwind_is_latched_after_cumulative_retreat():
  c = make()
  for i in range(100):
    step(c, i)
  assert c.reduction > 0
  for j in range(45):
    a = step(c, 100 + j, k=0.00065 * 0.99**j)
    b = c.reference.path_angle_last
    assert abs(a.path_angle) <= abs(b) + 1e-12
    assert abs(a.path_angle - b) <= min(0.01, 0.15 * abs(b)) + 1e-12
    if j >= 12:
      assert c.reduction == 0 and abs(c.final_residual) < 1e-12
  assert c.unwind_latched


def test_owner_factors_and_fixed_pinion_model_unchanged():
  from opendbc.car.ford.lateral_angle import _FORD_LOW_SPEED_FACTOR_ANG, _FORD_HIGH_SPEED_FACTOR_ANG

  assert (_FORD_LOW_SPEED_FACTOR_ANG, _FORD_HIGH_SPEED_FACTOR_ANG) == (1.20, 0.97)
  c = make()
  fixed = c.pinion_model.sR
  for i in range(100):
    step(c, i)
  assert c.pinion_model.sR == fixed == c.CP.steerRatio
  assert c.reference.application_model.sR == 22.5


@pytest.mark.parametrize(
  "kwargs",
  [{"body_ratio": 0.8}, {"body_ratio": 1.0}, {"body_valid": False}, {"body_age": 0.2}, {"body_age": -0.1}, {"body_std": float("nan")}, {"body_std": 0.1}],
)
def test_wheel_excess_without_fresh_body_confirmation_never_reduces(kwargs):
  c = make()
  for i in range(100):
    a = step(c, i, **kwargs)
    assert a.path_angle == c.reference.path_angle_last
  assert c.reduction == 0


@pytest.mark.parametrize("kwargs", [{"body_ratio": 0.8}, {"body_valid": False}, {"body_age": 0.2}])
def test_body_withdrawal_aborts_existing_correction(kwargs):
  c = make()
  for i in range(100):
    step(c, i)
  assert c.reduction > 0
  step(c, 100, **kwargs)
  assert c.reduction == 0 and c.final_residual == 0


def test_withdrawal_does_not_rearm_on_lower_plateau():
  c = make()
  for i in range(100):
    step(c, i)
  for i in range(100, 180):
    step(c, i, k=0.0005)
  assert c.reduction == 0
  # Direction change establishes a new bend, subject to new body/persistence.
  for i in range(180, 240):
    step(c, i, k=-0.00065)
  assert c.reduction > 0


def test_missing_motion_context_is_default_reference():
  c = make()
  # Bypass only the fixture's motion feed, leaving real state/context/target.
  if hasattr(c, "set_motion"):
    c.set_motion = lambda **kw: None
  for i in range(100):
    step(c, i)
  assert c.reduction == 0


def test_demand_before_admission_cannot_latch_future_correction():
  c = make()
  for i in range(30):
    step(c, i, body_ratio=0.8)
  for i in range(30, 40):
    step(c, i, k=0.0005, body_ratio=0.8)
  for i in range(40, 100):
    step(c, i, k=0.0008)
  assert c.reduction > 0


@pytest.mark.parametrize("mutation", ["future_source", "future_arrival", "arrival_before_source", "duplicate", "out_of_order"])
def test_motion_temporal_rejection(mutation):
  c = make()
  original = c.set_motion

  def modify(**kw):
    if mutation == "future_source":
      kw["sample_ns"] = kw["now_ns"] + 1
    if mutation == "future_arrival":
      kw["received_ns"] = kw["now_ns"] + 1
    if mutation == "arrival_before_source":
      kw["received_ns"] = kw["sample_ns"] - 1
    if mutation == "duplicate":
      kw["sample_ns"] = 1_000_000_000
    if mutation == "out_of_order":
      kw["sample_ns"] = 1_000_000_000 - kw["now_ns"] // 10
    original(**kw)

  c.set_motion = modify
  for i in range(100):
    step(c, i)
  assert c.reduction == 0


@pytest.mark.parametrize("decay", [0.998, 0.995, 0.99])
def test_cumulative_unwind_cannot_forget_peak(decay):
  c = make()
  for i in range(100):
    step(c, i)
  for j in range(160):
    step(c, 100 + j, k=0.00065 * decay**j)
    if decay ** max(0, j - 1) < 0.9:
      assert c.reduction == 0
  if 0.00065 * decay**159 * (83 * 0.44704) ** 2 >= 0.25:
    assert c.unwind_latched


def test_recent_out_of_order_source_revokes_active_body_confirmation():
  c = make()
  for i in range(100):
    step(c, i)
  assert c.reduction > 0
  prior = c._motion_highwater
  original = c.set_motion

  def stale_order(**kw):
    kw["sample_ns"] = prior - 1
    original(**kw)

  c.set_motion = stale_order
  step(c, 100)
  assert c.body_reason == "invalid" and c.reduction == 0


def test_unique_body_history_span_boundary():
  c = make()
  v = 83 * 0.44704
  k = 0.00065
  for elapsed in [0, 40_000_000, 79_999_999]:
    source = 1_000_000_000 + elapsed
    now = source + 1_000_000
    c.set_motion(now_ns=now, sample_ns=source, received_ns=now, valid=True, yaw_rate=k * v * 1.35, yaw_std=0.013)
    assert not c._body_confirms(now, k, v)
  source = 1_080_000_000
  now = source + 1_000_000
  c.set_motion(now_ns=now, sample_ns=source, received_ns=now, valid=True, yaw_rate=k * v * 1.35, yaw_std=0.013)
  assert c._body_confirms(now, k, v)


def test_calibration_generation_discards_old_body_history():
  c = make()
  for i in range(100):
    step(c, i)
  assert c.reduction > 0
  original = c.set_motion

  def new_generation(**kw):
    kw["generation"] = 1
    original(**kw)

  c.set_motion = new_generation
  step(c, 100)
  assert c.reduction == 0 and len(c.body_history) == 1


def test_new_rising_demand_above_retained_peak_can_rearm():
  c = make()
  for i in range(100):
    step(c, i)
  for i in range(100, 140):
    step(c, i, k=0.0005)
  assert c.reduction == 0 and c.unwind_latched
  for i in range(140, 190):
    step(c, i, k=0.0008)
  assert c.reduction > 0 and not c.unwind_latched


def test_new_peak_requires_body_confirmation_to_rearm():
  c = make()
  for i in range(100):
    step(c, i)
  for i in range(100, 140):
    step(c, i, k=0.0005)
  for i in range(140, 190):
    step(c, i, k=0.0008, body_ratio=0.8)
  assert c.reduction == 0 and c.unwind_latched


@pytest.mark.parametrize(
  "bad_reset",
  [
    {"mph": 0.0, "state_valid": False},
    {"k": 0.0, "valid": False},
    {"k": -0.00065, "state_valid": False},
    {"active": False, "valid": False},
    {"driver": True, "state_valid": False},
  ],
)
def test_invalid_reset_evidence_cannot_clear_unwind_latch(bad_reset):
  c = make()
  for i in range(100):
    step(c, i)
  for i in range(100, 120):
    step(c, i, k=0.0005)
  assert c.unwind_latched
  step(c, 120, **bad_reset)
  assert c.unwind_latched
  for i in range(121, 160):
    step(c, i, k=0.0005)
  assert c.reduction == 0 and c.unwind_latched


def test_one_frame_demand_dip_is_not_persistent_unwind():
  c = make()
  for i in range(100):
    step(c, i)
  step(c, 100, k=0.00054)
  assert not c.unwind_latched
  for i in range(101, 130):
    step(c, i)
  assert c.reduction > 0


@pytest.mark.parametrize("sign", [-1, 1])
def test_driver_release_during_fast_rising_reference_preserves_native_slew(sign):
  c = make()
  for i in range(100):
    step(c, i, k=sign * 0.00065)
  previous = c.path_angle_last
  step(c, 100, k=sign * 0.0014, driver=True)
  assert c.reduction == 0
  assert abs(c.path_angle_last - previous) <= 0.009 + 1e-12
