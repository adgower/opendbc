import pytest
from opendbc.car.ford.lateral_angle_midrange_admission import MidrangeAdmission
from opendbc.car.ford.tests.test_observer_continuity import make as make_controller
from opendbc.car.ford.tests.test_wheel_target import step


def make():
  return make_controller(MidrangeAdmission)


@pytest.mark.parametrize("body,excess", [(0.718, 0), (0.716, 0), (0.799, 2.4), (0.795, 0), (1.281, 0), (1.537, 2.4)])
def test_disagreeing_transition_cannot_admit(body, excess):
  c = make()
  for i in range(12):
    step(c, i, mph=45, k=0.002 * (1 + 0.03 * i), roll=-0.04, body_ratio=body, excess=excess)
  assert c.reference.map_delta == 0


def test_no_instantaneous_admission():
  c = make()
  for i in range(8):
    step(c, i, mph=45, k=0.002, roll=-0.04, body_ratio=1.2)
  assert c.reference.map_delta == 0


@pytest.mark.parametrize("fault", [dict(body_age=0.2), dict(body_valid=False), dict(driver=True), dict(lane=True), dict(valid=False)])
def test_earned_admission_releases_on_abort(fault):
  c = make()
  for i in range(50):
    step(c, i, mph=45, k=0.002, roll=0.04, body_ratio=1.2)
  assert c.admission_mode == -1 and c.reference.map_delta
  step(c, 50, mph=45, k=0.002, roll=0.04, body_ratio=1.2, **fault)
  assert c.admission_mode == 0 and c.reference.map_delta == 0


def test_target_jump_requires_new_qualified_history():
  c = make()
  for i in range(50):
    step(c, i, mph=45, k=0.002, roll=0.04, body_ratio=1.2)
  step(c, 50, mph=45, k=0.003, roll=0.04, body_ratio=1.2)
  assert c.admission_mode == 0


def test_unqualified_fallback_retains_original_mapping_history():
  from opendbc.car.ford.lateral_angle_observer_continuity import ObserverContinuity

  c = make()
  b = ObserverContinuity(c.CP, enabled=True)
  b.input_state_valid = True
  for i in range(80):
    kw = dict(mph=52, k=0.002, roll=0.04, body_ratio=0.7)
    assert step(c, i, **kw) == step(b, i, **kw)


def test_fresh_scheduler_jitter_keeps_independent_history():
  from types import SimpleNamespace as NS

  # Use real step provenance, but controlled ~75ms scheduler cadence.
  c = make()
  for i in range(50):
    step(c, i, mph=45, k=0.002, roll=0.04, body_ratio=1.2)
  # Physical source freshness is still independently enforced by Observer.
  assert c.admission_mode == -1
  observer = c.admission_observer
  old = observer.rows[-1]
  # A fresh75ms interval should retain history rather than a cadence reset.
  k = 0.002
  now = old[0] + 75_000_000
  c.set_context(now_ns=now, state_ns=now, control_ns=now, model_ns=now, model_valid=True, lane_change=False)
  c.set_calibration(now_ns=now, sample_ns=now, valid=True, roll=0.04, ratio=22.5, stiffness=1.0, offset_deg=-1.5)
  c.set_motion(now_ns=now, sample_ns=now, received_ns=now, valid=True, yaw_rate=k * 45 * 0.44704 * 1.2, yaw_std=0.013)
  import math

  vm = observer.model
  angle = math.degrees(vm.get_steer_from_curvature(-k, 45 * 0.44704, 0.04)) - 1.5
  state = NS(vEgoRaw=45 * 0.44704, vEgo=45 * 0.44704, steeringAngleDeg=angle - 2.4, canValid=True, steeringPressed=False)
  assert observer.feed(c, NS(latActive=True), NS(out=state), NS(curvature=k, steeringAngleDeg=angle)) is not None


@pytest.mark.parametrize("sign", [-1, 1])
def test_coherent_underresponse_admits_only_bounded_increase(sign):
  c = make()
  for i in range(50):
    step(c, i, mph=45, k=sign * 0.002, roll=-sign * 0.04, body_ratio=0.8, excess=-2)
  assert c.admission_mode == 1
  assert c.reference.map_reason == "mapped"
  assert c.reference.map_delta * c.reference.map_base > 0
  assert abs(c.reference.map_delta) <= min(0.01, 0.15 * abs(c.reference.map_base)) + 1e-12


def test_explicit_factory_preserves_specialized_baselines():
  from opendbc.car.ford.lateral_angle_very_low_speed_prototype import create_tracking_trial
  from opendbc.car.ford.lateral_angle_observer_continuity import ObserverContinuity

  c = make()
  built = create_tracking_trial(c.CP, enabled=True)
  assert isinstance(built, ObserverContinuity)
  assert len(built.admission_leaves) == 16
  assert type(built.reference.baseline).__name__ == "EarlyLow"
  assert built.reference.baseline.observer_reason == "initial"
  assert type(built.stock.reference.baseline).__name__ == "SustainedTracking"
  assert type(built.unchanged.reference.baseline).__name__ == "CalibratedTracking"
  assert not create_tracking_trial(c.CP).admission_leaves


def test_extension_boundaries_continuous():
  mapping = make().reference
  for mph in [35, 40, 50, 55]:
    weights = [mapping.mapping_weight((mph + offset) * 0.44704) for offset in [-1e-5, 0, 1e-5]]
    assert max(weights) - min(weights) < 1e-6
