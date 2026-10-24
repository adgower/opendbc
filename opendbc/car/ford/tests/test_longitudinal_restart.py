import pytest
from opendbc.car.ford.longitudinal_admission import ObservedClearBarrier
from opendbc.car.ford.longitudinal_coordination import Torque
from opendbc.car.ford.longitudinal_restart import RestartClearance


def update(b, now, *, speed=2.8, torque=0.0, sample=None, active=True, owner=True, brake=False, phase="pid", valid=True):
  return b.update(now, active, brake, speed, phase, Torque(now if sample is None else sample, torque, valid), owner_active=owner)


def warm(b):
  for now in [1.0, 1.02, 1.04, 1.06]:
    update(b, now)


def test_three_fresh_distinct_samples_carry_on_speed_only_crossing():
  b = RestartClearance(True)
  warm(b)
  r = update(b, 1.08, speed=3.01)
  assert r.allow and not r.fault and b.carried


@pytest.mark.parametrize(
  "kw",
  [
    {"torque": 6.0},
    {"torque": -1.0},
    {"torque": 32760.0},
    {"valid": False},
    {"sample": 0.8},
    {"sample": 1.09},
    {"sample": 1.03},
    {"brake": True},
    {"active": False},
    {"owner": False},
    {"phase": "stopping"},
  ],
)
def test_nonclear_invalid_override_or_phase_change_cannot_carry(kw):
  b = RestartClearance(True)
  warm(b)
  r = update(b, 1.08, speed=3.01, **kw)
  assert not b.carried
  assert not r.allow or not kw.get("active", True) or kw.get("phase") == "stopping"


@pytest.mark.parametrize("mode", ["one_sample", "too_short", "repeated", "late_callback", "sample_gap", "clock_reverse"])
def test_insufficient_or_discontinuous_evidence_cannot_carry(mode):
  b = RestartClearance(True)
  if mode == "one_sample":
    update(b, 1.0)
    now = 1.02
  elif mode == "too_short":
    for x in [1.0, 1.005, 1.01]:
      update(b, x)
    now = 1.015
  elif mode == "repeated":
    for x in [1.0, 1.02, 1.04, 1.06]:
      update(b, x, sample=1.0)
    now = 1.08
  elif mode == "late_callback":
    warm(b)
    now = 1.18
  elif mode == "sample_gap":
    warm(b)
    now = 1.12
  else:
    warm(b)
    now = 1.05
  r = update(b, now, speed=3.01, sample=1.0 if mode == "repeated" else None)
  assert not b.carried and not r.allow


@pytest.mark.parametrize("interruption", [{"brake": True}, {"active": False}, {"owner": False}, {"phase": "stopping"}, {"torque": 8.0}, {"valid": False}])
def test_interruption_discards_qualification_before_crossing(interruption):
  b = RestartClearance(True)
  warm(b)
  update(b, 1.08, **interruption)
  r = update(b, 1.10, speed=3.01)
  assert not b.carried and not r.allow


def test_clearance_survives_small_speed_hunting_with_fresh_measurements():
  b = RestartClearance(True)
  for n in range(8):
    update(b, 1.0 + n * 0.02, speed=3.1)
  update(b, 1.16, speed=2.99)
  assert update(b, 1.18, speed=3.01).allow and b.carried


@pytest.mark.parametrize("phase", ["pid", "stopping", "starting", "off"])
def test_below_boundary_outputs_match_installed(phase):
  a, b = RestartClearance(True), ObservedClearBarrier(True)
  for n in range(100):
    kw = dict(speed=2.8, phase=phase, torque=12.0 if n % 3 == 0 else 0.0, brake=n % 4 == 0, active=phase != "off", owner=phase != "off")
    assert update(a, 1 + n * 0.02, **kw) == update(b, 1 + n * 0.02, **kw)


def test_steady_moving_output_and_release_timeout_match_installed():
  a, b = RestartClearance(True), ObservedClearBarrier(True)
  for n in range(100):
    kw = dict(speed=18.0, torque=0.0 if n < 8 else 156.0)
    assert update(a, 1 + n * 0.02, **kw) == update(b, 1 + n * 0.02, **kw)
  assert a.fault == b.fault == "release_timeout_disengage"


def test_latched_fault_cannot_be_cleared_by_speed_crossing():
  b = RestartClearance(True)
  for n in range(60):
    update(b, 1 + n * 0.02, speed=18.0, torque=156.0)
  for n in range(10):
    assert update(b, 2.2 + n * 0.02).fault
  assert update(b, 2.4, speed=3.01).fault and not b.carried


def test_owner_reset_requires_new_clearance():
  b = RestartClearance(True)
  warm(b)
  b.reset()
  assert not update(b, 1.08, speed=3.01).allow


def test_latched_fault_survives_owner_enabled_pedal_override_and_crossing():
  b = RestartClearance(True)
  for n in range(60):
    update(b, 1 + n * 0.02, speed=18.0, torque=156.0)
  assert update(b, 2.2, active=False, owner=True).fault
  assert update(b, 2.22).fault
  assert update(b, 2.24, speed=3.01).fault and not b.carried


def test_controller_context_fault_still_withdraws_after_prequalification():
  from opendbc.car.ford.tests.test_longitudinal_admission import Scene

  s = Scene()
  for _ in range(100):
    out, _ = s.step(raw=0.7, pitch=0.0, speed=2.8)
  assert out.navigatorLongitudinal.gas > 0.69
  s.c.coordination_control_valid = False
  out, _ = s.step(raw=0.7, pitch=0.0, speed=3.01)
  assert out.navigatorLongitudinal.fault and out.navigatorLongitudinal.gas == -5.0


def test_controller_preserves_request_at_speed_only_crossing():
  from opendbc.car.ford.tests.test_longitudinal_admission import Scene

  s = Scene()
  for _ in range(150):
    out, _ = s.step(raw=0.7, pitch=0.0, speed=2.8)
  assert out.navigatorLongitudinal.gas > 0.69
  out, _ = s.step(raw=0.7, pitch=0.0, speed=3.01)
  assert out.navigatorLongitudinal.gas > 0.69 and not out.navigatorLongitudinal.brake and not out.navigatorLongitudinal.fault
