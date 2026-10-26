import math
from opendbc.car.ford.carcontroller import CarController
from opendbc.car.ford.tests import test_longitudinal_admission as contracts
from opendbc.car.ford.longitudinal_coordination import GRADE_OFFSET
from opendbc.car.ford.values import DBC


class ProtocolController(CarController):
  def update(self, CC, CS, now_ns):
    cc = CC.as_builder()
    q = cc.navigatorLongitudinal
    q.allocatorVersion = getattr(CS, "test_version", 1)
    q.targetAcceleration = getattr(CS, "test_target", float(CC.actuators.accel))
    q.targetMonoTime = getattr(CS, "test_target_ns", now_ns - 20_000_000)
    q.targetValid = getattr(CS, "test_target_valid", True)
    return super().update(cc.as_reader(), CS, now_ns)


class ProtocolScene(contracts.Scene):
  def __init__(self):
    super().__init__()
    self.c = ProtocolController(DBC[self.c.CP.carFingerprint], self.c.CP)
    self.c.longitudinal_coordination_enabled = True
    self.c.coordination_control_valid = True
    self.c.grade_map.amount = GRADE_OFFSET


def test_precharge_is_actuation_not_release_and_budget_commits_brake():
  s = ProtocolScene()
  s.warm()
  seen = []
  for _ in range(12):
    out, _ = s.step(0.3, math.asin((-0.13 - 0.3 - GRADE_OFFSET) / 9.81), torque=50)
    seen.append((s.now, s.c.precharge_request, out.navigatorLongitudinal.brake))
    assert out.navigatorLongitudinal.gas == -5 and not out.navigatorLongitudinal.fault
    if s.c.precharge_request and not out.navigatorLongitudinal.brake:
      assert not s.c.last_observer.allow and s.c.release_barrier.started is None
      assert out.navigatorLongitudinal.limited
  assert any(p and not b for _, p, b in seen)
  assert all(b for _, p, b in seen[5:])


def test_release_clock_starts_after_both_bits_off_then_requires_received_clearance():
  s = ProtocolScene()
  s.warm()
  for _ in range(12):
    s.step(0.3, math.asin((-0.13 - 0.3 - GRADE_OFFSET) / 9.81), torque=50)
  for _ in range(15):
    out, _ = s.step(0.6, 0.0, torque=50)
    assert out.navigatorLongitudinal.gas == -5 and not out.navigatorLongitudinal.fault
  assert not s.c.precharge_request and not out.navigatorLongitudinal.brake and s.c.release_barrier.started is not None
  for _ in range(2):
    out, _ = s.step(0.6, 0.0, torque=0)
    assert not s.c.last_observer.allow and out.navigatorLongitudinal.gas == -5
  out, _ = s.step(0.6, 0.0, torque=0)
  assert s.c.last_observer.allow and not out.navigatorLongitudinal.fault


def test_stalled_release_still_faults_without_gas_after_precharge():
  s = ProtocolScene()
  s.warm()
  for _ in range(12):
    s.step(0.3, math.asin((-0.13 - 0.3 - GRADE_OFFSET) / 9.81), torque=50)
  rows = []
  for _ in range(70):
    out, _ = s.step(0.6, 0.0, torque=50)
    rows.append(out.navigatorLongitudinal.fault)
    assert out.navigatorLongitudinal.gas == -5
  assert any(rows)


def test_retained_torque_cannot_rearm_allocator_even_with_positive_intent():
  s = ProtocolScene()
  s.warm()
  for _ in range(12):
    s.step(0.3, math.asin((-0.13 - 0.3 - GRADE_OFFSET) / 9.81), torque=50)
  assert s.c.precharge_allocator.locked
  for _ in range(35):
    s.step(0.6, 0.0, torque=50)
  assert s.c.precharge_allocator.locked and not s.c.last_observer.allow
  for _ in range(35):
    s.step(0.6, 0.0, torque=0)
  assert s.c.last_observer.allow and not s.c.precharge_allocator.locked


def test_missing_future_stale_or_invalid_request_metadata_selects_nominal():
  for changes in [
    dict(test_version=0),
    dict(test_version=2),
    dict(test_target_valid=False),
    dict(test_target_ns=1),
    dict(test_target_ns=99_000_000_000),
    dict(test_target=float("nan")),
  ]:
    s = ProtocolScene()
    s.warm()
    for key, value in changes.items():
      setattr(s.cs, key, value)
    out, _ = s.step(0.3, math.asin((-0.13 - 0.3 - GRADE_OFFSET) / 9.81), torque=50)
    assert out.navigatorLongitudinal.brake and not out.navigatorLongitudinal.fault
    assert out.navigatorLongitudinal.allocatorPhase != "precharge"


def test_old_request_keeps_original_control_even_without_new_fields():
  s = contracts.Scene()
  s.warm()
  out, _ = s.step(0.3, math.asin((-0.13 - 0.3 - GRADE_OFFSET) / 9.81), torque=50)
  assert out.navigatorLongitudinal.brake
  assert out.navigatorLongitudinal.allocatorVersion == 1  # capability, not implicit activation
