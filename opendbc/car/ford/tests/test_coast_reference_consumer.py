import pytest
from opendbc.car.ford.tests.test_history_release import controller
from opendbc.car.ford.values import CAR


def test_off_creates_no_reference_helper():
  c = controller()
  c.configure_coast_reference()
  assert not c.coast_reference_enabled
  assert not hasattr(c, "coast_reference")


def test_startup_only_and_supported_profile():
  for car, alpha, coordinated in [(CAR.FORD_F_150_MK14, True, True), (CAR.LINCOLN_NAVIGATOR_MK4, False, True), (CAR.LINCOLN_NAVIGATOR_MK4, True, False)]:
    c = controller(car, alpha, coordinated)
    with pytest.raises(ValueError):
      c.configure_coast_reference(enabled=True)
  c = controller(enabled=True)
  c.configure_coast_reference(enabled=True)
  assert c.coast_reference.reference.history_release_enabled
  c.frame = 1
  with pytest.raises(RuntimeError, match="startup-only"):
    c.configure_coast_reference(enabled=True)


def test_retention_combination_rejected_in_both_orders():
  c = controller()
  c.configure_retention(enabled=True)
  with pytest.raises(ValueError, match="retention"):
    c.configure_coast_reference(enabled=True)
  c = controller()
  c.configure_coast_reference(enabled=True)
  with pytest.raises(ValueError, match="retention"):
    c.configure_retention(enabled=True)


from collections import defaultdict
from types import SimpleNamespace as NS
from opendbc.car import structs
from opendbc.car.ford.longitudinal_coordination import Torque
from opendbc.car.ford.longitudinal_reference_control import TargetTrend


def context(i=0, version=4):
  now = 10.0 + i * 0.01
  ns = round(now * 1e9)
  cc = structs.CarControl.new_message(enabled=True, longActive=True, latActive=False, orientationNED=[0.0, 0.0, 0.0])
  cc.actuators.accel = 0.2
  cc.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
  q = cc.navigatorLongitudinal
  q.enabled, q.epoch, q.sequence = True, 7, i + 1
  q.producerMonoTime = q.poseMonoTime = q.targetMonoTime = ns
  q.poseValid = q.referenceValid = q.targetValid = True
  q.targetAcceleration, q.allocatorVersion = 0.1, version
  state = structs.CarState.new_message(vEgo=5.0, vEgoRaw=5.0, aEgo=0.0, canValid=True, cruiseState={"enabled": True, "standstill": False})
  cs = NS(
    out=state,
    buttons_stock_values=defaultdict(int),
    lkas_status_stock_values=defaultdict(int),
    acc_tja_status_stock_values=defaultdict(int),
    longitudinal_brake_torque=Torque(now, 0.0),
  )
  return cc, cs, ns


def update(c, i=0, version=4):
  cc, cs, ns = context(i, version)
  c.coordination_control_valid = c.coordination_state_valid = True
  c.coordination_state_ns = c.coordination_speed_ns = ns
  return c.update(cc.as_reader(), cs, ns)


def test_reference_only_updates_at_acc_cadence_and_keeps_nonacc_feedback():
  c = controller()
  c.configure_coast_reference(enabled=True)
  first, _ = update(c)
  assert c.coast_reference.callbacks == 1
  old = c.last_feedback
  odd, _ = update(c, 1)
  assert c.coast_reference.callbacks == 1
  assert c.last_feedback is old
  assert odd.navigatorLongitudinal.producerMonoTime == first.navigatorLongitudinal.producerMonoTime
  update(c, 2)
  assert c.coast_reference.callbacks == 2


@pytest.mark.parametrize("version", [3, 5])
def test_mixed_and_unknown_request_decline_new_borrowing(version):
  c = controller()
  c.configure_coast_reference(enabled=True)
  for i in range(0, 10, 2):
    out, _ = update(c, i, version)
    assert not c.coast_reference.gate.scope
    assert not c.coast_reference.gate.selected
    assert not out.navigatorLongitudinal.reason.startswith("bounded-release-reference-v2:")


def test_off_request4_has_no_reference_overlay():
  c = controller()
  update(c)
  assert not hasattr(c, "coast_reference")
  assert type(c.release_gate).__name__ == "ReleaseGate"


def test_exit_reference_failure_uses_actual_coordination_fault():
  c = controller()
  c.configure_coast_reference(enabled=True)
  r = c.coast_reference
  r.handoff.recovering = True
  r.handoff.recovery_started, r.handoff.release_until, r.handoff.epoch = 10.0, 10.3, 7
  r.reference.update = lambda *a: NS(handoff=None, feedback=None)
  output, _ = update(c, 2)
  f = output.navigatorLongitudinal
  assert c.coordination_fault == "exit_reference_invalid"
  assert f.fault and not f.valid and f.allocatorPhase == "fault"
  assert f.gas == -5 and not f.brake and not f.precharge
  assert output.accel == 0


def test_target_trend_fresh_monotonic_and_conflicts():
  t = TargetTrend()
  assert not t.update(1, 1000000000, 0.1, 1000000000)
  assert t.update(1, 1020000000, 0.1, 1020000000)
  assert not t.update(1, 1020000000, 0.2, 1020000000)
  assert not t.update(1, 1020000000, 0.1, 1020000000)
  assert t.update(1, 1040000000, 0.2, 1040000000)
  assert not t.update(1, 1040000000, 0.2, 1200000000)
  assert not t.update(2, 1200000000, 0.2, 1200000000)


def test_canonical_helper_and_policy_ast_parity():
  # Intentionally optimized fingerprints; numerical/guard parity is checked separately.
  import ast
  import hashlib
  from pathlib import Path
  import opendbc.car.ford.longitudinal_nominal_reference as nominal
  import opendbc.car.ford.longitudinal_reference_control as selected

  expected = {
    "NominalResult": "16197692f35698c67a3f3e6ce4be8191790334a79d043f7995770793793c3128",
    "apply_creep_compensation": "1679a0fb4ffabfd6cd71a1809831fd7089ea5c1e767df2f9816501e2962e317e",
    "NominalReference": "05fc951330a93b9e01621430c46508183c30601dccb9bdf051dcc2cb068c612d",
    "BoundedResponseEntry": "5160dd3ba1f08e59808c29db4ae283f6d5eefe3a0c848447082a15d16b816c5e",
    "QualifiedRearmEntry": "3a749e7b4aaf80d92043a796c3d82cf25e461adbf49f73a758d6125153c373be",
    "PendingRearmEntry": "988adbb6f66d2cd39d75d0f287e0890f83ac1446e728e04d049ff35638656619",
    "NominalOwnerGate": "5ad0f3dbab11fe1cf571f6d5581a90fa5d19bf36c0808b3fd79f7be84d19da64",
    "OwnerAllocator": "fb21718e777168bc8c422b539e0b162e2c42234f336e2a52ab622657cd06f03d",
    "OwnerHandoff": "13a6db64adbb063eeff187b52dbd34ccf4709954f142e09fced37b50600f4f69",
    "TargetTrend": "2b57c098b1efd08c93b1e32d2c9036921105597db570c6735df516f3de87276d",
    "ExitHandoff": "cc6915a227058f97a3dc3e34d7c081419815096b2b69c19f9ba971fb5c886fff",
  }
  found = {}
  for module in (nominal, selected):
    for node in ast.parse(Path(module.__file__).read_text()).body:
      if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in expected:
        found[node.name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
  assert found == expected


def test_receipt_binds_wire_gas_preserves_raw_nominal_and_actual_facts():
  import struct
  from dataclasses import replace
  from opendbc.car.ford.longitudinal_coordination import Feedback
  from opendbc.car.ford.longitudinal_reference_feedback import ReferenceReceipt, MODEL_ID

  c = controller()
  c.configure_coast_reference(enabled=True)
  r = c.coast_reference
  cc, _, ns = context(2)
  raw = float(cc.actuators.accel)
  f = Feedback(10.02, 7, 3, raw, 0.123456789, False, False, allocator_version=3, allocator_phase="drive")
  nominal = replace(f, gas=0.12345678123456)
  r.result = NS(feedback=nominal)
  r.actual_accel = 0.2
  c.last_grade = NS(offset=0.0, gravity=0.0)
  g = r.gate
  g.scope, g.deferred, g.arm, g.area, g.allocator_area = True, True, 10.0, 0.001, 0.001
  r.allocator.area = 0.001
  selected = r.finish(c, cc, f, ns)
  receipt = ReferenceReceipt.parse(selected.reason)
  assert receipt is not None and receipt.model_id == MODEL_ID
  assert receipt.actual_gas == struct.unpack("f", struct.pack("f", f.gas))[0]
  assert receipt.nominal["gas"] == nominal.gas
  assert replace(selected, reason=f.reason) == f
  old_budget = receipt.budget_remaining
  g.area = 0.0001
  renewed = ReferenceReceipt.parse(r.finish(c, cc, f, ns).reason)
  assert renewed.budget_remaining == old_budget
  expired = r.finish(c, cc, f, 10300000000)
  assert r.episode is None and expired.reason == f.reason
