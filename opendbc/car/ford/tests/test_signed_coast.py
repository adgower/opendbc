from dataclasses import replace
from opendbc.car.ford.longitudinal_coordination import Feedback, freeze_integral
from types import SimpleNamespace as NS
from opendbc.car.ford.longitudinal_coordination import allocator_feedback_valid
from opendbc.car.ford.longitudinal_precharge import allocator_request_valid

repair = NS(valid_feedback=allocator_feedback_valid, freeze=freeze_integral, valid_request=allocator_request_valid)


def coast(**kwargs):
  d = dict(
    t=1.0,
    epoch=1,
    sequence=1,
    requested=0.105,
    gas=-5.0,
    brake=False,
    limited=True,
    valid=True,
    fault=False,
    allocator_version=3,
    precharge=False,
    reference_brake=True,
    allocator_phase="release",
    precharge_age=0.04,
    budget_remaining=0.009,
  )
  return Feedback(**(d | kwargs))


def test_revision2_coast_is_explicit_and_bounded():
  assert repair.valid_feedback(coast())
  for update in [
    dict(precharge_age=0.31),
    dict(budget_remaining=0.0),
    dict(gas=0.1),
    dict(limited=False),
    dict(allocator_version=1),
    dict(precharge_age=float("nan")),
  ]:
    assert not repair.valid_feedback(coast(**update))


def test_signed_coast_preserves_nominal_braking_integral_role():
  for error in [-0.2, 0.2]:
    assert not repair.freeze(1.01, 1, 2, error, 0.1, 0.105, 20.0, "pid", coast(), enabled=True)


def test_ordinary_ramp_keeps_installed_numerical_margin():
  f = coast(reference_brake=False, gas=0.1, allocator_phase="engineRamp", precharge_age=0.0, budget_remaining=0.0)
  assert (
    repair.freeze(1.01, 1, 2, 0.2, 0.1, 0.105, 20.0, "pid", f, enabled=True)
    == freeze_integral(1.01, 1, 2, 0.2, 0.1, 0.105, 20.0, "pid", replace(f, allocator_version=1), enabled=True)
    is False
  )


def test_wrong_revision_cannot_claim_signed_coast_role():
  assert not repair.valid_feedback(coast(allocator_version=1))
  # Existing policy may freeze positive growth for malformed/unknown contexts;
  # no new signed-braking exemption may be granted by an unknown peer.
  assert repair.freeze(1.01, 1, 2, 0.2, 0.1, 0.105, 20.0, "pid", coast(allocator_version=1), enabled=True)


def test_unknown_peer_cannot_request_trial():
  from types import SimpleNamespace as NS

  r = NS(allocatorVersion=1, targetValid=True, targetAcceleration=0.1, targetMonoTime=1_000_000_000, producerMonoTime=1_010_000_000)
  assert not repair.valid_request(r, 1_020_000_000)
  r.allocatorVersion = 3
  assert repair.valid_request(r, 1_020_000_000)


def test_controller_emits_truthful_bounded_coast_then_commits_budget_brake():
  import math
  from opendbc.car.ford.tests.test_precharge_clearance import ProtocolScene
  from opendbc.car.ford.longitudinal_coordination import GRADE_OFFSET, feedback_from_wire
  scene = ProtocolScene()
  scene.warm()
  observed = []
  for _ in range(10):
    output, _ = scene.step(.3, math.asin((-.13 - .3 - GRADE_OFFSET) / 9.81), torque=0)
    feedback = feedback_from_wire(output.navigatorLongitudinal)
    if scene.c.last_allocation.phase == 'bounded_true_coast':
      observed.append(feedback)
      assert allocator_feedback_valid(feedback)
      assert not feedback.brake and not feedback.precharge and feedback.reference_brake
      assert feedback.allocator_phase == 'release' and feedback.gas == -5.
      assert scene.c.release_barrier.needs and scene.c.release_barrier.started is None
  assert observed
  assert scene.c.last_allocation.brake and scene.c.precharge_allocator.locked
