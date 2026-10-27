from opendbc.car.ford.longitudinal_precharge import BoundedPrechargeAllocator as CoordinatedAllocator, PRECHARGE_PROTOCOL_VERSION
from opendbc.car.ford.longitudinal_coordination import Feedback, freeze_integral, allocator_feedback_valid


def update(a, t, nominal=False, net=-0.03, raw=0.1, target=0.1, **kw):
  args = dict(owner=True, active=True, eligible=True, measured_clear=True)
  args.update(kw)
  return a.update(t, args.pop("owner"), args.pop("active"), args.pop("eligible"), nominal, raw, net, target, **args)


def pending_after_coast():
  a = CoordinatedAllocator()
  update(a, 1.0, net=0.1)
  for i in range(1, 30):
    update(a, 1.0 + 0.02 * i)
  return a


def test_continuous_negative_coast_is_charged_to_existing_allowance():
  a = pending_after_coast()
  assert a.start is not None and a.locked and a.area > 0
  assert update(a, 1.60, nominal=True).brake


def test_repaid_budget_and_fresh_clear_rearm_without_extra_integral_debt():
  a = CoordinatedAllocator()
  update(a, 1.0, net=0.1)
  for i in range(1, 10):
    update(a, 1.0 + i * 0.02, nominal=True, net=-0.1)
  assert a.locked
  # Net-positive allocation repays the signed ledger before the clearance dwell begins.
  for i in range(10, 25):
    update(a, 1.0 + i * 0.02, nominal=False, net=0.2, raw=0.3, target=0.1)
  assert a.area == 0 and not a.locked
  assert not update(a, 1.50, nominal=True, net=-0.02).brake


def test_bounded_precharge_and_actual_brake_have_equal_signed_integral_role():
  common = dict(t=1.0, epoch=1, sequence=1, requested=0.1, gas=-5.0, valid=True, fault=False,
                allocator_version=PRECHARGE_PROTOCOL_VERSION, reference_brake=True)
  pre = Feedback(**common, brake=False, limited=True, precharge=True, allocator_phase="precharge", precharge_age=0.02, budget_remaining=0.009)
  brake = Feedback(**common, brake=True, limited=False, precharge=True, allocator_phase="brake")
  assert allocator_feedback_valid(pre)
  for error in [-0.2, 0.2]:
    precharge_frozen = freeze_integral(1.01, 1, 2, error, 0.1, 0.1, 25.0, "pid", pre, enabled=True)
    brake_frozen = freeze_integral(1.01, 1, 2, error, 0.1, 0.1, 25.0, "pid", brake, enabled=True)
    assert precharge_frozen == brake_frozen
    assert not precharge_frozen


def test_no_allowance_forgiveness_during_short_positive_crossing():
  a = CoordinatedAllocator()
  update(a, 1.0, net=0.1)
  for i in range(1, 10):
    update(a, 1.0 + 0.02 * i, nominal=True, net=-0.1)
  assert a.locked
  update(a, 1.20, net=0.01)
  assert a.locked and update(a, 1.22, nominal=True).brake


def test_strong_intent_commits_without_precharge_only():
  for raw, net, target in [(0.1, -0.15, 0.1), (-0.6, -0.02, 0.1), (0.1, -0.02, -0.5)]:
    a = CoordinatedAllocator()
    update(a, 1.0, net=0.1)
    assert update(a, 1.02, nominal=True, raw=raw, net=net, target=target).brake


def test_area_exhaustion_cannot_be_reset_by_nominal_off_pulses():
  a = CoordinatedAllocator()
  update(a, 1.0, net=0.1)
  for i in range(1, 20):
    update(a, 1.0 + i * 0.02, nominal=(i % 2 == 1), net=-0.06)
  assert a.locked and update(a, 1.40, nominal=True).brake


def test_gap_reversal_and_ineligible_context_cannot_create_allowance():
  for t, eligible in [(1.2, True), (0.9, True), (1.02, False)]:
    a = CoordinatedAllocator()
    update(a, 1.0, net=0.1)
    assert update(a, t, nominal=True, eligible=eligible).brake


def test_committed_braking_is_not_demoted_to_precharge():
  a = CoordinatedAllocator()
  update(a, 1.0, net=0.1)
  update(a, 1.02, nominal=True, net=-0.2)
  assert update(a, 1.04, nominal=True, net=-0.02).brake


def test_measured_clear_positive_recovery_rearms():
  a = CoordinatedAllocator()
  update(a, 1.0, net=0.1)
  for i in range(1, 10):
    update(a, 1.0 + 0.02 * i, nominal=True, net=-0.1)
  assert a.locked
  for i in range(10, 37):
    update(a, 1.0 + 0.02 * i, net=0.1, measured_clear=True)
  assert not a.locked and a.start is None and a.area == 0
  assert not update(a, 1.74, nominal=True).brake


def test_uncleared_pressure_does_not_rearm_spent_allowance():
  a = CoordinatedAllocator()
  update(a, 1.0, net=0.1)
  for i in range(1, 10):
    update(a, 1.0 + 0.02 * i, nominal=True, net=-0.1)
  for i in range(10, 50):
    update(a, 1.0 + 0.02 * i, net=0.1, measured_clear=False)
  assert a.locked and update(a, 2.0, nominal=True).brake


def test_signed_budget_feedback_is_valid_until_deadline_then_actual_brake():
  a = CoordinatedAllocator()
  update(a, 1.0, net=0.1)
  seen = False
  for i in range(1, 20):
    t = 1.0 + 0.02 * i
    d = update(a, t, nominal=True, net=-0.02)
    if not d.brake:
      seen = True
      f = Feedback(
        t,
        1,
        i,
        0.1,
        -5.0,
        False,
        True,
        True,
        False,
        allocator_version=PRECHARGE_PROTOCOL_VERSION,
        precharge=True,
        reference_brake=True,
        allocator_phase="precharge",
        precharge_age=t - a.start,
        budget_remaining=a.budget - a.area,
      )
      assert allocator_feedback_valid(f)
  assert seen and d.brake and a.locked


def test_actual_strong_braking_does_not_create_unserved_weak_brake_debt():
  a = CoordinatedAllocator()
  update(a, 1.0, net=0.1)
  for i in range(1, 30):
    d = update(a, 1.0 + i * 0.02, nominal=True, net=-1.0, raw=-1.0, target=-1.0)
    assert d.brake
  assert a.area == 0.0


def test_precharge_and_earlier_coast_are_both_charged():
  a = CoordinatedAllocator()
  update(a, 1.0, net=0.1)
  update(a, 1.02, nominal=False, net=-0.03)
  initial = a.area
  assert initial > 0
  d = update(a, 1.04, nominal=True, net=-0.03)
  assert not d.brake and a.area > initial


def test_existing_omission_debt_is_retained_when_actual_braking_begins():
  a = CoordinatedAllocator()
  update(a, 1.0, net=0.1)
  update(a, 1.02, nominal=False, net=-0.03)
  debt = a.area
  assert debt > 0
  assert update(a, 1.04, nominal=True, net=-1.0, raw=-1.0, target=-1.0).brake
  assert a.area == debt


def test_positive_qualification_cannot_rearm_before_debt_is_repaid():
  a = CoordinatedAllocator()
  update(a, 1.0, net=0.1)
  for i in range(1, 9):
    update(a, 1.0 + i * 0.02, nominal=True, net=-0.1)
  for i in range(9, 14):
    update(a, 1.0 + i * 0.02, nominal=False, net=0.06, raw=0.1, target=0.1)
  assert a.area > 0 and a.recovery is None and a.locked


def test_override_preserves_budget_and_recovery_needs_new_qualification():
  a = CoordinatedAllocator()
  update(a, 1.0, net=0.1)
  for i in range(1, 9):
    update(a, 1.0 + i * 0.02, nominal=True, net=-0.1)
  debt = a.area
  update(a, 1.18, active=False, nominal=False, net=0.2)
  assert a.area == debt and a.locked and a.recovery is None
