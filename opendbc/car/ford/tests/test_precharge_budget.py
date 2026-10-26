from opendbc.car.ford.longitudinal_precharge import BoundedPrechargeAllocator as GuardedAllocator


def step(a, i, net=-0.02, nominal=False, raw=0.3, target=0.1, owner=True, active=True, dt=None, measured_clear=True):
  t = 10 + i * 0.02 if dt is None else dt
  return a.update(t, owner, active, True, nominal, raw, net, target, measured_clear=measured_clear)


def test_budget_counts_before_nominal_admission_and_returns_brake_at_deadline():
  a = GuardedAllocator()
  r = []
  for i in range(30):
    r.append(step(a, i, nominal=i >= 16))
  assert r[16].brake and a.locked


def test_precharge_cannot_restart_or_extend_negative_exposure_budget():
  a = GuardedAllocator()
  step(a, 0, net=0.1, nominal=False)
  r = [step(a, i, net=-0.13, nominal=True) for i in range(1, 21)]
  assert any(x.precharge and not x.brake for x in r)
  assert all(x.brake for x in r[5:])


def test_micro_crossings_do_not_rearm_and_positive_recovery_does():
  a = GuardedAllocator()
  for i in range(60):
    step(a, i, net=0.001 if i % 2 else -0.04, nominal=True)
  assert a.locked
  for i in range(60, 95):
    step(a, i, net=0.1, nominal=False)
  assert not a.locked


def test_bad_timing_strong_request_and_owner_off():
  a = GuardedAllocator()
  step(a, 0, net=-0.13, nominal=True)
  assert step(a, 1, net=-0.13, nominal=True, dt=10.06).brake
  a = GuardedAllocator()
  assert step(a, 0, net=-1, nominal=True, raw=-1, target=-1).brake
  step(a, 1, owner=False, active=False)
  assert a.start is None and not a.locked


def test_pedal_override_does_not_rearm_an_exhausted_owner_budget():
  a = GuardedAllocator()
  for i in range(20):
    step(a, i, net=-0.13, nominal=True)
  step(a, 20, active=False)
  assert a.locked


def test_committed_actual_brake_cannot_revert_to_precharge_in_same_reference_episode():
  a = GuardedAllocator()
  assert step(a, 0, net=-0.07, nominal=True).brake  # Cold/unqualified entry uses actual brake.
  r = step(a, 1, net=-0.13, nominal=True)
  assert r.brake, "Precharge is entry-only, never a downgrade of committed actual braking"


def test_nominal_weak_brake_is_precharged_even_above_bp_precharge_threshold():
  a = GuardedAllocator()
  step(a, 0, net=0.1, nominal=False)
  r = step(a, 1, net=-0.07, nominal=True)
  assert r.precharge and not r.brake
  for i in range(2, 20):
    r = step(a, i, net=-0.07, nominal=True)
  assert r.brake and a.locked


def test_rearming_requires_measured_clear_for_entire_recovery_dwell():
  a = GuardedAllocator()
  step(a, 0, net=0.1)
  for i in range(1, 20):
    step(a, i, net=-0.13, nominal=True)
  assert a.locked
  for i in range(20, 65):
    step(a, i, net=0.1, measured_clear=False)
  assert a.locked
  for i in range(65, 100):
    step(a, i, net=0.1, measured_clear=True)
  assert not a.locked


def test_override_interrupts_the_clean_recovery_dwell():
  a = GuardedAllocator()
  step(a, 0, net=0.1)
  for i in range(1, 20):
    step(a, i, net=-0.13, nominal=True)
  for i in range(20, 40):
    step(a, i, net=0.1)
  for i in range(40, 50):
    step(a, i, net=0.1, active=False)
  for i in range(50, 60):
    step(a, i, net=0.1)
  assert a.locked
  for i in range(60, 85):
    step(a, i, net=0.1)
  assert not a.locked


def test_invalid_context_interrupts_the_clean_recovery_dwell():
  a = GuardedAllocator()
  step(a, 0, net=0.1)
  for i in range(1, 20):
    step(a, i, net=-0.13, nominal=True)
  for i in range(20, 40):
    step(a, i, net=0.1)
  a.update(10.8, True, True, False, False, 0.3, 0.1, 0.1, measured_clear=True)
  for i in range(41, 50):
    step(a, i, net=0.1)
  assert a.locked


def test_latched_nominal_demand_with_positive_net_still_has_absolute_deadline():
  a = GuardedAllocator()
  step(a, 0, net=0.1, nominal=False)
  r = [step(a, i, net=0.001, nominal=True) for i in range(1, 30)]
  assert any(x.precharge and not x.brake for x in r)
  assert all(x.brake for x in r[16:]) and a.locked


def test_randomized_precharge_always_has_live_finite_budget_and_nominal_demand():
  import random

  rng = random.Random(17)
  a = GuardedAllocator()
  now = 10.0
  for _ in range(5000):
    now += rng.choice([0.01, 0.02, 0.04, 0.06])
    owner = rng.random() > 0.03
    active = rng.random() > 0.1
    eligible = rng.random() > 0.05
    nominal = rng.random() > 0.4
    raw = rng.uniform(-0.8, 0.6)
    net = rng.uniform(-0.4, 0.3)
    target = rng.uniform(-1, 0.5)
    r = a.update(now, owner, active, eligible, nominal, raw, net, target, measured_clear=rng.random() > 0.3)
    if r.precharge and not r.brake:
      assert owner and active and eligible and nominal and not a.locked
      assert a.start is not None and 0 <= now - a.start < 0.300000001 and a.area < 0.01
