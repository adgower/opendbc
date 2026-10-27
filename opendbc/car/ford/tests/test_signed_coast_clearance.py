from types import SimpleNamespace as NS
from opendbc.car.ford.longitudinal_coordination import Torque
from opendbc.car.ford.longitudinal_restart import CoastClearance


def call(b, t, torque_t=None, value=0.0):
  return b.update(
    t,
    True,
    False,
    20.0,
    "pid",
    Torque(t if torque_t is None else torque_t, value),
    owner_active=True,
    coast_lease=getattr(b, "coast_context", lambda t: None)(t),
  )


def ready():
  b = CoastClearance(True)
  for t in [1.0, 1.02, 1.04]:
    r = call(b, t)
  assert r.allow and not b.needs
  b.coast_context = lambda now: NS(enabled=True, epoch=1, sequence=round(now * 100), age=now - 1.06, remaining=0.009)
  return b


def test_bounded_coast_arms_without_granting_clearance_or_starting_deadline():
  b = ready()
  for t in [1.06, 1.08, 1.10, 1.12]:
    r = call(b, t)
    assert not r.allow and not r.fault and b.needs and b.started is None
    assert b.clear == [] and b.last_sample is None


def test_exit_needs_three_new_distinct_post_coast_samples():
  b = ready()
  call(b, 1.06)
  call(b, 1.08)
  b.coast_context = lambda now: None
  assert not call(b, 1.10).allow
  assert not call(b, 1.12, torque_t=1.10).allow
  assert not call(b, 1.14).allow
  assert call(b, 1.16).allow


def test_pending_ordinary_release_cannot_be_reset_as_coast():
  b = CoastClearance(True)
  call(b, 1.0, value=100.0)
  started = b.started
  b.coast_context = lambda now: NS(enabled=True, epoch=1, sequence=1, age=0.02, remaining=0.009)
  r = call(b, 1.02)
  assert r.fault and not r.allow and b.started == started


def test_reverse_and_stale_torque_cannot_waive_barrier():
  for sample in [1.03, 0.8]:
    b = ready()
    call(b, 1.06)
    r = call(b, 1.08, torque_t=sample)
    assert r.fault and not r.allow


def test_owner_change_and_clock_gap_cannot_restart_coast_wait():
  b = ready()
  call(b, 1.06)
  b.coast_context = lambda now: NS(enabled=True, epoch=2, sequence=110, age=0.04, remaining=0.009)
  assert call(b, 1.08).fault
  b = ready()
  call(b, 1.06)
  assert call(b, 1.30).fault


def test_exhausted_lease_and_nonclear_torque_do_not_continue():
  b = ready()
  call(b, 1.06)
  b.coast_context = lambda now: NS(enabled=True, epoch=1, sequence=110, age=0.04, remaining=0.0)
  assert call(b, 1.08).fault
  b = ready()
  call(b, 1.06)
  assert call(b, 1.08, value=60.0).fault
