from types import SimpleNamespace as NS
from opendbc.car.ford.longitudinal_coordination import freeze_integral, _freeze_integral_base as BASE


class PhaseIntegral:
  def __call__(self, *args, **kwargs):
    return freeze_integral(*args, **kwargs)


def f(timestamp, **kw):
  d = dict(
    t=timestamp,
    epoch=7,
    sequence=1,
    valid=True,
    fault=False,
    brake=False,
    precharge=True,
    limited=True,
    requested=0.2,
    gas=-5.0,
    allocator_version=1,
    allocator_phase="precharge",
    reference_brake=True,
    precharge_age=0.04,
    budget_remaining=0.004,
  )
  d.update(kw)
  return NS(**d)


def call(a, t, feedback, error=0.1, target=0.1):
  return a(t, 7, 3, error, target, 0.2, 25.0, "pid", feedback, enabled=True)


def test_bounded_precharge_matches_signed_nominal_brake_integral_role():
  a = PhaseIntegral()
  fb = f(10.0)
  assert BASE(10.0, 7, 3, 0.1, 0.1, 0.2, 25.0, "pid", fb, enabled=True)
  assert not call(a, 10.0, fb) and not call(a, 10.0, fb, error=-0.1)
  nominal = f(10.0, brake=True, precharge=True, limited=False, allocator_phase="brake")
  assert call(a, 10.0, fb) == call(a, 10.0, nominal)


def test_release_and_engine_ramp_immediately_restore_existing_antiwindup():
  a = PhaseIntegral()
  for phase in ["release", "engineRamp"]:
    fb = f(10.0, precharge=False, allocator_phase=phase, precharge_age=None, budget_remaining=None)
    assert call(a, 10.0, fb) and not call(a, 10.0, fb, error=-0.1)
  assert not call(a, 10.0, f(10.0, precharge=False, allocator_phase="drive", limited=False, gas=0.2))


def test_expired_unowned_or_unbudgeted_precharge_does_not_bypass_antiwindup():
  for change in [
    dict(precharge_age=0.31),
    dict(precharge_age=-0.01),
    dict(budget_remaining=0),
    dict(reference_brake=False),
    dict(allocator_phase="drive"),
    dict(precharge_age=None),
  ]:
    assert call(PhaseIntegral(), 10.0, f(10.0, **change))


def test_stale_foreign_and_future_feedback_keep_original_policy():
  for change in [dict(t=9.0), dict(t=11.0), dict(epoch=8), dict(sequence=5), dict(valid=False), dict(fault=True)]:
    fb = f(10.0, **change)
    a = PhaseIntegral()
    assert call(a, 10.0, fb) == BASE(10.0, 7, 3, 0.1, 0.1, 0.2, 25.0, "pid", fb, enabled=True)
