from types import SimpleNamespace as NS
import pytest
from opendbc.car.ford.longitudinal_coordination import freeze_integral as policy, _freeze_integral_base as original


def feedback(**change):
  d = dict(
    t=10.0,
    epoch=7,
    sequence=10,
    requested=1.74,
    gas=0.64,
    brake=False,
    limited=True,
    valid=True,
    fault=False,
    allocator_version=1,
    allocator_phase="engineRamp",
    precharge=False,
    reference_brake=False,
    precharge_age=0.0,
    budget_remaining=0.0,
  )
  d.update(change)
  return NS(**d)


def call(f=None, **change):
  d = dict(now=10.0, epoch=7, sequence=11, error=0.65, target=1.17, wanted=1.74, speed=1.65, phase="pid", feedback=f or feedback(), enabled=True)
  d.update(change)
  return policy(**d)


def test_unqualified_low_speed_ramp_does_not_broadly_freeze_integral():
  assert not call()


@pytest.mark.parametrize(
  "change",
  [
    dict(error=-0.1),
    dict(error=0),
    dict(target=-0.1),
    dict(enabled=False),
    dict(phase="starting"),
    dict(phase="stopping"),
    dict(speed=0),
    dict(speed=float("nan")),
    dict(epoch=8),
    dict(sequence=9),
    dict(now=10.2),
    dict(now=9.99),
  ],
)
def test_only_positive_unattainable_pid_demand_in_current_owner_is_held(change):
  assert not call(**change)


@pytest.mark.parametrize(
  "change",
  [
    dict(allocator_version=0),
    dict(allocator_version=2),
    dict(allocator_phase="release"),
    dict(allocator_phase="drive"),
    dict(allocator_phase="precharge", precharge=True, reference_brake=True, gas=-5, precharge_age=0.1, budget_remaining=0.005),
    dict(reference_brake=True),
    dict(brake=True),
    dict(precharge=True),
    dict(limited=False),
    dict(valid=False),
    dict(fault=True),
    dict(gas=-5),
    dict(gas=1.74),
  ],
)
def test_uncertain_metadata_braking_and_nonramp_keep_original_low_speed_policy(change):
  assert not call(feedback(**change))


@pytest.mark.parametrize("speed", [3.0, 4.0, 15.0, 30.0, 38.0])
@pytest.mark.parametrize("change", [{}, dict(error=-0.1), dict(target=-0.5)])
def test_moving_policy_is_exactly_unchanged(speed, change):
  d = dict(now=10.0, epoch=7, sequence=11, error=0.65, target=1.17, wanted=1.74, speed=speed, phase="pid", feedback=feedback(), enabled=True)
  d.update(change)
  assert policy(**d) == original(**d)
