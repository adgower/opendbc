from types import SimpleNamespace as NS
from dataclasses import replace
import math
import pytest
from opendbc.car.ford.longitudinal_recovery import RollingGuard
from opendbc.car.ford.longitudinal_coordination import Feedback


def step(
  g,
  t,
  speed=1.5,
  *,
  active=True,
  stopped=False,
  target=0.5,
  error=0.3,
  wanted=0.9,
  fbkw=None,
  cskw=None,
  epoch=1,
  sequence=100,
  should_stop=False,
  growth=0.001,
):
  cs = NS(vEgo=speed, canValid=True, gasPressed=False, brakePressed=False, standstill=stopped, cruiseState=NS(standstill=False))
  for k, v in (cskw or {}).items():
    setattr(cs, k, v)
  fb = Feedback(t - 0.005, 1, 1, 0.9, 0.2, False, True, True, False, allocator_version=1, precharge=False, reference_brake=False, allocator_phase="engineRamp")
  fb = replace(fb, **(fbkw or {}))
  g.observe(active, cs, target, should_stop, fb, t, epoch, sequence, True)
  g.growth_cost = growth
  return g.freeze(t, epoch, sequence, error, target, wanted, speed, "pid", fb, enabled=True)


def prime(g):
  step(g, 1.0, 5.0, fbkw=dict(allocator_phase="drive", limited=False, gas=0.8))
  step(g, 1.01, 2.0, fbkw=dict(allocator_phase="brake", brake=True, precharge=True, reference_brake=True, gas=-5.0))


def test_qualified_rolling_restart_holds_growth_that_installed_code_allows():
  g = RollingGuard()
  prime(g)
  assert step(g, 1.02)


def test_full_stop_launch_is_not_changed():
  g = RollingGuard()
  prime(g)
  assert not step(g, 1.02, 0.1, stopped=True)
  assert not step(g, 1.03, 1.0)


def test_engagement_at_crawl_without_moving_history_is_not_qualified():
  assert not step(RollingGuard(), 1.0)


@pytest.mark.parametrize(
  "kw",
  [
    dict(error=-0.2),
    dict(target=-0.1),
    dict(wanted=0.1),
    dict(fbkw=dict(gas=-5.0)),
    dict(fbkw=dict(gas=0.9)),
    dict(fbkw=dict(limited=False)),
    dict(fbkw=dict(brake=True)),
    dict(fbkw=dict(precharge=True)),
    dict(fbkw=dict(reference_brake=True)),
    dict(fbkw=dict(fault=True)),
    dict(fbkw=dict(valid=False)),
    dict(fbkw=dict(epoch=2)),
    dict(fbkw=dict(sequence=101)),
    dict(fbkw=dict(t=0.8)),
    dict(fbkw=dict(allocator_version=0)),
    dict(fbkw=dict(allocator_phase="drive")),
    dict(active=False),
    dict(cskw=dict(gasPressed=True)),
    dict(cskw=dict(brakePressed=True)),
    dict(cskw=dict(canValid=False)),
    dict(error=math.nan),
  ],
)
def test_signed_recovery_invalid_context_and_braking_stay_on_reference(kw):
  g = RollingGuard()
  prime(g)
  assert not step(g, 1.02, **kw)


def test_moving_installed_antiwindup_is_preserved():
  g = RollingGuard()
  assert step(g, 1.0, 3.2)


def test_expired_approach_does_not_start_hold():
  g = RollingGuard()
  prime(g)
  for j in range(1, 321):
    step(g, 1.01 + j * 0.01, 2.0, fbkw=dict(allocator_phase="release", gas=-5.0))
  assert not step(g, 4.23)


def test_bounded_episode_cannot_rearm_by_speed_boundary_wiggle():
  g = RollingGuard()
  prime(g)
  assert step(g, 1.02)
  for j in range(1, 311):
    step(g, 1.02 + j * 0.01)
  assert not step(g, 4.14)
  assert step(g, 4.15, 3.1)  # reference moving guard, not rolling rearm
  assert not step(g, 4.16, 1.5)


def test_new_actual_brake_after_normal_motion_can_qualify_again():
  g = RollingGuard()
  prime(g)
  assert step(g, 1.02)
  step(g, 1.03, 5.0, fbkw=dict(allocator_phase="drive", limited=False, gas=0.8))
  step(g, 1.04, 2.0, fbkw=dict(allocator_phase="brake", brake=True, precharge=True, reference_brake=True, gas=-5.0))
  assert step(g, 1.05)


def test_owner_disable_discards_rolling_history():
  g = RollingGuard()
  prime(g)
  step(g, 1.02, active=False)
  assert not step(g, 1.03)


def test_current_sequence_rollback_discards_rolling_history():
  g = RollingGuard()
  prime(g)
  assert not step(g, 1.02, sequence=99)


@pytest.mark.parametrize("t", [0.99, 1.2, math.nan])
def test_clock_discontinuity_discards_qualification(t):
  g = RollingGuard()
  prime(g)
  assert not step(g, t)


def test_epoch_change_is_not_a_low_speed_rearm():
  g = RollingGuard()
  prime(g)
  assert not step(g, 1.02, epoch=2, fbkw=dict(epoch=2))


def test_should_stop_cancels_hold_without_repayment():
  g = RollingGuard()
  prime(g)
  assert step(g, 1.02)
  assert not step(g, 1.03, should_stop=True)
  assert not step(g, 1.04)


def test_bad_feedback_cannot_restore_a_started_hold():
  g = RollingGuard()
  prime(g)
  assert step(g, 1.02)
  assert not step(g, 1.03, fbkw=dict(valid=False))
  assert not step(g, 1.04)


def test_deadline_exact_boundary():
  g = RollingGuard()
  prime(g)
  assert step(g, 1.02, growth=0.00001)
  for j in range(1, 300):
    assert step(g, 1.02 + j * 0.01, growth=0.00001)
  assert not step(g, 4.02)


def test_withheld_integral_budget_does_not_freeze_an_extra_step():
  g = RollingGuard()
  prime(g)
  assert step(g, 1.02, growth=0.1)
  assert step(g, 1.03, growth=0.1)
  assert not step(g, 1.04, growth=0.1)
  assert not step(g, 1.05, growth=0.001)


@pytest.mark.parametrize("growth", [0.0, -0.01, math.nan, math.inf])
def test_no_positive_or_unknown_native_growth_is_not_held(growth):
  g = RollingGuard()
  prime(g)
  assert not step(g, 1.02, growth=growth)
