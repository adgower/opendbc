from dataclasses import replace
import pytest
from opendbc.car.ford.navigator_a3 import Inputs, PROFILES
from opendbc.car.ford.navigator_a3_scheduler import ShadowScheduler


def sample(t, **kw):
  return replace(Inputs(t, t, .002, 20., True, False, True, 0., t, True), **kw)


@pytest.mark.parametrize('profile', PROFILES.values())
@pytest.mark.parametrize('sign', [-1, 1])
def test_49_51_no_advance_and_no_counter(profile, sign):
  s = ShadowScheduler()
  a = s.step(profile, sample(1_000_000_000, curvature_inv_m=sign*.002))
  history = s.state
  early = s.step(profile, sample(1_049_000_000, curvature_inv_m=sign*.002))
  assert early.output is None and early.proposal_counter is None and early.waiting
  assert s.state == history
  later = s.step(profile, sample(1_100_000_000, curvature_inv_m=sign*.002))
  assert later.output.mode == 1 and later.proposal_counter == 1
  assert a.proposal_counter == 0


def test_pause_is_immediate_once_and_release_waits():
  s = ShadowScheduler(); p = next(iter(PROFILES.values()))
  s.step(p, sample(1_000_000_000))
  pause = s.step(p, sample(1_010_000_000, driver_pressed=True))
  assert pause.output.mode == 0 and pause.reset
  history = s.state
  assert s.step(p, sample(1_020_000_000, driver_pressed=True)).output is None
  assert s.state == history
  assert s.step(p, sample(1_050_000_000)).output is None
  assert s.step(p, sample(1_060_000_000)).output.mode == 1


def test_clock_gap_and_no_catchup():
  s = ShadowScheduler(); p = next(iter(PROFILES.values()))
  s.step(p, sample(1_000_000_000)); state = s.state
  for t in (1_000_000_000, 999_999_999):
    assert s.step(p, sample(t)).output is None
    assert s.state == state
  gap = s.step(p, sample(1_100_000_001))
  assert gap.reset and gap.output.mode == 0 and gap.reason == 'timing_gap'
  assert s.step(p, sample(1_110_000_001)).output is None
  assert s.step(p, sample(1_150_000_001)).output.mode == 1

@pytest.mark.parametrize('bad', [dict(valid=False), dict(measurement_valid=False), dict(source_ns=0),
                               dict(measurement_ns=0), dict(active=False)])
def test_invalid_neutral_between_due_times(bad):
  s = ShadowScheduler(); p = next(iter(PROFILES.values()))
  s.step(p, sample(1_000_000_000))
  neutral = s.step(p, sample(1_010_000_000, **bad))
  assert neutral.output.mode == 0 and neutral.reset
  assert s.step(p, sample(1_020_000_000, **bad)).output is None
  assert s.step(p, sample(1_059_999_999)).output is None
  assert s.step(p, sample(1_060_000_000)).output.mode == 1


def test_100hz_jitter_emits_latest_input_and_only_one_step():
  s = ShadowScheduler(); p = next(iter(PROFILES.values()))
  emitted = []
  for i, offset in enumerate((0, 10, 20, 30, 40, 49, 59, 70, 80, 90, 100, 109, 120, 130, 140, 150, 160)):
    d = s.step(p, sample(1_000_000_000 + offset*1_000_000, curvature_inv_m=.001 if i < 6 else -.002))
    if d.output is not None:
      emitted.append((offset, d))
  assert [t for t, _ in emitted] == [0, 59, 109, 160]
  assert [d.proposal_counter for _, d in emitted] == [0, 1, 2, 3]
  assert emitted[1][1].output.raw_path_angle_rad < 0
  assert all(not d.output.transmission_allowed for _, d in emitted)
