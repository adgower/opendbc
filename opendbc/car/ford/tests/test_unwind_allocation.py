"""Contracts for bounded unwind allocation; no vehicle or private-data inputs."""

import math
import pytest
from opendbc.car.ford.lateral_angle_unwind_allocation import UnwindAllocation
from opendbc.car.ford.tests.test_midrange_admission import make as installed
from opendbc.car.ford.tests.test_wheel_target import step
from opendbc.car.vehicle_model import VehicleModel


def make(enabled=True):
  c = UnwindAllocation(installed(), enabled=enabled)
  c.input_state_valid = True
  return c


def for_target(c, target, mph=82.0, roll=-0.018):
  vm = VehicleModel(c.CP)
  vm.update_params(1.0, 22.5)
  return -vm.calc_curvature(math.radians(target), mph * 0.44704, roll)


def test_default_off_is_exact():
  c = make(False)
  b = installed()
  for i in range(60):
    kw = dict(mph=46.0, k=0.003 * (0.99 ** max(0, i - 20)), excess=3.0, body_ratio=1.35)
    assert step(c, i, **kw) == step(b, i, **kw)


def test_low_unwind_gets_bounded_reduction():
  c = make()
  reductions = []
  for i in range(45):
    k = 0.003 * (0.985 ** max(0, i - 10))
    out = step(c, i, mph=46, k=k, excess=3.0, body_ratio=1.35)
    reductions.append(abs(c.stock_path) - abs(out.path_angle))
  assert max(reductions) > 0.0005
  assert max(reductions) <= 0.003 + 1e-12


def test_high_confirmation_survives_target_floor_crossing():
  c = make()
  values = []
  for i in range(22):
    target = 1.7 if i < 8 else 1.4
    out = step(c, i, mph=82, k=for_target(c, target), roll=-0.018, excess=1.7, body_ratio=1.9)
    values.append(abs(c.stock_path) - abs(out.path_angle))
  assert max(values[8:]) > 0.0005
  assert max(values) <= 0.002 + 1e-12


@pytest.mark.parametrize("fault", [dict(driver=True), dict(lane=True), dict(valid=False), dict(body_valid=False), dict(body_age=0.2), dict(state_valid=False)])
def test_abort_releases_without_earning(fault):
  c = make()
  for i in range(12):
    step(c, i, mph=46, k=0.003 * 0.985**i, excess=3, body_ratio=1.35)
  step(c, 12, mph=46, k=0.003 * 0.985**12, excess=3, body_ratio=1.35, **fault)
  assert c.term == 0


@pytest.mark.parametrize("mph", [15, 25, 35, 55, 60, 65, 70, 74, 86])
def test_protected_speed_ranges(mph):
  c = make()
  b = installed()
  for i in range(24):
    kw = dict(mph=mph, k=0.0009, excess=3, body_ratio=1.5)
    assert step(c, i, **kw) == step(b, i, **kw)


def test_no_unwind_no_low_change():
  c = make()
  b = installed()
  for i in range(35):
    kw = dict(mph=46, k=0.003, excess=3, body_ratio=1.35)
    assert step(c, i, **kw) == step(b, i, **kw)


def test_quiet_rearming_requires_fresh_clear_dwell():
  c = make()
  for i in range(18):
    step(c, i, mph=46, k=0.003 * 0.985**i, excess=3, body_ratio=1.35)
  assert c.area > 0
  spent = c.area
  for i in range(18, 25):
    step(c, i, mph=46, k=0.0, excess=0.0, body_ratio=1.0)
  assert c.area >= spent
  for i in range(25, 55):
    step(c, i, mph=46, k=0.0, excess=0.0, body_ratio=1.0)
  assert c.area == 0 and c.start is None


def test_no_confirmation_earned_entirely_below_target_floor():
  c = make()
  for i in range(40):
    step(c, i, mph=82, k=for_target(c, 1.4), roll=-0.018, excess=1.7, body_ratio=1.9)
  assert c.term == 0


def test_closing_error_releases_while_still_positive():
  c = make()
  for i in range(10):
    step(c, i, mph=46, k=0.003 * 0.985**i, excess=3, body_ratio=1.35)
  assert c.term > 0
  for j in range(7):
    step(c, 10 + j, mph=46, k=0.003 * 0.985 ** (10 + j), excess=3 - 0.35 * j, body_ratio=1.35)
  assert c.term == 0


def test_abort_does_not_refund_spent_area():
  c = make()
  for i in range(12):
    step(c, i, mph=46, k=0.003 * 0.985**i, excess=3, body_ratio=1.35)
  area = c.area
  step(c, 12, mph=46, k=0.003 * 0.985**12, excess=3, body_ratio=1.35, driver=True)
  assert c.area >= area and c.term == 0


def test_target_neutral_releases_confirmation():
  c = make()
  for i in range(10):
    step(c, i, mph=82, k=for_target(c, 1.7), roll=-0.018, excess=1.7, body_ratio=1.9)
  assert c.term > 0
  step(c, 10, mph=82, k=for_target(c, 0.7), roll=-0.018, excess=1.7, body_ratio=1.9)
  assert c.term == 0 and not c.rows


def test_inactive_reset_preserves_original_zero_without_active_slew_constraint():
  c = make()
  b = installed()
  for i in range(12):
    kw = dict(mph=46, k=0.003 * 0.985**i, excess=3, body_ratio=1.35)
    step(c, i, **kw)
    step(b, i, **kw)
  assert step(c, 12, mph=46, k=0.0, active=False) == step(b, 12, mph=46, k=0.0, active=False)
  assert c.term == 0


def test_unmodified_wait_does_not_consume_delivered_trim_time():
  c = make()
  for i in range(10):
    step(c, i, mph=46, k=0.003 * 0.985**i, excess=3, body_ratio=1.35)
  area = c.area
  for i in range(10, 65):
    step(c, i, mph=46, k=0.0026, excess=0.0, body_ratio=1.0)
  assert c.area >= area
  for i in range(65, 75):
    step(c, i, mph=46, k=0.0026 * 0.985 ** (i - 65), excess=3, body_ratio=1.35)
  assert c.term > 0
  assert c.active_time <= 1.0 and c.area <= 0.0015


def test_long_excess_cannot_recharge_time_or_area_without_quiet():
  c = make()
  for i in range(180):
    step(c, i, mph=82, k=for_target(c, 1.7), roll=-0.018, excess=1.7, body_ratio=1.9)
  assert c.active_time <= 1.0 and c.area <= 0.0015 and c.term == 0


@pytest.mark.parametrize("direction", [-1, 1])
def test_transition_and_reversal_envelopes(direction):
  c = make()
  previous = 0.0
  from opendbc.car.ford.lateral_angle import _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES
  import numpy as np

  for i in range(500):
    mph = 46 if i < 100 else 46 + min(40, (i - 100) * 0.2)
    k = direction * 0.002 * (0.99 ** (i % 45)) if mph < 60 else direction * 0.00035
    if i > 350:
      k = -k
    out = step(c, i, mph=mph, k=k, excess=2.0, body_ratio=1.5)
    assert abs(out.path_angle - previous) <= float(np.interp(mph * 0.44704, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES)) + 1e-12
    assert abs(out.path_angle) <= abs(c.stock_path) + 1e-12
    assert c.area <= 0.0015 and c.active_time <= 1.0
    previous = out.path_angle


def test_old_above_floor_target_cannot_enable_later_new_excess():
  c = make()
  for i in range(6):
    step(c, i, mph=82, k=for_target(c, 1.7), roll=-0.018, excess=0, body_ratio=1.0)
  for i in range(6, 20):
    step(c, i, mph=82, k=for_target(c, 1.4), roll=-0.018, excess=0, body_ratio=1.0)
  for i in range(20, 35):
    step(c, i, mph=82, k=for_target(c, 1.4), roll=-0.018, excess=1.7, body_ratio=1.9)
  assert c.term == 0


def test_duplicate_quiet_sources_never_rearm():
  c = make()
  for i in range(12):
    step(c, i, mph=46, k=0.003 * 0.985**i, excess=3, body_ratio=1.35)
  area = c.area
  for i in range(20, 55):
    step(c, i, mph=46, k=0, excess=0, body_ratio=1.0, stepdt_ns=40_000_000, state_lag_ns=40_000_000 if i % 2 else 0)
  assert c.area >= area and c.start is not None


def test_reversed_clock_cannot_refund_retained_slew_release():
  from types import SimpleNamespace as NS
  from opendbc.car.ford.lateral_angle import AngleLateralResult

  c = make()
  c.previous = 0.099
  c.term = 0.001
  c.area = 0.0004
  c.active_time = 0.2
  c.last = 2_000_000_000
  c.input_state_valid = False
  c.context = {"now_ns": 1_990_000_000}
  c.reference.actual_reference_path = 0.109
  c.reference.update = lambda *a: AngleLateralResult(path_angle=0.109, shadow_curvature=0.0)
  before = c.area
  c.update(NS(latActive=True), NS(out=NS(vEgoRaw=25.0)), NS())
  assert c.area >= before and c.active_time >= 0.2


@pytest.mark.parametrize("attribute", ["brakePressed", "gasPressed", "leftBlinker", "rightBlinker", "steerFaultPermanent"])
def test_pedal_signal_fault_suppresses_new_term(attribute):
  c = make()
  original = c.update

  def update(CC, CS, a):
    setattr(CS.out, attribute, True)
    return original(CC, CS, a)

  c.update = update
  for i in range(20):
    step(c, i, mph=46, k=0.003 * 0.985**i, excess=3, body_ratio=1.35)
  assert c.term == 0 and c.area == 0


def test_onset_increment_and_fraction_caps_are_explicit():
  c = make()
  last = 0.0
  for i in range(20):
    step(c, i, mph=46, k=0.003 * 0.985**i, excess=3, body_ratio=1.35)
    assert c.term <= last + 0.0003 + 1e-12
    assert c.term <= min(0.003, 0.05 * abs(c.stock_path)) + 1e-12
    last = c.term


def test_small_demand_jitter_does_not_spend_unwind_allowance():
  c = make()
  maximum = 0.0
  for i in range(50):
    k = 0.003 * (1 + 0.02 * math.sin(i))
    step(c, i, mph=46, k=k, excess=3, body_ratio=1.35)
    maximum = max(maximum, c.term)
  assert maximum == 0


def test_near_exhaustion_allocates_affordable_onset_without_refund():
  c = make()
  for i in range(4):
    step(c, i, mph=46, k=0.003 * 0.97**i, excess=3, body_ratio=1.35)
  c.area = 0.0013183436
  before = c.area
  step(c, 4, mph=46, k=0.003 * 0.97**4, excess=3, body_ratio=1.35)
  assert 0 < c.term <= 0.0003 + 1e-12
  assert c.area > before
  assert c.area + 0.00015 <= 0.0015 + 1e-12
  for i in range(5, 40):
    previous_area = c.area
    step(c, i, mph=46, k=0.003 * 0.97**i, excess=3, body_ratio=1.35)
    assert c.area >= previous_area
    assert c.area <= 0.0015 + 1e-12


def test_exhausted_admission_budget_does_not_earn_new_correction():
  c = make()
  for i in range(4):
    step(c, i, mph=46, k=0.003 * 0.97**i, excess=3, body_ratio=1.35)
  c.area = 0.00135
  step(c, 4, mph=46, k=0.003 * 0.97**4, excess=3, body_ratio=1.35)
  assert c.term == 0 and c.area == 0.00135


@pytest.mark.parametrize("direction", [-1, 1])
def test_sustained_unwind_spreads_area_across_existing_active_allowance(direction):
  c = make()
  terms = []
  for i in range(21):
    step(c, i, mph=46, k=direction * 0.003 * 0.97**i, excess=3, body_ratio=1.35)
    terms.append(c.term)
  assert min(terms[15:21]) > 0.0005
  assert c.area + 0.00015 <= 0.0015 + 1e-12
  assert c.active_time <= 0.9 + 1e-9


@pytest.mark.parametrize("direction", [-1, 1])
def test_fast_closing_low_unwind_does_not_spend_extra_allowance(direction):
  c = make()
  for i in range(5):
    step(c, i, mph=46, k=direction * 0.003 * 0.97**i, excess=4.0 - 0.4 * i, body_ratio=1.35)
  # Wheel excess2.4deg, measured slope-8deg/s: already approaching clearance
  # within0.3s, despite lagging body excess. Preserve baseline; do not earn trim.
  assert c.term == 0 and c.area == 0 and c.active_time == 0


def test_exhausted_duration_is_not_reported_as_cause_of_absent_demand():
  c = make()
  for i in range(4):
    step(c, i, mph=46, k=0.003, excess=0, body_ratio=1.0)
  c.active_time = 0.9
  c.area = 0.00135
  step(c, 4, mph=46, k=0.003, excess=0, body_ratio=1.0)
  assert c.reason == "closing_or_unconfirmed"
  assert c.term == 0 and c.area == 0.00135


def test_exhausted_duration_still_blocks_real_confirmed_demand():
  c = make()
  for i in range(4):
    step(c, i, mph=46, k=0.003 * 0.97**i, excess=3, body_ratio=1.35)
  c.active_time = 0.9
  c.area = 0.00135
  step(c, 4, mph=46, k=0.003 * 0.97**4, excess=3, body_ratio=1.35)
  assert c.reason == "duration" and c.term == 0 and c.area == 0.00135
