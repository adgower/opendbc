import math
import pytest
from opendbc.car.ford.lateral_angle_early_low_high_onset import EarlyLowHighOnset as Installed
from opendbc.car.ford.lateral_angle_observer_continuity import ObserverContinuity as Candidate
from opendbc.car.ford.tests.test_high_speed import make as oldmake
from opendbc.car.ford.tests.test_wheel_target import step


def make(cls=Candidate, enabled=True):
  c = cls(oldmake().CP, enabled=enabled)
  c.input_state_valid = True
  return c


def sequence(c, mph=77, **fault):
  rows = []
  for i in range(50):
    kw = dict(mph=mph, k=0.0009, excess=2.4, body_ratio=0.95 if i < 25 else 1.5)
    if i >= 25:
      kw.update(fault)
    a = step(c, i, **kw)
    rows.append(a.path_angle)
  return rows


@pytest.mark.parametrize("mph", [71, 74, 76, 79])
def test_valid_observer_history_admits_earlier_after_body_confirmation(mph):
  c = make()
  b = make(Installed)
  x = sequence(c, mph)
  y = sequence(b, mph)
  assert any(abs(a) < abs(v) - 1e-6 for a, v in zip(x[25:], y[25:], strict=False))
  assert all(abs(a) <= abs(v) + 0.01 + 1e-12 for a, v in zip(x, y, strict=False))


@pytest.mark.parametrize(
  "fault",
  [
    dict(driver=True),
    dict(active=False),
    dict(valid=False),
    dict(state_valid=False),
    dict(fault=True),
    dict(body_valid=False),
    dict(body_age=0.2),
    dict(body_std=0.1),
    dict(lane=True),
    dict(mismatch=1.0),
  ],
)
def test_invalid_evidence_cannot_earn_extra_trim(fault):
  assert sequence(make(), **fault) == sequence(make(Installed), **fault)


@pytest.mark.parametrize("mph", [15, 20, 22, 25, 28, 45, 55, 65, 70, 80, 83, 85])
def test_static_protected_speed_parity(mph):
  assert sequence(make(), mph) == sequence(make(Installed), mph)


def test_default_off():
  assert sequence(make(enabled=False)) == sequence(make(Installed, enabled=False))


def test_clipped_low_observer_preserves_original_target_meaning():
  c = make()
  for i in range(12):
    step(c, i, mph=23, k=0.008, excess=-15, body_ratio=0.35)
  low = c.reference.baseline
  assert low.observer_valid and low.observer_reason == "headroom_limited"
  assert low.continuity_term == 0
  assert low.ramp_extra_area <= low.RAMP_EXTRA_AREA_LIMIT + 1e-12


@pytest.mark.parametrize("sign", [-1, 1])
def test_low_observation_only_preserves_installed_actuation(sign):
  from opendbc.car.vehicle_model import VehicleModel

  c = make()
  b = make(Installed)
  changed = []
  for i in range(22):
    k = sign * 0.0012 * 1.015**i
    v = 23 * 0.44704
    vm = VehicleModel(c.CP)
    vm.update_params(1.0, 22.5)
    target = abs(math.degrees(vm.get_steer_from_curvature(-k, v, 0.0)))
    kw = dict(mph=23, k=k, excess=-0.6 * target, body_ratio=0.95 if i < 12 else 0.35, body_std=0.005)
    x = step(c, i, **kw)
    y = step(b, i, **kw)
    if abs(x.path_angle - y.path_angle) > 1e-10:
      changed.append(i)
    assert x.shadow_curvature == y.shadow_curvature
    assert c.reference.baseline.ramp_extra_area <= c.reference.baseline.RAMP_EXTRA_AREA_LIMIT + 1e-12
  assert not changed, "Low actuation withdrawn: retain observation without spending installed budget"


@pytest.mark.parametrize("mph", [70.01, 71, 74.99, 75, 75.01, 77, 79, 79.99])
@pytest.mark.parametrize("sign", [-1, 1])
def test_trim_stays_reduction_only_and_within_existing_caps(mph, sign):
  c = make()
  for i in range(100):
    k = sign * 0.0009 * (1 if i < 50 else 0.99 ** (i - 50))
    a = step(c, i, mph=mph, k=k, body_ratio=1.5)
    # The selected policy has an independent untrimmed reference; all layers share it here.
    base = c.reference.path_angle_last
    assert abs(a.path_angle - base) <= min(0.01, 0.15 * abs(base)) + 1e-10
    assert a.path_angle * base >= 0 and abs(a.path_angle) <= abs(base) + 1e-10


@pytest.mark.parametrize(
  "fault",
  [
    dict(driver=True),
    dict(active=False),
    dict(valid=False),
    dict(state_valid=False),
    dict(fault=True),
    dict(body_valid=False),
    dict(body_age=0.2),
    dict(body_std=0.1),
    dict(lane=True),
    dict(mismatch=1.0),
    dict(excess=-0.2),
    dict(state_lag_ns=100_000_000),
  ],
)
def test_existing_trim_withdraws_on_abort(fault):
  c = make()
  for i in range(30):
    step(c, i, mph=77, body_ratio=1.5)
  for i in range(30, 38):
    a = step(c, i, mph=77, body_ratio=1.5, **fault)
  assert abs(a.path_angle - c.reference.path_angle_last) < 1e-10


@pytest.mark.parametrize("start,end", [(69, 81), (81, 69), (21, 26), (26, 21)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_speed_transitions_preserve_slew_shadow_and_protected_steady_output(start, end, sign):
  c = make()
  b = make(Installed)
  last = 0.0
  from numpy import interp
  from opendbc.car.ford.lateral_angle import _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES

  for i in range(220):
    mph = start + (end - start) * min(i / 160, 1)
    k = sign * (0.0009 if start > 30 else 0.0014)
    kw = dict(mph=mph, k=k, body_ratio=1.5 if start > 30 else 0.4, excess=2.4 if start > 30 else -3.0)
    a = step(c, i, **kw)
    r = step(b, i, **kw)
    assert a.shadow_curvature == r.shadow_curvature
    assert abs(a.path_angle - last) <= float(interp(mph * 0.44704, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES)) + 1e-10
    assert -0.5 <= a.path_angle <= 0.5235
    last = a.path_angle
    if i > 200 and (end >= 80 or end <= 22 or 25 <= end <= 70):
      assert a == r


def test_reversal_does_not_reuse_observer_history():
  c = make()
  for i in range(20):
    step(c, i, mph=77)
  step(c, 20, mph=77, k=-0.00065)
  assert c.high_observer.reason == "reversal_duplicate_or_cadence" and not c.high_observer.rows


@pytest.mark.parametrize("sign", [-1, 1])
def test_highway_fade_does_not_accelerate_baseline_admission_above80(sign):
  c = make()
  b = make(Installed)
  for i in range(65):
    mph = 79.7 if i < 30 else min(81.0, 79.7 + 0.04 * (i - 30))
    kw = dict(mph=mph, k=sign * 0.0009, excess=2.4, body_ratio=0.95 if i < 30 else 1.5)
    a = step(c, i, **kw)
    r = step(b, i, **kw)
    if mph >= 80:
      assert a == r


@pytest.mark.parametrize("sign", [-1, 1])
def test_total_trim_onset_keeps_installed_ramp_cap(sign):
  c = make()
  previous = 0.0
  for i in range(100):
    a = step(c, i, mph=77, k=sign * 0.0009, body_ratio=0.95 if i < 25 else 1.5)
    raw = c.unchanged.stock.upper_reference.reference.path_angle_last
    total = max(0.0, abs(raw) - abs(a.path_angle))
    assert total <= previous + 0.0003 + 1e-10
    previous = total
