import math
import pytest
from opendbc.car.ford.lateral_angle_early_low_high_onset import EarlyLowHighOnset as Candidate
from opendbc.car.ford.lateral_angle_earlier_unwind_onset import EarlierUnwindOnset as Installed
from opendbc.car.ford.tests.test_high_speed import make as oldmake
from opendbc.car.ford.tests.test_wheel_target import step
from opendbc.car.vehicle_model import VehicleModel
from opendbc.can import CANPacker
from opendbc.car.ford import fordcan
from opendbc.car.ford.values import DBC
from opendbc.car import Bus


def make(cls=Candidate, enabled=True):
  c = cls(oldmake().CP, enabled=enabled)
  c.input_state_valid = True
  return c


def pack(c, value):
  d = fordcan.create_lat_ctl2_msg(CANPacker(DBC[c.CP.carFingerprint][Bus.pt]), fordcan.CanBus(c.CP), 1, 0.0, -value, 0.0, 0.0, 0)[1]
  return -((((d[3] & 31) << 6) | (d[4] >> 2)) * 0.0005 - 0.5)


def low_sequence(c, sign=1, **fault):
  rows = []
  for i in range(32):
    v = 22 * 0.44704
    k = sign * 0.0012 * 1.015**i
    vm = VehicleModel(c.CP)
    vm.update_params(1.0, 22.5)
    target = abs(math.degrees(vm.get_steer_from_curvature(-k, v, 0.0)))
    kw = dict(mph=22, k=k, excess=-0.6 * target, body_ratio=0.35, body_std=0.005)
    kw.update(fault)
    a = step(c, i, **kw)
    rows.append((pack(c, a.path_angle), a.shadow_curvature, a.path_angle))
  return rows


@pytest.mark.parametrize("sign", [-1, 1])
def test_preclip_rising_demand_gets_bounded_earlier_packed_effort(sign):
  c = make()
  b = make(Installed)
  a = low_sequence(c, sign)
  r = low_sequence(b, sign)
  changed = [i for i, (x, y) in enumerate(zip(a, r, strict=False)) if x[0] != y[0]]
  assert changed, "No earlier encoded pre-clip correction"
  assert max(abs(x[2] - y[2]) for x, y in zip(a, r, strict=False)) <= 0.0015 + 1e-12
  assert all(x[1] == y[1] for x, y in zip(a, r, strict=False))


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
def test_early_low_respects_guards(fault):
  assert low_sequence(make(), **fault) == low_sequence(make(Installed), **fault)


def high_sequence(c, mph, sign=1, **fault):
  rows = []
  for i in range(50):
    kw = dict(mph=mph, k=sign * 0.0009 * (1.0 if i < 30 else 0.985 ** (i - 29)), excess=2.4, body_ratio=0.95 if i < 30 else 1.5)
    if i >= 30:
      kw.update(fault)
    a = step(c, i, **kw)
    rows.append((i * 0.05, pack(c, c.actual_reference_path) - pack(c, a.path_angle), a.path_angle, a.shadow_curvature))
  return rows


@pytest.mark.parametrize("mph", [71, 72, 74, 74.9])
@pytest.mark.parametrize("sign", [-1, 1])
def test_70to75_withdrawal_has_earlier_packed_onset(mph, sign):
  a = high_sequence(make(), mph, sign)
  b = high_sequence(make(Installed), mph, sign)

  def first(r):
    return next(t for t, d, *_ in r if d * sign > 1e-10)

  assert first(a) <= first(b) - 0.049


@pytest.mark.parametrize("mph", [5, 15, 19, 25, 27, 30, 45, 55, 60, 65, 69, 70, 75, 83, 85])
def test_protected_ranges(mph):
  assert high_sequence(make(), mph) == high_sequence(make(Installed), mph)


def test_default_off():
  assert low_sequence(make(enabled=False)) == low_sequence(make(Installed, enabled=False))
  assert high_sequence(make(enabled=False), 74) == high_sequence(make(Installed, enabled=False), 74)


def baseline(c):
  return c.installed_baseline.reference.baseline


def wheel_step(c, i, *, wheel_ratio=None, **kw):
  if wheel_ratio is not None:
    v = kw.get("mph", 23) * 0.44704
    k = kw.get("k", 0.004)
    vm = VehicleModel(c.CP)
    vm.update_params(1.0, 22.5)
    target = abs(math.degrees(vm.get_steer_from_curvature(-k, v, kw.get("roll", 0.0))))
    kw["excess"] = (wheel_ratio - 1) * target
  return step(c, i, **kw)


@pytest.mark.parametrize("mph", [71, 72, 74, 74.9])
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
def test_high_onset_does_not_bypass_current_guards(mph, fault):
  assert high_sequence(make(), mph, **fault) == high_sequence(make(Installed), mph, **fault)


@pytest.mark.parametrize("kind", ["duplicate_state", "stale_state", "duplicate_motion", "stale_motion", "generation", "reverse", "closing"])
def test_high_credit_breaks_on_invalid_history(kind):
  c = make()
  for i in range(33):
    step(c, i, mph=74, k=0.0009 * (1 if i < 30 else 0.985 ** (i - 29)), excess=2.4, body_ratio=0.95 if i < 30 else 1.5)
  for i in range(33, 37):
    kw = dict(mph=74, k=0.0009 * 0.985 ** (i - 29), excess=2.4, body_ratio=1.5)
    if kind == "duplicate_state":
      kw["state_lag_ns"] = (i - 32) * 50_000_000
    if kind == "stale_state":
      kw["state_lag_ns"] = 100_000_000
    if kind == "duplicate_motion":
      kw["body_age"] = (i - 32) * 0.05
    if kind == "stale_motion":
      kw["body_age"] = 0.2
    if kind == "generation":
      kw["generation"] = 1
    if kind == "reverse":
      kw["k"] = -kw["k"]
    if kind == "closing":
      kw["excess"] = 0.1
    step(c, i, **kw)
    assert c.credit_receipt is None and c.upper_reference.credit_receipt is None


@pytest.mark.parametrize("mph", [71, 74])
@pytest.mark.parametrize("kind", ["constant", "rising"])
def test_high_credit_requires_withdrawing_demand(mph, kind):
  c = make()
  for i in range(80):
    k = 0.0009 * (1 if kind == "constant" else 1 + 0.002 * i)
    step(c, i, mph=mph, k=k, excess=2.4, body_ratio=1.5)
    assert c.credit_receipt is None and c.upper_reference.credit_receipt is None


@pytest.mark.parametrize("sign", [-1, 1])
def test_early_low_shares_budget_and_stays_blocked_after_abort(sign):
  c = make()
  low_sequence(c, sign)
  low = baseline(c)
  first = low.early_total_area
  assert first > 0 and low.ramp_extra_area <= 0.00025 + 1e-12
  # No settled neutral dwell: a new direction cannot rearm the spent episode.
  for i in range(32, 90):
    wheel_step(c, i, mph=22, k=-sign * 0.0012 * 1.015 ** (i - 32), wheel_ratio=0.4, body_ratio=0.35, body_std=0.005)
  assert low.early_total_area == first


def test_early_area_uses_measured_sixty_ms_cadence():
  c = make()
  low = baseline(c)
  physical_area = 0.0
  for i in range(18):
    wheel_step(c, i, mph=22, k=0.0012 * 1.015**i, wheel_ratio=0.4, body_ratio=0.35, body_std=0.005, stepdt_ns=60_000_000)
    physical_area += abs(low.early_term) * 0.06
  assert physical_area <= low.RAMP_EXTRA_AREA_LIMIT + 1e-12
  assert abs(low.ramp_extra_area - physical_area) < 1e-12
