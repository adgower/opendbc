import math
import pytest
from opendbc.car.ford.longitudinal_coast import BoundedCoastAdmission
from opendbc.car.ford.longitudinal_coordination import Torque


def call(a, t, net=-0.02, **kw):
  values = dict(eligible=True, baseline=net < 0, previous=False, raw=0.1, net=net, cleared=True, torque=Torque(t, 0.0))
  values.update(kw)
  return a.update(t, **values)


def test_weak_oscillation_does_not_repeatedly_brake_or_withdraw_clear_propulsion():
  a = BoundedCoastAdmission()
  brakes = []
  coasts = []
  for i in range(1000):
    t = 1 + i * 0.02
    r = call(a, t, 0.04 * math.sin(2 * math.pi * 2 * i * 0.02))
    brakes.append(r.brake)
    coasts.append(r.coast)
  assert sum(brakes) == 0 and not any(coasts)


def test_persistent_weak_negative_demand_is_admitted_within_nominal_300ms():
  a = BoundedCoastAdmission()
  call(a, 1.0, 0.04)
  times = [1.02 + i * 0.02 for i in range(30)]
  first = next(t for t in times if call(a, t, -0.01).brake)
  assert first <= 1.32 + 1e-9
  assert first >= 1.3 - 1e-9


@pytest.mark.parametrize(
  "kw",
  [
    {"raw": 0.0},
    {"net": -0.1},
    {"previous": True},
    {"cleared": False},
    {"eligible": False},
    {"torque": Torque(0.5, 0.0)},
    {"torque": Torque(1.02, 8.0)},
    {"torque": Torque(1.02, 0.0, False)},
    {"raw": float("nan")},
  ],
)
def test_unsafe_or_meaningful_braking_retains_baseline(kw):
  a = BoundedCoastAdmission()
  call(a, 1.0, 0.03)
  assert call(a, 1.02, **kw).brake


@pytest.mark.parametrize("gap", [0.0, -0.02, 0.041, 0.2])
def test_bad_timing_retains_baseline(gap):
  a = BoundedCoastAdmission()
  call(a, 1.0, 0.03)
  assert call(a, 1.0 + gap).brake


def test_budget_admits_before_continuous_deadline():
  a = BoundedCoastAdmission()
  call(a, 1.0, 0.03)
  outputs = [(1.02 + i * 0.02, call(a, 1.02 + i * 0.02, -0.049)) for i in range(20)]
  first = next(t for t, r in outputs if r.brake)
  assert 1.18 <= first <= 1.24


def test_strong_braking_immediate_after_coasting():
  a = BoundedCoastAdmission()
  call(a, 1.0, 0.03)
  call(a, 1.02, -0.01)
  assert call(a, 1.04, -0.5, raw=-0.5).brake


def test_irregular_39ms_callbacks_admit_at_next_sample_after_deadline():
  a = BoundedCoastAdmission()
  call(a, 1.0, 0.03)
  times = [1.02 + i * 0.039 for i in range(12)]
  first = next(t for t in times if call(a, t, -0.01).brake)
  assert 0.3 <= first - 1.02 < 0.34


def test_sampled_budget_can_overshoot_but_by_less_than_one_40ms_increment():
  a = BoundedCoastAdmission()
  call(a, 1.0, 0.03)
  area = 0.0
  last = 1.0
  for i in range(20):
    t = 1.04 + i * 0.04
    area += 0.049 * (t - last)
    last = t
    r = call(a, t, -0.049)
    if r.brake:
      break
  assert a.budget <= area < a.budget + 0.002
