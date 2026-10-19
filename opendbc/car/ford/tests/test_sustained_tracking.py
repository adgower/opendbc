import pytest

from opendbc.car.ford.tests.test_wheel_target import step, make as oldmake
from opendbc.car.ford.lateral_angle_sustained_unwind import SustainedErrorUnwind as Candidate


def make(enabled=True):
  c = Candidate(oldmake().CP, enabled=enabled)
  c.input_state_valid = True
  return c


def baseline(c):
  return c.installed_baseline.reference.baseline


def spent(c):
  b = baseline(c)
  b.ramp_episode = True
  b.ramp_extra_area = b.RAMP_EXTRA_AREA_LIMIT


@pytest.mark.parametrize("sign", [-1, 1])
def test_sustained_deficit_gets_finite_extra_after_transient_budget_spent(sign):
  c = make()
  b = oldmake()
  spent(c)
  spent(b)
  extra = []
  for i in range(100):
    kw = dict(mph=15, k=sign * 0.01, excess=-20, body_ratio=0.45)
    x = step(c, i, **kw)
    y = step(b, i, **kw)
    extra.append(sign * (x.path_angle - y.path_angle))
  assert max(extra) > 0.0005
  assert max(extra) <= 0.0025 + 1e-12
  assert sum(extra) * 0.05 <= 0.0025 + 1e-10
  assert extra[-1] == 0


@pytest.mark.parametrize("sign", [-1, 1])
def test_64mph_unwind_feedback_has_bounded_reduction(sign):
  c = make()
  b = oldmake()
  ds = []
  for i in range(100):
    k = sign * 0.0009 if i < 60 else sign * 0.0009 * 0.99 ** (i - 60)
    x = step(c, i, mph=64, k=k)
    y = step(b, i, mph=64, k=k)
    if i >= 65:
      ds.append(abs(x.path_angle) - abs(y.path_angle))
  assert min(ds) < -0.00005


@pytest.mark.parametrize(
  "kwargs",
  [
    {"driver": True},
    {"active": False},
    {"body_ratio": 0.85},
    {"body_valid": False},
    {"body_age": 0.2},
    {"state_valid": False},
    {"state_lag_ns": 50_000_000},
    {"mismatch": 1.0},
    {"fault": True},
    {"lane": True},
    {"age": 0.3},
  ],
)
def test_low_invalid_no_added_correction(kwargs):
  c = make()
  b = oldmake()
  spent(c)
  spent(b)
  for i in range(100):
    kw = dict(mph=15, k=0.01, excess=-20, body_ratio=0.45)
    kw.update(kwargs)
    assert step(c, i, **kw) == step(b, i, **kw)


@pytest.mark.parametrize("mph", [25, 30, 40, 45, 50, 55, 60, 75, 83, 85])
def test_unchanged_ranges_whole_strategy_parity(mph):
  c = make()
  b = oldmake()
  for i in range(140):
    kw = dict(mph=mph, k=0.0008 if i < 70 else 0.0008 * 0.99 ** (i - 70), driver=105 <= i < 108)
    assert step(c, i, **kw) == step(b, i, **kw)


@pytest.mark.parametrize("mph", [15, 22, 64, 72, 83])
def test_disabled_whole_strategy_parity(mph):
  c = make(False)
  b = oldmake(False)
  for i in range(120):
    kw = dict(mph=mph, k=0.01 if mph < 25 else 0.0008, excess=-20 if mph < 25 else 2.4, body_ratio=0.45 if mph < 25 else 1.35)
    assert step(c, i, **kw) == step(b, i, **kw)


@pytest.mark.parametrize("kind", ["driver", "body_invalid", "body_closing", "state_old", "lane", "fault", "generation", "inactive"])
def test_existing_extra_withdraws_on_bad_or_closing_evidence(kind):
  c = make()
  spent(c)
  for i in range(23):
    step(c, i, mph=15, k=0.01, excess=-20, body_ratio=0.45)
  assert baseline(c).sustained_extra > 0
  kw = (
    {"driver": True}
    if kind == "driver"
    else {"body_valid": False}
    if kind == "body_invalid"
    else {"body_ratio": 0.8}
    if kind == "body_closing"
    else {"state_lag_ns": 50_000_000}
    if kind == "state_old"
    else {"lane": True}
    if kind == "lane"
    else {"fault": True}
    if kind == "fault"
    else {"generation": 1}
    if kind == "generation"
    else {"active": False}
  )
  step(c, 23, mph=15, k=0.01, excess=-20, body_ratio=kw.pop("body_ratio", 0.45), **kw)
  # Metadata may describe the preceding correction after an inactive update;
  # actual inactive output is verified separately by parity tests.
  if kind != "inactive":
    assert baseline(c).sustained_extra == 0
  assert baseline(c).sustained_area <= 0.0025 + 1e-12


@pytest.mark.parametrize("sign", [-1, 1])
def test_repeated_turns_without_full_quiet_do_not_refill(sign):
  c = make()
  spent(c)
  delivered = 0
  for i in range(800):
    k = sign * 0.01 * (1 if (i // 60) % 2 == 0 else -1)
    step(c, i, mph=15, k=k, excess=-20, body_ratio=0.45)
    delivered += abs(baseline(c).sustained_extra) * 0.05
  assert 0 < delivered <= 0.0025 + 1e-10


@pytest.mark.parametrize("valid", [False, True])
def test_only_valid_full_quiet_can_refill(valid):
  c = make()
  spent(c)
  for i in range(100):
    step(c, i, mph=15, k=0.01, excess=-20, body_ratio=0.45)
  before = baseline(c).sustained_area
  assert before > 0
  for i in range(100, 125):
    step(c, i, mph=15, k=0, excess=1.5, body_ratio=0, body_valid=valid)
  assert (baseline(c).sustained_area == 0) == valid


@pytest.mark.parametrize("mph", [5, 10, 15, 20, 24])
def test_low_peak_duration_and_area_bounds_across_speed(mph):
  c = make()
  spent(c)
  for i in range(300):
    step(c, i, mph=mph, k=0.06 if mph < 10 else 0.02 if mph < 15 else 0.01, excess=-20, body_ratio=0.45)
    b = baseline(c)
    assert abs(b.sustained_extra) <= 0.0025 * b.speed_weight(mph * 0.44704) + 1e-12
    assert b.sustained_area <= 0.0025 + 1e-12
    if b.sustained_start is not None and (1_000_000_000 + i * 50_000_000 - b.sustained_start) >= 1_000_000_000:
      assert b.sustained_extra == 0


@pytest.mark.parametrize("sign", [-1, 1])
def test_speed_exit_25mph_joins_current_reference(sign):
  c = make()
  b = oldmake()
  spent(c)
  spent(b)
  for i in range(150):
    mph = 15 if i < 23 else min(45, 15 + (i - 23) * 0.5)
    kw = dict(mph=mph, k=sign * 0.01, excess=-20, body_ratio=0.45)
    x = step(c, i, **kw)
    y = step(b, i, **kw)
    if mph >= 25:
      assert x == y


def test_invalid_body_dwell_then_one_valid_frame_cannot_refill():
  c = make()
  b = baseline(c)
  b.sustained_area = 0.0025
  b.sustained_start = 500_000_000
  b.sustained_sign = 1
  b.sustained_aborted = True
  for i in range(21):
    step(c, i, mph=15, k=0, excess=1.5, body_ratio=0, body_valid=(i == 20))
  assert b.sustained_area == 0.0025 and b.sustained_aborted
  for i in range(21, 41):
    step(c, i, mph=15, k=0, excess=1.5, body_ratio=0)
  assert b.sustained_area == 0 and not b.sustained_aborted


@pytest.mark.parametrize("sign", [-1, 1])
def test_body_catching_up_with_still_low_wheel_withdraws(sign):
  c = make()
  spent(c)
  for i in range(30):
    step(c, i, mph=15, k=sign * 0.01, excess=-20, body_ratio=0.45)
  assert sign * baseline(c).sustained_extra > 0
  for j, r in enumerate([0.49, 0.53, 0.57, 0.61, 0.65]):
    step(c, 30 + j, mph=15, k=sign * 0.01, excess=-20, body_ratio=r)
  assert baseline(c).sustained_extra == 0


def test_body_still_turning_cannot_refill_quiet_allowance():
  c = make()
  b = baseline(c)
  b.sustained_area = 0.0025
  b.sustained_start = 500_000_000
  b.sustained_sign = 1
  b.sustained_aborted = True
  original = c.set_motion

  def force_turn(**kw):
    kw["yaw_rate"] = 0.1
    original(**kw)

  c.set_motion = force_turn
  for i in range(25):
    step(c, i, mph=15, k=0, excess=1.5, body_ratio=0)
  assert b.sustained_area == 0.0025 and b.sustained_aborted


@pytest.mark.parametrize("kind", ["driver", "inactive", "lane", "body_invalid", "mismatch"])
def test_abort_during_active_extra_rejoins_reference(kind):
  c = make()
  b = oldmake()
  spent(c)
  spent(b)
  for i in range(23):
    for z in [c, b]:
      step(z, i, mph=15, k=0.01, excess=-20, body_ratio=0.45)
  assert baseline(c).sustained_extra > 0
  kw = (
    {"driver": True}
    if kind == "driver"
    else {"active": False}
    if kind == "inactive"
    else {"lane": True}
    if kind == "lane"
    else {"body_valid": False}
    if kind == "body_invalid"
    else {"mismatch": 1.0}
  )
  assert step(c, 23, mph=15, k=0.01, excess=-20, body_ratio=0.45, **kw) == step(b, 23, mph=15, k=0.01, excess=-20, body_ratio=0.45, **kw)
  assert baseline(c).sustained_extra == 0


@pytest.mark.parametrize("new_k", [-0.01, 0.002])
def test_sudden_reversal_or_abort_retains_native_bounded_handoff(new_k):
  c = make()
  b = oldmake()
  spent(c)
  spent(b)
  for i in range(23):
    xc = step(c, i, mph=15, k=0.01, excess=-20, body_ratio=0.45)
    step(b, i, mph=15, k=0.01, excess=-20, body_ratio=0.45)
  previous = xc.path_angle
  tail = []
  for i in range(23, 35):
    x = step(c, i, mph=15, k=new_k, excess=-20, body_ratio=0.45)
    y = step(b, i, mph=15, k=new_k, excess=-20, body_ratio=0.45)
    assert abs(x.path_angle - previous) <= 0.055 + 1e-12
    assert abs(x.path_angle - y.path_angle) <= 0.0025 + 1e-12
    tail.append(abs(x.path_angle - y.path_angle))
    previous = x.path_angle
  assert tail[-1] < 1e-12 and all(x < 1e-12 for x in tail[6:])
