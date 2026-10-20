"""Local-only behavioral acceptance; frozen motion is not a handling prediction."""

import math
import pytest
from opendbc.car.ford.lateral_angle_sustained_unwind import SustainedErrorUnwind as Installed
from opendbc.car.ford.tests.test_wheel_target import step as original_step
from opendbc.car.ford.tests.test_high_speed import make as oldmake
from opendbc.car.vehicle_model import VehicleModel
from opendbc.car.ford.lateral_angle_calibrated_mid_unwind import CalibratedMidUnwind as Candidate


def make(enabled=True):
  c = Candidate(oldmake().CP, enabled=enabled)
  c.input_state_valid = True
  return c


def step(c, i, *, wheel_ratio=None, **kw):
  if wheel_ratio is not None:
    vm = VehicleModel(c.CP)
    vm.update_params(1.0, 22.5)
    v = kw.get("mph", 23) * 0.44704
    k = kw.get("k", 0.004)
    roll = kw.get("roll", 0.0)
    desired = math.degrees(vm.get_steer_from_curvature(-k, v, roll))
    kw["excess"] = (wheel_ratio - 1) * abs(desired)
  return original_step(c, i, **kw)


def baseline(c):
  return c.installed_baseline.reference.baseline


@pytest.mark.parametrize("sign", [-1, 1])
def test_calibrated_deficit_not_vetoed_by_fixed_pinion(sign):
  c = make()
  old = Installed(c.CP, enabled=True)
  old.input_state_valid = True
  changed = []
  corrections = []
  for i in range(100):
    kw = dict(mph=23, k=sign * 0.004, wheel_ratio=0.60, body_ratio=0.60)
    x = step(c, i, **kw)
    y = step(old, i, **kw)
    changed.append(sign * (x.path_angle - y.path_angle))
    corrections.append(sign * baseline(c).tracking_correction)
  assert max(corrections) > 0.0005
  if sign > 0:
    assert max(changed) > 0.00005
  assert getattr(baseline(c), "feedback_definition", None) == "calibrated_wheel_body"
  assert max(changed) <= 0.0075 + 1e-12


@pytest.mark.parametrize("sign", [-1, 1])
def test_mid_band_has_finite_bounded_correction(sign):
  c = make()
  old = Installed(c.CP, enabled=True)
  old.input_state_valid = True
  delta = []
  for i in range(160):
    kw = dict(mph=27, k=sign * 0.008, wheel_ratio=0.80, body_ratio=0.80)
    x = step(c, i, **kw)
    y = step(old, i, **kw)
    delta.append(sign * (x.path_angle - y.path_angle))
  assert max(delta) > 0.0005
  assert min(delta) >= -1e-12 and max(delta) <= 0.0025 + 1e-12
  assert sum(delta) * 0.05 <= 0.0015 + 1e-10
  assert abs(delta[-1]) < 1e-12


@pytest.mark.parametrize("sign", [-1, 1])
def test_unwind_trim_exceeds_single_wire_step_when_excess_persists(sign):
  c = make()
  old = Installed(c.CP, enabled=True)
  old.input_state_valid = True
  delta = []
  for i in range(110):
    k = sign * 0.0009 * (1 if i < 50 else 0.997 ** (i - 50))
    kw = dict(mph=64, k=k, excess=2.4, body_ratio=1.4)
    x = step(c, i, **kw)
    y = step(old, i, **kw)
    if i >= 65:
      delta.append(abs(y.path_angle) - abs(x.path_angle))
  assert max(delta) > 0.0005
  assert max(delta) < 0.01


@pytest.mark.parametrize("mph", [5, 15, 23, 27, 30, 45, 64, 72, 75, 83])
def test_default_off_matches_installed(mph):
  c = make(False)
  b = Installed(c.CP, enabled=False)
  b.input_state_valid = True
  for i in range(100):
    kw = dict(mph=mph, k=0.008 if mph < 35 else 0.0008, wheel_ratio=0.6 if mph < 35 else 1.3, body_ratio=0.6 if mph < 35 else 1.3)
    assert step(c, i, **kw) == step(b, i, **kw)


@pytest.mark.parametrize("mph", [35, 40, 45, 50, 55, 60, 75, 83, 85])
def test_protected_speed_ranges_match_whole_stream(mph):
  c = make()
  b = Installed(c.CP, enabled=True)
  b.input_state_valid = True
  for i in range(120):
    kw = dict(mph=mph, k=0.0008 * (1 if i < 60 else 0.99 ** (i - 60)), excess=2.4, body_ratio=1.4, driver=90 <= i < 93)
    assert step(c, i, **kw) == step(b, i, **kw)


@pytest.mark.parametrize("mph", [23, 27, 64])
@pytest.mark.parametrize(
  "fault",
  [
    {"driver": True},
    {"active": False},
    {"valid": False},
    {"state_valid": False},
    {"body_valid": False},
    {"body_age": 0.2},
    {"state_lag_ns": 100_000_000},
    {"lane": True},
    {"fault": True},
    {"mismatch": 1.0},
  ],
)
def test_invalid_inputs_never_create_extra(mph, fault):
  c = make()
  b = Installed(c.CP, enabled=True)
  b.input_state_valid = True
  for i in range(60):
    kw = dict(mph=mph, k=0.008 if mph < 30 else 0.0008, wheel_ratio=0.6 if mph < 30 else 1.4, body_ratio=0.6 if mph < 30 else 1.4)
    kw.update(fault)
    assert step(c, i, **kw) == step(b, i, **kw)


@pytest.mark.parametrize("sign", [-1, 1])
def test_calibrated_feedback_cannot_deliver_unbounded_steady_extra(sign):
  c = make()
  area = 0.0
  for i in range(400):
    step(c, i, mph=20, k=sign * 0.006, wheel_ratio=0.55, body_ratio=0.55)
    area += abs(baseline(c).tracking_correction) * 0.05
  assert area <= 0.00275 + 1e-10
  assert baseline(c).tracking_correction == 0


@pytest.mark.parametrize("kind", ["driver", "inactive", "body", "state", "mismatch", "lane", "generation", "reverse", "closing"])
def test_mid_active_term_withdraws_and_does_not_refill(kind):
  c = make()
  for i in range(10):
    step(c, i, mph=27, k=0.008, wheel_ratio=0.8, body_ratio=0.8)
  assert c.mid.term > 0 and c.mid.area > 0
  before = c.mid.area
  for i in range(10, 60):
    kw = dict(mph=27, k=0.008, wheel_ratio=0.8, body_ratio=0.8)
    if kind == "driver":
      kw["driver"] = i < 20
    if kind == "inactive":
      kw["active"] = i >= 20
    if kind == "body":
      kw["body_valid"] = i >= 20
    if kind == "state":
      kw["state_lag_ns"] = 100_000_000 if i < 20 else 0
    if kind == "mismatch":
      kw["mismatch"] = 1.0 if i < 20 else 0
    if kind == "lane":
      kw["lane"] = i < 20
    if kind == "generation":
      kw["generation"] = 1
    if kind == "reverse":
      kw["k"] = -0.008
    if kind == "closing":
      kw["wheel_ratio"] = 0.95
      kw["body_ratio"] = 0.95
    step(c, i, **kw)
    assert c.mid.term == 0
  assert c.mid.area >= before and c.mid.area <= 0.0015 + 1e-12


@pytest.mark.parametrize("valid", [False, True])
def test_mid_only_full_valid_quiet_dwell_rearms(valid):
  c = make()
  for i in range(60):
    step(c, i, mph=27, k=0.008, wheel_ratio=0.8, body_ratio=0.8)
  assert c.mid.area > 0
  for i in range(60, 85):
    step(c, i, mph=27, k=0, excess=0, body_ratio=0, body_valid=valid)
  assert (c.mid.area == 0) == valid


@pytest.mark.parametrize("valid", [False, True])
def test_calibrated_regular_and_sustained_rearm_only_after_valid_quiet(valid):
  c = make()
  for i in range(120):
    step(c, i, mph=20, k=0.006, wheel_ratio=0.55, body_ratio=0.55)
  b = baseline(c)
  assert b.ramp_extra_area > 0
  for i in range(120, 150):
    step(c, i, mph=20, k=0, excess=0, body_ratio=0, body_valid=valid)
  assert (b.ramp_extra_area == 0) == valid
  assert (b.sustained_start is None) == valid
  if valid:
    terms = []
    for i in range(150, 190):
      step(c, i, mph=20, k=0.006, wheel_ratio=0.55, body_ratio=0.55)
      terms.append(b.tracking_correction)
    assert max(terms) > 0


@pytest.mark.parametrize("kind", ["mismatch", "closing", "speed_exit"])
def test_mid_single_frame_abort_cannot_readmit_without_quiet(kind):
  c = make()
  for i in range(7):
    step(c, i, mph=27, k=0.008, wheel_ratio=0.8, body_ratio=0.8)
  assert c.mid.term > 0
  kw = dict(mph=27, k=0.008, wheel_ratio=0.8, body_ratio=0.8)
  if kind == "mismatch":
    kw["mismatch"] = 1.0
  if kind == "closing":
    kw["wheel_ratio"] = 0.95
    kw["body_ratio"] = 0.95
  if kind == "speed_exit":
    kw["mph"] = 30.1
  step(c, 7, **kw)
  assert c.mid.term == 0
  for i in range(8, 18):
    step(c, i, mph=27, k=0.008, wheel_ratio=0.8, body_ratio=0.8)
    assert c.mid.term == 0


@pytest.mark.parametrize("sign", [-1, 1])
def test_mid_cannot_add_effort_when_native_deviation_clip_is_binding(sign):
  c = make()
  b = Installed(c.CP, enabled=True)
  b.input_state_valid = True
  for i in range(30):
    kw = dict(mph=27, k=sign * 0.008, wheel_ratio=0.2, body_ratio=0.2)
    assert step(c, i, **kw) == step(b, i, **kw)
    assert c.mid.term == 0


@pytest.mark.parametrize("kind", ["duplicate_body", "demand_abort", "predicted_catchup"])
def test_mid_transient_invalid_or_recovering_evidence_latches_active_abort(kind):
  c = make()
  for i in range(7):
    step(c, i, mph=27, k=0.008, wheel_ratio=0.8, body_ratio=0.8)
  assert c.mid.term > 0
  kw = dict(mph=27, k=0.008, wheel_ratio=0.8, body_ratio=0.8)
  if kind == "duplicate_body":
    kw["body_age"] = 0.05
  if kind == "demand_abort":
    kw["k"] = 0.007
  if kind == "predicted_catchup":
    kw["wheel_ratio"] = 0.89
    kw["body_ratio"] = 0.89
  step(c, 7, **kw)
  for i in range(8, 18):
    step(c, i, mph=27, k=0.008, wheel_ratio=0.8, body_ratio=0.8)
    assert c.mid.term == 0
