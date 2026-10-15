from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR, FordFlags
from opendbc.car.ford.lateral_angle_tracking_budget import BudgetedTrackingPrototype

WaitCatchupPrototype = BudgetedTrackingPrototype


def make(cls=WaitCatchupPrototype):
  cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)
  cp.flags |= int(FordFlags.PINION_CURVATURE | FordFlags.LOW_SPEED_GAIN)
  return cls(cp, enabled=True)


def step(c, i, k, m, driver=False, lane=False, mph=22):
  now = 1_000_000_000 + round(i * 50_000_000)
  c.set_context(now_ns=now, state_ns=now, control_ns=now, model_ns=now, model_valid=True, lane_change=lane)
  v = mph * 0.44704
  return c.correction(v, k, m, k * v * c.get_curvature_factor(v, k), driver)


def test_request_dip_without_measured_recovery_keeps_wait():
  c = make()
  for i in range(30):
    step(c, i, 0.0043, 0.0025)
  assert step(c, 30, 0.004, 0.0025) > 0


def test_real_recovery_releases():
  c = make()
  for i in range(30):
    step(c, i, 0.0043, 0.0025)
  assert step(c, 30, 0.0043, 0.0029) == 0


def test_actual_unwind_releases_at_ratio_threshold():
  c = make()
  for i in range(30):
    step(c, i, 0.0043, 0.0025)
  assert step(c, 30, 0.0035, 0.0025) == 0


def test_driver_lane_and_high_speed_release():
  for kwargs in [dict(driver=True), dict(lane=True), dict(mph=45)]:
    c = make()
    for i in range(30):
      step(c, i, 0.0043, 0.0025)
    assert step(c, 30, 0.004, 0.0025, **kwargs) == 0


def test_sustained_unwind_keeps_predictive_release():
  c = make()
  for i in range(30):
    step(c, i, 0.0043, 0.0025)
  for i, k in enumerate([0.0042, 0.0041, 0.0040], 30):
    x = step(c, i, k, 0.0025)
  assert x == 0 and c.admission_kind == "catchup_veto"


def test_noise_exception_budget_survives_reset_and_only_quiet_dwell_rearms():
  c = make()
  for i in range(30):
    step(c, i, 0.0043, 0.0025)
  for i in range(30, 70):
    step(c, i, 0.0040 if i % 2 == 0 else 0.0043, 0.0025)
  assert c.ramp_episode and 0 < c.ramp_extra_area <= 0.00025
  spent = c.ramp_extra_area
  c.reset_tracking()
  assert c.ramp_episode and c.ramp_extra_area == spent
  for i in range(70, 95):
    step(c, i, 0.0043, 0.0025, driver=True)
  assert c.ramp_extra_area == spent and c.rearm_count == 0
  for i in range(95, 117):
    step(c, i, 0.0, 0.0)
  assert c.ramp_extra_area == 0 and c.rearm_count == 1


def test_left_right_noise_exception_is_symmetric():
  a, b = make(), make()
  for i in range(60):
    k = 0.0043 if i < 30 or i % 2 else 0.0040
    x = step(a, i, k, 0.0025)
    y = step(b, i, -k, -0.0025)
    assert abs(x + y) < 1e-12
  assert a.ramp_extra_area == b.ramp_extra_area
