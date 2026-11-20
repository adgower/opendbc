from types import SimpleNamespace as NS
import math

import pytest
from opendbc.car.ford.lateral_angle import AngleLateralResult
from opendbc.car.ford.lateral_angle_steady_curve import SteadyCurveRelief


def model(rates):
  return NS(orientationRate=NS(t=[0., .25, .5, .75, 1., 1.25], z=rates), velocity=NS(x=[30.] * 6))


class Fixture:
  def __init__(self, direction=1):
    self.direction = direction
    self.speed = .44704 * 65
    self.policy = NS(model=model([direction * .03] * 6), model_ns=0, human_turn_active=False, latest_evidence=None)
    self.state = NS(vEgo=self.speed, vEgoRaw=self.speed, cruiseState=NS(enabled=True), vehicleSensorsInvalid=False,
                    steeringPressed=False, brakePressed=False, gasPressed=False, leftBlinker=False, rightBlinker=False)
    self.control = NS(latActive=True)
    self.relief = SteadyCurveRelief()
    self.yaw_dps = -direction * math.degrees(1.15 / self.speed)
    self.unsaturated = True
    self.yaw_age = self.sat_age = 0
    self.generation = 0
    self.relief.read_inputs = lambda: (self.now - self.sat_age, self.unsaturated, self.now - self.yaw_age, self.yaw_dps)
    self.now = 1_000_000_000
    self.path = direction * .05

  def step(self, *, valid=True):
    self.now += 50_000_000
    self.policy.model_ns = self.now
    self.policy.latest_evidence = dict(t=self.now, stamps=(self.now,) * 3, k=self.direction / self.speed ** 2,
                                      actual=self.direction * 3., target=self.direction * 2.,
                                      yaw=self.direction * 1.15 / self.speed, generation=self.generation) if valid else None
    result = self.relief.apply(AngleLateralResult(self.path, .00077), self.policy, self.control, self.state)
    assert result.shadow_curvature == .00077
    return result.path_angle


@pytest.mark.parametrize('direction', [-1, 1])
def test_single_count_requires_steady_history_and_preview(direction):
  f = Fixture(direction)
  for _ in range(34):
    assert f.step() == f.path
  assert [f.step() for _ in range(8)][-1] == pytest.approx(f.path - direction * .0005)
  for forecast in ([.03, .03, .032, .03, .03, .03], [.03, .03, .03, -.03, .03, .03], [.03] * 5 + [math.nan]):
    f.policy.model = model([direction * rate for rate in forecast])
    assert f.step() == f.path


@pytest.mark.parametrize('fault', ['stale_yaw', 'sensor_fault', 'stale_saturation', 'saturation', 'generation', 'no_cruise', 'bad_pinion',
                                    'uncorroborated', 'stale_policy_evidence', 'below_band', 'above_band', 'lost_model'])
def test_interrupted_evidence_requires_fresh_history(fault):
  f = Fixture()
  for _ in range(45):
    f.step()
  if fault == 'stale_yaw':
    f.yaw_age = 100_000_000
  elif fault == 'sensor_fault':
    f.yaw_dps = 4094 * .03663 - 75.
  elif fault == 'stale_saturation':
    f.sat_age = 100_000_000
  elif fault == 'saturation':
    f.unsaturated = False
  elif fault == 'generation':
    f.generation += 1
  elif fault == 'no_cruise':
    f.state.cruiseState.enabled = False
  elif fault == 'bad_pinion':
    f.state.vehicleSensorsInvalid = True
  elif fault == 'uncorroborated':
    f.yaw_dps = 0.
  elif fault == 'below_band':
    f.state.vEgo = 62.99 * .44704
  elif fault == 'above_band':
    f.state.vEgo = 66.01 * .44704
  elif fault == 'lost_model':
    f.policy.model = None
  assert f.step(valid=fault != 'stale_policy_evidence') == f.path


@pytest.mark.parametrize('direction', [-1, 1])
def test_exit_at_baseline_rate_and_inactive_neutral(direction):
  f = Fixture(direction)
  for _ in range(45):
    previous = f.step()
  f.unsaturated = False
  f.path += direction * .009
  output = f.step()
  assert output == pytest.approx(previous + direction * .009)
  assert abs(output) < abs(f.path)
  assert f.step() == f.path
  f.control.latActive = False
  f.path = 0.
  assert f.step(valid=False) == 0.

@pytest.mark.parametrize('field', ['steeringPressed', 'brakePressed', 'gasPressed', 'leftBlinker', 'rightBlinker', 'latActive', 'vEgoRaw', 'vEgo'])
def test_driver_controls_invalid_speed_and_disable_return_immediately_during_handoff(field):
  f = Fixture()
  for _ in range(45):
    previous = f.step()
  if field == 'latActive':
    f.control.latActive = False
  elif field in ('vEgo', 'vEgoRaw'):
    setattr(f.state, field, math.nan)
  else:
    setattr(f.state, field, True)
  f.path += .009
  assert f.step() == f.path
  assert not f.relief.history and not f.relief.handoff
  # Immediate return to the original path can cost one count above the
  # baseline's own full soft slew, inside the Ford integer safety allowance.
  assert round((f.path - previous) / .0005) == 19


def test_baseline_discontinuity_during_release_cannot_grow_trim():
  f = Fixture()
  for _ in range(45):
    f.step()
  f.unsaturated = False
  f.path += .025
  assert f.step() == f.path

@pytest.mark.parametrize('mph', [63., 66.])
def test_entire_history_must_be_inside_inclusive_speed_band(mph):
  f = Fixture()
  f.state.vEgo = (62.99 if mph == 63. else 66.01) * .44704
  for _ in range(45):
    assert f.step() == f.path
  f.state.vEgo = f.state.vEgoRaw = mph * .44704
  for _ in range(34):
    assert f.step() == f.path
  assert [f.step() for _ in range(10)][-1] == pytest.approx(f.path - .0005)


def test_crossing_above_band_stops_accrual_and_only_slew_limited_release_survives():
  f = Fixture()
  for _ in range(45):
    previous = f.step()
  f.state.vEgo = f.state.vEgoRaw = 66.01 * .44704
  f.path += .009
  assert f.step() == pytest.approx(previous + .009)
  assert not f.relief.history and not f.relief.demands
  assert f.step() == f.path
