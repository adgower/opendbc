import math

import numpy as np
import pytest

from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR
from opendbc.car.ford.lateral_angle import LateralAngle
from opendbc.car.ford.lateral_angle_low_speed_prototype import LowSpeedGainPrototype


def controller(cls, car=CAR.FORD_EXPEDITION_MK4):
  return cls(CarInterface.get_non_essential_params(car))


def test_gentle_and_tight_low_speed_gains():
  original, trial = controller(LateralAngle), controller(LowSpeedGainPrototype)
  assert trial.get_curvature_factor(25 * .44704, .0002) == pytest.approx(1.15 * original.get_curvature_factor(25 * .44704, .0002))
  assert trial.get_curvature_factor(25 * .44704, .002) == pytest.approx(1.05 * original.get_curvature_factor(25 * .44704, .002))


def test_high_speed_gain_is_exactly_unchanged():
  original, trial = controller(LateralAngle), controller(LowSpeedGainPrototype)
  for speed in np.linspace(45 * .44704, 35, 50):
    for kappa in np.linspace(-.01, .01, 101):
      assert trial.get_curvature_factor(speed, kappa) == original.get_curvature_factor(speed, kappa)


def test_other_platform_unchanged():
  original, trial = controller(LateralAngle, CAR.FORD_F_150_MK14), controller(LowSpeedGainPrototype, CAR.FORD_F_150_MK14)
  for speed in [0, 10, 15, 20]:
    for kappa in [.0001, .0008, .002]:
      assert trial.get_curvature_factor(speed, kappa) == original.get_curvature_factor(speed, kappa)


def test_monotone_odd_command_and_smooth_fade():
  trial = controller(LowSpeedGainPrototype)
  for speed in [10, 15, 18, 45 * .44704]:
    k = np.linspace(0, .003, 1001)
    command = np.array([x * trial.get_curvature_factor(speed, x) for x in k])
    assert np.all(np.diff(command) > 0)
    for x in [.0002, .0008, .002]:
      assert trial.get_curvature_factor(speed, x) == trial.get_curvature_factor(speed, -x)
  v = 45 * .44704
  for x in [.0002, .0008, .002]:
    assert math.isclose(trial.get_curvature_factor(v - 1e-6, x), trial.get_curvature_factor(v, x), abs_tol=1e-6)


def test_transition_incremental_gain_is_reduced():
  trial, original = controller(LowSpeedGainPrototype), controller(LateralAngle)
  k = np.linspace(.0004, .0014, 10001)
  for speed in [20 * .44704, 30 * .44704]:
    before = np.gradient([x * original.get_curvature_factor(speed, x) for x in k], k)
    after = np.gradient([x * trial.get_curvature_factor(speed, x) for x in k], k)
    assert after.max() < before.max()


def test_trial_does_not_weaken_existing_commands():
  original, trial = controller(LateralAngle), controller(LowSpeedGainPrototype)
  for speed in np.linspace(0., 45 * .44704, 101):
    for kappa in np.linspace(0., .002, 201):
      assert trial.get_curvature_factor(speed, kappa) >= original.get_curvature_factor(speed, kappa)


def test_inherited_limits_and_override_remain_effective():
  from types import SimpleNamespace

  trial = controller(LowSpeedGainPrototype)
  cs = SimpleNamespace(out=SimpleNamespace(vEgoRaw=12., yawRate=0., steeringPressed=False, steeringAngleDeg=0.))
  cc = SimpleNamespace(latActive=True)
  result = trial.update(cc, cs, SimpleNamespace(curvature=.1))
  assert result.shadow_curvature == pytest.approx(.002)
  assert abs(result.path_angle) <= .055

  cs.out.vEgoRaw = 8.
  last = trial.path_angle_last
  for _ in range(30):
    result = trial.update(cc, cs, SimpleNamespace(curvature=1.))
    assert -.5 <= result.path_angle <= .5235
    assert abs(result.path_angle - last) <= .055 + 1e-12
    last = result.path_angle

  cs.out.steeringPressed = True
  cs.out.steeringAngleDeg = 60.
  for _ in range(7):
    result = trial.update(cc, cs, SimpleNamespace(curvature=.002))
  assert trial.human_turn_active and result.path_angle == 0.
  cc.latActive = False
  result = trial.update(cc, cs, SimpleNamespace(curvature=.002))
  assert result.path_angle == 0. and not trial.human_turn_active


def test_deployment_selector_requires_pinion_and_is_opt_in():
  from opendbc.car import car_helpers
  from opendbc.car.ford.carcontroller import CarController
  from opendbc.car.ford.values import DBC, FordFlags
  from types import SimpleNamespace
  from unittest.mock import patch

  def get(pinion, gain):
    cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)

    class Fixture:
      @staticmethod
      def get_params(*args, **kwargs):
        return cp

      def __new__(cls, config):
        return SimpleNamespace(CP=config)

    with patch.object(car_helpers, 'fingerprint', return_value=(CAR.FORD_EXPEDITION_MK4, {}, 'TESTVIN', [], 'fixed', True)), \
         patch.dict(car_helpers.interfaces, {CAR.FORD_EXPEDITION_MK4: Fixture}):
      ci = car_helpers.get_car(None, None, None, False, False, ford_pinion_curvature=pinion, ford_low_speed_gain=gain)
    return ci.CP, CarController(DBC[CAR.FORD_EXPEDITION_MK4], ci.CP)

  for pinion, gain in [(False, False), (False, True), (True, False), (True, True)]:
    cp, cc = get(pinion, gain)
    assert bool(cp.flags & FordFlags.LOW_SPEED_GAIN) == (pinion and gain)
    assert isinstance(cc.lateral_angle, LowSpeedGainPrototype) == (pinion and gain)
    assert cp.safetyConfigs[-1].safetyParam == (10 if pinion else 2)
