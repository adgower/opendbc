import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from opendbc.car import car_helpers
from opendbc.car.ford import lateral_angle, values
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR
from opendbc.car.vehicle_model import VehicleModel


class TestPinionOption(unittest.TestCase):
  def setUp(self):
    self.cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)

  def configure(self, cp, enabled):
    from opendbc.car.ford.pinion import configure_pinion_curvature
    return configure_pinion_curvature(cp, enabled)

  def test_owner_gains_preserved(self):
    self.assertEqual(lateral_angle._FORD_LOW_SPEED_FACTOR_ANG, 1.20)
    self.assertEqual(lateral_angle._FORD_HIGH_SPEED_FACTOR_ANG, 0.97)
    self.assertEqual(lateral_angle._GAIN_EXPEDITION, (1.425, 1.425))

  def test_default_off_keeps_flags_and_yaw(self):
    flags, safety = self.cp.flags, self.cp.safetyConfigs[-1].safetyParam
    self.configure(self.cp, False)
    self.assertEqual(self.cp.flags, flags)
    self.assertEqual(self.cp.safetyConfigs[-1].safetyParam, safety)
    cs = SimpleNamespace(out=SimpleNamespace(vEgoRaw=20., yawRate=.1, steeringAngleDeg=30.))
    self.assertAlmostEqual(lateral_angle.LateralAngle(self.cp).get_current_curvature(cs), -.005)

  def test_enabled_pairs_controller_and_safety_flags(self):
    self.configure(self.cp, True)
    self.assertTrue(self.cp.flags & values.FordFlags.PINION_CURVATURE)
    self.assertEqual(self.cp.safetyConfigs[-1].safetyParam, 10)
    vm = VehicleModel(self.cp)
    for speed in (0., 5., 10., 20., 30.):
      cs = SimpleNamespace(out=SimpleNamespace(vEgoRaw=speed, yawRate=2., steeringAngleDeg=30.))
      self.assertAlmostEqual(lateral_angle.LateralAngle(self.cp).get_current_curvature(cs),
                             -vm.calc_curvature(math.radians(30.), speed, 0.), places=8)

  def test_disable_clears_previous_option(self):
    self.configure(self.cp, True)
    self.configure(self.cp, False)
    self.assertFalse(self.cp.flags & values.FordFlags.PINION_CURVATURE)
    self.assertEqual(self.cp.safetyConfigs[-1].safetyParam, 2)

  def test_unsupported_platform_unchanged(self):
    cp = CarInterface.get_non_essential_params(CAR.FORD_EXPLORER_MK6)
    before = (cp.flags, cp.safetyConfigs[-1].safetyParam)
    self.configure(cp, True)
    self.assertEqual((cp.flags, cp.safetyConfigs[-1].safetyParam), before)

  def test_geometry_mismatch_rejected_before_flags_change(self):
    self.cp.steerRatio = 22.5
    before = (self.cp.flags, self.cp.safetyConfigs[-1].safetyParam)
    with self.assertRaisesRegex(ValueError, 'geometry'):
      self.configure(self.cp, True)
    self.assertEqual((self.cp.flags, self.cp.safetyConfigs[-1].safetyParam), before)

  def test_get_car_applies_option_before_controller_construction(self):
    cp = self.cp

    def construct(config):
      self.assertTrue(config.flags & values.FordFlags.PINION_CURVATURE)
      return SimpleNamespace(CP=config)

    class Fixture:
      @staticmethod
      def get_params(*args, **kwargs):
        return cp

      def __new__(cls, config):
        return construct(config)
    with patch.object(car_helpers, 'fingerprint', return_value=(CAR.FORD_EXPEDITION_MK4, {}, 'TESTVIN', [], 'fixed', True)), \
         patch.dict(car_helpers.interfaces, {CAR.FORD_EXPEDITION_MK4: Fixture}):
      ci = car_helpers.get_car(None, None, None, False, False, None, ford_pinion_curvature=True)
    self.assertEqual(ci.CP.safetyConfigs[-1].safetyParam, 10)

  def test_get_car_clears_option_even_with_prior_mode_cache(self):
    self.configure(self.cp, True)
    cached = self.cp.as_reader()
    fresh = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)

    class Fixture:
      @staticmethod
      def get_params(*args, **kwargs):
        return fresh

      def __new__(cls, config):
        return SimpleNamespace(CP=config)

    with patch.object(car_helpers, 'fingerprint', return_value=(CAR.FORD_EXPEDITION_MK4, {}, 'TESTVIN', [], 'fixed', True)), \
         patch.dict(car_helpers.interfaces, {CAR.FORD_EXPEDITION_MK4: Fixture}):
      ci = car_helpers.get_car(None, None, None, False, False, cached, ford_pinion_curvature=False)
    self.assertEqual(ci.CP.flags, 1)
    self.assertEqual(ci.CP.safetyConfigs[-1].safetyParam, 2)

  def test_enable_preserves_longitudinal_flag(self):
    self.cp.safetyConfigs[-1].safetyParam |= 1
    self.configure(self.cp, True)
    self.assertEqual(self.cp.safetyConfigs[-1].safetyParam, 11)


if __name__ == '__main__':
  unittest.main()
