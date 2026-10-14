import importlib.util
import unittest

from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR
from opendbc.car.vehicle_model import VehicleModel


class TestNavigatorMigration(unittest.TestCase):
  def test_navigator_geometry_on_fork_profile(self):
    cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)
    self.assertAlmostEqual(cp.wheelbase, 3.1115, places=5)
    self.assertAlmostEqual(cp.steerRatio, 19.6, places=5)
    self.assertAlmostEqual(cp.centerToFront, 3.1115 * .44, places=5)

  def test_pinion_option_is_available(self):
    self.assertIsNotNone(importlib.util.find_spec('opendbc.car.ford.pinion'))

  def test_mass_normalized_model_matches_preserved_fixed_model(self):
    cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)
    vm = VehicleModel(cp)
    wheelbase, mass = 3.1115, 2878.
    front, rear = 353037.28125, 438491.46875
    a_front, a_rear = wheelbase * .44, wheelbase * .56
    slip = mass * (front * a_front - rear * a_rear) / (wheelbase ** 2 * front * rear)
    for speed in (0., 2., 5., 10., 15., 20., 30., 40.):
      with self.subTest(speed=speed):
        expected = 1. / ((1. - slip * speed ** 2) * wheelbase)
        self.assertAlmostEqual(vm.curvature_factor(speed), expected, delta=2e-8)
