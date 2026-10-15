from opendbc.car.ford.carcontroller import CarController
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR, DBC, FordFlags
from opendbc.car.ford.lateral_angle_low_speed_prototype import LowSpeedGainPrototype


def test_existing_trial_selects_new_correction():
  cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)
  cp.flags |= int(FordFlags.PINION_CURVATURE | FordFlags.LOW_SPEED_GAIN)
  controller = CarController(DBC[CAR.FORD_EXPEDITION_MK4], cp)
  old = LowSpeedGainPrototype(cp)
  for mph, extra in [(15, 1.10), (20, 1.05), (25, 1.), (45, 1.)]:
    v = mph * .44704
    assert abs(controller.lateral_angle.get_curvature_factor(v, .003) - old.get_curvature_factor(v, .003) * extra) < 1e-12
