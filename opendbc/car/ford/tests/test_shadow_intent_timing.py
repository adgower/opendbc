from types import SimpleNamespace as NS
from opendbc.car.ford.carcontroller import CarController as SynchronizedMarkerController
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR, FordFlags, DBC
from opendbc.car.ford.lateral_angle import AngleLateralResult
from opendbc.car.structs import car
from opendbc.can.parser import CANParser


def setup_controller(cls, frame=5, active=True):
  cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)
  cp.flags |= int(FordFlags.PINION_CURVATURE | FordFlags.LOW_SPEED_GAIN)
  c = cls(DBC[CAR.FORD_EXPEDITION_MK4], cp)
  c.frame = frame
  c.angle_mode_engaged = True
  c.shadow_curvature = -0.001606

  class Strategy:
    human_turn_active = False

    def update(self, *args):
      return AngleLateralResult(path_angle=-0.031, shadow_curvature=-0.0019750712507235826)

  c.lateral_angle = Strategy()
  p0 = CANParser("ford_lincoln_base_pt", [("Steering_Data_FD1", 0)], 0)
  p2 = CANParser("ford_lincoln_base_pt", [("ACCDATA_3", 0), ("IPMA_Data", 0)], 2)
  state = car.CarState.new_message()
  state.vEgoRaw = 12.713889
  state.steeringAngleDeg = 15.1
  cs = NS(
    out=state, buttons_stock_values=p0.vl["Steering_Data_FD1"], acc_tja_status_stock_values=p2.vl["ACCDATA_3"], lkas_status_stock_values=p2.vl["IPMA_Data"]
  )
  cc = car.CarControl.new_message()
  cc.enabled = True
  cc.latActive = active
  cc.actuators.curvature = -0.0013682859716936946
  return (c, cc.as_reader(), cs)


def test_fresh_actual_shadow_precedes_active_command_between_regular_marker_frames():
  c, cc, cs = setup_controller(SynchronizedMarkerController)
  _, sends = c.update(cc, cs, 1000000000)
  steer = next((i for i, (addr, _, bus) in enumerate(sends) if addr == 982 and bus == 0))
  markers = [(i, d) for i, (addr, d, bus) in enumerate(sends) if addr == 970 and bus == 0]
  assert len(markers) == 1 and markers[0][0] < steer
  d = markers[0][1]
  raw = d[5] << 8 | d[6]
  assert raw == 1975 and d[4] & 1


import pytest


@pytest.mark.parametrize("frame", [0, 1, 3, 5, 10, 15])
def test_active_pinion_requests_always_have_one_current_marker_before_steering(frame):
  c, cc, cs = setup_controller(SynchronizedMarkerController, frame)
  _, messages = c.update(cc, cs, 1_000_000_000)
  steer = next((i for i, f in enumerate(messages) if f[0] == 0x3D6 and (f[1][0] >> 4) & 7), None)
  if steer is not None:
    markers = [(i, f) for i, f in enumerate(messages) if f[0] == 0x3CA and f[2] == 0]
    assert len(markers) == 1 and markers[0][0] < steer
    data = markers[0][1][1]
    assert (data[5] << 8 | data[6]) == 1975
