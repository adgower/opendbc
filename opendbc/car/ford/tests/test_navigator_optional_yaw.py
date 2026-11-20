"""The optional Navigator data source does not add a powertrain CAN timeout."""
from opendbc.can import CANPacker
from opendbc.car import Bus
from opendbc.car.ford.carstate import CarState
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR


def test_optional_compensated_yaw_is_navigator_oem_only():
  cp = CarInterface.get_non_essential_params(CAR.LINCOLN_NAVIGATOR_MK4)
  cp.openpilotLongitudinalControl = False
  p = CarState.get_can_parsers(cp)[Bus.pt]
  assert 119 in p.addresses and p.message_states[119].ignore_alive
  # Keep the bus alive without 119; its absence alone cannot trigger a global CAN error.
  for i in range(20):
    p.update([[(i + 1) * 1_000_000_000, [(131, bytes(8), p.bus)]]])
    assert p.can_valid
  msg = CANPacker('ford_lincoln_base_pt').make_can_msg('BrakeSnData_3', p.bus, {'VehYawComp_W_Actl': 1.15})
  p.update([[21_000_000_000, [msg]]])
  assert p.ts_nanos['BrakeSnData_3']['VehYawComp_W_Actl'] == 21_000_000_000
  assert abs(p.vl['BrakeSnData_3']['VehYawComp_W_Actl'] - 1.15) < .02
  cp.openpilotLongitudinalControl = True
  assert 119 not in CarState.get_can_parsers(cp)[Bus.pt].addresses
  cp.carFingerprint = CAR.FORD_EXPEDITION_MK4
  cp.openpilotLongitudinalControl = False
  assert 119 not in CarState.get_can_parsers(cp)[Bus.pt].addresses
