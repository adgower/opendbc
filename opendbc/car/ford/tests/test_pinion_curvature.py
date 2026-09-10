import math
import unittest
from types import SimpleNamespace

from opendbc.car import structs
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.lateral_angle import LateralAngle
from opendbc.car.ford.values import CAR
from opendbc.car.vehicle_model import VehicleModel
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.safety.tests.common import CANPackerSafety
from opendbc.safety.tests.test_ford import checksum


class TestPinionCurvature(unittest.TestCase):
  def setUp(self):
    self.cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(6, 10)
    self.safety.init_tests()
    self.packer = CANPackerSafety('ford_lincoln_base_pt')

  def rx(self, name, values, bus=0):
    return self.safety.safety_rx_hook(self.packer.make_can_msg_safety(name, bus, values, fix_checksum=checksum))

  def pinion(self, angle=30., counter=0, quality=3, bus=0):
    return self.rx('SteeringPinion_Data', {'StePinComp_An_Est': angle, 'StePinAn_No_Cnt': counter,
                                        'StePinCompAnEst_D_Qf': quality}, bus)

  def steer(self, active=True):
    values = {'LatCtl_D2_Rq': int(active), 'LatCtlCurv_No_Actl': 0., 'LatCtlCrv_NoRate2_Actl': 0.,
              'LatCtlPath_An_Actl': 0., 'LatCtlPathOffst_L_Actl': 0.}
    return self.safety.safety_tx_hook(self.packer.make_can_msg_safety('LateralMotionControl2', 0, values))

  def test_platform_selection(self):
    for car in CAR:
      cp = CarInterface.get_non_essential_params(car)
      expected = car == CAR.FORD_EXPEDITION_MK4
      self.assertEqual(bool(cp.flags & 2), expected, car)
      self.assertEqual(bool(cp.safetyConfigs[-1].safetyParam & 8), expected, car)

  def test_python_and_firmware_measure_same_fixed_model(self):
    lateral = LateralAngle(self.cp)
    vm = VehicleModel(self.cp)
    for speed in (0., 5., 10., 20., 30., 40.):
      for angle in (-1599., -90., -10., 0., 10., 90., 1599.):
        for counter in range(6):
          self.rx('BrakeSysFeatures', {'Veh_V_ActlBrk': speed * 3.6, 'VehVActlBrk_D_Qf': 3, 'VehVActlBrk_No_Cnt': counter})
          self.assertTrue(self.pinion(angle, counter))
        expected = vm.calc_curvature(math.radians(angle), speed, 0.)
        state = SimpleNamespace(out=structs.CarState(vEgoRaw=speed, steeringAngleDeg=angle, yawRate=2.))
        self.assertAlmostEqual(lateral.get_current_curvature(state), -expected, places=8)
        self.assertLessEqual(abs(self.safety.get_curvature_meas_min() - round(expected * 50000)), 1)
        self.assertLessEqual(abs(self.safety.get_curvature_meas_max() - round(expected * 50000)), 1)

  def test_no_active_request_before_first_valid_sample(self):
    self.safety.set_controls_allowed(True)
    self.assertFalse(self.steer())
    self.assertTrue(self.steer(False))
    self.pinion(angle=0., counter=1)
    self.assertTrue(self.steer())

  def test_bad_quality_and_counter_block_active_requests(self):
    for fault in ('quality', 'counter'):
      self.setUp()
      for counter in range(1, 8):
        self.pinion(0., counter)
      for counter in range(8, 15):
        self.pinion(0., counter if fault == 'quality' else 7, 0 if fault == 'quality' else 3)
      self.safety.set_controls_allowed(True)
      self.assertFalse(self.steer(), fault)

  def test_stale_sample_blocks_active_before_safety_tick(self):
    self.safety.set_timer(0)
    self.pinion(0., 1)
    self.safety.set_timer(1000001)
    self.safety.set_controls_allowed(True)
    self.assertFalse(self.steer())

  def test_yaw_cannot_replace_pinion_and_init_clears_history(self):
    for counter in range(1, 7):
      self.pinion(30., counter)
    previous = self.safety.get_curvature_meas_max()
    for counter in range(6):
      self.rx('Yaw_Data_FD1', {'VehYaw_W_Actl': 1., 'VehYawWActl_D_Qf': 3, 'VehRollYaw_No_Cnt': counter})
    self.assertEqual(self.safety.get_curvature_meas_max(), previous)
    self.safety.set_safety_hooks(6, 10)
    self.assertEqual(self.safety.get_curvature_meas_max(), 0)
    self.safety.set_controls_allowed(True)
    self.assertFalse(self.steer())

  def test_flag_off_retains_yaw_measurement(self):
    self.cp.flags &= ~2
    state = SimpleNamespace(out=structs.CarState(vEgoRaw=20., steeringAngleDeg=90., yawRate=0.2))
    self.assertAlmostEqual(LateralAngle(self.cp).get_current_curvature(state), -0.01)
    self.safety.set_safety_hooks(6, 2)
    self.safety.set_controls_allowed(True)
    self.assertTrue(self.steer())

  def test_wrong_bus_or_length_does_not_initialize_pinion(self):
    for bus, length in ((1, 8), (0, 7)):
      self.setUp()
      self.safety.safety_rx_hook(libsafety_py.make_CANPacket(0x7e, bus, bytes(length)))
      self.safety.set_controls_allowed(True)
      self.assertFalse(self.steer())

  def test_counter_wrap_and_invalid_quality(self):
    for counter in range(1, 40):
      self.assertTrue(self.pinion(0., counter % 16))
    for quality in range(3):
      self.assertFalse(self.pinion(0., 8, quality))

  def test_braking_and_controls_disallowed_still_reject_active(self):
    self.pinion(0., 1)
    self.safety.set_controls_allowed(False)
    self.assertFalse(self.steer())
    self.safety.set_controls_allowed(True)
    self.rx('EngBrakeData', {'BpedDrvAppl_D_Actl': 2, 'CcStat_D_Actl': 5})
    self.assertFalse(self.steer())

  def test_non_canfd_safety_ignores_pinion_flag(self):
    self.safety.set_safety_hooks(6, 9)
    for counter in range(6):
      self.rx('Yaw_Data_FD1', {'VehYaw_W_Actl': 1., 'VehYawWActl_D_Qf': 3, 'VehRollYaw_No_Cnt': counter})
    self.assertGreater(self.safety.get_curvature_meas_max(), 0)

  def test_model_keeps_initial_geometry_and_raw_yaw(self):
    lateral = LateralAngle(self.cp)
    state = SimpleNamespace(out=structs.CarState(vEgoRaw=20., steeringAngleDeg=30., yawRate=0.2))
    raw_yaw = state.out.yawRate
    expected = lateral.get_current_curvature(state)
    self.cp.steerRatio = 5.
    self.cp.tireStiffnessFront *= 2.
    self.assertEqual(lateral.get_current_curvature(state), expected)
    self.assertEqual(state.out.yawRate, raw_yaw)

  def test_shadow_curvature_error_band_is_not_widened(self):
    from opendbc.car import Bus
    from opendbc.car.ford import fordcan
    from opendbc.car.ford.values import DBC
    from opendbc.can import CANPacker
    packer = CANPacker(DBC[CAR.FORD_EXPEDITION_MK4][Bus.pt])
    can = fordcan.CanBus(self.cp)
    for counter in range(6):
      self.rx('BrakeSysFeatures', {'Veh_V_ActlBrk': 72., 'VehVActlBrk_D_Qf': 3, 'VehVActlBrk_No_Cnt': counter})
      self.pinion(0., counter)
    self.safety.set_controls_allowed(True)
    for shadow, allowed in ((0.002, True), (-0.002, True), (0.0021, False), (-0.0021, False)):
      addr, data, bus = fordcan.create_lka_msg(packer, can, True, shadow)
      self.assertTrue(self.safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, data)))
      self.assertEqual(bool(self.steer()), allowed)
    addr, data, bus = fordcan.create_lka_msg(packer, can, False, 0.)
    self.safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, data))
