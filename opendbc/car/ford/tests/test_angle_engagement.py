import unittest
from collections import defaultdict
from types import SimpleNamespace

from opendbc.car import structs
from opendbc.car.ford import fordcan
from opendbc.car.ford.carcontroller import CarController
from opendbc.car.ford.values import CAR, DBC, FordFlags
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.safety.tests.test_ford import checksum


class TestAngleEngagement(unittest.TestCase):
  def setup_controller(self, speed):
    cp = structs.CarParams(carFingerprint=CAR.FORD_EXPEDITION_MK4, flags=int(FordFlags.CANFD),
                           safetyConfigs=[{"safetyModel": "ford", "safetyParam": 3}])
    self.controller = CarController(DBC[CAR.FORD_EXPEDITION_MK4], cp)
    self.command = structs.CarControl()
    self.state = SimpleNamespace(out=structs.CarState(vEgo=speed, vEgoRaw=speed),
                                 buttons_stock_values=defaultdict(int), lkas_status_stock_values=defaultdict(int),
                                 acc_tja_status_stock_values=defaultdict(int))
    self.safety = libsafety_py.libsafety
    self.assertEqual(self.safety.set_safety_hooks(6, 3), 0)
    self.safety.init_tests()
    self.safety.set_controls_allowed(False)
    # Ford's custom angle history is not reset by set_safety_hooks. Neutralize it
    # through the real TX hook so subtests cannot inherit the previous command.
    self.assertTrue(self.tx(fordcan.create_lka_msg(self.controller.packer, self.controller.CAN, True, 0.)))
    self.assertTrue(self.tx(fordcan.create_lat_ctl2_msg(self.controller.packer, self.controller.CAN, 0, 0., 0., 0., 0., 0)))
    self.assertTrue(self.tx(fordcan.create_lka_msg(self.controller.packer, self.controller.CAN, False, 0.)))
    for counter in range(6):
      msg = self.controller.packer.make_can_msg("BrakeSysFeatures", 0, {
        "Veh_V_ActlBrk": speed * 3.6, "VehVActlBrk_D_Qf": 3, "VehVActlBrk_No_Cnt": counter,
      })
      addr, data, bus = checksum(msg)
      self.assertTrue(self.safety.safety_rx_hook(libsafety_py.make_CANPacket(addr, bus, data)))
    self.safety.set_curvature_meas(0, 0)

  def tx(self, msg):
    addr, data, bus = msg
    return self.safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, data))

  def step(self, active):
    self.command.latActive = active
    self.safety.set_controls_allowed(active)
    frame = self.controller.frame
    self.safety.set_timer(frame * 10000)
    _, messages = self.controller.update(self.command.as_reader(), self.state, frame * 10000000)
    for msg in messages:
      if msg[0] in (0x3CA, 0x3D6):
        self.assertTrue(self.tx(msg), f"rejected {msg[0]:#x} at frame {frame}, active={active}")
    return messages

  def test_first_and_following_requests_pass_at_every_lka_phase(self):
    for speed in (10., 20., 30.):
      for curvature in (-0.001, 0.001):
        for engagement_frame in (15, 20, 25):
          with self.subTest(speed=speed, curvature=curvature, frame=engagement_frame):
            self.setup_controller(speed)
            self.command.actuators.curvature = curvature
            for frame in range(engagement_frame + 16):
              self.step(frame >= engagement_frame)

  def test_reengagement_and_driver_takeover(self):
    self.setup_controller(20.)
    self.command.actuators.curvature = 0.001
    for frame in range(110):
      self.state.out.steeringPressed = 30 <= frame < 75
      self.state.out.steeringAngleDeg = 60. if self.state.out.steeringPressed else 0.
      self.step(5 <= frame < 85 or frame >= 95)

  def test_high_speed_reengagement_resets_angle_history(self):
    for curvature in (-0.001, 0.001):
      for off_frame in (25, 30, 35):
        for inactive_frames in (10, 15, 20):
          with self.subTest(curvature=curvature, off_frame=off_frame, inactive_frames=inactive_frames):
            self.setup_controller(30.)
            self.command.actuators.curvature = curvature
            for frame in range(off_frame + inactive_frames + 16):
              self.step(5 <= frame < off_frame or frame >= off_frame + inactive_frames)

  def test_steady_lka_cadence_is_preserved(self):
    self.setup_controller(20.)
    for frame in range(90):
      messages = self.step(frame >= 5)
      if frame >= 15:
        self.assertEqual(sum(msg[0] == 0x3CA for msg in messages), int(frame % 3 == 0))

  def test_invalid_requests_still_rejected(self):
    self.setup_controller(30.)
    self.assertTrue(self.tx(fordcan.create_lka_msg(self.controller.packer, self.controller.CAN, True, 0.)))
    request = fordcan.create_lat_ctl2_msg(self.controller.packer, self.controller.CAN, 1, 0., 0.3, 0., 0., 0)
    self.safety.set_controls_allowed(True)
    self.assertFalse(self.tx(request))  # Excessive change from the previous neutral angle.
    self.safety.set_controls_allowed(False)
    self.assertFalse(self.tx(request))  # An active request without permission.
