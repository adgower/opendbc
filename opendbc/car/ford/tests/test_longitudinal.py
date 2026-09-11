import math
import unittest
from collections import defaultdict
from types import SimpleNamespace

from opendbc.can import CANParser
from opendbc.car import structs
from opendbc.car.ford.carcontroller import CarController
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR, DBC
from opendbc.safety.tests.libsafety import libsafety_py


class TestExpeditionLongitudinal(unittest.TestCase):
  def setUp(self):
    self.setup_controller()

  def setup_controller(self, car=CAR.FORD_EXPEDITION_MK4, alpha_long=True):
    cp = CarInterface.get_non_essential_params(car)
    cp.openpilotLongitudinalControl = alpha_long
    self.controller = CarController(DBC[car], cp)
    self.command = structs.CarControl(enabled=True, longActive=True)
    self.command.actuators.longControlState = "pid"
    self.command.orientationNED = [0., 0., 0.]
    self.state = SimpleNamespace(out=structs.CarState(vEgo=33., vEgoRaw=33.),
                                 buttons_stock_values=defaultdict(int), lkas_status_stock_values=defaultdict(int),
                                 acc_tja_status_stock_values=defaultdict(int))
    self.parser = CANParser("ford_lincoln_base_pt", [("ACCDATA", 50)], self.controller.CAN.main)

  def step(self, accel, pitch=0., active=True):
    self.command.actuators.accel = accel
    self.command.orientationNED = [0., pitch, 0.]
    self.command.longActive = active
    message = None
    for _ in range(2):
      now = self.controller.frame * 10_000_000
      output, messages = self.controller.update(self.command.as_reader(), self.state, now)
      for msg in messages:
        if msg[0] == 0x186:
          message = msg
          self.parser.update([(now, [msg])])
    self.assertIsNotNone(message)
    return dict(self.parser.vl["ACCDATA"]), output, message

  def test_mild_braking_withdraws_propulsion_without_changing_brake_demand(self):
    for _ in range(10):
      values, output, _ = self.step(-0.2)
    self.assertEqual(values["AccBrkDecel_B_Rq"], 1)
    self.assertEqual(values["AccBrkPrchg_B_Rq"], 1)
    self.assertAlmostEqual(values["AccBrkTot_A_Rq"], -0.2, delta=0.0039)
    self.assertEqual(values["AccPrpl_A_Rq"], -5.)
    self.assertEqual(output.gas, -5.)
    self.assertEqual(values["AccPrpl_A_Pred"], -5.)

  def test_propulsion_stays_inactive_until_pitch_adjusted_brake_release(self):
    # Negative pitch requests braking even with positive desired acceleration,
    # as observed near the second braking interval in the supplied Alpha log.
    pitch = math.asin(-0.3 / 9.81)
    values, _, _ = self.step(0.2, pitch)
    self.assertEqual(values["AccBrkDecel_B_Rq"], 1)
    self.assertEqual(values["AccPrpl_A_Rq"], -5.)
    values, _, _ = self.step(0.5, pitch)
    self.assertEqual(values["AccBrkDecel_B_Rq"], 1)
    self.assertEqual(values["AccPrpl_A_Rq"], -5.)
    values, _, _ = self.step(0.7, pitch)
    self.assertEqual(values["AccBrkDecel_B_Rq"], 0)
    self.assertEqual(values["AccBrkPrchg_B_Rq"], 0)
    self.assertAlmostEqual(values["AccPrpl_A_Rq"], 0.7)

  def test_positive_pitch_does_not_disable_propulsion_without_brake_request(self):
    values, _, _ = self.step(-0.2, math.asin(0.4 / 9.81))
    self.assertEqual(values["AccBrkDecel_B_Rq"], 0)
    self.assertAlmostEqual(values["AccPrpl_A_Rq"], -0.2)

  def test_low_speed_restart_waits_for_brake_release(self):
    self.state.out.vEgo = self.state.out.vEgoRaw = 0.
    self.command.actuators.longControlState = "stopping"
    self.step(-0.2)
    self.command.actuators.longControlState = "starting"
    for accel in (0., 0.2, 0.25):
      values, _, _ = self.step(accel)
      self.assertEqual(values["AccBrkDecel_B_Rq"], 1)
      self.assertEqual(values["AccPrpl_A_Rq"], -5.)
    self.command.actuators.longControlState = "pid"
    values, _, _ = self.step(0.31)
    self.assertEqual(values["AccBrkDecel_B_Rq"], 0)
    self.assertAlmostEqual(values["AccPrpl_A_Rq"], 0.31)

  def test_disengagement_clears_braking_and_reengagement_restores_propulsion(self):
    self.step(-0.2)
    values, _, _ = self.step(0., active=False)
    for signal in ("Cmbb_B_Enbl", "AccResumEnbl_B_Rq", "AccBrkDecel_B_Rq", "AccBrkPrchg_B_Rq"):
      self.assertEqual(values[signal], 0)
    self.assertEqual(values["AccPrpl_A_Rq"], -5.)
    self.assertAlmostEqual(values["AccBrkTot_A_Rq"], 0., delta=0.0039)
    values, _, _ = self.step(0.2)
    self.assertEqual(values["AccBrkDecel_B_Rq"], 0)
    self.assertAlmostEqual(values["AccPrpl_A_Rq"], 0.2)

  def test_stopping_starting_and_brake_limits(self):
    self.state.out.vEgo = self.state.out.vEgoRaw = 0.
    self.command.actuators.longControlState = "stopping"
    values, output, _ = self.step(-10.)
    # The existing 3.5 m/s^3 rate limit allows -0.07 on the first 50 Hz update.
    self.assertAlmostEqual(output.accel, -0.07, places=6)
    self.assertEqual(values["AccStopStat_B_Rq"], 1)
    self.assertEqual(values["AccPrpl_A_Rq"], -5.)
    for _ in range(60):
      values, output, _ = self.step(-10.)
    self.assertAlmostEqual(output.accel, -3.5)
    self.assertAlmostEqual(values["AccBrkTot_A_Rq"], -3.5, delta=0.0039)
    self.command.actuators.longControlState = "starting"
    values, output, _ = self.step(10.)
    self.assertEqual(values["AccStopStat_B_Rq"], 0)
    self.assertEqual(values["AccBrkDecel_B_Rq"], 0)
    self.assertAlmostEqual(values["AccPrpl_A_Rq"], 2.)
    self.assertAlmostEqual(output.accel, 2.)

  def test_other_fords_and_oem_acc_are_unchanged(self):
    for car in (CAR.FORD_EXPLORER_MK6, CAR.FORD_F_150_MK14):
      self.setup_controller(car)
      values, _, _ = self.step(-0.2)
      self.assertEqual(values["AccBrkDecel_B_Rq"], 1)
      self.assertAlmostEqual(values["AccPrpl_A_Rq"], -0.2)
    self.setup_controller(alpha_long=False)
    for frame in range(100):
      _, messages = self.controller.update(self.command.as_reader(), self.state, frame * 10_000_000)
      self.assertFalse(any(msg[0] == 0x186 for msg in messages))

  def test_braking_messages_pass_existing_safety_and_are_rejected_when_disallowed(self):
    safety = libsafety_py.libsafety
    self.assertEqual(safety.set_safety_hooks(6, 3), 0)
    safety.init_tests()
    values, _, msg = self.step(-0.2)
    self.assertEqual(values["AccPrpl_A_Rq"], -5.)
    packet = libsafety_py.make_CANPacket(msg[0], msg[2], msg[1])
    safety.set_controls_allowed(True)
    self.assertTrue(safety.safety_tx_hook(packet))
    safety.set_controls_allowed(False)
    self.assertFalse(safety.safety_tx_hook(packet))
    _, _, msg = self.step(0., active=False)
    self.assertTrue(safety.safety_tx_hook(libsafety_py.make_CANPacket(msg[0], msg[2], msg[1])))
