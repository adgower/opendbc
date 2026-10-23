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


class LongitudinalHarness:
  def __init__(self, speed=0., pitch=0., car=CAR.FORD_EXPEDITION_MK4, alpha=True):
    cp = CarInterface.get_non_essential_params(car)
    cp.openpilotLongitudinalControl = alpha
    self.controller = CarController(DBC[car], cp)
    self.state = SimpleNamespace(out=structs.CarState(vEgo=speed, vEgoRaw=speed),
                                 buttons_stock_values=defaultdict(int), lkas_status_stock_values=defaultdict(int),
                                 acc_tja_status_stock_values=defaultdict(int))
    self.command = structs.CarControl(enabled=True, longActive=True)
    self.command.orientationNED = [0., pitch, 0.]
    self.parser = CANParser('ford_lincoln_base_pt', [('ACCDATA', 50)], 0)
    self.values = None
    self.output = None
    self.acc_messages = []

  def step(self, demand, phase='pid', active=True, frames=2):
    self.command.actuators.accel = demand
    self.command.actuators.longControlState = phase
    self.command.longActive = active
    for _ in range(frames):
      now = self.controller.frame * 10_000_000
      self.output, messages = self.controller.update(self.command.as_reader(), self.state, now)
      for msg in messages:
        if msg[0] == 0x186:
          self.acc_messages.append(msg)
          self.parser.update([(now, [msg])])
          self.values = dict(self.parser.vl['ACCDATA'])
    return self.values


class TestLongitudinalCoordination(unittest.TestCase):
  def test_near_cruise_cap_releases_latched_brake(self):
    cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)
    self.assertEqual(float(CarInterface.get_pid_accel_limits(cp, 10., 10.)[1]), .2)
    h = LongitudinalHarness()
    h.step(-.2, phase='stopping', frames=100)
    values = h.step(.2, phase='pid', frames=100)
    self.assertEqual(values['AccBrkDecel_B_Rq'], 0)
    self.assertEqual(values['AccBrkPrchg_B_Rq'], 0)
    self.assertAlmostEqual(values['AccPrpl_A_Rq'], .2)
    self.assertAlmostEqual(values['AccBrkTot_A_Rq'], .2, delta=.0039)

  def test_grade_keeps_negative_net_braking_and_releases_positive_intent(self):
    h = LongitudinalHarness(pitch=math.asin(-.3 / 9.81))
    h.step(-.2, phase='stopping', frames=100)
    values = h.step(.2, frames=100)
    self.assertEqual(values['AccBrkDecel_B_Rq'], 1)
    self.assertEqual(values['AccPrpl_A_Rq'], -5)
    values = h.step(.4, frames=100)
    self.assertEqual(values['AccBrkDecel_B_Rq'], 0)
    self.assertAlmostEqual(values['AccPrpl_A_Rq'], .4)

  def test_grades_speeds_and_demand_reversals(self):
    for speed in [0., 2., 15., 33.]:
      for pitch_accel in [-.6, -.3, 0., .3, .6]:
        with self.subTest(speed=speed, pitch_accel=pitch_accel):
          h = LongitudinalHarness(speed, math.asin(pitch_accel / 9.81))
          for demand in [-.5, -.2, 0., .2, .25, .31, .5, .7, 1., 0., -.2]:
            values = h.step(demand, frames=100)
            if values['AccBrkDecel_B_Rq']:
              self.assertEqual(values['AccPrpl_A_Rq'], -5)
            if h.output.accel + pitch_accel < -.00001:
              self.assertEqual(values['AccBrkDecel_B_Rq'], 1)

  def test_stop_phase_does_not_release_and_disengagement_is_neutral(self):
    h = LongitudinalHarness()
    h.step(-.2, phase='stopping', frames=100)
    values = h.step(.2, phase='stopping', frames=100)
    self.assertEqual(values['AccBrkDecel_B_Rq'], 1)
    self.assertEqual(values['AccPrpl_A_Rq'], -5)
    values = h.step(0., active=False)
    self.assertEqual(values['AccBrkDecel_B_Rq'], 0)
    self.assertEqual(values['AccBrkPrchg_B_Rq'], 0)
    self.assertEqual(values['AccPrpl_A_Rq'], -5)
    self.assertAlmostEqual(values['AccBrkTot_A_Rq'], 0., delta=.0039)

  def test_invalid_pitch_retains_previous_brake_and_withdraws_propulsion(self):
    for pitch in [[], [0., float('nan'), 0.], [0., float('inf'), 0.]]:
      h = LongitudinalHarness()
      h.step(-.2, frames=100)
      h.command.orientationNED = pitch
      values = h.step(.7, frames=100)
      self.assertEqual(values['AccBrkDecel_B_Rq'], 1)
      self.assertEqual(values['AccPrpl_A_Rq'], -5)

  def test_invalid_demand_does_not_reapply_creep_and_disengages_neutrally(self):
    h = LongitudinalHarness()
    h.step(-.2, frames=100)
    processed = h.output.accel
    for _ in range(100):
      values = h.step(float('nan'))
      self.assertAlmostEqual(h.output.accel, processed)
      self.assertEqual(values['AccPrpl_A_Rq'], -5)
      self.assertEqual(values['AccBrkDecel_B_Rq'], 1)
    values = h.step(float('nan'), active=False)
    self.assertEqual(h.output.accel, 0.)
    self.assertEqual(values['AccBrkDecel_B_Rq'], 0)
    self.assertAlmostEqual(values['AccBrkTot_A_Rq'], 0., delta=.0039)

  def test_other_platform_and_stock_mode_preserve_behavior(self):
    for car in [CAR.FORD_EXPLORER_MK6, CAR.FORD_F_150_MK14]:
      h = LongitudinalHarness(speed=33., car=car)
      values = h.step(-.2, frames=100)
      self.assertEqual(values['AccBrkDecel_B_Rq'], 1)
      self.assertAlmostEqual(values['AccPrpl_A_Rq'], -.2)
    h = LongitudinalHarness(alpha=False)
    self.assertIsNone(h.step(-.2, frames=100))
    self.assertFalse(h.acc_messages)

  def test_native_permission_boundary_unchanged(self):
    safety = libsafety_py.libsafety
    self.assertEqual(safety.set_safety_hooks(6, 11), 0)
    safety.init_tests()
    h = LongitudinalHarness()
    h.step(-.2, frames=100)
    msg = h.acc_messages[-1]
    packet = libsafety_py.make_CANPacket(msg[0], msg[2], msg[1])
    safety.set_controls_allowed(True)
    self.assertTrue(safety.safety_tx_hook(packet))
    safety.set_controls_allowed(False)
    self.assertFalse(safety.safety_tx_hook(packet))
    h.step(0., active=False)
    msg = h.acc_messages[-1]
    self.assertTrue(safety.safety_tx_hook(libsafety_py.make_CANPacket(msg[0], msg[2], msg[1])))
