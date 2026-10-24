"""Actual controller contract; only IPC/physical state is synthetic."""

from collections import defaultdict
from types import SimpleNamespace as NS
import math
import pytest
from opendbc.car.ford.carcontroller import CarController


def controller_class(enabled=True):
  assert enabled
  return CarController


from opendbc.car import structs
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR, DBC
from opendbc.car.ford.longitudinal_coordination import Torque, GRADE_OFFSET


class Scene:
  def __init__(self, enabled=True):
    cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)
    cp.openpilotLongitudinalControl = True
    self.c = controller_class(enabled)(DBC[cp.carFingerprint], cp)
    self.c.longitudinal_coordination_enabled = True
    self.c.coordination_control_valid = True
    self.c.grade_map.amount = GRADE_OFFSET
    self.now = 10.0
    self.seq = 0
    self.cs = NS(
      out=structs.CarState(vEgo=18.0, vEgoRaw=18.0, canValid=True),
      buttons_stock_values=defaultdict(int),
      lkas_status_stock_values=defaultdict(int),
      acc_tja_status_stock_values=defaultdict(int),
    )

  def step(self, raw=0.3, pitch=-0.02, torque=0.0, dt=0.02, active=True, owner=True, phase="pid", speed=18.0, stale=False, age=0.0):
    self.now += dt
    self.seq += 1
    self.c.frame = (self.seq - 1) * 2
    cc = structs.CarControl(enabled=owner, longActive=active)
    cc.actuators.accel = raw
    cc.actuators.longControlState = phase
    cc.orientationNED = [0.0, pitch, 0.0]
    q = cc.navigatorLongitudinal
    q.enabled = True
    q.epoch = 1
    q.sequence = self.seq
    q.producerMonoTime = round(self.now * 1e9)
    q.poseMonoTime = round((self.now - 0.02) * 1e9)
    q.poseValid = True
    q.referenceValid = True
    self.cs.out.vEgo = self.cs.out.vEgoRaw = speed
    self.cs.longitudinal_brake_torque = Torque(self.now - (0.2 if stale else age), torque)
    out, sends = self.c.update(cc.as_reader(), self.cs, round(self.now * 1e9))
    return out, sends

  def warm(self):
    for _ in range(6):
      self.step()
    assert not self.c.release_barrier.needs


def test_single_near_zero_grade_crossing_coasts_instead_of_latching_actual_brake():
  s = Scene()
  s.warm()
  out, _ = s.step(0.30177226662635803, -0.046179983764886856)
  assert not out.navigatorLongitudinal.brake
  assert out.navigatorLongitudinal.gas == -5.0
  assert out.navigatorLongitudinal.limited
  for _ in range(7):
    out, _ = s.step(0.30191993713378906, -0.04455951973795891)
    assert not out.navigatorLongitudinal.brake and not out.navigatorLongitudinal.fault


def test_unexpected_received_torque_withholds_propulsion_even_after_prior_clearance():
  s = Scene()
  s.warm()
  out, _ = s.step(torque=156.0)
  assert out.navigatorLongitudinal.gas == -5.0
  assert not s.c.last_observer.allow
  for _ in range(52):
    out, _ = s.step(torque=156.0)
  assert out.navigatorLongitudinal.fault
  assert str(out.navigatorLongitudinal.reason) == "release_timeout_disengage"


def test_sustained_mild_negative_intent_is_not_dropped():
  s = Scene()
  s.warm()
  pitch = math.asin((-0.02 - 0.3 - GRADE_OFFSET) / 9.81)
  out, _ = s.step(0.3, pitch)
  assert not out.navigatorLongitudinal.brake
  out, _ = s.step(0.3, pitch)
  assert not out.navigatorLongitudinal.brake
  out, _ = s.step(0.3, pitch)
  assert out.navigatorLongitudinal.brake


def test_alternating_nearzero_demand_has_bounded_negative_exposure():
  s = Scene()
  s.warm()
  braked = False
  for i in range(12):
    net = -0.04 if i % 2 == 0 else 0.001
    out, _ = s.step(0.3, math.asin((net - 0.3 - GRADE_OFFSET) / 9.81))
    braked |= out.navigatorLongitudinal.brake
  assert braked


@pytest.mark.parametrize("dt", [0.041, 0.06, 0.099])
def test_long_delivery_interval_does_not_extend_pending_mild_qualification(dt):
  s = Scene()
  s.warm()
  pitch = math.asin((-0.02 - 0.3 - GRADE_OFFSET) / 9.81)
  s.step(0.3, pitch)
  out, _ = s.step(0.3, pitch, dt=dt)
  assert out.navigatorLongitudinal.brake


def test_observed_release_still_requires_three_distinct_clear_samples():
  s = Scene()
  s.warm()
  s.step(torque=156.0)
  for _ in range(20):
    out, _ = s.step(torque=156.0)
    assert out.navigatorLongitudinal.gas == -5.0 and not out.navigatorLongitudinal.fault
  for _ in range(2):
    out, _ = s.step(torque=0.0)
    assert out.navigatorLongitudinal.gas == -5.0 and not s.c.last_observer.allow
  out, _ = s.step(torque=0.0)
  assert s.c.last_observer.allow and not out.navigatorLongitudinal.fault


@pytest.mark.parametrize("mode", ["stale", "nan", "negative", "sentinel"])
def test_invalid_observed_feedback_after_clearance_is_not_permission(mode):
  s = Scene()
  s.warm()
  value = {"stale": 0.0, "nan": float("nan"), "negative": -1.0, "sentinel": 32760.0}[mode]
  out, _ = s.step(torque=value, stale=mode == "stale")
  assert out.navigatorLongitudinal.fault and out.navigatorLongitudinal.gas == -5.0


def test_reversed_fresh_torque_cannot_reuse_prior_clearance():
  s = Scene()
  s.warm()
  out, _ = s.step(torque=0.0, age=0.03)
  assert out.navigatorLongitudinal.fault
  assert str(out.navigatorLongitudinal.reason) == "reversed_release_feedback"


@pytest.mark.parametrize("net", [-0.049, -0.02, -0.001])
@pytest.mark.parametrize("dt", [0.005, 0.01, 0.019, 0.02, 0.031, 0.04])
def test_qualification_actual_delivery_and_command_exposure_are_bounded(net, dt):
  from opendbc.car.ford.longitudinal_admission import BrakeAdmission

  a = BrakeAdmission()
  now = 1.0
  a.update(now, True, False, False, 0.3, 0.1, True, Torque(now, 0.0))
  first = None
  area = 0.0
  for _ in range(100):
    now += dt
    if first is None:
      first = now
    r = a.update(now, True, True, False, 0.3, net, True, Torque(now, 0.0))
    if r.brake:
      assert now - first <= 0.080000001
      assert area < 0.002000001
      break
    area += -net * dt
  else:
    pytest.fail("sustained demand never admitted")
