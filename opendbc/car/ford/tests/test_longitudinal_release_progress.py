from types import SimpleNamespace as NS
from collections import defaultdict
from pathlib import Path
import json
from opendbc.car import structs
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.carcontroller import CarController
from opendbc.car.ford.values import CAR, DBC

import math
from opendbc.car.ford.longitudinal_coordination import Torque
from opendbc.car.ford.longitudinal_restart import RestartClearance


def series(values):
  b = RestartClearance(True)
  rows = []
  for i, value in enumerate(values):
    t = 10.0 + i * 0.02
    r = b.update(t, True, False, 5.0, "pid", Torque(t, value), owner_active=True)
    rows.append((t, r))
  return b, rows


def test_stalled_torque_still_faults_at_one_second():
  b, r = series([500.0] * 80)
  assert next(t for t, x in r if x.fault) == 11.0 and b.fault == "release_timeout_disengage"
  assert not any(x.allow for t, x in r)


def test_early_progress_then_plateau_cannot_extend_wait():
  b, r = series([max(200.0, 1000.0 - i * 40.0) for i in range(80)])
  assert next(t for t, x in r if x.fault) == 11.0


def test_reversal_after_progress_cannot_extend_wait():
  values = [max(100.0, 1000.0 - i * 15.0) for i in range(80)]
  values[47:] = [700.0] * (80 - 47)
  b, r = series(values)
  assert next(t for t, x in r if x.fault) == 11.0


def test_even_continuing_progress_cannot_wait_indefinitely():
  b, r = series([1000.0 * math.exp(-0.9 * i * 0.02) for i in range(120)])
  assert next(t for t, x in r if x.fault) == 12.0 and not any(x.allow for t, x in r)


def test_noise_floor_cannot_earn_extra_wait():
  b, r = series([20.0 - i * 0.1 for i in range(80)])
  assert next(t for t, x in r if x.fault) == 11.0


def test_clock_stale_feedback_and_reversed_sample_keep_faults():
  for now, torque, reason in [
    (10.02, Torque(9.0, 50.0), "invalid_release_feedback"),
    (10.0, Torque(10.0, 50.0), "clock_discontinuity"),
    (10.02, Torque(9.99, 50.0), "reversed_release_feedback"),
  ]:
    b = RestartClearance(True)
    b.update(10.0, True, False, 5.0, "pid", Torque(10.0, 500.0))
    assert b.update(now, True, False, 5.0, "pid", torque).reason == reason


def test_repeated_zero_sample_is_not_three_measured_samples():
  b = RestartClearance(True)
  for i in range(5):
    assert not b.update(10.0 + i * 0.02, True, False, 5.0, "pid", Torque(10.0, 0.0)).allow


def test_restart_speed_crossing_retains_installed_preclear():
  b = RestartClearance(True)
  for i, speed in enumerate([2.8, 2.9, 3.1]):
    r = b.update(10.0 + i * 0.02, True, False, speed, "pid", Torque(10.0 + i * 0.02, 0.0), owner_active=True)
  assert r.allow and b.carried


def test_fault_does_not_rearm_on_override_but_disabled_owner_resets():
  b, r = series([500.0] * 60)
  assert b.fault
  assert b.update(11.2, False, False, 5.0, "pid", Torque(11.2, 0.0), owner_active=True).fault
  assert not b.update(11.3, False, False, 5.0, "pid", Torque(11.3, 0.0), owner_active=False).fault
  assert b.release_progress.samples == []


def test_actual_controller_retains_inactive_gas_until_recorded_release_clears():
  cases = json.loads((Path(__file__).parent / "fixtures/release_progress.json").read_text())
  for case in cases:
    cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)
    cp.flags = 15
    cp.openpilotLongitudinalControl = True
    c = CarController(DBC[cp.carFingerprint], cp)
    c.longitudinal_coordination_enabled = True
    observed = []
    for i, x in enumerate(case["inputs"]):
      now = 10 + x["now"]
      ns = round(now * 1e9)
      cc = structs.CarControl.new_message(enabled=True, longActive=True, latActive=False, orientationNED=[0.0, 0.0, 0.0])
      q = cc.navigatorLongitudinal
      q.enabled = True
      q.epoch = 7
      q.sequence = i + 1
      q.producerMonoTime = ns
      q.poseMonoTime = ns
      q.poseValid = True
      q.referenceValid = True
      cc.actuators.accel = 2.0
      cc.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
      state = structs.CarState.new_message(vEgo=x["speed"], vEgoRaw=x["speed"], aEgo=0.0, canValid=True, cruiseState={"enabled": True, "standstill": False})
      cs = NS(
        out=state,
        buttons_stock_values=defaultdict(int),
        lkas_status_stock_values=defaultdict(int),
        acc_tja_status_stock_values=defaultdict(int),
        longitudinal_brake_torque=Torque(10 + x["torque_t"], x["torque"], x["valid"]),
      )
      c.frame = i * 2
      c.coordination_control_valid = True
      output, _ = c.update(cc.as_reader(), cs, ns)
      f = output.navigatorLongitudinal
      assert not f.fault
      if x["torque"] > 4.0:
        assert f.gas == -5.0
      observed.append((x["now"], f.gas))
    first_valid = min(t for t, gas in observed if gas > -5.0)
    assert 1.5 <= first_valid < 1.8
    assert any(gas > 0 for t, gas in observed)


def replay(cls):
  outcomes = []
  for case in json.loads((Path(__file__).parent / "fixtures/release_progress.json").read_text()):
    b = cls(True)
    rows = []
    for x in case["inputs"]:
      result = b.update(10 + x["now"], True, False, x["speed"], "pid", Torque(10 + x["torque_t"], x["torque"], x["valid"]))
      rows.append((x, result))
    outcomes.append(rows)
  return outcomes


def test_recorded_maneuver_release_progress_waits_for_actual_clearance():
  for rows in replay(RestartClearance):
    assert not any(result.fault for _, result in rows)
    allowed = [sample for sample, result in rows if result.allow]
    assert allowed and allowed[0]["now"] < 2.0
    assert all(not result.allow for sample, result in rows if sample["torque"] > 4.0)
