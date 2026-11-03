import math
import pytest
from opendbc.car.ford.carcontroller import CarController
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR, DBC
from opendbc.car.ford.longitudinal_coordination import BrakeReleaseProgress, Torque


def controller(car=CAR.FORD_EXPEDITION_MK4, alpha=True, coordinated=True, enabled=None):
  cp = CarInterface.get_non_essential_params(car)
  cp.openpilotLongitudinalControl = alpha
  c = CarController(DBC[car], cp)
  c.longitudinal_coordination_enabled = coordinated
  if enabled is not None:
    assert callable(getattr(c, "configure_history_release", None)), "startup history selection API is missing"
    c.configure_history_release(enabled=enabled)
  return c


def test_default_off_keeps_original_progress():
  c = controller()
  assert getattr(c, "history_release_enabled", None) is False, "history must have an explicit false startup default"
  assert type(c.release_barrier.release_progress) is BrakeReleaseProgress


@pytest.mark.parametrize("car,alpha,coordinated,want", [
  (CAR.FORD_EXPEDITION_MK4, True, True, True),
  (CAR.FORD_EXPEDITION_MK4, False, True, False),
  (CAR.FORD_EXPEDITION_MK4, True, False, False),
  (CAR.FORD_F_150_MK14, True, True, False),
])
def test_startup_selection_requires_supported_alpha_and_coordination(car, alpha, coordinated, want):
  c = controller(car, alpha, coordinated, True)
  assert c.history_release_enabled is want
  assert (type(c.release_barrier.release_progress) is BrakeReleaseProgress) is not want


def test_explicit_off_restores_original_and_post_start_cannot_change_selection():
  c = controller(enabled=True)
  c.configure_history_release(enabled=False)
  assert not c.history_release_enabled
  assert type(c.release_barrier.release_progress) is BrakeReleaseProgress
  c.frame = 1
  with pytest.raises(RuntimeError, match="startup-only"):
    c.configure_history_release(enabled=True)
  assert not c.history_release_enabled


def test_reset_and_new_owner_preserve_selected_factory():
  c = controller(enabled=True)
  cls = type(c.release_barrier.release_progress)
  assert cls is not BrakeReleaseProgress
  c.release_barrier.reset()
  assert type(c.release_barrier.release_progress) is cls
  c.release_barrier.update(10., False, False, 5., "pid", Torque(10., 0.))
  assert type(c.release_barrier.release_progress) is cls
  assert c.release_barrier.release_progress.samples == []


def test_history_never_grants_clearance_without_three_fresh_samples():
  c = controller(enabled=True)
  b = c.release_barrier
  for i, value in enumerate([8., 4., 4., 4.]):
    t = 10 + i * .02
    result = b.update(t, True, False, 5., "pid", Torque(t, value), owner_active=True)
    assert result.allow is (i == 3)
    assert not result.fault


def test_history_has_original_two_second_hard_deadline():
  b = controller(enabled=True).release_barrier
  for i in range(101):
    t = 10 + i * .02
    result = b.update(t, True, False, 5., "pid", Torque(t, 1000 * math.exp(-.9 * i * .02)), owner_active=True)
    assert not result.allow
    assert result.fault is (i == 100)


def test_actual_controller_history_wait_preserves_inactive_gas_and_observed_clearance():
  from collections import defaultdict
  from types import SimpleNamespace
  from opendbc.car import structs

  outcomes = {}
  for enabled in (False, True):
    c = controller(enabled=enabled)
    results = []
    for i in range(56):
      t = round(i * .02, 9)
      if t <= .6:
        torque = 328. - 280. * t
      elif t <= .86:
        torque = 160. - 64. * (t - .6) / .26
      else:
        torque = 96. - 8. * (t - .86) / .24
      now = 10. + t
      ns = round(now * 1e9)
      cc = structs.CarControl.new_message(enabled=True, longActive=True, latActive=False, orientationNED=[0., 0., 0.])
      q = cc.navigatorLongitudinal
      q.enabled = True
      q.epoch = 7
      q.sequence = i + 1
      q.producerMonoTime = ns
      q.poseMonoTime = ns
      q.poseValid = True
      q.referenceValid = True
      cc.actuators.accel = 2.
      cc.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
      state = structs.CarState.new_message(vEgo=5., vEgoRaw=5., aEgo=0., canValid=True,
                                          cruiseState={"enabled": True, "standstill": False})
      cs = SimpleNamespace(out=state, buttons_stock_values=defaultdict(int),
                           lkas_status_stock_values=defaultdict(int), acc_tja_status_stock_values=defaultdict(int),
                           longitudinal_brake_torque=Torque(now, torque))
      c.frame = i * 2
      c.coordination_control_valid = True
      output, _ = c.update(cc.as_reader(), cs, ns)
      f = output.navigatorLongitudinal
      assert f.gas == -5., "progress must not authorize propulsion above 4 Nm"
      results.append(bool(f.fault))
    outcomes[enabled] = results
  assert any(outcomes[False]), "fixture must expose the original recent-rate timeout"
  assert not any(outcomes[True]), "qualified historical progress should extend only the wait"
