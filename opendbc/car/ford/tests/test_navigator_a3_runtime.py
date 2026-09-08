"""Controller integration: inert proposals, frozen A2 wire equality, requested lockout."""
from dataclasses import FrozenInstanceError
import math
import subprocess
from pathlib import Path
import types
from collections import defaultdict

import pytest

from opendbc.car import Bus, structs
from opendbc.car.ford.carcontroller import CarController
from opendbc.car.ford.values import CAR, FordFlags


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
  monkeypatch.delenv('NAVIGATOR_A3_MODE', raising=False)
  monkeypatch.delenv('NAVIGATOR_A3_PROFILE', raising=False)


def params(**changes):
  values = dict(brand='ford', carFingerprint=CAR.FORD_EXPEDITION_MK4, flags=int(FordFlags.CANFD),
                wheelbase=3.1115, openpilotLongitudinalControl=True)
  values.update(changes)
  return structs.CarParams(**values)


def controller(monkeypatch, mode=None, **changes):
  if mode is not None:
    monkeypatch.setenv('NAVIGATOR_A3_MODE', mode)
    monkeypatch.setenv('NAVIGATOR_A3_PROFILE', 'expedition-provisional-v1')
  return CarController({Bus.pt: 'ford_lincoln_base_pt'}, params(**changes))


def sample(frame=0):
  cc = structs.CarControl(enabled=True, latActive=True, longActive=True)
  cc.actuators.curvature = math.sin(frame / 23.) * .005
  cc.actuators.accel = math.sin(frame / 35.)
  cc.hudControl.leadDistanceBars = frame // 100 % 4 + 1
  cc.hudControl.leftLaneVisible = frame % 31 < 20
  cc.hudControl.rightLaneDepart = frame % 59 < 8
  cc.cruiseControl.cancel = frame % 71 == 0
  cc.cruiseControl.resume = frame % 41 < 5
  cc.latActive = frame % 137 > 12
  cc.longActive = frame % 171 > 15
  cc.hudControl.visualAlert = ['none', 'steerRequired', 'fcw'][frame // 60 % 3]
  out = structs.CarState(vEgoRaw=20., vEgo=20., yawRate=.005, steeringPressed=frame % 101 < 5)
  out.cruiseState.available = frame % 199 > 5
  out.cruiseState.standstill = frame % 97 < 5
  cs = types.SimpleNamespace(out=out, buttons_stock_values=defaultdict(int),
                             acc_tja_status_stock_values=defaultdict(int), lkas_status_stock_values=defaultdict(int))
  return cc.as_reader(), cs


def test_startup_default_frozen_and_no_environment_reread(monkeypatch):
  c = controller(monkeypatch)
  assert c.navigator_a3.config.mode == 'a2'
  with pytest.raises(FrozenInstanceError):
    c.navigator_a3.config.mode = 'requested'
  monkeypatch.setenv('NAVIGATOR_A3_MODE', 'requested')
  cc, cs = sample()
  c.update(cc, cs, 1_000_000_000)
  assert c.navigator_a3.config.mode == 'a2'


@pytest.mark.parametrize('mode,profile', [('unknown', None), ('shadow', None), ('requested', 'off'), ('shadow', 'bad')])
def test_bad_startup_fails(monkeypatch, mode, profile):
  monkeypatch.setenv('NAVIGATOR_A3_MODE', mode)
  if profile is not None:
    monkeypatch.setenv('NAVIGATOR_A3_PROFILE', profile)
  with pytest.raises(ValueError):
    controller(monkeypatch)


@pytest.mark.parametrize('changes', [{'brand': 'other'}, {'carFingerprint': CAR.FORD_F_150_MK14}, {'flags': 0}, {'wheelbase': 3.69}])
def test_wrong_physical_profile_rejected(monkeypatch, changes):
  with pytest.raises(ValueError):
    controller(monkeypatch, 'shadow', **changes)


@pytest.mark.parametrize('mode', ['a2', 'shadow'])
def test_all_actual_frames_equal_frozen_controller(monkeypatch, mode):
  frozen = types.ModuleType('frozen_a2_controller')
  source = subprocess.check_output(['git', 'show', '27e76255:opendbc/car/ford/carcontroller.py'], text=True, cwd=Path(__file__).parents[4])
  exec(compile(source, 'frozen_a2_controller.py', 'exec'), frozen.__dict__)
  c = controller(monkeypatch, mode)
  old = frozen.CarController({Bus.pt: 'ford_lincoln_base_pt'}, params())
  for frame in range(1000):
    cc, cs = sample(frame)
    now = 1_000_000_000 + frame * 10_000_000
    c.set_navigator_a3_evidence(now, True, now, True, None)
    actual, sends = c.update(cc, cs, now)
    expected, original = old.update(cc, cs, now)
    assert sends == original, frame
    assert actual.to_dict() == expected.to_dict(), frame


def test_requested_never_falls_back_to_active_a2(monkeypatch):
  c = controller(monkeypatch, 'requested')
  assert c.navigator_a3.fault_reason == 'physical_enforcement_unvalidated'
  cc, cs = sample(25)
  for frame in range(20):
    now = 1_000_000_000 + frame * 10_000_000
    c.set_navigator_a3_evidence(now, True, now, True, None)
    actuators, sends = c.update(cc, cs, now)
    for address, data, _ in sends:
      if address == 0x3d6:
        assert data[0] >> 4 & 7 == 0
        assert ((data[2] << 3) | data[3] >> 5) == 1000
    assert actuators.curvature == 0.
    assert cc.latActive  # do not mutate caller's control packet


def test_shadow_diagnostics_evidence_age_and_actual_bus(monkeypatch):
  c = controller(monkeypatch, 'shadow')
  c.CAN.offset = 4
  cc, cs = sample(25)
  c.set_navigator_a3_evidence(950_000_000, True, 900_000_000, True, None)
  _, sends = c.update(cc, cs, 1_000_000_000)
  d = c.navigator_a3.diagnostic
  assert d['input']['source_ns'] == 950_000_000
  assert d['output']['measurement_age_ns'] == 100_000_000
  assert d['output']['requested_gain'] > 0
  assert d['proposed_frame']['bus'] == 4
  assert d['actual_frame']['bus'] == 4
  assert d['actual_frame']['data'] == next(data.hex() for addr, data, bus in sends if addr == 0x3d6)
  assert d['proposed_frame']['data'] != d['actual_frame']['data']
  assert not d['output']['transmission_allowed']


@pytest.mark.parametrize('source_valid,measurement_valid,reason', [(False, True, 'invalid'), (True, False, 'measurement_invalid')])
def test_invalid_evidence_resets_candidate(monkeypatch, source_valid, measurement_valid, reason):
  c = controller(monkeypatch, 'shadow')
  cc, cs = sample(25)
  c.set_navigator_a3_evidence(1_000_000_000, source_valid, 1_000_000_000, measurement_valid, None)
  c.update(cc, cs, 1_000_000_000)
  assert c.navigator_a3.diagnostic['output']['reason'] == reason
  assert c.navigator_a3.state.path_angle_rad == 0.


@pytest.mark.parametrize('change,reason', [('missing', 'invalid'), ('stale_source', 'stale'), ('future_source', 'stale'),
                                         ('stale_measurement', 'measurement_stale'), ('observer_fault', 'invalid'),
                                         ('driver', 'driver_override'), ('inactive', 'inactive')])
def test_candidate_lifecycle_neutralizes_without_changing_shadow_frames(monkeypatch, change, reason):
  c = controller(monkeypatch, 'shadow')
  baseline = controller(monkeypatch, 'a2')
  cc, cs = sample(25)
  if change == 'driver':
    cs.out.steeringPressed = True
  if change == 'inactive':
    cc = cc.as_builder()
    cc.latActive = False
    cc = cc.as_reader()
  source = 1_000_000_000
  measurement = source
  if change == 'stale_source':
    source -= 100_000_001
  elif change == 'future_source':
    source += 1
  elif change == 'stale_measurement':
    measurement -= 100_000_001
  if change != 'missing':
    c.set_navigator_a3_evidence(source, True, measurement, True, 'observer_failed' if change == 'observer_fault' else None)
  actual, sends = c.update(cc, cs, 1_000_000_000)
  expected, a2 = baseline.update(cc, cs, 1_000_000_000)
  assert sends == a2
  assert actual.to_dict() == expected.to_dict()
  assert c.navigator_a3.diagnostic['output']['reason'] == reason
  assert c.navigator_a3.diagnostic['output']['mode'] == 0


def test_diagnostics_update_only_on_lateral_cadence_and_recover_after_dropout(monkeypatch):
  c = controller(monkeypatch, 'shadow')
  cc, cs = sample(25)
  for frame in range(16):
    now = 1_000_000_000 + frame * 10_000_000
    c.set_navigator_a3_evidence(now, frame < 5 or frame >= 10, now, True, None)
    c.update(cc, cs, now)
    assert c.navigator_a3.diagnostic['input']['now_ns'] == 1_000_000_000 + frame // 5 * 50_000_000
    assert c.navigator_a3.diagnostic['output']['mode'] == (0 if 5 <= frame < 10 else 1)

@pytest.mark.parametrize('profile', list(__import__('opendbc.car.ford.navigator_a3', fromlist=['PROFILES']).PROFILES))
@pytest.mark.parametrize('sign', [-1, 1])
def test_scheduled_shadow_jitter_and_nonmonotonic_clock(monkeypatch, profile, sign):
  c = controller(monkeypatch, 'shadow')
  monkeypatch.setenv('NAVIGATOR_A3_PROFILE', profile)
  c = controller(monkeypatch)  # startup consumes the explicit profile
  cc, cs = sample(25)
  cc = cc.as_builder()
  cc.actuators.curvature = sign * .002
  cs.out.yawRate = 0.
  times = [1_000_000_000, 1_049_000_000, 1_100_000_000]
  for now in times:
    c.set_navigator_a3_evidence(now, True, now, True, None)
    c.navigator_a3.observe(cc, cs, now, c.packer, c.CAN, 0)
    assert c.navigator_a3.diagnostic['output']['mode'] == 1
  state = c.navigator_a3.state
  for now in (times[-1], times[-1] - 1):
    c.set_navigator_a3_evidence(now, True, now, True, None)
    c.navigator_a3.observe(cc, cs, now, c.packer, c.CAN, 0)
    assert c.navigator_a3.state == state
    assert c.navigator_a3.diagnostic['output']['mode'] == 0
  now = times[-1] + 100_000_001
  c.set_navigator_a3_evidence(now, True, now, True, None)
  c.navigator_a3.observe(cc, cs, now, c.packer, c.CAN, 0)
  assert c.navigator_a3.state.path_angle_rad == 0
  assert c.navigator_a3.diagnostic['output']['mode'] == 0

@pytest.mark.parametrize('mode', ['shadow', 'requested'])
def test_direct_rejection_shadow_eligibility_and_other_faults(monkeypatch, mode):
  c = controller(monkeypatch, mode)
  cc, cs = sample(25)
  c.set_navigator_a3_evidence(1_000_000_000, True, 1_000_000_000, True, 'direct_steering_rejection')
  _, sends = c.update(cc, cs, 1_000_000_000)
  d = c.navigator_a3.diagnostic
  assert d['evidence_fault_reason'] == 'direct_steering_rejection'
  assert d['calculation_eligible'] == (mode == 'shadow')
  assert d['output']['mode'] == (1 if mode == 'shadow' else 0)
  assert not d['output']['transmission_allowed']
  if mode == 'requested':
    assert all(data[0] >> 4 & 7 == 0 for address, data, _ in sends if address == 982)
  # A later independently latched fault cannot be hidden by the first evidence reason.
  c.set_navigator_a3_evidence(1_050_000_000, True, 1_050_000_000, True,
                            'direct_steering_rejection', 'configuration_mismatch')
  c.navigator_a3.observe(cc, cs, 1_050_000_000, c.packer, c.CAN, 1)
  assert not c.navigator_a3.diagnostic['calculation_eligible']
  assert c.navigator_a3.diagnostic['output']['mode'] == 0
