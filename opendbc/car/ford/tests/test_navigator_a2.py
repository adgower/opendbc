"""Synthetic configuration tests. The 2700 kg fixture is not a vehicle specification."""
import pytest

from opendbc.car import STD_CARGO_KG, gen_empty_fingerprint, scale_rot_inertia, scale_tire_stiffness, structs
from opendbc.car.ford.values import CAR, FordFlags
from opendbc.car.ford import navigator_a2


@pytest.fixture(autouse=True)
def isolate_profile_environment(monkeypatch):
  monkeypatch.delenv('NAVIGATOR_A2_PROFILE', raising=False)
  monkeypatch.delenv('REPLAY', raising=False)


def seed():
  return structs.CarParams(brand='ford', carFingerprint=CAR.FORD_EXPEDITION_MK4,
                          flags=int(FordFlags.CANFD), wheelbase=3.69, mass=2000.,
                          steerRatio=17., tireStiffnessFactor=1.)


def test_disabled_preserves_complete_parameters_even_with_unused_mass():
  cp = seed()
  before = cp.to_dict()
  navigator_a2.apply_navigator_a2_parameters(cp, enabled=False, curb_mass_kg=None)
  assert cp.to_dict() == before


def test_wheelbase_only_changes_exactly_one_input():
  cp = seed()
  before = cp.to_dict()
  navigator_a2.apply_navigator_a2_parameters(cp, enabled=True, curb_mass_kg=None)
  after = cp.to_dict()
  assert cp.wheelbase == pytest.approx(3.1115, abs=1e-6)
  assert {key for key in after if after[key] != before[key]} == {'wheelbase'}


def test_helper_uses_pre_payload_curb_mass():
  cp = seed()
  navigator_a2.apply_navigator_a2_parameters(cp, enabled=True, curb_mass_kg=2700.)
  assert cp.mass == 2700.


@pytest.mark.parametrize('changes', [{'brand': 'tesla'}, {'carFingerprint': CAR.FORD_F_150_MK14}, {'flags': 0}])
def test_incompatible_opt_in_rejected_atomically(changes):
  cp = seed()
  for key, value in changes.items():
    setattr(cp, key, value)
  before = cp.to_dict()
  with pytest.raises(ValueError):
    navigator_a2.apply_navigator_a2_parameters(cp, enabled=True, curb_mass_kg=2700.)
  assert cp.to_dict() == before


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf'), -1., 0., True, False, '2700', [], object(), 1e100, 1e-100, 10**100, 10**1000])
def test_bad_mass_rejected_atomically(bad):
  cp = seed()
  before = cp.to_dict()
  with pytest.raises((TypeError, ValueError)):
    navigator_a2.apply_navigator_a2_parameters(cp, enabled=True, curb_mass_kg=bad)
  assert cp.to_dict() == before


@pytest.mark.parametrize('bad', [0, 1, 'true', None])
def test_enable_requires_boolean(bad):
  cp = seed()
  before = cp.to_dict()
  with pytest.raises(TypeError):
    navigator_a2.apply_navigator_a2_parameters(cp, enabled=bad, curb_mass_kg=None)
  assert cp.to_dict() == before


@pytest.mark.parametrize('value,expected', [(None, False), ('off', False), ('2023-navigator-swb-4wd', True)])
def test_startup_selector_exact_values(value, expected):
  assert navigator_a2.navigator_a2_profile_enabled(value, docs=False) is expected


@pytest.mark.parametrize('value', ['', 'on', '2023-navigator-l-4wd', 'OFF', ' 2023-navigator-swb-4wd', True])
def test_unknown_selector_rejected(value):
  with pytest.raises((TypeError, ValueError)):
    navigator_a2.navigator_a2_profile_enabled(value, docs=False)


def test_docs_suppresses_local_selector():
  assert not navigator_a2.navigator_a2_profile_enabled('2023-navigator-swb-4wd', docs=True)
  assert not navigator_a2.navigator_a2_profile_enabled('unknown', docs=True)


def public_params(candidate=CAR.FORD_EXPEDITION_MK4, *, docs=False):
  from opendbc.car.ford.interface import CarInterface
  fingerprint = gen_empty_fingerprint()
  fingerprint[0][0x5A] = 8  # same synthetic automatic-transmission evidence for both profiles
  fingerprint[2].update({0x3D6: 8, 0x186: 8})
  return CarInterface.get_params(candidate, fingerprint, [], True, False, docs)


def test_public_default_off_and_docs_are_serialized_identical(monkeypatch):
  monkeypatch.delenv('NAVIGATOR_A2_PROFILE', raising=False)
  baseline = public_params().to_dict()
  docs_baseline = public_params(docs=True).to_dict()
  monkeypatch.setenv('NAVIGATOR_A2_PROFILE', 'off')
  assert public_params().to_dict() == baseline
  monkeypatch.setenv('NAVIGATOR_A2_PROFILE', '2023-navigator-swb-4wd')
  assert public_params(docs=True).to_dict() == docs_baseline
  monkeypatch.setenv('NAVIGATOR_A2_PROFILE', 'bad')
  assert public_params(docs=True).to_dict() == docs_baseline


@pytest.mark.parametrize('curb_mass', [None, 2700.])
def test_public_profile_changes_only_physical_inputs_and_normal_derivations(monkeypatch, curb_mass):
  monkeypatch.delenv('NAVIGATOR_A2_PROFILE', raising=False)
  baseline = public_params().to_dict()
  monkeypatch.setattr(navigator_a2, 'APPROVED_CURB_MASS_KG', curb_mass)
  monkeypatch.setenv('NAVIGATOR_A2_PROFILE', '2023-navigator-swb-4wd')
  cp = public_params()
  after = cp.to_dict()
  expected_mass = (2000. if curb_mass is None else curb_mass) + STD_CARGO_KG
  assert cp.mass == pytest.approx(expected_mass)
  assert cp.wheelbase == pytest.approx(3.1115, abs=1e-6)
  assert cp.centerToFront == pytest.approx(cp.wheelbase * .44)
  assert cp.rotationalInertia == pytest.approx(scale_rot_inertia(cp.mass, cp.wheelbase))
  front, rear = scale_tire_stiffness(cp.mass, cp.wheelbase, cp.centerToFront, cp.tireStiffnessFactor)
  assert cp.tireStiffnessFront == pytest.approx(front)
  assert cp.tireStiffnessRear == pytest.approx(rear)
  changes = {key for key in after if after[key] != baseline[key]}
  assert {'wheelbase', 'centerToFront', 'rotationalInertia'} <= changes
  assert changes <= {'wheelbase', 'mass', 'centerToFront', 'rotationalInertia', 'tireStiffnessFront', 'tireStiffnessRear'}
  assert ('mass' in changes) == (curb_mass is not None)


@pytest.mark.parametrize('candidate', [CAR.FORD_F_150_MK14, CAR.FORD_ESCAPE_MK4])
def test_public_other_platform_off_unchanged_and_opt_in_rejected(monkeypatch, candidate):
  monkeypatch.delenv('NAVIGATOR_A2_PROFILE', raising=False)
  baseline = public_params(candidate).to_dict()
  monkeypatch.setenv('NAVIGATOR_A2_PROFILE', 'off')
  assert public_params(candidate).to_dict() == baseline
  monkeypatch.setenv('NAVIGATOR_A2_PROFILE', '2023-navigator-swb-4wd')
  with pytest.raises(ValueError):
    public_params(candidate)


def test_public_unknown_selector_fails(monkeypatch):
  monkeypatch.setenv('NAVIGATOR_A2_PROFILE', 'on')
  with pytest.raises(ValueError):
    public_params()


def test_replay_suppresses_today_profile_but_live_startup_applies_it(monkeypatch):
  monkeypatch.delenv('NAVIGATOR_A2_PROFILE', raising=False)
  monkeypatch.delenv('REPLAY', raising=False)
  baseline = public_params().to_dict()
  monkeypatch.setenv('NAVIGATOR_A2_PROFILE', '2023-navigator-swb-4wd')
  monkeypatch.setenv('REPLAY', '1')
  assert public_params().to_dict() == baseline
  monkeypatch.setenv('NAVIGATOR_A2_PROFILE', 'unknown')
  assert public_params().to_dict() == baseline
  monkeypatch.setenv('REPLAY', '0')
  monkeypatch.setenv('NAVIGATOR_A2_PROFILE', '2023-navigator-swb-4wd')
  assert public_params().wheelbase == pytest.approx(3.1115, abs=1e-6)
