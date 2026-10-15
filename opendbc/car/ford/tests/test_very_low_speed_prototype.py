"""Offline prototype contracts; the vehicle response is not simulated."""
import math
from types import SimpleNamespace as NS
import numpy as np
import pytest
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR,FordFlags
from opendbc.car.ford.lateral_angle import _FORD_LOW_SPEED_FACTOR_ANG,_FORD_HIGH_SPEED_FACTOR_ANG
from opendbc.car.ford.lateral_angle_low_speed_prototype import LowSpeedGainPrototype
from opendbc.car.ford.lateral_angle_very_low_speed_prototype import VeryLowSpeedGainPrototype

def cp():
  value=CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)
  value.flags |= int(FordFlags.PINION_CURVATURE|FordFlags.LOW_SPEED_GAIN)
  return value

@pytest.mark.parametrize('mph,multiplier',[(5,1.1),(15,1.1),(20,1.05),(25,1),(30,1),(45,1),(55,1)])
@pytest.mark.parametrize('curvature',[-.003,-.0004,.0004,.003])
def test_command_correction_and_fade(mph,multiplier,curvature):
  c=cp();base=LowSpeedGainPrototype(c);trial=VeryLowSpeedGainPrototype(c,enabled=True)
  v=mph*.44704
  assert math.isclose(trial.get_curvature_factor(v,curvature),base.get_curvature_factor(v,curvature)*multiplier,rel_tol=1e-12)

def test_default_off_preserves_map():
  c=cp();base=LowSpeedGainPrototype(c);trial=VeryLowSpeedGainPrototype(c)
  for mph in [0,10,15,20,24,25,45]:
    for k in [-.01,-.0007,0,.0007,.01]:
      assert trial.get_curvature_factor(mph*.44704,k)==base.get_curvature_factor(mph*.44704,k)

@pytest.mark.parametrize('missing',[FordFlags.PINION_CURVATURE,FordFlags.LOW_SPEED_GAIN])
def test_requires_existing_paired_trial(missing):
  c=cp();c.flags &= ~int(missing)
  assert VeryLowSpeedGainPrototype(c,enabled=True).get_curvature_factor(6,.003)==LowSpeedGainPrototype(c).get_curvature_factor(6,.003)

def test_other_platform_unchanged():
  c=cp();c.carFingerprint=CAR.FORD_F_150_MK14
  assert VeryLowSpeedGainPrototype(c,enabled=True).get_curvature_factor(6,.003)==LowSpeedGainPrototype(c).get_curvature_factor(6,.003)

def test_existing_bounds_inactive_and_driver_override():
  c=cp();trial=VeryLowSpeedGainPrototype(c,enabled=True)
  cs=NS(out=NS(vEgoRaw=15*.44704,yawRate=0,steeringAngleDeg=0,steeringPressed=False));control=NS(latActive=True)
  outputs=[trial.update(control,cs,NS(curvature=1)).path_angle for _ in range(30)]
  assert max(np.diff([0]+outputs))<=.055+1e-12 and max(outputs)<=.5235
  control.latActive=False
  assert trial.update(control,cs,NS(curvature=1)).path_angle==0
  control.latActive=True;cs.out.steeringPressed=True;cs.out.steeringAngleDeg=60
  for _ in range(7):result=trial.update(control,cs,NS(curvature=.003))
  assert trial.human_turn_active and result.path_angle==0

def test_owner_factors_unchanged():
  assert (_FORD_LOW_SPEED_FACTOR_ANG,_FORD_HIGH_SPEED_FACTOR_ANG)==(1.20,.97)

def test_dense_map_is_bounded_odd_and_smooth_at_fade_endpoints():
  c=cp();base=LowSpeedGainPrototype(c);trial=VeryLowSpeedGainPrototype(c,enabled=True)
  for mph in np.linspace(0,55,111):
    v=mph*.44704
    for k in np.linspace(.00001,.02,100):
      b=base.get_curvature_factor(v,k);g=trial.get_curvature_factor(v,k)
      assert b<=g<=b*1.1+1e-12
      assert g==trial.get_curvature_factor(v,-k)
      if mph>=25:assert g==b
  for mph in [15,25]:
    v=mph*.44704;eps=1e-5
    center=trial.get_curvature_factor(v,.003)
    # Small finite-difference derivative at the full-strength/faded boundaries.
    assert abs((trial.get_curvature_factor(v+eps,.003)-center)/eps)<1e-4
    assert abs((center-trial.get_curvature_factor(v-eps,.003))/eps)<1e-4

def test_high_speed_state_resets_match_after_disengagement():
  c=cp();base=LowSpeedGainPrototype(c);trial=VeryLowSpeedGainPrototype(c,enabled=True)
  cs=NS(out=NS(vEgoRaw=15*.44704,yawRate=0,steeringAngleDeg=0,steeringPressed=False))
  control=NS(latActive=True);act=NS(curvature=.004)
  for _ in range(20):base.update(control,cs,act);trial.update(control,cs,act)
  assert trial.path_angle_last>base.path_angle_last
  control.latActive=False
  base.update(control,cs,act);trial.update(control,cs,act)
  control.latActive=True;cs.out.vEgoRaw=45*.44704
  for _ in range(20):assert base.update(control,cs,act)==trial.update(control,cs,act)
