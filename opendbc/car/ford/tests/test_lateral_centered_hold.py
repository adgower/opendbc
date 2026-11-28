import math
from types import SimpleNamespace as NS

import pytest

from opendbc.car.ford.lateral_angle import AngleLateralResult
from opendbc.car.ford.lateral_angle_centered_hold import CenteredCommandHold, straight_preview


def model():
  line = lambda y: NS(y=[y])
  return NS(laneLines=[line(-4.), line(-1.8), line(1.8), line(4.)], laneLineProbs=[0., 1., 1., 0.], meta=NS(laneChangeState='off'),
            orientationRate=NS(t=[0., .15, .4, .75, 1., 1.5], z=[0.] * 6), velocity=NS(t=[0., .15, .4, .75, 1., 1.5], x=[36.] * 6))


class Fixture:
  def __init__(self):
    self.now = 1_000_000_000
    self.raw = NS(path_angle_last=0.)
    self.policy = NS(human_turn_active=False, model=model(), latest_evidence=None, generation=1,
                     unchanged=NS(stock=NS(upper_reference=NS(reference=self.raw))))
    self.state = NS(vEgo=36., vEgoRaw=36., cruiseState=NS(enabled=True), vehicleSensorsInvalid=False, steeringPressed=False,
                    brakePressed=False, gasPressed=False, leftBlinker=False, rightBlinker=False)
    self.control = NS(latActive=True)
    self.sat_age = self.yaw_age = self.pin_age = self.ev_age = 0
    self.gyro_multiplier = 1.
    self.sat = False
    self.ford_override = None
    self.pin_quality = 3
    self.parser = NS(ts_nanos={}, vl={})
    self.installed = NS(apply=lambda command, *_: command, read_inputs=self.read)
    self.hold = CenteredCommandHold(self.installed, self.parser)

  def read(self):
    return self.now - self.sat_age, not self.sat, self.now - self.yaw_age, self.ford if self.ford_override is None else self.ford_override

  def step(self, i, reserved=0., forced_sign=None):
    self.now += 50_000_000
    sign = (-1 if (i // 6) % 2 else 1) if forced_sign is None else forced_sign
    self.ford = -math.degrees(sign * .20 / self.state.vEgo)
    self.parser.ts_nanos['SteeringPinion_Data'] = {'StePinComp_An_Est': self.now - self.pin_age}
    self.parser.vl['SteeringPinion_Data'] = {'StePinCompAnEst_D_Qf': self.pin_quality}
    self.policy.latest_evidence = dict(t=self.now, k=sign*.09/self.state.vEgo**2, yaw=sign*self.gyro_multiplier*.20/self.state.vEgo,
                                      generation=self.policy.generation, stamps=(self.now-self.ev_age, self.now, self.now))
    self.policy.model_ns = self.now
    cmd = AngleLateralResult(sign*.003, .0008)
    self.raw.path_angle_last = cmd.path_angle + reserved
    result = self.hold.apply(cmd, self.policy, self.control, self.state)
    assert result.shadow_curvature == cmd.shadow_curvature
    assert cmd.path_angle * result.path_angle >= 0
    assert abs(result.path_angle) <= abs(cmd.path_angle)
    assert abs(result.path_angle - cmd.path_angle) <= .0005000001
    return result.path_angle != cmd.path_angle


def test_alternating_requires_history_and_preserves_constant_command():
  f = Fixture()
  assert not any(f.step(i) for i in range(40))
  assert any([f.step(i) for i in range(40,47)])
  # After the first of a run of identical baseline counts there is no spontaneous new movement.
  path = f.hold.previous
  f.step(46)
  assert f.hold.previous == path


@pytest.mark.parametrize('field,value', [
  ('laneLineProbs', []), ('laneLineProbs', [0., float('nan'), 1., 0.]), ('laneLineProbs', [0., .9, float('nan'), 0.]),
  ('laneLineProbs', [0., .8, .99, 0.]), ('laneLineProbs', [0., 1.01, .99, 0.]), ('laneLines', []),
  ('orientationRate.t', [0., .1, .3]), ('orientationRate.t', [0., .2, float('nan'), .8, 1., 2.]),
  ('orientationRate.t', [0., .2, .7, .5, 1., 2.]), ('orientationRate.t', [0., .2, .4, .5, .65, .7]),
  ('orientationRate.t', [.01, .2, .4, .5, 1., 2.]), ('orientationRate.t', [0., .2, .4, .5, 1., float('inf')]),
  ('orientationRate.z', [0., float('nan'), 0., 0., 0., 0.]), ('velocity.x', []), ('velocity.t', [0., .1, .4, .75, 1., 1.5]), ('velocity.t', []),
  ('orientationRate.z', [0., 0., 0., 0., .1, 0.]), ('meta.laneChangeState','laneChangeStarting')])
def test_uncertain_or_curved_model(field, value):
  m=model()
  root, _, key=field.rpartition('.')
  setattr(getattr(m,root) if root else m,key,value)
  assert not straight_preview(m)


@pytest.mark.parametrize('field,value', [
  ('pin_quality',2),('pin_age',120_000_000),('yaw_age',120_000_000),('sat_age',120_000_000),('ev_age',40_000_001),
  ('sat',True),('ford_override',4094*.03663-75.),('ford_override',float('nan')),('gyro_multiplier',float('nan')),
  ('state.vEgo',78*.44704-.001),('state.vEgoRaw',85*.44704),('state.steeringPressed',True),('state.leftBlinker',True),
  ('state.brakePressed',True),('state.gasPressed',True),('state.vehicleSensorsInvalid',True),('policy.human_turn_active',True),
  ('control.latActive',False),('state.cruiseState.enabled',False)])
def test_interruption_releases_and_requires_new_history(field,value):
  f=Fixture()
  assert any([f.step(i) for i in range(49)])
  root=f
  for key in field.split('.')[:-1]:root=getattr(root,key)
  key=field.split('.')[-1]; old=getattr(root,key);setattr(root,key,value)
  assert not f.step(49)
  setattr(root,key,old)
  assert not any([f.step(i) for i in range(50,89)])


def test_existing_physical_trim_takes_priority_in_both_directions():
  f=Fixture()
  assert any([f.step(i) for i in range(49)])
  assert not any([f.step(i,reserved=(-1)**i * .0005) for i in range(49,70)])
  assert not any([f.step(i) for i in range(70,110)])


def test_one_direction_or_opposed_yaw_sensors_never_confirm():
  for fixed,gyro in ((1,1.),(None,-1.)):
    f=Fixture();f.gyro_multiplier=gyro
    assert not any([f.step(i,forced_sign=fixed) for i in range(100)])


def test_generation_and_sensor_timestamp_regression_clear_history():
  f=Fixture()
  assert any([f.step(i) for i in range(49)])
  f.policy.generation += 1
  assert not any([f.step(i) for i in range(49,89)])
  assert any([f.step(i) for i in range(89,99)])
  f.yaw_age=60_000_000
  assert not f.step(99)
  f.yaw_age=0
  assert not any([f.step(i) for i in range(100,140)])
