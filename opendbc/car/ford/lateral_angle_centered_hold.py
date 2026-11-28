"""One-count suppression of repeated, small Navigator steering corrections on a straight highway."""
import math
from collections import deque

from opendbc.car.ford.lateral_angle import AngleLateralResult, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES, interp


def straight_preview(model):
  if str(model.meta.laneChangeState) != 'off' or len(model.laneLines) != 4 or len(model.laneLineProbs) < 3:
    return False
  if not all(.82 <= model.laneLineProbs[i] <= 1. and len(model.laneLines[i].y) for i in (1, 2)):
    return False
  left, right = model.laneLines[1].y[0], model.laneLines[2].y[0]
  if not (2.3 < right - left < 4.5 and abs(left + right) <= .5):
    return False
  times, yaws = model.orientationRate.t, model.orientationRate.z
  speed_times, speeds = model.velocity.t, model.velocity.x
  if not (5 <= len(times) == len(yaws) == len(speed_times) == len(speeds) and times[0] <= 0. and times[-1] >= 1.):
    return False
  past = -math.inf
  horizon_checked = False
  for i, t in enumerate(times):
    if not math.isfinite(t) or t <= past or speed_times[i] != t:
      return False
    past = t
    if not horizon_checked:
      if not abs(yaws[i] * speeds[i]) < .35:
        return False
      horizon_checked = t >= 1.
  return horizon_checked


class CenteredCommandHold:
  def __init__(self, installed, parser):
    self.installed = installed
    self.parser = parser
    self.history = deque()
    self.previous = None
    self.generation = None
    self.stamps = ()
    self.preview_stamp = None
    self.preview_ok = False
    self.handoff = False

  @property
  def read_inputs(self):
    return self.installed.read_inputs

  def reset(self):
    self.history.clear()
    self.stamps = ()
    self.preview_stamp = None
    self.handoff = False

  def alternating(self, policy, state):
    ev = policy.latest_evidence
    if ev is None or policy.model is None or self.read_inputs is None:
      return False
    now = ev['t']
    sat_stamp, unsaturated, yaw_stamp, ford = self.read_inputs()
    pin_stamp = self.parser.ts_nanos.get('SteeringPinion_Data', {}).get('StePinComp_An_Est', 0)
    pin_quality = self.parser.vl.get('SteeringPinion_Data', {}).get('StePinCompAnEst_D_Qf', 0)
    stamps = (*ev['stamps'], policy.model_ns, sat_stamp, yaw_stamp, pin_stamp)
    if (len(stamps) != 7 or not unsaturated or pin_quality != 3 or not -75. <= ford <= 4093 * .03663 - 75.
        or not all(0 < stamp <= now for stamp in stamps)
        or any(now - stamp > limit for stamp, limit in zip(stamps[:3], (40_000_000, 40_000_000, 100_000_000), strict=True))
        or now - min(stamps[3:]) >= 120_000_000
        or any(a < b for a, b in zip(stamps, self.stamps, strict=False))):
      return False
    if self.generation != ev['generation'] or (self.history and not 0 < now - self.history[-1][0] < 120_000_000):
      self.reset()
    self.generation = ev['generation']
    self.stamps = stamps
    if self.preview_stamp != policy.model_ns:
      self.preview_ok = straight_preview(policy.model)
      self.preview_stamp = policy.model_ns
    request = ev['k'] * state.vEgo ** 2
    gyro_ay = ev['yaw'] * state.vEgo
    if not self.preview_ok or not abs(request) < .35 or not math.isfinite(gyro_ay):
      return False
    ford_ay = -math.radians(ford) * state.vEgo
    self.history.append((now, request, min(ford_ay, gyro_ay), max(ford_ay, gyro_ay)))
    while now - self.history[0][0] > 3_000_000_000:
      self.history.popleft()
    if now - self.history[0][0] < 2_000_000_000:
      return None
    direction = reversals = 0
    positive = negative = False
    for _, demand, low, high in self.history:
      positive |= low > .12
      negative |= high < -.12
      side = 1 if demand > .035 else -1 if demand < -.035 else 0
      if side:
        reversals += bool(direction and side != direction)
        direction = side
      if reversals >= 2 and positive and negative:
        return True
    return None

  def apply(self, stock, policy, control, state):
    result = self.installed.apply(stock, policy, control, state)
    path = result.path_angle
    previous, self.previous = self.previous, path
    if (not control.latActive or policy.human_turn_active or state.vehicleSensorsInvalid or not state.cruiseState.enabled
        or state.steeringPressed or state.brakePressed or state.gasPressed or state.leftBlinker or state.rightBlinker
        or not (78 * .44704 <= state.vEgo < 85 * .44704 and 78 * .44704 <= state.vEgoRaw < 85 * .44704)):
      self.reset()
      return result
    # This is the installed high-speed branch's own untrimmed wheel-target reference.
    # Give every earlier physical trim and its handoff exclusive use of the command.
    reference_path = policy.unchanged.stock.upper_reference.reference.path_angle_last
    if not (math.isfinite(path) and abs(path - reference_path) <= 1e-12):
      self.reset()
      return result
    eligible = self.alternating(policy, state)
    if eligible is False:
      self.reset()
      return result
    if previous is None or not (eligible or self.handoff):
      self.handoff = False
      return result

    rate = interp(state.vEgoRaw, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES)
    neutral = math.copysign(max(0., abs(path) - .0005), path)
    held = min(max(previous, min(path, neutral)), max(path, neutral))
    value = held if eligible and abs(held - previous) <= rate else path
    if self.handoff and abs(value - previous) > rate:
      limited = previous + math.copysign(rate, value - previous)
      if min(path, neutral) <= limited <= max(path, neutral):
        value = limited
    self.handoff = value != path
    self.previous = value
    return AngleLateralResult(value, result.shadow_curvature) if self.handoff else result
