"""Single-count relief during sustained Navigator wheel and yaw overtracking."""
import math
from collections import deque
from statistics import median

import numpy as np

from opendbc.car.ford.lateral_angle import AngleLateralResult, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES, interp


def opening_preview(model, direction):
  """Require same-direction curvature without tightening over 0.25–1.25 s."""
  yaw = np.asarray(model.orientationRate.z)
  velocity = np.asarray(model.velocity.x)
  horizon = np.asarray(model.orientationRate.t)
  if not (len(yaw) == len(velocity) == len(horizon) >= 5 and horizon[0] <= .25 and horizon[-1] >= 1.25
          and np.isfinite(horizon).all() and np.all(np.diff(horizon) > 0) and np.isfinite(yaw).all() and np.isfinite(velocity).all()):
    return False
  signed = direction * np.interp((.25, .5, .75, 1., 1.25), horizon, yaw / np.maximum(2., velocity))
  return bool(min(signed) > 0 and max(signed[1:]) <= 1.05 * signed[0])


def lower_quantile(ordered):
  # The linearly interpolated 15th percentile of a small sorted window.
  position = .15 * (len(ordered) - 1)
  i = int(position)
  return ordered[i] + (position - i) * (ordered[i + 1] - ordered[i])


class SteadyCurveRelief:
  def __init__(self):
    self.read_inputs = None
    self.demands = deque()
    self.history = deque()
    self.previous = None
    self.handoff = False
    self.generation = None
    self.source_stamps = ()
    self.preview_key = None
    self.preview_ok = False

  def reset_history(self):
    self.demands.clear()
    self.history.clear()

  def confirmed(self, stock_path, policy, state):
    speed = state.vEgo
    evidence = policy.latest_evidence
    if (not 63 * .44704 <= speed <= 66 * .44704 or evidence is None or
        policy.model is None or not state.cruiseState.enabled or state.vehicleSensorsInvalid or policy.human_turn_active):
      self.reset_history()
      return False

    now = evidence['t']
    sat_ns, unsaturated, yaw_ns, yaw_dps = self.read_inputs()
    stamps = (*evidence['stamps'], policy.model_ns, sat_ns, yaw_ns)
    ordered = all(0 < t <= now for t in stamps) and all(t >= old for t, old in zip(stamps, self.source_stamps, strict=False))
    if ordered:
      self.source_stamps = stamps
    direction = 1 if evidence['k'] > 0 else -1
    if (not ordered or not unsaturated or not 0 < sat_ns <= now < sat_ns + 100_000_000
        or not 0 < yaw_ns <= now < yaw_ns + 100_000_000 or not -75. <= yaw_dps <= 4093 * .03663 - 75.
        or not direction * stock_path > .015 or not evidence['k']):
      self.reset_history()
      return False
    if evidence['generation'] != self.generation or (self.demands and not 0 < now - self.demands[-1][0] <= 120_000_000):
      self.reset_history()
    self.generation = evidence['generation']
    requested = evidence['k'] * speed ** 2
    if self.demands and self.demands[-1][1] * requested <= 0:
      self.reset_history()
    self.demands.append((now, requested))
    oldest = now - 400_000_000
    while len(self.demands) > 1 and self.demands[1][0] < oldest:
      self.demands.popleft()
    if self.demands[0][0] > oldest:
      return False
    times, loads = zip(*self.demands, strict=True)
    preceding = interp(oldest, times, loads)
    demand = max(abs(requested), abs(interp(now - 200_000_000, times, loads)), abs(preceding))
    if demand <= .45 or abs(requested) >= 1.5 or direction * preceding <= 0:
      self.history.clear()
      return False
    excess = min(-direction * math.radians(yaw_dps), direction * evidence['yaw']) * speed - demand
    wheel = (evidence['actual'] - evidence['target']) * (1 if evidence['target'] > 0 else -1 if evidence['target'] < 0 else 0)
    self.history.append((now, demand, excess, wheel))
    while self.history[0][0] < now - 1_500_000_000:
      self.history.popleft()
    if (len(self.history) < 28 or now - self.history[0][0] < 1_300_000_000
        or excess < max(.035, .05 * demand) or wheel <= .5):
      return False
    _, demands, excesses, wheels = zip(*self.history, strict=True)
    if max(demands) - min(demands) > .17:
      return False
    excesses = sorted(excesses)
    wheels = sorted(wheels)
    if (median(excesses) < max(.06, .075 * demand) or lower_quantile(excesses) < .005
        or median(wheels) <= .55 or lower_quantile(wheels) <= .15):
      return False
    key = (policy.model_ns, direction)
    if self.preview_key != key:
      self.preview_key = key
      self.preview_ok = opening_preview(policy.model, direction)
    return self.preview_ok

  def apply(self, stock, policy, control, state):
    if self.read_inputs is None:
      return stock
    path = stock.path_angle
    previous = self.previous
    self.previous = path
    if (not control.latActive or policy.human_turn_active or state.steeringPressed or state.brakePressed or state.gasPressed
        or state.leftBlinker or state.rightBlinker or not math.isfinite(path)
        or not math.isfinite(state.vEgoRaw) or not math.isfinite(state.vEgo)):
      self.reset_history()
      self.handoff = False
      return stock
    trim = self.confirmed(path, policy, state)
    if previous is None or not (trim or self.handoff):
      self.handoff = False
      return stock
    rate = interp(state.vEgoRaw, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES)
    proposed = path - math.copysign(.0005, path)
    value = proposed if trim and abs(proposed - previous) <= rate + 1e-12 else path
    # If the baseline is already slewing outward at its limit, rejoin it
    # with only the remaining attenuation needed to respect the actual rate.
    if self.handoff and abs(value - previous) > rate:
      limited = previous + math.copysign(rate, value - previous)
      attenuation = math.copysign(1., path) * (path - limited)
      value = limited if path * limited > 0 and 0 < attenuation <= .0005 + 1e-12 else path
    self.handoff = value != path
    self.previous = value
    return AngleLateralResult(value, stock.shadow_curvature) if self.handoff else stock
