"""Bounded rising-demand admission with fresh-context and recovery guards."""

import math
from opendbc.car.ford.lateral_angle_tracking_release import ReleaseTrackingPrototype
from opendbc.car.ford.lateral_angle_bounded_tracking import BoundedTrackingPrototype
from opendbc.car.ford.values import CAR, FordFlags


class RampTrackingPrototype(ReleaseTrackingPrototype):
  MAX_RELATIVE_RAMP_PER_S = 2.0
  MAX_STEP_BACK_FRACTION = 0.02

  def __init__(self, CP, *, enabled=False):
    super().__init__(CP, enabled=enabled)
    self.admission_kind = "none"

  def reset_tracking(self):
    super().reset_tracking()
    self.admission_kind = "none"

  def correction(self, v, requested, measured, feedforward, driver_pressed):
    ctx = self.context
    self.context = None
    self.context_ok = False
    if ctx is not None:
      now, state, control, model, lane_change, model_valid = ctx
      ages = ((now - state) / 1e9, (now - control) / 1e9, (now - model) / 1e9)
      cadence_ok = self.last_time_ns is None or 40_000_000 <= now - self.last_time_ns <= 60_000_000
      self.last_time_ns = now
      self.context_ok = (
        cadence_ok and model_valid and not lane_change and all(math.isfinite(a) and 0 <= a <= limit for a, limit in zip(ages, (0.08, 0.10, 0.15)))
      )
    eligible = (
      self.context_ok
      and self.tracking_enabled
      and self.feedback_valid
      and self.speed_weight(v) > 0
      and not driver_pressed
      and self.CP.carFingerprint == CAR.FORD_EXPEDITION_MK4
      and self.CP.flags & FordFlags.PINION_CURVATURE
      and self.CP.flags & FordFlags.LOW_SPEED_GAIN
      and all(math.isfinite(x) for x in (requested, measured, feedforward))
      and abs(requested) > 0.001
      and requested * measured > 0
      and abs(measured) >= 0.1 * abs(requested)
    )
    if not eligible:
      self.reset_tracking()
      return 0.0
    if self.history and requested * self.history[-1][1] <= 0:
      self.reset_tracking()
    self.history.append((v, requested))
    self.response_history.append((now, requested, measured))
    stable = ramping = False
    if len(self.history) == self.WINDOW_FRAMES:
      vs, ks = zip(*self.history)
      signed = [k * (1 if requested > 0 else -1) for k in ks]
      span = (self.response_history[-1][0] - self.response_history[0][0]) / 1e9
      median = sorted(abs(k) for k in ks)[len(ks) // 2]
      common = span >= 0.45 and all(k * requested > 0 for k in ks) and max(vs) - min(vs) < 0.5
      stable = common and max(ks) - min(ks) <= 0.25 * median
      relative_rise = (signed[-1] - signed[0]) / (span * median)
      ramping = (
        common
        and 0 < relative_rise <= self.MAX_RELATIVE_RAMP_PER_S
        and all(
          -self.MAX_STEP_BACK_FRACTION * abs(requested) <= b - a <= self.MAX_RELATIVE_RAMP_PER_S * median * ((tb - ta) / 1e9)
          for a, b, ta, tb in zip(signed, signed[1:], [r[0] for r in self.response_history], [r[0] for r in list(self.response_history)[1:]])
        )
      )
      ratios = [m / k for _, k, m in self.response_history]
      # Release using fractional recovery. A proportional response to a rising
      # demand is not catching up just because absolute curvature is increasing.
      recovery = max(0.0, (ratios[-1] - ratios[0]) / span)
      projected = ratios[-1] + self.PREDICTION_S * recovery
      if projected >= 0.7:
        self.hold_frames = 0
        BoundedTrackingPrototype.reset_tracking(self)
        self.admission_kind = "catchup_veto"
        return 0.0
    deficit = abs(measured) < 0.7 * abs(requested)
    self.hold_frames = self.hold_frames + 1 if (stable or ramping) and deficit else 0
    if self.hold_frames < self.HOLD_FRAMES:
      BoundedTrackingPrototype.reset_tracking(self)
      self.admission_kind = "waiting"
      return 0.0
    self.admission_kind = "rising" if ramping and signed[-1] > 1.05 * signed[0] else "steady"
    BoundedTrackingPrototype.correction(self, v, requested, measured, feedforward, driver_pressed)
    if self.admission_kind == "rising":
      self.tracking_correction *= 0.5
    return self.tracking_correction
