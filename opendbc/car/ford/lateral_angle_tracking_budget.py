"""Experimental bounded ramp correction; selected by the existing opt-in only.

The command-area budget is pre-limiter, not a physical-response guarantee.
Final range/slew limits remain inherited. A quiet, settled valid dwell is the
only way to replenish a spent rising-demand budget during this strategy's life.
"""

import math
from opendbc.car.ford.lateral_angle_rising_demand import RampTrackingPrototype
from opendbc.car.ford.values import CAR, FordFlags


class BudgetedTrackingPrototype(RampTrackingPrototype):
  RAMP_EXTRA_AREA_LIMIT = 0.00025  # rad*s, before final range/slew limiting.
  REARM_DWELL_S = 1.0
  QUIET_CURVATURE = 0.0005
  QUIET_RANGE = 0.0001
  QUIET_SLOPE_PER_S = 0.0002

  def __init__(self, CP, *, enabled=False):
    super().__init__(CP, enabled=enabled)
    self.ramp_extra_area = 0.0
    self.ramp_episode = False
    self.rearm_count = 0
    self._clear_quiet()

  def _clear_quiet(self):
    self.quiet_start_ns = None
    self.quiet_last = None
    self.quiet_min = self.quiet_max = 0.0
    self.quiet_rearmed = False

  def _quiet_dwell(self, now, requested, measured, valid):
    if not valid or abs(requested) > self.QUIET_CURVATURE or abs(measured) > self.QUIET_CURVATURE:
      self._clear_quiet()
      return
    if self.quiet_rearmed:
      return
    if self.quiet_last is not None:
      previous_ns, previous = self.quiet_last
      dt = (now - previous_ns) / 1e9
      if (
        not 0.04 <= dt <= 0.06
        or abs(measured - previous) > self.QUIET_SLOPE_PER_S * dt
        or max(self.quiet_max, measured) - min(self.quiet_min, measured) > self.QUIET_RANGE
      ):
        self._clear_quiet()
        return
    if self.quiet_start_ns is None:
      self.quiet_start_ns = now
      self.quiet_min = self.quiet_max = measured
    self.quiet_min = min(self.quiet_min, measured)
    self.quiet_max = max(self.quiet_max, measured)
    self.quiet_last = (now, measured)
    if now - self.quiet_start_ns >= round(self.REARM_DWELL_S * 1e9):
      if self.ramp_episode:
        self.ramp_extra_area = 0.0
        self.ramp_episode = False
        self.rearm_count += 1
      self.quiet_rearmed = True

  def recovery_rate(self, requested, span, signed, ratios):
    fractional = super().recovery_rate(requested, span, signed, ratios)
    sign = 1 if requested > 0 else -1
    measured = max(0.0, sign * (self.response_history[-1][2] - self.response_history[0][2]) / (span * abs(requested)))
    # Small request dips alone must not impersonate actuator catch-up. Keep
    # predictive release for a sustained unwind (three declines totaling >5%).
    sustained_unwind = all(b < a for a, b in zip(signed[-4:-1], signed[-3:], strict=True)) and signed[-1] < 0.95 * signed[-4]
    recovery = fractional if sustained_unwind else min(fractional, measured)
    if ratios[-1] + self.PREDICTION_S * recovery < 0.7 <= ratios[-1] + self.PREDICTION_S * fractional:
      # Exceptions spend the existing episode budget, including subsequent
      # steady terms. Ordinary tracking resets must never replenish it.
      self.ramp_episode = True
    return recovery

  def correction(self, v, requested, measured, feedforward, driver_pressed):
    now = self.context[0] if self.context is not None else None
    previous = self.last_time_ns
    term = super().correction(v, requested, measured, feedforward, driver_pressed)
    valid = (
      self.context_ok
      and self.tracking_enabled
      and self.feedback_valid
      and not driver_pressed
      and self.speed_weight(v) > 0
      and self.CP.carFingerprint == CAR.FORD_EXPEDITION_MK4
      and self.CP.flags & FordFlags.PINION_CURVATURE
      and self.CP.flags & FordFlags.LOW_SPEED_GAIN
      and all(math.isfinite(x) for x in (requested, measured, feedforward))
    )
    self._quiet_dwell(now, requested, measured, valid)
    if not term:
      return 0.0
    if self.admission_kind == "rising":
      self.ramp_episode = True
    if self.ramp_episode:
      dt = (now - previous) / 1e9 if previous is not None else self.STEER_DT_S
      remaining = max(0.0, self.RAMP_EXTRA_AREA_LIMIT - self.ramp_extra_area)
      self.tracking_correction = math.copysign(min(abs(term), remaining / dt), term)
      self.ramp_extra_area = min(self.RAMP_EXTRA_AREA_LIMIT, self.ramp_extra_area + abs(self.tracking_correction) * dt)
      if not self.tracking_correction:
        self.admission_kind = "ramp_limit_reached"
    return self.tracking_correction

  def update(self, CC, CS, actuators):
    if not CC.latActive:
      self._clear_quiet()
    result = super().update(CC, CS, actuators)
    if self.human_turn_active:
      self._clear_quiet()
    return result
