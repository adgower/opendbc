"""Opt-in calibrated wheel-target trim with predictive unwind release."""

import math
import copy
from numpy import interp
from opendbc.car.ford.lateral_angle import AngleLateralResult, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES
from collections import deque
from opendbc.car.ford.lateral_angle_high_speed import HighSpeedReduction as InstalledHighSpeedReduction
from opendbc.car.ford.lateral_angle_low_speed_prototype import smoothstep


class ExtendedWheelTarget(InstalledHighSpeedReduction):
  WHEEL_DEADBAND_DEG = 0.5
  PREDICTION_S = 0.15

  def __init__(self, CP, *, enabled=False):
    super().__init__(CP, enabled=enabled)
    self.installed_baseline = InstalledHighSpeedReduction(CP, enabled=enabled)
    self.handoff_pending = False
    self.wheel_history = deque(maxlen=5)
    self.wheel_axis = 0
    self._wheel_state_highwater = 0
    self._wheel_control_highwater = 0
    self.wheel_reason = "initial"
    self.predicted_wheel_excess_deg = None

  @property
  def target_speed(self):
    return self.reference.target_speed

  @target_speed.setter
  def target_speed(self, value):
    self.reference.target_speed = value
    baseline = self.__dict__.get("installed_baseline")
    if baseline is not None:
      baseline.target_speed = value

  def set_context(self, **kw):
    super().set_context(**kw)
    self.installed_baseline.set_context(**kw)

  def set_calibration(self, **kw):
    super().set_calibration(**kw)
    self.installed_baseline.set_calibration(**kw)

  def set_motion(self, **kw):
    if kw.get("generation", 0) != self._motion_generation:
      self.wheel_history.clear()
      self.wheel_axis = 0
    super().set_motion(**kw)
    self.installed_baseline.set_motion(**kw)

  def _fraction(self, CC, CS, a):
    original = super()._fraction(CC, CS, a)
    self.predicted_wheel_excess_deg = None
    v = float(CS.out.vEgoRaw)
    blend = smoothstep((v / 0.44704 - 60) / 10) if math.isfinite(v) else 0.0
    if not self.enabled or blend <= 0:
      self.wheel_history.clear()
      self.wheel_axis = 0
      self.wheel_reason = "installed_reference"
      return original
    # Both outcomes require all inherited current wheel/body, atomic-target,
    # calibration, timing and driver checks to pass. Retained peak alone differs.
    if self.reason not in ("eligible", "unwind_rising_confirmation"):
      self.wheel_history.clear()
      self.wheel_axis = 0
      self.wheel_reason = "withdraw_" + self.reason
      return original * (1 - blend)
    now, state_ns, control_ns, *_ = self.reference.context
    if state_ns <= self._wheel_state_highwater or control_ns <= self._wheel_control_highwater or now - state_ns > 40_000_000 or now - control_ns > 40_000_000:
      self.wheel_history.clear()
      self.wheel_axis = 0
      self.wheel_reason = "duplicate_or_old_wheel_source"
      return original * (1 - blend)
    self._wheel_state_highwater = state_ns
    self._wheel_control_highwater = control_ns
    offset = self.reference.calibration[-1]
    desired = float(a.steeringAngleDeg) - offset
    actual = float(CS.out.steeringAngleDeg) - offset
    axis = 1 if desired > 0 else -1
    error = axis * (actual - desired)
    if axis != self.wheel_axis:
      self.wheel_history.clear()
    self.wheel_axis = axis
    self.wheel_history.append((state_ns, error))
    fraction = 0.0
    self.wheel_reason = "fresh_wheel_history"
    if len(self.wheel_history) == 5:
      first = self.wheel_history[0][0]
      span = (state_ns - first) / 1e9
      if 0.16 <= span <= 0.30:
        # Causal five-sample slope; prediction may release, never amplify onset.
        times = [(stamp - first) / 1e9 for stamp, _ in self.wheel_history]
        values = [value for _, value in self.wheel_history]
        mt = sum(times) / 5
        me = sum(values) / 5
        slope = sum((t - mt) * (e - me) for t, e in zip(times, values, strict=True)) / sum((t - mt) ** 2 for t in times)
        predicted = error + self.PREDICTION_S * slope
        self.predicted_wheel_excess_deg = predicted
        useful = min(error, predicted) - self.WHEEL_DEADBAND_DEG
        self.wheel_reason = "closing_or_under_target"
        if useful > 0:
          k = float(a.curvature)
          scope = smoothstep((v / 0.44704 - 60) / 10) * (1 - smoothstep((abs(k) - 0.001) / 0.0005)) * smoothstep((abs(k) * v * v - 0.25) / 0.25)
          fraction = scope * min(self.MAX_FRACTION, useful / abs(actual))
          self.wheel_reason = "confirmed_target_excess"
    result = original * (1 - blend) + fraction * blend
    if result > 0:
      self.reason = "wheel_target_eligible"
    return result

  def update(self, CC, CS, actuators):
    # Advance a separate installed-policy state every frame. New high-speed
    # onset/latch history must never become the lower-speed policy's history.
    self.installed_baseline.enabled = self.enabled
    self.installed_baseline.input_state_valid = self.input_state_valid
    installed = self.installed_baseline.update(CC, CS, actuators)
    if not self.enabled or float(CS.out.vEgoRaw) / 0.44704 <= 60:
      previous = self.path_angle_last
      reference = self.reference.update(CC, CS, actuators)
      for name, value in vars(self.installed_baseline).items():
        if name not in ("reference", "enabled"):
          setattr(self, name, copy.copy(value))
      self.wheel_history.clear()
      self.wheel_axis = 0
      self.wheel_reason = "installed_reference"
      self.predicted_wheel_excess_deg = None
      result = installed.path_angle
      self.handoff_pending = False
      if self.enabled and CC.latActive and not self.reference.human_turn_active:
        b = reference.path_angle
        cap = min(0.01, 0.15 * abs(b))
        edge = b - math.copysign(cap, b)
        rate = float(interp(float(CS.out.vEgoRaw), _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES))
        low = max(min(b, edge), previous - rate)
        high = min(max(b, edge), previous + rate)
        if low > high + 1e-12:
          self.infeasible_intersections += 1
          raise AssertionError("Installed handoff and native envelopes have no intersection")
        result = max(low, min(high, result))
        self.handoff_pending = abs(result - installed.path_angle) > 1e-12
        if self.handoff_pending:
          self.reason = "installed_handoff"
          self.release_pending = True
      self.path_angle_last = result
      self.final_residual = result - reference.path_angle
      return AngleLateralResult(path_angle=result, shadow_curvature=installed.shadow_curvature)
    self.handoff_pending = False
    return super().update(CC, CS, actuators)
