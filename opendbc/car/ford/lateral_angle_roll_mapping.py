"""Opt-in Navigator calibrated-target/roll-aware mapping trial.

Native fixed pinion model, shadow intent, 1.20/.97 and final range/slew limits
are inherited unchanged. Calibration is an explicitly timestamped input, not a
parameter or device API. Existing below25mph tracking remains the baseline.
"""

import math
from numpy import clip, interp
from opendbc.car.ford.lateral_angle import AngleLateralResult, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES, FORD_DBC_PATH_ANGLE_MIN, FORD_DBC_PATH_ANGLE_MAX
from opendbc.car.vehicle_model import VehicleModel
from opendbc.car.ford.lateral_angle_tracking_budget import BudgetedTrackingPrototype
from opendbc.car.ford.lateral_angle_low_speed_prototype import smoothstep
from opendbc.car.ford.values import CAR, FordFlags, CarControllerParams


class RollAwareMapping(BudgetedTrackingPrototype):
  MAX_DELTA_RAD = 0.01
  MAX_RELATIVE_DELTA = 0.15
  DELTA_SLEW_PER_FRAME = 0.001
  MAX_PARAM_AGE_S = 0.15
  TARGET_TOLERANCE_DEG = 0.5

  def __init__(self, CP, *, enabled=False, mapping_enabled=None):
    super().__init__(CP, enabled=enabled)
    self.mapping_enabled = bool(enabled if mapping_enabled is None else mapping_enabled)
    self.baseline = BudgetedTrackingPrototype(CP, enabled=enabled)
    self.baseline_previous = 0.0
    self.target_speed = None
    self.release_pending = False
    self.release_frames = 0
    self.final_residual = 0.0
    self.application_model = VehicleModel(CP)
    self.calibration = None
    self.map_delta = 0.0
    self.map_reason = "initial"
    self.map_base = 0.0
    self.map_target = 0.0
    self.map_weight = 0.0
    self._last_map_ns = None
    self._map_input = None
    self._previous_request = 0.0

  def set_calibration(self, *, now_ns, sample_ns, valid, roll, ratio, stiffness, offset_deg):
    self.calibration = (now_ns, sample_ns, bool(valid), roll, ratio, stiffness, offset_deg)

  @staticmethod
  def mapping_weight(v):
    mph = v / 0.44704
    return smoothstep((mph - 50) / 5) * (1 - smoothstep((mph - 65) / 5)) if math.isfinite(mph) else 0.0

  def _prepare_map(self, CC, CS, actuators):
    previous_request = self._previous_request
    self._previous_request = float(actuators.curvature)
    self._map_input = None
    self.map_target = 0.0
    self.map_weight = 0.0
    self.map_base = 0.0
    self.map_reason = "fallback"
    ctx = self.context
    cal = self.calibration
    v = float(CS.out.vEgoRaw)
    k = float(actuators.curvature)
    now = ctx[0] if ctx is not None else None
    previous = self._last_map_ns
    self._last_map_ns = now
    state = CS.out
    valid = (
      self.mapping_enabled
      and CC.latActive
      and not state.steeringPressed
      and getattr(state, "canValid", False)
      and not getattr(state, "steerFaultTemporary", False)
      and not getattr(state, "steerFaultPermanent", False)
      and self.CP.carFingerprint == CAR.FORD_EXPEDITION_MK4
      and self.CP.flags & FordFlags.PINION_CURVATURE
      and self.CP.flags & FordFlags.LOW_SPEED_GAIN
      and math.isfinite(v)
      and math.isfinite(k)
      and abs(k) > 0.0007
      and self.mapping_weight(v) > 0
      and ctx is not None
      and cal is not None
    )
    if not valid:
      self.map_delta = 0.0
      return
    _, state_ns, control_ns, model_ns, lane_change, model_valid = ctx
    ages = [(now - ns) / 1e9 for ns in [state_ns, control_ns, model_ns]]
    stamp_now, stamp, good, roll, ratio, stiffness, offset = cal
    finite = all(math.isfinite(x) for x in [roll, ratio, stiffness, offset])
    if (
      not model_valid
      or lane_change
      or any(not 0 <= age <= limit for age, limit in zip(ages, [0.08, 0.1, 0.15], strict=False))
      or (previous is not None and not 40_000_000 <= now - previous <= 60_000_000)
      or stamp_now != now
      or not good
      or not 0 <= (now - stamp) / 1e9 <= self.MAX_PARAM_AGE_S
      or not finite
      or not 10 <= ratio <= 35
      or not 0.2 <= stiffness <= 3
      or abs(offset) > 10
      or abs(roll) > 0.15
    ):
      self.map_delta = 0.0
      self.map_reason = "invalid_calibration_or_context"
      return
    target = float(getattr(actuators, "steeringAngleDeg", float("nan")))
    model_v = float(self.target_speed if self.target_speed is not None else getattr(state, "vEgo", v))
    if not math.isfinite(target) or not math.isfinite(model_v) or model_v < 0.1:
      self.map_delta = 0.0
      self.map_reason = "missing_angle_target"
      return
    self.application_model.update_params(stiffness, ratio)
    expected = math.degrees(self.application_model.get_steer_from_curvature(-k, model_v, roll)) + offset
    if abs(target - expected) > self.TARGET_TOLERANCE_DEG:
      self.map_delta = 0.0
      self.map_reason = "target_calibration_mismatch"
      return
    # Transform the already roll-aware wheel target once, using the application
    # model without roll. This is not a degree-to-path-angle wire conversion.
    tire_target = -self.application_model.calc_curvature(math.radians(target - offset), model_v, 0.0)
    current = self.get_current_curvature(CS)
    clipped = max(current - CarControllerParams.CURVATURE_ERROR, min(current + CarControllerParams.CURVATURE_ERROR, k)) if v > 9 else k
    factor = super().get_curvature_factor(v, clipped)
    base = clipped * v * factor
    weight = self.mapping_weight(v) * smoothstep((abs(k) - 0.0007) / 0.0008)
    desired = (tire_target - k) * v * factor * weight
    cap = min(self.MAX_DELTA_RAD, self.MAX_RELATIVE_DELTA * abs(base))
    desired = max(-cap, min(cap, desired))
    # An abrupt withdrawal/reversal may clear the internal term immediately;
    # inherited final rate limiting still governs the transmitted command.
    if k * previous_request < 0:
      self.map_delta = 0.0
    self.map_delta = max(self.map_delta - self.DELTA_SLEW_PER_FRAME, min(self.map_delta + self.DELTA_SLEW_PER_FRAME, desired))
    self.map_delta = max(-cap, min(cap, self.map_delta))
    self._map_input = (v, clipped)
    self.map_base = base
    self.map_target = desired
    self.map_weight = weight
    self.map_reason = "mapped"

  def update(self, CC, CS, actuators):
    self.baseline.context = self.context
    previous = self.path_angle_last
    previous_delta = self.map_delta
    previous_request = self._previous_request
    self._prepare_map(CC, CS, actuators)
    base = self.baseline.update(CC, CS, actuators)
    for name in ("tracking_correction", "ramp_extra_area", "rearm_count"):
      setattr(self, name, getattr(self.baseline, name))
    reversal = float(actuators.curvature) * previous_request < 0
    correction_reversal = previous_delta * self.map_target < 0
    invalid = self.map_reason != "mapped"
    if (invalid or reversal or correction_reversal) and abs(previous - self.baseline_previous) > 1e-12:
      self.release_pending = True
    if self.release_pending:
      self.map_delta = 0.0
      self.map_reason = "releasing"
    if not self.mapping_enabled:
      result = base.path_angle
    elif not CC.latActive or self.baseline.human_turn_active:
      result = base.path_angle
      self.map_delta = 0.0
      self.release_pending = False
    else:
      target = base.path_angle + self.map_delta
      target = float(clip(target, FORD_DBC_PATH_ANGLE_MIN, FORD_DBC_PATH_ANGLE_MAX))
      rate = float(interp(float(CS.out.vEgoRaw), _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES))
      result = float(clip(target, previous - rate, previous + rate))
    self.final_residual = result - base.path_angle
    if self.release_pending:
      self.release_frames += 1
      if abs(self.final_residual) < 1e-12:
        self.release_pending = False
    self.baseline_previous = base.path_angle
    self.path_angle_last = result
    self.kappa_cmd = base.shadow_curvature
    self.human_turn_active = self.baseline.human_turn_active
    self._map_input = None
    return AngleLateralResult(path_angle=result, shadow_curvature=base.shadow_curvature)
