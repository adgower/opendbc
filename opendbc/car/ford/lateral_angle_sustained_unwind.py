"""Opt-in bounded sustained tracking and independent-history unwind policy."""

import math
from collections import deque
from opendbc.car.ford.lateral_angle_tracking_budget import BudgetedTrackingPrototype
from opendbc.car.vehicle_model import VehicleModel
from opendbc.car.ford.lateral_angle_extended_wheel import ExtendedWheelTarget
from opendbc.car.ford.lateral_angle_wheel_target import CalibratedWheelTargetReduction as InstalledUpper
from opendbc.car.ford.lateral_angle import AngleLateralResult, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES
from opendbc.car.ford.lateral_angle_low_speed_prototype import smoothstep
from numpy import interp


class SustainedTracking(BudgetedTrackingPrototype):
  EXTRA_CAP_RAD = 0.0025
  EXTRA_AREA_RAD_S = 0.0025
  EXTRA_DURATION_S = 1.0

  def __init__(self, CP, *, enabled=True, extra_enabled=False):
    super().__init__(CP, enabled=enabled)
    self.extra_enabled = extra_enabled
    self.evidence = None
    self.sustained_history = deque(maxlen=5)
    self.sustained_area = 0.0
    self.sustained_start = None
    self.sustained_sign = 0
    self.sustained_aborted = False
    self.sustained_extra = 0.0
    self.sustained_reason = "initial"
    self.sustained_vm = VehicleModel(CP)
    self._sustained_highwater = (0, 0, 0)
    self._sustained_generation = None
    self._last_recovery = None
    self._clear_sustained_quiet()

  def recovery_rate(self, *args):
    self._last_recovery = super().recovery_rate(*args)
    return self._last_recovery

  def set_evidence(self, e):
    self.evidence = e
    self.sustained_extra = 0.0
    if self.sustained_start is not None and (not e["active"] or e["driver_or_fault"]):
      self.sustained_aborted = True
    if e["generation"] != self._sustained_generation:
      if self._sustained_generation is not None and self.sustained_start is not None:
        self.sustained_aborted = True
      self._sustained_generation = e["generation"]
      self.sustained_history.clear()
      self._sustained_highwater = (0, 0, 0)

  def basic_evidence(self):
    e = self.evidence
    if e is None or not e["active"] or not e["state_valid"] or e["driver_or_fault"]:
      return False
    ctx = e["context"]
    cal = e["calibration"]
    mo = e["motion"]
    if ctx is None or cal is None or mo is None:
      return False
    now, st, ct, mt, lane, model_valid = ctx
    cn, source, good, roll, ratio, stiffness, offset = cal
    mn, ms, mr, valid, yaw, std = mo
    return bool(
      model_valid
      and not lane
      and good
      and cn == now
      and mn == now
      and valid
      and 0 < ms <= mr <= now
      and now - ms <= 100_000_000
      and now - mr <= 100_000_000
      and all(0 < x <= now and now - x <= limit for x, limit in [(st, 40_000_000), (ct, 40_000_000), (mt, 150_000_000), (source, 150_000_000)])
      and all(math.isfinite(x) for x in [roll, ratio, stiffness, offset, yaw, std, e["target"], e["actual"], e["model_v"]])
      and abs(roll) <= 0.15
      and 10 <= ratio <= 35
      and 0.2 <= stiffness <= 3
      and abs(offset) <= 10
      and 0 <= std <= 0.03
      and e["model_v"] > 0.1
    )

  def observe_deficit(self, v, k):
    if not self.basic_evidence() or not self.speed_weight(v):
      self.sustained_history.clear()
      return False
    e = self.evidence
    now, st, ct, *_ = e["context"]
    _, ms, *_ = e["motion"]
    _, _, _, roll, ratio, stiffness, offset = e["calibration"]
    stamps = (st, ct, ms)
    if any(a <= b for a, b in zip(stamps, self._sustained_highwater, strict=False)):
      self.sustained_history.clear()
      return False
    self._sustained_highwater = stamps
    self.sustained_vm.update_params(stiffness, ratio)
    expected = math.degrees(self.sustained_vm.get_steer_from_curvature(-k, e["model_v"], roll)) + offset
    target = e["target"] - offset
    actual = e["actual"] - offset
    if abs(expected - e["target"]) > 0.5 or abs(target) < 5 or target * actual <= 0 or abs(k) * v * v < 0.25:
      self.sustained_history.clear()
      return False
    wr = actual / target
    br = e["motion"][4] / (k * v)
    if not (0.1 <= wr < 0.7 and 0.1 <= br < 0.7) or (abs(k) * v * v - abs(e["motion"][4] * v)) <= max(0.1, 0.1 * abs(k) * v * v):
      self.sustained_history.clear()
      return False
    axis = 1 if k > 0 else -1
    if self.sustained_history and self.sustained_history[-1][2] != axis:
      self.sustained_history.clear()
    self.sustained_history.append((st, wr, axis, ms, br))
    if len(self.sustained_history) < 5:
      return False
    t0 = self.sustained_history[0][0]
    span = (st - t0) / 1e9
    if not 0.16 <= span <= 0.30:
      return False
    tt = [(x[0] - t0) / 1e9 for x in self.sustained_history]
    rr = [x[1] for x in self.sustained_history]
    mt = sum(tt) / 5
    mr = sum(rr) / 5
    slope = sum((t - mt) * (r - mr) for t, r in zip(tt, rr, strict=False)) / sum((t - mt) ** 2 for t in tt)
    body_t0 = self.sustained_history[0][3]
    body_span = (ms - body_t0) / 1e9
    if not 0.16 <= body_span <= 0.30:
      return False
    bt = [(x[3] - body_t0) / 1e9 for x in self.sustained_history]
    brs = [x[4] for x in self.sustained_history]
    bmt = sum(bt) / 5
    bmr = sum(brs) / 5
    body_slope = sum((t - bmt) * (r - bmr) for t, r in zip(bt, brs, strict=False)) / sum((t - bmt) ** 2 for t in bt)
    return wr + 0.30 * max(0.0, slope) < 0.7 and br + 0.30 * max(0.0, body_slope) < 0.7

  def _clear_sustained_quiet(self):
    self.sustained_quiet_start = None
    self.sustained_quiet_last = None
    self.sustained_quiet_min = self.sustained_quiet_max = 0.0
    self.sustained_body_min = self.sustained_body_max = 0.0
    self.sustained_quiet_rearmed = False

  def _quiet_dwell(self, now, requested, measured, valid):
    # The installed budget keeps its exact original dwell. The new allowance
    # independently requires good wheel/body/calibration evidence at EVERY frame.
    super()._quiet_dwell(now, requested, measured, valid)
    if not valid or not self.basic_evidence() or abs(requested) > self.QUIET_CURVATURE or abs(measured) > self.QUIET_CURVATURE:
      self._clear_sustained_quiet()
      return
    body_curvature = self.evidence["motion"][4] / self.evidence["model_v"]
    if abs(body_curvature) > self.QUIET_CURVATURE:
      self._clear_sustained_quiet()
      return
    if self.sustained_quiet_rearmed:
      return
    if self.sustained_quiet_last is not None:
      previous_ns, previous, previous_body = self.sustained_quiet_last
      dt = (now - previous_ns) / 1e9
      if (
        not 0.04 <= dt <= 0.06
        or abs(measured - previous) > self.QUIET_SLOPE_PER_S * dt
        or max(self.sustained_quiet_max, measured) - min(self.sustained_quiet_min, measured) > self.QUIET_RANGE
        or abs(body_curvature - previous_body) > self.QUIET_SLOPE_PER_S * dt
        or max(self.sustained_body_max, body_curvature) - min(self.sustained_body_min, body_curvature) > self.QUIET_RANGE
      ):
        self._clear_sustained_quiet()
        return
    if self.sustained_quiet_start is None:
      self.sustained_quiet_start = now
      self.sustained_quiet_min = self.sustained_quiet_max = measured
      self.sustained_body_min = self.sustained_body_max = body_curvature
    self.sustained_quiet_min = min(self.sustained_quiet_min, measured)
    self.sustained_quiet_max = max(self.sustained_quiet_max, measured)
    self.sustained_body_min = min(self.sustained_body_min, body_curvature)
    self.sustained_body_max = max(self.sustained_body_max, body_curvature)
    self.sustained_quiet_last = (now, measured, body_curvature)
    if now - self.sustained_quiet_start >= round(self.REARM_DWELL_S * 1e9):
      self.sustained_area = 0.0
      self.sustained_start = None
      self.sustained_sign = 0
      self.sustained_aborted = False
      self.sustained_quiet_rearmed = True

  def correction(self, v, requested, measured, feedforward, driver_pressed):
    confirmed = self.observe_deficit(v, requested) if self.extra_enabled else False
    previous = self.last_time_ns
    regular = super().correction(v, requested, measured, feedforward, driver_pressed)
    self.sustained_extra = 0.0
    self.sustained_reason = "not_admitted"
    if (
      not self.extra_enabled
      or not confirmed
      or not self.context_ok
      or not self.feedback_valid
      or self.hold_frames < self.HOLD_FRAMES
      or self.admission_kind not in ("steady", "ramp_limit_reached")
    ):
      return regular
    if len(self.history) != self.WINDOW_FRAMES or len(self.response_history) != self.WINDOW_FRAMES:
      return regular
    ks = [x[1] for x in self.history]
    span = (self.response_history[-1][0] - self.response_history[0][0]) / 1e9
    median = sorted(abs(x) for x in ks)[len(ks) // 2]
    if span < 0.45 or max(ks) - min(ks) > 0.15 * median or abs(ks[-1]) - abs(ks[0]) > 0.10 * median:
      return regular
    now = self.evidence["context"][0]
    axis = 1 if requested > 0 else -1
    if self.sustained_aborted or (self.sustained_sign and axis != self.sustained_sign):
      return regular
    if self.sustained_start is not None and now - self.sustained_start >= self.EXTRA_DURATION_S * 1e9:
      self.sustained_reason = "duration_spent"
      return regular
    dt = (now - previous) / 1e9 if previous is not None else 0.05
    if not 0.04 <= dt <= 0.06:
      return regular
    raw = self.KP * self.filtered_error * v * self.get_curvature_factor(v, requested)
    cap = min(self.MAX_EXTRA_PATH_RAD, self.MAX_RELATIVE_EXTRA * abs(feedforward))
    desired = min(abs(raw), cap) * self.speed_weight(v)
    remaining = max(0.0, self.EXTRA_AREA_RAD_S - self.sustained_area)
    extra = min(self.EXTRA_CAP_RAD * self.speed_weight(v), max(0.0, desired - abs(regular)), remaining / dt)
    if extra <= 0 or raw * requested <= 0:
      return regular
    if self.sustained_start is None:
      self.sustained_start = now
      self.sustained_sign = axis
    # Charge before the final native limits; clipping must never refund exposure.
    self.sustained_area = min(self.EXTRA_AREA_RAD_S, self.sustained_area + extra * dt)
    self.sustained_extra = math.copysign(extra, requested)
    self.tracking_correction = regular + self.sustained_extra
    self.sustained_reason = "sustained"
    return self.tracking_correction


class SustainedErrorUnwind(ExtendedWheelTarget):
  def __init__(self, CP, *, enabled=False):
    super().__init__(CP, enabled=enabled)
    self.installed_baseline.reference.baseline = SustainedTracking(CP, enabled=True, extra_enabled=enabled)
    self.reference.baseline = SustainedTracking(CP, enabled=True, extra_enabled=enabled)
    self.upper_reference = InstalledUpper(CP, enabled=enabled)
    self._actual_output_last = 0.0
    self.upper_handoff_pending = False
    self.upper_extension_weight = 1.0
    self.actual_final_residual = 0.0
    self.upper_output_delta = 0.0
    self.actual_reference_path = 0.0
    self.actual_reduction = 0.0
    self.actual_release_pending = False

  @property
  def target_speed(self):
    return self.reference.target_speed

  @target_speed.setter
  def target_speed(self, value):
    ExtendedWheelTarget.target_speed.fset(self, value)
    upper = self.__dict__.get("upper_reference")
    if upper is not None:
      upper.target_speed = value

  def set_context(self, **kw):
    super().set_context(**kw)
    upper = self.__dict__.get("upper_reference")
    if upper is not None:
      upper.set_context(**kw)

  def set_calibration(self, **kw):
    super().set_calibration(**kw)
    upper = self.__dict__.get("upper_reference")
    if upper is not None:
      upper.set_calibration(**kw)

  def set_motion(self, **kw):
    super().set_motion(**kw)
    upper = self.__dict__.get("upper_reference")
    if upper is not None:
      upper.set_motion(**kw)

  def update(self, CC, CS, actuators):
    s = CS.out
    self.upper_reference.enabled = self.enabled
    self.upper_reference.input_state_valid = self.input_state_valid
    upper = self.upper_reference.update(CC, CS, actuators)
    for parent in [self.installed_baseline.reference, self.reference]:
      low = parent.baseline
      low.extra_enabled = self.enabled
      low.set_evidence(
        dict(
          active=bool(CC.latActive),
          state_valid=bool(self.input_state_valid),
          driver_or_fault=any(
            bool(getattr(s, k, False)) for k in ["steeringPressed", "brakePressed", "leftBlinker", "rightBlinker", "steerFaultTemporary", "steerFaultPermanent"]
          ),
          context=parent.context,
          calibration=parent.calibration,
          motion=self.motion,
          generation=self._motion_generation,
          target=float(actuators.steeringAngleDeg),
          actual=float(s.steeringAngleDeg),
          model_v=float(parent.target_speed if parent.target_speed is not None else s.vEgo),
        )
      )
    extension = super().update(CC, CS, actuators)
    mph = float(s.vEgoRaw) / 0.44704
    weight = (1.0 - smoothstep((mph - 70.0) / 5.0)) if self.enabled and math.isfinite(mph) else 0.0
    self.upper_extension_weight = weight
    # The two branches retain independent onset histories. Only final output
    # is blended; neither branch is overwritten by the other's internal state.
    desired = (
      extension.path_angle if weight >= 1.0 else upper.path_angle if weight <= 0.0 else upper.path_angle + weight * (extension.path_angle - upper.path_angle)
    )
    result = desired
    if CC.latActive and not self.reference.human_turn_active:
      rate = float(interp(float(s.vEgoRaw), _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES))
      result = max(self._actual_output_last - rate, min(self._actual_output_last + rate, desired))
    self.upper_handoff_pending = abs(result - desired) > 1e-12
    self.handoff_pending = self.handoff_pending or self.upper_handoff_pending
    self._actual_output_last = result
    self.upper_output_delta = result - upper.path_angle
    base = self.reference.path_angle_last if weight > 0 else self.upper_reference.reference.path_angle_last
    self.actual_reference_path = base
    self.actual_final_residual = result - base
    self.actual_reduction = max(0.0, abs(base) - abs(result))
    self.actual_release_pending = self.upper_handoff_pending or (self.release_pending if weight > 0 else self.upper_reference.release_pending)
    return AngleLateralResult(path_angle=result, shadow_curvature=extension.shadow_curvature if weight > 0 else upper.shadow_curvature)
