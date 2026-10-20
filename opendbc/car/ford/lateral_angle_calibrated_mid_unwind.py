"""Bounded calibrated tracking, mid-speed episodes and independent unwind feedback."""

import math
from collections import deque
from statistics import median
from numpy import interp
from opendbc.car.ford.lateral_angle import AngleLateralResult, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES
from opendbc.car.ford.lateral_angle_sustained_unwind import SustainedErrorUnwind as Installed, SustainedTracking
from opendbc.car.ford.lateral_angle_high_speed import HighSpeedReduction
from opendbc.car.ford.values import CarControllerParams
from opendbc.car.ford.lateral_angle_low_speed_prototype import smoothstep


def _calibrated_ratios(observer, k, v):
  if not observer.basic_evidence() or not observer.feedback_valid or not math.isfinite(k) or abs(k) * v * v < 0.25:
    return None
  e = observer.evidence
  _, _, _, roll, ratio, stiffness, offset = e["calibration"]
  observer.sustained_vm.update_params(stiffness, ratio)
  expected = math.degrees(observer.sustained_vm.get_steer_from_curvature(-k, e["model_v"], roll)) + offset
  target = e["target"] - offset
  actual = e["actual"] - offset
  if abs(expected - e["target"]) > 0.5 or abs(target) < 5 or actual * target <= 0 or not k * v:
    return None
  wr = actual / target
  br = e["motion"][4] / (k * v)
  return (wr, br) if all(math.isfinite(x) and 0.1 <= x <= 3 for x in (wr, br)) else None


class CalibratedTracking(SustainedTracking):
  def __init__(self, CP, **kw):
    super().__init__(CP, **kw)
    self.feedback_definition = "fixed_pinion"
    self.feedback_ratios = None

  def correction(self, v, requested, measured, feedforward, driver_pressed):
    self.feedback_definition = "fixed_pinion"
    self.feedback_ratios = None
    if not self.extra_enabled:
      return super().correction(v, requested, measured, feedforward, driver_pressed)
    pair = _calibrated_ratios(self, requested, v) if self.speed_weight(v) > 0 else None
    if pair is None:
      if self.basic_evidence() and self.feedback_valid and abs(requested) <= 0.0005:
        # Quiet evidence is not deficit evidence; still advance both dwell guards.
        return super().correction(v, requested, measured, feedforward, driver_pressed)
      now = self.evidence["context"][0] if self.evidence and self.evidence["context"] else 0
      self._quiet_dwell(now, requested, measured, False)
      self.reset_tracking()
      self.sustained_history.clear()
      self.sustained_extra = 0.0
      self.sustained_reason = "invalid_calibrated_feedback"
      return 0.0
    self.feedback_definition = "calibrated_wheel_body"
    self.feedback_ratios = pair
    # Reinterpreting feedback must not create an unlimited new steady term.
    # All regular calibrated effort spends the existing transient allowance.
    self.ramp_episode = True
    # Experimental feedback only. Native pinion/deviation/shadow remains fixed.
    # The better-tracking sensor is conservative: both must establish a deficit.
    return super().correction(v, requested, requested * max(pair), feedforward, driver_pressed)


class MidEpisode:
  CAP = 0.0025
  AREA = 0.0015
  DURATION = 0.75

  def __init__(self):
    self.area = 0.0
    self.start = None
    self.axis = 0
    self.aborted = False
    self.history = deque(maxlen=5)
    self.highwater = (0, 0, 0)
    self.quiet_stamps = (0, 0, 0)
    self.generation = None
    self.quiet = []
    self.term = 0.0
    self.reason = "initial"
    self.rearm_count = 0
    self.total_area = 0.0
    self.quiet_reason = "initial"
    self.last_rearm = None

  def _observe_quiet(self, observer, k, wheel_k, body_k):
    e = observer.evidence
    now, st, ct, *_ = e["context"]
    mo = e["motion"]
    stamps = (st, ct, mo[1])
    self.quiet_reason = "collecting"
    distinct = all(a > b for a, b in zip(stamps, self.quiet_stamps, strict=False))
    if distinct:
      self.quiet_stamps = stamps
    if not distinct or not all(math.isfinite(x) for x in (k, wheel_k, body_k)) or max(abs(k), abs(wheel_k), abs(body_k)) > 0.0005:
      self.quiet = []
      self.quiet_reason = "duplicate_or_non_neutral"
      return
    # A 0.1-degree quantization increment expressed in the validated application
    # model. Two independently rounded samples can move by two endpoint quanta.
    quantum = abs(observer.sustained_vm.calc_curvature(math.radians(0.1), e["model_v"], 0.0))
    noise = mo[5] / e["model_v"]
    if not math.isfinite(quantum) or not math.isfinite(noise) or quantum <= 0 or noise < 0:
      self.quiet = []
      self.quiet_reason = "invalid_resolution"
      return
    if self.quiet:
      last = self.quiet[-1]
      dt = (now - last[0]) / 1e9
      source_dt = (mo[1] - last[4]) / 1e9
      if not 0.04 <= dt <= 0.06 or not 0.035 <= source_dt <= 0.065:
        self.quiet = []
        self.quiet_reason = "cadence"
      elif abs(wheel_k - last[1]) > min(0.00008, 0.0002 * dt + 2 * max(quantum, last[5])) or abs(body_k - last[2]) > 0.0002 * dt + min(
        0.00025, 0.25 * min(noise, last[6])
      ):
        self.quiet = []
        self.quiet_reason = "step"
    self.quiet.append((now, wheel_k, body_k, k, mo[1], quantum, noise))
    # Sliding full-second window: a previous large span cannot poison later
    # settled evidence, and no invalid or duplicate sample contributes dwell.
    while len(self.quiet) > 2 and now - self.quiet[1][0] >= 1_000_000_000:
      self.quiet.pop(0)
    if now - self.quiet[0][0] < 1_000_000_000:
      return
    wheel = [x[1] for x in self.quiet]
    body = [x[2] for x in self.quiet]
    demand = [x[3] for x in self.quiet]
    wheel_range = min(0.0002, 0.0001 + 2 * min(x[5] for x in self.quiet))
    body_range = 0.0001 + min(0.0003, 0.5 * min(x[6] for x in self.quiet))
    if max(wheel) - min(wheel) > wheel_range or max(body) - min(body) > body_range:
      self.quiet_reason = "range"
      return
    # A bounded noise allowance must not disguise cumulative steering motion.
    if any(abs(median(values[-5:]) - median(values[:5])) > 0.0001 for values in [wheel, body, demand]):
      self.quiet_reason = "drift"
      return
    self.quiet_reason = "confirmed"
    if self.start is not None or self.area > 0 or self.aborted:
      self.last_rearm = dict(
        now_ns=now,
        span_s=(now - self.quiet[0][0]) / 1e9,
        frames=len(self.quiet),
        wheel_range=max(wheel) - min(wheel),
        body_range=max(body) - min(body),
        wheel_range_limit=wheel_range,
        body_range_limit=body_range,
      )
      self.rearm_count += 1
      self.area = 0.0
      self.start = None
      self.axis = 0
      self.aborted = False

  def update(self, observer, k, v, base):
    self.term = 0.0
    e = observer.evidence
    self.reason = "outside_scope"
    if e is None:
      return 0.0
    now, st, ct, *_ = e["context"] if e["context"] is not None else (0, 0, 0)
    generation = e["generation"]
    if self.generation is not None and generation != self.generation:
      if self.start is not None:
        self.aborted = True
      self.history.clear()
      self.highwater = (0, 0, 0)
      self.quiet_stamps = (0, 0, 0)
      self.quiet = []
    self.generation = generation
    if not observer.basic_evidence() or not observer.feedback_valid:
      if self.start is not None:
        self.aborted = True
      self.history.clear()
      self.quiet = []
      self.reason = "invalid_evidence"
      return 0.0
    mo = e["motion"]
    body_k = mo[4] / max(e["model_v"], 0.1)
    _, _, _, roll, ratio, stiffness, offset = e["calibration"]
    observer.sustained_vm.update_params(stiffness, ratio)
    wheel_k = -observer.sustained_vm.calc_curvature(math.radians(e["actual"] - offset), e["model_v"], roll)
    self._observe_quiet(observer, k, wheel_k, body_k)
    mph = v / 0.44704
    if not 25 < mph < 30:
      if self.start is not None:
        self.aborted = True
      self.history.clear()
      return 0.0
    pair = _calibrated_ratios(observer, k, v)
    if pair is None:
      if self.start is not None:
        self.aborted = True
      self.history.clear()
      self.reason = "invalid_target"
      return 0.0
    wr, br = pair
    fixed = -observer.pinion_model.calc_curvature(math.radians(e["actual"]), v, 0.0)
    if v > 9 and abs(k - fixed) > CarControllerParams.CURVATURE_ERROR:
      if self.start is not None:
        self.aborted = True
      self.history.clear()
      self.reason = "native_deviation_clip"
      return 0.0
    stamps = (st, ct, mo[1])
    if any(a <= b for a, b in zip(stamps, self.highwater, strict=False)):
      if self.start is not None:
        self.aborted = True
      self.history.clear()
      self.reason = "duplicate_evidence"
      return 0.0
    self.highwater = stamps
    axis = 1 if k > 0 else -1
    if self.axis and axis != self.axis:
      self.aborted = True
      self.history.clear()
    if self.aborted or self.area >= self.AREA or (self.start is not None and now - self.start >= self.DURATION * 1e9):
      self.reason = "spent_or_aborted"
      return 0.0
    if not (0.1 <= wr < 0.9 and 0.1 <= br < 0.9):
      if self.start is not None:
        self.aborted = True
      self.history.clear()
      self.reason = "no_agreed_deficit"
      return 0.0
    self.history.append((st, wr, mo[1], br, k, v))
    if len(self.history) < 5:
      self.reason = "confirming"
      return 0.0
    first = self.history[0]
    last = self.history[-1]
    span = (last[0] - first[0]) / 1e9
    bs = (last[2] - first[2]) / 1e9
    if not (0.16 <= span <= 0.30 and 0.16 <= bs <= 0.30) or max(x[5] for x in self.history) - min(x[5] for x in self.history) > 0.5:
      self.reason = "invalid_window"
      return 0.0
    ks = [axis * x[4] for x in self.history]
    if any(x <= 0 for x in ks) or (ks[-1] - ks[0]) / (span * max(ks)) > 2.0 or ks[-1] < 0.95 * max(ks):
      if self.start is not None:
        self.aborted = True
      self.reason = "demand_unwind_or_jump"
      return 0.0

    def slope(index, ti):
      t = [(x[ti] - first[ti]) / 1e9 for x in self.history]
      y = [x[index] for x in self.history]
      mt = sum(t) / 5
      my = sum(y) / 5
      return sum((a - mt) * (b - my) for a, b in zip(t, y, strict=False)) / sum((a - mt) ** 2 for a in t)

    projected = max(wr + 0.3 * max(0.0, slope(1, 0)), br + 0.3 * max(0.0, slope(3, 2)))
    if projected >= 0.9:
      if self.start is not None:
        self.aborted = True
      self.reason = "catching_up"
      return 0.0
    weight = smoothstep(mph - 25) * (1 - smoothstep(mph - 29))
    amount = min(self.CAP, 0.05 * abs(base), 0.25 * (1 - projected) * abs(base)) * weight
    # Keep the combined effort inside the unchanged native deviation envelope,
    # even when an unclipped request has only a fraction of a wire step left.
    if v > 9:
      edge_k = fixed + axis * CarControllerParams.CURVATURE_ERROR
      edge_path = edge_k * v * observer.get_curvature_factor(v, edge_k)
      amount = min(amount, max(0.0, axis * (edge_path - base)))
    dt = (last[0] - self.history[-2][0]) / 1e9
    if not 0.04 <= dt <= 0.06:
      self.reason = "invalid_cadence"
      return 0.0
    amount = min(amount, max(0.0, self.AREA - self.area) / dt)
    if amount <= 0:
      return 0.0
    if self.start is None:
      self.start = now
      self.axis = axis
    self.area = min(self.AREA, self.area + amount * dt)
    self.total_area += amount * dt
    self.term = math.copysign(amount, k)
    self.reason = "bounded_correction"
    return self.term


class CalibratedMidUnwind(Installed):
  def __init__(self, CP, *, enabled=False):
    super().__init__(CP, enabled=enabled)
    self.installed_baseline.reference.baseline = CalibratedTracking(CP, enabled=True, extra_enabled=enabled)
    self.reference.baseline = CalibratedTracking(CP, enabled=True, extra_enabled=enabled)
    self.stock = Installed(CP, enabled=enabled)
    self.mid = MidEpisode()
    self.prototype_last = 0.0
    self.prototype_pending = False
    self._independent_wheel_history = deque(maxlen=5)
    self._independent_wheel_stamps = (0, 0)
    self._independent_wheel_generation = None

  def set_context(self, **kw):
    super().set_context(**kw)
    if hasattr(self, "stock"):
      self.stock.set_context(**kw)

  def set_calibration(self, **kw):
    super().set_calibration(**kw)
    if hasattr(self, "stock"):
      self.stock.set_calibration(**kw)

  def set_motion(self, **kw):
    super().set_motion(**kw)
    if hasattr(self, "stock"):
      self.stock.set_motion(**kw)

  @property
  def target_speed(self):
    return Installed.target_speed.fget(self)

  @target_speed.setter
  def target_speed(self, value):
    Installed.target_speed.fset(self, value)
    if hasattr(self, "stock"):
      self.stock.target_speed = value

  def _fraction(self, CC, CS, a):
    observer = self.reference.baseline
    e = observer.evidence
    v = float(CS.out.vEgoRaw)
    mph = v / 0.44704
    blend = smoothstep((mph - 60) / 10) if math.isfinite(mph) else 0.0
    history = self._independent_wheel_history
    valid = bool(e and observer.basic_evidence() and blend > 0)
    generation = e["generation"] if e else None
    if generation != self._independent_wheel_generation:
      history.clear()
      self._independent_wheel_stamps = (0, 0)
      self._independent_wheel_generation = generation
    error = actual = desired = 0.0
    if valid:
      now, st, ct, *_ = e["context"]
      offset = e["calibration"][-1]
      desired = e["target"] - offset
      actual = e["actual"] - offset
      _, _, _, roll, ratio, stiffness, _ = e["calibration"]
      observer.sustained_vm.update_params(stiffness, ratio)
      expected = math.degrees(observer.sustained_vm.get_steer_from_curvature(-float(a.curvature), e["model_v"], roll)) + offset
      valid = (
        abs(expected - e["target"]) <= 0.5
        and abs(desired) >= 1.5
        and desired * actual > 0
        and st > self._independent_wheel_stamps[0]
        and ct > self._independent_wheel_stamps[1]
      )
      if valid:
        if history and desired * history[-1][2] <= 0:
          history.clear()
        error = math.copysign(1.0, desired) * (actual - desired)
        history.append((st, error, desired))
        self._independent_wheel_stamps = (st, ct)
    if not valid:
      history.clear()
    # Current inherited wheel/body permission remains mandatory. A valid wheel
    # history is retained independently across body-excess-margin crossings.
    original = HighSpeedReduction._fraction(self, CC, CS, a)
    self.predicted_wheel_excess_deg = None
    if blend <= 0 or not self.enabled:
      self.wheel_reason = "installed_reference"
      return original
    if self.reason not in ("eligible", "unwind_rising_confirmation") or not valid:
      self.wheel_reason = "withdraw_" + self.reason
      return original * (1 - blend)
    fraction = 0.0
    self.wheel_reason = "fresh_wheel_history"
    if len(history) == 5:
      first = history[0][0]
      span = (history[-1][0] - first) / 1e9
      if 0.16 <= span <= 0.30:
        tt = [(x[0] - first) / 1e9 for x in history]
        yy = [x[1] for x in history]
        mt = sum(tt) / 5
        my = sum(yy) / 5
        slope = sum((t - mt) * (y - my) for t, y in zip(tt, yy, strict=False)) / sum((t - mt) ** 2 for t in tt)
        predicted = error + self.PREDICTION_S * slope
        self.predicted_wheel_excess_deg = predicted
        useful = min(error, predicted) - self.WHEEL_DEADBAND_DEG
        self.wheel_reason = "closing_or_under_target"
        if useful > 0:
          k = float(a.curvature)
          scope = (1 - smoothstep((abs(k) - 0.001) / 0.0005)) * smoothstep((abs(k) * v * v - 0.25) / 0.25)
          fraction = scope * min(self.MAX_FRACTION, useful / abs(actual))
          self.wheel_reason = "confirmed_target_excess"
    result = original * (1 - blend) + fraction * blend
    if result > 0:
      self.reason = "wheel_target_eligible"
    return result

  def update(self, CC, CS, actuators):
    self.stock.enabled = self.enabled
    self.stock.input_state_valid = self.input_state_valid
    stock = self.stock.update(CC, CS, actuators)
    result = super().update(CC, CS, actuators)
    v = float(CS.out.vEgoRaw)
    mph = v / 0.44704
    low = self.installed_baseline.reference.baseline
    extra = self.mid.update(low, float(actuators.curvature), v, result.path_angle) if self.enabled else 0.0
    desired = result.path_angle + extra
    if not self.enabled or mph <= 5 or 30 <= mph <= 60 or mph >= 75:
      desired = stock.path_angle
    if self.enabled and mph < 30 and not low.basic_evidence():
      desired = stock.path_angle
    if not CC.latActive or self.reference.human_turn_active:
      desired = stock.path_angle
    desired = max(-0.5, min(0.5235, desired))
    actual = desired
    if self.enabled and CC.latActive and not self.reference.human_turn_active:
      rate = float(interp(v, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES))
      actual = max(self.prototype_last - rate, min(self.prototype_last + rate, desired))
    self.prototype_pending = abs(actual - desired) > 1e-12
    self.prototype_last = actual
    self.actual_final_residual = actual - self.actual_reference_path
    return AngleLateralResult(path_angle=actual, shadow_curvature=result.shadow_curvature if self.enabled else stock.shadow_curvature)
