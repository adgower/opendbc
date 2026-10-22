"""Bounded earlier admission at20–25mph and withdrawing-demand onset at70–75mph."""

import math
from collections import deque
from numpy import interp
from opendbc.car.ford.lateral_angle_earlier_unwind_onset import EarlierUnwindOnset as Installed
from opendbc.car.ford.lateral_angle_calibrated_mid_unwind import CalibratedTracking
from opendbc.car.ford.lateral_angle_extended_wheel import ExtendedWheelTarget
from opendbc.car.ford.lateral_angle_low_speed_prototype import smoothstep
from opendbc.car.ford.lateral_angle import AngleLateralResult, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES
from opendbc.car.ford.values import CarControllerParams


def slope(rows, column):
  tt = [(r[0] - rows[0][0]) / 1e9 for r in rows]
  yy = [r[column] for r in rows]
  mt = sum(tt) / len(tt)
  my = sum(yy) / len(yy)
  return sum((t - mt) * (y - my) for t, y in zip(tt, yy, strict=False)) / sum((t - mt) ** 2 for t in tt)


class EarlyLow(CalibratedTracking):
  def __init__(self, *args, **kw):
    super().__init__(*args, **kw)
    self.intent = 0.0
    self.early_history = deque(maxlen=5)
    self.early_start = None
    self.early_aborted = False
    self.early_axis = 0
    self.early_epoch = 0
    self.early_reason = "initial"
    self.early_term = 0.0
    self.early_total_area = 0.0

  def correction(self, v, requested, measured, feedforward, driver_pressed):
    regular = super().correction(v, requested, measured, feedforward, driver_pressed)
    self.early_term = 0.0
    self.early_reason = "outside_scope"
    e = self.evidence
    mph = v / 0.44704
    if self.rearm_count != self.early_epoch:
      self.early_epoch = self.rearm_count
      self.early_start = None
      self.early_aborted = False
      self.early_axis = 0
      self.early_history.clear()

    def stop(reason):
      self.early_reason = reason
      self.early_history.clear()
      if self.early_start is not None:
        self.early_aborted = True
      return regular

    if not self.extra_enabled or not 20 < mph < 25:
      return stop("outside_scope")
    if not self.basic_evidence() or not self.feedback_valid or driver_pressed:
      return stop("invalid_evidence")
    k = self.intent
    now, st, ct, *_ = e["context"]
    ms = e["motion"][1]
    if not all(math.isfinite(x) for x in [k, requested, measured, feedforward]):
      return stop("nonfinite")
    _, _, _, roll, ratio, stiffness, offset = e["calibration"]
    self.sustained_vm.update_params(stiffness, ratio)
    expected = math.degrees(self.sustained_vm.get_steer_from_curvature(-k, e["model_v"], roll)) + offset
    if abs(expected - e["target"]) > 0.5:
      return stop("original_target_mismatch")
    if abs(k - requested) > 1e-10:
      return stop("command_deviation_limited")
    target = e["target"] - offset
    actual = e["actual"] - offset
    if abs(target) < 1.5 or actual * target <= 0 or abs(k) * v * v < 0.10:
      return stop("small_or_direction")
    wr = actual / target
    br = e["motion"][4] / (k * v)
    deficit = abs(k) * v * v - math.copysign(1.0, k) * e["motion"][4] * v
    if not (0 <= wr < 0.9 and 0 <= br < 0.9) or deficit <= max(0.05, 0.1 * abs(k) * v * v, 0.25 * e["motion"][5] * v):
      return stop("no_current_agreed_deficit")
    axis = 1 if k > 0 else -1
    if self.early_axis and axis != self.early_axis:
      self.early_aborted = True
    current = (now, wr, br, k, st, ct, ms, e["generation"])
    if self.early_history:
      old = self.early_history[-1]
      if old[7] != current[7] or any(current[j] <= old[j] for j in [0, 4, 5, 6]) or not 40_000_000 <= now - old[0] <= 60_000_000:
        return stop("duplicate_generation_or_cadence")
      if k * old[3] <= 0:
        return stop("reversal")
    self.early_history.append(current)
    if regular:
      return stop("installed_effort")
    if self.early_aborted or (self.early_start is not None and now - self.early_start >= 250_000_000):
      return stop("episode_spent")
    if len(self.early_history) < 5:
      self.early_reason = "collecting"
      return regular
    rows = list(self.early_history)
    if any(not 160_000_000 <= rows[-1][j] - rows[0][j] <= 300_000_000 for j in [0, 4, 5, 6]):
      return stop("span")
    ks = [abs(r[3]) for r in rows]
    if ks[-1] < 1.02 * ks[0] or max(ks) > 2 * min(ks) or ks[-1] < 0.98 * ks[-2]:
      return stop("not_bounded_rising")
    if max(wr + 0.15 * max(0.0, slope(rows, 1)), br + 0.15 * max(0.0, slope(rows, 2))) >= 0.9:
      return stop("predicted_catchup")
    weight = smoothstep(mph - 20) * (1 - smoothstep(mph - 24))
    amount = min(0.0015, 0.05 * abs(feedforward)) * weight
    edge = measured + axis * CarControllerParams.CURVATURE_ERROR
    edge_path = edge * v * self.get_curvature_factor(v, edge)
    dt = (rows[-1][0] - rows[-2][0]) / 1e9
    amount = min(amount, max(0.0, axis * (edge_path - feedforward)), max(0.0, self.RAMP_EXTRA_AREA_LIMIT - self.ramp_extra_area) / dt)
    if amount <= 0:
      return stop("no_headroom_or_budget")
    if self.early_start is None:
      self.early_start = now
      self.early_axis = axis
    self.ramp_extra_area = min(self.RAMP_EXTRA_AREA_LIMIT, self.ramp_extra_area + amount * dt)
    self.early_total_area += amount * dt
    self.ramp_episode = True
    self.early_term = math.copysign(amount, k)
    self.early_reason = "bounded_preclip"
    return regular + self.early_term


class Credit70:
  def __init__(self, *args, **kw):
    super().__init__(*args, **kw)
    self.credit_history = deque(maxlen=5)
    self.credit_receipt = None
    self.credit_generation = None

  def _fraction(self, CC, CS, a):
    f = super()._fraction(CC, CS, a)
    self.credit_receipt = None
    v = float(CS.out.vEgoRaw)
    mph = v / 0.44704
    ctx = self.reference.context
    cal = self.reference.calibration
    mo = self.motion
    s = CS.out
    if (
      not self.enabled
      or not 70 < mph < 75
      or not self.input_state_valid
      or not ctx
      or not cal
      or not mo
      or not CC.latActive
      or any(
        bool(getattr(s, k, False)) for k in ["steeringPressed", "brakePressed", "leftBlinker", "rightBlinker", "steerFaultTemporary", "steerFaultPermanent"]
      )
      or not getattr(s, "canValid", False)
    ):
      self.credit_history.clear()
      return f
    now, st, ct, mt, lane, mvalid = ctx
    cn, source, good, roll, ratio, stiffness, offset = cal
    mn, ms, mr, mok, yaw, std = mo
    k = float(a.curvature)
    if not (
      mvalid
      and not lane
      and good
      and mok
      and cn == mn == now
      and all(
        0 < t <= now and now - t <= lim
        for t, lim in [(st, 40_000_000), (ct, 40_000_000), (mt, 150_000_000), (source, 150_000_000), (ms, 100_000_000), (mr, 100_000_000)]
      )
      and ms <= mr
      and all(math.isfinite(x) for x in [k, roll, ratio, stiffness, offset, yaw, std])
      and 0 <= std <= 0.03
      and abs(roll) <= 0.15
      and 10 <= ratio <= 35
      and 0.2 <= stiffness <= 3
      and abs(offset) <= 10
    ):
      self.credit_history.clear()
      return f
    vm = self.reference.application_model
    vm.update_params(stiffness, ratio)
    mv = float(self.target_speed if self.target_speed is not None else s.vEgo)
    if not math.isfinite(mv) or mv <= 0.1:
      self.credit_history.clear()
      return f
    expected = math.degrees(vm.get_steer_from_curvature(-k, mv, roll)) + offset
    target = float(a.steeringAngleDeg) - offset
    actual = float(s.steeringAngleDeg) - offset
    if abs(expected - float(a.steeringAngleDeg)) > 0.5 or abs(target) < 1.5 or target * actual <= 0:
      self.credit_history.clear()
      return f
    current = (now, math.copysign(1.0, target) * (actual - target), k, st, ct, ms)
    if self.credit_history:
      old = self.credit_history[-1]
      if (
        self._motion_generation != self.credit_generation
        or any(current[j] <= old[j] for j in [0, 3, 4, 5])
        or not 40_000_000 <= now - old[0] <= 60_000_000
        or k * old[2] <= 0
      ):
        self.credit_history.clear()
        return f
    self.credit_generation = self._motion_generation
    self.credit_history.append(current)
    if len(self.credit_history) < 5:
      return f
    rows = list(self.credit_history)
    if any(not 160_000_000 <= rows[-1][j] - rows[0][j] <= 300_000_000 for j in [0, 3, 4, 5]):
      return f
    demand = [abs(x[2]) for x in rows]
    if demand[-1] > 0.98 * max(demand) or demand[-1] > demand[-2]:
      return f
    if self.body_reason != "confirmed" or self.reason not in ["eligible", "unwind_rising_confirmation", "wheel_target_eligible"] or self.release_pending:
      return f
    error = current[1]
    predicted = error + self.PREDICTION_S * slope(rows, 1)
    useful = min(error, predicted) - self.WHEEL_DEADBAND_DEG
    if useful <= 0:
      return f
    scope = (1 - smoothstep((abs(k) - 0.001) / 0.0005)) * smoothstep((abs(k) * v * v - 0.25) / 0.25)
    f = max(f, scope * min(self.MAX_FRACTION, useful / abs(actual)))
    if f > 0:
      self._persistent = max(self._persistent, self.PERSISTENCE_FRAMES - 1)
      self.reason = "wheel_target_eligible"
      self.credit_receipt = dict(now_ns=now, frames=5, state_span_s=(st - rows[0][3]) / 1e9, motion_span_s=(ms - rows[0][5]) / 1e9)
    return f


class UpperCredit(Credit70, ExtendedWheelTarget):
  pass


class EarlyLowHighOnset(Credit70, Installed):
  def __init__(self, CP, *, enabled=False):
    super().__init__(CP, enabled=enabled)
    self.installed_baseline.reference.baseline = EarlyLow(CP, enabled=True, extra_enabled=enabled)
    self.reference.baseline = EarlyLow(CP, enabled=True, extra_enabled=enabled)
    self.upper_reference = UpperCredit(CP, enabled=enabled)
    self.unchanged = Installed(CP, enabled=enabled)
    self.outer_last = 0.0
    self.outer_pending = False

  def set_context(self, **kw):
    super().set_context(**kw)
    if "unchanged" in self.__dict__:
      self.unchanged.set_context(**kw)

  def set_calibration(self, **kw):
    super().set_calibration(**kw)
    if "unchanged" in self.__dict__:
      self.unchanged.set_calibration(**kw)

  def set_motion(self, **kw):
    super().set_motion(**kw)
    if "unchanged" in self.__dict__:
      self.unchanged.set_motion(**kw)

  def update(self, CC, CS, a):
    self.unchanged.enabled = self.enabled
    self.unchanged.input_state_valid = self.input_state_valid
    self.unchanged.target_speed = self.target_speed
    stock = self.unchanged.update(CC, CS, a)
    for low in [self.installed_baseline.reference.baseline, self.reference.baseline]:
      low.intent = float(a.curvature)
    out = super().update(CC, CS, a)
    mph = float(CS.out.vEgoRaw) / 0.44704
    desired = out.path_angle if self.enabled and (20 < mph < 25 or 70 < mph < 75) else stock.path_angle
    actual = desired
    if CC.latActive and not self.reference.human_turn_active:
      rate = float(interp(float(CS.out.vEgoRaw), _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES))
      actual = max(self.outer_last - rate, min(self.outer_last + rate, desired))
    self.outer_pending = abs(actual - desired) > 1e-12
    self.outer_last = actual
    return AngleLateralResult(actual, stock.shadow_curvature)
