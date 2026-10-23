"""Bounded highway observer continuity over unchanged installed policy."""

import math
from collections import deque
from numpy import interp
from opendbc.car.ford.lateral_angle_early_low_high_onset import EarlyLowHighOnset as Installed, slope
from opendbc.car.ford.lateral_angle import AngleLateralResult, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES
from opendbc.car.ford.lateral_angle_low_speed_prototype import smoothstep
from opendbc.car.ford.values import CAR, FordFlags
from opendbc.car.vehicle_model import VehicleModel


class Observer:
  def __init__(self):
    self.rows = deque(maxlen=5)
    self.generation = None
    self.reason = "initial"
    self.model = None

  def clear(self, reason):
    self.rows.clear()
    self.reason = reason

  def feed(self, owner, CC, CS, a):
    ctx = owner.reference.context
    cal = owner.reference.calibration
    mo = owner.motion
    s = CS.out
    if (
      not owner.enabled
      or not owner.input_state_valid
      or ctx is None
      or cal is None
      or mo is None
      or not CC.latActive
      or any(
        bool(getattr(s, k, False)) for k in ["steeringPressed", "brakePressed", "leftBlinker", "rightBlinker", "steerFaultTemporary", "steerFaultPermanent"]
      )
      or not s.canValid
    ):
      self.clear("invalid_state")
      return None
    now, st, ct, mt, lane, mvalid = ctx
    cn, source, good, roll, ratio, stiffness, offset = cal
    mn, ms, mr, mok, yaw, std = mo
    k = float(a.curvature)
    v = float(s.vEgoRaw)
    mv = float(owner.target_speed if owner.target_speed is not None else s.vEgo)
    if not (
      mvalid
      and not lane
      and good
      and mok
      and cn == mn == now
      and 0 < ms <= mr <= now
      and now - ms <= 100_000_000
      and now - mr <= 100_000_000
      and all(0 < t <= now and now - t <= lim for t, lim in [(st, 40_000_000), (ct, 40_000_000), (mt, 150_000_000), (source, 150_000_000)])
      and all(math.isfinite(x) for x in [k, v, mv, roll, ratio, stiffness, offset, yaw, std, a.steeringAngleDeg, s.steeringAngleDeg])
      and v > 0.1
      and mv > 0.1
      and 0 <= std <= 0.03
      and abs(roll) <= 0.15
      and 10 <= ratio <= 35
      and 0.2 <= stiffness <= 3
      and abs(offset) <= 10
    ):
      self.clear("invalid_provenance")
      return None

    if self.model is None:
      self.model = VehicleModel(owner.CP)
    vm = self.model
    vm.update_params(stiffness, ratio)
    expected = math.degrees(vm.get_steer_from_curvature(-k, mv, roll)) + offset
    target = float(a.steeringAngleDeg) - offset
    actual = float(s.steeringAngleDeg) - offset
    if abs(expected - float(a.steeringAngleDeg)) > 0.5 or abs(target) < 1.5 or actual * target <= 0 or abs(k) * v * v < 0.1:
      self.clear("target_or_direction")
      return None
    axis = 1 if target > 0 else -1
    row = (now, axis * (actual - target), k, st, ct, ms, actual / target, yaw / (k * v), std, yaw, v)
    generation = owner._motion_generation
    if self.generation != generation:
      self.rows.clear()
      self.generation = generation
    if self.rows:
      old = self.rows[-1]
      if k * old[2] <= 0 or any(row[j] <= old[j] for j in [0, 3, 4, 5]) or not 40_000_000 <= now - old[0] <= 60_000_000:
        self.clear("reversal_duplicate_or_cadence")
        return None
    self.rows.append(row)
    self.reason = "observing"
    if len(self.rows) == 5 and all(160_000_000 <= row[j] - self.rows[0][j] <= 300_000_000 for j in [0, 3, 4, 5]):
      self.reason = "ready"
    return row


class ObserverContinuity(Installed):
  def __init__(self, CP, *, enabled=False):
    super().__init__(CP, enabled=enabled)
    self.high_observer = Observer()
    self.low_observer = Observer()
    self.experiment_last = 0.0
    self.experiment_residual = 0.0
    self.experiment_release_pending = False
    self.experiment_reason = "initial"
    self.experiment_total_trim_last = 0.0
    for low in [self.reference.baseline, self.installed_baseline.reference.baseline]:
      low.observer_valid = False
      low.observer_reason = "initial"
      low.continuity_term = 0.0

  def update(self, CC, CS, a):
    # The entire installed strategy advances with unmodified history and budget.
    reference = super().update(CC, CS, a)
    mph = float(CS.out.vEgoRaw) / 0.44704
    lowrow = self.low_observer.feed(self, CC, CS, a)
    for low in [self.reference.baseline, self.installed_baseline.reference.baseline]:
      low.observer_valid = lowrow is not None
      low.observer_reason = self.low_observer.reason
      low.continuity_term = 0.0
      if lowrow is not None and abs(float(a.curvature) - low.kappa_cmd) > 1e-10:
        low.observer_reason = "headroom_limited"
    row = self.high_observer.feed(self, CC, CS, a)
    self.experiment_reason = "reference"
    extra = 0.0
    scope = smoothstep(mph - 70) * (1 - smoothstep(mph - 79)) if math.isfinite(mph) else 0.0
    b = float(reference.path_angle)
    # Every selected production path has its own untouched untrimmed reference.
    raw = self.actual_reference_path if mph < 75 else self.unchanged.stock.upper_reference.reference.path_angle_last
    if (
      self.enabled
      and self.CP.carFingerprint == CAR.FORD_EXPEDITION_MK4
      and self.CP.flags & FordFlags.PINION_CURVATURE
      and self.CP.flags & FordFlags.LOW_SPEED_GAIN
      and scope > 0
      and row is not None
      and self.high_observer.reason == "ready"
    ):
      rows = list(self.high_observer.rows)
      v = float(CS.out.vEgoRaw)
      k = float(a.curvature)
      target = float(a.steeringAngleDeg) - self.reference.calibration[-1]
      actual = float(CS.out.steeringAngleDeg) - self.reference.calibration[-1]
      expected_no_roll = math.degrees(
        self.high_observer.model.get_steer_from_curvature(-k, float(self.target_speed if self.target_speed is not None else CS.out.vEgo), 0.0)
      )
      error = row[1]
      predicted = error + 0.15 * slope([(x[3], x[1]) for x in rows], 1)
      useful = min(error, predicted) - 0.5
      demand = abs(k) * v * v
      margin = max(0.1, 0.1 * demand)
      excess = math.copysign(1.0, k) * row[9] * v - demand
      history = sorted(math.copysign(1.0, k) * x[9] * v - demand for x in rows)
      selected = self if mph < 75 else self.unchanged.stock.upper_reference
      if (
        b * raw > 0
        and selected.body_reason == "confirmed"
        and target * expected_no_roll > 0
        and useful > 0
        and excess > margin
        and history[len(history) // 2] > margin
      ):
        weight = (1 - smoothstep((abs(k) - 0.001) / 0.0005)) * smoothstep((demand - 0.25) / 0.25)
        desired = min(0.01, 0.15 * abs(raw), abs(raw) * weight * min(0.15, useful / abs(actual)))
        already = max(0.0, abs(raw) - abs(b))
        extra = min(max(0.0, desired - already) * scope, max(0.0, self.experiment_total_trim_last + 0.0003 - already))
        if self.experiment_release_pending:
          extra = 0.0
        self.experiment_reason = "current_confirmed_observer" if extra else "installed_already_sufficient"
    desired = b - math.copysign(extra, raw) if extra else b
    if not self.enabled or not CC.latActive or self.reference.human_turn_active:
      result = b
      self.experiment_release_pending = False
    else:
      rate = float(interp(float(CS.out.vEgoRaw), _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES))
      result = max(self.experiment_last - rate, min(self.experiment_last + rate, desired))
      self.experiment_release_pending = abs(result - desired) > 1e-12
    self.experiment_last = result
    self.experiment_residual = result - b
    self.experiment_total_trim_last = max(0.0, abs(raw) - abs(result)) if scope > 0 else 0.0
    if self.experiment_release_pending:
      self.experiment_reason = "bounded_release"
    return AngleLateralResult(path_angle=result, shadow_curvature=reference.shadow_curvature)
