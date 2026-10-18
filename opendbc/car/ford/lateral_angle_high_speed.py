"""Opt-in body-confirmed high-speed reduction, with an independent reference.

The independent installed strategy remains the reference. Replay measures command
changes, never counterfactual vehicle motion. No CAN interfaces or parameter I/O.
"""

import math
from collections import deque
from statistics import median
from numpy import interp
from opendbc.car.ford.lateral_angle_roll_mapping import RollAwareMapping
from opendbc.car.ford.lateral_angle import AngleLateralResult, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES
from opendbc.car.ford.lateral_angle_low_speed_prototype import smoothstep
from opendbc.car.ford.values import CAR, FordFlags


class HighSpeedReduction:
  MAX_FRACTION = 0.15
  MAX_RAD = 0.01
  ONSET_RAD_PER_FRAME = 0.0003
  PERSISTENCE_FRAMES = 5

  def __init__(self, CP, *, enabled=False):
    self.reference = RollAwareMapping(CP, enabled=True)
    self.enabled = enabled
    self.reduction = 0.0
    self.path_angle_last = 0.0
    self.final_residual = 0.0
    self.release_pending = False
    self.reason = "initial"
    self._previous_ns = None
    self._previous_request = 0.0
    self._persistent = 0
    self.infeasible_intersections = 0
    self.input_state_valid = False
    self.motion = None
    self._motion_generation = None
    self.body_history = deque(maxlen=8)
    self._motion_highwater = 0
    self.body_excess_accel = None
    self.body_reason = "missing"
    self.demand_peak = 0.0
    self.filtered_demand = 0.0
    self.demand_window = deque(maxlen=3)
    self.demand_sign = 0
    self.unwind_latched = False
    self._armed_bend = False
    self._rising_rearm = 0

  def __getattr__(self, name):
    return getattr(self.reference, name)

  @property
  def target_speed(self):
    return self.reference.target_speed

  @target_speed.setter
  def target_speed(self, value):
    self.reference.target_speed = value

  def set_context(self, **kw):
    self.reference.set_context(**kw)

  def set_calibration(self, **kw):
    self.reference.set_calibration(**kw)

  def set_motion(self, *, now_ns, sample_ns, received_ns, valid, yaw_rate, yaw_std, generation=0):
    if generation != self._motion_generation:
      self.body_history.clear()
      self._motion_generation = generation
    self.motion = (now_ns, sample_ns, received_ns, bool(valid), yaw_rate, yaw_std)

  def _reset_bend(self):
    self.demand_window.clear()
    self.filtered_demand = 0.0
    self.demand_peak = 0.0
    self.demand_sign = 0
    self.unwind_latched = False
    self._armed_bend = False
    self._rising_rearm = 0

  def _demand_gate(self, k, v, active):
    # A fixed bend peak cannot forget a slow cumulative withdrawal. A neutral
    # interval or direction change starts a new bend; a lower plateau does not.
    sign = 1 if k > 0 else -1 if k < 0 else 0
    if not active or not math.isfinite(k) or not math.isfinite(v) or v / 0.44704 <= 60 or abs(k) * v * v < 0.25:
      self._reset_bend()
      return False
    if sign != self.demand_sign:
      self._reset_bend()
      self.demand_sign = sign
    self.demand_window.append(abs(k))
    self.filtered_demand = median(self.demand_window)
    # Track cumulative withdrawal only after this correction has been admitted.
    # Unrelated pre-entry jitter must not disable a later rising bend.
    if not self.unwind_latched:
      self.demand_peak = max(self.demand_peak, self.filtered_demand) if self._armed_bend else self.filtered_demand
    if self._armed_bend and (self.filtered_demand < 0.9 * self.demand_peak or abs(k) < 0.8 * self.demand_peak):
      self.unwind_latched = True
    return not self.unwind_latched

  def _body_confirms(self, now, k, v):
    self.body_excess_accel = None
    self.body_reason = "missing"
    if self.motion is None:
      self.body_history.clear()
      return False
    stamp_now, source, received, good, yaw, std = self.motion
    valid = (
      good
      and stamp_now == now
      and 0 < source <= received <= now
      and 0 <= (now - source) / 1e9 <= 0.1
      and 0 <= (now - received) / 1e9 <= 0.1
      and source >= self._motion_highwater
      and math.isfinite(yaw)
      and math.isfinite(std)
      and 0 <= std <= 0.03
    )
    if not valid:
      self.body_history.clear()
      self.body_reason = "invalid"
      return False
    if source > self._motion_highwater:
      self.body_history.append((source, yaw))
      self._motion_highwater = source
    while self.body_history and now - self.body_history[0][0] > 300_000_000:
      self.body_history.popleft()
    self.body_excess_accel = math.copysign(1.0, k) * yaw * v - abs(k) * v * v
    # These are conservative engineering admission margins, not statistical
    # confidence intervals: the source yaw covariance is temporally correlated.
    margin = max(0.1, 0.1 * abs(k) * v * v)
    if self.body_excess_accel <= margin:
      self.body_reason = "no_current_excess"
      return False
    if len(self.body_history) < 3 or self.body_history[-1][0] - self.body_history[0][0] < 80_000_000:
      self.body_reason = "insufficient_history"
      return False
    values = sorted(math.copysign(1.0, k) * y * v - abs(k) * v * v for _, y in self.body_history)
    if median(values) <= margin:
      self.body_reason = "no_sustained_excess"
      return False
    self.body_reason = "confirmed"
    return True

  def _fraction(self, CC, CS, a):
    ctx = self.reference.context
    cal = self.reference.calibration
    s = CS.out
    k = float(a.curvature)
    v = float(s.vEgoRaw)
    previous = self._previous_request
    self._previous_request = k
    now = ctx[0] if ctx is not None else None
    old_now = self._previous_ns
    self._previous_ns = now
    self.reason = "ineligible"
    self.body_excess_accel = None
    self.body_reason = "not_evaluated"
    if not (
      self.enabled
      and self.input_state_valid
      and getattr(s, "canValid", False)
      and not getattr(s, "steerFaultTemporary", False)
      and not getattr(s, "steerFaultPermanent", False)
      and not getattr(s, "leftBlinker", False)
      and not getattr(s, "rightBlinker", False)
      and not getattr(s, "brakePressed", False)
      and self.CP.carFingerprint == CAR.FORD_EXPEDITION_MK4
      and self.CP.flags & FordFlags.PINION_CURVATURE
      and self.CP.flags & FordFlags.LOW_SPEED_GAIN
      and ctx is not None
      and cal is not None
      and math.isfinite(k)
      and math.isfinite(v)
    ):
      return 0.0
    _, state_ns, control_ns, model_ns, lane_change, model_valid = ctx
    stamp_now, stamp, good, roll, ratio, stiffness, offset = cal
    target = float(getattr(a, "steeringAngleDeg", float("nan")))
    actual = float(s.steeringAngleDeg)
    model_v = float(self.target_speed if self.target_speed is not None else s.vEgo)
    if not (
      model_valid
      and not lane_change
      and good
      and stamp_now == now
      and all(0 < ns <= now and (now - ns) / 1e9 <= limit for ns, limit in zip((state_ns, control_ns, model_ns, stamp), (0.08, 0.1, 0.15, 0.15), strict=False))
      and (old_now is None or 40_000_000 <= now - old_now <= 60_000_000)
      and all(math.isfinite(x) for x in (roll, ratio, stiffness, offset, target, actual, model_v))
      and 10 <= ratio <= 35
      and 0.2 <= stiffness <= 3
      and abs(offset) <= 10
      and abs(roll) <= 0.15
      and model_v > 0.1
    ):
      self.reason = "invalid_context"
      return 0.0
    # Only fresh validated evidence may reset or advance a bend's peak/latch.
    self._demand_gate(k, v, CC.latActive and not s.steeringPressed)
    if not CC.latActive or s.steeringPressed:
      self.reason = "driver_or_inactive"
      return 0.0
    mph = v / 0.44704
    weight = smoothstep((mph - 60) / 10) * (1 - smoothstep((abs(k) - 0.001) / 0.0005)) * smoothstep((abs(k) * v * v - 0.25) / 0.25)
    if weight <= 0 or k * previous < 0:
      self.reason = "outside_scope_or_unwinding"
      return 0.0
    vm = self.reference.application_model
    vm.update_params(stiffness, ratio)
    expected = math.degrees(vm.get_steer_from_curvature(-k, model_v, roll)) + offset
    if abs(expected - target) > 0.5:
      self.reason = "target_mismatch"
      return 0.0
    desired = target - offset
    measured = actual - offset
    # Derive turn axes from the application model rather than assuming CAN signs.
    expected_no_roll = math.degrees(vm.get_steer_from_curvature(-k, model_v, 0.0))
    if abs(desired) < 1.5 or desired * measured <= 0 or desired * expected_no_roll <= 0:
      self.reason = "neutral_or_direction"
      return 0.0
    excess = math.copysign(1.0, desired) * (measured - desired) - 0.5
    if excess <= 0:
      self.reason = "no_excess"
      return 0.0
    if not self._body_confirms(now, k, v):
      self.reason = "body_" + self.body_reason
      return 0.0
    if self.unwind_latched:
      self._rising_rearm = self._rising_rearm + 1 if self.filtered_demand > 1.02 * self.demand_peak else 0
      if self._rising_rearm < self.PERSISTENCE_FRAMES:
        self.reason = "unwind_rising_confirmation"
        return 0.0
      # A genuinely new demand peak can re-arm without waiting for a straight;
      # a continuously declining or lower plateau request can never do this.
      self.unwind_latched = False
      self._armed_bend = False
      self._rising_rearm = 0
      self.demand_peak = self.filtered_demand
      self._persistent = 0
    self.reason = "eligible"
    return weight * min(self.MAX_FRACTION, excess / abs(measured))

  def update(self, CC, CS, actuators):
    fraction = self._fraction(CC, CS, actuators)
    base = self.reference.update(CC, CS, actuators)
    b = base.path_angle
    previous = self.path_angle_last
    if not self.enabled:
      self.reduction = 0.0
      self.path_angle_last = b
      return base
    if not CC.latActive or self.reference.human_turn_active:
      self.reduction = 0.0
      self._persistent = 0
      self.release_pending = False
      self.path_angle_last = b
      self.final_residual = 0.0
      self.reason = "driver_or_inactive"
      return base
    if fraction <= 0:
      self._persistent = 0
      if self.reason != "unwind_rising_confirmation":
        self._rising_rearm = 0
      if abs(self.final_residual) > 1e-12:
        self.release_pending = True
      self.reduction = 0.0
    elif not self.release_pending:
      self._persistent += 1
      target = min(self.MAX_RAD, abs(b) * fraction) if self._persistent >= self.PERSISTENCE_FRAMES else 0.0
      self.reduction = min(target, self.reduction + self.ONSET_RAD_PER_FRAME)
      if self.reduction > 0:
        self._armed_bend = True
    else:
      self.reduction = 0.0
    cap = min(self.MAX_RAD, self.MAX_FRACTION * abs(b))
    desired = b - math.copysign(min(cap, self.reduction), b)
    rate = float(interp(float(CS.out.vEgoRaw), _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES))
    # Intersect reduction-only envelope with the inherited final slew envelope.
    edge = b - math.copysign(cap, b)
    low = max(min(b, edge), previous - rate)
    high = min(max(b, edge), previous + rate)
    if low > high + 1e-12:
      self.infeasible_intersections += 1
      raise AssertionError("Installed reference and final reduction/slew bounds have no intersection")
    result = max(low, min(high, desired))
    self.final_residual = result - b
    if self.release_pending:
      self.reason = "releasing"
      if abs(self.final_residual) < 1e-12:
        self.release_pending = False
    elif self.reduction > 0:
      self.reason = "reducing"
    self.path_angle_last = result
    return AngleLateralResult(path_angle=result, shadow_curvature=base.shadow_curvature)
