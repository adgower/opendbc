"""Bounded Navigator unwind allocation and fresh near-neutral confirmation.

Pure strategy: existing opt-in only, no I/O or new settings. Invariant failures
stop control rather than bypass native envelopes or refund unknown exposure.
"""

import math
from collections import deque
import numpy as np
from opendbc.can.dbc import DBC
from opendbc.car.vehicle_model import VehicleModel
from opendbc.car.ford.lateral_angle import AngleLateralResult, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES
from opendbc.car.ford.lateral_angle_low_speed_prototype import smoothstep


def planned_exit(position, direction):
  """Signed spatial curvature at 60 m and 120 m from one predicted path."""
  x = np.asarray(position.x)
  y = np.asarray(position.y)
  if len(x) != len(y) or len(x) < 3 or x[0] > 40 or x[-1] < 140:
    return False
  if not (np.isfinite(x).all() and np.isfinite(y).all() and np.all(np.diff(x) > 0)):
    return False
  y40, y60, y80, y100, y120, y140 = np.interp((40, 60, 80, 100, 120, 140), x, y)
  near = direction * (y40 - 2*y60 + y80) / 400
  far = direction * (y100 - 2*y120 + y140) / 400
  return near >= 0.00025 and -0.00005 <= far <= min(0.00030, 0.60 * near)


class PlannedExit:
  def __init__(self):
    self.rows = deque(maxlen=3)
    self.stamp = 0
    self.direction = 0

  def observe(self, model, stamp, now, direction):
    if model is None or direction == 0 or not 0 < stamp <= now or now - stamp > 150_000_000:
      self.rows.clear()
      return False
    if direction != self.direction or stamp < self.stamp or stamp - self.stamp > 100_000_000:
      self.rows.clear()
      self.stamp = 0
    self.direction = direction
    if stamp > self.stamp:
      self.stamp = stamp
      if not planned_exit(model.position, direction):
        self.rows.clear()
      else:
        self.rows.append(stamp)
    return len(self.rows) == 3 and self.rows[-1] - self.rows[0] >= 70_000_000


class UnwindAllocation:
  def __init__(self, reference, *, enabled=False):
    self.reference = reference
    self.CP = reference.CP
    self.enabled = enabled is True
    self.vm = VehicleModel(self.CP)
    self.path_signal = DBC("ford_lincoln_base_pt").addr_to_msg[0x3d6].sigs["LatCtlPath_An_Actl"]
    self.rows = deque(maxlen=5)
    self.motion = None
    self.generation = None
    self.term = 0.0
    self.area = 0.0
    self.active_time = 0.0
    self.start = None
    self.mid_start = None
    self.mid_area = 0.0
    self.mid_active_time = 0.0
    self.mid_term = False
    self.quiet = None
    self.earned_axis = 0
    self.last = None
    self.previous = 0.0
    self.stock_path = 0.0
    self.admitted_demand = 0.0
    self.reason = "disabled"
    self.context = None
    self.calibration = None
    self._input_state_valid = False
    self._target_speed = None
    self.events = []
    self.native_intersection_failures = 0
    self.last_stamps = None

  def __getattr__(self, name):
    return getattr(self.reference, name)

  @property
  def input_state_valid(self):
    return self._input_state_valid

  @input_state_valid.setter
  def input_state_valid(self, v):
    self._input_state_valid = bool(v)
    self.reference.input_state_valid = bool(v)

  @property
  def target_speed(self):
    return self._target_speed

  @target_speed.setter
  def target_speed(self, v):
    self._target_speed = v
    self.reference.target_speed = v

  def set_context(self, **kw):
    self.context = kw.copy()
    self.reference.set_context(**kw)

  def set_calibration(self, **kw):
    self.calibration = kw.copy()
    self.reference.set_calibration(**kw)

  def set_motion(self, **kw):
    self.motion = kw.copy()
    self.reference.set_motion(**kw)

  def clear(self, reason):
    self.rows.clear()
    self.earned_axis = 0
    self.quiet = None
    self.reason = reason

  def evidence(self, CC, CS, a):
    s = CS.out
    c = self.context
    p = self.calibration
    m = self.motion
    if (
      not self.input_state_valid
      or c is None
      or p is None
      or m is None
      or not CC.latActive
      or not getattr(s, "canValid", False)
      or any(
        bool(getattr(s, k, False))
        for k in ["steeringPressed", "brakePressed", "gasPressed", "leftBlinker", "rightBlinker", "steerFaultTemporary", "steerFaultPermanent"]
      )
    ):
      return None
    now = c["now_ns"]
    v = float(s.vEgoRaw)
    mv = float(self.target_speed if self.target_speed is not None else s.vEgo)
    k = float(a.curvature)
    if not (c["model_valid"] and not c["lane_change"] and p["valid"] and m["valid"] and p["now_ns"] == m["now_ns"] == now):
      return None
    stamps = [c["state_ns"], c["control_ns"], m["sample_ns"]]
    if not all(
      0 < t <= now and now - t <= lim
      for t, lim in [
        (stamps[0], 40_000_000),
        (stamps[1], 40_000_000),
        (c["model_ns"], 150_000_000),
        (p["sample_ns"], 150_000_000),
        (stamps[2], 100_000_000),
        (m["received_ns"], 100_000_000),
      ]
    ):
      return None
    if m["sample_ns"] > m["received_ns"]:
      return None
    vals = [v, mv, k, p["roll"], p["ratio"], p["stiffness"], p["offset_deg"], m["yaw_rate"], m["yaw_std"], a.steeringAngleDeg, s.steeringAngleDeg]
    if (
      not all(math.isfinite(x) for x in vals)
      or v <= 0.1
      or mv <= 0.1
      or not 10 <= p["ratio"] <= 35
      or not 0.2 <= p["stiffness"] <= 3
      or abs(p["roll"]) > 0.15
      or abs(p["offset_deg"]) > 10
      or not 0 <= m["yaw_std"] <= 0.03
    ):
      return None
    self.vm.update_params(p["stiffness"], p["ratio"])
    expected = math.degrees(self.vm.get_steer_from_curvature(-k, mv, p["roll"])) + p["offset_deg"]
    target = float(a.steeringAngleDeg) - p["offset_deg"]
    actual = float(s.steeringAngleDeg) - p["offset_deg"]
    no_roll = math.degrees(self.vm.get_steer_from_curvature(-k, mv, 0.0))
    if abs(expected - float(a.steeringAngleDeg)) > 0.5:
      return None
    return dict(
      t=now, stamps=stamps, v=v, k=k, target=target, actual=actual, no_roll=no_roll, yaw=m["yaw_rate"], std=m["yaw_std"], generation=m.get("generation", 0)
    )

  def update(self, CC, CS, a):
    stock = self.reference.update(CC, CS, a)
    b = float(stock.path_angle)
    self.stock_path = b
    self.admitted_demand = 0.0
    if not self.enabled:
      self.term = 0.0
      self.mid_term = False
      self.previous = b
      self.reason = "disabled"
      return stock
    e = self.evidence(CC, CS, a)
    desired = 0.0
    projected_wheel_error = 0.0
    self.reason = "guard"
    now = self.context["now_ns"] if self.context else 0
    dt = (now - self.last) / 1e9 if self.last is not None else 0.05
    self.last = now
    previous_term = self.term
    previous_mid = self.mid_term
    cadence_good = 0.04 <= dt <= 0.06
    charge_dt = dt if cadence_good else max(0.06, abs(dt))
    valid = e is not None and 0.04 <= dt <= 0.06
    if valid:
      if self.last_stamps is not None and any(x <= y for x, y in zip(e["stamps"], self.last_stamps, strict=False)):
        valid = False
      else:
        self.last_stamps = e["stamps"]
    if not valid:
      self.clear("invalid_or_cadence")
    else:
      if e["generation"] != self.generation:
        self.clear("generation")
        self.generation = e["generation"]
      target = e["target"]
      axis = 1 if target > 0 else -1
      demand = abs(e["k"]) * e["v"] ** 2
      excess = math.copysign(1.0, e["k"]) * e["yaw"] * e["v"] - demand
      quiet = abs(e["k"]) < 0.00015 and abs(e["actual"] - target) <= 0.5 and abs(e["yaw"] * e["v"]) < 0.1
      if quiet:
        self.quiet = now if self.quiet is None else self.quiet
        if now - self.quiet >= 1_000_000_000 and self.term == 0:
          self.area = 0.0
          self.active_time = 0.0
          self.start = self.mid_start = None
          self.mid_area = self.mid_active_time = 0.0
          self.earned_axis = 0
      else:
        self.quiet = None
      if demand < 0.25 or abs(target) < 1.0 or target * e["actual"] <= 0 or target * e["no_roll"] <= 0:
        self.rows.clear()
        self.earned_axis = 0
        self.reason = "neutral_or_direction"
        if not quiet:
          self.quiet = None
      else:
        if self.rows and (any(x <= y for x, y in zip(e["stamps"], self.rows[-1]["stamps"], strict=False)) or e["k"] * self.rows[-1]["k"] <= 0):
          self.clear("source_or_reversal")
        e["error"] = axis * (e["actual"] - target)
        e["excess"] = excess
        e["demand"] = demand
        self.rows.append(e)
        self.earned_axis = axis if any(abs(r["target"]) >= 1.5 for r in self.rows) else 0
        mph = e["v"] / 0.44704
        low = smoothstep((mph - 40) / 3) * (1 - smoothstep((mph - 50) / 5))
        mid = smoothstep((mph - 62) / 2) * (1 - smoothstep((mph - 72) / 2))
        high = smoothstep((mph - 75) / 2) * (1 - smoothstep((mph - 83) / 2))
        if len(self.rows) == 5 and 0.16 <= (now - self.rows[0]["t"]) / 1e9 <= 0.30:
          rr = list(self.rows)
          tt = np.array([(r["t"] - rr[0]["t"]) / 1e9 for r in rr])
          xx = np.array([r["error"] for r in rr])
          kk = np.array([abs(r["k"]) for r in rr])
          den = float(np.sum((tt - tt.mean()) ** 2))
          es = float(np.sum((tt - tt.mean()) * (xx - xx.mean())) / den)
          ks = float(np.sum((tt - tt.mean()) * (kk - kk.mean())) / den)
          # Low unwind avoids spending extra trim during already-fast recovery.
          # Keep the highway confirmation horizon exactly unchanged.
          clearance_horizon = 0.3 if low > 0 or mid > 0 else 0.15
          projected_wheel_error = min(e["error"], e["error"] + clearance_horizon * es)
          useful = projected_wheel_error - 0.5
          body = excess > max(0.1, 0.1 * demand) and np.median([r["excess"] for r in rr[-3:]]) > max(0.1, 0.1 * demand)
          coherent = all(r["target"] * target > 0 and r["error"] > 0.5 for r in rr)
          if useful > 0 and body and coherent:
            if low > 0 and ks < -0.1 * abs(e["k"]) and kk[-1] <= 0.95 * max(kk):
              desired = min(0.003, 0.05 * abs(b), 0.15 * (-ks) * abs(b) / max(abs(e["k"]), 1e-8)) * low
              self.reason = "unwind"
            elif (
              mid > 0
              and ks < -0.1 * abs(e["k"])
              and kk[-1] <= 0.95 * max(kk)
              and all(r["excess"] > max(0.1, 0.1 * r["demand"]) for r in rr)
              and all(x >= y for x, y in zip(kk, kk[1:], strict=False))
            ):
              # Admit a small, fresh dual-feedback unwind using the installed
              # transient budget, release reserve and native angle/rate checks.
              desired = min(0.0015, 0.05 * abs(b), 0.15 * (-ks) * abs(b) / max(abs(e["k"]), 1e-8)) * mid
              self.reason = "mid_high_unwind"
            elif high > 0 and self.earned_axis == axis:
              desired = min(0.002, 0.10 * abs(b)) * high
              self.reason = "near_neutral"
          else:
            self.reason = "closing_or_unconfirmed"
    if not CC.latActive or self.reference.human_turn_active:
      if not cadence_good:
        self.area += previous_term * charge_dt
        self.active_time += charge_dt if previous_term > 1e-12 else 0.0
        if previous_mid and previous_term > 1e-12:
          self.mid_area += previous_term * charge_dt
          self.mid_active_time += charge_dt
        if self.area > 0.0015 + 1e-12 or self.active_time > 1.0 + 1e-9:
          raise AssertionError("unobserved hold exceeds prototype allowance")
      self.term = 0.0
      self.mid_term = False
      self.previous = b
      return stock
    raw = float(getattr(self.reference, "actual_reference_path", b))
    remaining = max(0.0, min(0.01, 0.15 * abs(raw)) - max(0.0, abs(raw) - abs(b))) if raw * b > 0 else 0.0
    desired = min(desired, remaining)
    self.admitted_demand = desired
    # Before first spend require both meaningful body excess and the
    # existing earned-axis wheel-angle scale on the same fresh sample.
    # Continuation, inherited speed bands and all shared limits stay unchanged.
    mid_proposal = self.reason == "mid_high_unwind" and desired > 0
    # A fresh burst must requalify, but elapsed gaps never restore any
    # shared or midband area or active time. Legacy episode age is untouched.
    if not mid_proposal and not previous_mid:
      self.mid_start = None
    if mid_proposal and self.mid_start is None and (e["excess"] < 0.25 or np.median([r["excess"] for r in self.rows]) < 0.25):
      desired = 0.0
      self.reason = "mid_onset_evidence"
    start = self.mid_start if mid_proposal else self.start
    # Reserve shared time for an eventual transition into the installed
    # near-neutral band. Spend only a short exploratory allowance near that
    # transition and only for persistent coherent wheel excess.
    mid_time_cap = 0.675
    if mid_proposal and e["v"] >= 70 * 0.44704:
      mid_time_cap = 0.15
      if projected_wheel_error <= 0.5 * abs(e["target"]):
        desired = 0.0
        self.reason = "legacy_reserve"
    if desired > 0 and ((start is not None and now - start >= 8_000_000_000) or self.active_time >= 0.9
                        or (mid_proposal and self.mid_active_time + charge_dt > mid_time_cap + 1e-9)):
      desired = 0.0
      self.reason = "duration"
    if desired > 0 and self.start is None and not mid_proposal:
      self.start = now
    # Allocate only the deliverable onset step; retain the full release reserve.
    # Spent area is never refunded by a smaller request or an evidence reset.
    affordable = max(0.0, (0.0015 - 0.00015 - self.area) / charge_dt)
    # The midband can use at most one quarter of the common admission
    # allowance. Legacy pacing uses its own delivered usage, and the shared
    # admission and release caps still bound the combined command.
    if mid_proposal:
      affordable = min(affordable, max(0.0, (0.0003375 - self.mid_area) / charge_dt))
    deliverable = min(desired, self.term + 0.0003)
    if self.reason in ("unwind", "mid_high_unwind") and deliverable > 0:
      # Spend the remaining admission area over the remaining active allowance.
      # No forecast, new allowance or repayment from hypothetical changed motion.
      if mid_proposal:
        horizon = max(charge_dt, 0.225)
        paced = max(0.0, (0.0003375 - self.mid_area) / horizon)
      else:
        horizon = max(charge_dt, 0.9 - (self.active_time - self.mid_active_time))
        paced = max(0.0, (0.0015 - 0.00015 - (self.area - self.mid_area)) / horizon)
      deliverable = min(deliverable, paced)
    desired = min(deliverable, affordable)
    if desired < deliverable:
      self.reason = "budget_partial" if desired > 0 else "budget"
    wanted = b - math.copysign(desired, b) if desired else b
    rate = float(np.interp(float(CS.out.vEgoRaw), _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES))
    cap = min(0.003, remaining)
    lo = max(min(b, b - math.copysign(cap, b)), self.previous - rate)
    hi = min(max(b, b - math.copysign(cap, b)), self.previous + rate)
    if lo > hi + 1e-12:
      self.native_intersection_failures += 1
      raise AssertionError("prototype envelopes have no intersection")
    value = max(lo, min(hi, wanted))
    # A fresh middle-band onset has no ECU effect if the exact Ford path
    # field is unchanged after the controller's sign inversion and packing.
    # Leave release/rate-limited outputs alone and charge actual raw trims.
    if mid_proposal and desired > 0 and previous_term == 0 and lo <= b <= hi:
      signal = self.path_signal
      if math.floor((-value - signal.offset) / signal.factor + 0.5) == math.floor((-b - signal.offset) / signal.factor + 0.5):
        value = b
        self.reason = "mid_quantized_no_effect"
    self.term = abs(b - value)
    self.mid_term = self.term > 1e-12 and (mid_proposal and desired > 0 or previous_mid and desired == 0)
    if self.mid_term and self.mid_start is None:
      self.mid_start = now
    charged_term = self.term if cadence_good else max(previous_term, self.term)
    self.area += charged_term * charge_dt
    self.active_time += charge_dt if charged_term > 1e-12 else 0.0
    if charged_term > 1e-12 and (self.mid_term or (not cadence_good and previous_mid)):
      self.mid_area += charged_term * charge_dt
      self.mid_active_time += charge_dt
    self.previous = value
    if self.active_time > 1.0 + 1e-9:
      raise AssertionError("prototype release exceeds active duration")
    if self.area > 0.0015 + 1e-12:
      raise AssertionError("prototype release exceeds sampled budget")
    return AngleLateralResult(path_angle=value, shadow_curvature=stock.shadow_curvature)



class WheelTargetTrial(UnwindAllocation):
  """Shadow the installed strategy and select only scoped wheel-target reductions."""

  def __init__(self, reference):
    super().__init__(reference, enabled=True)
    self.preview = PlannedExit()
    self.model = None
    self.model_ns = 0
    self.previous_output = 0.0
    self.handoff_pending = False
    self.previous_curvature = 0.0
    self.preview_eligible = False
    self.wheel_seeded = False
    self.shadow_reduction = 0.0
    leaf = reference.unchanged.stock.upper_reference
    from opendbc.car.ford.lateral_angle_wheel_target import CalibratedWheelTargetReduction
    assert type(leaf) is CalibratedWheelTargetReduction
    from types import MethodType

    def fraction(policy, CC, CS, a):
      value = CalibratedWheelTargetReduction._fraction(policy, CC, CS, a)
      self.wheel_fraction = value
      self.wheel_release = policy.release_pending
      self.stock_reduction_before = policy.reduction
      self.wheel_reason_selected = policy.wheel_reason
      self.body_reason_selected = policy.body_reason
      return value

    leaf._fraction = MethodType(fraction, leaf)
    self.selected_leaf = leaf

  def evidence(self, CC, CS, a):
    self.latest_evidence = super().evidence(CC, CS, a)
    return self.latest_evidence

  def set_path_preview(self, model, stamp):
    self.model = model
    self.model_ns = stamp

  def update(self, CC, CS, a):
    v = float(CS.out.vEgoRaw)
    k = float(a.curvature)
    mph = v / .44704
    self.wheel_seeded = False
    self.wheel_reason_selected = "not_called"
    self.body_reason_selected = "not_called"
    self.wheel_fraction = 0.0
    self.latest_evidence = None
    original = super().update(CC, CS, a)
    context = self.context
    direction = int(math.copysign(1, k)) if k else 0
    eligible = 82 < mph < 84 and self.latest_evidence is not None
    model = self.model if eligible and self.model_ns == context["model_ns"] else None
    now = context["now_ns"] if context else 0
    self.preview_eligible = self.preview.observe(model, self.model_ns, now, direction)
    falling = self.previous_curvature * k > 0 and abs(k) < abs(self.previous_curvature)
    self.previous_curvature = k
    value = original.path_angle
    steer = CC.latActive and not self.reference.human_turn_active
    rate = float(np.interp(v, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES)) if self.preview_eligible or self.handoff_pending else 0.0
    leaf = self.selected_leaf
    can_track = (steer and not getattr(CS.out, "steeringPressed", False) and self.latest_evidence is not None
                 and self.wheel_fraction > 0 and not self.wheel_release
                 and self.wheel_reason_selected == "confirmed_target_excess")
    if can_track and self.preview_eligible and falling and self.stock_reduction_before == 0 and self.shadow_reduction == 0:
      self.wheel_seeded = True
    if can_track and (self.wheel_seeded or self.shadow_reduction > 0):
      reference_path = leaf.path_angle_last - leaf.final_residual
      target = min(leaf.MAX_RAD, abs(reference_path) * self.wheel_fraction)
      self.shadow_reduction = min(target, self.shadow_reduction + leaf.ONSET_RAD_PER_FRAME)
      if self.preview_eligible and self.term == 0 and abs(value - leaf.path_angle_last) <= 1e-12:
        trial_path = reference_path - math.copysign(self.shadow_reduction, reference_path)
        if value * trial_path > 0 and abs(trial_path) < abs(value):
          reduced = math.copysign(max(abs(trial_path), abs(value) - .0009), value)
          if abs(reduced - self.previous_output) <= rate:
            value = reduced
            self.reason = "wheel_target_exit"
      if leaf.reduction >= self.shadow_reduction:
        self.shadow_reduction = 0.0
    else:
      self.shadow_reduction = 0.0
    if steer and self.handoff_pending:
      limited = max(self.previous_output - rate, min(self.previous_output + rate, value))
      if limited != value:
        self.reason = "wheel_target_handoff"
      value = limited
    self.handoff_pending = steer and value != original.path_angle
    self.previous_output = value
    return AngleLateralResult(path_angle=value, shadow_curvature=original.shadow_curvature)


def create_tracking_trial(CP, *, enabled=False, curve_scope_enabled=False):
  """Select the wrapper only for the existing Navigator prerequisite flags."""
  from opendbc.car.ford.values import CAR, FordFlags
  from opendbc.car.ford.lateral_angle_midrange_admission import MidrangeAdmission

  if not enabled or CP.carFingerprint != CAR.LINCOLN_NAVIGATOR_MK4 or not CP.flags & FordFlags.PINION_CURVATURE or not CP.flags & FordFlags.LOW_SPEED_GAIN:
    return None
  strategy = WheelTargetTrial(MidrangeAdmission(CP, enabled=True))
  if curve_scope_enabled is True:
    from opendbc.car.ford.lateral_angle_wheel_target import enable_calibrated_curve_scope
    enable_calibrated_curve_scope(strategy)
  return strategy
