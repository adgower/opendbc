"""Bounded Navigator unwind allocation and fresh near-neutral confirmation.

Pure strategy: existing opt-in only, no I/O or new settings. Invariant failures
stop control rather than bypass native envelopes or refund unknown exposure.
"""

import math
from collections import deque
import numpy as np
from opendbc.car.vehicle_model import VehicleModel
from opendbc.car.ford.lateral_angle import AngleLateralResult, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES
from opendbc.car.ford.lateral_angle_low_speed_prototype import smoothstep


class UnwindAllocation:
  def __init__(self, reference, *, enabled=False):
    self.reference = reference
    self.CP = reference.CP
    self.enabled = enabled is True
    self.vm = VehicleModel(self.CP)
    self.rows = deque(maxlen=5)
    self.motion = None
    self.generation = None
    self.term = 0.0
    self.area = 0.0
    self.active_time = 0.0
    self.start = None
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
      self.previous = b
      self.reason = "disabled"
      return stock
    e = self.evidence(CC, CS, a)
    desired = 0.0
    self.reason = "guard"
    now = self.context["now_ns"] if self.context else 0
    dt = (now - self.last) / 1e9 if self.last is not None else 0.05
    self.last = now
    previous_term = self.term
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
          self.start = None
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
          clearance_horizon = 0.3 if low > 0 else 0.15
          useful = min(e["error"], e["error"] + clearance_horizon * es) - 0.5
          body = excess > max(0.1, 0.1 * demand) and np.median([r["excess"] for r in rr[-3:]]) > max(0.1, 0.1 * demand)
          coherent = all(r["target"] * target > 0 and r["error"] > 0.5 for r in rr)
          if useful > 0 and body and coherent:
            if low > 0 and ks < -0.1 * abs(e["k"]) and kk[-1] <= 0.95 * max(kk):
              desired = min(0.003, 0.05 * abs(b), 0.15 * (-ks) * abs(b) / max(abs(e["k"]), 1e-8)) * low
              self.reason = "unwind"
            elif high > 0 and self.earned_axis == axis:
              desired = min(0.002, 0.10 * abs(b)) * high
              self.reason = "near_neutral"
          else:
            self.reason = "closing_or_unconfirmed"
    if not CC.latActive or self.reference.human_turn_active:
      if not cadence_good:
        self.area += previous_term * charge_dt
        self.active_time += charge_dt if previous_term > 1e-12 else 0.0
        if self.area > 0.0015 + 1e-12 or self.active_time > 1.0 + 1e-9:
          raise AssertionError("unobserved hold exceeds prototype allowance")
      self.term = 0.0
      self.previous = b
      return stock
    raw = float(getattr(self.reference, "actual_reference_path", b))
    remaining = max(0.0, min(0.01, 0.15 * abs(raw)) - max(0.0, abs(raw) - abs(b))) if raw * b > 0 else 0.0
    desired = min(desired, remaining)
    self.admitted_demand = desired
    if desired > 0 and ((self.start is not None and now - self.start >= 8_000_000_000) or self.active_time >= 0.9):
      desired = 0.0
      self.reason = "duration"
    if desired > 0 and self.start is None:
      self.start = now
    # Allocate only the deliverable onset step; retain the full release reserve.
    # Spent area is never refunded by a smaller request or an evidence reset.
    affordable = max(0.0, (0.0015 - 0.00015 - self.area) / charge_dt)
    deliverable = min(desired, self.term + 0.0003)
    if self.reason == "unwind" and deliverable > 0:
      # Spend the remaining admission area over the remaining active allowance.
      # No forecast, new allowance or repayment from hypothetical changed motion.
      horizon = max(charge_dt, 0.9 - self.active_time)
      paced = max(0.0, (0.0015 - 0.00015 - self.area) / horizon)
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
    self.term = abs(b - value)
    charged_term = self.term if cadence_good else max(previous_term, self.term)
    self.area += charged_term * charge_dt
    self.active_time += charge_dt if charged_term > 1e-12 else 0.0
    self.previous = value
    if self.active_time > 1.0 + 1e-9:
      raise AssertionError("prototype release exceeds active duration")
    if self.area > 0.0015 + 1e-12:
      raise AssertionError("prototype release exceeds sampled budget")
    return AngleLateralResult(path_angle=value, shadow_curvature=stock.shadow_curvature)


def create_tracking_trial(CP, *, enabled=False):
  """Select the wrapper only for the existing Navigator prerequisite flags."""
  from opendbc.car.ford.values import CAR, FordFlags
  from opendbc.car.ford.lateral_angle_midrange_admission import MidrangeAdmission

  if not enabled or CP.carFingerprint != CAR.FORD_EXPEDITION_MK4 or not CP.flags & FordFlags.PINION_CURVATURE or not CP.flags & FordFlags.LOW_SPEED_GAIN:
    return None
  return UnwindAllocation(MidrangeAdmission(CP, enabled=True), enabled=True)
