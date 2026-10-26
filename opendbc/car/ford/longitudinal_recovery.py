"""Bounded rolling-demand reduction and anti-windup for its local constraint."""

import math
from collections import Counter

from opendbc.car.ford.longitudinal_coordination import allocator_feedback_valid, freeze_integral


class RollingGuard:
  def __init__(self):
    self.counts = Counter()
    self.reset()

  def reset(self):
    self.last = None
    self.epoch = None
    self.sequence = None
    self.moving = False
    self.brake_seen = None
    self.start = None
    self.spent = False
    self.context = False
    self.last_reason = "reset"
    self.withheld = 0.0
    self.growth_cost = 0.0

  def finish(self, reason):
    if self.start is not None:
      self.spent = True
    self.last_reason = reason

  def observe(self, active, cs, target, should_stop, feedback, now, epoch, sequence, coordinated):
    gap = None if self.last is None else now - self.last
    if (
      not coordinated
      or not active
      or not cs.canValid
      or cs.gasPressed
      or cs.brakePressed
      or not all(math.isfinite(v) for v in [now, cs.vEgo, target])
      or epoch <= 0
      or sequence <= 0
    ):
      self.reset()
      return
    if self.epoch != epoch or (gap is not None and not 0 < gap <= 0.100000001) or (self.sequence is not None and sequence < self.sequence):
      self.reset()
    self.last = now
    self.epoch = epoch
    self.sequence = sequence
    self.context = False
    self.growth_cost = 0.0
    if cs.standstill or cs.cruiseState.standstill or cs.vEgo <= 0.3:
      self.moving = False
      self.brake_seen = None
      self.start = None
      self.spent = False
      self.withheld = 0.0
      self.last_reason = "actual_stop"
      return
    if cs.vEgo >= 4.0:
      self.moving = True
      self.brake_seen = None
      self.start = None
      self.spent = False
      self.withheld = 0.0
      self.last_reason = "normal_motion"
      return
    f = feedback
    fresh = (
      f is not None
      and all(math.isfinite(v) for v in [f.t, f.gas, f.requested])
      and 0 <= now - f.t <= 0.100000001
      and f.epoch == epoch
      and 0 < f.sequence <= sequence
      and f.valid
      and not f.fault
      and allocator_feedback_valid(f)
    )
    if not fresh:
      self.brake_seen = None
      self.finish("invalid_feedback")
      return
    braking = f.brake or f.precharge or f.reference_brake
    if self.moving and braking:
      if self.brake_seen is None:
        self.counts["armed_approaches"] += 1
      self.brake_seen = now
      self.finish("braking")
    if should_stop or target < 0:
      self.finish("signed_brake_recovery")
      return
    if cs.vEgo >= 3.0:
      self.finish("moving_reference")
      return
    if f.allocator_phase != "engineRamp" or not f.limited:
      self.finish("authority_returned")
      return
    if self.start is not None and now - self.start >= 3.0 - 1e-9:
      self.spent = True
      self.last_reason = "deadline"
      return
    recent = self.start is not None or (self.brake_seen is not None and 0 <= now - self.brake_seen <= 3.0)
    self.context = bool(self.moving and recent and not self.spent and not braking and f.gas >= 0)
    self.last_reason = "rolling_ready" if self.context else "unqualified"

  def freeze(self, now, epoch, sequence, error, target, wanted, speed, phase, feedback, enabled=False):
    nominal = freeze_integral(now, epoch, sequence, error, target, wanted, speed, phase, feedback, enabled=enabled)
    if nominal:
      return True
    f = feedback
    hold = (
      enabled
      and self.context
      and self.last == now
      and self.epoch == epoch
      and f is not None
      and all(math.isfinite(v) for v in [now, error, target, wanted, speed, f.t, f.gas, f.requested])
      and phase == "pid"
      and 0.3 < speed < 3.0
      and error > 0
      and target >= 0
      and wanted > 0.02
      and f.requested > 0.02
      and wanted > f.gas + 0.011
    )
    hold = bool(hold and math.isfinite(self.growth_cost) and self.growth_cost > 1e-12)
    if hold and self.withheld + self.growth_cost > 0.25 + 1e-12:
      self.spent = True
      self.context = False
      self.last_reason = "integral_budget"
      return False
    if hold:
      if self.start is None:
        self.start = now
        self.counts["held_episodes"] += 1
      self.withheld += self.growth_cost
      self.counts["max_withheld"] = max(self.counts["max_withheld"], self.withheld)
      self.counts["total_withheld"] += self.growth_cost
      self.counts["held_updates"] += 1
      self.last_reason = "rolling_hold"
    return bool(hold)


class CommandRecovery:
  def __init__(self):
    self.rolling_guard = RollingGuard()
    self.counts = Counter()
    self.reset()

  def reset(self):
    self.rolling_guard.reset()
    self.reduction = 0.0
    self.overlay_epoch = None
    self.overlay_origin = None
    self.previous_time = None
    self.previous_sequence = None
    self.previous_epoch = None
    self.last_command_reduction = 0.0
    self.valid = False
    self.sample = None

  def prepare(self, active, CS, a_target, should_stop, feedback, now, epoch, sequence, coordinated, source_valid, urgent):
    dt = None if self.previous_time is None else now - self.previous_time
    continuity = self.previous_sequence is None or (epoch == self.previous_epoch and sequence > self.previous_sequence)
    clock_ok = dt is None or 0 < dt <= 0.100000001
    if not continuity or not clock_ok:
      self.counts["invalidated_reduction"] += self.reduction
      self.reduction = 0.0
      self.overlay_epoch = None
      self.overlay_origin = None
      self.rolling_guard.reset()
    self.previous_time = now
    self.previous_epoch = epoch
    self.previous_sequence = sequence
    self.last_command_reduction = 0.0
    f = feedback
    valid = bool(
      continuity
      and clock_ok
      and active
      and coordinated
      and source_valid
      and CS.canValid
      and not CS.gasPressed
      and not CS.brakePressed
      and not should_stop
      and not urgent
      and all(math.isfinite(v) for v in [now, a_target, CS.aEgo, CS.vEgo])
      and f is not None
      and all(math.isfinite(v) for v in [f.t, f.requested, f.gas])
      and f.valid
      and not f.fault
      and f.epoch == epoch
      and epoch > 0
      and 0 < f.sequence <= sequence
      and 0 <= now - f.t <= 0.100000001
      and dt is not None
      and 0 < dt <= 0.100000001
      and allocator_feedback_valid(f)
      and (self.overlay_epoch is None or self.overlay_epoch == epoch)
    )
    self.rolling_guard.observe(active, CS, a_target, should_stop, f, now, epoch, sequence, coordinated and source_valid)
    if self.reduction > 0 and self.rolling_guard.start != self.overlay_origin:
      self.rolling_guard.context = False
    self.valid = valid
    self.sample = (now, epoch, sequence, a_target, CS.aEgo, CS.vEgo, feedback, dt)

  def freeze(self, t, owner, seq, error, target, wanted, speed, phase, fb, enabled=False):
    nominal = freeze_integral(t, owner, seq, error, target, wanted, speed, phase, fb, enabled=enabled)
    local_limit = bool(
      self.valid
      and enabled
      and self.reduction > 0
      and phase == "pid"
      and 3.0 <= speed <= 12.0
      and error > 0
      and target > 0
      and wanted > 0.02
      and fb is not None
      and not fb.brake
      and not fb.precharge
      and not fb.reference_brake
      and fb.gas >= 0
      and fb.requested > 0.02
      and wanted > fb.gas + 0.011
    )
    if local_limit:
      self.counts["local_constraint_freeze"] += 1
    return nominal or local_limit

  def apply(self, output, integral_delta, wanted, phase):
    if self.sample is None:
      return output
    now, epoch, sequence, a_target, actual_a, speed, f, dt = self.sample
    if self.valid and phase == "pid":
      previous = self.rolling_guard.counts["total_withheld"]
      self.rolling_guard.growth_cost = max(0.0, integral_delta)
      if self.rolling_guard.context:
        self.rolling_guard.freeze(now, epoch, sequence, a_target - actual_a, a_target, wanted, speed, "pid", f, enabled=True)
      withheld = self.rolling_guard.counts["total_withheld"] - previous
      if withheld > 0:
        self.reduction = min(0.25, self.reduction + withheld)
        self.overlay_epoch = epoch
        self.overlay_origin = self.rolling_guard.start
        self.counts["admitted"] += withheld
      elif (
        3.0 <= speed <= 12.0
        and a_target > 0
        and output > 0.02
        and not f.brake
        and not f.precharge
        and not f.reference_brake
        and not f.limited
        and f.allocator_phase == "drive"
        and f.gas >= 0
        and f.requested > 0.02
        and abs(f.gas - f.requested) <= 0.02
      ):
        step = min(self.reduction, 0.001, 0.1 * dt)
        self.reduction = max(0.0, self.reduction - step)
        self.counts["released"] += step
        if self.reduction <= 1e-12:
          self.reduction = 0.0
          self.overlay_epoch = None
          self.overlay_origin = None
      if (
        a_target > 0
        and output > 0.02
        and 0.3 < speed <= 12.0
        and not f.brake
        and not f.precharge
        and not f.reference_brake
        and f.gas >= 0
        and f.allocator_phase in ("engineRamp", "drive")
      ):
        self.last_command_reduction = min(self.reduction, max(0.0, float(output) - 0.02))
        output = float(output) - self.last_command_reduction

    self.counts["max_reduction"] = max(self.counts["max_reduction"], self.reduction)
    return output
