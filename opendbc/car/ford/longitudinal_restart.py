"""Retain fresh measured clearance across a speed-only moving-band crossing."""

import math
from opendbc.car.ford.longitudinal_admission import ObservedClearBarrier


class RestartClearance(ObservedClearBarrier):
  def reset(self):
    super().reset()
    self.preclear = []
    self.pre_last_torque = None
    self.pre_last_now = None
    self.pre_last_speed = None
    self.carried = False

  def update(self, now, active, brake, speed, phase, torque, *, owner_active=False):
    self.carried = False
    dt = None if self.pre_last_now is None else now - self.pre_last_now
    fresh = (
      torque is not None
      and torque.valid
      and all(math.isfinite(x) for x in [now, speed, torque.t, torque.value])
      and 0 <= now - torque.t <= 0.1
      and 0 <= torque.value <= 4.0
    )
    eligible = (
      self.enabled
      and not self.fault
      and active
      and owner_active
      and not brake
      and phase == "pid"
      and fresh
      and speed >= 0
      and (dt is None or 0 < dt <= 0.100000001)
    )
    crossing = eligible and self.pre_last_speed is not None and self.pre_last_speed < 3.0 <= speed and dt is not None
    if not eligible:
      self.preclear = []
      self.pre_last_torque = None
    else:
      if self.pre_last_torque is not None:
        gap = torque.t - self.pre_last_torque
        if gap < 0 or gap > 0.050000001:
          self.preclear = []
          # Never carry on the discontinuous update itself.
          crossing = False
      if self.pre_last_torque is None or torque.t != self.pre_last_torque:
        self.preclear.append(torque.t)
      self.pre_last_torque = torque.t
      self.preclear = [t for t in self.preclear if 0 <= now - t <= 0.1]
      if crossing and len(self.preclear) >= 3 and self.preclear[-1] - self.preclear[0] + 1e-9 >= 0.04:
        self.needs = False
        self.started = None
        self.clear = []
        self.last_sample = None
        self.carried = True
    self.pre_last_now = now if math.isfinite(now) else None
    self.pre_last_speed = speed if eligible else None
    return super().update(now, active, brake, speed, phase, torque, owner_active=owner_active)


from dataclasses import dataclass
from opendbc.car.ford.longitudinal_coordination import Release


@dataclass(frozen=True)
class CoastLease:
  enabled: bool
  epoch: int
  sequence: int
  age: float
  remaining: float


class CoastClearance(RestartClearance):
  def reset(self):
    super().reset()
    self.coast_epoch = None
    self.coast_sequence = None

  def update(self, now, active, brake, speed, phase, torque, *, owner_active=False, coast_lease=None):
    ctx = coast_lease
    if ctx is None:
      self.coast_epoch = None
      self.coast_sequence = None
      return super().update(now, active, brake, speed, phase, torque, owner_active=owner_active)
    if self.fault:
      return Release(False, True, self.fault)
    finite = all(math.isfinite(x) for x in [now, speed, ctx.age, ctx.remaining])
    if not finite or not active or not owner_active or brake or phase != "pid" or speed < 3 or not ctx.enabled or ctx.epoch <= 0 or ctx.sequence <= 0:
      return self.fail("invalid_coast_context")
    if self.last is not None and not 0 < now - self.last <= 0.100000001:
      return self.fail("clock_discontinuity")
    if not 0 <= ctx.age < 0.300000001 or not 0 < ctx.remaining <= 0.01:
      return self.fail("invalid_coast_lease")
    if torque is None or not torque.valid or not all(math.isfinite(x) for x in [torque.t, torque.value]) or not 0 <= now - torque.t <= 0.1:
      return self.fail("invalid_release_feedback")
    if self.observed_last is not None and torque.t < self.observed_last:
      return self.fail("reversed_release_feedback")
    if not 0 <= torque.value <= 4:
      return self.fail("coast_lost_observed_clearance")
    if self.coast_epoch is None:
      if self.needs or self.started is not None:
        return self.fail("coast_without_entry_clearance")
      self.coast_epoch = ctx.epoch
    elif ctx.epoch != self.coast_epoch or ctx.sequence < self.coast_sequence:
      return self.fail("coast_owner_changed")
    self.coast_sequence = ctx.sequence
    # Arm a logical obligation, not a physical brake command. Run the original
    # observer with the actual brake=False argument and all original validation.
    self.needs = True
    self.started = None
    self.clear = []
    self.last_sample = None
    result = super().update(now, active, brake, speed, phase, torque, owner_active=owner_active)
    if result.fault:
      return result
    self.needs = True
    self.started = None
    self.clear = []
    self.last_sample = None
    self.preclear = []
    self.pre_last_torque = None
    return Release(False, False, "bounded_coast_pending")
