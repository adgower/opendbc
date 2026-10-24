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
