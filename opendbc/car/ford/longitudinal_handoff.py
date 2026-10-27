"""Ford propulsion withdrawal and bounded request resumption."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Result:
  gas: float
  state: str
  reason: str


class Handoff:
  def __init__(self, rise_rate=0.5, qualify_s=0.02):
    assert math.isfinite(rise_rate) and rise_rate > 0
    assert math.isfinite(qualify_s) and qualify_s >= 0.02
    self.rise_rate = rise_rate
    self.qualify_s = qualify_s
    self.last_time = None
    self.since = None
    self.last_valid = None

  def withdraw(self, state, reason):
    self.since = None
    self.last_valid = None
    return Result(-5.0, state, reason)

  def update(self, now, active, brake, target, pitch_valid, stopping, *, bounded_coast=False):
    if not math.isfinite(now):
      self.last_time = None
      return self.withdraw("inactive", "invalid_time")
    dt = None if self.last_time is None else now - self.last_time
    self.last_time = now
    if dt is not None and (dt <= 0 or dt > 0.100000001):
      return self.withdraw("inactive", "time_discontinuity")
    if not active:
      return self.withdraw("inactive", "disengaged")
    if brake:
      return self.withdraw("braking", "baseline_brake_latched")
    if stopping:
      return self.withdraw("braking", "stopping")
    if not pitch_valid:
      return self.withdraw("inactive", "pitch_unavailable")
    if bounded_coast:
      self.withdraw("coast", "bounded_true_coast")
      return Result(-5., "qualify", "bounded_true_coast")
    if not math.isfinite(target) or target < -0.5 or target > 2.0:
      return self.withdraw("coast", "inactive_or_invalid_request")
    if self.since is None:
      self.since = now
    if now - self.since + 1e-9 < self.qualify_s:
      return Result(-5.0, "qualify", "brake_clear_qualification")
    if self.last_valid is None:
      gas = min(target, 0.0)
    else:
      gas = min(target, self.last_valid + self.rise_rate * dt)
    self.last_valid = gas
    return Result(gas, "resume" if gas < target - 1e-9 else "drive", "bounded_rise" if gas < target - 1e-9 else "at_target")
