"""Bound weak Navigator brake entries while retaining measured clearance."""

import math
from opendbc.car.ford.longitudinal_admission import Admission, BrakeAdmission


class BoundedCoastAdmission(BrakeAdmission):
  budget = 0.01  # Experimental accumulated speed-debt, m/s.
  deadline = 0.3  # Experimental continuous-negative decision deadline, seconds.

  def update(self, now, eligible, baseline, previous, raw, net, cleared, torque):
    dt = None if self.last is None else now - self.last
    self.last = now if math.isfinite(now) else None
    fresh = (
      torque is not None
      and torque.valid
      and all(math.isfinite(x) for x in [torque.t, torque.value])
      and 0 <= now - torque.t <= 0.1
      and 0 <= torque.value <= 4.0
    )
    scope = (
      eligible
      and not previous
      and cleared
      and fresh
      and all(math.isfinite(x) for x in [now, raw, net])
      and raw > 0.02
      and net > -0.05 + 1e-6
      and dt is not None
      and 0 < dt <= 0.040000001
    )
    if not scope:
      self.since = None
      self.area = 0.0
      return Admission(bool(baseline))
    if net >= 0:
      self.area = max(0.0, self.area - net * dt)
      self.since = None
      return Admission(bool(baseline))
    if not baseline:
      self.since = None
      self.area = 0.0
      return Admission(False)
    self.area += -net * dt
    if self.since is None:
      self.since = now
    if self.area + 1e-9 >= self.budget or now - self.since + 1e-9 >= self.deadline:
      self.since = None
      self.area = 0.0
      return Admission(True, False, "bounded_coast_admit")
    # Existing measured observer still runs; qualification alone does not reset propulsion.
    return Admission(False, False, "bounded_coast_clear")
