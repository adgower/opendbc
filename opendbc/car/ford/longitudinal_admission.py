"""Bound mild grade-crossing brake admission and observe received brake clearance."""

from dataclasses import dataclass
import math
from opendbc.car.ford.longitudinal_coordination import ReleaseBarrier


@dataclass(frozen=True)
class Admission:
  brake: bool
  coast: bool = False
  reason: str = "baseline"


class BrakeAdmission:
  def __init__(self):
    self.last = None
    self.since = None
    self.area = 0.0

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
      and -0.05 + 1e-6 < net
      and dt is not None
      and 0 < dt <= 0.040000001
    )
    if not scope:
      self.since = None
      self.area = 0.0
      return Admission(bool(baseline))
    if net >= 0.0:
      # Tiny positive crossings must not reset accumulated negative exposure.
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
    if now - self.since + 1e-9 >= 0.04 or self.area + 1e-9 >= 0.002:
      self.since = None
      self.area = 0.0
      return Admission(True, False, "qualified_mild_brake")
    return Admission(False, True, "qualify_mild_brake_coast")


class ObservedClearBarrier(ReleaseBarrier):
  """Never infer clearance from absence of our own command after coasting."""

  def reset(self):
    super().reset()
    self.observed_last = None

  def update(self, now, active, brake, speed, phase, torque, *, owner_active=False):
    if self.enabled and not self.fault and active and phase == "pid" and speed >= 3.0 and not brake:
      fresh = (
        torque is not None
        and torque.valid
        and all(math.isfinite(x) for x in [torque.t, torque.value])
        and 0 <= now - torque.t <= 0.1
        and 0 <= torque.value <= 32756.0
      )
      if not fresh:
        return self.fail("invalid_release_feedback")
      if self.observed_last is not None and torque.t < self.observed_last:
        return self.fail("reversed_release_feedback")
      self.observed_last = torque.t
      if torque.value > 4.0 and not self.needs:
        self.needs = True
        self.started = None
        self.clear = []
        self.last_sample = None
    return super().update(now, active, brake, speed, phase, torque, owner_active=owner_active)
