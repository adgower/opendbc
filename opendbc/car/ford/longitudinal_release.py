"""Qualified Ford brake release without changing brake acceleration."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ReleaseResult:
  brake: bool
  reason: str


class ReleaseGate:
  def __init__(self, dwell=0.06, positive_margin=0.02):
    assert math.isfinite(dwell) and dwell >= 0.02
    assert math.isfinite(positive_margin) and positive_margin >= 0
    self.dwell = dwell
    self.positive_margin = positive_margin
    self.last_time = None
    self.since = None

  def hold(self, brake, reason):
    self.since = None
    return ReleaseResult(bool(brake), reason)

  def update(self, now, active, baseline_brake, previous_brake, raw, net, phase, pitch_valid):
    if not active:
      self.last_time = now if math.isfinite(now) else None
      return self.hold(baseline_brake, "disengaged")
    retained = bool(baseline_brake or previous_brake)
    if not math.isfinite(now):
      self.last_time = None
      return self.hold(retained, "invalid_time")
    dt = None if self.last_time is None else now - self.last_time
    self.last_time = now
    if dt is not None and (dt <= 0 or dt > 0.100000001):
      return self.hold(retained, "time_discontinuity")
    if not pitch_valid or not all(math.isfinite(x) for x in [raw, net]):
      return self.hold(retained, "invalid_pitch_or_demand")
    if phase not in ["pid", "starting"]:
      return self.hold(retained, "not_driving_phase")
    if not baseline_brake:
      return self.hold(False, "baseline_released")
    if raw <= self.positive_margin or net <= self.positive_margin:
      return self.hold(True, "braking_or_nonpositive_intent")
    if self.since is None:
      self.since = now
    if now - self.since + 1e-9 < self.dwell:
      return ReleaseResult(True, "positive_intent_qualification")
    return ReleaseResult(False, "qualified_positive_release")
