"""Opt-in earlier onset using independently validated wheel and body histories.

All native limits and the
installed magnitude/rate/predictive-release rules remain in the parent.
"""

import math
from collections import deque
from opendbc.car.ford.lateral_angle_calibrated_mid_unwind import CalibratedMidUnwind as Installed


class EarlierUnwindOnset(Installed):
  def __init__(self, CP, *, enabled=False):
    super().__init__(CP, enabled=enabled)
    self.onset_history = deque(maxlen=5)
    self.onset_generation = None
    self.onset_receipt = None

  def _fraction(self, CC, CS, a):
    fraction = super()._fraction(CC, CS, a)
    self.onset_receipt = None
    observer = self.reference.baseline
    e = observer.evidence
    mph = float(CS.out.vEgoRaw) / 0.44704
    if not self.enabled or not 60 < mph < 70 or not getattr(CS.out, "canValid", False) or not observer.basic_evidence():
      self.onset_history.clear()
      return fraction
    now, st, ct, *_ = e["context"]
    motion_stamp = e["motion"][1]
    generation = e["generation"]
    if generation != self.onset_generation:
      self.onset_history.clear()
      self.onset_generation = generation
    # Independent wheel history is only advanced for a validated atomic target.
    wheel = self._independent_wheel_history
    k = float(a.curvature)
    if not wheel or wheel[-1][0] != st or not math.isfinite(k):
      self.onset_history.clear()
      return fraction
    current = (now, st, ct, motion_stamp, k)
    if self.onset_history:
      previous = self.onset_history[-1]
      if any(current[j] <= previous[j] for j in range(4)) or not 40_000_000 <= now - previous[0] <= 60_000_000:
        self.onset_history.clear()
        return fraction
      if k * previous[4] <= 0:
        self.onset_history.clear()
    self.onset_history.append(current)
    if fraction <= 0 or self.wheel_reason != "confirmed_target_excess" or self.release_pending or len(self.onset_history) < 5:
      return fraction
    first, last = self.onset_history[0], self.onset_history[-1]
    if any(not 160_000_000 <= last[j] - first[j] <= 300_000_000 for j in range(4)):
      return fraction
    # Only a withdrawing request receives early onset, not a rising entry.
    demand = [abs(row[4]) for row in self.onset_history]
    if demand[-1] > 0.98 * max(demand) or demand[-1] > demand[-2]:
      return fraction
    body = self.body_history
    if len(body) < 5 or body[-1][0] != motion_stamp or not 160_000_000 <= body[-1][0] - body[0][0] <= 300_000_000:
      return fraction
    # Current wheel/body agreement and predictive release already passed above.
    # Credit the existing confirmation, but never reset or accelerate the ramp.
    self._persistent = max(self._persistent, self.PERSISTENCE_FRAMES - 1)
    self.onset_receipt = dict(
      now_ns=now,
      frames=5,
      state_span_s=(st - first[1]) / 1e9,
      control_span_s=(ct - first[2]) / 1e9,
      motion_span_s=(motion_stamp - first[3]) / 1e9,
      body_frames=len(body),
      body_span_s=(body[-1][0] - body[0][0]) / 1e9,
    )
    return fraction
