"""Bounded precharge entry; nominal brake admission and release remain authoritative."""

from dataclasses import dataclass
import math

PRECHARGE_PROTOCOL_VERSION = 3


def allocator_request_valid(request, now_ns):
  return bool(
    request.allocatorVersion == PRECHARGE_PROTOCOL_VERSION
    and request.targetValid
    and math.isfinite(request.targetAcceleration)
    and 0 < request.targetMonoTime <= request.producerMonoTime <= now_ns
    and now_ns - request.targetMonoTime <= 100_000_000
  )


@dataclass(frozen=True)
class PrechargeDecision:
  brake: bool
  precharge: bool
  phase: str


class BoundedPrechargeAllocator:
  budget = 0.01  # Signed demand-area guard, not measured physical speed error.
  deadline = 0.3  # Decision deadline; commands persist until the next consumer delivery.

  def __init__(self):
    self.brake_proposal = False
    self.last = None
    self.start = None
    self.area = 0.0
    self.locked = False
    self.recovery = None
    self.committed = False

  def reference(self, nominal, phase):
    if nominal:
      self.committed = True
    return PrechargeDecision(bool(nominal), bool(nominal), phase)

  def update(self, now, owner, active, eligible, nominal, raw, net, target, *, measured_clear=False):
    dt = None if self.last is None else now - self.last
    self.last = now if math.isfinite(now) else None
    if not owner:
      self.__init__()
      self.last = now if math.isfinite(now) else None
      return self.reference(nominal, "inactive")
    if not nominal:
      self.committed = False
    if not active:
      self.brake_proposal = False
    elif math.isfinite(net):
      if net < -0.14:
        self.brake_proposal = True
      elif net > -0.06:
        self.brake_proposal = False
    if not active:
      self.recovery = None
      return self.reference(nominal, "override")
    good = eligible and all(math.isfinite(v) for v in [now, raw, net, target]) and dt is not None and 0 < dt <= 0.040000001
    if not good:
      self.recovery = None
      if self.start is not None:
        self.locked = True
      return self.reference(nominal, "invalid_or_installed")
    # Rearm only after sampled weak-demand debt is repaid and fresh clearance
    # remains qualified; positive demand alone must not forgive debt.
    recovery = measured_clear and not nominal and net > 0.05 and raw > 0.05 and target >= 0 and self.area == 0.0
    self.recovery = now if recovery and self.recovery is None else self.recovery if recovery else None
    if self.recovery is not None and now - self.recovery >= 0.1 - 1e-9:
      self.start = None
      self.area = 0.0
      self.locked = False
    if (net < 0 or (nominal and not self.committed)) and self.start is None:
      self.start = now
    strong = target <= -0.5 or raw < -0.5 or net < -0.14
    proposed_precharge = nominal and not self.committed and not self.locked and not self.brake_proposal and not strong
    deferred_weak_entry = not nominal and net < 0
    if self.start is not None:
      # Sampled right-endpoint demand guard, not a measured physical deficit.
      # Full-brake requests retain existing debt without adding new deferral.
      if net >= 0 or proposed_precharge or deferred_weak_entry:
        self.area = max(0.0, self.area - net * dt)
      if self.area >= self.budget - 1e-9 or now - self.start >= self.deadline - 1e-9:
        self.locked = True
    if strong:
      if self.start is not None:
        self.locked = True
      return self.reference(nominal, "strong_reference")
    if self.locked:
      return self.reference(nominal, "budget_reference")
    if not nominal:
      return PrechargeDecision(False, False, "reference_coast")
    if self.committed:
      return self.reference(nominal, "committed_reference")
    if self.brake_proposal:
      return self.reference(True, "brake")
    # A required weak brake is represented by precharge within the remaining
    # budget, never discarded because another allocator threshold is higher.
    return PrechargeDecision(False, True, "precharge")


class SignedCoastAllocator(BoundedPrechargeAllocator):
  def update(self, now, owner, active, eligible, nominal, raw, net, target, *, measured_clear=False):
    d = super().update(now, owner, active, eligible, nominal, raw, net, target, measured_clear=measured_clear)
    if (
      d.precharge
      and not d.brake
      and getattr(self, "coast_allowed", False)
      and eligible
      and (measured_clear or getattr(self, "continuation_clear", False))
      and raw > 0.02
      and target >= 0
    ):
      return PrechargeDecision(False, False, "bounded_true_coast")
    return d
