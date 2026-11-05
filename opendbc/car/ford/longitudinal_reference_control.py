"""Bounded weak-coast selection with an independent nominal policy witness.

The measured clearance observer and main controller remain the only actuation
path. This module owns no packer, output messages, or inferred ECU clearance.
"""

import math
import struct
from collections import Counter
from dataclasses import asdict, replace
from opendbc.car.ford.longitudinal_release import ReleaseGate, ReleaseResult
from opendbc.car.ford.longitudinal_precharge import SignedCoastAllocator, PrechargeDecision
from opendbc.car.ford.longitudinal_handoff import Handoff, Result
from opendbc.car.ford.longitudinal_nominal_reference import NominalReference
from opendbc.car.ford.longitudinal_reference_feedback import MODEL_ID, REQUEST_VERSION, SOURCE, ReferenceReceipt


class BoundedResponseEntry(ReleaseGate):
  budget = 0.01
  deadline = 0.3
  response_budget = 0.01

  def __init__(self):
    super().__init__()
    self.scope = False
    self.actual = self.target = 0.0
    self.clear = False
    self.owner = True
    self.arm = self.permission = None
    self.area = self.allocator_area = 0.0
    self.response_area = 0.0
    self.spent = self.deferred = False
    self.recovery = self.sample = None

  def update(self, now, active, baseline_brake, previous_brake, raw, net, phase, pitch_valid, *, _common_valid=None):
    r = super().update(now, active, baseline_brake, previous_brake, raw, net, phase, pitch_valid)
    dt = None if self.sample is None else now - self.sample
    self.sample = now if math.isfinite(now) else None
    self.deferred = False
    if not self.owner:
      if self.arm is not None or self.permission is not None or max(self.area, self.allocator_area) > 0:
        self.spent = True
      self.arm = self.permission = self.recovery = None
      return r
    good = (
      _common_valid if _common_valid is not None else (
      self.scope
      and active
      and pitch_valid
      and phase == "pid"
      and all(math.isfinite(v) for v in [now, raw, net, self.actual, self.target, self.allocator_area])
      and dt is not None
      and 0 < dt <= 0.040000001)
    )
    if not good:
      if self.arm is not None or self.permission is not None:
        self.spent = True
      self.arm = self.permission = self.recovery = None
      return r
    if self.arm is not None:
      self.response_area += max(0.0, self.actual - self.target) * dt
      if self.response_area >= self.response_budget - 1e-9:
        self.spent = True
        self.arm = self.permission = None
        return r
    if net > 0:
      self.area = max(0.0, self.area - net * dt)
    recovered = self.clear and max(self.area, self.allocator_area) == 0.0 and net > 0.05 and raw > 0.05 and self.target >= 0
    self.recovery = now if recovered and self.recovery is None else self.recovery if recovered else None
    if self.recovery is not None and now - self.recovery >= 0.1 - 1e-9:
      self.spent = False
      self.arm = self.permission = None
      self.response_area = 0.0
    strong = self.target <= -0.5 or raw < -0.5 or net < -0.14
    if strong or (self.arm is not None and now - self.arm >= self.deadline - 1e-9):
      self.spent = True
      self.arm = self.permission = None
      return r
    if r.reason == "qualified_positive_release":
      if self.arm is not None:
        self.spent = True
        self.arm = self.permission = None
      elif not self.spent and self.permission is None:
        self.permission = now
      return r
    if self.permission is not None and self.arm is None and now - self.permission > self.deadline + 1e-9:
      self.permission = None
      self.spent = True
    intent = raw > 0.02 and abs(self.target) <= 0.2
    entry_error_ok = self.actual <= self.target + 0.02
    if self.arm is None and not self.spent and self.permission is not None and r.brake and -0.14 <= net < 0 and intent and entry_error_ok:
      self.arm = now
      self.response_area = max(0.0, self.actual - self.target) * dt
    if self.arm is None or self.spent or not r.brake:
      return r
    if not intent:
      self.spent = True
      self.arm = self.permission = None
      return r
    proposed = max(self.area, self.allocator_area) + max(0.0, -net) * dt
    if proposed >= self.budget - 1e-9:
      self.spent = True
      self.arm = self.permission = None
      return r
    self.area = proposed
    self.deferred = True
    return ReleaseResult(False, "bounded_release_continuation")


class QualifiedRearmEntry(BoundedResponseEntry):
  def __init__(self):
    super().__init__()
    self.lease_not_before = -math.inf

  def update(self, now, active, baseline_brake, previous_brake, raw, net, phase, pitch_valid, *, _common_valid=None):
    dt = None if self.sample is None else now - self.sample
    valid = (
      _common_valid if _common_valid is not None else (
      self.scope
      and self.owner
      and active
      and pitch_valid
      and phase == "pid"
      and all(math.isfinite(v) for v in [now, raw, net, self.actual, self.target, self.allocator_area])
      and dt is not None
      and 0 < dt <= 0.040000001)
    )
    old_arm, old_permission = self.arm, self.permission
    result = super().update(now, active, baseline_brake, previous_brake, raw, net, phase, pitch_valid, _common_valid=valid)
    if old_arm is None and self.arm is not None:
      self.lease_not_before = max(self.lease_not_before, self.arm + self.deadline)
    if self.arm is None and self.permission is not None and old_permission is None:
      if now < self.lease_not_before - 1e-9 or max(self.area, self.allocator_area) > 0:
        self.permission = None
        self.spent = True
      else:
        self.lease_not_before = now + self.deadline
    if (
      result.reason == "qualified_positive_release"
      and valid
      and old_arm is None
      and self.arm is None
      and self.permission is None
      and now >= self.lease_not_before - 1e-9
      and self.area == 0.0
      and self.allocator_area == 0.0
    ):
      self.permission = now
      self.spent = False
      self.response_area = 0.0
      self.lease_not_before = now + self.deadline
    return result


class PendingRearmEntry(QualifiedRearmEntry):
  def __init__(self):
    super().__init__()
    self.pending_qualified = None

  def update(self, now, active, baseline_brake, previous_brake, raw, net, phase, pitch_valid, *, _common_valid=None):
    dt = None if self.sample is None else now - self.sample
    _common_valid = (_common_valid if _common_valid is not None else (
      self.scope
      and self.owner
      and active
      and pitch_valid
      and phase == "pid"
      and all(math.isfinite(v) for v in [now, raw, net, self.actual, self.target, self.allocator_area])
      and dt is not None
      and 0 < dt <= 0.040000001))
    valid = (
      _common_valid
      and 0.02 < raw <= 0.3
      and abs(self.target) <= 0.2
      and self.target > -0.5
      and net >= -0.14
    )
    old_arm = self.arm
    old_deadline = self.lease_not_before
    result = super().update(now, active, baseline_brake, previous_brake, raw, net, phase, pitch_valid, _common_valid=_common_valid)
    if not valid or result.brake or old_arm is not None or self.arm is not None:
      self.pending_qualified = None
      return result
    if self.pending_qualified is not None and now >= self.pending_qualified + self.deadline - 1e-9:
      self.pending_qualified = None
    # Only the genuine original ReleaseGate qualification can seed this proof.
    # A spent episode must already have passed its immutable deadline.
    if result.reason == "qualified_positive_release" and self.permission is None and self.pending_qualified is None and now >= old_deadline - 1e-9:
      self.pending_qualified = now
    if self.permission is not None:
      self.pending_qualified = None
      return result
    if self.pending_qualified is not None and now >= self.lease_not_before - 1e-9 and self.area == 0.0 and self.allocator_area == 0.0:
      proof = self.pending_qualified
      self.permission = proof
      self.spent = False
      self.response_area = 0.0
      self.lease_not_before = proof + self.deadline
      self.pending_qualified = None
    return result


class NominalOwnerGate(PendingRearmEntry):
  def __init__(self):
    super().__init__()
    self.nominal_gate = ReleaseGate()
    self.selected = False
    self.target_trend_ok = False

  def update(self, *args, **kw):
    nominal = self.nominal_gate.update(*args, **kw)
    scope = self.scope
    previous_brake = args[3] if len(args) > 3 else kw["previous_brake"]
    raw = args[4] if len(args) > 4 else kw["raw"]
    if not math.isfinite(raw) or not 0.02 < raw <= 0.3:
      self.scope = False
    if self.arm is None and nominal.brake and (previous_brake or not self.target_trend_ok):
      self.scope = False
    super().update(*args, **kw)
    self.scope = scope
    self.deferred = bool(self.deferred and nominal.brake)
    self.nominal_result = nominal
    return nominal


class OwnerAllocator(SignedCoastAllocator):
  def __init__(self, gate=None):
    saved = gate if gate is not None else getattr(self, "gate", None)
    assert saved is not None
    super().__init__()
    self.gate = saved

  def update(self, now, owner, active, eligible, nominal, raw, net, target, *, measured_clear=False):
    prior = self.last
    self.gate.selected = False
    r = super().update(now, owner, active, eligible, nominal, raw, net, target, measured_clear=measured_clear)
    valid = bool(
      prior is not None
      and 0 < now - prior <= 0.040000001
      and eligible
      and owner
      and active
      and nominal
      and getattr(self, "coast_allowed", False)
      and self.gate.deferred
      and all(math.isfinite(v) for v in [now, raw, net, target, self.area, self.gate.area])
      and target > -0.5
      and 0.02 < raw <= 0.3
      and net >= -0.14
    )
    if not valid:
      return r
    area = max(self.area, self.gate.area)
    if area >= self.budget - 1e-9:
      return r
    self.area = area
    self.gate.selected = True
    return PrechargeDecision(False, False, "bounded_release_continuation")


class OwnerHandoff(Handoff):
  def __init__(self, gate):
    super().__init__()
    self.gate = gate

  def update(self, *args, bounded_coast=False, **kw):
    return super().update(*args, bounded_coast=bool(bounded_coast or self.gate.selected), **kw)


class TargetTrend:
  def __init__(self):
    self.epoch = None
    self.latest = None
    self.previous = None
    self.conflicted = False

  def update(self, epoch, ns, value, now):
    if any(type(v) is not int or v <= 0 for v in [epoch, ns, now]) or not math.isfinite(value) or ns > now:
      return False
    if epoch != self.epoch:
      self.epoch = epoch
      self.latest = None
      self.previous = None
      self.conflicted = False
    if self.latest is None:
      self.latest = (ns, value)
    elif ns > self.latest[0]:
      self.previous = self.latest
      self.latest = (ns, value)
      self.conflicted = False
    elif ns == self.latest[0]:
      if value != self.latest[1]:
        self.conflicted = True
    else:
      return False
    if self.conflicted or self.previous is None:
      return False
    return bool(
      0 <= now - self.previous[0] <= 100_000_000
      and 0 <= now - self.latest[0] <= 100_000_000
      and self.previous[0] < self.latest[0]
      and self.latest[1] >= self.previous[1]
    )


class ExitHandoff(OwnerHandoff):
  horizon = 1.0

  def __init__(self, gate):
    super().__init__(gate)
    self.recovering = False
    self.recovery_started = None
    self.release_until = None
    self.epoch = None
    self.failures = Counter()
    self.completed = 0
    self.started = 0
    self.capped_s = 0.0
    self.max_recovery_s = 0.0

  def update(self, now, active, brake, target, pitch_valid, stopping, *, bounded_coast=False):
    prior = self.last_time
    actual = super().update(now, active, brake, target, pitch_valid, stopping, bounded_coast=bounded_coast)
    if not active:
      self.recovering = False
      self.recovery_started = self.release_until = self.epoch = None
      return actual
    if self.gate.selected and not self.recovering:
      self.recovering = True
      self.recovery_started = now
      self.epoch = self.gate.current_epoch
      self.release_until = self.gate.arm + self.gate.deadline
      self.started += 1
    if not self.recovering:
      return actual
    if self.gate.current_epoch != self.epoch:
      self.failures["epoch_changed"] += 1
      self.recovering = False
      return self.withdraw("qualify", "exit_reference_epoch_changed")
    if not math.isfinite(now) or now < self.recovery_started:
      self.failures["invalid_time"] += 1
      return self.withdraw("qualify", "exit_reference_invalid_time")
    elapsed = now - self.recovery_started
    self.max_recovery_s = max(self.max_recovery_s, elapsed)
    if elapsed >= self.horizon - 1e-9:
      self.failures["recovery_deadline"] += 1
      return self.withdraw("qualify", "exit_reference_deadline")
    s = getattr(self.gate, "nominal_ramp_sample", {})
    valid = bool(
      self.gate.scope
      and s.get("ns") == round(now * 1e9)
      and s.get("epoch") == self.epoch
      and s.get("sequence") == self.gate.current_sequence
      and s.get("raw") == self.gate.current_raw
    )
    ceiling = s.get("gas", float("nan"))
    valid = bool(
      valid
      and type(ceiling) in (int, float)
      and math.isfinite(ceiling)
      and (ceiling == -5.0 or -0.5 <= ceiling <= 2.0)
      and s.get("state") in ("inactive", "braking", "coast", "qualify", "resume", "drive")
    )
    if not valid:
      self.failures["invalid_reference"] += 1
      return self.withdraw("qualify", "exit_reference_invalid")
    result = actual
    if actual.gas != -5.0 and ceiling < actual.gas:
      if ceiling == -5.0:
        # The real update has already checked actual withdrawal and qualification.
        # Keep that actual qualifier; hold numerical ramp memory at its start.
        self.last_valid = None
        result = Result(-5.0, "qualify", "nominal_ramp_hold")
      else:
        self.last_valid = ceiling
        result = Result(ceiling, "resume", "nominal_ramp_ceiling")
      if prior is not None and 0 < now - prior <= 0.100000001:
        self.capped_s += now - prior
    if not self.gate.selected and now >= self.release_until and result.state in ("resume", "drive") and s["state"] == result.state and result.gas == ceiling:
      self.recovering = False
      self.completed += 1
    return result


class ReferenceControl:
  def __init__(self, CP, history_enabled=False):
    self.reference = NominalReference(CP, history_enabled=history_enabled)
    self.gate = NominalOwnerGate()
    self.allocator = OwnerAllocator(self.gate)
    self.handoff = ExitHandoff(self.gate)
    self.target_trend = TargetTrend()
    self.result = None
    self.episode = None
    self.callbacks = 0
    self.actual_accel = 0.

  def install(self, controller):
    controller.release_gate = self.gate
    controller.precharge_allocator = self.allocator
    controller.long_handoff = self.handoff

  def begin(self, controller, CC, CS, now_ns):
    for key in ("coordination_control_valid", "coordination_state_ns", "coordination_speed_ns", "coordination_state_valid"):
      setattr(self.reference, key, getattr(controller, key, False if key.endswith("valid") else 0))
    self.result = self.reference.update(CC, CS, now_ns)
    self.callbacks += 1
    g, q = self.gate, CC.navigatorLongitudinal
    h = self.result.handoff
    g.nominal_ramp_sample = (
      dict(ns=round(now_ns / 1e9 * 1e9), epoch=int(q.epoch), sequence=int(q.sequence), raw=float(CC.actuators.accel), gas=float(h.gas), state=h.state)
      if h is not None
      else {}
    )
    g.current_epoch = int(q.epoch)
    g.current_sequence = int(q.sequence)
    g.current_raw = float(CC.actuators.accel)
    g.target_trend_ok = self.target_trend.update(int(q.epoch), int(q.targetMonoTime), float(q.targetAcceleration), int(now_ns))
    g.actual = float(CS.out.aEgo)
    g.target = float(q.targetAcceleration)
    g.allocator_area = float(self.allocator.area)
    g.owner = bool(CC.enabled)
    tq = getattr(CS, "longitudinal_brake_torque", None)
    g.clear = bool(
      tq is not None
      and tq.valid
      and 0 <= now_ns / 1e9 - tq.t <= 0.1
      and 0 <= tq.value <= 4
      and not controller.release_barrier.needs
      and controller.last_observer is not None
      and controller.last_observer.allow
    )
    state_ns = getattr(controller, "coordination_state_ns", 0)
    g.scope = bool(
      q.allocatorVersion == REQUEST_VERSION
      and getattr(controller, "coordination_state_valid", False)
      and 0 < state_ns <= now_ns
      and now_ns - state_ns <= 100_000_000
      and CC.enabled
      and CC.longActive
      and CS.out.canValid
      and CS.out.vEgo >= 3
      and not CS.out.gasPressed
      and not CS.out.brakePressed
      and str(CC.actuators.longControlState) == "pid"
      and q.poseValid
      and q.referenceValid
      and q.targetValid
      and 0 < q.targetMonoTime <= q.producerMonoTime <= now_ns
      and now_ns - q.targetMonoTime <= 100_000_000
      and str(CC.hudControl.visualAlert) != "fcw"
    )

  def finish(self, controller, CC, output, now_ns):
    """Attach only policy provenance to the final actual feedback facts."""
    g, q, f = self.gate, CC.navigatorLongitudinal, output
    if g.selected:
      age = now_ns / 1e9 - g.arm
      remaining = g.budget - max(g.area, self.allocator.area)
      f = replace(f, precharge_age=age, budget_remaining=remaining, reason="bounded-release-selection-v2:unowned")
    actual_gas = struct.unpack('f', struct.pack('f', f.gas))[0]
    valid = bool(
      q.allocatorVersion == REQUEST_VERSION
      and CC.enabled
      and CC.longActive
      and g.scope
      and f.valid
      and not f.fault
      and not f.brake
      and not f.precharge
      and math.isfinite(actual_gas)
      and (actual_gas == -5 or -0.5 <= actual_gas <= 0.3)
      and 0.02 < f.requested <= 0.3
      and abs(float(q.targetAcceleration)) <= 0.2
      and self.actual_accel + controller.last_grade.gravity >= -0.14
    )
    if not valid:
      self.episode = None
      return f
    if g.deferred and g.arm is not None:
      start = int(round(g.arm * 1e9))
      until = start + int(round(g.deadline * 1e9))
      remaining = g.budget - max(g.area, g.allocator_area)
      if self.episode is None or self.episode["start"] != start:
        self.episode = dict(start=start, until=until, remaining=remaining)
    episode = self.episode
    if episode is None:
      return f
    if not episode["start"] <= now_ns < episode["until"]:
      self.episode = None
      return f
    episode["remaining"] = min(episode["remaining"], g.budget - max(g.area, g.allocator_area))
    if not 0 < episode["remaining"] <= 0.01:
      self.episode = None
      return f
    # The receiving process consumes the Float32 actuator field. Preserve raw
    # actual/nominal numerical control state; round only its envelope binding.
    actual_gas = struct.unpack("f", struct.pack("f", f.gas))[0]
    receipt = ReferenceReceipt(
      epoch=int(q.epoch),
      sequence=int(q.sequence),
      producer_ns=int(q.producerMonoTime),
      selected_ns=int(now_ns),
      start_ns=episode["start"],
      until_ns=episode["until"],
      budget_remaining=float(episode["remaining"]),
      episode=str(episode["start"]),
      source=SOURCE,
      target_ns=int(q.targetMonoTime),
      grade_offset=float(controller.last_grade.offset),
      actual_gas=actual_gas,
      actual_limited=bool(f.limited),
      actual_phase=str(f.allocator_phase),
      nominal=asdict(self.result.feedback),
      model_id=MODEL_ID,
    )
    return replace(f, reason=receipt.encode())
