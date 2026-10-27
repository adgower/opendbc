"""Navigator moving-speed longitudinal coordination contracts."""

from dataclasses import dataclass, asdict
import json
import math
from opendbc.car.ford.longitudinal_precharge import PRECHARGE_PROTOCOL_VERSION


@dataclass(frozen=True)
class Torque:
  t: float
  value: float
  valid: bool = True


@dataclass(frozen=True)
class Feedback:
  t: float
  epoch: int
  sequence: int
  requested: float
  gas: float
  brake: bool
  limited: bool
  valid: bool = True
  fault: bool = False
  allocator_version: int = 0
  precharge: bool = False
  reference_brake: bool = False
  allocator_phase: str = 'inactive'
  precharge_age: float = 0.
  budget_remaining: float = 0.

  def encode(self):
    return json.dumps(asdict(self), allow_nan=False).encode()

  @classmethod
  def decode(cls, b):
    return cls(**json.loads(b))


@dataclass(frozen=True)
class Release:
  allow: bool
  fault: bool = False
  reason: str = "baseline"


class BrakeReleaseProgress:
  """Bound extra fault-wait time using received torque, never inferred clearance."""

  def __init__(self):
    self.episode = None
    self.samples = []

  def expired(self, now, started, torque):
    if started != self.episode:
      self.episode = started
      self.samples = []
    if torque.t >= started and (not self.samples or torque.t != self.samples[-1][0]):
      self.samples.append((torque.t, torque.value))

    elapsed = now - started
    if elapsed + 1e-9 < 1.0:
      return False
    if elapsed + 1e-9 >= 2.0:
      return True

    initial = [value for timestamp, value in self.samples if 0 <= timestamp - started <= 0.25]
    recent = [(timestamp, value) for timestamp, value in self.samples if now - 0.25 <= timestamp <= now]
    if not initial or len(recent) < 3 or recent[-1][0] - recent[0][0] < 0.15 - 1e-9:
      return True
    if now - torque.t > 0.05 or any(b[0] - a[0] > 0.050000001 for a, b in zip(recent, recent[1:], strict=False)):
      return True

    peak = max(initial)
    # Require substantial overall release and continuing recent progress.
    if peak < 40.0 or torque.value > 0.6 * peak:
      return True
    if recent[0][1] - recent[-1][1] < max(8.0, 0.04 * peak):
      return True

    floor = recent[0][1]
    for _, value in recent:
      if value - floor > max(8.0, 0.02 * peak):
        return True
      floor = min(floor, value)
    return False


class ReleaseBarrier:
  def __init__(self, enabled=False):
    self.enabled = enabled
    self.reset()

  def reset(self):
    self.needs = True  # moving entry must establish physical brake clearance
    self.started = None
    self.clear = []
    self.last_sample = None
    self.last = None
    self.fault = None
    self.release_progress = BrakeReleaseProgress()

  def fail(self, reason):
    self.fault = reason
    return Release(False, True, reason)

  def update(self, now, active, brake, speed, phase, torque, *, owner_active=False):
    if not self.enabled:
      return Release(not brake, False, "disabled")
    if not active:
      if owner_active:
        # A pedal override is not a new owner/engagement. Keep the physical
        # release obligation, but require fresh qualification after returning.
        # Manual control must not consume the automatic-release fault deadline.
        self.needs = True
        self.started = None
        self.clear = []
        self.last_sample = None
        self.last = None
        return Release(False, bool(self.fault), self.fault or "override_pending_release")
      self.reset()
      return Release(not brake, False, "inactive")
    if self.fault:
      return Release(False, True, self.fault)
    if not math.isfinite(now) or not math.isfinite(speed):
      return self.fail("invalid_clock_or_speed")
    dt = None if self.last is None else now - self.last
    self.last = now
    if dt is not None and not 0 < dt <= 0.100000001:
      return self.fail("clock_discontinuity")
    if phase != "pid" or speed < 3.0:
      self.needs = True
      self.started = None
      self.clear = []
      self.last_sample = None
      return Release(not brake, False, "installed_stop_creep")
    if brake:
      self.needs = True
      self.started = None
      self.clear = []
      self.last_sample = None
      return Release(False, False, "actual_brake")
    if not self.needs:
      return Release(True, False, "no_release_pending")
    if self.started is None:
      self.started = now
    fresh = (
      torque is not None
      and torque.valid
      and all(math.isfinite(x) for x in [torque.t, torque.value])
      and 0 <= now - torque.t <= 0.1
      and 0 <= torque.value <= 32756.0
    )
    if not fresh:
      return self.fail("invalid_release_feedback")
    if self.last_sample is not None and torque.t < self.last_sample:
      return self.fail("reversed_release_feedback")
    if torque.t >= self.started and torque.t != self.last_sample:
      if self.last_sample is not None and not 0 < torque.t - self.last_sample <= 0.050000001:
        self.clear = []
      self.last_sample = torque.t
      if torque.value <= 4.0:
        self.clear.append(torque.t)
      else:
        self.clear = []
    if len(self.clear) >= 3 and self.clear[-1] - self.clear[0] + 1e-9 >= 0.04:
      self.needs = False
      self.started = None
      self.clear = []
      return Release(True, False, "observed_clear")
    # A progressing measured release may outlast the nominal one-second deadline.
    # The bounded wait never grants propulsion without the original measured clear.
    if self.release_progress.expired(now, self.started, torque):
      return self.fail("release_timeout_disengage")
    return Release(False, False, "wait_observed_clear")


def _freeze_integral_base(now, epoch, sequence, error, target, wanted, speed, phase, feedback, enabled=False):
  if not enabled or feedback is None:
    return False
  f = feedback
  if not all(math.isfinite(x) for x in [now, error, target, wanted, speed, f.t, f.requested, f.gas]):
    return False
  fresh = 0 <= now - f.t <= 0.100000001 and f.epoch == epoch and 0 <= f.sequence <= sequence
  return bool(
    fresh
    and f.valid
    and not f.fault
    and not f.brake
    and f.limited
    and phase == "pid"
    and speed >= 3.0
    and error > 0
    and target >= 0
    and wanted > 0.02
    and f.requested > 0.02
    and wanted > max(f.gas, 0.0) + 0.011
  )


def _allocator_feedback_valid(f, version):
  if getattr(f, 'allocator_version', 0) != version or getattr(f, 'allocator_phase', '') not in (
    'inactive', 'brake', 'precharge', 'release', 'engineRamp', 'drive', 'fault',
  ):
    return False
  try:
    finite_budget = all(math.isfinite(v) for v in [f.precharge_age, f.budget_remaining])
  except (TypeError, ValueError):
    return False
  if not finite_budget:
    return False
  if f.precharge and not f.brake:
    return bool(f.allocator_phase == 'precharge' and f.reference_brake and f.limited and f.gas == -5.
                and 0 <= f.precharge_age < .300000001 and 0 < f.budget_remaining <= .01)
  return f.allocator_phase != 'precharge'


def coast_marker(f):
  return bool(getattr(f, 'allocator_phase', '') == 'release' and f.reference_brake and not f.brake and not f.precharge)


def allocator_feedback_valid(f):
  if getattr(f, 'allocator_version', 0) != PRECHARGE_PROTOCOL_VERSION:
    return False
  if coast_marker(f):
    return bool(f.limited and f.gas == -5. and all(math.isfinite(v) for v in [f.precharge_age, f.budget_remaining])
                and 0 <= f.precharge_age < .300000001 and 0 < f.budget_remaining <= .01)
  return _allocator_feedback_valid(f, PRECHARGE_PROTOCOL_VERSION)


def freeze_integral(now, epoch, sequence, error, target, wanted, speed, phase, feedback, enabled=False):
  f = feedback
  valid = (enabled and f is not None
           and all(math.isfinite(v) for v in [now, error, target, wanted, speed, f.t, f.requested, f.gas])
           and 0 <= now - f.t <= .100000001 and f.epoch == epoch and 0 < f.sequence <= sequence
           and epoch > 0 and f.valid and not f.fault and phase == 'pid' and speed >= 3)
  # Bounded precharge is signed brake control. Keep both signs of PID error,
  # as with nominal actual braking; the selected brake bit stays truthful.
  if valid and ((allocator_feedback_valid(f) and ((f.precharge and not f.brake) or coast_marker(f)))
                or (getattr(f, 'allocator_version', 0) == 1 and _allocator_feedback_valid(f, 1) and f.precharge and not f.brake)):
    return False
  return _freeze_integral_base(now, epoch, sequence, error, target, wanted, speed, phase, feedback, enabled=enabled)


# OEM-only fit: 15 windows / four moving routes, equal route weighting.
# See navigator-grade-resolution-2026-10-23/calibration.json in the research record.
# This is a relative decision-reference correction, never a camera calibration reset.
GRADE_OFFSET = 0.14991323424078207


@dataclass(frozen=True)
class GradeResult:
  gravity: float
  offset: float
  reason: str


class GradeLease:
  def __init__(self, enabled=False):
    self.enabled = enabled
    self.amount = 0.0
    self.last = None
    self.last_pose = None

  def update(self, now, pitch, pose_t, pose_valid, active, speed, phase, reference_valid=True):
    dt = None if self.last is None else now - self.last
    self.last = now if math.isfinite(now) else None
    finite = all(math.isfinite(v) for v in [now, pitch, pose_t, speed])
    valid = (finite and pose_valid and 0 <= now - pose_t <= 0.1
             and (dt is None or 0 < dt <= 0.100000001)
             and (self.last_pose is None or pose_t >= self.last_pose))
    if not self.enabled:
      return GradeResult(pitch, 0.0, "disabled")
    if not active:
      self.amount = 0.0
      self.last_pose = None
      return GradeResult(pitch, 0.0, "inactive")
    if not valid:
      return GradeResult(pitch, 0.0, "invalid_pose_disengage")
    self.last_pose = pose_t
    if phase == "pid" and speed >= 3.0:
      target = GRADE_OFFSET if reference_valid else 0.0
      step = 0.005 * (dt or 0.0)
      self.amount += max(-step, min(step, target - self.amount))
      used = self.amount * max(0.0, min(1.0, (speed - 3.0) / 2.0))
      return GradeResult(pitch + used, used, "mapped" if reference_valid else "reference_slew_out")
    return GradeResult(pitch, 0.0, "installed_stop_creep")


def feedback_from_wire(f):
  phase = getattr(f, 'allocatorPhase', 'inactive')
  phase_raw = getattr(phase, 'raw', None)
  phases = ('inactive', 'brake', 'precharge', 'release', 'engineRamp', 'drive', 'fault')
  phase_name = (phases[phase_raw] if 0 <= phase_raw < len(phases) else 'unknown') if phase_raw is not None else str(phase)
  return Feedback(
    f.producerMonoTime / 1e9, int(f.epoch), int(f.sequence), float(f.requested), float(f.gas), bool(f.brake), bool(f.limited), bool(f.valid), bool(f.fault),
    int(getattr(f, 'allocatorVersion', 0)), bool(getattr(f, 'precharge', False)), bool(getattr(f, 'referenceBrake', False)),
    phase_name, float(getattr(f, 'prechargeAge', 0.)), float(getattr(f, 'budgetRemaining', 0.)),
  )


def coordination_available(CP):
  from opendbc.car.ford.values import CAR

  return CP.carFingerprint == CAR.FORD_EXPEDITION_MK4 and CP.openpilotLongitudinalControl
