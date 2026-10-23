"""Navigator moving-speed longitudinal coordination contracts."""

from dataclasses import dataclass, asdict
import json
import math


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
    # One real Alpha release retains60–90Nm for~0.7s. Deadline only requests
    # disengagement; elapsed time never grants propulsion without measured clear.
    if now - self.started + 1e-9 >= 1.0:
      return self.fail("release_timeout_disengage")
    return Release(False, False, "wait_observed_clear")


def freeze_integral(now, epoch, sequence, error, target, wanted, speed, phase, feedback, enabled=False):
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
  return Feedback(
    f.producerMonoTime / 1e9, int(f.epoch), int(f.sequence), float(f.requested), float(f.gas), bool(f.brake), bool(f.limited), bool(f.valid), bool(f.fault)
  )


def coordination_available(CP):
  from opendbc.car.ford.values import CAR

  return CP.carFingerprint == CAR.FORD_EXPEDITION_MK4 and CP.openpilotLongitudinalControl
