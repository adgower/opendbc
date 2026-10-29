"""Finite offline A/B lease. Transforms supplied bytes; never sends vehicle commands."""

from dataclasses import dataclass
import math
import re

MIN_SPEED = 3.0
MAX_SPEED = 20 * 0.44704
MAX_ACTIVE = 5.0
MAX_TOTAL = 30.0
QUALIFY = 0.1
TERMINAL = ("disabled", "finished", "cancelled")


@dataclass(frozen=True)
class Request:
  trial_id: str
  condition: str
  source_id: str
  duration_s: float = 3.0
  wait_s: float = 30.0


@dataclass(frozen=True)
class Context:
  source_id: str
  alpha: bool
  enabled: bool
  long_active: bool
  phase: str
  speed_ms: float
  can_valid: bool
  control_valid: bool
  source_valid: bool
  pose_valid: bool
  reference_valid: bool
  request_valid: bool
  epoch: int
  sequence: int
  producer_ns: int
  state_ns: int
  pose_ns: int
  target_ns: int
  health_valid: bool
  controls_allowed: bool
  selfdrive_active: bool
  mismatch: bool
  fault: bool
  in_drive: bool
  steering_override: bool
  gas_pedal: bool
  brake_pedal: bool
  observer_clear: bool
  torque_valid: bool
  torque_ns: int
  torque: float


def finite(value):
  try:
    return math.isfinite(value)
  except (ValueError, TypeError):
    return False


class Runner:
  def __init__(self, *, enabled=False):
    self.enabled = enabled is True
    self.state = "disabled" if not self.enabled else "idle"
    self.request = None
    self.seen = set()
    self.events = []
    self.last_decision = {}
    self.last = None
    self.epoch = None
    self.sequence = None
    self.producer_ns = None
    self.since = None
    self.wait_deadline = None
    self.active_deadline = None

  def event(self, event, now, reason):
    self.events.append(
      dict(
        event=event,
        t=now if finite(now) else None,
        trial_id=self.request.trial_id if self.request else None,
        condition=self.request.condition if self.request else None,
        reason=reason,
      )
    )

  def arm(self, request, now):
    if not self.enabled:
      raise ValueError("Explicit local enable required")
    if self.state in ("waiting", "active"):
      raise ValueError("Trial busy; cannot replace active request")
    if not isinstance(request, Request):
      raise ValueError("Typed request required")
    if not isinstance(request.trial_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", request.trial_id):
      raise ValueError("Invalid trial ID")
    if request.trial_id in self.seen:
      raise ValueError("Trial ID already consumed")
    if request.condition not in ("A", "B"):
      raise ValueError("Condition must be A or B")
    if not isinstance(request.source_id, str) or not request.source_id or len(request.source_id) > 256:
      raise ValueError("Source ID required")
    if not finite(now) or now < 0:
      raise ValueError("Invalid arm clock")
    for value, limit in [(request.duration_s, MAX_ACTIVE), (request.wait_s, MAX_TOTAL)]:
      if isinstance(value, bool) or not finite(value) or not 0 < value <= limit:
        raise ValueError("Invalid bounded trial duration")
    self.request = request
    self.seen.add(request.trial_id)
    self.state = "waiting"
    self.last = None
    self.epoch = None
    self.sequence = None
    self.producer_ns = None
    self.since = None
    self.wait_deadline = now + request.wait_s
    self.active_deadline = None
    self.armed_at = now
    self.event("arm", now, "explicit_request")

  def cancel(self, reason, now):
    if self.state in ("waiting", "active"):
      self.state = "cancelled"
      self.since = None
      self.event("cancel", now, reason)

  def finish(self, reason, now):
    self.state = "finished"
    self.since = None
    self.event("finish", now, reason)

  @staticmethod
  def readiness(c, now):
    for key in [
      "enabled",
      "long_active",
      "can_valid",
      "control_valid",
      "source_valid",
      "pose_valid",
      "reference_valid",
      "request_valid",
      "health_valid",
      "controls_allowed",
      "selfdrive_active",
      "in_drive",
      "observer_clear",
      "torque_valid",
    ]:
      if getattr(c, key) is not True:
        return key
    if c.phase != "pid":
      return "phase"
    if c.gas_pedal or c.brake_pedal or c.steering_override:
      return "driver_override"
    if not finite(c.speed_ms) or not MIN_SPEED <= c.speed_ms <= MAX_SPEED:
      return "speed"
    if not finite(c.torque) or not 0 <= c.torque <= 4:
      return "torque"
    if not isinstance(c.epoch, int) or c.epoch <= 0 or not isinstance(c.sequence, int) or c.sequence <= 0:
      return "identity"
    now_ns = round(now * 1e9)
    for key in ["producer_ns", "state_ns", "pose_ns", "target_ns", "torque_ns"]:
      n = getattr(c, key)
      if not isinstance(n, int) or not 0 < n <= now_ns or now_ns - n > 100_000_000:
        return key
    if c.target_ns > c.producer_ns:
      return "target_after_producer"
    return None

  @staticmethod
  def packet_ready(msg):
    if not isinstance(msg, (list, tuple)) or len(msg) != 3 or msg[0] != 390 or msg[2] != 0:
      return "packet_route"
    if not isinstance(msg[1], (bytes, bytearray)) or len(msg[1]) != 8:
      return "packet_size"
    d = msg[1]
    gas = ((d[6] & 3) << 8) | d[7]
    pred = ((d[2] & 3) << 8) | d[3]
    if pred != 0:
      return "noninactive_baseline_prediction"
    if gas == 0:
      return "engine_inactive"
    if not 450 <= gas <= 700:
      return "engine_outside_bounds"
    if not d[6] & 4 or d[6] & 0xC0:
      return "packet_inactive_or_braking"
    # Stop status lies in byte4 bit2; do not transform stopping commands.
    if d[4] & 4:
      return "packet_stopping"
    return None

  def process(self, now, c, msg):
    self.last_decision = dict(state=self.state, reason="inert", modified=False)
    if self.state in TERMINAL or self.state == "idle":
      return msg
    reason = None
    if not finite(now) or now < self.armed_at:
      reason = "invalid_clock"
    elif self.last is not None and not 0 < now - self.last <= (0.040000001 if self.state == "active" else 0.100000001):
      reason = "callback_clock_discontinuity"
    elif not isinstance(c, Context):
      reason = "invalid_context"
    elif c.source_id != self.request.source_id:
      reason = "source_changed"
    elif not c.alpha:
      reason = "owner_changed"
    elif c.fault or c.mismatch:
      reason = "fault_or_mismatch"
    elif finite(c.speed_ms) and c.speed_ms > MAX_SPEED:
      reason = "overspeed"
    elif self.epoch is not None and (not isinstance(c.epoch, int) or c.epoch != self.epoch):
      reason = "epoch_changed"
    if reason:
      self.cancel(reason, now)
      self.last_decision = dict(state=self.state, reason=reason, modified=False)
      return msg
    self.last = now
    if self.state == "waiting" and now >= self.wait_deadline - 1e-9:
      self.finish("wait_expired", now)
      self.last_decision = dict(state=self.state, reason="wait_expired", modified=False)
      return msg
    reason = self.readiness(c, now) or self.packet_ready(msg)
    if reason:
      if self.state == "active" or reason in ("packet_route", "packet_size", "noninactive_baseline_prediction", "engine_outside_bounds"):
        self.cancel(reason, now)
      else:
        self.since = None
      self.last_decision = dict(state=self.state, reason=reason, modified=False)
      return msg
    if self.sequence is not None:
      if (
        c.sequence < self.sequence
        or (c.sequence == self.sequence and c.producer_ns != self.producer_ns)
        or (c.sequence > self.sequence and c.producer_ns <= self.producer_ns)
      ):
        self.cancel("producer_identity_discontinuity", now)
        self.last_decision = dict(state=self.state, reason="producer_identity_discontinuity", modified=False)
        return msg
    self.epoch = c.epoch
    self.sequence = c.sequence
    self.producer_ns = c.producer_ns
    if self.state == "active" and (now >= self.active_deadline - 1e-9 or now >= self.wait_deadline - 1e-9):
      self.finish("lease_expired", now)
      self.last_decision = dict(state=self.state, reason="lease_expired", modified=False)
      return msg
    if self.state == "waiting":
      if self.since is None:
        self.since = now
      if now - self.since + 1e-9 < QUALIFY:
        self.last_decision = dict(state=self.state, reason="qualifying", modified=False)
        return msg
      self.state = "active"
      self.active_deadline = min(now + self.request.duration_s, self.wait_deadline)
      self.event("start", now, "healthy_engine_context")
    if self.request.condition == "A":
      self.last_decision = dict(state=self.state, reason="A_original", modified=False)
      return msg
    d = bytearray(msg[1])
    gas = ((d[6] & 3) << 8) | d[7]
    d[2] = (d[2] & 0xFC) | (gas >> 8)
    d[3] = gas & 255
    out = list(msg)
    out[1] = bytes(d)
    self.last_decision = dict(state=self.state, reason="B_mirror", modified=out[1] != bytes(msg[1]))
    return tuple(out) if isinstance(msg, tuple) else out
