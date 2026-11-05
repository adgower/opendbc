"""Bounded nominal witness transport; never actuator or clearance authority."""

from collections import Counter
from dataclasses import dataclass, asdict, fields
from numbers import Real
import json
import math

RELEASE_PREFIX = "bounded-release-owner-v1:"
PREFIX = RELEASE_PREFIX
MODEL_ID = "ford-nominal-reference-v1"
REQUEST_VERSION = 4
MAX_WITNESS = 8192
SOURCE = "navigator_longcontrol_v1"


def _finite(value):
  """JSON integers need not fit a floating-point value."""
  try:
    if type(value) in (int, float):
      return math.isfinite(value)
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
  except (OverflowError, TypeError, ValueError):
    return False


def _time_ns(value):
  try:
    return int(round(value * 1e9)) if _finite(value) else None
  except (OverflowError, TypeError, ValueError):
    return None


@dataclass(frozen=True)
class ReleaseReceipt:
  epoch: int
  sequence: int
  producer_ns: int
  selected_ns: int
  start_ns: int
  until_ns: int
  budget_remaining: float
  episode: str
  source: str = SOURCE
  target_ns: int = 0
  grade_offset: float = 0.0

  def encode(self):
    return _encode(RELEASE_PREFIX, self)

  @classmethod
  def parse(cls, text):
    try:
      if not isinstance(text, str) or not text.startswith(RELEASE_PREFIX) or len(text) > MAX_WITNESS:
        return None
      pairs = json.loads(text[len(RELEASE_PREFIX) :], object_pairs_hook=lambda p: p)
      if not isinstance(pairs, list) or len(dict(pairs)) != len(pairs):
        return None
      r = cls(**dict(pairs))
      ints = (r.epoch, r.sequence, r.producer_ns, r.selected_ns, r.start_ns, r.until_ns, r.target_ns)
      if any(type(v) is not int or v > 2**64 - 1 for v in ints):
        return None
      if any(v <= 0 for v in ints[:-1]) or r.target_ns < 0:
        return None
      if not isinstance(r.episode, str) or not 0 < len(r.episode) <= 128 or not isinstance(r.source, str):
        return None
      if type(r.budget_remaining) not in (int, float) or not _finite(r.budget_remaining):
        return None
      if type(r.grade_offset) not in (int, float) or not _finite(r.grade_offset) or not 0 <= r.grade_offset <= 0.149914:
        return None
      return r
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
      return None


def _encode(prefix, receipt):
  if len(repr(asdict(receipt))) > MAX_WITNESS:
    raise ValueError("reference witness too large")
  text = prefix + json.dumps(asdict(receipt), sort_keys=True, separators=(",", ":"), allow_nan=False)
  if len(text) > MAX_WITNESS:
    raise ValueError("reference witness too large")
  return text


class IssuedRegistry:
  """Bounded immutable issuance, with high-water tombstones after eviction."""

  def __init__(self):
    self.issued = {}
    self.episodes = {}
    self.last_reason = "reset"
    self.last_observation = None
    self.epoch = None
    self.highwater = 0
    self.overflow = False

  def issue(self, epoch, sequence, now, target, source_valid=True, source=SOURCE, target_ns=0):
    if type(epoch) is not int or type(sequence) is not int or type(target_ns) is not int or target_ns < 0:
      return
    if not (_finite(now) and _finite(target)) or epoch <= 0 or sequence <= 0:
      return
    if self.epoch is not None and epoch < self.epoch:
      return
    if epoch != self.epoch:
      IssuedRegistry.__init__(self)
      self.epoch = epoch
    key = (epoch, sequence)
    now_ns = _time_ns(now)
    if now_ns is None:
      return
    item = (now_ns, float(target), bool(source_valid), source, target_ns)
    if key in self.issued:
      if self.issued[key] != item:
        self.issued[key] = None
      return
    if sequence <= self.highwater or self.overflow:
      return
    self.highwater = sequence
    for old_key, old in list(self.issued.items()):
      if old is None or now - old[0] / 1e9 > 0.100000001:
        del self.issued[old_key]
    if len(self.issued) >= 64:
      self.overflow = True
      self.issued.clear()
      return
    self.issued[key] = item


REFERENCE_PREFIX = "bounded-release-reference-v2:"
PHASES = ("inactive", "brake", "precharge", "release", "engineRamp", "drive", "fault")


@dataclass(frozen=True)
class ReferenceReceipt(ReleaseReceipt):
  actual_gas: float = -5.0
  actual_limited: bool = True
  actual_phase: str = "release"
  nominal: dict = None
  model_id: str = ""

  def encode(self):
    return _encode(REFERENCE_PREFIX, self)

  @classmethod
  def parse(cls, text):
    try:
      if not isinstance(text, str) or not text.startswith(REFERENCE_PREFIX) or len(text) > MAX_WITNESS:
        return None

      def unique(pairs):
        out = {}
        for k, v in pairs:
          if k in out:
            raise ValueError("duplicate")
          out[k] = v
        return out

      data = json.loads(text[len(REFERENCE_PREFIX) :], object_pairs_hook=unique)
      if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)}:
        return None
      r = cls(**data)
      base = {f.name: getattr(r, f.name) for f in fields(ReleaseReceipt)}
      if ReleaseReceipt.parse(ReleaseReceipt(**base).encode()) is None:
        return None
      if type(r.actual_gas) not in (int, float) or not _finite(r.actual_gas):
        return None
      if type(r.actual_limited) is not bool or r.actual_phase not in PHASES:
        return None
      if not isinstance(r.nominal, dict) or not isinstance(r.model_id, str) or not r.model_id:
        return None
      return r
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
      return None


def nominal_feedback(r):
  from opendbc.car.ford.longitudinal_coordination import Feedback, allocator_feedback_valid

  d = r.nominal
  if set(d) != {f.name for f in fields(Feedback)}:
    return None
  bools = ("brake", "limited", "valid", "fault", "precharge", "reference_brake")
  ints = ("epoch", "sequence", "allocator_version")
  nums = ("t", "requested", "gas", "precharge_age", "budget_remaining")
  if any(type(d[k]) is not bool for k in bools):
    return None
  if any(type(d[k]) is not int for k in ints):
    return None
  if any(type(d[k]) not in (int, float) or not _finite(d[k]) for k in nums):
    return None
  if not isinstance(d["reason"], str) or d["allocator_phase"] not in PHASES:
    return None
  if not d["valid"] or d["fault"] or d["epoch"] <= 0 or d["sequence"] <= 0:
    return None
  f = Feedback(**d)
  if not allocator_feedback_valid(f):
    return None
  return f


class ReferenceRegistry(IssuedRegistry):
  def __init__(self, model_id):
    super().__init__()
    self.model_id = model_id
    self.last_nominal = None

  def verify(self, f, **kw):
    self.last_nominal = None

    def fail(reason):
      self.last_reason = reason
      return False

    r = ReferenceReceipt.parse(getattr(f, "reason", None))
    if r is None:
      return fail("invalid_reference_receipt")
    if r.model_id != MODEL_ID or self.model_id != MODEL_ID:
      return fail("model_id")
    if self.overflow:
      return fail("registry_overflow")
    if not (all(type(getattr(f, k)) is bool for k in ("brake", "precharge", "valid", "fault")) and not f.brake and not f.precharge and f.valid and not f.fault):
      return fail("actual_flags")
    if not (type(f.limited) is bool and f.limited == r.actual_limited and f.gas == r.actual_gas and f.allocator_phase == r.actual_phase):
      return fail("actual_match")
    if not (r.actual_gas == -5.0 or -0.5 <= r.actual_gas <= 0.3):
      return fail("actual_gas_range")
    n = nominal_feedback(r)
    if n is None:
      return fail("invalid_nominal")
    if any(type(getattr(f, k)) is not int for k in ("epoch", "sequence", "allocator_version")):
      return fail("actual_types")
    if f.allocator_version != 3:
      return fail("actual_version")
    if not (n.epoch == f.epoch == r.epoch and n.sequence == f.sequence == r.sequence and n.t == f.t and n.requested == f.requested):
      return fail("nominal_binding")
    # V2 references the COMPLETE nominal antiwindup policy. Current PID
    # inputs remain real and finite, but need not remain in V1's weak band.
    # Actor/source admission remains weak; no actuator permission is changed.
    now = kw["now"]
    epoch = kw["epoch"]
    sequence = kw["sequence"]
    target = kw["target"]
    error = kw["error"]
    wanted = kw["wanted"]
    speed = kw["speed"]
    source = kw.get("source", SOURCE)
    if any(type(v) is not int for v in (epoch, sequence)) or epoch <= 0 or sequence <= 0:
      return fail("current_identity")
    if not all(_finite(v) for v in (now, target, error, wanted, speed, f.t, f.gas, f.requested)):
      return fail("nonfinite")
    if not (kw.get("source_valid", True) and kw.get("eligible", False) and speed >= 3.0):
      return fail("current_scope")
    if not (0.02 < f.requested <= 0.3):
      return fail("issued_raw_scope")
    if not (r.source == SOURCE == source and r.epoch == epoch == f.epoch and 0 < r.sequence == f.sequence <= sequence):
      return fail("identity")
    now_ns = _time_ns(now)
    if now_ns is None:
      return fail("invalid_clock")
    if not (r.selected_ns == int(round(f.t * 1e9)) and 0 <= now_ns - r.selected_ns <= 100_000_001 and 0 <= now_ns - r.producer_ns <= 100_000_001):
      return fail("freshness")
    if not (
      0 < r.start_ns <= r.selected_ns < r.until_ns <= r.start_ns + 300_000_000
      and now_ns < r.until_ns
      and r.producer_ns <= r.selected_ns
      and 0 < r.budget_remaining <= 0.01
    ):
      return fail("lease")
    item = self.issued.get((epoch, r.sequence))
    if item is None or item[0] != r.producer_ns or not item[2] or item[3] != source or item[4] != r.target_ns:
      return fail("not_issued")
    if not (abs(item[1]) <= 0.2):
      return fail("issued_scope")
    observation = (r.epoch, r.sequence, r.selected_ns)
    if self.last_observation is not None and observation < self.last_observation:
      return fail("out_of_order")
    key = (epoch, r.episode)
    old = self.episodes.get(key)
    identity = (r.start_ns, r.until_ns, r.source)
    if old is not None and (identity != old[:3] or r.budget_remaining > old[3] + 1e-12):
      return fail("renewal")
    for k, v in self.episodes.items():
      if k != key and (r.start_ns <= v[1] or r.start_ns == v[0]):
        return fail("overlapping_episode")
    self.episodes = {key: (*identity, r.budget_remaining)}
    self.last_observation = observation
    self.last_nominal = n
    self.last_reason = "bounded_release_reference"
    return True


@dataclass(frozen=True)
class ReferenceContext:
  now_ns: int = 0
  epoch: int = 0
  sequence: int = 0
  pose_valid: bool = False
  pitch: float = 0.0
  eligible: bool = False


def make_reference_context(CC, CS, now_ns, source_valid=True, urgent=False):
  q = CC.navigatorLongitudinal
  cs = getattr(CS, "out", CS)
  orientation = CC.orientationNED
  pose_time = q.poseMonoTime
  pose = bool(
    q.poseValid
    and q.referenceValid
    and 0 < pose_time <= now_ns
    and now_ns - pose_time <= 100_000_000
    and len(orientation) == 3
    and all(_finite(v) for v in orientation)
  )
  eligible = bool(source_valid and pose and CC.enabled and CC.longActive and cs.canValid and not cs.gasPressed and not cs.brakePressed and not urgent)
  return ReferenceContext(int(now_ns), int(q.epoch), int(q.sequence), pose, float(orientation[1]) if pose else 0.0, eligible)


class ReferenceIntegral:
  def __init__(self):
    self.counts = Counter()
    self.reset()

  def reset(self):
    self.registry = ReferenceRegistry(MODEL_ID)
    self.current = ReferenceContext()
    self.diagnostic = {}
    self.counts["reset_invalidations"] += 1

  def begin(self, context):
    self.current = context

  def issue(self, epoch, seq, now, target, source_valid=True, source=SOURCE, target_ns=0):
    self.registry.issue(epoch, seq, now, target, source_valid, source, target_ns)

  def nominal(self, rc, t, owner, seq, error, target, wanted, speed, phase, fb, enabled=False):
    from opendbc.car.ford.longitudinal_coordination import freeze_integral

    actual = freeze_integral(t, owner, seq, error, target, wanted, speed, phase, fb, enabled=enabled)
    r = ReferenceReceipt.parse(fb.reason) if fb is not None else None
    current = self.current
    exact = current.now_ns == _time_ns(t) and current.epoch == owner and current.sequence == seq
    age = t - r.selected_ns / 1e9 if r else float("inf")
    gravity = float("nan")
    if r and _finite(age) and age >= 0 and current.pose_valid:
      offset = max(0.0, r.grade_offset - 0.005 * age) * max(0.0, min(1.0, (speed - 3) / 2))
      gravity = math.sin(current.pitch) * 9.81 + offset
    scoped = bool(
      exact
      and enabled
      and phase == "pid"
      and rc.valid
      and rc.sample is not None
      and rc.sample[:3] == (t, owner, seq)
      and current.eligible
      and _finite(gravity)
      and fb is not None
      and _finite(fb.requested)
      and 0.02 < fb.requested <= 0.3
    )
    owned = self.registry.verify(
      fb,
      now=t,
      epoch=owner,
      sequence=seq,
      target=target,
      error=error,
      wanted=wanted,
      speed=speed,
      source_valid=current.eligible,
      source=SOURCE,
      eligible=scoped,
    )
    self.counts["owned" if owned else "declined"] += 1
    reference = freeze_integral(t, owner, seq, error, target, wanted, speed, phase, self.registry.last_nominal, enabled=enabled) if owned else None
    self.diagnostic = dict(
      nominal_frozen=bool(actual),
      owned=bool(owned),
      receipt_reason=self.registry.last_reason,
      net_lower_bound=wanted + gravity if _finite(gravity) else None,
      selected_ns=r.selected_ns if r else None,
      reference_frozen=reference,
    )
    return bool(reference if reference is not None else actual)
