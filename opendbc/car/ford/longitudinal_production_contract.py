"""Versioned bounded facts over trusted IPC; never a native300 receipt.

Source and process identities detect incompatible peers and stale histories. They
are provenance checks, not cryptographic authentication of the local IPC peer.
"""
from dataclasses import dataclass, asdict, fields
import json
import math
import struct

from opendbc.car.ford.longitudinal_coordination import Feedback, allocator_feedback_valid

CONTRACT_ID = 'navigator-production-v5'
SOURCE_ID = 'navigator-production-producer-v1'
REQUEST_VERSION = 5
PREFIX = 'navigator-production-v5:'
MAX_WITNESS = 8192
MAX_ISSUED = 64
FRESH_NS = 100_000_000
NATIVE_NS = 300_000_000
ACTOR_NS = 500_000_000


def finite(value):
  return type(value) in (int, float) and math.isfinite(value)


def _unique(pairs):
  out = {}
  for key, value in pairs:
    if key in out:
      raise ValueError('duplicate field')
    out[key] = value
  return out


def _feedback(data):
  if type(data) is not dict or set(data) != {f.name for f in fields(Feedback)}:
    return None
  if any(type(data[k]) is not bool for k in ('brake', 'limited', 'valid', 'fault', 'precharge', 'reference_brake')):
    return None
  if any(type(data[k]) is not int for k in ('epoch', 'sequence', 'allocator_version')):
    return None
  if any(not finite(data[k]) for k in ('t', 'requested', 'gas', 'precharge_age', 'budget_remaining')):
    return None
  if type(data['reason']) is not str or len(data['reason']) > 256:
    return None
  f = Feedback(**data)
  return f if allocator_feedback_valid(f) else None


@dataclass(frozen=True)
class ProductionWitness:
  contract_id: str
  source_id: str
  run_id: int
  epoch: int
  sequence: int
  producer_ns: int
  target_ns: int
  target: float
  pose_ns: int
  selected_ns: int
  actual_raw: float
  nominal_raw: float
  nominal_phase: str
  actual: dict
  nominal: dict
  grade_offset: float
  executed_pose_ns: int = 0
  birth_ns: int = 0
  native_until_ns: int = 0
  actor_until_ns: int = 0
  remaining: float = 0.
  selected: bool = False
  retired: bool = False

  @property
  def until_ns(self):
    return self.actor_until_ns

  @property
  def raw(self):
    return self.actual_raw

  def actual_feedback(self):
    return _feedback(self.actual)

  def nominal_feedback(self):
    return _feedback(self.nominal)

  def valid(self):
    if self.contract_id != CONTRACT_ID or any(type(v) is not str or not 0 < len(v) <= 128 for v in (self.source_id,)):
      return False
    ints = (self.run_id, self.epoch, self.sequence, self.producer_ns, self.target_ns, self.pose_ns, self.selected_ns)
    if any(type(v) is not int or not 0 < v < 2**64 for v in ints):
      return False
    if (type(self.executed_pose_ns) is not int or not 0 < self.executed_pose_ns <= self.selected_ns
        or self.selected_ns - self.executed_pose_ns > FRESH_NS
        or self.executed_pose_ns < self.pose_ns):
      return False
    if not (self.target_ns <= self.producer_ns <= self.selected_ns and self.pose_ns <= self.selected_ns):
      return False
    if any(not finite(v) for v in (self.target, self.actual_raw, self.nominal_raw, self.grade_offset, self.remaining)):
      return False
    if not 0 <= self.grade_offset <= .149914 or self.nominal_phase not in ('off', 'pid', 'stopping', 'starting'):
      return False
    if type(self.selected) is not bool or type(self.retired) is not bool or self.selected and self.retired:
      return False
    a, n = self.actual_feedback(), self.nominal_feedback()
    if a is None or n is None:
      return False
    if any((f.epoch, f.sequence, round(f.t * 1e9)) != (self.epoch, self.sequence, self.selected_ns) for f in (a, n)):
      return False
    if a.requested != self.actual_raw or n.requested != self.nominal_raw:
      return False
    horizon = (self.birth_ns, self.native_until_ns, self.actor_until_ns)
    if any(type(v) is not int or not 0 <= v < 2**64 for v in horizon):
      return False
    if self.birth_ns:
      if (self.native_until_ns != self.birth_ns + NATIVE_NS or self.actor_until_ns != self.birth_ns + ACTOR_NS
          or not self.birth_ns <= self.selected_ns or not 0 <= self.remaining <= .01):
        return False
    elif any(horizon) or self.remaining != 0 or self.selected or self.retired:
      return False
    if self.selected and not (self.selected_ns < self.actor_until_ns and self.remaining > 0
        and a.valid and not a.fault and not a.brake and not a.precharge and a.gas == -5.
        and a.budget_remaining == self.remaining
        and a.precharge_age == (self.selected_ns - self.birth_ns) / 1e9):
      return False
    return True

  def encode(self):
    if not self.valid():
      raise ValueError('invalid production witness')
    text = PREFIX + json.dumps(asdict(self), sort_keys=True, separators=(',', ':'), allow_nan=False)
    if len(text) > MAX_WITNESS:
      raise ValueError('production witness too large')
    return text

  @classmethod
  def parse(cls, text):
    try:
      if type(text) is not str or not text.startswith(PREFIX) or len(text) > MAX_WITNESS:
        return None
      data = json.loads(text[len(PREFIX):], object_pairs_hook=_unique)
      if type(data) is not dict or set(data) != {f.name for f in fields(cls)}:
        return None
      witness = cls(**data)
      return witness if witness.valid() and witness.encode() == text else None
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
      return None


def request_identity(CC):
  q = CC.navigatorLongitudinal
  return (int(q.productionRunId), int(q.epoch), int(q.sequence), int(q.producerMonoTime),
          int(q.targetMonoTime), float(q.targetAcceleration), int(q.productionPoseMonoTime),
          float(CC.actuators.accel), float(q.nominalRawAcceleration), str(q.nominalLongControlState))


def request_valid(CC, now_ns, source_id=SOURCE_ID):
  try:
    q = CC.navigatorLongitudinal
    values = (q.productionRunId, q.epoch, q.sequence, q.producerMonoTime)
    return bool(q.allocatorVersion == REQUEST_VERSION and q.productionContractId == CONTRACT_ID
      and q.productionSourceId == source_id and all(type(v) is int and 0 < v < 2**64 for v in values)
      and q.nominalValid and all(finite(v) for v in (CC.actuators.accel, q.nominalRawAcceleration, q.targetAcceleration))
      and struct.unpack('f', struct.pack('f', q.nominalRawAcceleration))[0] == q.nominalRawAcceleration
      and q.nominalLongControlState.raw in (0, 1, 2, 3)
      and str(q.nominalLongControlState) in ('off', 'pid', 'stopping', 'starting')
      and 0 < q.producerMonoTime <= now_ns and now_ns - q.producerMonoTime <= FRESH_NS)
  except (AttributeError, ValueError, TypeError, OverflowError, RuntimeError):
    return False


class ProductionIssuedRegistry:
  """Producer-owned bounded issuance; feedback never creates request history."""
  def __init__(self, source_id=SOURCE_ID):
    self.source_id = source_id
    self.issued = {}
    self.run_id = None
    self.highwater = None
    self.birth = None
    self.remaining = .01
    self.observation = None
    self.observation_text = None
    self.last_reason = 'unissued'
    self.overflow = False

  def issue(self, CC):
    q = CC.navigatorLongitudinal
    if not request_valid(CC, int(q.producerMonoTime), self.source_id):
      self.last_reason = 'invalid_issue'
      return False
    identity = request_identity(CC)
    run, epoch, sequence, producer_ns = identity[:4]
    if self.run_id is None:
      self.run_id = run
    if run != self.run_id or self.overflow:
      self.last_reason = 'run_or_overflow'
      return False
    key = (epoch, sequence)
    if key in self.issued:
      if self.issued[key] != identity:
        self.issued[key] = None
      return self.issued[key] is not None
    order = (epoch, sequence, producer_ns)
    if self.highwater is not None and (epoch < self.highwater[0] or sequence <= self.highwater[1] and epoch == self.highwater[0]
                                      or producer_ns < self.highwater[2]):
      self.last_reason = 'issue_order'
      return False
    self.highwater = order
    self.issued = {k: v for k, v in self.issued.items() if v is not None and 0 <= producer_ns - v[3] <= FRESH_NS}
    if len(self.issued) >= MAX_ISSUED:
      self.issued.clear()
      self.overflow = True
      self.last_reason = 'overflow'
      return False
    self.issued[key] = identity
    return True

  def verify(self, feedback, *, now_ns, cutoff_ns, epoch, sequence):
    def fail(reason):
      self.last_reason = reason
      return None
    w = ProductionWitness.parse(getattr(feedback, 'reason', None))
    if w is None:
      return fail('witness')
    if self.overflow or w.source_id != self.source_id or w.run_id != self.run_id:
      return fail('source_run')
    if (type(now_ns) is not int or type(cutoff_ns) is not int or not 0 < cutoff_ns <= now_ns
        or w.epoch != epoch or not 0 < w.sequence <= sequence):
      return fail('current_identity_cutoff')
    if not (0 <= now_ns - w.selected_ns <= FRESH_NS and 0 <= now_ns - w.producer_ns <= FRESH_NS
            and w.selected_ns <= cutoff_ns):
      return fail('freshness_cutoff')
    issued = self.issued.get((w.epoch, w.sequence))
    binding = (w.run_id, w.epoch, w.sequence, w.producer_ns, w.target_ns, w.target, w.pose_ns,
               w.actual_raw, w.nominal_raw, w.nominal_phase)
    if issued != binding:
      return fail('not_issued')
    actual = w.actual_feedback()
    observed = asdict(feedback)
    observed['reason'] = actual.reason
    if observed != asdict(actual):
      return fail('actual_facts')
    if not w.birth_ns or not now_ns < w.actor_until_ns:
      return fail('actor_horizon')
    # The actor's500ms lease cannot lend nominal antiwindup across native300.
    if w.selected_ns < w.native_until_ns <= now_ns:
      return fail('native_horizon')
    birth = (w.epoch, w.birth_ns, w.native_until_ns, w.actor_until_ns)
    if self.birth is not None and birth != self.birth:
      return fail('renewal')
    if w.remaining > self.remaining + 1e-12:
      return fail('budget_refund')
    observation = (w.epoch, w.sequence, w.selected_ns)
    if self.observation is not None:
      if observation < self.observation or w.selected_ns < self.observation[2]:
        return fail('feedback_order')
      if observation == self.observation and feedback.reason != self.observation_text:
        return fail('conflicting_selection')
    self.birth, self.remaining, self.observation = birth, w.remaining, observation
    self.observation_text = feedback.reason
    self.last_reason = 'production_history'
    return w


WIRE_FIELDS = {'epoch': 'epoch', 'sequence': 'sequence', 'requested': 'requested', 'gas': 'gas',
  'brake': 'brake', 'limited': 'limited', 'valid': 'valid', 'fault': 'fault', 'allocatorVersion': 'allocator_version',
  'precharge': 'precharge', 'referenceBrake': 'reference_brake', 'allocatorPhase': 'allocator_phase',
  'prechargeAge': 'precharge_age', 'budgetRemaining': 'budget_remaining', 'reason': 'reason'}


def write_nominal_wire(wire, nominal, *, source_id, run_id):
  wire.productionContractId = CONTRACT_ID
  wire.productionSourceId = source_id
  wire.productionRunId = run_id
  if nominal is not None and nominal.allocator_phase not in ('inactive', 'brake', 'precharge', 'release', 'engineRamp', 'drive', 'fault'):
    nominal = None
  wire.nominalValid = nominal is not None
  if nominal is None:
    return
  n = wire.nominalFeedback
  n.producerMonoTime = round(nominal.t * 1e9)
  for name, attribute in WIRE_FIELDS.items():
    setattr(n, name, getattr(nominal, attribute))


def nominal_from_wire(wire, *, source_id=SOURCE_ID, run_id, now_ns, cutoff_ns, epoch, sequence):
  try:
    if (wire.productionContractId != CONTRACT_ID or wire.productionSourceId != source_id
        or type(run_id) is not int or run_id <= 0 or wire.productionRunId != run_id or not wire.nominalValid
        or not 0 < cutoff_ns <= now_ns):
      return None
    n = wire.nominalFeedback
    if not (n.epoch == wire.epoch == epoch and 0 < n.sequence == wire.sequence <= sequence
            and 0 < n.producerMonoTime == wire.producerMonoTime <= cutoff_ns
            and 0 <= now_ns - n.producerMonoTime <= FRESH_NS):
      return None
    data = {attribute: getattr(n, name) for name, attribute in WIRE_FIELDS.items()}
    data['allocator_phase'] = str(data['allocator_phase'])
    data['t'] = n.producerMonoTime / 1e9
    return _feedback(data)
  except (AttributeError, ValueError, TypeError, OverflowError, RuntimeError):
    return None
