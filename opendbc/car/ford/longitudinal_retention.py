"""Optional Navigator retention ownership; nominal brake history remains authoritative."""

import math
from dataclasses import dataclass

REQUEST_VERSION = 1
TAG = 'navigator_retention_v1'
HOLD_NS = 100_000_000


class ExtraRetentionOwner:
  def __init__(self):
    self.last = None
    self.epoch = None
    self.sequence = None
    self.holding = False
    self.spent = False
    self.until_ns = 0
    self.tracking = None
    self.track_source = None
    self.latest_source = None

  def update(
    self, t, owner, active, eligible, reference, nominal_precharge, qualified, previous_actual, clear, raw, net, actual, target, epoch, sequence, source_ns=None
  ):
    if (
      not isinstance(t, (int, float)) or not math.isfinite(t) or not isinstance(source_ns, int) or not 0 < source_ns <= t * 1e9 or t * 1e9 - source_ns > HOLD_NS
    ):
      self.holding = False
      self.tracking = None
      self.track_source = None
      if self.until_ns:
        self.spent = True
      self.last = None
      return False
    previous = self.last
    self.last = t
    self.holding = False
    if not owner:
      self.__init__()
      self.last = t
      return False
    identity = epoch > 0 and sequence > 0 and (self.epoch is None or self.epoch == epoch) and (self.sequence is None or sequence >= self.sequence)
    clock = math.isfinite(t) and (previous is None or 0 < t - previous <= 0.100000001)
    if not identity:
      self.spent = True
      self.tracking = None
      return False
    self.epoch = epoch
    self.sequence = sequence
    if clock and active and eligible and not reference and clear:
      self.spent = False
      self.until_ns = 0
      self.tracking = None
    source_ok = source_ns > 0 and (self.latest_source is None or source_ns >= self.latest_source)
    self.latest_source = source_ns
    scoped = source_ok and clock and active and eligible and all(math.isfinite(v) for v in [raw, net, actual, target]) and target > -0.5
    if scoped and actual <= target:
      if self.tracking is None:
        self.tracking = t
        self.track_source = source_ns
    else:
      self.tracking = None
      self.track_source = None
    good = scoped and raw > 0.02 and 0 <= net <= 0.3
    if not good or reference or nominal_precharge:
      if self.until_ns:
        self.spent = True
      return False
    if self.until_ns and round(t * 1e9) >= self.until_ns:
      self.spent = True
    if self.tracking is not None and t - self.tracking >= 0.06 - 1e-9 and source_ns - self.track_source >= 60_000_000:
      if self.until_ns:
        self.spent = True
      return False
    if self.spent:
      return False
    if not self.until_ns:
      if not qualified or not previous_actual:
        return False
      self.until_ns = round(t * 1e9) + HOLD_NS
    self.holding = True
    return True


class OwnerReceipt:
  def __init__(self):
    self.latest = None
    self.payload = None
    self.conflicted = False
    self.episode_until = None

  def freeze(self, t, epoch, sequence, error, target, wanted, speed, phase, f, enabled=False, current_valid=False):
    if f is None:
      return False
    required = ['epoch', 't', 'sequence', 'requested', 'gas', 'valid', 'fault', 'allocator_version', 'brake', 'precharge', 'reference_brake', 'allocator_phase']
    if any(not hasattr(f, k) for k in required):
      return False
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in [t, error, target, wanted, speed, f.t, f.requested, f.gas, f.epoch, f.sequence]):
      return False
    if not (epoch > 0 and f.epoch == epoch and 0 < f.sequence <= sequence and 0 <= t - f.t <= 0.100000001):
      return False
    if not f.valid or f.fault or f.allocator_version != 3:
      return False
    key = (f.epoch, f.t, f.sequence)
    payload = tuple(getattr(f, k) for k in required) + (getattr(f, 'reason', ''),)
    if self.latest is not None and self.latest[0] == epoch:
      if f.t < self.latest[1] or f.sequence < self.latest[2]:
        return False
      if key == self.latest:
        if payload != self.payload:
          self.conflicted = True
        if self.conflicted:
          return False
      else:
        self.conflicted = False
    else:
      self.conflicted = False
      self.episode_until = None
    self.latest = key
    self.payload = payload
    prefix = TAG + '|until_ns='
    reason = getattr(f, 'reason', '')
    if not isinstance(reason, str):
      return False
    if not reason.startswith(prefix):
      if not reason.startswith(TAG):
        self.episode_until = None
      return False
    receipt = RetentionReceipt.parse(reason)
    if receipt is None:
      return False
    until = receipt.until_ns
    now = round(t * 1e9)
    ft = round(f.t * 1e9)
    if not ft < until <= ft + HOLD_NS:
      return False
    if self.episode_until is not None and until != self.episode_until:
      return False
    self.episode_until = until
    return bool(
      enabled
      and current_valid
      and phase == 'pid'
      and speed >= 40 * 0.44704
      and all(math.isfinite(v) for v in [t, error, target, wanted, speed, f.t, f.requested, f.gas])
      and ft < until <= ft + HOLD_NS
      and now < min(ft + HOLD_NS, until + HOLD_NS)
      and 0 <= t - f.t <= 0.100000001
      and epoch > 0
      and f.epoch == epoch
      and 0 < f.sequence <= sequence
      and f.valid
      and not f.fault
      and f.allocator_version == 3
      and f.brake
      and f.precharge
      and not f.reference_brake
      and f.allocator_phase == 'brake'
      and f.gas == -5.0
      and error > 0
      and target >= 0
      and wanted > 0.02
      and f.requested > 0.02
    )


@dataclass(frozen=True)
class RetentionReceipt:
  until_ns: int

  def encode(self):
    if not isinstance(self.until_ns, int) or not 0 < self.until_ns < 2**64:
      raise ValueError("invalid retention deadline")
    return TAG + '|until_ns=' + str(self.until_ns)

  @classmethod
  def parse(cls, reason):
    prefix = TAG + '|until_ns='
    if not isinstance(reason, str) or not reason.startswith(prefix):
      return None
    suffix = reason[len(prefix) :]
    if not suffix or len(suffix) > 20 or not suffix.isascii() or not suffix.isdecimal():
      return None
    value = int(suffix)
    return cls(value) if 0 < value < 2**64 else None


def make_retention_owner(*, enabled=False):
  return ExtraRetentionOwner() if enabled else None


def make_receipt_tracker(*, enabled=False):
  return OwnerReceipt() if enabled else None


def speed_sample_ns(parser):
  try:
    value = parser.ts_nanos['BrakeSysFeatures']['Veh_V_ActlBrk']
    return value if isinstance(value, int) and 0 < value < 2**64 else 0
  except (AttributeError, KeyError, TypeError):
    return 0


def consumer_scope(controller, CC, CS, now_ns):
  q = CC.navigatorLongitudinal
  state_ns = getattr(controller, 'coordination_state_ns', 0)
  speed_ns = getattr(controller, 'coordination_speed_ns', 0)
  sources = bool(
    getattr(controller, 'coordination_state_valid', False)
    and 0 < speed_ns <= state_ns <= now_ns
    and now_ns - speed_ns <= HOLD_NS
    and now_ns - state_ns <= HOLD_NS
  )
  target = bool(
    q.poseValid and q.referenceValid and q.targetValid and 0 < q.targetMonoTime <= q.producerMonoTime <= now_ns and now_ns - q.targetMonoTime <= HOLD_NS
  )
  return bool(
    sources
    and target
    and getattr(q, 'retentionVersion', 0) == REQUEST_VERSION
    and CS.out.canValid
    and CS.out.vEgo >= 40 * 0.44704
    and not CS.out.gasPressed
    and not CS.out.brakePressed
    and str(CC.actuators.longControlState) == 'pid'
    and str(CC.hudControl.visualAlert) != 'fcw'
  )
