"""Host-only hypothetical frame cadence. Emission does not imply admission."""
from dataclasses import dataclass, replace
from opendbc.car.ford.navigator_a3 import CADENCE_NS, MAX_AGE_NS, Inputs, Output, Profile, State, update


@dataclass(frozen=True)
class Decision:
  output: Output | None
  reason: str
  waiting: bool
  reset: bool
  next_eligible_ns: int | None
  proposal_counter: int | None


class ShadowScheduler:
  def __init__(self):
    self.state = State()
    self.last_call_ns = None
    self.last_emission_ns = None
    self.counter = 0
    self.neutral = False

  @property
  def next_eligible_ns(self):
    return None if self.last_emission_ns is None else self.last_emission_ns + CADENCE_NS

  def step(self, profile: Profile | None, sample: Inputs) -> Decision:
    now = sample.now_ns
    if self.last_call_ns is not None and now <= self.last_call_ns:
      return Decision(None, 'nonmonotonic_time', False, False, self.next_eligible_ns, None)
    gap = self.last_call_ns is not None and now - self.last_call_ns > MAX_AGE_NS
    self.last_call_ns = now
    # Probe validity on a copy; only emitted proposals commit demand history.
    # The scheduler owns timing, so the pure strategy's cadence check is disabled.
    result = update(profile, replace(self.state, last_ns=None), replace(sample, scheduled_update=True))
    if gap:
      result = replace(result, state=State(0., now), mode=0, path_angle_rad=0., reason='timing_gap',
                       equivalent_curvature_inv_m=None)
    if result.mode == 0:
      if self.neutral:
        return Decision(None, result.reason, False, False, self.next_eligible_ns, None)
      self.neutral = True
    else:
      if self.next_eligible_ns is not None and now < self.next_eligible_ns:
        return Decision(None, 'waiting', True, False, self.next_eligible_ns, None)
      self.neutral = False
    self.state = result.state
    self.last_emission_ns = now
    counter = self.counter
    self.counter = (self.counter + 1) % 16
    return Decision(result, result.reason, False, result.mode == 0, self.next_eligible_ns, counter)
