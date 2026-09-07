"""Core path-angle experiment: offline only, never imported by vehicle control.

The donor's executable mapping is kappa * speed * gain, NOT its docstring's
half-kappa-d_ref formula. Profiles below are unvalidated research constants.
No method transmits CAN. Stock safety and the production controller are untouched.
"""
from dataclasses import dataclass
import math

from opendbc.car.ford import fordcan

DONOR_REVISION = '3210caa02d9e09b46d08be490f689edb23b22b20'
CADENCE_NS = 50_000_000
MAX_AGE_NS = 100_000_000


@dataclass(frozen=True)
class Profile:
  name: str
  base_gain: float
  low_factor: float
  high_factor: float
  dampening: float


PROFILES = {
  'expedition-provisional-v1': Profile('expedition-provisional-v1', 1.425, 1., 1., 1.),
  'bof-reference-v1': Profile('bof-reference-v1', .95, 1., 1., 1.),
  'sensitivity-low-v1': Profile('sensitivity-low-v1', 1.425, .9, 1., 1.),
  'sensitivity-high-v1': Profile('sensitivity-high-v1', 1.425, 1., .9, .9),
}


def select_profile(name: str | None) -> Profile | None:
  """Explicit offline startup configuration. No environment or persistent reads."""
  if name is None or name == 'off':
    return None
  if name not in PROFILES:
    raise ValueError(f'Unknown offline profile: {name!r}')
  return PROFILES[name]


def interp(x: float, xs: tuple, ys: tuple) -> float:
  for i in range(1, len(xs)):
    if x < xs[i]:
      weight = max(0., (x - xs[i - 1]) / (xs[i] - xs[i - 1]))
      return ys[i - 1] + weight * (ys[i] - ys[i - 1])
  return ys[-1]


def gain_for(profile: Profile, speed: float, curvature: float) -> float:
  low = interp(speed, (13.5, 26.82), (1., profile.base_gain * profile.dampening))
  high = interp(speed, (13.5, 26.82), (1.30 * profile.low_factor, profile.base_gain * profile.high_factor))
  return interp(abs(curvature), (.0007, .001), (low, high))


@dataclass(frozen=True)
class Inputs:
  now_ns: int
  source_ns: int
  curvature_inv_m: float  # same bounded general-controls request used by A2
  speed_mps: float
  active: bool
  driver_pressed: bool
  valid: bool = True


@dataclass(frozen=True)
class State:
  path_angle_rad: float = 0.
  last_ns: int | None = None


@dataclass(frozen=True)
class Output:
  state: State
  mode: int
  path_angle_rad: float
  raw_path_angle_rad: float
  effective_gain: float
  reason: str
  profile: str
  transmission_allowed: bool = False


def update(profile: Profile | None, state: State, sample: Inputs) -> Output:
  """Advance at 20 Hz; reset neutral on invalidity/driver input, never pulse.

  This immediate driver override differs from the donor's sustained human-turn
  detector. Its availability and re-entry behavior require independent review.
  Bounds here are not permission to actuate; compiled admission is separate.
  """
  reason = ''
  if profile is None:
    reason = 'disabled'
  elif profile not in PROFILES.values():
    raise ValueError('Only named audited offline profiles are supported')
  elif not sample.valid or not all(math.isfinite(v) for v in (sample.speed_mps, sample.curvature_inv_m, state.path_angle_rad)):
    reason = 'invalid'
  elif abs(sample.curvature_inv_m) > .02 or sample.speed_mps > 60.:
    reason = 'invalid'
  elif not 0 <= sample.now_ns - sample.source_ns <= MAX_AGE_NS:
    reason = 'stale'
  elif not sample.active:
    reason = 'inactive'
  elif sample.driver_pressed:
    reason = 'driver_override'
  elif sample.speed_mps < 1.:
    reason = 'low_speed'
  elif state.last_ns is not None and not CADENCE_NS <= sample.now_ns - state.last_ns <= MAX_AGE_NS:
    reason = 'cadence'
  name = profile.name if profile else 'off'
  if reason:
    return Output(State(0., sample.now_ns), 0, 0., 0., 0., reason, name)
  assert profile is not None
  gain = gain_for(profile, sample.speed_mps, sample.curvature_inv_m)
  raw = sample.curvature_inv_m * sample.speed_mps * gain
  roc = interp(sample.speed_mps, (9., 10., 15., 25.), (.055, .055, .0425, .009))
  # Host sign is opposite the Ford wire. Intersect rate and asymmetric DBC
  # bounds on the integer grid; truncating magnitude alone can violate unwind ROC.
  lower = math.ceil((max(-.5235, state.path_angle_rad - roc) - 1e-12) / .0005)
  upper = math.floor((min(.5, state.path_angle_rad + roc) + 1e-12) / .0005)
  if lower > upper:
    return Output(State(0., sample.now_ns), 0, 0., raw, gain, 'invalid_state', name)
  angle = min(upper, max(lower, round(raw / .0005))) * .0005
  return Output(State(angle, sample.now_ns), 1, angle, raw, gain, 'limited' if abs(angle - raw) > 1e-9 else 'tracking', name)


def encode_offline(packer, output: Output, counter: int):
  """Return inert (address, bytes, bus) for offline compiled-safety evaluation."""
  if not 0 <= counter <= 15:
    raise ValueError('LMC2 counter must be 0..15')
  return fordcan.create_lat_ctl2_msg(packer, fordcan.CanBus(fingerprint={0: {}, 1: {}, 2: {}}), output.mode, 0., -output.path_angle_rad, 0., 0., counter)
