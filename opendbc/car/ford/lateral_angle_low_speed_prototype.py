"""EXPERIMENTAL VEHICLE TRIAL. Explicit opt-in, with pinion feedback required.

Trial calibration, not validated vehicle tuning. Retains the owner's 1.20/0.97
coefficients and all inherited feedback, override, signal, and rate limits.
"""
from numpy import interp

from opendbc.car.ford.lateral_angle import LateralAngle
from opendbc.car.ford.values import CAR

GENTLE_TRIAL_MULTIPLIER = 1.15
TIGHT_TRIAL_MULTIPLIER = 1.05
FADE_START_MPS = 30 * .44704
FADE_END_MPS = 45 * .44704
BLEND_START = .00025
BLEND_END = .00125


def smoothstep(value: float) -> float:
  value = min(max(value, 0.), 1.)
  return value * value * (3. - 2. * value)


class LowSpeedGainPrototype(LateralAngle):
  def get_curvature_factor(self, v_ego: float, kappa_cmd: float) -> float:
    original = super().get_curvature_factor(v_ego, kappa_cmd)
    if self.CP.carFingerprint != CAR.FORD_EXPEDITION_MK4 or v_ego >= FADE_END_MPS:
      return original

    # Query unchanged endpoint gains so owner's coefficients stay authoritative.
    gentle = super().get_curvature_factor(v_ego, 0.) * GENTLE_TRIAL_MULTIPLIER
    tight = super().get_curvature_factor(v_ego, .01) * TIGHT_TRIAL_MULTIPLIER
    blend = smoothstep((abs(kappa_cmd) - BLEND_START) / (BLEND_END - BLEND_START))
    trial = float(interp(blend, [0., 1.], [gentle, tight]))
    weight = 1. - smoothstep((v_ego - FADE_START_MPS) / (FADE_END_MPS - FADE_START_MPS))
    return original + weight * (trial - original)
