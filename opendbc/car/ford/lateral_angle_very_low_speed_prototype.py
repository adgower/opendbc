"""Experimental Navigator trial selected by the existing startup low-speed option.

A 10% feedforward sensitivity trial, not a validated vehicle calibration.
All inherited clipping, rate bounds, shadow feedback and override remain active.
"""
from opendbc.car.ford.lateral_angle_low_speed_prototype import LowSpeedGainPrototype, smoothstep
from opendbc.car.ford.values import CAR, FordFlags

EXTRA_LOW_SPEED_GAIN = 0.10
FADE_START_MPS = 15 * .44704
FADE_END_MPS = 25 * .44704


class VeryLowSpeedGainPrototype(LowSpeedGainPrototype):
  def __init__(self, CP, *, enabled: bool = False):
    super().__init__(CP)
    self.very_low_speed_enabled = bool(enabled)

  def get_curvature_factor(self, v_ego: float, kappa_cmd: float) -> float:
    current = super().get_curvature_factor(v_ego, kappa_cmd)
    if (not self.very_low_speed_enabled or self.CP.carFingerprint != CAR.FORD_EXPEDITION_MK4
        or not self.CP.flags & FordFlags.PINION_CURVATURE
        or not self.CP.flags & FordFlags.LOW_SPEED_GAIN or v_ego >= FADE_END_MPS):
      return current
    weight = 1. - smoothstep((v_ego - FADE_START_MPS) / (FADE_END_MPS - FADE_START_MPS))
    return current * (1. + EXTRA_LOW_SPEED_GAIN * weight)
