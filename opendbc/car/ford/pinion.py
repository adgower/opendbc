"""Configure the startup-only Expedition fixed-geometry feedback option."""
import math

from opendbc.car.ford.values import CAR, FordFlags, FordSafetyFlags
from opendbc.car.structs import CarParams

# Must match the geometry baked into ford_pinion_curvature_factor in ford.h.
PINION_GEOMETRY = {
  'wheelbase': 3.1115,
  'mass': 2878.,
  'centerToFront': 3.1115 * .44,
  'steerRatio': 19.6,
  'steerRatioRear': 0.,
  'tireStiffnessFront': 353037.28125,
  'tireStiffnessRear': 438491.46875,
}


def configure_pinion_curvature(cp: CarParams, enabled: bool) -> None:
  if cp.brand != 'ford':
    return
  supported = cp.carFingerprint == CAR.FORD_EXPEDITION_MK4 and bool(cp.flags & FordFlags.CANFD)
  if enabled and supported:
    if not all(math.isclose(getattr(cp, key), value, rel_tol=1e-6, abs_tol=1e-6) for key, value in PINION_GEOMETRY.items()):
      raise ValueError('Ford pinion feedback geometry does not match the safety model')
    cp.flags |= FordFlags.PINION_CURVATURE.value
    cp.safetyConfigs[-1].safetyParam |= FordSafetyFlags.PINION_CURVATURE.value
  else:
    cp.flags &= ~FordFlags.PINION_CURVATURE.value
    cp.safetyConfigs[-1].safetyParam &= ~FordSafetyFlags.PINION_CURVATURE.value
