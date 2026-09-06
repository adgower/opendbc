"""Owner-selected startup experiment, not automatic Navigator identification.

A2-WB: standard-wheelbase 2023 Navigator 4WD, mass unresolved. The retained
steering ratio, front-CG fraction and tire factor are model assumptions. See the
parent checkout's docs/navigator_a2_parameter_audit.md for evidence and enablement.
Recorded CarParams must remain authoritative for historical replay/analysis.
"""
import math
import struct

from opendbc.car import structs
from opendbc.car.ford.values import CAR, FordFlags

PROFILE_NAME = '2023-navigator-swb-4wd'
# 122.5 inches * 0.0254 m/in. Enable only after the accompanying evidence audit.
WHEELBASE_M = 3.1115
# No generic mass environment override. None retains existing pre-payload mass.
APPROVED_CURB_MASS_KG: float | None = None


def navigator_a2_profile_enabled(selector: str | None, *, docs: bool, replay: bool = False) -> bool:
  """Parse the developer-only startup selector; generic documentation and replay ignore it."""
  if docs or replay:
    return False
  if selector is None or selector == 'off':
    return False
  if not isinstance(selector, str):
    raise TypeError('NAVIGATOR_A2_PROFILE must be a string or unset')
  if selector != PROFILE_NAME:
    raise ValueError(f'Unknown NAVIGATOR_A2_PROFILE: {selector!r}')
  return True


def apply_navigator_a2_parameters(ret: structs.CarParams, *, enabled: bool, curb_mass_kg: float | None) -> None:
  """Validate atomically, then change only scoped pre-payload physical inputs."""
  if not isinstance(enabled, bool):
    raise TypeError('enabled must be a bool')
  if not enabled:
    return
  if ret.brand != 'ford' or ret.carFingerprint != CAR.FORD_EXPEDITION_MK4 or not ret.flags & FordFlags.CANFD:
    raise ValueError('Navigator A2 requires the explicitly selected Ford Expedition CAN-FD platform')
  if curb_mass_kg is not None:
    if isinstance(curb_mass_kg, bool) or not isinstance(curb_mass_kg, (int, float)):
      raise TypeError('curb_mass_kg must be a real mass in kilograms or None')
    # CarParams.mass is Float32. Reject overflow/underflow before either write.
    try:
      curb_mass_kg = struct.unpack('f', struct.pack('f', float(curb_mass_kg)))[0]
    except (OverflowError, struct.error) as exc:
      raise ValueError('curb_mass_kg must be representable as a positive finite CarParams mass') from exc
    if not math.isfinite(curb_mass_kg) or curb_mass_kg <= 0:
      raise ValueError('curb_mass_kg must be representable as a positive finite CarParams mass')
  ret.wheelbase = WHEELBASE_M
  if curb_mass_kg is not None:
    ret.mass = curb_mass_kg
