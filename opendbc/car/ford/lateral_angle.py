"""
Ford CAN-FD path-angle-primary lateral control.

Converts planner curvature (kappa) into path_angle (c1) for the PSCM.
The steering intent is c1; c0 (path_offset), c2 (curvature), and c3 (curvature_rate)
are forced to zero in angle mode.

Live map: path_angle = kappa_cmd * v_ego * curvature_factor

Human-turn mode-0: when the driver manually turns (steeringPressed + large wheel angle),
lateral control is forced inactive (mode 0) so the PSCM releases cleanly.
"""
import math
from dataclasses import dataclass
from numpy import clip, interp

from opendbc.car import DT_CTRL
from opendbc.car.carlog import carlog
from opendbc.car.ford.values import CAR, CarControllerParams, FordFlags
from opendbc.car.vehicle_model import VehicleModel


@dataclass
class AngleLateralResult:
  """Result from angle-mode lateral strategy, used by CarController."""
  path_angle: float = 0.0
  shadow_curvature: float = 0.0


# DBC path_angle signal limits (rad). Safety uses the same in ford.h.
FORD_DBC_PATH_ANGLE_MIN = -0.5
FORD_DBC_PATH_ANGLE_MAX = 0.5235


# Neutral gain for non-tuned platforms (simple/neutral baseline)
_GAIN_NEUTRAL = (1.0, 1.0)

# PROVISIONAL FORK TUNING: Expedition gain from BluePilot bp-dev-expedition.
# NOT Navigator-verified. Subject to change with real-world validation.
_GAIN_EXPEDITION = (1.425, 1.425)

# PROVISIONAL FORK TUNING: BluePilot-equivalent speed factors.
# FordLowSpeedFactor_ang = 1.15 (+0.15 from default 1.0)
# FordHighSpeedFactor_ang = 0.98 (-0.02 from default 1.0)
# Applied to high-curvature gain arm only; dampening stays 1.0.
# Not a Params/UI port; baked constants for Navigator angle-mode testing.
_FORD_LOW_SPEED_FACTOR_ANG = 1.15
_FORD_HIGH_SPEED_FACTOR_ANG = 0.98

_EXPEDITION_CARS = frozenset({
  CAR.FORD_EXPEDITION_MK4,
})


def _get_platform_gains(car_fingerprint: str) -> tuple[float, float]:
  """Returns (low_curvature_gain, high_curvature_gain) for the platform.

  Only Expedition has tuned gains (provisional). All other platforms use neutral 1.0.
  """
  if car_fingerprint in _EXPEDITION_CARS:
    return _GAIN_EXPEDITION
  return _GAIN_NEUTRAL


# Soft ROC limit: path_angle rate-of-change per 20Hz frame.
# Matches lateral_angle_ext.py _soft_roc, scaled for STEER_STEP=5 (20Hz cadence).
# ford.h's FORD_PATH_ANGLE_LIMITS mirrors this with +2% headroom.
_SOFT_ROC_SPEEDS = [9., 10., 15., 25.]
_SOFT_ROC_VALUES = [0.055, 0.055, 0.0425, 0.009]  # rad/frame at 20Hz


class HumanTurnDetector:
  """Detects sustained driver steering that should pause lateral control."""

  PRESS_THRESHOLD_S = 0.3
  ANGLE_THRESHOLD_DEG = 45.0
  STEER_DT = CarControllerParams.STEER_STEP * DT_CTRL

  def __init__(self):
    self.press_timer = 0.0
    self.active = False

  def update(self, lat_active: bool, steering_pressed: bool, steering_angle_deg: float) -> bool:
    """Returns True if human-turn override should be active."""
    if not lat_active:
      self.reset()
      return False

    if steering_pressed and abs(steering_angle_deg) > self.ANGLE_THRESHOLD_DEG:
      self.press_timer += self.STEER_DT
      if self.press_timer >= self.PRESS_THRESHOLD_S:
        self.active = True
    else:
      self.reset()

    return self.active

  def reset(self):
    self.press_timer = 0.0
    self.active = False


class LateralAngle:
  """Angle-mode lateral control strategy for Ford vehicles."""

  def __init__(self, CP):
    self.CP = CP
    # Private model: fixed initialization geometry, never learned ratio/stiffness, offset or roll.
    self.pinion_model = VehicleModel(CP) if CP.flags & FordFlags.PINION_CURVATURE else None
    if self.pinion_model is not None:
      carlog.info("Ford pinion curvature enabled: wheelbase=%s steerRatio=%s mass=%s centerToFront=%s " +
                  "tireStiffnessFront=%s tireStiffnessRear=%s flags=%s safetyParam=%s",
                  CP.wheelbase, CP.steerRatio, CP.mass, CP.centerToFront, CP.tireStiffnessFront,
                  CP.tireStiffnessRear, CP.flags, CP.safetyConfigs[-1].safetyParam)
    low_gain, high_gain = _get_platform_gains(CP.carFingerprint)
    self.gain_low_curv = low_gain
    self.gain_high_curv = high_gain

    self.path_angle_last = 0.0
    self.kappa_cmd = 0.0

    self.human_turn_detector = HumanTurnDetector()
    self.human_turn_active = False

  def get_current_curvature(self, CS) -> float:
    """Returns controller-sign curvature; preserves the raw CarState yaw rate."""
    if self.pinion_model is not None:
      return -self.pinion_model.calc_curvature(math.radians(CS.out.steeringAngleDeg), CS.out.vEgoRaw, 0.)
    return -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)

  def update(self, CC, CS, actuators) -> AngleLateralResult:
    """
    Compute path_angle from planner curvature.

    In angle mode:
      - path_angle = kappa_cmd * v_ego * curvature_factor
      - c0 (path_offset) = 0 (handled by caller)
      - c2 (curvature) = 0 (handled by caller)
      - c3 (curvature_rate) = 0 (handled by caller)

    Returns AngleLateralResult with path_angle and shadow_curvature for safety.
    """
    v_ego = float(CS.out.vEgoRaw)
    current_curvature = self.get_current_curvature(CS)

    # Inactive: zero everything
    if not CC.latActive:
      self.path_angle_last = 0.0
      self.kappa_cmd = current_curvature
      self.human_turn_detector.reset()
      self.human_turn_active = False
      return AngleLateralResult(path_angle=0.0, shadow_curvature=current_curvature)

    # Human-turn override: force mode 0 while driver manually steers
    self.human_turn_active = self.human_turn_detector.update(
      CC.latActive, CS.out.steeringPressed, CS.out.steeringAngleDeg)
    if self.human_turn_active:
      self.path_angle_last = 0.0
      self.kappa_cmd = current_curvature
      return AngleLateralResult(path_angle=0.0, shadow_curvature=current_curvature)

    # Start from planner curvature
    kappa_cmd = float(actuators.curvature)

    # Deviation clip: kappa_cmd must stay within CURVATURE_ERROR of measured (v > 9 m/s)
    if v_ego > 9:
      kappa_cmd = float(clip(kappa_cmd,
                             current_curvature - CarControllerParams.CURVATURE_ERROR,
                             current_curvature + CarControllerParams.CURVATURE_ERROR))

    # Speed-interpolated gain: 1.0 at low speed, platform-specific at high speed
    # Low-curvature arm: unchanged (dampening=1.0 implicit)
    low_gain_interp = float(interp(v_ego, [13.5, 26.82], [1.0, self.gain_low_curv]))
    # High-curvature arm: apply BP speed factors (low_speed_curv_factor at 13.5, high_speed_curv_factor at 26.82)
    high_gain_interp = float(interp(v_ego, [13.5, 26.82],
                                    [1.30 * _FORD_LOW_SPEED_FACTOR_ANG,
                                     self.gain_high_curv * _FORD_HIGH_SPEED_FACTOR_ANG]))

    # Curvature magnitude selects between low-curv and high-curv gain
    curvature_factor = float(interp(abs(kappa_cmd), [0.0007, 0.001], [low_gain_interp, high_gain_interp]))

    # Core angle-mode formula
    path_angle = kappa_cmd * v_ego * curvature_factor

    # Clamp to DBC signal range
    path_angle = float(clip(path_angle, FORD_DBC_PATH_ANGLE_MIN, FORD_DBC_PATH_ANGLE_MAX))

    # Soft ROC limit
    soft_roc = float(interp(v_ego, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES))
    path_angle = float(clip(path_angle, self.path_angle_last - soft_roc, self.path_angle_last + soft_roc))

    self.path_angle_last = path_angle
    # Shadow curvature: for driver pressing, use measured; otherwise use clipped planner kappa
    self.kappa_cmd = current_curvature if CS.out.steeringPressed else kappa_cmd

    return AngleLateralResult(path_angle=path_angle, shadow_curvature=self.kappa_cmd)
