"""
Ford CAN-FD path-angle-primary lateral control.

Converts planner curvature (kappa) into path_angle (c1) for the PSCM.
The steering intent is c1; c0 (path_offset), c2 (curvature), and c3 (curvature_rate)
are forced to zero in angle mode.

Live map: path_angle = kappa_cmd * v_ego * curvature_factor

Gain schedules are platform-specific. Expedition/Navigator uses ~1.425 (1.5x BOF baseline)
based on real-world tuning data from BluePilot bp-dev-expedition.

Human-turn mode-0: when the driver manually turns (steeringPressed + large wheel angle),
lateral control is forced inactive (mode 0) so the PSCM releases cleanly.
"""
from dataclasses import dataclass
from numpy import clip, interp

from opendbc.car import DT_CTRL
from opendbc.car.ford.values import CAR, CarControllerParams


@dataclass
class AngleLateralResult:
  """Result from angle-mode lateral strategy, used by CarController."""
  path_angle: float = 0.0
  shadow_curvature: float = 0.0


# DBC path_angle signal limits (rad). Safety uses the same in ford.h.
FORD_DBC_PATH_ANGLE_MIN = -0.5
FORD_DBC_PATH_ANGLE_MAX = 0.5235


# Per-platform gain defaults.
# CAN vehicles (Escape MK4, Bronco Sport, Explorer, Maverick, Edge, Focus)
_GAIN_CAN = (1.00, 1.15)
# CAN-FD body-on-frame trucks (F-150, Lightning, Ranger)
_GAIN_CANFD_BOF = (0.95, 0.95)
# CAN-FD unibody SUVs (Mustang Mach-E, Escape MK4.5)
_GAIN_CANFD_SUV = (1.00, 1.05)
# Expedition MK4: 1.5x BOF CANFD values based on real-world tuning (bp-dev-expedition)
_GAIN_EXPEDITION = (_GAIN_CANFD_BOF[0] * 1.5, _GAIN_CANFD_BOF[1] * 1.5)  # ~1.425, ~1.425

_CANFD_BOF_CARS = frozenset({
  CAR.FORD_F_150_MK14,
  CAR.FORD_F_150_LIGHTNING_MK1,
  CAR.FORD_RANGER_MK2,
})
_CANFD_SUV_CARS = frozenset({
  CAR.FORD_MUSTANG_MACH_E_MK1,
  CAR.FORD_ESCAPE_MK4_5,
})
_EXPEDITION_CARS = frozenset({
  CAR.FORD_EXPEDITION_MK4,
})


def _get_platform_gains(car_fingerprint: str) -> tuple[float, float]:
  """Returns (low_curvature_gain, high_curvature_gain) for the platform."""
  if car_fingerprint in _EXPEDITION_CARS:
    return _GAIN_EXPEDITION
  elif car_fingerprint in _CANFD_BOF_CARS:
    return _GAIN_CANFD_BOF
  elif car_fingerprint in _CANFD_SUV_CARS:
    return _GAIN_CANFD_SUV
  else:
    return _GAIN_CAN


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
    low_gain, high_gain = _get_platform_gains(CP.carFingerprint)
    self.gain_low_curv = low_gain
    self.gain_high_curv = high_gain

    self.path_angle_last = 0.0
    self.kappa_cmd = 0.0

    self.human_turn_detector = HumanTurnDetector()
    self.human_turn_active = False

  def get_current_curvature(self, CS) -> float:
    """Returns measured curvature from yaw rate and vehicle speed."""
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
    low_gain_interp = float(interp(v_ego, [13.5, 26.82], [1.0, self.gain_low_curv]))
    high_gain_interp = float(interp(v_ego, [13.5, 26.82], [1.30, self.gain_high_curv]))

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
