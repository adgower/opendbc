"""Offline Navigator tracking sensitivity prototype; not installed or selected.

The update sequence is copied from the pinned installed strategy so the new term
is before its unchanged range/rate bounds. No learned/integral state is used.
"""
import math
from numpy import clip, interp
from opendbc.car.ford.lateral_angle import (
  AngleLateralResult, FORD_DBC_PATH_ANGLE_MIN, FORD_DBC_PATH_ANGLE_MAX,
  _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES,
)
from opendbc.car.ford.lateral_angle_low_speed_prototype import smoothstep
from opendbc.car.ford.lateral_angle_very_low_speed_prototype import VeryLowSpeedGainPrototype
from opendbc.car.ford.values import CAR, CarControllerParams, FordFlags


class BoundedTrackingPrototype(VeryLowSpeedGainPrototype):
  KP = .5
  FILTER_TAU_S = .30
  MAX_EXTRA_PATH_RAD = .015
  MAX_RELATIVE_EXTRA = .25
  STEER_DT_S = .05

  def __init__(self, CP, *, enabled=False):
    # The current installed extra-low-speed feedforward remains the baseline.
    super().__init__(CP, enabled=True)
    self.tracking_enabled = bool(enabled)
    self.filtered_error = 0.
    self.tracking_correction = 0.

  @staticmethod
  def speed_weight(v):
    if not math.isfinite(v) or v < 5*.44704 or v >= 25*.44704:
      return 0.
    return 1. - smoothstep((v-15*.44704)/(10*.44704))

  def reset_tracking(self):
    self.filtered_error = 0.
    self.tracking_correction = 0.

  def correction(self, v, requested, measured, feedforward, driver_pressed):
    weight = self.speed_weight(v)
    if (not self.tracking_enabled or not weight or driver_pressed
        or self.CP.carFingerprint != CAR.FORD_EXPEDITION_MK4
        or not self.CP.flags & FordFlags.PINION_CURVATURE
        or not self.CP.flags & FordFlags.LOW_SPEED_GAIN
        or not all(math.isfinite(x) for x in (requested, measured, feedforward))):
      self.reset_tracking()
      return 0.
    alpha = self.STEER_DT_S/(self.FILTER_TAU_S+self.STEER_DT_S)
    self.filtered_error += alpha*((requested-measured)-self.filtered_error)
    raw = self.KP*self.filtered_error*v*self.get_curvature_factor(v,requested)
    cap = min(self.MAX_EXTRA_PATH_RAD,self.MAX_RELATIVE_EXTRA*abs(feedforward))
    self.tracking_correction = float(clip(raw,-cap,cap))*weight
    return self.tracking_correction

  def update(self, CC, CS, actuators):
    v_ego = float(CS.out.vEgoRaw)
    current_curvature = self.get_current_curvature(CS)
    if not CC.latActive:
      self.reset_tracking()
      self.path_angle_last = 0.
      self.kappa_cmd = current_curvature
      self.human_turn_detector.reset()
      self.human_turn_active = False
      return AngleLateralResult(path_angle=0.,shadow_curvature=current_curvature)
    self.human_turn_active = self.human_turn_detector.update(
      CC.latActive,CS.out.steeringPressed,CS.out.steeringAngleDeg)
    if self.human_turn_active:
      self.reset_tracking()
      self.path_angle_last = 0.
      self.kappa_cmd = current_curvature
      return AngleLateralResult(path_angle=0.,shadow_curvature=current_curvature)
    kappa_cmd = float(actuators.curvature)
    if v_ego > 9:
      kappa_cmd = float(clip(kappa_cmd,
        current_curvature-CarControllerParams.CURVATURE_ERROR,
        current_curvature+CarControllerParams.CURVATURE_ERROR))
    path_angle = kappa_cmd*v_ego*self.get_curvature_factor(v_ego,kappa_cmd)
    path_angle += self.correction(v_ego,kappa_cmd,current_curvature,path_angle,CS.out.steeringPressed)
    path_angle = float(clip(path_angle,FORD_DBC_PATH_ANGLE_MIN,FORD_DBC_PATH_ANGLE_MAX))
    soft_roc = float(interp(v_ego,_SOFT_ROC_SPEEDS,_SOFT_ROC_VALUES))
    path_angle = float(clip(path_angle,self.path_angle_last-soft_roc,self.path_angle_last+soft_roc))
    self.path_angle_last = path_angle
    self.kappa_cmd = current_curvature if CS.out.steeringPressed else kappa_cmd
    return AngleLateralResult(path_angle=path_angle,shadow_curvature=self.kappa_cmd)
