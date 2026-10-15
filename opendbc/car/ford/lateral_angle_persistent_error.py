"""Offline-only persistent deficit admission; no installed strategy selection."""
import math
from collections import deque
from opendbc.car.ford.lateral_angle_bounded_tracking import BoundedTrackingPrototype
from opendbc.car.ford.values import CAR, FordFlags


class PersistentErrorPrototype(BoundedTrackingPrototype):
  WINDOW_FRAMES = 11  # Ten 50 ms intervals.
  HOLD_FRAMES = 6

  def __init__(self, CP, *, enabled=False):
    super().__init__(CP, enabled=enabled)
    self.history = deque(maxlen=self.WINDOW_FRAMES)
    self.hold_frames = 0
    self.feedback_valid = True

  def reset_tracking(self):
    super().reset_tracking()
    self.history.clear()
    self.hold_frames = 0

  def update(self, CC, CS, actuators):
    self.feedback_valid = bool(getattr(CS.out, 'canValid', True)) and not (
      getattr(CS.out, 'steerFaultTemporary', False) or getattr(CS.out, 'steerFaultPermanent', False))
    return super().update(CC, CS, actuators)

  def correction(self, v, requested, measured, feedforward, driver_pressed):
    eligible = (self.tracking_enabled and self.feedback_valid and self.speed_weight(v)>0
      and not driver_pressed and self.CP.carFingerprint == CAR.FORD_EXPEDITION_MK4
      and self.CP.flags & FordFlags.PINION_CURVATURE
      and self.CP.flags & FordFlags.LOW_SPEED_GAIN
      and all(math.isfinite(x) for x in (requested, measured, feedforward))
      and abs(requested)>.001)
    if not eligible:
      self.reset_tracking()
      return 0.
    if self.history and requested*self.history[-1][1]<=0:
      self.reset_tracking()
    self.history.append((v,requested))
    stable = len(self.history)==self.WINDOW_FRAMES
    if stable:
      vs,ks=zip(*self.history)
      ordered=sorted(abs(k) for k in ks)
      stable = (all(k*requested>0 for k in ks) and max(vs)-min(vs)<.5
                and max(ks)-min(ks)<=.25*ordered[len(ordered)//2])
    deficit = requested*measured>=0 and abs(measured)<.7*abs(requested)
    self.hold_frames = self.hold_frames+1 if stable and deficit else 0
    if self.hold_frames<self.HOLD_FRAMES:
      # Clear correction state without destroying the admission window.
      super().reset_tracking()
      return 0.
    return super().correction(v,requested,measured,feedforward,driver_pressed)
