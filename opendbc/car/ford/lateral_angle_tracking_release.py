"""Local-only release-aware correction. Requires explicit fresh context."""
import math
from collections import deque
from opendbc.car.ford.lateral_angle_persistent_error import PersistentErrorPrototype


class ReleaseTrackingPrototype(PersistentErrorPrototype):
  HOLD_FRAMES = 6
  KP = .25
  MAX_EXTRA_PATH_RAD = .0075
  MAX_RELATIVE_EXTRA = .125
  PREDICTION_S = 1.

  def __init__(self, CP, *, enabled=False):
    super().__init__(CP,enabled=enabled)
    self.context = None
    self.last_time_ns = None
    self.response_history = deque(maxlen=self.WINDOW_FRAMES)
    self.context_ok = False

  def set_context(self, *, now_ns, state_ns, control_ns, model_ns, lane_change, model_valid):
    self.context = (now_ns,state_ns,control_ns,model_ns,lane_change,model_valid)

  def reset_tracking(self):
    super().reset_tracking()
    self.response_history.clear()

  def correction(self,v,requested,measured,feedforward,driver_pressed):
    ctx=self.context
    self.context = None  # Each evaluated steering frame requires new context.
    self.context_ok=False
    if ctx is not None:
      now,state,control,model,lane_change,model_valid=ctx
      ages=((now-state)/1e9,(now-control)/1e9,(now-model)/1e9)
      cadence_ok=self.last_time_ns is None or 40_000_000 <= now-self.last_time_ns <= 60_000_000
      self.last_time_ns=now
      self.context_ok=(cadence_ok and model_valid and not lane_change
        and all(math.isfinite(a) and 0<=a<=limit for a,limit in zip(ages,(.08,.10,.15))))
    if not self.context_ok:
      self.reset_tracking()
      return 0.
    if not math.isfinite(measured) or not math.isfinite(requested):
      self.reset_tracking()
      return 0.
    if self.history and requested*self.history[-1][1]<=0:
      self.reset_tracking()
    # A silent actuator can be delayed or unavailable. Never boost before an
    # observed same-direction response; this prototype does not rescue stalls.
    if requested*measured<=0 or abs(measured)<.1*abs(requested):
      self.reset_tracking()
      return 0.
    self.response_history.append((now,measured))
    # Positive response slope means the car is already catching up. Do not
    # add effort that will remain in the plant after it reaches the request.
    if len(self.response_history)==self.WINDOW_FRAMES:
      sign=1 if requested>=0 else -1
      slope=max(0.,sign*(self.response_history[-1][1]-self.response_history[0][1])/
                ((self.response_history[-1][0]-self.response_history[0][0])/1e9))
      projected=sign*measured+self.PREDICTION_S*slope
      if projected>=.7*abs(requested):
        # Keep response samples to prevent immediate re-admission, but reset
        # demand persistence/filter. Next admissible sample restarts the hold.
        samples=list(self.response_history)
        self.reset_tracking()
        self.response_history.extend(samples)
        return 0.
    return super().correction(v,requested,measured,feedforward,driver_pressed)
