"""Bounded temporal admission for the existing opt-in Navigator tracking trial."""

import math
from collections import deque
from opendbc.car.vehicle_model import VehicleModel
from opendbc.car.ford.lateral_angle_roll_mapping import RollAwareMapping
from opendbc.car.ford.lateral_angle_observer_continuity import ObserverContinuity as Installed
from opendbc.car.ford.lateral_angle_low_speed_prototype import smoothstep


class AdmissionObserver:
  def __init__(self):
    self.rows = deque(maxlen=5)
    self.generation = None
    self.reason = "initial"
    self.model = None

  def clear(self, reason):
    self.rows.clear()
    self.reason = reason

  def feed(self, owner, CC, CS, a):
    ctx = owner.reference.context
    cal = owner.reference.calibration
    mo = owner.motion
    s = CS.out
    if (
      not owner.enabled
      or not owner.input_state_valid
      or ctx is None
      or cal is None
      or mo is None
      or not CC.latActive
      or any(
        bool(getattr(s, k, False)) for k in ["steeringPressed", "brakePressed", "leftBlinker", "rightBlinker", "steerFaultTemporary", "steerFaultPermanent"]
      )
      or not s.canValid
    ):
      self.clear("invalid_state")
      return None
    now, st, ct, mt, lane, mvalid = ctx
    cn, source, good, roll, ratio, stiffness, offset = cal
    mn, ms, mr, mok, yaw, std = mo
    k = float(a.curvature)
    v = float(s.vEgoRaw)
    mv = float(owner.target_speed if owner.target_speed is not None else s.vEgo)
    if not (
      mvalid
      and not lane
      and good
      and mok
      and cn == mn == now
      and 0 < ms <= mr <= now
      and now - ms <= 100_000_000
      and now - mr <= 100_000_000
      and all(0 < t <= now and now - t <= lim for t, lim in [(st, 40_000_000), (ct, 40_000_000), (mt, 150_000_000), (source, 150_000_000)])
      and all(math.isfinite(x) for x in [k, v, mv, roll, ratio, stiffness, offset, yaw, std, a.steeringAngleDeg, s.steeringAngleDeg])
      and v > 0.1
      and mv > 0.1
      and 0 <= std <= 0.03
      and abs(roll) <= 0.15
      and 10 <= ratio <= 35
      and 0.2 <= stiffness <= 3
      and abs(offset) <= 10
    ):
      self.clear("invalid_provenance")
      return None

    if self.model is None:
      self.model = VehicleModel(owner.CP)
    vm = self.model
    vm.update_params(stiffness, ratio)
    expected = math.degrees(vm.get_steer_from_curvature(-k, mv, roll)) + offset
    target = float(a.steeringAngleDeg) - offset
    actual = float(s.steeringAngleDeg) - offset
    if abs(expected - float(a.steeringAngleDeg)) > 0.5 or abs(target) < 1.5 or actual * target <= 0 or abs(k) * v * v < 0.1:
      self.clear("target_or_direction")
      return None
    axis = 1 if target > 0 else -1
    row = (now, axis * (actual - target), k, st, ct, ms, actual / target, yaw / (k * v), std, yaw, v)
    generation = owner._motion_generation
    if self.generation != generation:
      self.rows.clear()
      self.generation = generation
    if self.rows:
      old = self.rows[-1]
      if k * old[2] <= 0 or any(row[j] <= old[j] for j in [0, 3, 4, 5]) or not 20_000_000 <= now - old[0] <= 100_000_000:
        self.clear("reversal_duplicate_or_cadence")
        return None
    self.rows.append(row)
    self.reason = "observing"
    if len(self.rows) == 5 and all(160_000_000 <= row[j] - self.rows[0][j] <= 300_000_000 for j in [0, 3, 4, 5]):
      self.reason = "ready"
    return row


class MidrangeMapping(RollAwareMapping):
  def __init__(self, CP, *, baseline):
    super().__init__(CP, enabled=True)
    # Preserve the explicitly constructed wrapper's specialized baseline and its
    # continuity fields. This seam is used only during initial construction.
    self.baseline = baseline
    self._original_mapping = False

  def mapping_weight(self, v):
    if self._original_mapping:
      return RollAwareMapping.mapping_weight(v)
    mph = v / 0.44704
    return max(RollAwareMapping.mapping_weight(v), smoothstep((mph - 35) / 5) * (1 - smoothstep((mph - 50) / 5)))

  def _prepare_map(self, CC, CS, actuators):
    previous = (self.map_delta, self._last_map_ns, self._previous_request)
    super()._prepare_map(CC, CS, actuators)
    mph = float(CS.out.vEgoRaw) / 0.44704
    if not 35 < mph < 55:
      return
    direction = -1 if self.map_target * self.map_base < 0 else 1
    if getattr(self, "admission_mode", 0) == direction:
      return
    # Re-evaluate the exact original policy from its pre-step state. Scaling an
    # already slew-limited full-extension state would distort original history.
    self.map_delta, self._last_map_ns, self._previous_request = previous
    self._original_mapping = True
    try:
      super()._prepare_map(CC, CS, actuators)
    finally:
      self._original_mapping = False


class MidrangeAdmission(Installed):
  def __init__(self, CP, *, enabled=False):
    super().__init__(CP, enabled=enabled)
    self.admission_observer = AdmissionObserver()
    self.admission_rows = deque(maxlen=32)
    self.admission_mode = 0
    self.admission_leaves = []
    if enabled:
      # The installed graph has explicit highway wrapper pairs. Keep every
      # wrapper and specialized low-speed observer; construct only mapping leaves.
      owners = (
        self,
        self.installed_baseline,
        self.upper_reference,
        self.upper_reference.installed_baseline,
        self.stock,
        self.stock.installed_baseline,
        self.stock.upper_reference,
        self.stock.upper_reference.installed_baseline,
        self.unchanged,
        self.unchanged.installed_baseline,
        self.unchanged.upper_reference,
        self.unchanged.upper_reference.installed_baseline,
        self.unchanged.stock,
        self.unchanged.stock.installed_baseline,
        self.unchanged.stock.upper_reference,
        self.unchanged.stock.upper_reference.installed_baseline,
      )
      for owner in owners:
        owner.reference = MidrangeMapping(CP, baseline=owner.reference.baseline)
        self.admission_leaves.append(owner.reference)

  def update(self, CC, CS, a):
    mph = float(CS.out.vEgoRaw) / 0.44704
    row = self.admission_observer.feed(self, CC, CS, a) if 35 < mph < 55 else None
    if row is None:
      self.admission_rows.clear()
      self.admission_mode = 0
    else:
      k = float(a.curvature)
      v = float(CS.out.vEgoRaw)  # noqa: F841 - retain verified admission statement
      cal = self.reference.calibration
      wheel = (
        -self.admission_observer.model.calc_curvature(
          math.radians(float(CS.out.steeringAngleDeg) - cal[-1]), float(self.target_speed if self.target_speed is not None else CS.out.vEgo), cal[3]
        )
        / k
      )
      body = row[7]
      value = (row[0], k, body, wheel)
      if self.admission_rows and (
        not 20_000_000 <= row[0] - self.admission_rows[-1][0] <= 100_000_000
        or k * self.admission_rows[-1][1] <= 0
        or abs(k / self.admission_rows[-1][1] - 1) > 0.15
      ):
        self.admission_rows.clear()
        self.admission_mode = 0
      self.admission_rows.append(value)
      while len(self.admission_rows) > 2 and row[0] - self.admission_rows[1][0] >= 1_000_000_000:
        self.admission_rows.popleft()
      rows = list(self.admission_rows)
      coherent = all(abs(x[2] - x[3]) <= 0.15 and 0.5 <= x[2] <= 1.8 and 0.5 <= x[3] <= 1.8 for x in rows)
      stable = max(abs(x[1]) for x in rows) <= 1.15 * min(abs(x[1]) for x in rows)
      ready = len(rows) >= 12 and rows[-1][0] - rows[0][0] >= 950_000_000
      if not coherent:
        self.admission_mode = 0
      elif self.admission_mode == -1 and min(body, wheel) < 1.03:
        self.admission_mode = 0
      elif self.admission_mode == 1 and max(body, wheel) > 0.97:
        self.admission_mode = 0
      elif ready and stable:
        if all(min(x[2], x[3]) > 1.08 for x in rows):
          self.admission_mode = -1
        elif all(max(x[2], x[3]) < 0.92 for x in rows):
          self.admission_mode = 1
        elif self.admission_mode == -1 and min(body, wheel) < 1.03:
          self.admission_mode = 0
        elif self.admission_mode == 1 and max(body, wheel) > 0.97:
          self.admission_mode = 0
    for leaf in self.admission_leaves:
      leaf.admission_mode = self.admission_mode
    return super().update(CC, CS, a)
