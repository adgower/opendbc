"""Startup-configured proposal observer. Requested actuation is always blocked."""
from dataclasses import asdict, dataclass
import math
import os

from opendbc.car.ford import fordcan
from opendbc.car.ford.navigator_a3 import Inputs, PROFILES, State, update
from opendbc.car.ford.values import CAR, FordFlags


@dataclass(frozen=True)
class Config:
  mode: str
  profile: str | None

  @classmethod
  def from_startup(cls, cp):
    mode = os.environ.get('NAVIGATOR_A3_MODE', 'a2')
    profile = os.environ.get('NAVIGATOR_A3_PROFILE')
    if mode not in ('a2', 'shadow', 'requested'):
      raise ValueError(f'Unknown NAVIGATOR_A3_MODE: {mode!r}')
    if profile is not None and profile not in PROFILES:
      raise ValueError(f'Unknown NAVIGATOR_A3_PROFILE: {profile!r}')
    if mode != 'a2':
      if profile is None:
        raise ValueError('NAVIGATOR_A3_PROFILE must be explicit for shadow/requested')
      if (cp.brand != 'ford' or cp.carFingerprint != CAR.FORD_EXPEDITION_MK4 or not cp.flags & FordFlags.CANFD
          or not math.isclose(cp.wheelbase, 3.1115, rel_tol=0., abs_tol=1e-6)):
        raise ValueError('Navigator A3 requires Ford Expedition CAN FD with physical A2 wheelbase 3.1115 m')
    return cls(mode, profile)


class Runtime:
  def __init__(self, cp):
    self.config = Config.from_startup(cp)
    self.state = State()
    self.fault_reason = 'physical_enforcement_unvalidated' if self.config.mode == 'requested' else None
    self.source_ns = 0
    self.source_valid = False
    self.measurement_ns = None
    self.measurement_valid = False
    self.evidence_fault_reason = None
    self.diagnostic = {}

  def set_evidence(self, source_ns: int, source_valid: bool, measurement_ns: int | None,
                   measurement_valid: bool, fault_reason: str | None):
    self.source_ns = source_ns
    self.source_valid = source_valid
    self.measurement_ns = measurement_ns
    self.measurement_valid = measurement_valid
    self.evidence_fault_reason = fault_reason

  def observe(self, cc, cs, now_ns, packer, can_bus, counter):
    if self.config.mode == 'a2':
      return
    sample = Inputs(now_ns, self.source_ns, cc.actuators.curvature, cs.out.vEgoRaw,
                    cc.latActive, cs.out.steeringPressed, self.source_valid and self.evidence_fault_reason is None,
                    -cs.out.yawRate / max(cs.out.vEgoRaw, .1), self.measurement_ns, self.measurement_valid)
    result = update(PROFILES[self.config.profile], self.state, sample)
    self.state = result.state
    proposed = fordcan.create_lat_ctl2_msg(packer, can_bus, result.mode, 0., -result.path_angle_rad, 0., 0., counter)
    self.diagnostic = {'config': asdict(self.config), 'input': asdict(sample), 'output': asdict(result),
                       'fault_reason': self.fault_reason, 'evidence_fault_reason': self.evidence_fault_reason,
                       'proposed_frame': self.frame_record(proposed), 'actual_frame': None}

  @staticmethod
  def frame_record(frame):
    address, data, bus = frame
    return {'address': address, 'data': data.hex(), 'bus': bus}
