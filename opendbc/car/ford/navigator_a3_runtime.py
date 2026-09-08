"""Startup-configured proposal observer. Requested actuation is always blocked."""
from dataclasses import asdict, dataclass
import math
import os

from opendbc.car.ford import fordcan
from opendbc.car.ford.navigator_a3 import Inputs, PROFILES
from opendbc.car.ford.navigator_a3_scheduler import ShadowScheduler
from opendbc.car.ford.navigator_a3_command import select_command
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
    self.scheduler = ShadowScheduler()
    self.fault_reason = 'physical_enforcement_unvalidated' if self.config.mode == 'requested' else None
    self.source_ns = 0
    self.source_valid = False
    self.measurement_ns = None
    self.measurement_valid = False
    self.evidence_fault_reason = None
    self.calculation_fault_reason = None
    self.diagnostic = {}
    self.proposal_frame = None
    self.selection = select_command(self.config.mode, None, None)

  def set_evidence(self, source_ns: int, source_valid: bool, measurement_ns: int | None,
                   measurement_valid: bool, fault_reason: str | None, calculation_fault_reason: str | None = None):
    self.source_ns = source_ns
    self.source_valid = source_valid
    self.measurement_ns = measurement_ns
    self.measurement_valid = measurement_valid
    self.evidence_fault_reason = fault_reason
    self.calculation_fault_reason = calculation_fault_reason or (
      None if self.config.mode == 'shadow' and fault_reason == 'direct_steering_rejection' else fault_reason)

  def observe(self, cc, cs, now_ns, packer, can_bus, counter):
    self.proposal_frame = None
    if self.config.mode == 'a2':
      return
    # scheduled_update bypasses legacy strategy cadence; ShadowScheduler owns
    # proposal eligibility while observe is checked at the 100 Hz control rate.
    sample = Inputs(now_ns, self.source_ns, cc.actuators.curvature, cs.out.vEgoRaw,
                    cc.latActive, cs.out.steeringPressed, self.source_valid and self.calculation_fault_reason is None,
                    -cs.out.yawRate / max(cs.out.vEgoRaw, .1), self.measurement_ns, self.measurement_valid, scheduled_update=True)
    previous_ns = self.scheduler.last_call_ns
    decision = self.scheduler.step(PROFILES[self.config.profile], sample)
    result = decision.output
    proposed = None if result is None else fordcan.create_lat_ctl2_msg(
      packer, can_bus, result.mode, 0., -result.path_angle_rad, 0., 0., decision.proposal_counter)
    self.proposal_frame = proposed
    self.diagnostic = {'schema_version': 3, 'calculation_eligible': sample.valid,
                       'calculation_fault_reason': self.calculation_fault_reason,
                       'timing': {'scheduled_update': True, 'scheduler_checked_at_control_rate': True, 'previous_ns': previous_ns,
                                  'elapsed_ns': None if previous_ns is None else now_ns - previous_ns},
                       'scheduler': {**asdict(decision), 'output': None},
                       'config': asdict(self.config), 'input': asdict(sample),
                       'output': None if result is None else asdict(result),
                       'fault_reason': self.fault_reason, 'evidence_fault_reason': self.evidence_fault_reason,
                       'proposed_frame': None if proposed is None else self.frame_record(proposed), 'actual_frame': None}
    self.diagnostic['scheduler'].pop('output')
    self.diagnostic['scheduler']['new_proposal'] = result is not None

  @property
  def state(self):
    return self.scheduler.state

  @staticmethod
  def frame_record(frame):
    address, data, bus = frame
    return {'address': address, 'data': data.hex(), 'bus': bus}
