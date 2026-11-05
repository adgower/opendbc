"""Local nominal longitudinal reference, extracted from the original ACC callback.

No transport, controller shell or actuator-output construction. Each update is one
50 Hz ACC callback; callers own scheduling and refresh coordination context.
"""
import math
from dataclasses import dataclass
import numpy as np
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, DT_CTRL, structs
from opendbc.car.ford.longitudinal_handoff import Handoff, Result as HandoffResult
from opendbc.car.ford.longitudinal_release import ReleaseGate
from opendbc.car.ford.longitudinal_coast import BoundedCoastAdmission
from opendbc.car.ford.longitudinal_restart import CoastClearance, CoastLease
from opendbc.car.ford.longitudinal_coordination import GradeLease, GradeResult, Feedback, coordination_available
from opendbc.car.ford.longitudinal_precharge import SignedCoastAllocator, PRECHARGE_PROTOCOL_VERSION, allocator_request_valid
from opendbc.car.ford.values import CarControllerParams, CAR

LongCtrlState = structs.CarControl.Actuators.LongControlState

@dataclass(frozen=True)
class NominalResult:
  feedback: Feedback
  handoff: HandoffResult
  grade: GradeResult
  accel: float
  gas: float

def apply_creep_compensation(accel: float, v_ego: float) -> float:
  if v_ego >= 3.0 and math.isfinite(accel):
    return float(accel)
  creep_accel = np.interp(v_ego, [1., 3.], [0.6, 0.])
  creep_accel = np.interp(accel, [0., 0.2], [creep_accel, 0.])
  accel -= creep_accel
  return float(accel)


class NominalReference:
  def __init__(self, CP, *, history_enabled=False, retention_enabled=False):
    if not coordination_available(CP):
      raise ValueError("nominal reference supports Navigator Alpha longitudinal control only")
    if retention_enabled:
      raise NotImplementedError("retention-enabled nominal reference is not supported")
    self.CP = CP
    self.accel = 0.0
    self.gas = 0.0
    self.brake_request = False
    self.reference_brake = False
    self.precharge_request = False
    self.precharge_allocator = SignedCoastAllocator()
    self.last_allocation = None
    self.long_handoff = Handoff(0.5, 0.02)
    self.longitudinal_coordination_enabled = True  # helper always evaluates coordination
    self.history_release_enabled = bool(history_enabled)
    self.grade_map = GradeLease(True)
    self.release_barrier = CoastClearance(True)
    self.brake_admission = BoundedCoastAdmission()
    self.last_admission = None
    self.last_grade = None
    self.last_observer = None
    self.last_feedback = None
    self.coordination_fault = None
    self.coordination_epoch = None
    self.coordination_producer_ns = 0
    self.coordination_sequence = 0
    self.coordination_control_valid = False
    self.release_gate = ReleaseGate()
    self.retention_owner = None
    self.last_release = None
    self.last_handoff = None
    self.coordination_state_ns = 0
    self.coordination_speed_ns = 0
    self.coordination_state_valid = False
    if self.history_release_enabled:
      from opendbc.car.ford.longitudinal_progress_historical import HistoricalProgress
      self.release_barrier = CoastClearance(True, progress_factory=HistoricalProgress)

  def update(self, CC, CS, now_nanos) -> NominalResult:
    """Consume exactly one ACC callback and return its raw nominal facts."""
    actuators = CC.actuators
    long_active = CC.longActive
    long_state = CC.actuators.longControlState
    phase = str(long_state)
    navigator_longitudinal = self.CP.carFingerprint == CAR.FORD_EXPEDITION_MK4
    accel_valid = math.isfinite(actuators.accel)
    accel = (self.accel if long_active else 0.0) if navigator_longitudinal and not accel_valid else actuators.accel
    gas = CarControllerParams.INACTIVE_GAS if navigator_longitudinal and not accel_valid else accel

    creep_accel = None
    if long_active and (not navigator_longitudinal or accel_valid):
      # Compensate for engine creep at low speed.
      # Either the ABS does not account for engine creep, or the correction is very slow
      # TODO: verify this applies to EV/hybrid
      accel = apply_creep_compensation(accel, CS.out.vEgo)
      creep_accel = accel

      # The stock system has been seen rate limiting the brake accel to 5 m/s^3,
      # however even 3.5 m/s^3 causes some overshoot with a step response.
      accel = max(accel, self.accel - (3.5 * CarControllerParams.ACC_CONTROL_STEP * DT_CTRL))

    accel = float(
      CarControllerParams.ACCEL_MIN if accel < CarControllerParams.ACCEL_MIN
      else CarControllerParams.ACCEL_MAX if accel > CarControllerParams.ACCEL_MAX else accel
    )
    gas = float(
      CarControllerParams.ACCEL_MIN if gas < CarControllerParams.ACCEL_MIN
      else CarControllerParams.ACCEL_MAX if gas > CarControllerParams.ACCEL_MAX else gas
    )

    # Both gas and accel are in m/s^2, accel is used solely for braking
    if not long_active or gas < CarControllerParams.MIN_GAS:
      gas = CarControllerParams.INACTIVE_GAS

    # PCM applies pitch compensation to gas/accel, but we need to compensate for the brake/pre-charge bits
    accel_due_to_pitch = 0.0
    pitch_valid = len(CC.orientationNED) == 3 and all(math.isfinite(x) for x in CC.orientationNED)
    if len(CC.orientationNED) == 3 and (not navigator_longitudinal or pitch_valid):
      accel_due_to_pitch = math.sin(CC.orientationNED[1]) * ACCELERATION_DUE_TO_GRAVITY

    # Scalars from this same callback only; no policy or rate-limit state is shared.
    self.physical_prefix = (accel_valid, creep_accel, pitch_valid, accel_due_to_pitch, phase)

    coordination_now = now_nanos / 1e9
    request = CC.navigatorLongitudinal
    coordinated = navigator_longitudinal and self.longitudinal_coordination_enabled
    if coordinated:
      # Faults survive longActive dropping from our own fault or a pedal
      # override. Only a disabled owner followed by a new engagement rearms.
      if not CC.enabled:
        self.coordination_fault = None
        self.coordination_epoch = None
        self.coordination_producer_ns = 0
        self.release_barrier.reset()
        self.grade_map = GradeLease(True)
      elif self.coordination_epoch is None:
        self.coordination_epoch = request.epoch
        self.coordination_sequence = 0
      elif self.coordination_epoch != request.epoch:
        self.coordination_fault = 'owner_epoch_changed_while_enabled'
      valid_request = (request.enabled and request.epoch > 0 and request.sequence > 0
                       and request.sequence >= self.coordination_sequence
                       and request.producerMonoTime >= self.coordination_producer_ns
                       and 0 <= now_nanos - request.producerMonoTime <= 100_000_000
                       and self.coordination_control_valid)
      if CC.enabled and not valid_request:
        self.coordination_fault = 'invalid_control_context'
      self.coordination_sequence = request.sequence
      self.coordination_producer_ns = request.producerMonoTime
      installed_gravity = accel_due_to_pitch
      self.last_grade = self.grade_map.update(coordination_now, accel_due_to_pitch,
        request.poseMonoTime / 1e9, request.poseValid and pitch_valid,
        long_active, CS.out.vEgo, phase, request.referenceValid)
      accel_due_to_pitch = self.last_grade.gravity
      if actuators.accel < CarControllerParams.MIN_GAS:
        accel_due_to_pitch = min(installed_gravity, accel_due_to_pitch)
      if (long_active and CS.out.vEgo >= 3. and long_state == LongCtrlState.pid
          and (self.last_grade.reason == 'invalid_pose_disengage' or not accel_valid)):
        self.coordination_fault = 'invalid_pose_or_accel'

    brake_request_previous = self.brake_request
    if navigator_longitudinal and coordinated:
      brake_request_previous = self.reference_brake
      self.brake_request = self.reference_brake
    accel_pitch_compensated = accel + accel_due_to_pitch
    if accel_pitch_compensated > 0.3 or not long_active:
      self.brake_request = False
    elif accel_pitch_compensated < 0.0:
      self.brake_request = True

    if navigator_longitudinal:
      self.last_admission = self.brake_admission.update(coordination_now,
        coordinated and not self.coordination_fault and long_active and CS.out.vEgo >= 3.0
        and CS.out.canValid and request.poseValid and phase == 'pid' and pitch_valid and request.referenceValid,
        self.brake_request, brake_request_previous, actuators.accel, accel_pitch_compensated,
        not self.release_barrier.needs, getattr(CS, 'longitudinal_brake_torque', None))
      self.brake_request = self.last_admission.brake
      self.last_release = self.release_gate.update(now_nanos / 1e9, long_active, self.brake_request,
                                                   brake_request_previous, actuators.accel, accel_pitch_compensated,
                                                   phase, pitch_valid)
      self.brake_request = self.last_release.brake
      self.reference_brake = self.brake_request
      if coordinated:
        eligible_allocator = (not self.coordination_fault and long_active and CS.out.vEgo >= 3.
          and CS.out.canValid and request.poseValid and pitch_valid and request.referenceValid
          and phase == 'pid' and allocator_request_valid(request, now_nanos))
        coast_scope = bool(eligible_allocator and CC.enabled and not CS.out.gasPressed and not CS.out.brakePressed
                           and str(CC.hudControl.visualAlert) != 'fcw')
        prior = self.last_feedback
        torque = getattr(CS, 'longitudinal_brake_torque', None)
        continuing = (self.last_allocation is not None and self.last_allocation.phase == 'bounded_true_coast'
                      and prior is not None and prior.valid and not prior.fault and prior.epoch == request.epoch
                      and 0 < prior.sequence <= request.sequence and 0 <= coordination_now - prior.t <= .100000001)
        current_clear = (torque is not None and torque.valid and all(math.isfinite(v) for v in [torque.t, torque.value])
                         and 0 <= coordination_now - torque.t <= .1 and 0 <= torque.value <= 4)
        self.precharge_allocator.coast_allowed = coast_scope
        self.precharge_allocator.continuation_clear = bool(coast_scope and continuing and current_clear)
        self.last_allocation = self.precharge_allocator.update(coordination_now, CC.enabled, long_active,
          eligible_allocator, self.brake_request, float(actuators.accel), accel_pitch_compensated,
          float(request.targetAcceleration),
          measured_clear=(not self.release_barrier.needs and self.last_observer is not None and self.last_observer.allow
            and getattr(CS, 'longitudinal_brake_torque', None) is not None
            and CS.longitudinal_brake_torque.valid
            and all(math.isfinite(v) for v in [CS.longitudinal_brake_torque.t, CS.longitudinal_brake_torque.value])
            and 0 <= coordination_now - CS.longitudinal_brake_torque.t <= .1
            and 0 <= CS.longitudinal_brake_torque.value <= 4.))
        self.brake_request = self.last_allocation.brake
        self.precharge_request = self.last_allocation.precharge
      if coordinated:
        torque = getattr(CS, 'longitudinal_brake_torque', None)
        if torque is not None and not CS.out.canValid:
          torque = None
        self.last_observer = self.release_barrier.update(coordination_now, long_active,
          bool(self.brake_request or self.precharge_request), CS.out.vEgo, phase, torque, owner_active=CC.enabled,
          coast_lease=(CoastLease(coast_scope, int(request.epoch), int(request.sequence),
            coordination_now - self.precharge_allocator.start, self.precharge_allocator.budget - self.precharge_allocator.area)
            if self.last_allocation is not None and self.last_allocation.phase == 'bounded_true_coast' else None))
        if self.last_observer.fault:
          self.coordination_fault = self.last_observer.reason
        if self.coordination_fault:
          long_active = False
          long_state = LongCtrlState.off
          self.brake_request = False
          self.precharge_request = False
          accel = 0.
          gas = CarControllerParams.INACTIVE_GAS
        elif not self.last_observer.allow:
          gas = CarControllerParams.INACTIVE_GAS
      if coordinated and self.last_admission.coast:
        gas = CarControllerParams.INACTIVE_GAS
      self.last_handoff = self.long_handoff.update(now_nanos / 1e9, long_active, bool(self.brake_request or self.precharge_request), gas,
                                                  pitch_valid, long_state == LongCtrlState.stopping,
                                                  bounded_coast=(coordinated and self.last_allocation is not None
                                                    and self.last_allocation.phase == 'bounded_true_coast'))
      gas = self.last_handoff.gas

    if coordinated:
      if self.coordination_fault:
        allocator_phase = 'fault'
      elif not long_active:
        allocator_phase = 'inactive'
      elif self.brake_request:
        allocator_phase = 'brake'
      elif self.precharge_request:
        allocator_phase = 'precharge'
      elif not self.last_observer.allow:
        allocator_phase = 'release'
      elif self.last_handoff.state in ('qualify', 'resume'):
        allocator_phase = 'engineRamp'
      else:
        allocator_phase = 'drive'
      precharge_only = allocator_phase == 'precharge' or (self.last_allocation is not None
        and self.last_allocation.phase == 'bounded_true_coast')
      self.last_feedback = Feedback(coordination_now, request.epoch, request.sequence,
        float(actuators.accel) if accel_valid else 0., float(gas), bool(self.brake_request),
        bool(not self.brake_request and (self.precharge_request or not self.last_observer.allow or self.last_admission.coast
                                        or self.last_handoff.state in ('qualify', 'resume'))),
        bool(long_active), bool(self.coordination_fault),
        allocator_version=PRECHARGE_PROTOCOL_VERSION, precharge=bool(self.brake_request or self.precharge_request),
        reference_brake=bool(self.reference_brake), allocator_phase=allocator_phase,
        precharge_age=(coordination_now - self.precharge_allocator.start) if precharge_only else 0.,
        budget_remaining=(self.precharge_allocator.budget - self.precharge_allocator.area) if precharge_only else 0.,
        reason=(self.coordination_fault or (self.last_admission.reason if self.last_admission.coast else self.last_observer.reason)))
    self.accel = accel
    self.gas = gas
    return NominalResult(self.last_feedback, self.last_handoff, self.last_grade, accel, gas)
