import math
import numpy as np
from opendbc.can import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, apply_hysteresis, structs
from opendbc.car.ford import fordcan
from opendbc.car.ford.lateral_angle import LateralAngle
from opendbc.car.ford.longitudinal_handoff import Handoff
from opendbc.car.ford.longitudinal_release import ReleaseGate
from opendbc.car.ford.longitudinal_admission import BrakeAdmission
from opendbc.car.ford.longitudinal_restart import RestartClearance
from opendbc.car.ford.longitudinal_coordination import GradeLease, Feedback
from opendbc.car.ford.values import CarControllerParams, FordFlags, FordPrefLateralControl, CAR
from opendbc.car.interfaces import CarControllerBase, V_CRUISE_MAX

LongCtrlState = structs.CarControl.Actuators.LongControlState
VisualAlert = structs.CarControl.HUDControl.VisualAlert

def anti_overshoot(apply_curvature, apply_curvature_last, v_ego):
  diff = 0.1
  tau = 5  # 5s smooths over the overshoot
  dt = DT_CTRL * CarControllerParams.STEER_STEP
  alpha = 1 - np.exp(-dt / tau)

  lataccel = apply_curvature * (v_ego ** 2)
  last_lataccel = apply_curvature_last * (v_ego ** 2)
  last_lataccel = apply_hysteresis(lataccel, last_lataccel, diff)
  last_lataccel = alpha * lataccel + (1 - alpha) * last_lataccel

  output_curvature = last_lataccel / (max(v_ego, 1) ** 2)

  return float(np.interp(v_ego, [5, 10], [apply_curvature, output_curvature]))


def apply_creep_compensation(accel: float, v_ego: float) -> float:
  creep_accel = np.interp(v_ego, [1., 3.], [0.6, 0.])
  creep_accel = np.interp(accel, [0., 0.2], [creep_accel, 0.])
  accel -= creep_accel
  return float(accel)


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.CAN = fordcan.CanBus(CP)

    self.apply_curvature_last = 0
    self.anti_overshoot_curvature_last = 0
    self.accel = 0.0
    self.gas = 0.0
    self.brake_request = False
    self.long_handoff = Handoff(0.5, 0.02)
    self.longitudinal_coordination_enabled = False  # explicit offroad opt-in from card
    self.grade_map = GradeLease(True)
    self.release_barrier = RestartClearance(True)
    self.brake_admission = BrakeAdmission()
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
    self.last_release = None
    self.last_handoff = None
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False
    self.lead_distance_bars_last = None
    self.distance_bar_frame = 0

    # Fork policy: default to angle mode for improved steering feel
    self.lateral_control_mode = FordPrefLateralControl.angle
    if CP.flags & FordFlags.LOW_SPEED_GAIN and CP.flags & FordFlags.PINION_CURVATURE and CP.carFingerprint == CAR.FORD_EXPEDITION_MK4:
      from opendbc.car.ford.lateral_angle_very_low_speed_prototype import VeryLowSpeedGainPrototype
      self.lateral_angle = VeryLowSpeedGainPrototype(CP, enabled=True)
    else:
      self.lateral_angle = LateralAngle(CP)
    self.angle_mode_engaged = False
    self.shadow_curvature = 0.0

  def update(self, CC, CS, now_nanos):
    can_sends = []
    lka_msg_sent = False
    angle_mode_engaged_last = self.angle_mode_engaged

    actuators = CC.actuators
    hud_control = CC.hudControl

    main_on = CS.out.cruiseState.available
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)
    fcw_alert = hud_control.visualAlert == VisualAlert.fcw

    ### acc buttons ###
    if CC.cruiseControl.cancel:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, cancel=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, cancel=True))
    elif CC.cruiseControl.resume and (self.frame % CarControllerParams.BUTTONS_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, resume=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, resume=True))
    # if stock lane centering isn't off, send a button press to toggle it off
    # the stock system checks for steering pressed, and eventually disengages cruise control
    elif CS.acc_tja_status_stock_values["Tja_D_Stat"] != 0 and (self.frame % CarControllerParams.ACC_UI_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, tja_toggle=True))

    ### lateral control ###
    # send steer msg at 20Hz
    if (self.frame % CarControllerParams.STEER_STEP) == 0:
      # Determine lateral control mode
      use_angle_mode = self.lateral_control_mode == FordPrefLateralControl.angle and (self.CP.flags & FordFlags.CANFD)

      if use_angle_mode:
        # Angle mode: path_angle is the actuator; c0/c2/c3 are zero
        angle_result = self.lateral_angle.update(CC, CS, actuators)
        lat_active = CC.latActive and not self.lateral_angle.human_turn_active

        # In angle mode, curvature is held at 0 (inactive sentinel) on the wire
        apply_curvature = 0.0
        path_angle = angle_result.path_angle
        self.shadow_curvature = angle_result.shadow_curvature
        self.angle_mode_engaged = CC.latActive
      else:
        # Curvature mode: stock upstream behavior
        # Bronco and some other cars consistently overshoot curv requests
        # Apply some deadzone + smoothing convergence to avoid oscillations
        if self.CP.carFingerprint in (CAR.FORD_BRONCO_SPORT_MK1, CAR.FORD_F_150_MK14):
          self.anti_overshoot_curvature_last = anti_overshoot(actuators.curvature, self.anti_overshoot_curvature_last, CS.out.vEgoRaw)
          apply_curvature = self.anti_overshoot_curvature_last
        else:
          apply_curvature = actuators.curvature

        # apply rate limits, curvature error limit, and clip to signal range
        current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)
        # No blending at low speed due to lack of torque wind-up and inaccurate current curvature
        if CS.out.vEgoRaw > 9:
          apply_curvature = float(np.clip(apply_curvature, current_curvature - CarControllerParams.CURVATURE_ERROR,
                                         current_curvature + CarControllerParams.CURVATURE_ERROR))
        apply_curvature = CarControllerParams.CURVATURE_LIMITS.apply_limits(apply_curvature, self.apply_curvature_last, CS.out.vEgoRaw,
                                                                            0., CC.latActive, CarControllerParams.STEER_STEP)
        lat_active = CC.latActive
        path_angle = 0.0
        self.angle_mode_engaged = False
        self.shadow_curvature = 0.0

      self.apply_curvature_last = apply_curvature

      if self.CP.flags & FordFlags.CANFD:
        # Panda must receive the mode transition before the first request using it,
        # including steering frames between the regular 33 Hz LKA updates.
        # Pinion-mode safety checks must see this request's current intent,
        # including active requests between the regular metadata slots.
        sync_pinion_intent = (lat_active and self.CP.carFingerprint == CAR.FORD_EXPEDITION_MK4
                              and bool(self.CP.flags & FordFlags.PINION_CURVATURE))
        if use_angle_mode and self.angle_mode_engaged and ((self.frame % CarControllerParams.LKA_STEP) == 0 or
                                                         self.angle_mode_engaged != angle_mode_engaged_last or sync_pinion_intent):
          can_sends.append(fordcan.create_lka_msg(self.packer, self.CAN, self.angle_mode_engaged, -self.shadow_curvature))
          lka_msg_sent = True
        mode = 1 if lat_active else 0
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        # In angle mode: c0=0, c1=path_angle, c2=0, c3=0
        # In curvature mode: c0=0, c1=0, c2=curvature, c3=0
        can_sends.append(fordcan.create_lat_ctl2_msg(self.packer, self.CAN, mode, 0., -path_angle, -apply_curvature, 0., counter))
      else:
        can_sends.append(fordcan.create_lat_ctl_msg(self.packer, self.CAN, lat_active, 0., -path_angle, -apply_curvature, 0.))

    # Send LKA at 33 Hz and on mode transitions. On disengagement the neutral
    # steering request must reset Panda's angle history before clearing its mode.
    if ((self.frame % CarControllerParams.LKA_STEP) == 0 or self.angle_mode_engaged != angle_mode_engaged_last) and not lka_msg_sent:
      # Pass angle mode state to ford.h via unused bytes in Lane_Assist_Data1
      # shadow_curvature is negated to match wire sign convention (see fordcan.py)
      can_sends.append(fordcan.create_lka_msg(self.packer, self.CAN, self.angle_mode_engaged, -self.shadow_curvature))

    ### longitudinal control ###
    # send acc msg at 50Hz
    if self.CP.openpilotLongitudinalControl and (self.frame % CarControllerParams.ACC_CONTROL_STEP) == 0:
      navigator_longitudinal = self.CP.carFingerprint == CAR.FORD_EXPEDITION_MK4
      accel_valid = math.isfinite(actuators.accel)
      accel = (self.accel if CC.longActive else 0.0) if navigator_longitudinal and not accel_valid else actuators.accel
      gas = CarControllerParams.INACTIVE_GAS if navigator_longitudinal and not accel_valid else accel

      if CC.longActive and (not navigator_longitudinal or accel_valid):
        # Compensate for engine creep at low speed.
        # Either the ABS does not account for engine creep, or the correction is very slow
        # TODO: verify this applies to EV/hybrid
        accel = apply_creep_compensation(accel, CS.out.vEgo)

        # The stock system has been seen rate limiting the brake accel to 5 m/s^3,
        # however even 3.5 m/s^3 causes some overshoot with a step response.
        accel = max(accel, self.accel - (3.5 * CarControllerParams.ACC_CONTROL_STEP * DT_CTRL))

      accel = float(np.clip(accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      gas = float(np.clip(gas, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))

      # Both gas and accel are in m/s^2, accel is used solely for braking
      if not CC.longActive or gas < CarControllerParams.MIN_GAS:
        gas = CarControllerParams.INACTIVE_GAS

      # PCM applies pitch compensation to gas/accel, but we need to compensate for the brake/pre-charge bits
      accel_due_to_pitch = 0.0
      pitch_valid = len(CC.orientationNED) == 3 and all(math.isfinite(x) for x in CC.orientationNED)
      if len(CC.orientationNED) == 3 and (not navigator_longitudinal or pitch_valid):
        accel_due_to_pitch = math.sin(CC.orientationNED[1]) * ACCELERATION_DUE_TO_GRAVITY

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
          CC.longActive, CS.out.vEgo, str(CC.actuators.longControlState), request.referenceValid)
        accel_due_to_pitch = self.last_grade.gravity
        if actuators.accel < CarControllerParams.MIN_GAS:
          accel_due_to_pitch = min(installed_gravity, accel_due_to_pitch)
        if (CC.longActive and CS.out.vEgo >= 3. and CC.actuators.longControlState == LongCtrlState.pid
            and (self.last_grade.reason == 'invalid_pose_disengage' or not accel_valid)):
          self.coordination_fault = 'invalid_pose_or_accel'

      brake_request_previous = self.brake_request
      accel_pitch_compensated = accel + accel_due_to_pitch
      if accel_pitch_compensated > 0.3 or not CC.longActive:
        self.brake_request = False
      elif accel_pitch_compensated < 0.0:
        self.brake_request = True

      if navigator_longitudinal:
        self.last_admission = self.brake_admission.update(coordination_now,
          coordinated and not self.coordination_fault and CC.longActive and CS.out.vEgo >= 3.0
          and CS.out.canValid and request.poseValid and str(CC.actuators.longControlState) == 'pid' and pitch_valid and request.referenceValid,
          self.brake_request, brake_request_previous, actuators.accel, accel_pitch_compensated,
          not self.release_barrier.needs, getattr(CS, 'longitudinal_brake_torque', None))
        self.brake_request = self.last_admission.brake
        self.last_release = self.release_gate.update(now_nanos / 1e9, CC.longActive, self.brake_request,
                                                     brake_request_previous, actuators.accel, accel_pitch_compensated,
                                                     str(CC.actuators.longControlState), pitch_valid)
        self.brake_request = self.last_release.brake
        if coordinated:
          torque = getattr(CS, 'longitudinal_brake_torque', None)
          if torque is not None and not CS.out.canValid:
            torque = None
          self.last_observer = self.release_barrier.update(coordination_now, CC.longActive,
            self.brake_request, CS.out.vEgo, str(CC.actuators.longControlState), torque, owner_active=CC.enabled)
          if self.last_observer.fault:
            self.coordination_fault = self.last_observer.reason
          if self.coordination_fault:
            CC = CC.as_builder()
            CC.longActive = False
            CC.actuators.accel = 0.
            CC.actuators.longControlState = LongCtrlState.off
            self.brake_request = False
            accel = 0.
            gas = CarControllerParams.INACTIVE_GAS
          elif not self.last_observer.allow:
            gas = CarControllerParams.INACTIVE_GAS
        if coordinated and self.last_admission.coast:
          gas = CarControllerParams.INACTIVE_GAS
        self.last_handoff = self.long_handoff.update(now_nanos / 1e9, CC.longActive, self.brake_request, gas,
                                                    pitch_valid, CC.actuators.longControlState == LongCtrlState.stopping)
        gas = self.last_handoff.gas

      if coordinated:
        self.last_feedback = Feedback(coordination_now, request.epoch, request.sequence,
          float(actuators.accel) if accel_valid else 0., float(gas), bool(self.brake_request),
          bool(not self.brake_request and (not self.last_observer.allow or self.last_admission.coast or self.last_handoff.state in ('qualify', 'resume'))),
          bool(CC.longActive), bool(self.coordination_fault))
      stopping = CC.actuators.longControlState == LongCtrlState.stopping
      # TODO: look into using the actuators packet to send the desired speed
      can_sends.append(fordcan.create_acc_msg(self.packer, self.CAN, CC.longActive, gas, accel, stopping, self.brake_request, v_ego_kph=V_CRUISE_MAX))

      self.accel = accel
      self.gas = gas

    ### ui ###
    send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)
    # send lkas ui msg at 1Hz or if ui state changes
    if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_lkas_ui_msg(self.packer, self.CAN, main_on, CC.latActive, steer_alert, hud_control, CS.lkas_status_stock_values))

    # send acc ui msg at 5Hz or if ui state changes
    if hud_control.leadDistanceBars != self.lead_distance_bars_last:
      send_ui = True
      self.distance_bar_frame = self.frame

    if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      show_distance_bars = self.frame - self.distance_bar_frame < 400
      can_sends.append(fordcan.create_acc_ui_msg(self.packer, self.CAN, self.CP, main_on, CC.latActive,
                                                 fcw_alert, CS.out.cruiseState.standstill, show_distance_bars,
                                                 hud_control, CS.acc_tja_status_stock_values))

    self.main_on_last = main_on
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert
    self.lead_distance_bars_last = hud_control.leadDistanceBars

    new_actuators = actuators.as_builder()
    new_actuators.curvature = self.apply_curvature_last
    new_actuators.accel = self.accel
    new_actuators.gas = self.gas

    if self.longitudinal_coordination_enabled and self.last_feedback is not None:
      f = self.last_feedback
      output = new_actuators.navigatorLongitudinal
      output.producerMonoTime = int(round(f.t * 1e9))
      output.epoch = f.epoch
      output.sequence = f.sequence
      output.requested = f.requested
      output.gas = f.gas
      output.brake = f.brake
      output.limited = f.limited
      output.valid = f.valid
      output.fault = f.fault
      output.reason = self.coordination_fault or (self.last_admission.reason if self.last_admission.coast else self.last_observer.reason)
      output.gradeOffset = self.last_grade.offset

    self.frame += 1
    return new_actuators, can_sends
