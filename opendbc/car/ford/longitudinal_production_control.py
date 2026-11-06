"""Resident production consumer history and one nonrenewable500ms selection.

The native gate/allocator/clearance histories remain the actual owners. This
module changes only explicitly bounded episode endpoints and witness provenance.
"""
from dataclasses import asdict, replace
import math
import struct

from opendbc.car.ford.longitudinal_reference_control import ReferenceControl, BoundedResponseEntry
from opendbc.car.ford.longitudinal_nominal_reference import NominalReference
from opendbc.car.ford.longitudinal_production_contract import (
  CONTRACT_ID, SOURCE_ID, REQUEST_VERSION, FRESH_NS, ProductionWitness, request_valid,
  request_identity, write_nominal_wire,
)


def f32(value):
  return struct.unpack('f', struct.pack('f', value))[0]


class ProductionReferenceControl(ReferenceControl):
  def __init__(self, CP, history_enabled=False, source_id=SOURCE_ID):
    super().__init__(CP, history_enabled)
    self.production_nominal = NominalReference(CP, history_enabled=history_enabled)
    self.source_id = source_id
    self.run_id = None
    self.last_identity = None
    self.original = None
    self.current_valid = False
    self.independent_result = None
    self.native_result = None
    self.birth_ns = self.birth_epoch = 0
    self.retired = False
    self.remaining = .01
    self.native_published_start = 0
    self.proof = None
    self.proof_seen = False
    self.last_callback_ns = None
    self.last_grade_offset = None
    self.last_reason = 'unborn'

  def retire(self, reason):
    if self.birth_ns and not self.retired:
      self.retired = True
      self.episode = None
      self.last_reason = reason
    self.gate.deadline = BoundedResponseEntry.deadline

  def prepare(self, controller, CC, CS, now_ns):
    self.original = CC
    self.current_valid = False
    q = CC.navigatorLongitudinal
    if q.allocatorVersion != REQUEST_VERSION:
      self.retire('unpaired_current')
      return CC
    execution = CC.as_builder() if hasattr(CC, 'as_builder') else CC.as_reader().as_builder()
    valid = request_valid(CC, now_ns, self.source_id)
    identity = request_identity(CC) if valid else None
    if valid and self.run_id is not None and q.productionRunId != self.run_id:
      if not CC.enabled:
        self.retire('producer_restart')
        self.run_id = int(q.productionRunId)
        self.last_identity = None
        self.production_nominal = NominalReference(self.reference.CP, history_enabled=self.reference.history_release_enabled)
      else:
        valid = False
    if valid and self.last_identity is not None:
      old = self.last_identity
      valid = bool((identity[:3] == old[:3] and identity == old)
                   or (identity[1] >= old[1] and (identity[1] > old[1] or identity[2] > old[2]) and identity[3] >= old[3]))
    if not valid:
      self.retire('invalid_current')
      if CC.enabled:
        controller.coordination_fault = 'production_invalid_current'
      execution.navigatorLongitudinal.allocatorVersion = 0
      execution.navigatorLongitudinal.enabled = False
      return execution.as_reader()
    if self.run_id is None:
      self.run_id = int(q.productionRunId)
    self.last_identity = identity
    self.current_valid = True
    execution.navigatorLongitudinal.allocatorVersion = 4
    return execution.as_reader()

  def scope(self, controller, CC, CS, now_ns):
    q = self.original.navigatorLongitudinal
    tq = getattr(CS, 'longitudinal_brake_torque', None)
    try:
      return bool(self.current_valid and CC.enabled and CC.longActive and CS.out.canValid
        and not CS.out.gasPressed and not CS.out.brakePressed and str(CC.hudControl.visualAlert) != 'fcw'
        and q.productionSourceValid and not q.productionUrgent and q.noLead
        and q.targetValid and q.poseValid and q.referenceValid and q.nominalValid
        and len(CC.orientationNED) == 3 and all(math.isfinite(v) for v in CC.orientationNED)
        and 0 < q.targetMonoTime <= q.producerMonoTime <= now_ns and now_ns - q.targetMonoTime <= FRESH_NS
        and 0 < q.productionPoseMonoTime <= q.producerMonoTime
        and q.productionPoseMonoTime <= q.poseMonoTime <= now_ns and now_ns - q.poseMonoTime <= FRESH_NS
        and 0 < q.radarMonoTime <= now_ns and now_ns - q.radarMonoTime <= FRESH_NS
        and 0 < q.feedbackCutoffMonoTime <= q.producerMonoTime
        and str(CC.actuators.longControlState) == 'pid' and str(q.nominalLongControlState) == 'pid'
        and 20 * .44704 <= CS.out.vEgo <= 30 * .44704
        and 0 < q.targetAcceleration <= .2 and .02 < CC.actuators.accel <= .3
        and tq is not None and tq.valid and math.isfinite(tq.value) and 0 <= tq.value <= 4
        and 0 <= now_ns - round(tq.t * 1e9) <= FRESH_NS
        and not controller.release_barrier.needs and controller.last_observer is not None and controller.last_observer.allow)
    except (AttributeError, ValueError, TypeError, OverflowError, RuntimeError):
      return False

  def _proof_before(self, controller, CC, CS, now_ns):
    if self.proof is None:
      return
    proof_ns, epoch = self.proof
    if (self.last_callback_ns is None or not 0 < now_ns - self.last_callback_ns <= 40_000_001
        or now_ns < proof_ns or now_ns - proof_ns >= 300_000_000
        or epoch != int(CC.navigatorLongitudinal.epoch) or not self.scope(controller, CC, CS, now_ns)):
      self.proof = None
      return
    if now_ns < self.birth_ns + 500_000_000:
      return
    if self.last_grade_offset is None or controller.last_feedback is None:
      return
    prior = controller.last_feedback
    age = now_ns - round(prior.t * 1e9)
    if not 0 <= age <= FRESH_NS:
      return
    offset = self.last_grade_offset
    lower = max(0., offset - (2. ** (math.frexp(offset)[1] - 24) if offset else 0.))
    lower = max(0., lower - .005 * age / 1e9) * max(0., min(1., (CS.out.vEgo - 3) / 2))
    net = float(CC.actuators.accel) + 9.81 * math.sin(CC.orientationNED[1]) + lower
    g = self.gate
    if not math.isfinite(net) or net <= .02 or any(v != 0 for v in (g.area, g.allocator_area, self.allocator.area, g.response_area)):
      return
    if g.pending_qualified is None and g.permission is None and g.arm is None:
      g.pending_qualified = proof_ns / 1e9
      self.proof = None

  def begin(self, controller, CC, CS, now_ns):
    super().begin(controller, CC, CS, now_ns)
    self.native_result = self.result
    self.independent_result = None
    if self.current_valid:
      nominal = CC.as_builder() if hasattr(CC, 'as_builder') else CC.as_reader().as_builder()
      q = self.original.navigatorLongitudinal
      nominal.actuators.accel = q.nominalRawAcceleration
      nominal.actuators.longControlState = q.nominalLongControlState
      nominal.navigatorLongitudinal.allocatorVersion = 3
      for key in ('coordination_control_valid', 'coordination_state_ns', 'coordination_speed_ns', 'coordination_state_valid'):
        setattr(self.production_nominal, key, getattr(controller, key))
      self.independent_result = self.production_nominal.update(nominal.as_reader(), CS, now_ns)
    scoped = self.scope(controller, CC, CS, now_ns)
    if self.birth_ns and not self.retired:
      if now_ns >= self.birth_ns + 500_000_000 or int(CC.navigatorLongitudinal.epoch) != self.birth_epoch or not scoped:
        self.retire('expired_or_scope')
    if self.birth_ns and not self.retired and self.independent_result is not None:
      self.result = self.independent_result
      h = self.result.handoff
      if h is not None:
        self.gate.nominal_ramp_sample.update(gas=float(h.gas), state=h.state)
    self._proof_before(controller, CC, CS, now_ns)

  def finish(self, controller, CC, output, now_ns):
    g, a, h = self.gate, self.allocator, self.handoff
    scoped = self.scope(controller, CC, self._state, now_ns)
    valid_output = bool(output.valid and not output.fault and not output.brake and not output.precharge
                        and f32(output.gas) == -5. and self.actual_accel + controller.last_grade.gravity >= -.14)
    first = bool(not self.birth_ns and self.native_published_start != now_ns and self.episode is None and not self.retired
      and scoped and valid_output and g.selected and g.deferred and g.scope and not g.spent
      and g.arm is not None and round(g.arm * 1e9) == now_ns and g.deadline == .3
      and round(g.lease_not_before * 1e9) == now_ns + 300_000_000
      and h.recovering and round(h.recovery_started * 1e9) == now_ns
      and round(h.release_until * 1e9) == now_ns + 300_000_000 and h.epoch == g.current_epoch
      and a.deadline == .3 and g.budget == g.response_budget == .01
      and all(math.isfinite(v) and 0 <= v < .01 for v in (g.area, g.allocator_area, a.area, g.response_area)))
    selected = bool(first or self.birth_ns and not self.retired and scoped and valid_output and g.selected)
    if first:
      self.birth_ns, self.birth_epoch = now_ns, int(CC.navigatorLongitudinal.epoch)
      g.deadline = .5
      g.lease_not_before = (now_ns + 500_000_000) / 1e9
      h.release_until = (now_ns + 500_000_000) / 1e9
    if selected:
      self.remaining = min(self.remaining, .01 - max(g.area, g.allocator_area, a.area))
      if not 0 < self.remaining <= .01:
        self.retire('budget')
        selected = False
    if selected and self.independent_result is not None:
      q = self.original.navigatorLongitudinal
      output = replace(output, precharge_age=(now_ns - self.birth_ns) / 1e9, budget_remaining=self.remaining)
      actual = replace(output, requested=f32(output.requested), gas=f32(output.gas))
      n = self.independent_result.feedback
      witness = ProductionWitness(CONTRACT_ID, self.source_id, self.run_id,
        int(q.epoch), int(q.sequence), int(q.producerMonoTime), int(q.targetMonoTime), float(q.targetAcceleration),
        int(q.productionPoseMonoTime), now_ns, f32(output.requested), float(q.nominalRawAcceleration), str(q.nominalLongControlState),
        asdict(actual), asdict(n), f32(controller.last_grade.offset), int(CC.navigatorLongitudinal.poseMonoTime), self.birth_ns,
        self.birth_ns + 300_000_000, self.birth_ns + 500_000_000, self.remaining, True, False)
      output = replace(output, reason=witness.encode())
    else:
      if self.birth_ns and not self.retired:
        self.retire('selection_lost')
      current = self.result
      self.result = self.native_result
      try:
        output = super().finish(controller, CC, output, now_ns)
      finally:
        self.result = current
      if output.reason.startswith('bounded-release-reference-v2:'):
        self.native_published_start = round(g.arm * 1e9) if g.arm is not None else now_ns
    # Qualification is factual released history, distinct from permission to
    # withhold a brake. A normal rising engine command must not erase it.
    released = bool(output.valid and not output.fault and not output.brake and not output.precharge)
    if self.proof is not None and not (scoped and released):
      self.proof = None
    if (not self.proof_seen and self.birth_ns and now_ns < self.birth_ns + 500_000_000 and scoped and released
        and getattr(controller.last_release, 'reason', '') == 'qualified_positive_release'
        and self.actual_accel + controller.last_grade.gravity > 0):
      self.proof = (now_ns, int(CC.navigatorLongitudinal.epoch))
      self.proof_seen = True
    self.last_callback_ns = now_ns
    self.last_grade_offset = f32(controller.last_grade.offset)
    return output

  def write_nominal_wire(self, wire):
    if self.run_id is not None and self.independent_result is not None:
      write_nominal_wire(wire, self.independent_result.feedback, source_id=self.source_id, run_id=self.run_id)
