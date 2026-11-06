from opendbc.car.ford.tests.test_history_release import controller
from opendbc.car.ford.tests.test_coast_reference_consumer import context
from opendbc.car.ford.longitudinal_production_contract import CONTRACT_ID, SOURCE_ID


def command(i=0):
  cc, cs, ns = context(i, 5)
  cs.out.vEgo = cs.out.vEgoRaw = 11.
  q = cc.navigatorLongitudinal
  q.productionContractId = CONTRACT_ID
  q.productionSourceId = SOURCE_ID
  q.productionRunId = 123
  q.nominalRawAcceleration = .125
  q.nominalLongControlState = 'pid'
  q.nominalValid = q.productionSourceValid = q.noLead = True
  q.productionPoseMonoTime = q.radarMonoTime = q.feedbackCutoffMonoTime = ns
  return cc, cs, ns


def step(c, cc, cs, ns):
  c.coordination_control_valid = c.coordination_state_valid = True
  c.coordination_state_ns = c.coordination_speed_ns = ns
  return c.update(cc.as_reader(), cs, ns)


def test_production_off_allocates_nothing_and_selected_startup_only():
  c = controller()
  assert not getattr(c, 'production_reference_enabled', False)
  c.configure_production_reference()
  assert c.production_reference_enabled
  assert c.coast_reference.production_nominal is not c.coast_reference.reference


def test_independent_full_precision_nominal_wire_and_lateral_unchanged():
  c = controller()
  c.configure_production_reference()
  cc, cs, ns = command()
  cc.latActive = True
  original = cc.to_bytes()
  out, _ = step(c, cc, cs, ns)
  cc.clear_write_flag()
  assert cc.to_bytes() == original
  assert out.navigatorLongitudinal.nominalValid
  n = out.navigatorLongitudinal.nominalFeedback
  assert n.requested == cc.navigatorLongitudinal.nominalRawAcceleration
  assert n.requested != cc.actuators.accel
  assert not out.navigatorLongitudinal.reason.startswith('navigator-production-v5:')


def test_invalid_current_source_faults_without_throwing_or_second_update():
  c = controller()
  c.configure_production_reference()
  cc, cs, ns = command()
  cc.navigatorLongitudinal.productionRunId = 0
  out, _ = step(c, cc, cs, ns)
  f = out.navigatorLongitudinal
  assert f.fault and not f.valid
  assert f.gas == -5. and not f.brake and not f.precharge
  assert out.accel == 0
  assert c.frame == 1


def test_run_restart_requires_disabled_owner_then_preserves_native_history():
  c = controller()
  c.configure_production_reference()
  cc, cs, ns = command()
  step(c, cc, cs, ns)
  owner = c.coast_reference
  gate, allocator, handoff = owner.gate, owner.allocator, owner.handoff
  cc, cs, ns = command(2)
  cc.enabled = cc.longActive = False
  cc.navigatorLongitudinal.productionRunId = 456
  c.frame = 2
  step(c, cc, cs, ns)
  assert owner.run_id == 456
  assert owner.gate is gate and owner.allocator is allocator and owner.handoff is handoff
  cc, cs, ns = command(4)
  cc.navigatorLongitudinal.productionRunId = 456
  c.frame = 4
  out, _ = step(c, cc, cs, ns)
  assert not out.navigatorLongitudinal.fault


def selected_owner():
  from types import SimpleNamespace as NS
  from opendbc.car.ford.longitudinal_coordination import Feedback
  c = controller()
  c.configure_production_reference()
  cc, cs, ns = command()
  o = c.coast_reference
  o._state = cs
  execution = o.prepare(c, cc.as_reader(), cs, ns)
  c.coordination_control_valid = c.coordination_state_valid = True
  c.coordination_state_ns = c.coordination_speed_ns = ns
  c.release_barrier.needs = False
  c.last_observer = NS(allow=True)
  o.begin(c, execution, cs, ns)
  c.last_grade = NS(offset=.01, gravity=0.)
  o.actual_accel = .2
  g, a, h = o.gate, o.allocator, o.handoff
  g.selected = g.deferred = g.scope = True
  g.spent = False
  g.arm = ns / 1e9
  g.lease_not_before = ns / 1e9 + .3
  g.area = g.allocator_area = a.area = .001
  g.response_area = 0.
  h.recovering = True
  h.recovery_started = ns / 1e9
  h.release_until = ns / 1e9 + .3
  h.epoch = g.current_epoch
  actual = Feedback(ns / 1e9, 7, 1, float(cc.actuators.accel), -5., False, True,
                    allocator_version=3, allocator_phase='engineRamp')
  return c, o, execution, actual, ns


def test_first_selection_seals_distinct_horizons_without_changing_allocator_or_debt():
  from opendbc.car.ford.longitudinal_production_contract import ProductionWitness
  c, o, execution, actual, ns = selected_owner()
  debt = (o.gate.area, o.gate.allocator_area, o.allocator.area, o.gate.response_area)
  out = o.finish(c, execution, actual, ns)
  witness = ProductionWitness.parse(out.reason)
  assert witness is not None
  assert witness.native_until_ns == ns + 300_000_000
  assert witness.actor_until_ns == ns + 500_000_000
  assert o.allocator.deadline == .3
  assert debt == (o.gate.area, o.gate.allocator_area, o.allocator.area, o.gate.response_area)
  o.retire('test')
  assert o.gate.deadline == .3
  assert o.birth_ns == ns and o.retired
  assert debt == (o.gate.area, o.gate.allocator_area, o.allocator.area, o.gate.response_area)


def test_unqualified500_keeps_native_actual_feedback_and_can_exact():
  from opendbc.car.ford.longitudinal_coordination import feedback_from_wire
  native, production = controller(), controller()
  native.configure_coast_reference(enabled=True)
  production.configure_production_reference()
  for i in range(0, 180, 2):
    paired, cs, ns = command(i)
    paired.navigatorLongitudinal.noLead = False
    raw = .2 if i < 40 else .08 if i < 90 else .15
    paired.actuators.accel = raw
    paired.navigatorLongitudinal.nominalRawAcceleration = .125
    paired.orientationNED = [0., -.012, 0.]
    ordinary = paired.as_reader().as_builder()
    ordinary.navigatorLongitudinal.allocatorVersion = 4
    native.frame = production.frame = i
    no, nc = step(native, ordinary, cs, ns)
    po, pc = step(production, paired, cs, ns)
    assert nc == pc
    assert feedback_from_wire(no.navigatorLongitudinal) == feedback_from_wire(po.navigatorLongitudinal)
    assert (no.accel, no.gas) == (po.accel, po.gas)
  assert not production.coast_reference.birth_ns


def test_noncanonical_nominal_request_faults_before_selected_receipt_construction():
  c = controller()
  c.configure_production_reference()
  cc, cs, ns = command()
  cc.navigatorLongitudinal.nominalRawAcceleration = .1234567890123
  out, _ = step(c, cc, cs, ns)
  assert out.navigatorLongitudinal.fault
  assert not c.coast_reference.birth_ns


def retired_released_proof():
  from dataclasses import replace
  from types import SimpleNamespace as NS
  c, o, execution, actual, birth = selected_owner()
  o.finish(c, execution, actual, birth)
  o.retire('selected_released')
  cc, cs, ns = command(48)
  execution = o.prepare(c, cc.as_reader(), cs, ns)
  o._state = cs
  o.proof = (birth + 400_000_000, 7)
  o.proof_seen = True
  o.last_callback_ns = birth + 460_000_000
  o.gate.arm = o.gate.permission = o.gate.pending_qualified = None
  o.gate.deferred = o.gate.selected = False
  o.gate.area = o.gate.allocator_area = o.allocator.area = o.gate.response_area = 0.
  c.last_release = NS(reason='baseline_released')
  actual = replace(actual, t=ns / 1e9, sequence=49, gas=.01)
  return c, o, execution, actual, ns


def test_released_positive_gas_keeps_genuine_proof_and_promotes_only_pending():
  c, o, execution, actual, ns = retired_released_proof()
  proof = o.proof
  numeric = (o.gate.area, o.gate.allocator_area, o.allocator.area, o.gate.response_area,
             o.handoff.last_valid, o.handoff.since)
  o.finish(c, execution, actual, ns)
  assert o.proof == proof
  c.last_feedback = actual
  cc, cs, later = command(52)
  execution = o.prepare(c, cc.as_reader(), cs, later)
  o._proof_before(c, execution, cs, later)
  assert o.proof is None
  assert o.gate.pending_qualified == proof[0] / 1e9
  assert o.gate.permission is None and o.gate.arm is None
  assert numeric == (o.gate.area, o.gate.allocator_area, o.allocator.area, o.gate.response_area,
                     o.handoff.last_valid, o.handoff.since)


def test_genuine_proof_discarded_for_actual_brake_precharge_fault_or_lost_scope():
  from dataclasses import replace
  for change in ({'brake': True}, {'precharge': True}, {'fault': True}, {'valid': False}, {'scope': False}):
    c, o, execution, actual, ns = retired_released_proof()
    if 'scope' in change:
      cc = execution.as_builder()
      cc.navigatorLongitudinal.noLead = False
      o.original = cc.as_reader()
    else:
      actual = replace(actual, **change)
    o.finish(c, execution, actual, ns)
    assert o.proof is None


def test_retained_proof_malformed_orientation_reaches_native_fault_without_throwing():
  for orientation in ([], [0., float('inf'), 0.], [0., float('nan'), 0.]):
    c, o, execution, actual, ns = retired_released_proof()
    c.last_feedback = actual
    cc, cs, later = command(50)
    cc.orientationNED = orientation
    c.frame = 50
    out, _ = step(c, cc, cs, later)
    f = out.navigatorLongitudinal
    assert f.fault and not f.valid
    assert f.gas == -5. and not f.brake and not f.precharge
    assert out.accel == 0.
    assert o.proof is None
    assert o.gate.permission is None and o.gate.arm is None
    assert c.frame == 51


def test_unknown_nominal_phase_faults_in_real_controller_without_enum_exception():
  for paired in (False, True):
    c = controller()
    c.configure_production_reference()
    cc, cs, ns = command()
    if not paired:
      cc.navigatorLongitudinal.productionContractId = ''
    cc.navigatorLongitudinal.nominalLongControlState = 65535
    out, _ = step(c, cc, cs, ns)
    f = out.navigatorLongitudinal
    assert f.fault and not f.valid
    assert f.gas == -5. and not f.brake and not f.precharge
    assert out.accel == 0.
    assert c.frame == 1
