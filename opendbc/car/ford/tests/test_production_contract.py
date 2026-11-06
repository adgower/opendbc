from dataclasses import asdict, replace
from opendbc.car.ford.longitudinal_coordination import Feedback
from opendbc.car.ford.longitudinal_reference_feedback import ReferenceReceipt
from opendbc.car.ford.longitudinal_production_contract import ProductionWitness, CONTRACT_ID, PREFIX


def witness():
  actual = Feedback(1.02, 7, 10, .07, -5., False, True, allocator_version=3, allocator_phase='engineRamp', budget_remaining=.009)
  nominal = replace(actual, requested=.08, brake=True, limited=False, allocator_phase='brake')
  return ProductionWitness(contract_id=CONTRACT_ID, source_id='source-one', run_id=123,
    epoch=7, sequence=10, producer_ns=1_000_000_000, target_ns=990_000_000, target=.1,
    pose_ns=1_000_000_000, selected_ns=1_020_000_000, actual_raw=.07, nominal_raw=.08,
    nominal_phase='pid', actual=asdict(actual), nominal=asdict(nominal), grade_offset=.01,
    executed_pose_ns=1_000_000_000, birth_ns=1_020_000_000, native_until_ns=1_320_000_000, actor_until_ns=1_520_000_000,
    remaining=.009, selected=True, retired=False)


def test_self_contained_serialized_independent_facts():
  w = witness()
  wire = w.encode()
  assert wire.startswith(PREFIX)
  assert ProductionWitness.parse(wire) == w
  assert ReferenceReceipt.parse(wire) is None
  assert w.nominal_feedback().requested != w.actual_feedback().requested


def test_strict_shape_and_nonfinite_rejected():
  import json
  w = witness()
  assert ProductionWitness.parse(PREFIX + json.dumps({**asdict(w), 'surprise': 1})) is None
  assert ProductionWitness.parse(w.encode() + ' ') is None
  assert ProductionWitness.parse(PREFIX + '{"contract_id":"x","contract_id":"y"}') is None
  assert ProductionWitness.parse(PREFIX + 'x' * 8193) is None
  for key, value in [('remaining', float('nan')), ('native_until_ns', 1_520_000_000),
                     ('actor_until_ns', 1_620_000_000), ('source_id', ''), ('selected', 1)]:
    try:
      text = replace(w, **{key: value}).encode()
    except ValueError:
      continue
    assert ProductionWitness.parse(text) is None


def test_failed_production_marker_cannot_use_native_coast_freeze_exemption():
  from opendbc.car.ford.longitudinal_coordination import freeze_integral
  coast = Feedback(1., 7, 1, .2, -5., False, True, allocator_version=3,
                   reference_brake=True, allocator_phase='release', precharge_age=.02, budget_remaining=.009)
  args = (1.02, 7, 2, .2, .1, .2, 11., 'pid')
  assert not freeze_integral(*args, coast, enabled=True)
  forged = replace(coast, reason=PREFIX + 'broken')
  assert freeze_integral(*args, forged, enabled=True)


def test_selected_witness_binds_actual_episode_metadata_exactly():
  w = witness()
  assert ProductionWitness.parse(w.encode()) == w
  for change in ({'budget_remaining': .5}, {'budget_remaining': -.01}, {'precharge_age': 999.},
                 {'precharge_age': .001}):
    bad = replace(w, actual={**w.actual, **change})
    try:
      encoded = bad.encode()
    except ValueError:
      continue
    assert ProductionWitness.parse(encoded) is None


def test_unknown_peer_enums_decline_registry_and_nominal_wire_without_throwing():
  from opendbc.car import structs
  from opendbc.car.ford.tests.test_production_consumer import command
  from opendbc.car.ford.longitudinal_production_contract import (
    ProductionIssuedRegistry, nominal_from_wire, write_nominal_wire,
  )
  cc, _, _ = command()
  cc.navigatorLongitudinal.nominalLongControlState = 65535
  assert not ProductionIssuedRegistry().issue(cc.as_reader())
  w = witness()
  output = structs.CarControl.Actuators.new_message()
  wire = output.navigatorLongitudinal
  wire.producerMonoTime, wire.epoch, wire.sequence = w.selected_ns, w.epoch, w.sequence
  write_nominal_wire(wire, w.nominal_feedback(), source_id=w.source_id, run_id=w.run_id)
  wire.nominalFeedback.allocatorPhase = 65535
  assert nominal_from_wire(wire, source_id=w.source_id, run_id=w.run_id, now_ns=w.selected_ns,
                           cutoff_ns=w.selected_ns, epoch=w.epoch, sequence=w.sequence) is None
