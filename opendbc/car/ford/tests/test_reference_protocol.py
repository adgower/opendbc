from dataclasses import asdict, replace
from types import SimpleNamespace as NS
import struct

import pytest

from opendbc.car.ford.longitudinal_coordination import Feedback, allocator_feedback_valid, freeze_integral
from opendbc.car.ford.longitudinal_precharge import allocator_request_valid
from opendbc.car.ford.longitudinal_reference_feedback import (
  MODEL_ID,
  SOURCE,
  ReferenceReceipt,
  ReferenceRegistry,
  ReleaseReceipt,
)


def _float32(value):
  return struct.unpack("f", struct.pack("f", value))[0]


def sample():
  n = Feedback(1.02, 7, 10, _float32(0.07), -5.0, True, False, allocator_version=3, allocator_phase="brake")
  r = ReferenceReceipt(
    7,
    10,
    1_000_000_000,
    1_020_000_000,
    1_000_000_000,
    1_300_000_000,
    0.009,
    "one",
    SOURCE,
    990_000_000,
    actual_gas=_float32(0.05),
    actual_limited=True,
    actual_phase="engineRamp",
    nominal=asdict(n),
    model_id=MODEL_ID,
  )
  f = replace(n, brake=False, limited=True, gas=r.actual_gas, requested=_float32(n.requested), allocator_phase="engineRamp", reason=r.encode())
  reg = ReferenceRegistry(MODEL_ID)
  reg.issue(7, 10, 1.0, 0.08, True, SOURCE, 990_000_000)
  return r, f, reg


def check(reg, f):
  return reg.verify(f, now=1.03, epoch=7, sequence=11, target=0.08, error=0.2, wanted=0.3, speed=27.0, eligible=True)


def test_independent_serialized_witness():
  r, f, reg = sample()
  f = Feedback.decode(f.encode())
  assert ReferenceReceipt.parse(r.encode()) == r
  assert check(reg, f)
  assert reg.last_nominal.brake
  assert not f.brake


@pytest.mark.parametrize("version,valid", [(3, True), (4, True), (5, False)])
def test_known_request_versions(version, valid):
  q = NS(allocatorVersion=version, targetValid=True, targetAcceleration=0.1, targetMonoTime=990_000_000, producerMonoTime=1_000_000_000)
  assert allocator_request_valid(q, 1_010_000_000) is valid


def test_reserved_role_does_not_use_legacy_freeze_exemption():
  f = Feedback(
    1.0, 7, 1, 0.2, -5.0, False, True, allocator_version=3, reference_brake=True, allocator_phase="release", precharge_age=0.02, budget_remaining=0.009
  )
  args = (1.02, 7, 2, 0.2, 0.1, 0.2, 20.0, "pid")
  assert allocator_feedback_valid(f)
  assert not freeze_integral(*args, f, enabled=True)
  for reason in ("bounded-release-reference-v2:broken", "bounded-release-selection-v1:broken"):
    marked = replace(f, reason=reason)
    assert allocator_feedback_valid(marked)
    assert freeze_integral(*args, marked, enabled=True)


def test_immutable_issue_and_eviction_tombstone():
  _, f, reg = sample()
  reg.issue(7, 10, 1.0, 0.09, True, SOURCE, 990_000_000)
  assert not check(reg, f)
  reg.issue(7, 11, 1.2, 0.08)
  reg.issue(7, 10, 1.0, 0.08, True, SOURCE, 990_000_000)
  assert not check(reg, f)


def test_registry_overflow_fails_closed():
  _, f, reg = sample()
  for seq in range(11, 76):
    reg.issue(7, seq, 1.0, 0.08)
  assert len(reg.issued) <= 64
  assert reg.overflow
  assert not check(reg, f)


def test_constant_space_episodes_and_nonrenewal():
  r, f, reg = sample()
  assert check(reg, f)
  f = replace(f, reason=replace(r, budget_remaining=0.01).encode())
  assert not check(reg, f)
  assert len(reg.episodes) == 1
  f = replace(f, reason=replace(r, episode="alternate").encode())
  assert not check(reg, f)


@pytest.mark.parametrize(
  "change", [dict(model_id="other"), dict(source="other"), dict(target_ns=980_000_000), dict(actual_gas=0.051), dict(actual_limited=False), dict(sequence=True)]
)
def test_binding_conflicts(change):
  r, f, reg = sample()
  assert not check(reg, replace(f, reason=replace(r, **change).encode()))


def test_oversized_and_duplicate_json_decline():
  r, _, _ = sample()
  assert ReferenceReceipt.parse("bounded-release-reference-v2:" + " " * 8192) is None
  assert ReferenceReceipt.parse(r.encode().replace('"epoch":7', '"epoch":7,"epoch":7', 1)) is None
  with pytest.raises(ValueError):
    replace(r, episode="x" * 9000).encode()
  assert ReleaseReceipt.parse("bounded-release-owner-v1:[]") is None


def test_float32_does_not_relax_nominal_binding():
  r, f, reg = sample()
  n = dict(r.nominal, requested=0.07)
  assert not check(reg, replace(f, reason=replace(r, nominal=n).encode()))


def test_epoch_change_and_reset_invalidate_old_authority():
  _, f, reg = sample()
  reg.issue(8, 1, 1.01, 0.08)
  assert not check(reg, f)
  reg.issue(7, 10, 1.0, 0.08, True, SOURCE, 990_000_000)
  assert not check(reg, f)


@pytest.mark.parametrize("change", [dict(selected_ns=1_040_000_000), dict(until_ns=1_400_000_000), dict(budget_remaining=0), dict(budget_remaining=0.0101)])
def test_lease_boundaries(change):
  r, f, reg = sample()
  assert not check(reg, replace(f, reason=replace(r, **change).encode()))


def test_real_capnp_round_trip():
  from opendbc.car import structs
  from opendbc.car.ford.longitudinal_coordination import feedback_from_wire

  r, f, reg = sample()
  out = structs.CarControl.Actuators.new_message()
  q = out.navigatorLongitudinal
  q.producerMonoTime = r.selected_ns
  for field in ("epoch", "sequence", "requested", "gas", "brake", "limited", "valid", "fault", "precharge", "reason"):
    setattr(q, field, getattr(f, field))
  q.allocatorVersion = f.allocator_version
  q.allocatorPhase = f.allocator_phase
  with structs.CarControl.Actuators.from_bytes(out.to_bytes()) as packet:
    wire = feedback_from_wire(packet.navigatorLongitudinal)
    assert check(reg, wire)


@pytest.mark.parametrize("field", ["actual_gas", "budget_remaining", "grade_offset"])
def test_oversized_integer_envelope_declines(field):
  r, f, reg = sample()
  reason = replace(r, **{field: 10**500}).encode()
  assert ReferenceReceipt.parse(reason) is None
  assert not check(reg, replace(f, reason=reason))


@pytest.mark.parametrize("field", ["t", "requested", "gas", "precharge_age", "budget_remaining"])
def test_oversized_integer_nominal_declines(field):
  r, f, reg = sample()
  reason = replace(r, nominal=dict(r.nominal, **{field: 10**500})).encode()
  assert not check(reg, replace(f, reason=reason))


@pytest.mark.parametrize("field", ["actual_gas", "budget_remaining", "grade_offset"])
def test_oversized_integer_integral_declines_without_exception(field):
  from opendbc.car.ford.longitudinal_reference_feedback import ReferenceIntegral, ReferenceContext

  r, f, _ = sample()
  policy = ReferenceIntegral()
  policy.begin(ReferenceContext(1_030_000_000, 7, 11, True, 0.0, True))
  policy.issue(7, 10, 1.0, 0.08, True, SOURCE, 990_000_000)
  rc = NS(valid=True, sample=(1.03, 7, 11))
  malformed = replace(f, reason=replace(r, **{field: 10**500}).encode())
  assert policy.nominal(rc, 1.03, 7, 11, 0.2, 0.08, 0.3, 27.0, "pid", malformed, enabled=True)
  assert not policy.diagnostic["owned"]


@pytest.mark.parametrize("field", ["t", "requested", "gas", "precharge_age", "budget_remaining"])
def test_oversized_integer_integral_nominal_declines(field):
  from opendbc.car.ford.longitudinal_reference_feedback import ReferenceIntegral, ReferenceContext

  r, f, _ = sample()
  policy = ReferenceIntegral()
  policy.begin(ReferenceContext(1_030_000_000, 7, 11, True, 0.0, True))
  policy.issue(7, 10, 1.0, 0.08, True, SOURCE, 990_000_000)
  rc = NS(valid=True, sample=(1.03, 7, 11))
  malformed = replace(f, reason=replace(r, nominal=dict(r.nominal, **{field: 10**500})).encode())
  assert policy.nominal(rc, 1.03, 7, 11, 0.2, 0.08, 0.3, 27.0, "pid", malformed, enabled=True)
  assert not policy.diagnostic["owned"]


def test_finite_clock_conversion_overflow_declines():
  _, f, reg = sample()
  reg.issue(7, 11, 1e308, 0.08)
  assert (7, 11) not in reg.issued
  assert not reg.verify(f, now=1e308, epoch=7, sequence=11, target=0.08, error=0.2, wanted=0.3, speed=27.0, eligible=True)
