from types import SimpleNamespace as NS
from opendbc.car.ford.longitudinal_retention import ExtraRetentionOwner, OwnerReceipt, TAG


class TestOwner:
  def make(self):
    return ExtraRetentionOwner()

  def step(
    self, x, t, ref=False, qualified=False, clear=False, actual=0.2, target=0.1, previous=True, net=0.01, eligible=True, owner=True, epoch=1, sequence=None
  ):
    return x.update(
      t + 1,
      owner,
      True,
      eligible,
      ref,
      False,
      qualified,
      previous,
      clear,
      0.2,
      net,
      actual,
      target,
      epoch,
      round(t * 1000) + 1 if sequence is None else sequence,
      source_ns=round((t + 1) * 1000000000.0),
    )

  def test_first_owner_keeps_reference_false_actual_true(self):
    x = self.make()
    assert self.step(x, 0.0, qualified=True, net=0.03)
    assert x.holding
    assert self.step(x, 0.02, ref=False, qualified=False, net=0.01)
    assert x.holding
    assert x.until_ns == 1100000000

  def test_new_genuine_reference_brake_cancels_without_refund(self):
    x = self.make()
    self.step(x, 0.0, qualified=True)
    assert not self.step(x, 0.02, ref=True, net=-0.01)
    assert x.spent
    assert not self.step(x, 0.04, qualified=True)
    assert x.spent

  def test_expiry_never_rearms_on_sign_toggle(self):
    x = self.make()
    for i in range(10000):
      self.step(x, i * 0.02, qualified=True, actual=0.09 if i % 2 else 0.1000001)
    assert x.spent
    assert not x.holding
    assert x.until_ns == 1100000000

  def test_same_observer_clear_and_nominal_false_only(self):
    x = self.make()
    self.step(x, 0.0, qualified=True)
    self.step(x, 0.1)
    self.step(x, 0.12, ref=True, clear=True)
    assert x.spent
    self.step(x, 0.14, ref=False, clear=False)
    assert x.spent
    self.step(x, 0.16, ref=False, clear=True, qualified=False)
    assert not x.spent

  def test_requires_actual_previous_braking_and_initial_qualified_release(self):
    assert not self.step(self.make(), 0.0, qualified=False)
    assert not self.step(self.make(), 0.0, qualified=True, previous=False)

  def test_native_release_negative_scope_stale_owner_escape(self):
    for kwargs in [dict(net=0.31), dict(target=-0.5), dict(eligible=False), dict(epoch=2), dict(sequence=0)]:
      x = self.make()
      self.step(x, 0.0, qualified=True)
      assert not self.step(x, 0.02, **kwargs)
      assert x.spent

  def test_tracking_dwell_accumulates_before_nominal_release(self):
    x = self.make()
    for t in [0.0, 0.02, 0.04, 0.06]:
      self.step(x, t, ref=True, actual=0.05, target=0.1)
    assert not self.step(x, 0.08, qualified=True, actual=0.05, target=0.1)

  def test_cached_measurement_cannot_earn_dwell(self):
    x = self.make()
    for t in [0.0, 0.02, 0.04, 0.06]:
      x.update(t + 1, True, True, True, True, False, False, True, False, 0.2, 0.01, 0.05, 0.1, 1, round(t * 1000) + 1, source_ns=1000000000)
    assert x.update(1.08, True, True, True, False, False, True, True, False, 0.2, 0.03, 0.05, 0.1, 1, 81, source_ns=1000000000)

  def test_invalid_missing_future_source_fail_closed(self):
    for t, source in [(float('nan'), 1), (float('inf'), 1), (1.0, None), (1.0, 1000000001), (1.0, float('nan'))]:
      assert not self.make().update(t, True, True, True, False, False, True, True, False, 0.2, 0.03, 0.2, 0.1, 1, 1, source_ns=source)

  def test_invalid_receipt_times_and_missing_fields(self):
    for t in [float('nan'), float('inf')]:
      assert not OwnerReceipt().freeze(t, 1, 2, 0.1, 0.1, 0.2, 27.0, 'pid', self.feedback(), True, True)
      assert not OwnerReceipt().freeze(1.02, 1, 2, 0.1, 0.1, 0.2, 27.0, 'pid', self.feedback(t=t), True, True)
    assert not OwnerReceipt().freeze(1.02, 1, 2, 0.1, 0.1, 0.2, 27.0, 'pid', NS(), True, True)

  def test_equal_key_conflict_cannot_revive_withdrawal(self):
    r = OwnerReceipt()
    f = self.feedback(reason='actual_brake')
    assert not r.freeze(1.02, 1, 2, 0.1, 0.1, 0.2, 27.0, 'pid', f, True, True)
    assert not r.freeze(1.02, 1, 2, 0.1, 0.1, 0.2, 27.0, 'pid', self.feedback(), True, True)

  def test_sequence_timestamp_monotonic_and_future_epoch_not_poison(self):
    r = OwnerReceipt()
    assert r.freeze(1.02, 1, 3, 0.1, 0.1, 0.2, 27.0, 'pid', self.feedback(seq=2), True, True)
    assert not r.freeze(1.04, 1, 3, 0.1, 0.1, 0.2, 27.0, 'pid', self.feedback(t=1.03, seq=1), True, True)
    future = self.feedback(t=1.03, seq=3)
    future.epoch = 2
    assert not r.freeze(1.04, 1, 3, 0.1, 0.1, 0.2, 27.0, 'pid', future, True, True)
    assert r.freeze(1.05, 1, 3, 0.1, 0.1, 0.2, 27.0, 'pid', self.feedback(t=1.04, seq=3), True, True)

  def test_schedule_gap_scope_exit_and_negative_demand_abort(self):
    for t, kw in [(0.2, {}), (0.02, dict(eligible=False)), (0.02, dict(net=-0.01)), (0.02, dict(target=-0.5))]:
      x = self.make()
      self.step(x, 0.0, qualified=True)
      assert not self.step(x, t, **kw)
      assert x.spent

  def test_speed_boundary_phase_and_negative_recovery_live(self):
    for speed, phase, error in [(40 * 0.44704 - 1e-06, 'pid', 0.1), (27.0, 'stopping', 0.1), (27.0, 'starting', 0.1), (27.0, 'pid', -0.1)]:
      assert not OwnerReceipt().freeze(1.02, 1, 2, error, 0.1, 0.2, speed, phase, self.feedback(), True, True)
    assert OwnerReceipt().freeze(1.02, 1, 2, 0.1, 0.1, 0.2, 40 * 0.44704, 'pid', self.feedback(), True, True)

  def test_inactive_and_override_do_not_refund_live_lease(self):
    x = self.make()
    self.step(x, 0.0, qualified=True)
    assert not x.update(1.02, True, False, True, False, False, True, True, False, 0.2, 0.03, 0.2, 0.1, 1, 21, source_ns=1020000000)
    assert x.spent
    assert not self.step(x, 0.04, qualified=True)

  def test_receipt_continuity_after_decision_expiry_until_freshness(self):
    r = OwnerReceipt()
    f = self.feedback(t=1.05, reason=TAG + '|until_ns=1100000000')
    assert r.freeze(1.1, 1, 2, 0.1, 0.1, 0.2, 27.0, 'pid', f, True, True)
    assert r.freeze(1.149999, 1, 2, 0.1, 0.1, 0.2, 27.0, 'pid', f, True, True)
    assert not r.freeze(1.15, 1, 2, 0.1, 0.1, 0.2, 27.0, 'pid', f, True, True)

  def test_continuity_duplicate_cannot_renew_and_absolute_cap(self):
    r = OwnerReceipt()
    f = self.feedback(t=1.099999, reason=TAG + '|until_ns=1100000000')
    assert r.freeze(1.19, 1, 2, 0.1, 0.1, 0.2, 27.0, 'pid', f, True, True)
    assert not r.freeze(1.199999, 1, 2, 0.1, 0.1, 0.2, 27.0, 'pid', f, True, True)
    assert not r.freeze(1.2, 1, 2, 0.1, 0.1, 0.2, 27.0, 'pid', f, True, True)

  def test_continuity_withdrawal_and_signed_recovery(self):
    r = OwnerReceipt()
    f = self.feedback(t=1.05, reason=TAG + '|until_ns=1100000000', seq=2)
    assert not r.freeze(1.11, 1, 3, -0.1, 0.1, 0.2, 27.0, 'pid', f, True, True)
    assert not r.freeze(1.11, 1, 3, 0.1, -0.1, 0.2, 27.0, 'pid', f, True, True)
    assert r.freeze(1.11, 1, 3, 0.1, 0.1, 0.2, 27.0, 'pid', f, True, True)
    assert not r.freeze(1.12, 1, 3, 0.1, 0.1, 0.2, 27.0, 'pid', self.feedback(t=1.115, reason='wait_observed_clear', seq=3), True, True)
    assert not r.freeze(1.13, 1, 4, 0.1, 0.1, 0.2, 27.0, 'pid', f, True, True)

  def test_active_receipt_lease_end_immutable_until_withdrawal(self):
    r = OwnerReceipt()
    f = self.feedback(t=1.05, reason=TAG + '|until_ns=1100000000', seq=2)
    assert r.freeze(1.1, 1, 2, 0.1, 0.1, 0.2, 27.0, 'pid', f, True, True)
    changed = self.feedback(t=1.08, reason=TAG + '|until_ns=1180000000', seq=3)
    assert not r.freeze(1.11, 1, 3, 0.1, 0.1, 0.2, 27.0, 'pid', changed, True, True)
    normal = self.feedback(t=1.12, reason='wait_observed_clear', seq=4)
    assert not r.freeze(1.13, 1, 4, 0.1, 0.1, 0.2, 27.0, 'pid', normal, True, True)
    new = self.feedback(t=1.14, reason=TAG + '|until_ns=1240000000', seq=5)
    assert r.freeze(1.15, 1, 5, 0.1, 0.1, 0.2, 27.0, 'pid', new, True, True)

  def test_cross_component_fixed_consumer_lease_receipt_and_withdrawal(self):
    x = self.make()
    r = OwnerReceipt()
    end = None
    for n in range(7):
      t = 1 + n * 0.02
      held = x.update(t, True, True, True, False, False, n == 0, True, False, 0.2, 0.03, 0.2, 0.1, 1, n + 1, source_ns=round(t * 1000000000.0))
      if n == 0:
        end = x.until_ns
      assert x.until_ns == end
      reason = TAG + '|until_ns=' + str(end) if held else 'wait_observed_clear'
      f = self.feedback(t=t, reason=reason, seq=n + 1)
      f.brake = held
      f.precharge = held
      f.allocator_phase = 'brake' if held else 'release'
      assert r.freeze(t + 0.005, 1, n + 1, 0.1, 0.1, 0.2, 27.0, 'pid', f, True, True) == held
      assert held == (n < 5)
    assert end == 1100000000
    assert x.spent

  def test_original_release_clock_not_part_of_owner(self):
    x = self.make()
    self.step(x, 0.0, qualified=True)
    assert not hasattr(x, 'since')
    assert not hasattr(x, 'area')

  def feedback(self, t=1.0, reason=None, seq=1):
    return NS(
      t=t,
      epoch=1,
      sequence=seq,
      requested=0.2,
      gas=-5.0,
      brake=True,
      limited=False,
      valid=True,
      fault=False,
      allocator_version=3,
      precharge=True,
      reference_brake=False,
      allocator_phase='brake',
      precharge_age=0.0,
      budget_remaining=0.0,
      reason=reason or TAG + '|until_ns=1100000000',
    )

  def test_receipt_positive_only_before_expiry(self):
    r = OwnerReceipt()
    f = self.feedback()
    assert r.freeze(1.02, 1, 2, 0.1, 0.1, 0.2, 27.0, 'pid', f, True, True)
    assert not r.freeze(1.1, 1, 2, 0.1, 0.1, 0.2, 27.0, 'pid', f, True, True)
    assert not r.freeze(1.02, 1, 2, -0.1, 0.1, 0.2, 27.0, 'pid', f, True, True)

  def test_explicit_withdrawal_rejects_delayed_old_receipt(self):
    r = OwnerReceipt()
    f = self.feedback()
    assert r.freeze(1.02, 1, 2, 0.1, 0.1, 0.2, 27.0, 'pid', f, True, True)
    withdraw = self.feedback(t=1.03, reason='actual_brake', seq=2)
    assert not r.freeze(1.04, 1, 3, 0.1, 0.1, 0.2, 27.0, 'pid', withdraw, True, True)
    assert not r.freeze(1.05, 1, 4, 0.1, 0.1, 0.2, 27.0, 'pid', f, True, True)

  def test_malformed_unknown_stale_and_negative_target(self):
    for reason in ['actual_brake', TAG, TAG + '|until_ns=x', TAG + '|until_ns=1200000000']:
      assert not OwnerReceipt().freeze(1.02, 1, 2, 0.1, 0.1, 0.2, 27.0, 'pid', self.feedback(reason=reason), True, True)
    assert not OwnerReceipt().freeze(1.02, 1, 2, 0.1, -0.1, 0.2, 27.0, 'pid', self.feedback(), True, True)
