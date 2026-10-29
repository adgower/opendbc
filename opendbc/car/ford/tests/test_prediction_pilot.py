import unittest
from opendbc.car.ford.prediction_pilot import Runner, Request, Context

SOURCE = "source-fixture"


def context(t, **kw):
  d = dict(
    source_id=SOURCE,
    alpha=True,
    enabled=True,
    long_active=True,
    phase="pid",
    speed_ms=5.0,
    can_valid=True,
    control_valid=True,
    source_valid=True,
    pose_valid=True,
    reference_valid=True,
    request_valid=True,
    epoch=1,
    sequence=round(t * 100) + 1,
    producer_ns=round(t * 1e9),
    state_ns=round(t * 1e9),
    pose_ns=round(t * 1e9),
    target_ns=round(t * 1e9),
    health_valid=True,
    controls_allowed=True,
    selfdrive_active=True,
    mismatch=False,
    fault=False,
    in_drive=True,
    steering_override=False,
    gas_pedal=False,
    brake_pedal=False,
    observer_clear=True,
    torque_valid=True,
    torque_ns=round(t * 1e9),
    torque=0.0,
  )
  d.update(kw)
  return Context(**d)


def packet(gas=525, pred=0, brake=False):
  d = bytearray(8)
  d[0] = 20
  d[1] = 8
  d[2] = pred >> 8
  d[3] = pred & 255
  d[6] = 4 | (gas >> 8) | (128 if brake else 0)
  d[7] = gas & 255
  return (390, bytes(d), 0)


def request(condition="B", **kw):
  d = dict(trial_id="trial-1", condition=condition, source_id=SOURCE, duration_s=0.3, wait_s=1.0)
  d.update(kw)
  return Request(**d)


def active(condition="B"):
  r = Runner(enabled=True)
  r.arm(request(condition), 1.0)
  for i in range(7):
    r.process(1.0 + i * 0.02, context(1.0 + i * 0.02), packet())
  return r


class PilotContracts(unittest.TestCase):
  def test_disabled_default_is_inert(self):
    r = Runner()
    self.assertEqual(r.process(1.0, context(1.0), packet()), packet())
    with self.assertRaises(ValueError):
      r.arm(request(), 1.0)

  def test_b_only_prediction_bits(self):
    r = active()
    self.assertEqual(r.state, "active")
    p = packet()
    q = r.process(1.14, context(1.14), p)
    self.assertNotEqual(p, q)
    mask = bytes([0, 0, 3, 255, 0, 0, 0, 0])
    self.assertTrue(all(((a ^ b) & (~m & 255)) == 0 for a, b, m in zip(p[1], q[1], mask, strict=False)))
    self.assertEqual(((q[1][2] & 3) << 8) | q[1][3], 525)

  def test_a_tracks_trial_but_never_changes_packet(self):
    r = active("A")
    self.assertEqual(r.state, "active")
    self.assertEqual(r.process(1.14, context(1.14), packet()), packet())
    self.assertTrue(any(e["event"] == "start" for e in r.events))

  def test_duration_and_wait_are_finite_bounded(self):
    for kw in [
      dict(duration_s=0),
      dict(duration_s=5.01),
      dict(wait_s=30.01),
      dict(duration_s=float("nan")),
      dict(wait_s=float("inf")),
      dict(condition="C"),
      dict(trial_id=""),
      dict(source_id=""),
    ]:
      with self.subTest(kw=kw), self.assertRaises(ValueError):
        Runner(enabled=True).arm(request(**kw), 1.0)

  def test_lease_expires_before_transformation_and_does_not_resume(self):
    r = active()
    deadline = r.active_deadline
    for i in range(7, 21):
      t = 1 + i * 0.02
      out = r.process(t, context(t), packet())
    self.assertEqual(out, packet())
    self.assertEqual(r.state, "finished")
    self.assertEqual(r.process(deadline + 0.02, context(deadline + 0.02), packet()), packet())

  def test_wait_qualification_never_transforms_bad_context(self):
    r = Runner(enabled=True)
    r.arm(request(), 1.0)
    for i in range(5):
      self.assertEqual(r.process(1 + i * 0.02, context(1 + i * 0.02, enabled=False, long_active=False, brake_pedal=True), packet()), packet())
    self.assertEqual(r.state, "waiting")

  def test_wait_timeout(self):
    r = Runner(enabled=True)
    r.arm(request(wait_s=0.2), 1.0)
    self.assertEqual(r.process(1.2, context(1.2), packet()), packet())
    self.assertEqual(r.state, "finished")

  def test_actual_speed_ceiling_and_floor(self):
    r = active()
    self.assertEqual(r.process(1.14, context(1.14, speed_ms=8.9408001), packet()), packet())
    self.assertEqual(r.state, "cancelled")
    for speed in [2.99, 0.0, 15.0, 20.0, 38.0]:
      r = active()
      self.assertEqual(r.process(1.14, context(1.14, speed_ms=speed), packet()), packet())
      self.assertEqual(r.state, "cancelled")

  def test_active_guard_loss_latches(self):
    for key, value in [
      ("alpha", False),
      ("enabled", False),
      ("long_active", False),
      ("phase", "starting"),
      ("can_valid", False),
      ("control_valid", False),
      ("source_valid", False),
      ("pose_valid", False),
      ("reference_valid", False),
      ("request_valid", False),
      ("health_valid", False),
      ("controls_allowed", False),
      ("selfdrive_active", False),
      ("in_drive", False),
      ("steering_override", True),
      ("gas_pedal", True),
      ("brake_pedal", True),
      ("mismatch", True),
      ("fault", True),
      ("observer_clear", False),
      ("torque_valid", False),
      ("torque", 8.0),
      ("torque", -1.0),
      ("speed_ms", float("nan")),
    ]:
      with self.subTest(key=key):
        r = active()
        self.assertEqual(r.process(1.14, context(1.14, **{key: value}), packet()), packet())
        self.assertEqual(r.state, "cancelled")
        self.assertEqual(r.process(1.16, context(1.16), packet()), packet())

  def test_source_and_epoch_changes_cancel(self):
    for kw in [dict(source_id="other"), dict(epoch=2)]:
      r = active()
      self.assertEqual(r.process(1.14, context(1.14, **kw), packet()), packet())
      self.assertEqual(r.state, "cancelled")

  def test_stale_future_and_missing_timestamps_cancel(self):
    for field in ["producer_ns", "state_ns", "pose_ns", "target_ns", "torque_ns"]:
      for value in [0, round(1.0e9), round(1.2e9)]:
        r = active()
        self.assertEqual(r.process(1.14, context(1.14, **{field: value}), packet()), packet())
        self.assertEqual(r.state, "cancelled")

  def test_repeated_callback_clock_gap_and_reverse_cancel(self):
    for t in [1.12, 1.1, 1.3, float("nan")]:
      r = active()
      self.assertEqual(r.process(t, context(1.14), packet()), packet())
      self.assertEqual(r.state, "cancelled")

  def test_sequence_repetition_allowed_only_same_stamp(self):
    r = active()
    c = context(1.14, sequence=113, producer_ns=round(1.12e9), target_ns=round(1.12e9))
    self.assertNotEqual(r.process(1.14, c, packet()), packet())
    r = active()
    self.assertEqual(r.process(1.14, context(1.14, sequence=113), packet()), packet())
    self.assertEqual(r.state, "cancelled")

  def test_sequence_rollback_or_time_rollback_cancel(self):
    for kw in [dict(sequence=100), dict(sequence=115, producer_ns=round(1.1e9))]:
      r = active()
      self.assertEqual(r.process(1.14, context(1.14, **kw), packet()), packet())
      self.assertEqual(r.state, "cancelled")

  def test_busy_and_reused_request_are_rejected(self):
    r = active()
    with self.assertRaises(ValueError):
      r.arm(request(trial_id="trial-2"), 1.14)
    r.cancel("operator_cancel", 1.14)
    with self.assertRaises(ValueError):
      r.arm(request(), 1.16)
    r.arm(request(trial_id="trial-2"), 1.16)
    self.assertEqual(r.state, "waiting")

  def test_no_runtime_permission_survives_restart(self):
    active()
    new = Runner()
    self.assertEqual(new.process(1.14, context(1.14), packet()), packet())
    self.assertEqual(new.state, "disabled")

  def test_invalid_baseline_packet_never_modified(self):
    variants = [
      packet(gas=0),
      packet(gas=449),
      packet(gas=701),
      packet(brake=True),
      packet(pred=525),
      (391, packet()[1], 0),
      (390, packet()[1], 2),
      (390, b"bad", 0),
    ]
    for p in variants:
      r = active()
      self.assertEqual(r.process(1.14, context(1.14), p), p)
      self.assertEqual(r.state, "cancelled")

  def test_wait_fault_source_owner_and_overspeed_cancel(self):
    for kw in [dict(fault=True), dict(mismatch=True), dict(source_id="bad"), dict(alpha=False), dict(speed_ms=9.0)]:
      r = Runner(enabled=True)
      r.arm(request(), 1.0)
      self.assertEqual(r.process(1.02, context(1.02, **kw), packet()), packet())
      self.assertEqual(r.state, "cancelled")

  def test_brief_readiness_does_not_count_as_qualification(self):
    r = Runner(enabled=True)
    r.arm(request(), 1.0)
    for t in [1.0, 1.02, 1.04]:
      self.assertEqual(r.process(t, context(t), packet()), packet())
    r.process(1.06, context(1.06, long_active=False), packet())
    self.assertEqual(r.process(1.08, context(1.08), packet()), packet())
    self.assertEqual(r.state, "waiting")


class MoreContracts(unittest.TestCase):
  def test_unexpected_baseline_policy_aborts_even_before_start(self):
    r = Runner(enabled=True)
    r.arm(request(), 1.0)
    r.process(1.0, context(1.0), packet(pred=525))
    self.assertEqual(r.state, "cancelled")

  def test_exact_speed_boundary_is_allowed(self):
    r = active()
    self.assertNotEqual(r.process(1.14, context(1.14, speed_ms=20 * 0.44704), packet()), packet())


class EndAccounting(unittest.TestCase):
  def test_gap_beyond_deadline_is_retained_as_abort(self):
    r = active()
    self.assertEqual(r.process(1.5, context(1.5), packet()), packet())
    self.assertEqual(r.state, "cancelled")

  def test_fault_at_deadline_is_not_successful_completion(self):
    r = active()
    for i in range(7, 20):
      r.process(1 + i * 0.02, context(1 + i * 0.02), packet())
    r.process(1.4, context(1.4, fault=True), packet())
    self.assertEqual(r.state, "cancelled")


if __name__ == "__main__":
  unittest.main()
