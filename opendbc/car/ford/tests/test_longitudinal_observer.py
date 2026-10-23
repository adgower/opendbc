import math
import unittest
from opendbc.car.ford.longitudinal_coordination import Torque, Feedback, ReleaseBarrier, freeze_integral


class Contracts(unittest.TestCase):
  def feedback(self, **kw):
    d = dict(t=1.0, epoch=2, sequence=10, requested=0.3, gas=0.1, brake=False, limited=True)
    d.update(kw)
    return Feedback(**d)

  def freeze(self, **kw):
    d = dict(now=1.02, epoch=2, sequence=11, error=0.2, target=0.0, wanted=0.3, speed=15.0, phase="pid", feedback=self.feedback(), enabled=True)
    d.update(kw)
    return freeze_integral(**d)

  def release(self, b, t, **kw):
    d = dict(now=t, active=True, brake=False, speed=10.0, phase="pid", torque=Torque(t, 0.0))
    d.update(kw)
    return b.update(**d)

  def start(self):
    b = ReleaseBarrier(True)
    self.release(b, 0.0, brake=True, torque=Torque(0.0, 100.0))
    return b

  def test_positive_integral_growth_into_deliberate_cap_is_frozen(self):
    self.assertTrue(self.freeze())

  def test_negative_error_can_unwind_during_cap(self):
    self.assertFalse(self.freeze(error=-0.1))

  def test_genuine_braking_intent_can_integrate(self):
    self.assertFalse(self.freeze(target=-0.2))

  def test_actual_braking_is_not_frozen(self):
    self.assertFalse(self.freeze(feedback=self.feedback(brake=True)))

  def test_no_integral_reset_or_freeze_when_not_limited(self):
    self.assertFalse(self.freeze(feedback=self.feedback(limited=False)))

  def test_default_off_policy_is_parity(self):
    self.assertFalse(self.freeze(enabled=False))

  def test_stale_future_foreign_epoch_or_future_sequence_is_ignored(self):
    for fb in [
      self.feedback(t=0.8),
      self.feedback(t=1.03),
      self.feedback(epoch=1),
      self.feedback(sequence=12),
      self.feedback(valid=False),
      self.feedback(fault=True),
      self.feedback(gas=math.nan),
    ]:
      self.assertFalse(self.freeze(feedback=fb))

  def test_stop_restart_creep_policy_is_parity(self):
    for kw in [dict(speed=2.99), dict(phase="stopping"), dict(phase="starting")]:
      self.assertFalse(self.freeze(**kw))

  def test_packet_roundtrip_is_atomic(self):
    self.assertEqual(Feedback.decode(self.feedback().encode()), self.feedback())

  def test_nonfinite_packet_cannot_cross_transport(self):
    with self.assertRaises(ValueError):
      self.feedback(gas=math.nan).encode()

  def test_release_waits_for_observed_torque_clear(self):
    b = self.start()
    self.assertFalse(self.release(b, 0.02, torque=Torque(0.02, 80.0)).allow)

  def test_cached_clear_sample_cannot_qualify(self):
    b = self.start()
    for t in [0.02, 0.04, 0.06]:
      self.assertFalse(self.release(b, t, torque=Torque(0.02, 0.0)).allow)

  def test_three_distinct_clear_samples_and_dwell_qualify(self):
    b = self.start()
    for t in [0.02, 0.04]:
      self.assertFalse(self.release(b, t).allow)
    self.assertTrue(self.release(b, 0.06).allow)

  def test_high_torque_resets_clear_dwell(self):
    b = self.start()
    self.release(b, 0.02)
    self.release(b, 0.04, torque=Torque(0.04, 8.0))
    for t in [0.06, 0.08]:
      self.assertFalse(self.release(b, t).allow)
    self.assertTrue(self.release(b, 0.1).allow)

  def test_future_stale_fault_code_and_invalid_torque_fail_closed(self):
    for tq in [Torque(0.03, 0.0), Torque(-0.2, 0.0), Torque(0.02, 32760.0), Torque(0.02, math.nan), Torque(0.02, 0.0, False)]:
      b = self.start()
      r = self.release(b, 0.02, torque=tq)
      self.assertTrue(r.fault)
      self.assertFalse(r.allow)

  def test_timeout_never_becomes_blind_permission(self):
    b = self.start()
    for i in range(1, 57):
      r = self.release(b, i * 0.02, torque=Torque(i * 0.02, 80.0))
    self.assertTrue(r.fault)
    self.assertFalse(r.allow)

  def test_fault_requires_off_before_rearming(self):
    b = self.start()
    self.release(b, 0.02, torque=Torque(-1.0, 0.0))
    self.assertTrue(self.release(b, 0.04).fault)
    self.release(b, 0.06, active=False)
    self.assertFalse(self.release(b, 0.08).fault)

  def test_low_speed_restart_is_unchanged(self):
    b = self.start()
    self.assertTrue(self.release(b, 0.02, speed=0.0, torque=None).allow)

  def test_default_off_release_is_parity(self):
    b = ReleaseBarrier()
    self.release(b, 0.0, brake=True)
    self.assertTrue(self.release(b, 0.02).allow)

  def test_out_of_order_torque_cannot_rearm_release(self):
    b = self.start()
    self.release(b, 0.02)
    self.assertTrue(self.release(b, 0.04, torque=Torque(0.01, 0.0)).fault)

  def test_observed_seven_tenths_release_tail_never_grants_blind_propulsion(self):
    b = self.start()
    for i in range(1, 36):
      r = self.release(b, i * 0.02, torque=Torque(i * 0.02, 80.0))
      self.assertFalse(r.allow)
      self.assertFalse(r.fault)
    for t in [0.72, 0.74]:
      self.assertFalse(self.release(b, t).allow)
    self.assertTrue(self.release(b, 0.76).allow)


if __name__ == "__main__":
  unittest.main()


class TestIntegratedController(unittest.TestCase):
  def harness(self, enabled=True, speed=15.0):
    from opendbc.car.ford.tests.test_longitudinal_coordination import LongitudinalHarness
    from opendbc.car.ford.longitudinal_coordination import Torque

    h = LongitudinalHarness(speed=speed)
    h.controller.longitudinal_coordination_enabled = enabled
    h.controller.coordination_control_valid = True
    h.state.out.canValid = True
    update = h.controller.update
    state = {"epoch": 1, "pose": True, "torque": True, "delay": 0.152, "changed": None, "brake": False}

    def wrapped(command, CS, now_ns):
      CC = command.as_builder()
      req = CC.navigatorLongitudinal
      req.enabled = enabled
      req.epoch = state["epoch"]
      req.sequence = h.controller.frame + 1
      req.producerMonoTime = max(0, now_ns + state.get("control_shift_ns", 0))
      req.poseMonoTime = max(0, now_ns + state.get("pose_shift_ns", 0))
      req.poseValid = state["pose"]
      req.referenceValid = True
      now = now_ns / 1e9
      high = state["brake"] or (state["changed"] is not None and now - state["changed"] < state["delay"])
      CS.longitudinal_brake_torque = Torque(now, 160.0 if high else 0.0, state["torque"])
      result = update(CC.as_reader(), CS, now_ns)
      if h.controller.brake_request != state["brake"]:
        state["brake"] = h.controller.brake_request
        state["changed"] = now
      return result

    h.controller.update = wrapped
    return h, state

  def test_observed_release_then_bounded_resume(self):
    h, c = self.harness()
    h.step(-0.5, frames=100)
    v = h.step(0.7, frames=6)
    self.assertEqual(v["AccPrpl_A_Rq"], -5.0)
    self.assertEqual(h.controller.last_observer.reason, "wait_observed_clear")
    v = h.step(0.7, frames=60)
    self.assertGreater(v["AccPrpl_A_Rq"], 0.0)
    self.assertFalse(h.controller.coordination_fault)
    self.assertTrue(h.output.navigatorLongitudinal.valid)

  def test_fault_survives_override_and_epoch_change(self):
    h, c = self.harness()
    h.step(-0.5, frames=100)
    c["torque"] = False
    v = h.step(0.7, frames=6)
    self.assertEqual(v["Cmbb_B_Enbl"], 0)
    self.assertEqual(v["AccPrpl_A_Rq"], -5.0)
    self.assertAlmostEqual(v["AccBrkTot_A_Rq"], 0.0, delta=0.0039)
    self.assertTrue(h.output.navigatorLongitudinal.fault)
    c["torque"] = True
    h.step(0.0, active=False, frames=4)  # override is not owner disable
    c["epoch"] += 1
    self.assertEqual(h.step(0.7, frames=10)["Cmbb_B_Enbl"], 0)
    h.command.enabled = False
    h.step(0.0, active=False, frames=4)
    h.command.enabled = True
    c["epoch"] += 1
    self.assertEqual(h.step(0.7, frames=20)["Cmbb_B_Enbl"], 1)

  def test_pose_fault_latches_and_urgent_brake_preserves_onset(self):
    h, c = self.harness()
    h.step(0.7, frames=50)
    c["pose"] = False
    self.assertEqual(h.step(0.3, frames=4)["Cmbb_B_Enbl"], 0)
    c["pose"] = True
    self.assertEqual(h.step(0.3, frames=10)["Cmbb_B_Enbl"], 0)
    a, _ = self.harness()
    b, _ = self.harness(False)
    a.step(0.7, frames=4000)
    b.step(0.7, frames=4000)
    for _ in range(30):
      x = a.step(-2.0, frames=2)
      y = b.step(-2.0, frames=2)
      if y["AccBrkDecel_B_Rq"]:
        self.assertEqual(x["AccBrkDecel_B_Rq"], 1)
      self.assertEqual(x["AccPrpl_A_Rq"], -5.0)

  def test_stop_creep_packet_parity(self):
    a, _ = self.harness(speed=0.0)
    b, _ = self.harness(False, speed=0.0)
    for demand, phase in [(-0.2, "stopping"), (0.2, "pid"), (0.7, "pid"), (-0.2, "stopping")]:
      self.assertEqual(a.step(demand, phase, frames=100), b.step(demand, phase, frames=100))

  def test_override_does_not_erase_pending_physical_release(self):
    h, c = self.harness()
    c['delay'] = .7
    h.step(-.5, frames=100)
    h.step(0., active=False, frames=4)
    self.assertEqual(h.step(.7, frames=8)['AccPrpl_A_Rq'], -5.)
    self.assertTrue(h.controller.release_barrier.needs)
    self.assertFalse(h.controller.coordination_fault)
    self.assertGreater(h.step(.7, frames=100)['AccPrpl_A_Rq'], 0.)

  def test_long_override_does_not_expire_release_timeout(self):
    h, c = self.harness()
    h.step(-.5, frames=100)
    h.step(0., active=False, frames=200)
    self.assertFalse(h.controller.coordination_fault)
    self.assertEqual(h.step(.7, frames=2)['AccPrpl_A_Rq'], -5.)
    self.assertGreater(h.step(.7, frames=50)['AccPrpl_A_Rq'], 0.)

  def test_new_moving_entry_requires_valid_observed_clearance(self):
    h, c = self.harness()
    c['torque'] = False
    self.assertEqual(h.step(.7, frames=20)['Cmbb_B_Enbl'], 0)
    self.assertTrue(h.controller.coordination_fault)
    h, c = self.harness()
    c['brake'] = True
    self.assertEqual(h.step(.7, frames=2)['AccPrpl_A_Rq'], -5.)
    c['brake'] = True  # keep simulated physical torque high
    self.assertEqual(h.step(.7, frames=2)['AccPrpl_A_Rq'], -5.)
    self.assertTrue(h.controller.release_barrier.needs)

  def test_reversed_producer_or_pose_time_faults_even_if_fresh(self):
    for field in ('control_shift_ns', 'pose_shift_ns'):
      h, c = self.harness()
      h.step(.7, frames=20)
      c[field] = -80_000_000
      self.assertEqual(h.step(.7, frames=2)['Cmbb_B_Enbl'], 0)
      self.assertTrue(h.controller.coordination_fault)
