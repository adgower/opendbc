import unittest
from opendbc.car.ford.longitudinal_release import ReleaseGate


class TestRelease(unittest.TestCase):
  def run_gate(self, raw=0.2, net=0.2, phase="pid", valid=True):
    h = ReleaseGate()
    rows = []
    for i in range(11):
      rows.append(h.update(i * 0.02, True, True, True, raw, net, phase, valid))
    return rows

  def test_persistent_point_two_can_release(self):
    r = self.run_gate()
    self.assertTrue(r[0].brake)
    self.assertFalse(r[-1].brake)

  def test_actual_pid_after_stop_is_supported(self):
    self.assertFalse(self.run_gate(phase="pid")[-1].brake)

  def test_positive_raw_negative_net_grade_stays_braking(self):
    self.assertTrue(self.run_gate(raw=0.2, net=-0.1)[-1].brake)

  def test_small_positive_on_grade_qualifies_only_with_positive_net(self):
    self.assertFalse(self.run_gate(raw=0.4, net=0.1)[-1].brake)

  def test_zero_boundary_and_stopping_off_do_not_release(self):
    for raw, net, phase in [(0, 0.2, "pid"), (0.2, 0, "pid"), (0.2, 0.2, "stopping"), (0.2, 0.2, "off")]:
      self.assertTrue(self.run_gate(raw, net, phase)[-1].brake)

  def test_invalid_pitch_cannot_clear_previous_brake(self):
    h = ReleaseGate()
    r = h.update(1, True, False, True, 0.5, 0.5, "pid", False)
    self.assertTrue(r.brake)

  def test_invalid_demand_cannot_clear_previous_brake(self):
    for raw, net in [(float("nan"), 0.5), (0.5, float("inf"))]:
      self.assertTrue(ReleaseGate().update(1, True, False, True, raw, net, "pid", True).brake)

  def test_disengaged_preserves_neutral_baseline(self):
    self.assertFalse(ReleaseGate().update(1, False, False, True, 0.2, 0.2, "off", False).brake)

  def test_request_noise_resets_dwell(self):
    h = ReleaseGate()
    for i in range(20):
      self.assertTrue(h.update(i * 0.02, True, True, True, 0.2, 0.2 if i % 2 == 0 else 0, "pid", True).brake)

  def test_time_gap_reversal_and_duplicates_reset(self):
    for t in [0, 0.01, 0.3, float("nan")]:
      h = ReleaseGate()
      h.update(0, True, True, True, 0.2, 0.2, "pid", True)
      h.update(0.02, True, True, True, 0.2, 0.2, "pid", True)
      self.assertTrue(h.update(t, True, True, True, 0.2, 0.2, "pid", True).brake)

  def test_negative_demand_relatches_immediately(self):
    h = ReleaseGate()
    for i in range(10):
      h.update(i * 0.02, True, True, True, 0.2, 0.2, "pid", True)
    self.assertTrue(h.update(0.2, True, True, False, -0.1, -0.1, "pid", True).brake)

  def test_existing_valid_release_remains(self):
    self.assertFalse(ReleaseGate().update(1, True, False, True, 0.5, 0.5, "pid", True).brake)

  def test_near_zero_noise_does_not_create_brake_release_cycles(self):
    h = ReleaseGate()
    for i in range(300):
      self.assertTrue(h.update(i * 0.02, True, True, True, 0.2, 0.01 if i % 20 < 10 else -0.01, "pid", True).brake)

  def test_positive_margin_boundary_is_held(self):
    for raw, net in [(0.02, 0.2), (0.2, 0.02)]:
      self.assertTrue(self.run_gate(raw, net)[-1].brake)


if __name__ == "__main__":
  unittest.main()
