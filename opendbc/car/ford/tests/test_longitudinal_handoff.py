import unittest
from opendbc.car.ford.longitudinal_handoff import Handoff


class TestHandoff(unittest.TestCase):
  def settle(self, h, target=0.2):
    rows = []
    for i in range(1, 102):
      rows.append(h.update(i * 0.02, True, False, target, True, False))
    return rows

  def test_braking_withdraws_immediately(self):
    h = Handoff()
    self.settle(h)
    self.assertEqual(h.update(2.04, True, True, 0.3, True, False).gas, -5)

  def test_resume_is_bounded_without_sentinel_interpolation(self):
    h = Handoff()
    rows = self.settle(h, 1)
    valid = [r.gas for r in rows if r.gas != -5]
    self.assertEqual(valid[0], 0)
    self.assertTrue(all(b - a <= 0.01000001 for a, b in zip(valid, valid[1:], strict=False)))
    self.assertGreater(valid[-1], 0.9)

  def test_near_zero_request_cannot_exit_limiter(self):
    h = Handoff()
    self.settle(h, 0)
    r = h.update(2.04, True, False, 2, True, False)
    self.assertLessEqual(r.gas, 0.01000001)

  def test_decreasing_request_passes_immediately(self):
    h = Handoff()
    self.settle(h, 0.5)
    self.assertEqual(h.update(2.04, True, False, -0.2, True, False).gas, -0.2)

  def test_negative_valid_request_is_not_sentinel(self):
    h = Handoff()
    rows = self.settle(h, -0.4)
    self.assertEqual(rows[-1].gas, -0.4)

  def test_missing_pitch_stopping_disengage_and_sentinel(self):
    for active, brake, target, pitch, stopping in [
      (False, False, 1, True, False),
      (True, False, 1, False, False),
      (True, False, 1, True, True),
      (True, False, -5, True, False),
    ]:
      h = Handoff()
      self.settle(h)
      self.assertEqual(h.update(2.04, active, brake, target, pitch, stopping).gas, -5)

  def test_bad_time_or_gap_requalifies(self):
    for t in [2.02, 2.0, 3.0, float("nan")]:
      h = Handoff()
      self.settle(h)
      self.assertEqual(h.update(t, True, False, 1, True, False).gas, -5)

  def test_nonfinite_or_out_of_range_target_is_inactive(self):
    for target in [float("nan"), float("inf"), -1, 2.1]:
      self.assertEqual(Handoff().update(0.02, True, False, target, True, False).gas, -5)

  def test_brake_toggle_requalifies(self):
    h = Handoff()
    self.settle(h)
    h.update(2.04, True, True, 0.2, True, False)
    self.assertEqual(h.update(2.06, True, False, 0.2, True, False).gas, -5)
    self.assertEqual(h.update(2.08, True, False, 0.2, True, False).gas, 0)

  def test_qualification_uses_elapsed_time(self):
    h = Handoff(qualify_s=0.06)
    self.assertEqual(h.update(1, True, False, 1, True, False).gas, -5)
    self.assertEqual(h.update(1.02, True, False, 1, True, False).gas, -5)
    self.assertEqual(h.update(1.04, True, False, 1, True, False).gas, -5)
    self.assertEqual(h.update(1.06, True, False, 1, True, False).gas, 0)


if __name__ == "__main__":
  unittest.main()
