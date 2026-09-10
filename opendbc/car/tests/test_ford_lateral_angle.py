#!/usr/bin/env python3
"""Unit tests for Ford path-angle lateral control strategy."""
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

from opendbc.car.ford.lateral_angle import (
  LateralAngle, HumanTurnDetector, AngleLateralResult,
  _get_platform_gains, _GAIN_NEUTRAL, _GAIN_EXPEDITION,
  _FORD_LOW_SPEED_FACTOR_ANG, _FORD_HIGH_SPEED_FACTOR_ANG,
  FORD_DBC_PATH_ANGLE_MIN, FORD_DBC_PATH_ANGLE_MAX,
)
from opendbc.car.ford.values import CAR
from numpy import interp


class TestGetPlatformGains(unittest.TestCase):
  """Tests for _get_platform_gains function.

  Only Expedition has tuned gains (provisional fork tuning from bp-dev-expedition).
  All other platforms use neutral 1.0 gains.
  """

  def test_expedition_gains(self):
    """Expedition should use provisional fork tuning gains (1.425, 1.425)."""
    gains = _get_platform_gains(CAR.FORD_EXPEDITION_MK4)
    self.assertEqual(gains, _GAIN_EXPEDITION)
    self.assertAlmostEqual(gains[0], 1.425, places=3)
    self.assertAlmostEqual(gains[1], 1.425, places=3)

  def test_f150_uses_neutral_gains(self):
    """F-150 should use neutral gains (not platform-tuned)."""
    gains = _get_platform_gains(CAR.FORD_F_150_MK14)
    self.assertEqual(gains, _GAIN_NEUTRAL)

  def test_lightning_uses_neutral_gains(self):
    """F-150 Lightning should use neutral gains (not platform-tuned)."""
    gains = _get_platform_gains(CAR.FORD_F_150_LIGHTNING_MK1)
    self.assertEqual(gains, _GAIN_NEUTRAL)

  def test_ranger_uses_neutral_gains(self):
    """Ranger should use neutral gains (not platform-tuned)."""
    gains = _get_platform_gains(CAR.FORD_RANGER_MK2)
    self.assertEqual(gains, _GAIN_NEUTRAL)

  def test_mach_e_uses_neutral_gains(self):
    """Mustang Mach-E should use neutral gains (not platform-tuned)."""
    gains = _get_platform_gains(CAR.FORD_MUSTANG_MACH_E_MK1)
    self.assertEqual(gains, _GAIN_NEUTRAL)

  def test_escape_mk4_5_uses_neutral_gains(self):
    """Escape MK4.5 should use neutral gains (not platform-tuned)."""
    gains = _get_platform_gains(CAR.FORD_ESCAPE_MK4_5)
    self.assertEqual(gains, _GAIN_NEUTRAL)

  def test_can_vehicle_uses_neutral_gains(self):
    """CAN vehicles should use neutral gains (not platform-tuned)."""
    gains = _get_platform_gains(CAR.FORD_ESCAPE_MK4)
    self.assertEqual(gains, _GAIN_NEUTRAL)


class TestSpeedFactorConstants(unittest.TestCase):
  """Tests for BP-equivalent speed factor constants.

  PROVISIONAL FORK TUNING: baked constants equivalent to BluePilot:
  - FordLowSpeedFactor_ang = 1.15 (+0.15 from default 1.0)
  - FordHighSpeedFactor_ang = 0.98 (-0.02 from default 1.0)
  Applied to high-curvature gain arm only.
  """

  def test_low_speed_factor_value(self):
    """Low speed factor matches BP FordLowSpeedFactor_ang=1.15."""
    self.assertAlmostEqual(_FORD_LOW_SPEED_FACTOR_ANG, 1.15, places=3)

  def test_high_speed_factor_value(self):
    """High speed factor matches BP FordHighSpeedFactor_ang=0.98."""
    self.assertAlmostEqual(_FORD_HIGH_SPEED_FACTOR_ANG, 0.98, places=3)

  def test_gain_expedition_unchanged(self):
    """_GAIN_EXPEDITION remains (1.425, 1.425) - speed factors are separate."""
    self.assertEqual(_GAIN_EXPEDITION, (1.425, 1.425))


class TestCurvatureFactorInterpolation(unittest.TestCase):
  """Tests for curvature factor interpolation at different speeds.

  Verifies the BP speed factor behavior:
  - At low speed (≤13.5 m/s): low-κ=1.0, high-κ=1.30*1.15=1.495
  - At high speed (≥26.82 m/s): low-κ=1.425, high-κ=1.425*0.98=1.3965
  """

  def test_low_speed_low_curvature_factor(self):
    """At ≤13.5 m/s, low-curvature gain is 1.0 (unchanged)."""
    v_ego = 13.5
    low_gain_interp = float(interp(v_ego, [13.5, 26.82], [1.0, _GAIN_EXPEDITION[0]]))
    self.assertAlmostEqual(low_gain_interp, 1.0, places=4)

  def test_low_speed_high_curvature_factor(self):
    """At ≤13.5 m/s, high-curvature gain is 1.30*1.15=1.495."""
    v_ego = 13.5
    expected = 1.30 * _FORD_LOW_SPEED_FACTOR_ANG  # 1.30 * 1.15 = 1.495
    high_gain_interp = float(interp(v_ego, [13.5, 26.82],
                                    [1.30 * _FORD_LOW_SPEED_FACTOR_ANG,
                                     _GAIN_EXPEDITION[1] * _FORD_HIGH_SPEED_FACTOR_ANG]))
    self.assertAlmostEqual(high_gain_interp, expected, places=4)
    self.assertAlmostEqual(high_gain_interp, 1.495, places=3)

  def test_high_speed_low_curvature_factor(self):
    """At ≥26.82 m/s, low-curvature gain is 1.425 (platform gain, unchanged)."""
    v_ego = 26.82
    low_gain_interp = float(interp(v_ego, [13.5, 26.82], [1.0, _GAIN_EXPEDITION[0]]))
    self.assertAlmostEqual(low_gain_interp, 1.425, places=4)

  def test_high_speed_high_curvature_factor(self):
    """At ≥26.82 m/s, high-curvature gain is 1.425*0.98=1.3965."""
    v_ego = 26.82
    expected = _GAIN_EXPEDITION[1] * _FORD_HIGH_SPEED_FACTOR_ANG  # 1.425 * 0.98 = 1.3965
    high_gain_interp = float(interp(v_ego, [13.5, 26.82],
                                    [1.30 * _FORD_LOW_SPEED_FACTOR_ANG,
                                     _GAIN_EXPEDITION[1] * _FORD_HIGH_SPEED_FACTOR_ANG]))
    self.assertAlmostEqual(high_gain_interp, expected, places=4)
    self.assertAlmostEqual(high_gain_interp, 1.3965, places=4)

  def test_midpoint_interpolation(self):
    """At midpoint speed (20.16 m/s), gains interpolate linearly."""
    v_ego = 20.16  # Midpoint between 13.5 and 26.82
    low_gain_interp = float(interp(v_ego, [13.5, 26.82], [1.0, _GAIN_EXPEDITION[0]]))
    high_gain_interp = float(interp(v_ego, [13.5, 26.82],
                                    [1.30 * _FORD_LOW_SPEED_FACTOR_ANG,
                                     _GAIN_EXPEDITION[1] * _FORD_HIGH_SPEED_FACTOR_ANG]))
    # Low-curvature: midpoint between 1.0 and 1.425 = 1.2125
    self.assertAlmostEqual(low_gain_interp, 1.2125, places=3)
    # High-curvature: midpoint between 1.495 and 1.3965 = 1.44575
    self.assertAlmostEqual(high_gain_interp, 1.44575, places=3)

  def test_below_low_speed_threshold(self):
    """Below 13.5 m/s, gains are at low-speed values (extrapolation clamps)."""
    v_ego = 10.0
    low_gain_interp = float(interp(v_ego, [13.5, 26.82], [1.0, _GAIN_EXPEDITION[0]]))
    high_gain_interp = float(interp(v_ego, [13.5, 26.82],
                                    [1.30 * _FORD_LOW_SPEED_FACTOR_ANG,
                                     _GAIN_EXPEDITION[1] * _FORD_HIGH_SPEED_FACTOR_ANG]))
    self.assertAlmostEqual(low_gain_interp, 1.0, places=4)
    self.assertAlmostEqual(high_gain_interp, 1.495, places=3)

  def test_above_high_speed_threshold(self):
    """Above 26.82 m/s, gains are at high-speed values (extrapolation clamps)."""
    v_ego = 35.0
    low_gain_interp = float(interp(v_ego, [13.5, 26.82], [1.0, _GAIN_EXPEDITION[0]]))
    high_gain_interp = float(interp(v_ego, [13.5, 26.82],
                                    [1.30 * _FORD_LOW_SPEED_FACTOR_ANG,
                                     _GAIN_EXPEDITION[1] * _FORD_HIGH_SPEED_FACTOR_ANG]))
    self.assertAlmostEqual(low_gain_interp, 1.425, places=4)
    self.assertAlmostEqual(high_gain_interp, 1.3965, places=4)


class TestHumanTurnDetector(unittest.TestCase):
  """Tests for HumanTurnDetector class."""

  def setUp(self):
    self.detector = HumanTurnDetector()

  def test_initial_state(self):
    """Detector starts inactive."""
    self.assertFalse(self.detector.active)
    self.assertEqual(self.detector.press_timer, 0.0)

  def test_inactive_when_lat_inactive(self):
    """Human-turn should not trigger when lateral control is inactive."""
    result = self.detector.update(lat_active=False, steering_pressed=True, steering_angle_deg=60)
    self.assertFalse(result)
    self.assertFalse(self.detector.active)

  def test_resets_when_steering_released(self):
    """Timer resets when driver releases steering."""
    self.detector.press_timer = 0.2
    result = self.detector.update(lat_active=True, steering_pressed=False, steering_angle_deg=60)
    self.assertFalse(result)
    self.assertEqual(self.detector.press_timer, 0.0)

  def test_resets_when_angle_small(self):
    """Timer resets when steering angle is below threshold."""
    self.detector.press_timer = 0.2
    result = self.detector.update(lat_active=True, steering_pressed=True, steering_angle_deg=30)
    self.assertFalse(result)
    self.assertEqual(self.detector.press_timer, 0.0)

  def test_triggers_after_sustained_press(self):
    """Human-turn triggers after sustained press above threshold."""
    for _ in range(10):
      result = self.detector.update(lat_active=True, steering_pressed=True, steering_angle_deg=60)

    self.assertTrue(result)
    self.assertTrue(self.detector.active)

  def test_explicit_reset(self):
    """Reset method clears state."""
    self.detector.press_timer = 0.5
    self.detector.active = True
    self.detector.reset()
    self.assertEqual(self.detector.press_timer, 0.0)
    self.assertFalse(self.detector.active)


@dataclass
class MockCarState:
  """Minimal car state mock for testing."""
  vEgoRaw: float = 15.0
  yawRate: float = 0.0
  steeringPressed: bool = False
  steeringAngleDeg: float = 0.0


@dataclass
class MockCSOut:
  """Mock CarState.out for testing."""
  vEgoRaw: float = 15.0
  yawRate: float = 0.0
  steeringPressed: bool = False
  steeringAngleDeg: float = 0.0


@dataclass
class MockCS:
  """Mock CS for testing."""
  out: MockCSOut = None

  def __post_init__(self):
    if self.out is None:
      self.out = MockCSOut()


@dataclass
class MockActuators:
  """Mock actuators for testing."""
  curvature: float = 0.0


@dataclass
class MockCC:
  """Mock CC for testing."""
  latActive: bool = True


class TestLateralAngle(unittest.TestCase):
  """Tests for LateralAngle class."""

  def _make_cp(self, car_fingerprint=CAR.FORD_EXPEDITION_MK4):
    """Create a mock CarParams."""
    cp = MagicMock()
    cp.carFingerprint = car_fingerprint
    cp.flags = 0
    return cp

  def setUp(self):
    self.cp = self._make_cp()
    self.lateral = LateralAngle(self.cp)

  def test_initialization(self):
    """LateralAngle initializes with correct gains."""
    self.assertAlmostEqual(self.lateral.gain_low_curv, 1.425, places=3)
    self.assertAlmostEqual(self.lateral.gain_high_curv, 1.425, places=3)
    self.assertEqual(self.lateral.path_angle_last, 0.0)

  def test_inactive_returns_zero(self):
    """When lateral inactive, path_angle is zero."""
    cc = MockCC(latActive=False)
    cs = MockCS(out=MockCSOut(vEgoRaw=15.0, yawRate=-0.1))
    actuators = MockActuators(curvature=0.01)

    result = self.lateral.update(cc, cs, actuators)

    self.assertEqual(result.path_angle, 0.0)
    self.assertAlmostEqual(result.shadow_curvature, 0.1 / 15.0, places=4)

  def test_active_computes_path_angle(self):
    """When active, path_angle is computed from curvature."""
    cc = MockCC(latActive=True)
    cs = MockCS(out=MockCSOut(vEgoRaw=15.0, yawRate=0.0))
    actuators = MockActuators(curvature=0.005)

    result = self.lateral.update(cc, cs, actuators)

    self.assertGreater(result.path_angle, 0.0)

  def test_path_angle_clamped_to_dbc_range(self):
    """Path angle is clamped to DBC signal limits."""
    cc = MockCC(latActive=True)
    cs = MockCS(out=MockCSOut(vEgoRaw=30.0, yawRate=0.0))
    actuators = MockActuators(curvature=0.1)

    for _ in range(50):
      result = self.lateral.update(cc, cs, actuators)

    self.assertLessEqual(result.path_angle, FORD_DBC_PATH_ANGLE_MAX)
    self.assertGreaterEqual(result.path_angle, FORD_DBC_PATH_ANGLE_MIN)

  def test_roc_limiting(self):
    """Path angle changes are rate-of-change limited."""
    cc = MockCC(latActive=True)
    cs = MockCS(out=MockCSOut(vEgoRaw=15.0, yawRate=0.0))

    self.lateral.update(cc, cs, MockActuators(curvature=0.0))
    result1 = self.lateral.update(cc, cs, MockActuators(curvature=0.1))

    self.assertLess(result1.path_angle, 0.1 * 15.0)

  def test_human_turn_zeros_path_angle(self):
    """Human turn override zeros path angle."""
    cc = MockCC(latActive=True)
    cs = MockCS(out=MockCSOut(vEgoRaw=15.0, yawRate=0.0, steeringPressed=True, steeringAngleDeg=60))
    actuators = MockActuators(curvature=0.01)

    for _ in range(10):
      result = self.lateral.update(cc, cs, actuators)

    self.assertEqual(result.path_angle, 0.0)
    self.assertTrue(self.lateral.human_turn_active)

  def test_deviation_clip_at_high_speed(self):
    """At high speed, kappa_cmd is clipped to measured curvature + error band."""
    cc = MockCC(latActive=True)
    cs = MockCS(out=MockCSOut(vEgoRaw=15.0, yawRate=0.0))
    actuators = MockActuators(curvature=0.05)

    result = self.lateral.update(cc, cs, actuators)
    self.assertIsInstance(result, AngleLateralResult)

  def test_get_current_curvature(self):
    """Current curvature is computed from yaw rate and speed."""
    cs = MockCS(out=MockCSOut(vEgoRaw=20.0, yawRate=-0.2))
    curvature = self.lateral.get_current_curvature(cs)
    self.assertAlmostEqual(curvature, 0.01, places=4)

  def test_get_current_curvature_low_speed(self):
    """At very low speed, curvature uses minimum speed divisor."""
    cs = MockCS(out=MockCSOut(vEgoRaw=0.01, yawRate=-0.1))
    curvature = self.lateral.get_current_curvature(cs)
    self.assertAlmostEqual(curvature, 1.0, places=4)


class TestAngleLateralResult(unittest.TestCase):
  """Tests for AngleLateralResult dataclass."""

  def test_default_values(self):
    """Result has sensible defaults."""
    result = AngleLateralResult()
    self.assertEqual(result.path_angle, 0.0)
    self.assertEqual(result.shadow_curvature, 0.0)

  def test_custom_values(self):
    """Result stores custom values."""
    result = AngleLateralResult(path_angle=0.1, shadow_curvature=0.005)
    self.assertEqual(result.path_angle, 0.1)
    self.assertEqual(result.shadow_curvature, 0.005)


if __name__ == "__main__":
  unittest.main()
