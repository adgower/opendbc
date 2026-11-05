import math
import struct
from fractions import Fraction

import numpy as np

from opendbc.car.ford.longitudinal_nominal_reference import apply_creep_compensation
from opendbc.car.ford.longitudinal_reference_feedback import _finite


def test_creep_fast_path_matches_original_boundaries_and_signed_zero():
  speeds = [-math.inf, -1., 0., 1., 1.5, 2.999999, 3., 3.000001, 20., math.inf, math.nan]
  accelerations = [-math.inf, -3.5, -.5, -0., 0., .000001, .1, .2, .200001, 2., math.inf, math.nan]
  for speed in speeds:
    for acceleration in accelerations:
      creep = np.interp(speed, [1., 3.], [.6, 0.])
      expected = float(acceleration - np.interp(acceleration, [0., .2], [creep, 0.]))
      actual = apply_creep_compensation(acceleration, speed)
      if math.isnan(expected):
        assert math.isnan(actual)
      else:
        assert struct.pack('d', actual) == struct.pack('d', expected), (speed, acceleration)


def test_finite_builtin_fast_path_retains_boolean_overflow_and_real_fallback_rules():
  class FloatSubclass(float):
    pass

  for value in [0, -1, 2**64, 0., -0., 1e308, Fraction(1, 3), FloatSubclass(.25), np.float64(.25)]:
    assert _finite(value)
  for value in [True, False, math.nan, math.inf, -math.inf, 10**500, '1', None, object()]:
    assert not _finite(value)
