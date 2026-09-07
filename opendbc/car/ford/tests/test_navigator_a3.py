from dataclasses import replace
import math
import pytest
from opendbc.can import CANPacker
from opendbc.car.ford.navigator_a3 import Inputs, State, select_profile, update, gain_for, encode_offline


def inp(**kw):
  return replace(Inputs(1_000_000_000, 1_000_000_000, .001, 20., True, False), **kw)


def test_default_and_unknown():
  assert select_profile(None) is None
  assert select_profile('off') is None
  with pytest.raises(ValueError):
    select_profile('unknown')


@pytest.mark.parametrize('speed', [1., 9., 13.5, 20., 26.82, 40.])
@pytest.mark.parametrize('k', [-.001, -.0007, 0., .0007, .001])
def test_mapping_and_rate(speed, k):
  p = select_profile('expedition-provisional-v1')
  o = update(p, State(), inp(speed_mps=speed, curvature_inv_m=k))
  assert o.raw_path_angle_rad == pytest.approx(k * speed * gain_for(p, speed, k))
  assert math.copysign(1, o.path_angle_rad) == math.copysign(1, k)
  assert abs(o.path_angle_rad) <= abs(o.raw_path_angle_rad) + .0005
  assert o.transmission_allowed is False


@pytest.mark.parametrize('changes,reason', [({'driver_pressed': True}, 'driver_override'), ({'active': False}, 'inactive'),
  ({'speed_mps': 0.}, 'low_speed'), ({'source_ns': 1}, 'stale'), ({'curvature_inv_m': float('nan')}, 'invalid'),
  ({'speed_mps': float('inf')}, 'invalid'), ({'source_ns': 2_000_000_000}, 'stale')])
def test_neutral_reset(changes, reason):
  o = update(select_profile('expedition-provisional-v1'), State(.2, 950_000_000), inp(**changes))
  assert o.mode == 0 and o.path_angle_rad == 0 and o.state.path_angle_rad == 0
  assert o.reason == reason


def test_disabled():
  o = update(None, State(), inp())
  assert o.reason == 'disabled' and o.mode == 0


def test_cadence_and_reentry():
  p = select_profile('expedition-provisional-v1')
  o = update(p, State(), inp(curvature_inv_m=.02, speed_mps=25.))
  assert abs(o.path_angle_rad) <= .009 + 1e-12
  early = update(p, o.state, inp(now_ns=1_010_000_000, source_ns=1_010_000_000))
  assert early.reason == 'cadence' and early.mode == 0
  resumed = update(p, early.state, inp(now_ns=1_060_000_000, source_ns=1_060_000_000, speed_mps=25.))
  assert abs(resumed.path_angle_rad) <= .0095


def test_real_wire_quantization():
  o = update(select_profile('expedition-provisional-v1'), State(), inp())
  addr, data, bus = encode_offline(CANPacker('ford_lincoln_base_pt'), o, 3)
  raw = ((data[3] & 31) << 6) | (data[4] >> 2)
  assert addr == 0x3d6 and bus == 0
  assert raw * .0005 - .5 == pytest.approx(-o.path_angle_rad)
  assert ((data[2] << 3) | data[3] >> 5) == 1000


@pytest.mark.parametrize('k,speed', [(1e308, 20.), (.001, -5.), (.001, 1e308)])
def test_extreme_inputs_fail_neutral(k, speed):
  o = update(select_profile('expedition-provisional-v1'), State(), inp(curvature_inv_m=k, speed_mps=speed))
  assert o.mode == 0


def test_named_gain_variation():
  assert gain_for(select_profile('bof-reference-v1'), 30., .001) == pytest.approx(.95)
  assert gain_for(select_profile('expedition-provisional-v1'), 30., .001) == pytest.approx(1.425)
  assert gain_for(select_profile('sensitivity-low-v1'), 5., .001) == pytest.approx(1.17)
  assert gain_for(select_profile('sensitivity-high-v1'), 30., .001) == pytest.approx(1.2825)


def test_unwind_quantization_respects_rate():
  speed = 20.123
  from opendbc.car.ford.navigator_a3 import interp
  o = update(select_profile('expedition-provisional-v1'), State(.1, 950_000_000), inp(curvature_inv_m=0., speed_mps=speed))
  assert abs(o.path_angle_rad - .1) <= interp(speed, (9., 10., 15., 25.), (.055, .055, .0425, .009)) + 1e-12


@pytest.mark.parametrize('angle', [-1., 1., float('nan')])
def test_invalid_state_is_neutral(angle):
  o = update(select_profile('expedition-provisional-v1'), State(angle, 950_000_000), inp())
  assert o.mode == 0 and o.path_angle_rad == 0


@pytest.mark.parametrize('speed,previous,target', [(8.763889, .03, .0058520888), (23.76111, 0., .0047635166)])
def test_first_rejection_curvature_step(speed, previous, target):
  # Recorded A2 step and post-driver-release requests. The angle-only limiter
  # must also preserve the frozen Ford curvature envelope through quantization.
  from opendbc.car.lateral import MAX_LATERAL_JERK
  p = select_profile('expedition-provisional-v1')

  def inverse(angle):
    lo, hi = 0., .02
    for _ in range(60):
      mid = (lo + hi) / 2
      if mid * speed * gain_for(p, speed, mid) < abs(angle):
        lo = mid
      else:
        hi = mid
    return math.copysign((lo + hi) / 2, angle)
  o = update(p, State(previous, 950_000_000), inp(speed_mps=speed, curvature_inv_m=target))
  assert o.mode == 1  # Reject-everything is not a fix.
  assert abs(inverse(o.path_angle_rad) - inverse(previous)) <= MAX_LATERAL_JERK / speed**2 * .05 + 1e-12


def test_final_gain_and_inverse_history_are_logged_at_original_speed():
  from opendbc.car.ford.navigator_a3 import curvature_for
  p = select_profile('expedition-provisional-v1')
  o = update(p, State(), inp(speed_mps=23.76111, curvature_inv_m=.0047635166))
  equivalent = curvature_for(p, 23.76111, o.path_angle_rad)
  assert o.equivalent_curvature_inv_m == equivalent
  assert o.state.equivalent_curvature_inv_m == equivalent
  assert o.effective_gain == gain_for(p, 23.76111, equivalent)
  assert o.requested_gain == gain_for(p, 23.76111, .0047635166)
  assert o.effective_gain != o.requested_gain
