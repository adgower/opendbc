from dataclasses import replace
import math
import pytest
from opendbc.can import CANPacker
from opendbc.car.ford.navigator_a3 import Inputs, State, select_profile, update, gain_for, encode_offline


def inp(**kw):
  return replace(Inputs(1_000_000_000, 1_000_000_000, .001, 20., True, False,
                        measured_curvature_inv_m=0., measurement_ns=1_000_000_000, measurement_valid=True), **kw)


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


def test_missing_measurement_fails_neutral_above_nine():
  sample = Inputs(1_000_000_000, 1_000_000_000, .001, 20., True, False)
  o = update(select_profile('expedition-provisional-v1'), State(), sample)
  assert o.mode == 0
  assert o.reason == 'measurement_missing'


@pytest.mark.parametrize('profile', ['expedition-provisional-v1', 'bof-reference-v1', 'sensitivity-low-v1', 'sensitivity-high-v1'])
@pytest.mark.parametrize('sign', [-1., 1.])
@pytest.mark.parametrize('speed', [9., 9.0001, 13.5, 20., 26.82, 40., 60.])
def test_measurement_target_clips_before_rate_and_grid(profile, sign, speed):
  from opendbc.car.ford.navigator_a3 import angle_for, curvature_for
  from opendbc.car.ford.values import CarControllerParams
  p = select_profile(profile)
  sample = inp(speed_mps=speed, curvature_inv_m=sign * .01,
               measured_curvature_inv_m=0., measurement_ns=1_000_000_000, measurement_valid=True)
  # Start at the quantized mapping of the measurement band edge to make the
  # measurement clamp observable independently of the initial jerk ramp.
  edge = round(angle_for(p, speed, sign * .002) / .0005) * .0005
  if speed == 60.:
    edge = 0.  # The measurement band edge exceeds the acceleration bound here.
  previous = curvature_for(p, speed, edge)
  o = update(p, State(edge, 950_000_000, previous), sample)
  target = sign * (.002 if speed > 9. else .01)
  assert o.mode == 1
  assert o.measurement_limited_curvature_inv_m == pytest.approx(target)
  assert o.raw_path_angle_rad == angle_for(p, speed, sign * .01)
  if speed > 9.:
    assert abs(o.path_angle_rad) <= abs(angle_for(p, speed, target)) + .00025 + 1e-12
  assert o.measurement_age_ns == 0 and o.measured_curvature_inv_m == 0.
  assert o.path_angle_rad / .0005 == pytest.approx(round(o.path_angle_rad / .0005))
  limits = CarControllerParams.CURVATURE_LIMITS
  low = limits.apply_limits(-.02, previous, speed, 0., True, CarControllerParams.STEER_STEP)
  high = limits.apply_limits(.02, previous, speed, 0., True, CarControllerParams.STEER_STEP)
  assert low - 1e-12 <= o.equivalent_curvature_inv_m <= high + 1e-12


@pytest.mark.parametrize('changes,reason', [
  ({'measured_curvature_inv_m': None}, 'measurement_missing'),
  ({'measurement_ns': None}, 'measurement_missing'),
  ({'measurement_valid': False}, 'measurement_invalid'),
  ({'measured_curvature_inv_m': float('nan')}, 'measurement_invalid'),
  ({'measured_curvature_inv_m': float('inf')}, 'measurement_invalid'),
  ({'measured_curvature_inv_m': -float('inf')}, 'measurement_invalid'),
  ({'measurement_ns': 899_999_999}, 'measurement_stale'),
  ({'measurement_ns': 1_000_000_001}, 'measurement_stale'),
])
def test_measurement_failures(changes, reason):
  sample = inp(measured_curvature_inv_m=0., measurement_ns=1_000_000_000, measurement_valid=True)
  o = update(select_profile('expedition-provisional-v1'), State(.01, 950_000_000), replace(sample, **changes))
  assert o.mode == 0 and o.path_angle_rad == 0. and o.state.path_angle_rad == 0.
  assert o.reason == reason


@pytest.mark.parametrize('speed', [1., 8.9999, 9.])
def test_measurement_not_required_at_or_below_nine(speed):
  sample = Inputs(1_000_000_000, 1_000_000_000, .005, speed, True, False)
  p = select_profile('expedition-provisional-v1')
  missing = update(p, State(), sample)
  invalid = update(p, State(), replace(sample, measured_curvature_inv_m=float('nan'), measurement_ns=1))
  assert missing.mode == invalid.mode == 1
  assert missing.path_angle_rad == invalid.path_angle_rad
  assert missing.measurement_limited_curvature_inv_m == .005


@pytest.mark.parametrize('age', [0, 100_000_000])
def test_measurement_age_endpoints(age):
  o = update(select_profile('expedition-provisional-v1'), State(),
             inp(measured_curvature_inv_m=0., measurement_ns=1_000_000_000 - age, measurement_valid=True))
  assert o.mode == 1 and o.measurement_age_ns == age


@pytest.mark.parametrize('profile', ['expedition-provisional-v1', 'bof-reference-v1', 'sensitivity-low-v1', 'sensitivity-high-v1'])
@pytest.mark.parametrize('sign', [-1., 1.])
def test_outside_measurement_band_reenters_gradually(profile, sign):
  from opendbc.car.ford.navigator_a3 import curvature_for
  p = select_profile(profile)
  state = State(sign * .1, 950_000_000, curvature_for(p, 20., sign * .1))
  initial = abs(state.equivalent_curvature_inv_m)
  observed = []
  for i in range(20):
    now = 1_000_000_000 + i * 50_000_000
    o = update(p, state, inp(now_ns=now, source_ns=now, curvature_inv_m=sign * .01,
                            measured_curvature_inv_m=0., measurement_ns=now, measurement_valid=True))
    assert o.mode == 1
    assert o.measurement_limited_curvature_inv_m == sign * .002
    observed.append(abs(o.equivalent_curvature_inv_m))
    state = o.state
  assert .002 < observed[0] < initial  # No hard intersection or forced jump.
  assert observed[-1] <= .00203
  assert all(a >= b - 1e-12 for a, b in zip(observed[:-1], observed[1:], strict=True))


@pytest.mark.parametrize('measured', [-1e308, -.004, .004, 1e308])
def test_nonzero_finite_measurement_clips_target_before_other_limits(measured):
  sample = inp(curvature_inv_m=0., measured_curvature_inv_m=measured,
               measurement_ns=1_000_000_000, measurement_valid=True)
  o = update(select_profile('expedition-provisional-v1'), State(), sample)
  assert o.mode == 1
  assert o.raw_path_angle_rad == 0.
  assert o.measurement_limited_curvature_inv_m == pytest.approx(math.copysign(max(0., abs(measured) - .002), measured))
  assert math.isfinite(o.path_angle_rad)
  assert math.copysign(1, o.path_angle_rad) == math.copysign(1, measured)
  assert abs(o.path_angle_rad) <= .011

@pytest.mark.parametrize('profile_name', ['expedition-provisional-v1', 'bof-reference-v1', 'sensitivity-low-v1', 'sensitivity-high-v1'])
@pytest.mark.parametrize('sign', [-1, 1])
def test_scheduled_jitter_preserves_per_update_limits_and_driver_reentry(profile_name, sign):
  profile = select_profile(profile_name)
  regular_state, jitter_state = State(), State()
  jitter_now = 1_000_000_000
  for step in range(80):
    regular_now = 1_000_000_000 + step * 50_000_000
    if step:
      jitter_now += [49_000_000, 51_000_000, 48_000_000, 52_000_000][step % 4]
    speed = [9., 9.0001, 15., 25., 26.82, 40.][step // 14]
    common = dict(speed_mps=speed, curvature_inv_m=sign * .008, measured_curvature_inv_m=sign * .004,
                  driver_pressed=step in (25, 26), valid=step != 45, scheduled_update=True)
    regular = update(profile, regular_state, inp(now_ns=regular_now, source_ns=regular_now, measurement_ns=regular_now, **common))
    jitter = update(profile, jitter_state, inp(now_ns=jitter_now, source_ns=jitter_now, measurement_ns=jitter_now, **common))
    assert jitter.path_angle_rad == regular.path_angle_rad
    assert jitter.reason == regular.reason
    assert not jitter.transmission_allowed
    assert jitter.path_angle_rad / .0005 == pytest.approx(round(jitter.path_angle_rad / .0005))
    if step in (25, 26, 45):
      assert jitter.path_angle_rad == 0
    regular_state, jitter_state = regular.state, jitter.state
