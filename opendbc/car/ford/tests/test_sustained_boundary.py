from opendbc.car.ford.tests.test_sustained_tracking import make, oldmake, step
import pytest


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("rate", [0.2, 0.3, 0.5, 1.0, 2.0])
def test_rising_speed_above75_matches_independently_warmed_installed(sign, rate):
  c = make()
  b = oldmake()
  for i in range(60):
    kw = dict(mph=64, k=sign * 0.00065, excess=2.4)
    step(c, i, **kw)
    step(b, i, **kw)
  for j in range(70):
    kw = dict(mph=min(83, 70 + j * rate), k=sign * (0.0006 + 0.000002 * j), excess=2.4)
    x = step(c, 60 + j, **kw)
    y = step(b, 60 + j, **kw)
    if kw["mph"] >= 75:
      assert x == y, (j, kw, x, y)


@pytest.mark.parametrize("sign", [-1, 1])
def test_slow_and_fast_crossings_preserve_native_output_slew(sign):
  from opendbc.car.ford.lateral_angle import _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES
  from numpy import interp

  c = make()
  previous = 0.0
  speeds = [64] * 45 + [70, 72, 74, 74.9, 75, 75.1, 83, 65, 76, 45, 83] + [83] * 30
  for i, mph in enumerate(speeds):
    out = step(c, i, mph=mph, k=sign * 0.00065)
    assert abs(out.path_angle - previous) <= float(interp(mph * 0.44704, _SOFT_ROC_SPEEDS, _SOFT_ROC_VALUES)) + 1e-12
    assert -0.5 <= out.path_angle <= 0.5235
    previous = out.path_angle


@pytest.mark.parametrize("sign", [-1, 1])
def test_upper_shadow_receives_target_and_invalid_context(sign):
  c = make()
  b = oldmake()
  c.target_speed = b.target_speed = 33.0
  for i in range(80):
    kw = dict(mph=83, k=sign * 0.00065, body_valid=i < 50, age=0.3 if i >= 60 else 0, generation=1 if i >= 70 else 0)
    assert step(c, i, **kw) == step(b, i, **kw)
  assert c.upper_reference.target_speed == 33.0


@pytest.mark.parametrize("enabled", [False, True])
def test_upper_shadow_advances_while_below75(enabled):
  c = make(enabled)
  b = oldmake(enabled)
  for i in range(160):
    mph = 64 if i < 80 else 83
    kw = dict(mph=mph, k=0.00065, excess=2.4, driver=45 <= i < 50, lane=60 <= i < 65)
    x = step(c, i, **kw)
    y = step(b, i, **kw)
    if mph >= 75 and not c.upper_handoff_pending:
      assert x == y


def test_true_rate_limited_handoff_is_reported_and_settles():
  c = make()
  b = oldmake()
  for i in range(30):
    step(c, i, mph=64, k=0.00065)
    step(b, i, mph=64, k=0.00065)
  # A deliberately adverse previous actuator command tests the handoff contract;
  # it is not a possible-in-this-drive claim or a normal state reset.
  c._actual_output_last = 0.1
  out = step(c, 30, mph=83, k=0.00065)
  step(b, 30, mph=83, k=0.00065)
  assert c.upper_handoff_pending and abs(out.path_angle - 0.1) <= 0.009 + 1e-12
  for i in range(31, 60):
    out = step(c, i, mph=83, k=0.00065)
    ref = step(b, i, mph=83, k=0.00065)
  assert not c.upper_handoff_pending and out == ref


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("event", ["invalid", "generation", "driver", "disable", "reversal"])
def test_repeated_up_down_crossings_report_returned_output(sign, event):
  c = make()
  b = oldmake()
  for i in range(240):
    mph = 64 if i < 60 else [70, 73, 74.9, 75.1, 83, 75.1, 74.9, 72, 64][((i - 60) // 5) % 9]
    kw = dict(mph=mph, k=sign * 0.00065, excess=2.4)
    if 100 <= i < 120:
      if event == "invalid":
        kw.update(body_valid=False, age=0.3)
      if event == "generation":
        kw["generation"] = 1
      if event == "driver":
        kw["driver"] = True
      if event == "disable":
        c.enabled = b.enabled = False
      if event == "reversal":
        kw["k"] = -kw["k"]
    elif event == "disable":
      c.enabled = b.enabled = True
    x = step(c, i, **kw)
    y = step(b, i, **kw)
    assert abs(c.actual_final_residual - (x.path_angle - c.actual_reference_path)) < 1e-12
    assert c.actual_reduction == max(0.0, abs(c.actual_reference_path) - abs(x.path_angle))
    assert -0.5 <= x.path_angle <= 0.5235
    if mph >= 75 and not c.upper_handoff_pending:
      assert x == y
