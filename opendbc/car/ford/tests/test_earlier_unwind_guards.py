from opendbc.car.ford.tests.test_earlier_unwind_onset import make, step, Installed
import pytest


@pytest.mark.parametrize("mph", [60.001, 64, 67, 69.999, 70, 70.001, 74.999, 75])
@pytest.mark.parametrize("sign", [-1, 1])
def test_output_bounds_and_no_speed_boundary_jump(mph, sign):
  c = make()
  b = make(Installed)
  for i in range(90):
    speed = mph if i < 45 else 45 if i < 60 else 83
    k = sign * 0.0009 * (1 if i < 30 else 0.985 ** (i - 29))
    kw = dict(mph=speed, k=k, excess=2.4, body_ratio=0.95 if i < 30 else 1.5)
    x = step(c, i, **kw)
    y = step(b, i, **kw)
    assert -0.5 <= x.path_angle <= 0.5235
    if speed in [45, 83]:
      assert x == y


@pytest.mark.parametrize(
  "kind", ["duplicate_state", "stale_state", "duplicate_motion", "stale_motion", "generation", "driver", "invalid_calibration", "lane", "reverse", "closing"]
)
def test_new_credit_requires_five_distinct_valid_inputs_after_break(kind):
  c = make()
  for i in range(33):
    step(c, i, mph=67, k=0.0009 * (1 if i < 30 else 0.985 ** (i - 29)), excess=2.4, body_ratio=0.95 if i < 30 else 1.5)
  for i in range(33, 37):
    kw = dict(mph=67, k=0.0009 * 0.985 ** (i - 29), excess=2.4, body_ratio=1.5)
    if kind == "duplicate_state":
      kw["state_lag_ns"] = (i - 32) * 50_000_000
    if kind == "stale_state":
      kw["state_lag_ns"] = 100_000_000
    if kind == "duplicate_motion":
      kw["body_age"] = (i - 32) * 0.05
    if kind == "stale_motion":
      kw["body_age"] = 0.2
    if kind == "generation":
      kw["generation"] = 1
    if kind == "driver":
      kw["driver"] = True
    if kind == "invalid_calibration":
      kw["valid"] = False
    if kind == "lane":
      kw["lane"] = True
    if kind == "reverse":
      kw["k"] = -kw["k"]
    if kind == "closing":
      kw["excess"] = 0.1
    step(c, i, **kw)
    assert c.onset_receipt is None, (kind, i, c.onset_receipt)


@pytest.mark.parametrize("sign", [-1, 1])
def test_fresh_rising_entry_and_constant_demand_do_not_receive_credit(sign):
  for kind in ["constant", "rising"]:
    c = make()
    for i in range(90):
      k = sign * 0.0007 * (1 if kind == "constant" else 1 + 0.002 * i)
      step(c, i, mph=67, k=k, excess=2.4, body_ratio=1.5)
      assert c.onset_receipt is None


@pytest.mark.parametrize("kind", ["invalid_can", "invalid_state", "rising_after_dip"])
def test_invalid_recovery_or_new_rise_cannot_borrow_history(kind):
  c = make()
  for i in range(34):
    k = 0.0009 if i < 30 else 0.0009 * 0.98 ** (i - 29)
    if kind == "rising_after_dip" and i == 33:
      k = 0.0009 * 0.98**3 * 1.005
    if kind == "invalid_can" and i == 30:
      # The helper's state_valid option feeds canValid; event validity is separate.
      step(c, i, mph=67, k=k, excess=2.4, body_ratio=1.5, state_valid=False)
    elif kind == "invalid_state" and i == 30:
      c.input_state_valid = False
      step(c, i, mph=67, k=k, excess=2.4, body_ratio=1.5)
      c.input_state_valid = True
    else:
      step(c, i, mph=67, k=k, excess=2.4, body_ratio=1.5)
    if i >= 30 and (kind != "rising_after_dip" or i == 33):
      assert c.onset_receipt is None, (kind, i, c.onset_receipt)
