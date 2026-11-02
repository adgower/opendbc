import pytest
from opendbc.car.ford.tests.test_wheel_target import make, step


@pytest.mark.parametrize("sign", [-1, 1])
def test_moderate_highway_bend_can_use_calibrated_feedback(sign):
  c = make()
  c.curve_scope_extension_enabled = True
  for i in range(60):
    step(c, i, mph=75.8, k=sign * 0.00156, roll=sign * 0.021, excess=2.7, body_ratio=1.35, body_std=0.012)
  assert c.reduction > 0 and sign * c.final_residual < 0
  assert abs(c.final_residual) <= min(0.01, 0.15 * abs(c.reference.path_angle_last)) + 1e-12


@pytest.mark.parametrize("mph", [15, 25, 45, 60, 65, 70, 85, 90])
def test_protected_ranges_remain_exact(mph):
  c = make()
  c.curve_scope_extension_enabled = True
  b = make()
  for i in range(50):
    kw = dict(mph=mph, k=0.00156, roll=0.021, excess=2.7, body_ratio=1.35)
    assert step(c, i, **kw) == step(b, i, **kw)


@pytest.mark.parametrize("fault", [{"driver": True}, {"lane": True}, {"body_valid": False}, {"state_valid": False}, {"body_age": 0.2}, {"body_ratio": 0.8}])
def test_scope_extension_never_bypasses_evidence(fault):
  c = make()
  c.curve_scope_extension_enabled = True
  for i in range(60):
    kw = dict(mph=75.8, k=0.00156, roll=0.021, excess=2.7, body_ratio=1.35)
    kw.update(fault)
    step(c, i, **kw)
  assert c.reduction == 0


def test_scope_option_survives_low_high_and_disabled_transitions():
  c = make()
  c.curve_scope_extension_enabled = True
  for i in range(30):
    step(c, i, mph=45, k=0.00156, roll=0.021, excess=2.7, body_ratio=1.35)
  assert c.curve_scope_extension_enabled is True
  for i in range(30, 80):
    step(c, i, mph=75.8, k=0.00156, roll=0.021, excess=2.7, body_ratio=1.35)
  assert c.reduction > 0
  step(c, 80, mph=75.8, k=0.00156, roll=0.021, excess=2.7, body_ratio=1.35, active=False)
  assert c.curve_scope_extension_enabled is True
  for i in range(81, 130):
    step(c, i, mph=75.8, k=0.00156, roll=0.021, excess=2.7, body_ratio=1.35)
  assert c.reduction > 0


def test_actual_selection_keeps_original_policy_shadows_off():
  from opendbc.car.ford.lateral_angle_wheel_target import enable_calibrated_curve_scope as enable_curve_scope
  from opendbc.car.ford.tests.test_unwind_allocation import make as make_root
  from opendbc.car.ford.lateral_angle_wheel_target import CalibratedWheelTargetReduction

  root = enable_curve_scope(make_root())
  seen = set()
  selected = []

  def visit(obj):
    if id(obj) in seen:
      return
    seen.add(id(obj))
    if vars(obj).get("curve_scope_extension_enabled", False):
      assert type(obj) is CalibratedWheelTargetReduction
      selected.append(obj)
      assert obj.installed_baseline.curve_scope_extension_enabled is False
    for name in ("reference", "baseline", "installed_baseline", "upper_reference", "stock", "unchanged"):
      child = vars(obj).get(name)
      if child is not None and hasattr(child, "__dict__"):
        visit(child)

  visit(root)
  assert selected
  c = enable_curve_scope(make())
  for i in range(320):
    phase = i // 80 % 4
    mph = 45 if phase in [0, 2] else 75.8
    step(c, i, mph=mph, k=0.00156, roll=0.021, excess=2.7, body_ratio=1.35)
    assert c.curve_scope_extension_enabled is True
    assert c.installed_baseline.curve_scope_extension_enabled is False
