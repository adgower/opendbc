import importlib

import pytest

from opendbc.car.ford.tests.test_navigator_a3_runtime import controller, sample, params


def api():
  # Fail as a missing capability before implementation, rather than a collection error.
  assert importlib.util.find_spec('opendbc.car.ford.navigator_a3_command') is not None
  return importlib.import_module('opendbc.car.ford.navigator_a3_command')


def test_requested_selection_does_not_grant_production_permission():
  select = api().select_command
  from opendbc.can import CANPacker
  from opendbc.car.ford.fordcan import CanBus, create_lat_ctl2_msg
  packer = CANPacker('ford_lincoln_base_pt')
  bus = CanBus(params())
  neutral = create_lat_ctl2_msg(packer, bus, 0, 0., 0., 0., 0., 0)
  angle = create_lat_ctl2_msg(packer, bus, 1, 0., .001, 0., 0., 1)
  selection = select('requested', neutral, angle)
  assert selection.production_frame == neutral
  assert selection.experimental_frame == angle
  assert selection.production_inhibited
  assert select('requested', None, None).experimental_frame is None
  with pytest.raises(ValueError):
    select('requested', angle, angle)
  with pytest.raises(ValueError):
    select('simulated', neutral, angle)


@pytest.mark.parametrize('mode', ['a2', 'shadow', 'requested'])
def test_controller_uses_shared_selection_without_altering_schedules(monkeypatch, mode):
  api()
  c = controller(monkeypatch, mode)
  cc, cs = sample(25)
  for i, ms in enumerate((0, 10, 20, 30, 40, 49, 59)):
    now = 1_000_000_000 + ms * 1_000_000
    c.set_navigator_a3_evidence(now, True, now, True, None)
    _, sends = c.update(cc, cs, now)
    selected = c.navigator_a3.selection
    actual = [f for f in sends if f[0] == 982]
    assert actual == ([] if selected.production_frame is None else [selected.production_frame])
    assert bool(actual) == (i in (0, 5))
    if mode == 'shadow' and i == 6:
      assert selected.experimental_frame is not None
      assert selected.production_frame is None


def test_environment_cannot_select_simulated_transport(monkeypatch):
  api()
  monkeypatch.setenv('NAVIGATOR_A3_TRANSPORT', 'simulated')
  monkeypatch.setenv('NAVIGATOR_A3_ALLOW_ACTIVE', '1')
  c = controller(monkeypatch, 'requested')
  cc, cs = sample(25)
  c.set_navigator_a3_evidence(1_000_000_000, True, 1_000_000_000, True, None)
  _, sends = c.update(cc, cs, 1_000_000_000)
  assert c.navigator_a3.selection.production_inhibited
  assert all((d[0] >> 4) & 7 == 0 for a, d, _ in sends if a == 982)
