import pytest

from opendbc.car import Bus
from opendbc.car.ford.carstate import CarState
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR


@pytest.mark.parametrize("alpha", [False, True])
def test_camera_status_event_burst_and_missing_heartbeat(alpha):
  cp = CarInterface.get_non_essential_params(CAR.LINCOLN_NAVIGATOR_MK4)
  cp.openpilotLongitudinalControl = alpha
  parser = CarState.get_can_parsers(cp)[Bus.cam]
  _ = parser.vl["IPMA_Data"]

  def tick(ms, status_bus=None):
    # Keep the camera bus alive even when the monitored message is missing.
    frames = [(0x123, bytes(8), parser.bus)]
    if status_bus is not None:
      frames.append((0x3d8, bytes(8), status_bus))
    parser.update([[(ms + 1000) * 1_000_000, frames]])
    return parser.can_valid

  assert not tick(0)
  for ms in range(10, 900, 10):
    assert not tick(ms, parser.bus if ms % 30 == 10 else None)
  # The event burst qualifies only after spanning most of a heartbeat period.
  assert tick(910, parser.bus)
  for ms in range(920, 1420, 10):
    assert tick(ms, parser.bus)
  for ms in range(1420, 6430, 10):
    assert tick(ms, parser.bus if ms % 1000 == 420 else None)
  # One missing heartbeat is within budget. A second missing heartbeat expires it.
  # Status on the opposite bus throughout the loss cannot satisfy the watchdog.
  for ms in range(6430, 8480, 10):
    assert tick(ms, 0)
  for ms in range(8480, 8610, 10):
    valid = tick(ms, 0)
  assert not valid
  assert not tick(8610, parser.bus)
  for ms in range(8620, 9510, 10):
    assert not tick(ms, parser.bus if ms < 8650 else None)
  assert tick(9510, parser.bus)
  assert not parser.bus_timeout


def test_expedition_camera_status_still_uses_normal_parser():
  cp = CarInterface.get_non_essential_params(CAR.FORD_EXPEDITION_MK4)
  parser = CarState.get_can_parsers(cp)[Bus.cam]
  assert 0x3d8 not in parser.addresses
  _ = parser.vl["IPMA_Data"]
  assert parser.message_states[0x3d8].frequency == 0


@pytest.mark.parametrize("size", [0, 7, 9])
def test_malformed_camera_status_cannot_refresh_or_requalify(size):
  cp = CarInterface.get_non_essential_params(CAR.LINCOLN_NAVIGATOR_MK4)
  parser = CarState.get_can_parsers(cp)[Bus.cam]

  def send(ms, length):
    return parser.update([[ms * 1_000_000, [(0x3d8, bytes(length), parser.bus)]]])

  assert 0x3d8 in send(100, 8)
  assert 0x3d8 in send(1100, 8)
  assert parser.can_valid
  for ms in range(2100, 3800, 100):
    assert 0x3d8 not in send(ms, size)
    valid = parser.can_valid
  assert not valid
  assert 0x3d8 in send(3800, 8)
  assert not parser.can_valid
  assert 0x3d8 not in send(4800, size)
  assert not parser.can_valid
  assert 0x3d8 in send(4810, 8)
  assert parser.can_valid


def test_rewinding_time_cannot_revive_a_lost_camera_heartbeat():
  cp = CarInterface.get_non_essential_params(CAR.LINCOLN_NAVIGATOR_MK4)
  parser = CarState.get_can_parsers(cp)[Bus.cam]
  state = parser.message_states[0x3d8]

  def update(ms, status_frames=0):
    frames = [(0x123, bytes(8), parser.bus)] + [(0x3d8, bytes(8), parser.bus)] * status_frames
    return parser.update([[ms * 1_000_000, frames]])

  update(1000, 1)
  update(2000, 1)
  assert parser.can_valid
  for ms in range(4100, 4200, 10):
    update(ms)
    valid = parser.can_valid
  assert not valid

  update(2800)
  assert not parser.can_valid
  for ms in (2800, 3800):
    assert 0x3d8 not in update(ms, 1)
    assert not parser.can_valid

  old_count = len(state.timestamps)
  assert 0x3d8 in update(4300, 2)
  assert len(state.timestamps) == old_count + 2
  # A delayed frame can also arrive before the next validity check.
  assert 0x3d8 not in update(4250, 1)
  assert not parser.can_valid
  assert 0x3d8 in update(4400, 1)
  assert not parser.can_valid
  assert 0x3d8 in update(5200, 1)
  assert parser.can_valid
