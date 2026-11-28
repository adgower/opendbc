import pytest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus
from opendbc.car.ford.carstate import CarState
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR, FordFlags

MS = 1_000_000
PACKER = CANPacker("ford_lincoln_base_pt")


def status(side="Left", bus=2, **values):
  return PACKER.make_can_msg(f"Side_Detect_{side[0]}_Stat", bus, {f"Sod{side}_D_Stat": 2, **values})


def setup_car(car=CAR.LINCOLN_NAVIGATOR_MK4, has_bsm=True, canfd=True):
  cp = CarInterface.get_non_essential_params(car)
  cp.flags &= ~int(FordFlags.HAS_BSM | FordFlags.CANFD)
  if has_bsm:
    cp.flags |= int(FordFlags.HAS_BSM)
  if canfd:
    cp.flags |= int(FordFlags.CANFD)
  cs = CarState(cp)
  parsers = cs.get_can_parsers(cp)

  def update(*entries, sendcan=False):
    for parser in parsers.values():
      parser.update([(ms * MS, frames) for ms, frames in entries], sendcan=sendcan)
    ret = cs.update(parsers)
    return ((ret.navigatorLeftBlindspotClearMonoTime, ret.navigatorLeftBlindspotBlockMonoTime),
            (ret.navigatorRightBlindspotClearMonoTime, ret.navigatorRightBlindspotBlockMonoTime), ret.leftBlindspot, ret.rightBlindspot)
  return parsers[Bus.cam], update


def test_freshness_and_hidden_gap():
  _, update = setup_car()
  assert update((100, []))[:2] == ((0, 0), (0, 0))
  assert update((200, [status(), status("Right")]))[:2] == ((200*MS, 0), (200*MS, 0))
  assert update((650, []))[:2] == ((200*MS, 0), (200*MS, 0))
  # A fresh packet can arrive before the model has observed the expired packet.
  assert update((651, [status()]))[:2] == ((651*MS, 650*MS), (0, 650*MS))


@pytest.mark.parametrize(("field", "value"), [
  ("SodLeft_D_Stat", 0), ("SodLeft_D_Stat", 1), ("SodLeft_D_Stat", 3), ("SodLeft_D_Stat", 7),
  ("SodDetctLeft_D_Stat", 1), ("SodDetctLeft_D_Stat", 4), ("SodDetctLeft_D_Stat", 7),
  ("SodSnsLeft_D_Stat", 1), ("SodSnsLeft_D_Stat", 2), ("SodSnsLeft_D_Stat", 3),
  ("SodAlrtLeft_D_Stat", 1), ("SodAlrtLeft_D_Stat", 2), ("SodAlrtLeft_D_Stat", 3),
])
def test_transient_warning_and_legacy_boolean(field, value):
  _, update = setup_car()
  update((100, [status(), status("Right")]))
  warning = status(**{field: value})
  assert update((200, [warning])) == ((0, 200*MS), (100*MS, 0), field == "SodDetctLeft_D_Stat", False)
  # On the same timestamp a warning wins, even if a clear packet comes last.
  assert update((300, [warning, status()])) == ((0, 300*MS), (100*MS, 0), False, False)
  assert update((400, [warning]), (430, [status()]))[0] == (430*MS, 400*MS)


@pytest.mark.parametrize("length", [0, 2, 7, 9, 64, 65])
def test_bad_length_cannot_claim_clear(length):
  _, update = setup_car()
  addr, data, bus = status()
  update((100, [status()]))
  bad = (addr, (data + bytes(length))[:length], bus)
  assert update((110, [bad]))[0] == (0, 110*MS)
  assert update((120, [bad, status()]))[0] == (0, 120*MS)
  assert update((121, [status()]))[0] == (121*MS, 120*MS)


def test_real_camera_rx_only_and_timestamp_recovery():
  _, update = setup_car()
  clear = status("Right")
  update((200, [clear]))
  assert update((210, [status("Right", 0), status("Right", 130)]))[1] == (200*MS, 0)
  assert update((220, [clear]), sendcan=True)[1] == (200*MS, 0)
  assert update((190, [clear]))[1] == (0, 210*MS)
  assert update((210, [clear]))[1] == (0, 210*MS)
  assert update((230, [clear]))[1] == (230*MS, 210*MS)
  # A new loader starts in its own boot clock domain.
  _, new_boot = setup_car()
  assert new_boot((1, [clear]))[1] == (MS, 0)


@pytest.mark.parametrize(("car", "has_bsm", "canfd"), [
  (CAR.LINCOLN_NAVIGATOR_MK4, False, True),
  (CAR.LINCOLN_NAVIGATOR_MK4, True, False),
  (CAR.FORD_EXPEDITION_MK4, True, True),
])
def test_unaffected_cars(car, has_bsm, canfd):
  _, update = setup_car(car, has_bsm, canfd)
  assert update((100, [status(), status("Right")]))[:2] == ((0, 0), (0, 0))


def test_stock_camera_outputs_and_watchdogs():
  candidate, _ = setup_car()
  stock = CANParser(candidate.dbc_name, [], 2)
  stock.add_event_periodic_message("IPMA_Data", period_s=1., timeout_s=2.05, min_span_s=.9)
  # Production registers the two ordinary watchdogs on the first CarState read.
  assert candidate.addresses == stock.addresses == {0x3d8}
  for parser in (stock, candidate):
    _ = parser.vl["Side_Detect_L_Stat"], parser.vl["Side_Detect_R_Stat"]
  for ms in range(100, 3000, 100):
    frames = [(0x3d8, bytes(8), 2)] if ms % 1000 == 100 else []
    if ms % 200 == 0:
      frames.extend([status(), status("Right")])
    if ms in (400, 1000):
      frames.extend([status(SodAlrtLeft_D_Stat=2), (0x3a7, bytes(3), 2)])
    entry = (ms * MS, frames)
    assert candidate.update(entry) == stock.update(entry)
    assert candidate.can_valid == stock.can_valid
    assert candidate.bus_timeout == stock.bus_timeout
    assert candidate.vl == stock.vl
    assert candidate.ts_nanos == stock.ts_nanos
