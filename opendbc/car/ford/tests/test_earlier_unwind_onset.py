import pytest

from opendbc.car.ford.lateral_angle_earlier_unwind_onset import EarlierUnwindOnset as Candidate
from opendbc.car.ford.lateral_angle_calibrated_mid_unwind import CalibratedMidUnwind as Installed
from opendbc.car.ford.tests.test_high_speed import make as oldmake
from opendbc.car.ford.tests.test_wheel_target import step
from opendbc.can import CANPacker
from opendbc.car.ford import fordcan
from opendbc.car.ford.values import DBC
from opendbc.car import Bus


def make(cls=Candidate, enabled=True):
  c = cls(oldmake().CP, enabled=enabled)
  c.input_state_valid = True
  return c


def packed(c, value):
  m = fordcan.create_lat_ctl2_msg(CANPacker(DBC[c.CP.carFingerprint][Bus.pt]), fordcan.CanBus(c.CP), 1, 0.0, -value, 0.0, 0.0, 0)
  d = m[1]
  return -((((d[3] & 31) << 6) | (d[4] >> 2)) * 0.0005 - 0.5)


def sequence(c, mph=67, sign=1, **fault):
  rows = []
  for i in range(45):
    k = sign * 0.0009 * (1.0 if i < 30 else 0.985 ** (i - 29))
    kw = dict(mph=mph, k=k, excess=2.4, body_ratio=0.95 if i < 30 else 1.5)
    if i >= 30:
      kw.update(fault)
    out = step(c, i, **kw)
    rows.append((i * 0.05, abs(c.actual_reference_path) - abs(out.path_angle), abs(packed(c, c.actual_reference_path)) - abs(packed(c, out.path_angle))))
  return rows


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("mph", [64, 67, 69])
def test_confirmed_unwind_has_earlier_packed_onset(sign, mph):
  c = make()
  b = make(Installed)
  a = sequence(c, mph, sign)
  old = sequence(b, mph, sign)

  def first(rows):
    return next(t for t, _, delta in rows if delta > 1e-10)

  assert first(a) <= first(old) - 0.10
  assert max(r[1] for r in a) <= 0.01 + 1e-12


@pytest.mark.parametrize(
  "fault",
  [
    dict(driver=True),
    dict(active=False),
    dict(valid=False),
    dict(state_valid=False),
    dict(fault=True),
    dict(body_valid=False),
    dict(body_age=0.2),
    dict(body_std=0.1),
    dict(lane=True),
    dict(mismatch=1.0),
    dict(body_ratio=0.9),
    dict(excess=-1.0),
  ],
)
def test_guarded_input_creates_no_additional_correction(fault):
  a = sequence(make(), **fault)
  b = sequence(make(Installed), **fault)
  assert a == b


@pytest.mark.parametrize("mph", [5, 15, 23, 27, 30, 45, 55, 60, 75, 83, 85])
def test_protected_ranges_and_default_off(mph):
  assert sequence(make(), mph) == sequence(make(Installed), mph)
  assert sequence(make(enabled=False), mph) == sequence(make(Installed, enabled=False), mph)
