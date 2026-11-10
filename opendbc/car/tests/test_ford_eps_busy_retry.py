from collections import deque
from unittest.mock import patch

import pytest

from opendbc.car.can_definitions import CanData
from opendbc.car.isotp_parallel_query import IsoTpParallelQuery

EPS = (0x730, None)
TP = b'\x3e\x00'
READ = b'\x22\xf1\x88'


@pytest.mark.parametrize('replies,enabled,expected_sends,matched', [
  ([b'\x7f\x3e\x21', b'\x7e\x00'], True, [TP, TP, READ], True),
  ([b'\x7f\x3e\x21'] * 3, True, [TP, TP, TP], False),
  ([b'\x7f\x3e\x21'], False, [TP], False),
  ([b'\x7f\x22\x21'], True, [TP], False),
  ([b'\x7f\x3e\x31'], True, [TP], False),
  ([b'\x7f\x3e\x21'], True, [TP, TP], False),
])
def test_eps_busy_retry(replies, enabled, expected_sends, matched):
  pending = deque(replies)
  inbox = []
  sent = []
  clock = 0.

  def send(frames):
    for f in frames:
      assert f.address == EPS[0] and f.src == 0
      payload = f.dat[1:1 + f.dat[0]]
      sent.append((clock, payload))
      reply = (pending.popleft() if pending else None) if payload == TP else b'\x62\xf1\x88firm'
      if reply is not None:
        inbox.append(CanData(0x738, (bytes([len(reply)]) + reply).ljust(8, b'\x00'), 0))

  def receive(wait_for_one=False):
    nonlocal clock
    clock += .01
    result = inbox.copy()
    inbox.clear()
    return [result]

  options = {'retry_busy_addrs': {EPS}} if enabled else {}
  with patch('opendbc.car.isotp_parallel_query.time.monotonic', lambda: clock):
    query = IsoTpParallelQuery(send, receive, 0, [EPS], [TP, READ], [b'\x7e\x00', b'\x62\xf1\x88'], **options)
    data = query.get_data(.1, total_timeout=1.)
  assert [x[1] for x in sent] == expected_sends
  assert data == ({EPS: b'firm'} if matched else {})
  tester_times = [t for t, payload in sent if payload == TP]
  assert all(b - a >= .15 for a, b in zip(tester_times, tester_times[1:], strict=False))


def test_retry_only_enabled_for_the_ford_bus_zero_software_query():
  from opendbc.car.fw_versions import get_fw_versions

  calls = []

  class Query:
    def __init__(self, send, recv, bus, addrs, request, response, offset, *, retry_busy_addrs):
      if retry_busy_addrs:
        calls.append((bus, request, retry_busy_addrs))

    def get_data(self, timeout):
      return {}

  with patch('opendbc.car.fw_versions.IsoTpParallelQuery', Query):
    get_fw_versions(lambda _: None, lambda **_: [], lambda _: None, query_brand='mazda')
    assert not calls
    get_fw_versions(lambda _: None, lambda **_: [], lambda _: None, query_brand='ford')
  assert calls == [(0, [TP, READ], {EPS})]
