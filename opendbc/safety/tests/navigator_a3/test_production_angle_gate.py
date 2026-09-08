"""Firmware gate tests: final encoded values, independent of host controller models."""
import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]


class TestProductionAngleGate(unittest.TestCase):
  def test_final_encoded_angle_gate(self):
    # The fallback models the missing independent gate, allowing an assertion
    # failure before production integration without depending on compiler errors.
    source = '''
#include "opendbc/safety/tests/libsafety/safety.c"
bool final_angle_violation(unsigned int angle) {
#ifdef FORD_FINAL_PATH_ANGLE_GATE
  return ford_final_path_angle_checks(angle);
#else
  return false;
#endif
}
bool transmit_angle(unsigned int raw, unsigned int param, unsigned int mode, bool allowed) {
  set_safety_hooks(SAFETY_FORD, param);
  controls_allowed = allowed;
  CANPacket_t p = {0};
  p.data_len_code = 8;
  if (param & 2U) {
    p.addr = FORD_LateralMotionControl2;
    p.data[0] = mode << 4;
    p.data[2] = 1000U >> 3;
    p.data[3] = ((1000U & 7U) << 5) | (raw >> 6);
    p.data[4] = ((raw & 63U) << 2) | 2U;
    p.data[6] = 1024U >> 3;
  } else {
    p.addr = FORD_LateralMotionControl;
    p.data[0] = 1000U >> 3;
    p.data[1] = ((1000U & 7U) << 5) | (4096U >> 8);
    p.data[3] = raw >> 3;
    p.data[4] = ((raw & 7U) << 5) | (mode << 2);
    p.data[5] = 512U >> 2;
  }
  return safety_tx_hook(&p);
}
'''
    with tempfile.TemporaryDirectory() as tmp:
      c = Path(tmp) / 'gate.c'
      so = Path(tmp) / 'gate.so'
      c.write_text(source)
      for name, flags in (('release', []), ('debug', ['-DALLOW_DEBUG'])):
        so = Path(tmp) / f'{name}.so'
        subprocess.run(['cc', '-shared', '-fPIC', '-std=gnu11', '-Wno-pointer-to-int-cast', *flags, '-I', str(ROOT), str(c), '-o', str(so)], check=True)
        lib = ctypes.CDLL(str(so))
        gate = lib.final_angle_violation
        gate.argtypes = [ctypes.c_uint]
        gate.restype = ctypes.c_bool
        for raw in range(2048):
          self.assertEqual(gate(raw), raw != 1000, f'final 11-bit angle {raw}')
        self.assertTrue(gate(2048))
        self.assertTrue(gate(0xffffffff))
        tx = lib.transmit_angle
        tx.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_bool]
        tx.restype = ctypes.c_bool
        for param in (0, 1, 2, 3, 0xffff):
          for mode in range(8):
            for allowed in (False, True):
              for raw in range(2048):
                if raw != 1000:
                  self.assertFalse(tx(raw, param, mode, allowed), (raw, param, mode, allowed))
              self.assertEqual(tx(1000, param, mode, allowed), allowed or mode == 0)


if __name__ == '__main__':
  unittest.main()
