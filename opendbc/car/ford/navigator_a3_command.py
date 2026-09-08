"""Separate experimental command choice from immutable production inhibition.

No transport, environment switches or physical-permission claims live here.
"""
from dataclasses import dataclass

Frame = tuple[int, bytes, int]


@dataclass(frozen=True)
class CommandSelection:
  production_frame: Frame | None
  experimental_frame: Frame | None
  production_inhibited: bool


def select_command(mode: str, scheduled_frame: Frame | None, proposal: Frame | None) -> CommandSelection:
  if mode not in ('a2', 'shadow', 'requested'):
    raise ValueError('Unsupported command strategy')
  if mode == 'requested' and scheduled_frame is not None:
    address, d, _ = scheduled_frame
    if (address != 0x3d6 or len(d) != 8 or (d[0] >> 4) & 7
        or (d[2] << 3 | d[3] >> 5) != 1000
        or ((d[3] & 31) << 6 | d[4] >> 2) != 1000
        or ((d[4] & 3) << 8 | d[5]) != 512
        or (d[6] << 3 | d[7] >> 5) != 1024):
      raise ValueError('Requested A3 production output must remain inactive and neutral')
  return CommandSelection(scheduled_frame, scheduled_frame if mode == 'a2' else proposal, mode == 'requested')
