from opendbc.can import CANParser


# Ford's event-periodic side status has a 200 ms heartbeat.
MAX_BLINDSPOT_AGE_NS = 450_000_000


class BlindspotStatus:
  def __init__(self, message, side: str):
    self.size = message.size
    self.system = message.sigs[f"Sod{side}_D_Stat"]
    self.sensor = message.sigs[f"SodSns{side}_D_Stat"]
    self.detection = message.sigs[f"SodDetct{side}_D_Stat"]
    self.lamp = message.sigs[f"SodAlrt{side}_D_Stat"]
    self.clear_ns = 0
    self.block_ns = 0

  def block(self, nanos: int):
    self.clear_ns = 0
    self.block_ns = max(nanos, self.block_ns)

  def expire(self, nanos: int):
    if self.clear_ns and nanos - self.clear_ns > MAX_BLINDSPOT_AGE_NS:
      self.block(self.clear_ns + MAX_BLINDSPOT_AGE_NS)

  def observe(self, nanos: int, data: bytes):
    if (len(data) == self.size and self.system.get_raw_value(data) == 2 and self.sensor.get_raw_value(data) == 0 and
        self.detection.get_raw_value(data) == 0 and self.lamp.get_raw_value(data) == 0):
      # Multiple CAN frames can share one receive timestamp. A simultaneous
      # warning wins regardless of their order in the batch.
      if nanos > self.block_ns:
        self.clear_ns = nanos
    else:
      self.block(nanos)


class NavigatorBlindspotCamera(CANParser):
  def __init__(self, dbc_name, messages, bus):
    super().__init__(dbc_name, messages, bus)
    left = self.dbc.name_to_msg["Side_Detect_L_Stat"]
    right = self.dbc.name_to_msg["Side_Detect_R_Stat"]
    self.left = BlindspotStatus(left, "Left")
    self.right = BlindspotStatus(right, "Right")
    self.status_by_address = {left.address: self.left, right.address: self.right}
    self.last_observation_ns = 0

  def update(self, strings, sendcan=False):
    updated = super().update(strings, sendcan)
    if sendcan:
      return updated
    # Read the original envelopes: vl/ts_nanos retain only the final packet,
    # so they cannot timestamp a warning followed by clear in one update.
    entries = [strings] if strings and not isinstance(strings[0], (list, tuple)) else strings
    for nanos, frames in entries:
      if nanos < self.last_observation_ns or nanos <= 0:
        self.left.block(self.last_observation_ns)
        self.right.block(self.last_observation_ns)
        continue
      self.last_observation_ns = nanos
      self.left.expire(nanos)
      self.right.expire(nanos)
      for address, data, src in frames:
        if src == self.bus and (side := self.status_by_address.get(address)) is not None:
          side.observe(nanos, data)
    return updated
