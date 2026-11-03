"""Research-only historical-rate and recent observed-progress hypothesis.

No clearance authority. Quantization bounds are representational assumptions,
not a calibrated physical sensor accuracy claim.
"""
from opendbc.car.ford.longitudinal_coordination import BrakeReleaseProgress

class HistoricalProgress(BrakeReleaseProgress):
  def expired(self, now, started, torque):
    if started != self.episode:
      self.episode = started
      self.samples = []
    if torque.t >= started and (not self.samples or torque.t != self.samples[-1][0]):
      self.samples.append((torque.t, torque.value))

    elapsed = now - started
    if elapsed + 1e-9 < 1.0:
      return False
    if elapsed + 1e-9 >= 2.0:
      return True

    initial = [value for timestamp, value in self.samples if 0 <= timestamp - started <= 0.25]
    recent = [(timestamp, value) for timestamp, value in self.samples if now - 0.25 <= timestamp <= now]
    if not initial or len(recent) < 3 or recent[-1][0] - recent[0][0] < 0.15 - 1e-9:
      return True
    if now - torque.t > 0.05 or any(b[0] - a[0] > 0.050000001 for a, b in zip(recent, recent[1:], strict=False)):
      return True

    peak = max(initial)
    if peak <= 200.0:
      return super().expired(now, started, torque)
    # Require substantial overall release and continuing recent progress.
    if peak < 40.0 or torque.value > 0.6 * peak:
      return True
    if recent[0][1] - recent[-1][1] < 8.0:
      return True

    historical = [(timestamp, value) for timestamp, value in self.samples if now - 0.5 <= timestamp <= now]
    if len(historical) < 3 or historical[-1][0] - historical[0][0] < 0.4 - 1e-9:
      return True
    if any(b[0] - a[0] > 0.050000001 for a, b in zip(historical, historical[1:], strict=False)):
      return True
    span = historical[-1][0] - historical[0][0]
    if historical[0][1] - historical[-1][1] - 8.0 < max(8.0, 0.04 * (peak + 4.0) * span / 0.25):
      return True

    # A rebound and return to an old minimum must not recycle progress forever.
    lowest = float('inf')
    last_new_low = None
    for timestamp, value in self.samples:
      if value < lowest:
        lowest = value
        last_new_low = timestamp
    if last_new_low is None or now - last_new_low > 0.25 + 1e-9:
      return True

    floor = recent[0][1]
    for _, value in recent:
      if value - floor > max(8.0, 0.02 * peak):
        return True
      floor = min(floor, value)
    return False
