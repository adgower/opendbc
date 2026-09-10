# Expedition CAN-FD fixed-model pinion curvature

This candidate replaces the curvature measurement used by the Ford lateral controller and Panda for `FORD_EXPEDITION_MK4` CAN-FD. It estimates curvature from the existing absolute steering-pinion angle (`SteeringPinion_Data`, bus 0, `0x7e`). It does not select the compensated yaw signal at `0x77`.

## Source and scope

- Base: `adgower/opendbc` `eb70bef4f56d2e9e91cb5a0a8374890a6acfaf51`, branch `navigator-angle-v1`.
- Matching host base: `adgower/openpilot` `f35e6e5eb11ca9cddad3ed945d44d8f191679240`.
- Panda source remains `75aa44bec9140849868239b1f1e3f22624adb8fe`; firmware must be rebuilt with this opendbc safety source.
- Donor reference: BluePilot `bp-7.0` at `e1d051d7ba270261b4455068bd68f1a58db15a4a`, `opendbc_repo/opendbc/sunnypilot/car/ford/lateral_curv_ext.py`, `opendbc_repo/opendbc/sunnypilot/car/interfaces.py`, and `opendbc_repo/opendbc/safety/modes/ford.h`.

The donor's UI calls this “Use Pinion Yaw Sensor.” Only the signal-selection concept and signal decoding are used here. Its UI/Params framework, different Expedition geometry, learned corrections, and increased curvature allowance are not imported.

## Observable selection

`CarInterface` sets `FordFlags.PINION_CURVATURE` (bit `2`) and `FordSafetyFlags.PINION_CURVATURE` (bit `8`) only for Expedition CAN-FD. These are initialization-time flags, visible in recorded `carParams`. The controller also logs `Ford pinion curvature enabled` with the geometry, flags and safety parameter. Other safety bits remain intact; CAN-FD with longitudinal control normally changes safety parameter `3` to `11`.

The flag is on by default for this fingerprint in the candidate. There is no UI switch. Raw `CarState.yawRate` remains available unchanged for analysis.

## Matching calculation

Python uses a private `VehicleModel(CP)` frozen at initialization. Both the angle path and curvature fallback use the same helper. Panda uses the corresponding fixed model coefficients. Tests compare them across speeds and positive/negative pinion angles within CAN quantization.

| Model input | Value |
|---|---:|
| Wheelbase | 3.1115 m |
| Steering ratio | 19.6 |
| Mass including standard cargo | 2878 kg |
| Center to front | 0.44 × wheelbase |
| Front tire stiffness | 353037.28125 N/rad |
| Rear tire stiffness | 438491.46875 N/rad |

These are the existing branch's geometry assumptions, not a new road calibration. No learned ratio/stiffness, roll correction, or angle offset is applied. Controller sign is the negative of the Ford wire convention. Host and Panda consume their own speed snapshots, so equal formulas do not guarantee equal samples at every instant.

## Safety boundaries

Panda adds a 100 Hz pinion RX check with exact bus/length, quality `3`, and the 4-bit rolling counter. Active steering is blocked before the first valid sample, after invalid quality or excessive counter faults, and when the sample is older than the existing native 1-second liveness tolerance. Missing pinion never falls back silently to yaw. Existing yaw RX checks remain registered.

The pinion checksum algorithm is unknown and is not validated, matching the donor limitation. This remains an explicit review concern.

The curvature-error allowance remains **0.002 1/m**. Increasing it to `0.003` would allow 50% more disagreement between requested/shadow curvature and Panda's measured range; that is a separate safety-bound change, not a sensor correction. This candidate does not make that change. HIGH/LOW factors, steering limits, engagement order, takeover behavior and installed message cadence remain unchanged.

## Validation and limits

Run from the opendbc repository root with a compatible Python environment:

```sh
PYTHONPATH="$PWD" python -m unittest \
  opendbc.car.ford.tests.test_pinion_curvature \
  opendbc.car.ford.tests.test_angle_engagement \
  opendbc.car.tests.test_ford_lateral_angle \
  opendbc.safety.tests.test_ford -q
```

The focused run completed 186 tests, with 14 existing skips. Coverage includes platform selection, fixed-model parity, flag-off behavior, malformed/missing/stale input, quality and counter faults, braking, engagement/re-engagement, driver takeover, and excessive requests. Explicit boundaries accept ±0.002 and reject ±0.0021 in the controlled shadow-curvature test. Independent source/test review found no actionable findings.

The matching debug H7 firmware and ARM64 `pandad` target built offline. H7 text/data/BSS are 56284/23728/435352 bytes, changes of +396/+88/+8 bytes versus the preserved baseline. H7 signed artifact SHA-256: `4b101fd08cd042a7d419d83a30b5bd12157a1e830cee59804fd4177464e748ee`. Host SHA-256: `2028d13694e1ee103563b4bfafe13e4c19d1e4905c8b93cb10c515abd6c2dd80`.

The complete latest-route controller replay reproduces all 23,413 baseline LMC2 messages and all 39,027 LKA metadata messages byte-for-byte, with zero schedule differences. The candidate changes 814 LMC2 and 10,405 LKA payloads with the same schedule.

The native timing sweep is not a uniformly successful fix. At TX offsets 0/+5/+10/+20 ms it reduces rejections without newly rejecting baseline-accepted commands in the tested windows. At -10 ms it newly rejects two commands and accepts two previously rejected commands, leaving the total unchanged. These are artificial timing variations, not measured MCU latency. Keep this candidate in draft while reviewing these outcomes.

See the adjacent recorded-signal report and replay report for before/after evidence and exact route windows. Software tests and builds do not establish hardware timing or vehicle behavior. No device installation, firmware flash, or candidate drive was performed. Diagnostic instrumentation and the fresh-metadata experiment remain separate and unpublished.
