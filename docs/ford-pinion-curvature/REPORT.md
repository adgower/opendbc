# Expedition CAN-FD pinion measurement evidence

Observed signals from two complete recorded routes support evaluating pinion-derived curvature as a narrowly scoped measurement source. This evidence does not diagnose a failed yaw sensor and does not establish candidate driving performance.

## Reproduction and conventions

Run the configurable extraction, analysis and report commands in `README.md`. Requires private raw logs and matching local schemas; SHA-256 hashes, exact paths, raw CAN segment bounds, CP geometry and model source hash are in `evidence.json`. Parsing decompresses one segment at a time.

Pinion: bus 0, 0x7e, StePinComp_An_Est = (((byte2 & 127) << 8) | byte3) × 0.1 − 1600 degrees; quality bits byte5[3:2] = 3. Ford yaw: bus 0, 0x91, bytes2:3 big endian × 0.0002 − 6.5 rad/s; quality byte6[5:4] = 3. Pinion yaw = VehicleModel(CP).yaw_rate(radians(pinion), vEgoRaw, 0). Recorded fixed CP: 19.6000003815 steering ratio, 3.1115000248 m wheelbase; stiffness factor 1, no learned offset, ratio, stiffness or roll. The copied model source is executed unchanged mathematically; only dependency/type imports are replaced to avoid incompatible Cap’n Proto schemas.

Comma: project angularVelocityDevice onto calibrated vertical using extrinsics rpyCalib at publication time, then negate to Ford steering/yaw sign convention. Compare at deviceMotion.timestamp, using last CAN sample ≤ measurement time, both CAN ages <50 ms, quality=3, and valid/calibrated deviceMotion. No fitted slope, bias, sign optimization, lag shift or smoothing. Coordinate sign is fixed; raw unnegated Comma is opposite Ford.

## Whole-route numerical comparison

Yaw residuals are source − sign-aligned Comma, rad/s. Moving means vEgoRaw >5 m/s.

| Route | samples | pinion correlation | pinion residual median / RMSE | raw yaw correlation | raw yaw residual median / RMSE |
|---|---:|---:|---:|---:|---:|
| 00000007--ebffff8226 | 17210 | 0.996140 | -0.001762 / 0.009614 | 0.999667 | -0.017647 / 0.017817 |
| 00000004--e3907c22d6 | 14132 | 0.995713 | -0.001233 / 0.009711 | 0.999578 | -0.017978 / 0.018199 |

The raw yaw tracks turn dynamics more tightly but has a persistent negative level difference relative to Comma; pinion has lower bias and larger dynamic/model error. Neither correlation alone establishes which source is safe to substitute. Fixed geometry, steering compliance, tire slip, road bank, sensor calibration and timestamp age can affect residuals.

| Route | source | curvature median / RMSE (1/m) | p90 absolute | samples >0.002 |
|---|---|---:|---:|---:|
| 00000007--ebffff8226 | pinion | -0.0000872 / 0.0008815 | 0.0009049 | 737 / 17210 |
| 00000007--ebffff8226 | raw_yaw | -0.0008013 / 0.0012822 | 0.0019329 | 1576 / 17210 |
| 00000004--e3907c22d6 | pinion | -0.0000533 / 0.0008488 | 0.0008436 | 399 / 14132 |
| 00000004--e3907c22d6 | raw_yaw | -0.0007094 / 0.0011085 | 0.0015803 | 710 / 14132 |

These curvature residuals divide yaw-rate difference by measured speed. The 0.002 comparison is descriptive, not the Panda command-deviation check; Panda uses its own speed, quantization, sample range and timing. No limit was relaxed.

## Exact rejected-command windows

Dongle `0e09a1daaf2c4fd2`; route `00000007--ebffff8226`. Route seconds use initial initData.logMonoTime (38.510349203 s). Segment-local zero is the first raw CAN logMonoTime in that file. Repeated old initData in later segments is excluded from segment timing. Each window is exactly 10 s before/after the logged rejected frame. Cross-boundary coverage is calculated from raw CAN ranges, never segment number ×60. Viewer access was not checked, so these are explicit identifiers, not guessed viewer URLs.

| Event | brake / driver | logMonoTime | route s | segment : local s | ±10 s window coverage (segment : local start–end) |
|---|---|---:|---:|---|---|
| reject-1 | False / False | 202.314281640 | 163.803932437 | 2 : 42.791164931 | 2 : 32.791164931–52.791164931 |
| reject-2 | False / False | 202.616612213 | 164.106263010 | 2 : 43.093495504 | 2 : 33.093495504–53.093495504 |
| reject-3 | True / False | 503.185394859 | 464.675045656 | 7 : 43.655655192 | 7 : 33.655655192–53.655655192 |
| reject-4 | False / False | 696.169303275 | 657.658954072 | 10 : 56.623135499 | 10 : 46.623135499–59.978024872; 11 : 0.000000000–6.632861824 |
| reject-5 | False / False | 696.871217493 | 658.360868290 | 10 : 57.325049717 | 10 : 47.325049717–59.978024872; 11 : 0.000000000–7.334776042 |
| reject-6 | False / False | 718.184795558 | 679.674446355 | 11 : 18.648354107 | 11 : 8.648354107–28.648354107 |
| reject-7 | True / True | 739.145982425 | 700.635633222 | 11 : 39.609540974 | 11 : 29.609540974–49.609540974 |
| reject-8 | False / False | 1186.221847254 | 1147.711498051 | 19 : 6.673585414 | 18 : 56.684470235–60.000432426; 19 : 0.000000000–16.673585414 |

| Event ±10 s | pinion residual median / RMSE (rad/s) | raw yaw residual median / RMSE (rad/s) |
|---|---:|---:|
| reject-1 | -0.007642 / 0.011514 | -0.017198 / 0.017823 |
| reject-2 | -0.007592 / 0.011423 | -0.017196 / 0.017804 |
| reject-3 | -0.006719 / 0.011095 | -0.018214 / 0.018487 |
| reject-4 | 0.003103 / 0.017615 | -0.017387 / 0.016606 |
| reject-5 | 0.002708 / 0.017241 | -0.017455 / 0.016661 |
| reject-6 | 0.001085 / 0.003891 | -0.017788 / 0.017860 |
| reject-7 | 0.001769 / 0.005611 | -0.017804 / 0.017721 |
| reject-8 | 0.001541 / 0.007694 | -0.018128 / 0.018280 |

## Comparable turns without recorded rejection

Deterministic selection scans one-second centers with full ±10 s windows: calibrated valid measurements, speed >5 m/s, latActive throughout, no brake, center |Comma yaw| >0.03 rad/s, and no recorded active 0x3d6 bus192 frame within the window. Driver steering is allowed and not treated as a sensor fault. Centers are separated by >25 s. Postfix choices rank closeness of speed and absolute Comma yaw to non-brake rejected events; prior-route choices use evenly spaced reference times. This is observational comparison, not a randomized matched study. Full selection scores and per-window distributions are in JSON.

| Route | turn | center route s | segment : local s | pinion residual median / RMSE | raw yaw residual median / RMSE | ±10 s coverage (segment : local start–end) |
|---|---|---:|---|---:|---:|---|
| 00000007--ebffff8226 | clean-turn-1 | 578.489651 | 9 : 37.468555 | -0.000926 / 0.002318 | -0.017691 / 0.017718 | 9 : 27.468554962–47.468554962 |
| 00000007--ebffff8226 | clean-turn-2 | 179.489651 | 2 : 58.476883 | -0.016012 / 0.019505 | -0.019366 / 0.019450 | 2 : 48.476883291–59.991196852; 3 : 0.000000000–8.475583471 |
| 00000007--ebffff8226 | clean-turn-3 | 732.489651 | 12 : 11.462300 | 0.003616 / 0.011139 | -0.018377 / 0.018443 | 12 : 1.462299979–21.462299979 |
| 00000007--ebffff8226 | clean-turn-4 | 205.489651 | 3 : 24.475583 | -0.011320 / 0.019049 | -0.017298 / 0.018170 | 3 : 14.475583471–34.475583471 |
| 00000004--e3907c22d6 | clean-turn-1 | 508.684073 | 8 : 27.685525 | 0.001000 / 0.012215 | -0.018639 / 0.018955 | 8 : 17.685524755–37.685524755 |
| 00000004--e3907c22d6 | clean-turn-2 | 249.684073 | 4 : 8.655848 | -0.003916 / 0.006891 | -0.018178 / 0.018585 | 3 : 58.671490057–60.005343571; 4 : 0.000000000–18.655848257 |
| 00000004--e3907c22d6 | clean-turn-3 | 299.684073 | 4 : 58.655848 | 0.024063 / 0.025126 | -0.016218 / 0.016431 | 4 : 48.655848257–59.966870133; 5 : 0.000000000–8.678487759 |
| 00000004--e3907c22d6 | clean-turn-4 | 536.684073 | 8 : 55.685525 | -0.000524 / 0.003692 | -0.018019 / 0.018075 | 8 : 45.685524755–59.998612582; 9 : 0.000000000–5.676993111 |

## Timing and limits

- 00000007--ebffff8226: device publication minus motion timestamp median 24.153 ms, p10/p90 19.834/27.965 ms; accepted pinion/yaw sample-age medians 5.247/5.223 ms.
- 00000004--e3907c22d6: device publication minus motion timestamp median 24.161 ms, p10/p90 19.916/27.969 ms; accepted pinion/yaw sample-age medians 5.088/5.129 ms.

CAN log timestamps are receive/publication times, not ECU sensor sample times. Same-frame state and exact firmware RX-before-TX order remain uncertain. The nearest motion sample at each rejection is retained with its signed time delta, not asserted simultaneous. The six non-brake and two brake-associated frames remain separately identifiable; brake-associated rejection can reflect controls permission timing.

Changing measurement also changes future controller commands. This report compares recorded measurements only and does not replace CAN feedback while preserving old commands to claim candidate success. A separately identified paired replay must regenerate commands with the actual controller and compare identical inputs/state. Even that remains an open-loop replay, without counterfactual vehicle motion or road validation. The production scope under evaluation is Expedition CAN-FD only; this dataset does not justify other Ford platforms.
