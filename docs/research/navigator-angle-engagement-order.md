# Navigator Angle engagement ordering

Base: opendbc `58d7dba5bb0f3f6cab1c17010228931a14f0ab7a`, used by openpilot `143f4c80e992d009f4102985f7cc1b45769af097`.

## Fixed locally

The controller previously emitted an active LateralMotionControl2 request before updating Panda's Angle-mode metadata. This occurred both on coincident 20/33 Hz frames and on engagement between regular metadata frames. Panda therefore interpreted the first nonneutral angle under curvature-mode rules and rejected it. A following command could also fail because the first request had not initialized the Angle history.

Send true mode metadata before the first active request. On disengagement, send neutral steering before false metadata so the existing safety handler clears its angle history. Keep regular 33 Hz metadata updates and add a transition update only when necessary; never duplicate metadata in a frame. This adds a metadata packet on off-cadence transitions, so the steady cadence is preserved but transition spacing changes. No gains, yaw decoding, safety checks, or takeover thresholds were changed.

## Verification

- Existing controller/strategy tests: 30 passed before changes.
- New compiled-safety integration tests reproduced 20 failing subcases before the initial correction.
- Review exposed stale angle history on high-speed reengagement; 18 additional subcases failed before the final correction.
- Final controller/strategy/Ford tests: 35 passed, including five new tests with 36 parameterized engagement/reengagement combinations.
- Ford safety suite: 138 tests, 14 skipped, no failures.
- New integration tests followed by the safety suite in the same process: 143 tests, 14 skipped, no failures.
- Ruff on changed Python files and git diff --check passed.
- Independent review confirmed the high-speed regression was resolved and found no remaining actionable issue in this bounded fix.

Reproduction (from this checkout, with the repository's Python dependencies installed):

```sh
python -m unittest opendbc.car.tests.test_ford_lateral_angle opendbc.car.ford.tests.test_ford opendbc.car.ford.tests.test_angle_engagement -q
python -m unittest opendbc.car.ford.tests.test_angle_engagement opendbc.safety.tests.test_ford -q
ruff check opendbc/car/ford/carcontroller.py opendbc/car/ford/tests/test_angle_engagement.py
```

## Drive evidence and remaining work

Route `00000006--45c575552b`: 28 segments, 1624.55 seconds. Raw logs contain 39 rejected active steering messages; 34 occurred around 22–27 mph. The six segments containing all these events were copied read-only for diagnosis (0, 4, 5, 6, 22, 25).

A diagnostic wrapper compiled the unmodified safety source and inspected the recorded RX/TX stream. Four first-engagement requests encountered false Angle metadata; the second request at the initial engagement was also rejected in replay. Two rejects coincided with controls no longer allowed. For the remaining 32 mode-active/controls-allowed cases, the saved request shadow curvature was beyond the reconstructed check boundary by the time the rejection was logged. Only seven failed the shadow predicate at send-log time. This strongly supports stale metadata / measurement timing as the next hypothesis.

These are not 39 proven firmware rejection reasons: logger batching does not preserve exact MCU processing time, replay starts at separate segment boundaries, and the flashed firmware signature was not established. Complete exact-order parity remains unproven. The ordering patch does not solve the steady-state shadow-curvature timing issue.

The full-route read-only scan also found stationary Ford yaw median -0.0172 rad/s in raw CAN decoding; calibrated Comma yaw was about -0.00018 rad/s. The Ford quality flag was OK for all 162490 main-bus samples. The offset exists before controller calculations; the Ford signal definition/calibration still needs validation. No bias subtraction was implemented.

Local diagnostic inputs, wrapper, replay script, and per-event results: `/private/tmp/navigator-angle-followup/`. These paths are temporary local evidence, not portable dependencies of the regression tests.

## Installation status

Local candidate only. Nothing was pushed, installed, restarted, or flashed. The application gitlink still pins the original opendbc revision. Tests establish software behavior for the covered cases; there has been no validation of this patch on the vehicle. Preserve the current device baseline and treat any device installation as a separate step.
