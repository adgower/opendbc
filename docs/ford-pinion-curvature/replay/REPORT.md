# Paired controller and native safety replay

Route `00000007--ebffff8226`, all 20 supplied rlog segments. Candidate source `131f73e98bd20dede2744906f1be97421ebdeb16`.
Baseline `ford.h` in the immutable build snapshot was byte-verified against commit `eb70bef4f56d2e9e91cb5a0a8374890a6acfaf51`. Full source/log SHA-256 provenance is in `paired-native.json`.

## Reconstruction verification

The extraction process loads the recorded log schema and writes plain Python dictionaries. A separate process loads only the selected opendbc schema and regenerates controller output from nearest preceding full recorded carState and carControl. The initial steering counter comes from DBC `LatCtlPath_No_Cnt` (60|4): `(data[7] >> 1) & 15`. It supplies the initial frame phase; there is no subsequent fitting.

Baseline: **117,061 updates; 23,413/23,413 LMC2 and 39,027/39,027 LKA messages exactly matched recorded payloads and bus**, with zero schedule mismatches. This verifies the lateral reconstruction on this route.

Candidate: 814 LMC2 messages and 10,405 LKA messages changed; message schedules remained identical. Candidate TX is regenerated using the pinion controller and paired with pinion safety (parameter 11); baseline TX uses the baseline controller and baseline safety (parameter 3). No recorded steering command is silently substituted for candidate output.

## Six-window native safety results

Each offset processes 854 LMC2 requests across the six previously identified raw-CAN windows. Controls permission is driven by normal RX hooks; it is never forced. The table counts rejected LMC2 messages. Parentheses show rejections while `controls_allowed` was true immediately before TX.

| TX offset vs logged RX | Baseline rejected | Candidate rejected |
|---:|---:|---:|
| -10 ms | 122 (3) | 122 (3) |
| +0 ms | 7 (5) | 5 (3) |
| +5 ms | 7 (5) | 5 (3) |
| +10 ms | 11 (9) | 5 (3) |
| +20 ms | 12 (10) | 5 (3) |

The candidate has fewer reconstructed rejections at 0, +5, +10 and +20 ms, and the same total at -10 ms. This is **not uniformly better**: at -10 ms, window 4 increases from 1 to 3 rejected requests, while two other windows each decrease by one. The large -10 ms total is dominated by controls-disallowed ordering, including 115 rejected active requests in window 0 for both variants.

| Window | Logged monotonic interval (s) |
|---:|---|
| 0 | 197.314281640–204.616612213 |
| 1 | 498.185394859–505.185394859 |
| 2 | 691.169303275–698.871217493 |
| 3 | 713.184795558–720.184795558 |
| 4 | 734.145982425–741.145982425 |
| 5 | 1181.221847254–1188.221847254 |

## Interpretation limits

- Controller reconstruction uses nearest preceding full logged CS and CC; zero-filled stock UI/button messages are excluded from comparison.
- All baseline LMC2 and LKA payloads and schedule exactly match their recorded counterparts, validating the lateral reconstruction on this route.
- Native safety initializes separately at each seven-second window; it has no exact pre-window MCU state. No controls_allowed override is used.
- No periodic MCU safety tick, interrupt scheduling, SPI transport or vehicle feedback is modeled. RX quality/counter hooks and per-TX pinion freshness gate do run.
- The five offsets perturb TX against fixed RX log timestamps; none is asserted to equal MCU ordering.
- Candidate commands are regenerated but physical feedback remains the recorded baseline drive: this is counterfactual software replay, not a closed-loop vehicle outcome.

Native libraries are compiled into unique paths per variant, window and offset so macOS does not reuse one loaded image. Original non-lateral TX and raw physical RX are retained; baseline regenerated steering is asserted equal to each recorded steering packet before replay. Raw bus-128+ TX echoes are excluded from RX. Sources are uninstrumented except a read-only native state accessor.

These are software reconstruction results, not Panda hardware timing, closed-loop simulation or road-validation evidence. They support a narrower claim: on these recorded inputs, the approved pinion change reduces several timing-sensitive reconstructed rejections without eliminating them.

## Files

- `extract_controls.py`: standalone log-schema extraction.
- `controller_replay.py`: schema-isolated baseline/candidate regeneration.
- `paired_native.py`: paired regenerated-TX/native-safety harness.
- `baseline.pkl`, `candidate.pkl`: generated and recorded per-batch commands.
- `paired-native.json`: all decisions, pre-TX native state, case counts and provenance.
- `baseline.log`, `candidate.log`, `paired-native.log`: execution output.

## Observed targets and acknowledged neighbors

Each target is matched to exactly one byte-identical recorded send preceding its bus-192 return within 40 ms. Times below are the original target times relative to route init. Cells are **baseline / candidate** decisions; A means accepted, R rejected.

| Target seconds | Brake | -10 ms | 0 ms | +5 ms | +10 ms | +20 ms |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 163.804 | no | R / R | A / A | A / A | R / A | R / A |
| 164.106 | no | R / R | A / A | A / A | R / A | R / A |
| 464.675 | yes | A / A | R / R | R / R | R / R | R / R |
| 657.659 | no | A / A | R / A | R / A | R / A | A / A |
| 658.361 | no | A / A | A / A | A / A | R / A | A / A |
| 679.674 | no | A / A | R / A | R / A | R / A | R / A |
| 700.636 | yes | A / A | R / R | R / R | R / R | R / R |
| 1147.711 | no | A / A | A / A | A / A | R / A | R / A |

At +10 ms, baseline reproduces all eight observed target rejections. Candidate accepts the six non-braking target commands and retains both brake-related rejections. Its five total window rejections therefore include two targets plus three other requests; target counts must not be confused with all-window totals.

At -10 ms, both variants accept the two brake targets because shifted TX precedes brake RX. Both also have controls disallowed for the first two targets. This adverse offset fails recorded decision parity and must not be read as evidence of actual brake bypass.

Across the warmed windows (at least two seconds after each native initialization), 502 active requests have exactly one recorded byte-identical bus-128 acknowledgement within 40 ms. At 0, +5 and +10 ms neither variant rejects any of these; at +20 ms baseline rejects three and candidate rejects zero. At -10 ms each rejects 105, but the sets differ.

**Two new candidate rejections** occur only at -10 ms: relative send times **698.117646400 s** and **698.422472702 s**, both in window 4 after 2.482 s and 2.787 s of warmup, respectively. Both were acknowledged on the original drive and accepted by baseline replay. Candidate and baseline both have controls allowed at these requests. These regressions remain visible despite unchanged aggregate -10 ms rejection totals.

For visual review, these are segment **11 at 37.091554152 s and 37.396380454 s**, respectively, using the first raw CAN event as local zero. Their ±10-second windows are **segment 11, 27.091554152–47.091554152 s** and **27.396380454–47.396380454 s**. These were accepted on the recorded drive; the rejections exist only in the candidate's artificial -10 ms replay.

Independent comparison to the previous diagnostic-parity run found **zero baseline decision differences across all 4,270 offset/request pairs**. Original packet ordering and all unchanged TX packets are retained; baseline replacements are asserted byte-identical. The portable raw extractor reproduced the earlier cache exactly: every event, sequence, batch, window and identity record.

Compact results: `summary.json` includes per-target decisions, warmed-neighbor counts and the two new rejections with their native pre-TX state. The larger `paired-native.json` is reproducible output and need not be committed.
