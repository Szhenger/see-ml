"""Tests for tool/bench_compare.py, the Tier A regression gate (P5, #79).

Standard library only, like the tool. The reports are synthetic but shaped
like seeml-bench's (schema 3 without, schema 4 with the calibration block),
and the two incident replays use the recorded runner timings: Nightly
#20-#24 (bench jobs of 264-315 s against 230-235 s green ones, all on
commit 9768709) and 09-12 -> 09-14 (a uniform -12.6..-22% across all 18
keys on cbdc5b2).

    python3 -m unittest discover -s test/tool -p '*_test.py'
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tool"))

import bench_compare  # noqa: E402

FIXTURES = {"mlp_64x512x3": 14000.0, "mlp_128x1024x2": 8800.0,
            "dec_v256_d128_s32": 6900.0, "dec_v256_d128_s128": 6400.0,
            "dec_v512_d192_s32": 4100.0, "tok_v64_d64_s16": 52000.0}
THREADS = ("1", "2", "4")  # 6 fixtures x 3 widths = the nightly's 18 keys
CALIB = 61000.0


def wobble(i, amplitude):
    """A deterministic per-key noise term in [-amplitude, +amplitude]."""
    return amplitude * (((i * 37) % 11) / 5.0 - 1.0)


def report(runner=1.0, schema=4, noise=0.03, night=0, regress=None,
           per_key=None, calibration=True, kernel="sgemm_ikj_128_f32_v1",
           policy=None):
    """One night's report on a runner `runner` times the reference speed.
    `regress` {fixture: factor} is a REAL regression (the calibration
    kernel does not see it); `per_key` overrides the runner factor per key
    index (a non-uniform slowdown the calibration sees only on average)."""
    fixtures, i = {}, 0
    for name, base in FIXTURES.items():
        cells = {}
        for t in THREADS:
            speed = per_key[i] if per_key else runner
            real = (regress or {}).get(name, 1.0)
            rows = (base * int(t) ** 0.8 * speed * real
                    * (1.0 + wobble(i + night, noise)))
            cells[t] = {"rows_per_s": rows}
            i += 1
        fixtures[name] = {"threads": cells}
    out = {"seeml_version": "test", "schema": schema, "backend": "cpu",
           "fixtures": fixtures}
    if policy:
        out["kernel_policy"] = {"source": "table", "gemm_tile_k": policy[0],
                                "gemm_tile_n": policy[1]}
    if schema >= 4 and calibration:
        out["calibration"] = {
            "kernel": kernel,
            "calib_rows_per_s": CALIB * runner * (1.0 + wobble(night, 0.02))}
    return out


class Gate:
    """The nightly job's gate step, run night after night in one cache
    directory: bench_compare.py with --epoch-baseline, --state, --promote."""

    def __init__(self, tmp, state=True, epoch=True):
        self.dir = tmp
        self.baseline = os.path.join(tmp, "baseline", "bench.json")
        self.epoch = os.path.join(tmp, "epoch.json") if epoch else None
        self.state = os.path.join(tmp, "baseline", "gate_state.json")
        self.use_state = state
        self.log = ""

    def night(self, rep, extra=()):
        cur = os.path.join(self.dir, "current.json")
        with open(cur, "w") as f:
            json.dump(rep, f)
        argv = ["bench_compare.py", self.baseline, cur, "--promote"]
        if self.epoch:
            argv += ["--epoch-baseline", self.epoch]
        if self.use_state:
            argv += ["--state", self.state]
        argv += list(extra)
        buf, old = io.StringIO(), sys.argv
        sys.argv = argv
        try:
            with redirect_stdout(buf):
                status = bench_compare.main()
        finally:
            sys.argv = old
        self.log = buf.getvalue()
        return status

    def baseline_report(self):
        with open(self.baseline) as f:
            return json.load(f)


class GateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    # --- The incident replays (the issue's acceptance test) ----------------

    # Job wall time on identical work, #15 (the fast seed) then #20-#24.
    NIGHTLY_20_24 = [232.0, 264.0, 315.0, 290.0, 271.0, 233.0]

    def runner_speeds(self):
        return [self.NIGHTLY_20_24[0] / s for s in self.NIGHTLY_20_24]

    def test_replay_nightly_20_to_24_is_five_green_gates(self):
        gate = Gate(self.tmp)
        speeds = self.runner_speeds()
        self.assertEqual(gate.night(report(speeds[0])), 0)  # #15 seeds
        for n, speed in enumerate(speeds[1:], start=1):
            self.assertEqual(gate.night(report(speed, night=n)), 0, gate.log)
            self.assertNotIn("WARN", gate.log)
            self.assertIn("no Tier A regression", gate.log)

    def test_the_same_replay_without_calibration_is_the_incident(self):
        # Teeth: schema-3 reports (no calibration) reproduce the four red
        # nights, so the green replay above is the normalization's doing.
        gate = Gate(self.tmp, state=False)
        speeds = self.runner_speeds()
        statuses = [gate.night(report(s, schema=3, night=n))
                    for n, s in enumerate(speeds)]
        self.assertEqual(statuses, [0, 1, 1, 1, 1, 0])

    def test_replay_0912_to_0914_uniform_slowdown_is_green(self):
        # 09-13 and 09-14: every one of the 18 keys between -12.6% and -22%
        # on the commit that was green on 09-12, losses bit-identical.
        gate = Gate(self.tmp)
        self.assertEqual(gate.night(report(1.0, noise=0.0)), 0)
        for night, (lo, hi) in enumerate([(0.126, 0.22), (0.13, 0.215)], 1):
            per_key = [1.0 - (lo + (hi - lo) * (((i * 7) % 18) / 17.0))
                       for i in range(18)]
            mean = sum(per_key) / len(per_key)
            rep = report(mean, noise=0.0, night=night, per_key=per_key)
            self.assertEqual(gate.night(rep), 0, gate.log)
            self.assertNotIn("WARN", gate.log)

    # --- No missed true regression -----------------------------------------

    def test_injected_regression_warns_once_then_fails(self):
        gate = Gate(self.tmp)
        self.assertEqual(gate.night(report(1.0)), 0)
        seeded = gate.baseline_report()
        bad = {"dec_v512_d192_s32": 0.75}
        # Night 1, on a slow runner: a first red is a warning, exit 0, and
        # the warned run must NOT become the baseline.
        self.assertEqual(gate.night(report(0.8, night=1, regress=bad)), 0)
        self.assertIn("WARN dec_v512_d192_s32@1t/cpu", gate.log)
        self.assertNotIn("WARN mlp_64x512x3", gate.log)
        self.assertNotIn("promoted", gate.log)
        self.assertEqual(gate.baseline_report(), seeded)
        # Night 2, another runner, same regression: the gate fails.
        self.assertEqual(gate.night(report(0.95, night=2, regress=bad)), 1)
        self.assertIn("second consecutive red", gate.log)
        # ... and keeps failing until fixed (state and baseline unmoved).
        self.assertEqual(gate.night(report(1.1, night=3, regress=bad)), 1)
        self.assertEqual(gate.night(report(0.9, night=4)), 0)
        self.assertIn("promoted", gate.log)

    def test_uniform_real_regression_is_not_normalized_away(self):
        # Every fixture 20% slower while the calibration kernel is not:
        # exactly what a median-of-keys normalization would have hidden.
        gate = Gate(self.tmp)
        self.assertEqual(gate.night(report(1.0)), 0)
        bad = {name: 0.8 for name in FIXTURES}
        self.assertEqual(gate.night(report(1.0, night=1, regress=bad)), 0)
        self.assertEqual(gate.log.count("WARN "), 2 * 18)  # both references
        self.assertEqual(gate.night(report(0.85, night=2, regress=bad)), 1)

    def test_a_red_that_recovers_is_forgotten(self):
        gate = Gate(self.tmp)
        self.assertEqual(gate.night(report(1.0)), 0)
        bad = {"tok_v64_d64_s16": 0.7}
        for night, regress in enumerate([bad, None, bad, None], start=1):
            self.assertEqual(
                gate.night(report(1.0, night=night, regress=regress)), 0,
                gate.log)

    def test_the_epoch_is_seeded_once_and_only_by_a_green_gate(self):
        gate = Gate(self.tmp)
        self.assertEqual(gate.night(report(1.0)), 0)
        self.assertIn("seeded the epoch", gate.log)
        with open(gate.epoch) as f:
            seeded = f.read()
        self.assertEqual(gate.night(report(1.05, night=1)), 0)
        self.assertNotIn("seeded the epoch", gate.log)
        with open(gate.epoch) as f:
            self.assertEqual(f.read(), seeded)

    def test_without_state_every_red_fails_at_once(self):
        gate = Gate(self.tmp, state=False)
        self.assertEqual(gate.night(report(1.0)), 0)
        self.assertEqual(
            gate.night(report(1.0, regress={"mlp_64x512x3": 0.7})), 1)
        self.assertIn("FAIL mlp_64x512x3@1t/cpu", gate.log)

    def test_compounding_drift_trips_the_epoch(self):
        gate = Gate(self.tmp)
        self.assertEqual(gate.night(report(1.0, noise=0.0)), 0)
        drift, statuses = 1.0, []
        for night in range(1, 6):
            drift *= 0.94  # under the 10% rolling gate every night
            rep = report(1.0, noise=0.0, night=night,
                         regress={n: drift for n in FIXTURES})
            statuses.append(gate.night(rep))
        self.assertEqual(statuses, [0, 0, 0, 1, 1])  # -17% warns, -22% fails
        self.assertIn("vs epoch", gate.log)

    # --- Fail closed ---------------------------------------------------------

    def test_schema3_baseline_compares_unnormalized_with_a_note(self):
        gate = Gate(self.tmp, epoch=False)
        self.assertEqual(gate.night(report(1.0, schema=3)), 0)
        self.assertEqual(gate.night(report(1.0, night=1)), 0)
        self.assertIn("predates schema 4", gate.log)
        self.assertIn("un-normalized", gate.log)

    def test_schema4_without_calibration_is_an_error(self):
        gate = Gate(self.tmp, epoch=False)
        self.assertEqual(gate.night(report(1.0)), 0)
        self.assertEqual(gate.night(report(1.0, calibration=False)), 1)
        self.assertIn("without a usable calibration block", gate.log)
        rep = report(1.0)
        rep["calibration"]["calib_rows_per_s"] = 0.0
        self.assertEqual(gate.night(rep), 1)
        self.assertIn("non-positive", gate.log)

    def test_different_calibration_kernels_are_an_error(self):
        gate = Gate(self.tmp, epoch=False)
        self.assertEqual(gate.night(report(1.0)), 0)
        self.assertEqual(gate.night(report(1.0, kernel="sgemm_v2")), 1)
        self.assertIn("not comparable", gate.log)

    def test_a_different_host_class_is_an_error(self):
        gate = Gate(self.tmp, epoch=False)
        self.assertEqual(gate.night(report(1.0)), 0)
        self.assertEqual(gate.night(report(0.25)), 1)
        self.assertIn("not the same host class", gate.log)

    def test_zero_key_overlap_is_an_error(self):
        gate = Gate(self.tmp, epoch=False)
        self.assertEqual(gate.night(report(1.0)), 0)
        rep = report(1.0)
        rep["fixtures"] = {"renamed_" + k: v
                           for k, v in rep["fixtures"].items()}
        self.assertEqual(gate.night(rep), 1)
        self.assertIn("would be blind", gate.log)

    def test_different_kernel_policies_are_an_error(self):
        gate = Gate(self.tmp, epoch=False)
        self.assertEqual(gate.night(report(1.0)), 0)
        self.assertEqual(gate.night(report(1.0, policy=(128, 256))), 1)
        self.assertIn("compares like with like", gate.log)

    def test_an_unreadable_state_is_an_error(self):
        gate = Gate(self.tmp, epoch=False)
        self.assertEqual(gate.night(report(1.0)), 0)
        with open(gate.state, "w") as f:
            f.write("{\"schema\": 9}")
        self.assertEqual(gate.night(report(1.0)), 1)
        self.assertIn("cannot read the gate state", gate.log)

    def test_github_warning_annotation(self):
        gate = Gate(self.tmp, epoch=False)
        self.assertEqual(gate.night(report(1.0)), 0)
        os.environ["GITHUB_ACTIONS"] = "true"
        self.addCleanup(os.environ.pop, "GITHUB_ACTIONS", None)
        self.assertEqual(
            gate.night(report(1.0, regress={"mlp_64x512x3": 0.7})), 0)
        self.assertIn("::warning title=bench gate, first red::", gate.log)


if __name__ == "__main__":
    unittest.main()
