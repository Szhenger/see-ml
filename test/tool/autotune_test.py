"""Tests for tool/autotune.py, the offline autotuner.

Standard library only, like the tool. A fake seeml-bench (a Python script
the suite writes) answers every run with a report in the harness's
schema-3 shape whose rows/s depend on the arm, so the sweep, the
medians-of-medians scoring, the decision rule and the table merge are all
provable without a C++ build.

    python3 -m unittest discover -s test/tool -p '*_test.py'
"""

import io
import json
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tool"))

import autotune  # noqa: E402

HOST_KEY = "arm64;Test CPU;cores=4;l1d=65536;l2=4194304;simd=4"

# The fake bench: rows/s per fixture is a base rate times a per-arm factor
# (RATES, keyed by "KxN"; unknown arms 0.5), plus a wobble that cycles per
# arm occurrence, so medians-of-medians has something to median and the
# median over three rounds cancels it exactly.
FAKE_BENCH = r'''#!%(python)s
import json, os, sys
args = sys.argv[1:]
def take(flag, default=None):
    if flag in args:
        i = args.index(flag); v = args[i + 1]; del args[i:i + 2]; return v
    return default
out = take("--out"); tiles = take("--gemm-tiles"); fixtures = take("--fixtures", "")
for f in ("--threads", "--steps-lo", "--steps-hi", "--repeats", "--backend"):
    take(f)
if args:
    sys.stderr.write("fake-bench: unknown %%r\n" %% args); sys.exit(2)
rates = json.load(open(%(rates)r))
k, n = (64, 256) if tiles is None else tuple(int(x) for x in tiles.split(","))
arm = "%%dx%%d" %% (k, n)
if rates.get("fail") == arm:
    sys.stderr.write("fake-bench: simulated failure\n"); sys.exit(1)
count_path = %(count)r
counts = json.load(open(count_path)) if os.path.exists(count_path) else {}
seen = counts.get(arm, 0)
counts[arm] = seen + 1
json.dump(counts, open(count_path, "w"))
factor = rates["arms"].get(arm, 0.5)
wobble = [1.0, 1.10, 0.95][seen %% 3]  # per arm: medians cancel it
names = fixtures.split(",") if fixtures else ["fx_a", "fx_b"]
report = {"seeml_version": "test", "schema": 3, "host": "Test arm64",
          "backend": "cpu", "host_key": %(host)r,
          "host_arch": {"isa": "arm64", "cpu_model": "Test CPU"},
          "kernel_policy": {"source": "default" if tiles is None else "flag",
                            "gemm_tile_k": k, "gemm_tile_n": n},
          "analytic_gemm_tiles": {"k": rates["analytic"][0], "n": rates["analytic"][1]},
          "config": {}, "fixtures": {}}
for i, name in enumerate(names):
    base = 1000.0 * (i + 1)
    report["fixtures"][name] = {"threads": {
        "1": {"rows_per_s": base * factor * wobble},
        "8": {"rows_per_s": 4 * base * factor * wobble}}}
json.dump(report, open(out, "w"))
'''


class AutotuneTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="seeml_autotune_")
        self.rates_path = os.path.join(self.tmp, "rates.json")
        self.count_path = os.path.join(self.tmp, "count.txt")
        self.bench = os.path.join(self.tmp, "fake-bench")
        with open(self.bench, "w") as f:
            f.write(FAKE_BENCH % {"python": sys.executable,
                                  "rates": self.rates_path,
                                  "count": self.count_path,
                                  "host": HOST_KEY})
        os.chmod(self.bench, os.stat(self.bench).st_mode | stat.S_IXUSR)
        self.table = os.path.join(self.tmp, "kernel_policy.json")

    def set_rates(self, arms, analytic=(96, 16), fail=None):
        with open(self.rates_path, "w") as f:
            json.dump({"arms": arms, "analytic": list(analytic),
                       "fail": fail}, f)

    def runs(self):
        if not os.path.exists(self.count_path):
            return 0
        with open(self.count_path) as f:
            return sum(json.load(f).values())

    def tune(self, *extra, expect=0):
        err, out = io.StringIO(), io.StringIO()
        with redirect_stderr(err), redirect_stdout(out):
            rc = autotune.main(["tune", "--bench", self.bench, "--out",
                                self.table, "--rounds", "3", *extra])
        self.assertEqual(rc, expect, err.getvalue())
        return err.getvalue()

    def load(self):
        with open(self.table) as f:
            return json.load(f)

    # --- The sweep ------------------------------------------------------------

    def test_sweeps_every_arm_every_round_and_picks_the_winner(self):
        self.set_rates({"64x256": 1.0, "96x16": 0.7, "128x512": 1.25,
                        "32x64": 0.9})
        err = self.tune("--arms", "128x512,32x64")
        # default + analytic + two requested arms, three rounds each.
        self.assertEqual(self.runs(), 4 * 3)
        table = self.load()
        self.assertEqual(table["schema"], 1)
        entry = table["hosts"][HOST_KEY]
        self.assertEqual(entry["cpu"], {"gemm_tile_k": 128, "gemm_tile_n": 512})
        tuned = entry["tuned"]
        self.assertTrue(tuned["decision"].startswith("tuned:"))
        self.assertEqual(tuned["default_arm"], "64x256")
        self.assertEqual(tuned["analytic_arm"], "96x16")
        self.assertEqual(tuned["rounds"], 3)
        labels = {f"{a['gemm_tile_k']}x{a['gemm_tile_n']}": a["label"]
                  for a in tuned["arms"]}
        self.assertEqual(labels, {"64x256": "default", "96x16": "analytic",
                                  "128x512": "grid", "32x64": "grid"})
        # Scores are medians of medians relative to the default arm: the
        # wobble cancels because every arm sees the same three wobbles.
        by_arm = {f"{a['gemm_tile_k']}x{a['gemm_tile_n']}": a
                  for a in tuned["arms"]}
        self.assertAlmostEqual(by_arm["64x256"]["score"], 1.0, places=6)
        self.assertAlmostEqual(by_arm["128x512"]["score"], 1.25, places=6)
        self.assertAlmostEqual(by_arm["96x16"]["score"], 0.7, places=6)
        self.assertEqual(sorted(tuned["fixtures"]), ["fx_a", "fx_b"])
        self.assertIn("fx_a@8", by_arm["128x512"]["rows_per_s"])
        self.assertIn("128x512", err)

    def test_default_is_recorded_when_no_arm_beats_the_margin(self):
        self.set_rates({"64x256": 1.0, "96x16": 0.9, "128x512": 1.02})
        self.tune("--arms", "128x512")
        entry = self.load()["hosts"][HOST_KEY]
        self.assertEqual(entry["cpu"], {"gemm_tile_k": 64, "gemm_tile_n": 256})
        self.assertTrue(
            entry["tuned"]["decision"].startswith("default-within-margin"))
        # Lower the margin and the same arm wins.
        self.set_rates({"64x256": 1.0, "96x16": 0.9, "128x512": 1.02})
        self.tune("--arms", "128x512", "--min-gain", "0.01")
        self.assertEqual(self.load()["hosts"][HOST_KEY]["cpu"],
                         {"gemm_tile_k": 128, "gemm_tile_n": 512})

    def test_default_is_best_when_nothing_beats_it(self):
        self.set_rates({"64x256": 1.0, "96x16": 0.8, "16x16": 0.6})
        self.tune("--arms", "16x16")
        entry = self.load()["hosts"][HOST_KEY]
        self.assertEqual(entry["cpu"], {"gemm_tile_k": 64, "gemm_tile_n": 256})
        self.assertEqual(entry["tuned"]["decision"], "default-is-best")

    def test_default_grid_when_no_arms_are_named(self):
        self.set_rates({"64x256": 1.0})
        self.tune("--rounds", "1")
        grid = len(autotune.default_grid())
        # The grid contains 64x256 already; the analytic arm 96x16 is extra.
        self.assertEqual(self.runs(), grid + 1)

    def test_fixtures_flag_reaches_the_bench(self):
        self.set_rates({"64x256": 1.0, "8x8": 1.5})
        self.tune("--arms", "8x8", "--fixtures", "only_this", "--rounds", "1")
        entry = self.load()["hosts"][HOST_KEY]
        self.assertEqual(entry["tuned"]["fixtures"], ["only_this"])
        self.assertEqual(entry["cpu"], {"gemm_tile_k": 8, "gemm_tile_n": 8})

    # --- The table --------------------------------------------------------------

    def test_merges_into_an_existing_table_and_keeps_other_hosts(self):
        with open(self.table, "w") as f:
            json.dump({"schema": 1, "hosts": {
                "other-host": {"cpu": {"gemm_tile_k": 16, "gemm_tile_n": 32}},
                HOST_KEY: {"cpu": {"gemm_tile_k": 4, "gemm_tile_n": 4},
                           "stale": True}}}, f)
        self.set_rates({"64x256": 1.0, "128x512": 1.3})
        self.tune("--arms", "128x512", "--rounds", "1")
        table = self.load()
        self.assertEqual(table["hosts"]["other-host"]["cpu"],
                         {"gemm_tile_k": 16, "gemm_tile_n": 32})
        self.assertEqual(table["hosts"][HOST_KEY]["cpu"],
                         {"gemm_tile_k": 128, "gemm_tile_n": 512})
        self.assertNotIn("stale", table["hosts"][HOST_KEY])

    def test_refuses_to_overwrite_a_file_that_is_not_a_table(self):
        with open(self.table, "w") as f:
            f.write('{"something": "else"}')
        self.set_rates({"64x256": 1.0})
        err = self.tune("--arms", "64x256", "--rounds", "1", expect=1)
        self.assertIn("not a schema-1", err)
        self.assertEqual(open(self.table).read(), '{"something": "else"}')

    def test_bench_failure_is_exit_1_and_leaves_the_table_alone(self):
        self.set_rates({"64x256": 1.0, "8x8": 1.5}, fail="8x8")
        err = self.tune("--arms", "8x8", "--rounds", "1", expect=1)
        self.assertIn("bench exited 1", err)
        self.assertFalse(os.path.exists(self.table))

    def test_show_prints_the_entries(self):
        self.set_rates({"64x256": 1.0, "128x512": 1.3})
        self.tune("--arms", "128x512", "--rounds", "1")
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(autotune.main(["show", self.table]), 0)
        text = out.getvalue()
        self.assertIn(HOST_KEY, text)
        self.assertIn("K 128 N 512", text)
        self.assertIn("128x512", text)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            self.assertEqual(autotune.main(["show", self.table, "--host",
                                            "nobody"]), 1)

    # --- Argument discipline -------------------------------------------------------

    def test_arm_syntax_is_strict(self):
        for bad in ("64", "6x16", "0x16", "64x0", "64x256x4", "axb", "64,256"):
            with self.assertRaises(SystemExit) as cm, \
                    redirect_stderr(io.StringIO()):
                autotune.build_parser().parse_args(
                    ["tune", "--bench", "b", "--out", "t", "--arms", bad])
            self.assertEqual(cm.exception.code, 2)
        self.assertEqual(autotune.parse_arms("64x256,8x8"), [(64, 256), (8, 8)])

    def test_unknown_flag_is_exit_2(self):
        with self.assertRaises(SystemExit) as cm, \
                redirect_stderr(io.StringIO()):
            autotune.main(["tune", "--bench", "b", "--out", "t", "--bogus"])
        self.assertEqual(cm.exception.code, 2)

    def test_scoring_is_a_geometric_mean_over_keys(self):
        runs = {(64, 256): [{"a": 100.0, "b": 10.0}],
                (8, 8): [{"a": 200.0, "b": 5.0}]}  # 2x and 0.5x -> gmean 1.0
        scored = autotune.score_arms(runs, (64, 256))
        self.assertAlmostEqual(scored[(8, 8)]["score"], 1.0, places=9)
        chosen, decision = autotune.choose(scored, (64, 256), 0.03)
        self.assertEqual(chosen, (64, 256))
        self.assertEqual(decision, "default-is-best")


if __name__ == "__main__":
    unittest.main()
