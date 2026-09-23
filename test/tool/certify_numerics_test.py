"""Tests for tool/certify_numerics.py, the relaxed-reduction certifier
(P3, #77).

  stdlib   what a certificate binds to and how verification fails closed;
  numpy    the relaxed model itself: the lane reduction is the schedule it
           claims to be, and the any-order bound really bounds it — on
           benign operands and on catastrophic cancellation alike;
  C++      certificates over compiled plans: granted, refused, voided by a
           recompile, enforced by tool/pack_update.py, and checked against
           a measured run (a frontier_exec.py diff report).

    python3 -m unittest discover -s test/tool -p '*_test.py'
"""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tool"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import certify_numerics as cn  # noqa: E402
import frontier_exec as fx  # noqa: E402
import frontier_exec_test as fixtures  # noqa: E402
import pack_update  # noqa: E402

np = fixtures.np
N = fx.NULL_REF


def run(module, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        status = module.main(argv)
    return status, buf.getvalue()


def granted(plan_path, **over):
    cert = {"schema": 1, "kind": cn.CERT_KIND, "seeml_tool": "test",
            "plan": {"file": "p", "sha256": cn.sha256_file(plan_path),
                     "plan_hash": "0", "version": 11},
            "corpus": {"file": "c", "sha256": "0" * 64},
            "contract": {"family": cn.FAMILY, "model_lanes": 8,
                         "unit_roundoff": cn.UNIT_ROUNDOFF,
                         "sites": ["attn.fwd"]},
            "evidence": {}, "tolerances": {"site_rel": 1e-5,
                                           "loss_rel": 1e-4},
            "verdict": "granted", "reasons": []}
    cert.update(over)
    cert["digest"] = cn.self_digest(cert)
    return cert


class CertificateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.plan = os.path.join(self._tmp.name, "update_plan.seeu")
        with open(self.plan, "wb") as f:
            f.write(fixtures.assemble(64))

    def test_a_granted_certificate_verifies(self):
        self.assertEqual(cn.check_certificate(granted(self.plan), self.plan),
                         [])

    def test_verification_fails_closed(self):
        cert = granted(self.plan)
        edited = dict(cert, tolerances={"site_rel": 1.0, "loss_rel": 1.0})
        self.assertIn("edited", cn.check_certificate(edited, self.plan)[0])
        refusal = granted(self.plan, verdict="refused", reasons=["too loose"])
        self.assertIn("refusal: too loose",
                      cn.check_certificate(refusal, self.plan)[0])
        with open(self.plan, "ab") as f:
            f.write(b"\0")  # "recompiled"
        self.assertIn("different plan",
                      cn.check_certificate(cert, self.plan)[0])
        for junk in ({}, [], {"kind": cn.CERT_KIND, "schema": 99}):
            self.assertIn("not a schema-1",
                          cn.check_certificate(junk, self.plan)[0])
        hollow = {"schema": 1, "kind": cn.CERT_KIND}
        hollow["digest"] = cn.self_digest(hollow)
        self.assertIn("malformed", cn.check_certificate(hollow, self.plan)[0])

    def test_observed_runs_are_held_to_the_tolerances(self):
        cert = granted(self.plan)
        ok = {"rtol": 1e-5, "opcodes": {
            "attn.fwd": {"worst_ratio": 0.4}, "gemm.nn": {"worst_ratio": 0.2}}}
        self.assertEqual(cn.check_observed(cert, ok), [])
        loose = {"rtol": 1e-5, "opcodes": {"attn.fwd": {"worst_ratio": 3.0}}}
        self.assertIn("attn.fwd: measured", cn.check_observed(cert, loose)[0])
        stray = {"rtol": 1e-5, "opcodes": {"gemm.nn": {"worst_ratio": 1.5}}}
        self.assertIn("not a certified site",
                      cn.check_observed(cert, stray)[0])

    def test_gamma(self):
        self.assertAlmostEqual(cn.gamma(64), 64 * 2.0 ** -24, delta=1e-10)
        self.assertEqual(cn.gamma(2 ** 24), float("inf"))

    def test_pack_update_enforces_a_certificate_that_is_present(self):
        pkg = self._tmp.name
        self.assertIsNone(pack_update.check_certificate(pkg, self.plan))
        path = os.path.join(pkg, pack_update.CERTIFICATE_FILE)
        with open(path, "w") as f:
            json.dump(granted(self.plan), f)
        summary = pack_update.check_certificate(pkg, self.plan)
        self.assertEqual(summary["family"], cn.FAMILY)
        with open(self.plan, "ab") as f:
            f.write(b"\0")
        with self.assertRaisesRegex(pack_update.PackError,
                                    "does not vouch for this plan"):
            pack_update.check_certificate(pkg, self.plan)


@unittest.skipIf(np is None, "NumPy not installed")
class RelaxedModelTest(unittest.TestCase):
    def test_the_lane_sum_is_the_schedule_it_claims(self):
        x = cn.LaneBackend(4)
        v = np.random.default_rng(1).standard_normal((3, 11)).astype(
            np.float32)
        want = []
        for row in v:
            lanes = [np.float32(0)] * 4
            for i, value in enumerate(row):  # strided sequential f32 adds
                lanes[i % 4] = np.float32(lanes[i % 4] + value)
            want.append(np.float32(np.float32(lanes[0] + lanes[1]) +
                                   np.float32(lanes[2] + lanes[3])))
        got = x._lane_sum(v, -1)
        self.assertEqual(got.dtype, np.float32)
        np.testing.assert_array_equal(got, np.array(want, np.float32))

    def test_the_any_order_bound_bounds_every_schedule(self):
        rng = np.random.default_rng(2)
        benign = rng.standard_normal((64, 777))
        hostile = np.concatenate([benign * 1e6, -benign * 1e6 + 1e-3], -1)
        for data in (benign, hostile):
            exact = data.astype(np.float32).astype(np.float64).sum(-1)
            for lanes in (1, 2, 8, 32):
                x = cn.LaneBackend(lanes)
                got = x.sum(data, -1)
                err = np.abs(got - exact).max() / np.abs(got).max()
                self.assertLessEqual(err, x.bound_rel * (1 + 1e-12))
        self.assertGreater(x.bound_rel, 1e-2)  # cancellation is visible
        a, b = rng.standard_normal((5, 9, 64)), rng.standard_normal((5, 64, 7))
        x = cn.LaneBackend(8)
        got = x.matmul(a, b)
        exact = (a.astype(np.float32).astype(np.float64) @
                 b.astype(np.float32).astype(np.float64))
        self.assertLessEqual(np.abs(got - exact).max() / np.abs(got).max(),
                             x.bound_rel)
        self.assertGreater(np.abs(got - exact).max(), 0.0)  # it IS relaxed

    def test_lockstep_sees_cancellation_at_the_site_that_owns_it(self):
        rows, cols = 4096, 2
        big = np.tile([[3e7, 1.0], [-3e7, 1.0]], (rows // 2, 1))
        plan = fx.Plan(fixtures.assemble(
            4 * (rows * cols + 8),
            train=[(10, 0, (0, 4 * rows * cols, N, N), (rows, cols, 0)),
                   (19, 0, (4 * rows * cols + 16, fx.f32_bits(1.0), N, N),
                    (2, 0, 0))]))
        ex = cn.LockstepExecutor(plan, 8)
        ex.mem.stage(0, big.astype("<f4").tobytes())
        ex.execute("train")
        self.assertEqual(sorted(ex.sites), ["reduce.rows"])  # fill: no site
        stat = ex.sites["reduce.rows"]
        self.assertLessEqual(stat["observed_rel"], stat["bound_rel"])
        self.assertGreater(stat["bound_rel"], 1.0)  # vacuous: refuse-worthy


@unittest.skipIf(np is None, "NumPy not installed")
@unittest.skipUnless(fixtures.HAVE_CPP,
                     f"no built seeml tools under {fixtures.BUILD}")
class CertifyPlansTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="seeml-certify-test-")
        cls.dir = cls._tmp.name
        fixtures._write_models(cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def compile(self, name, model, batch, *flags):
        out = os.path.join(self.dir, name)
        shutil.rmtree(out, ignore_errors=True)
        done = subprocess.run(
            [fixtures.COMPILER, "--source", os.path.join(self.dir, model),
             "--out", out, "--data-batch", str(batch), "--no-embed", *flags],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(done.returncode, 0, done.stdout)
        return out

    def test_certify_verify_pack_and_void(self):
        corpus = os.path.join(self.dir, "decoder_corpus.sds")
        pkg = self.compile("dec", "decoder.smf", 12, "--lr", "2e-3")
        status, log = run(cn, ["certify", pkg, "--corpus", corpus,
                               "--steps", "6", "--val-fraction", "0.25"])
        self.assertEqual(status, 0, log)
        cert_path = os.path.join(pkg, cn.CERT_FILE)
        with open(cert_path) as f:
            cert = json.load(f)
        self.assertEqual(cert["verdict"], "granted")
        self.assertEqual(cert["contract"]["sites"], [
            "attn.dp", "attn.fwd", "rms_norm.bwd", "rms_norm.fwd",
            "softmax_rows.bwd", "softmax_xent.fwd"])
        for stat in cert["evidence"]["sites"].values():
            self.assertGreater(stat["instances"], 0)
            self.assertLessEqual(stat["observed_rel"], stat["bound_rel"])
            self.assertLess(stat["observed_rel"], 1e-5)
        free = cert["evidence"]["free_run"]
        self.assertLess(free["loss_max_rel_dev"], 1e-4)
        self.assertTrue(free["gate_improved_reference"])
        self.assertTrue(free["gate_improved_relaxed"])

        self.assertEqual(run(cn, ["verify", cert_path, "--plan", pkg,
                                  "--corpus", corpus])[0], 0)
        # A measured run (here: the exact C++ runtime itself) is inside the
        # certified tolerances.
        diff = os.path.join(self.dir, "diff.json")
        self.assertEqual(run(fx, ["diff", os.path.join(pkg, cn.PLAN_FILE),
                                  "--corpus", corpus, "--probe",
                                  fixtures.PROBE, "--steps", "2", "--report",
                                  diff])[0], 0)
        self.assertEqual(run(cn, ["verify", cert_path, "--plan", pkg,
                                  "--observed", diff])[0], 0)
        wrong = os.path.join(self.dir, "class.sds")
        status, log = run(cn, ["verify", cert_path, "--plan", pkg,
                               "--corpus", wrong])
        self.assertEqual(status, 3)
        self.assertIn("different corpus", log)

        # Packaging beside a valid certificate records it ...
        report = os.path.join(self.dir, "pack.json")
        self.assertEqual(run(pack_update, [pkg, "--report", report])[0], 0)
        with open(report) as f:
            self.assertEqual(json.load(f)["numerics_certificate"]["family"],
                             cn.FAMILY)
        # ... and a recompile voids it, loudly, at verify and at pack time.
        kept = os.path.join(self.dir, "kept.json")
        shutil.copy(cert_path, kept)
        pkg = self.compile("dec", "decoder.smf", 12, "--lr", "3e-3")
        shutil.copy(kept, os.path.join(pkg, cn.CERT_FILE))
        status, log = run(cn, ["verify", kept, "--plan", pkg])
        self.assertEqual(status, 3)
        self.assertIn("different plan", log)
        status, log = run(pack_update, [pkg])
        self.assertEqual(status, 1)
        self.assertIn("does not vouch for this plan", log)

    def test_a_relaxed_plan_is_packaged_only_beside_its_certificate(self):
        # --precision certified-bf16 (v18): the packer refuses the plan bare,
        # the certifier prices the gemm.relaxed site (bf16 input rounding,
        # observed inside 2^-7 and its own bound), and the packer then
        # accepts it — but not an older certificate that never saw the site.
        corpus = os.path.join(self.dir, "class.sds")
        pkg = self.compile("mlp_relaxed", "mlp.smf", 8,
                           "--precision", "certified-bf16")
        plan = fx.Plan.load(os.path.join(pkg, cn.PLAN_FILE))
        self.assertGreater(plan.relaxed_gemms, 0)
        status, log = run(pack_update, [pkg])
        self.assertEqual(status, 1)
        self.assertIn("relaxed GEMM", log)
        self.assertIn("no numerics certificate", log)
        # bf16 input rounding moves a short run's loss by ~1e-3 relative;
        # the 1e-4 default is the reduction family's, so a relaxed plan
        # states its loss tolerance explicitly (F2's acceptance is 1e-3 on
        # the 300-step validation loss, measured by the frontier harness).
        status, log = run(cn, ["certify", pkg, "--corpus", corpus,
                               "--steps", "4"])
        self.assertEqual(status, 3)
        self.assertIn("loss_max_rel_dev", log)
        status, log = run(cn, ["certify", pkg, "--corpus", corpus,
                               "--steps", "4", "--max-loss-deviation", "1e-2"])
        self.assertEqual(status, 0, log)
        with open(os.path.join(pkg, cn.CERT_FILE)) as f:
            cert = json.load(f)
        self.assertEqual(cert["verdict"], "granted")
        self.assertIn(cn.RELAXED_GEMM_SITE, cert["contract"]["sites"])
        self.assertEqual(cert["plan"]["relaxed_gemms"], plan.relaxed_gemms)
        site = cert["evidence"]["sites"][cn.RELAXED_GEMM_SITE]
        self.assertEqual(site["instances"], plan.relaxed_gemms * 4)
        self.assertGreater(site["observed_rel"], 1e-5)   # a real rounding
        self.assertLessEqual(site["observed_rel"], site["bound_rel"])
        self.assertLess(site["observed_rel"], 2.0 ** -7)
        self.assertEqual(cert["tolerances"]["relaxed_gemm_rel"], 2.0 ** -7)
        report = os.path.join(self.dir, "pack_relaxed.json")
        self.assertEqual(run(pack_update, [pkg, "--report", report])[0], 0)
        with open(report) as f:
            self.assertEqual(json.load(f)["relaxed_gemms"],
                             plan.relaxed_gemms)
        # A certificate over the exact plan does not cover the relaxed one.
        exact = self.compile("mlp_exact", "mlp.smf", 8)
        self.assertEqual(run(cn, ["certify", exact, "--corpus", corpus,
                                  "--steps", "4"])[0], 0)
        with open(os.path.join(exact, cn.CERT_FILE)) as f:
            other = json.load(f)
        other["plan"]["sha256"] = cert["plan"]["sha256"]
        other["digest"] = cn.self_digest(other)
        problems = cn.check_certificate(other, os.path.join(pkg, cn.PLAN_FILE))
        self.assertTrue(any("does not cover" in p for p in problems), problems)
        # And the exact plan's train program is the relaxed one's, minus
        # the bit: the format change is inert on the reference backend.
        a = fx.Plan.load(os.path.join(exact, cn.PLAN_FILE))
        for x, y in zip(a.sections["train"], plan.sections["train"]):
            self.assertEqual(x.opcode, y.opcode)
            self.assertEqual(x.flags, y.flags & ~fx.FLAG_RELAXED)
            self.assertEqual(x.src, y.src)

    def test_a_tolerance_the_plan_cannot_meet_is_a_refusal(self):
        pkg = self.compile("mlp", "mlp.smf", 8, "--clip-norm", "0.5")
        status, log = run(cn, ["certify", pkg, "--corpus",
                               os.path.join(self.dir, "class.sds"),
                               "--steps", "4", "--max-site-error", "1e-12"])
        self.assertEqual(status, 3, log)
        with open(os.path.join(pkg, cn.CERT_FILE)) as f:
            cert = json.load(f)
        self.assertEqual(cert["verdict"], "refused")
        self.assertTrue(any("exceeds 1.0e-12" in r for r in cert["reasons"]))
        self.assertEqual(run(cn, ["verify", os.path.join(pkg, cn.CERT_FILE),
                                  "--plan", pkg])[0], 3)
        status, log = run(pack_update, [pkg])  # a refusal packs nothing
        self.assertEqual(status, 1)
        self.assertIn("refusal", log)

    def test_every_loss_and_norm_family_certifies(self):
        for name, model, corpus, batch, flags in fixtures.MATRIX:
            if name not in ("mlp_mse_sgd", "mlp_xent_kl", "mlp_accum"):
                continue
            flags = [os.path.join(self.dir, f) if f.endswith(".smf") else f
                     for f in flags]
            pkg = self.compile(name, model, batch, *flags)
            with self.subTest(plan=name):
                status, log = run(cn, ["certify", pkg, "--corpus",
                                       os.path.join(self.dir, corpus),
                                       "--steps", "5"])
                self.assertEqual(status, 0, log)


if __name__ == "__main__":
    unittest.main()
