"""Tests for tool/frontier_exec.py, the frontier reference executor (P4, #78).

Three tiers, each skipping cleanly where its dependency is absent:

  stdlib   the .seeu reader, the f32 LR schedule, the feeder's RNG;
  numpy    the interpreter against ITSELF: every backward opcode against a
           central finite difference of its forward opcode, run through a
           hand-assembled plan — the oracle must not be trusted because it
           agrees with the C++ runtime, only if it is right on its own;
  C++      the differential suite: a fixture matrix compiled by
           seeml-update-compile (every loss, optimizer, storage precision,
           gradient accumulation, fused and unfused epilogues, both model
           families) and compared instruction by instruction against
           seeml-plan-probe. Needs the built tools: SEEML_BUILD_DIR, else
           ./build.

    python3 -m unittest discover -s test/tool -p '*_test.py'
"""

import contextlib
import io
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tool"))

import frontier_exec as fx  # noqa: E402

try:
    import numpy as np
except ImportError:  # the torch-free CI tier
    np = None

BUILD = os.environ.get("SEEML_BUILD_DIR", os.path.join(REPO, "build"))
COMPILER = os.path.join(BUILD, "seeml-update-compile")
PROBE = os.path.join(BUILD, "seeml-plan-probe")
HAVE_CPP = all(os.path.isfile(p) and os.access(p, os.X_OK)
               for p in (COMPILER, PROBE))


def assemble(arena_size, train=(), persistent=b"", rodata=b"", **header):
    """A minimal .seeu blob around hand-written instructions. The plan hash
    is left zero: the Python reader does not seal, only the C++ probe does."""
    fields = dict.fromkeys(fx._HEADER_FIELDS, 0)
    fields.update(magic=fx.SEEU_MAGIC, version=fx.SEEU_NEWEST,
                  arena_size=arena_size, persistent_size=len(persistent),
                  persist_init_size=len(persistent), input_ref=fx.NULL_REF,
                  label_ref=fx.NULL_REF, loss_ref=fx.NULL_REF, lr=1e-3,
                  beta1=0.9, beta2=0.999, eps=1e-8, batch=1)
    fields.update(header)
    body = b"".join(fx._INSTRUCTION.pack(op, flags, 0, *src, *out)
                    for op, flags, src, out in train)
    fields["train_instr_offset"] = fx._HEADER.size
    fields["train_instr_count"] = len(train)
    fields["persist_init_offset"] = fx._HEADER.size + len(body)
    fields["rodata_offset"] = fields["persist_init_offset"] + len(persistent)
    fields["rodata_size"] = len(rodata)
    head = fx._HEADER.pack(*(fields[name] for name in fx._HEADER_FIELDS))
    return bytearray(head + body + persistent + rodata)


N = fx.NULL_REF


class ReaderTest(unittest.TestCase):
    def test_header_round_trip_and_sections(self):
        blob = assemble(64, train=[(19, 0, (0, fx.f32_bits(2.5), N, N),
                                    (4, 0, 0))], persistent=b"\1" * 8)
        plan = fx.Plan(blob)
        self.assertEqual((plan.version, plan.arena_size), (13, 64))
        self.assertEqual(len(plan.sections["train"]), 1)
        self.assertEqual(plan.sections["train"][0].opcode, 19)
        self.assertEqual(plan.initial_arena()[:9], b"\1" * 8 + b"\0")
        self.assertEqual(plan.histogram(), {"fill": {"train": 1}})

    def test_refuses_what_it_cannot_prove(self):
        blob = assemble(64)
        for mutate, message in (
                (lambda b: b.__setitem__(slice(0, 4), b"NOPE"), "not a .seeu"),
                (lambda b: b.__setitem__(slice(4, 8), struct.pack("<I", 99)),
                 "version 99"),
                (lambda b: b.__delitem__(slice(100, None)), "truncated")):
            bad = bytearray(blob)
            mutate(bad)
            with self.assertRaisesRegex(fx.PlanError, message):
                fx.Plan(bad)
        with self.assertRaisesRegex(fx.PlanError, "outside the file"):
            fx.Plan(assemble(64, eval_instr_offset=10 ** 9,
                             eval_instr_count=1))

    def test_every_opcode_of_the_abi_is_interpreted(self):
        # source/plan/instruction.h is the ABI; a new opcode there must fail
        # here until the interpreter learns it.
        with open(os.path.join(REPO, "source/plan/instruction.h")) as f:
            text = f.read()
        body = text[text.index("enum class OpCode"):text.index("};")]
        declared = sorted(int(line.split("=")[1].split(",")[0])
                          for line in body.splitlines()
                          if line.strip().startswith("k") and "=" in line)
        self.assertEqual(declared, sorted(fx.INTERPRETER))

    def test_cosine_schedule_matches_the_engine(self):
        plan = fx.Plan(assemble(64, lr_schedule=1, warmup_steps=4,
                                default_steps=24, min_lr_factor=0.1, lr=2e-3))
        base = fx.f32(2e-3)
        self.assertAlmostEqual(plan.effective_lr(2), base / 2, places=9)
        self.assertEqual(plan.effective_lr(4), base)       # end of warmup
        self.assertAlmostEqual(plan.effective_lr(14),       # cosine midpoint
                               base * (0.1 + 0.9 * 0.5), places=9)
        self.assertEqual(plan.effective_lr(24), fx.f32(base * fx.f32(0.1)))
        self.assertEqual(plan.effective_lr(500), plan.effective_lr(24))
        self.assertEqual(fx.Plan(assemble(64)).effective_lr(7), fx.f32(1e-3))

    def test_splitmix64_reference_vector(self):
        # The published splitmix64 stream for seed 1234567.
        state, out = 1234567, []
        for _ in range(3):
            state, z = fx._splitmix64(state)
            out.append(z)
        self.assertEqual(out, [6457827717110365317, 3203168211198807973,
                               9817491932198370423])


@unittest.skipIf(np is None, "NumPy not installed")
class InterpreterSelfCheck(unittest.TestCase):
    """Backward opcodes against finite differences of forward opcodes."""

    def run_ops(self, arena_floats, instructions, values):
        plan = fx.Plan(assemble(4 * arena_floats, train=instructions))
        ex = fx.Executor(plan, fx.NumpyBackend("float64"))
        for off, array in values.items():
            ex.mem.stage(4 * off, np.asarray(array, "<f4").tobytes())
        ex.execute("train")
        return np.frombuffer(ex.mem.export(), "<f4").astype(np.float64)

    def check_vjp(self, n_in, n_out, fwd, bwd, x0, extra=None, eps=1e-2,
                  tol=2e-3):
        """fwd(x_off, y_off) / bwd(dy_off, x_off, dx_off) build instructions;
        asserts dx == J^T dy with J from central differences in f32-safe
        steps. Layout: x | y | dy | dx | scratch."""
        rng = np.random.default_rng(7)
        dy = rng.standard_normal(n_out)
        x_off, y_off, dy_off, dx_off = 0, n_in, n_in + n_out, n_in + 2 * n_out
        size = 2 * n_in + 2 * n_out + 4096
        base = {x_off: x0, dy_off: dy}
        base.update(extra or {})

        def forward(x):
            vals = dict(base)
            vals[x_off] = x
            return self.run_ops(size, fwd(x_off, y_off), vals)[
                y_off:y_off + n_out]
        out = self.run_ops(size, fwd(x_off, y_off) + bwd(dy_off, x_off,
                                                         dx_off, y_off), base)
        dx = out[dx_off:dx_off + n_in]
        numeric = np.zeros(n_in)
        for i in range(n_in):
            step = np.zeros(n_in)
            step[i] = eps
            numeric[i] = np.dot(dy, forward(x0 + step) -
                                forward(x0 - step)) / (2 * eps)
        scale = max(np.abs(numeric).max(), 1e-6)
        self.assertLess(np.abs(dx - numeric).max() / scale, tol)

    SCRATCH = 3000  # float offset of side outputs (cached statistics, P)

    def test_activations(self):
        x0 = np.linspace(-2.5, 2.5, 12)
        for f, b in ((7, 8), (22, 23), (24, 25)):
            self.check_vjp(
                12, 12,
                lambda x, y, f=f: [(f, 0, (4 * x, 4 * y, N, N), (12, 0, 0))],
                lambda dy, x, dx, y, b=b: [(b, 0, (4 * dy, 4 * x, 4 * dx, N),
                                            (12, 0, 0))],
                x0 + 0.013)  # keep ReLU's kink off the difference stencil

    def test_gelu_backward_survives_saturation(self):
        out = self.run_ops(16, [(23, 0, (0, 8, 16, N), (2, 0, 0))],
                           {0: [1.0, 1.0], 2: [3e12, -3e12]})
        self.assertEqual(list(out[4:6]), [1.0, 0.0])

    def test_norms(self):
        rows, dim, s = 3, 8, self.SCRATCH
        rng = np.random.default_rng(3)
        x0 = rng.standard_normal(rows * dim)
        gamma = {s: rng.standard_normal(dim), s + dim: rng.standard_normal(dim)}
        nd = rows << 32 | dim
        self.check_vjp(
            rows * dim, rows * dim,
            lambda x, y: [(26, 0, (4 * x, 4 * s, 4 * (s + dim), 4 * y),
                           (nd, 4 * (s + 64), 4 * (s + 96)))],
            lambda dy, x, dx, y: [(27, 0, (4 * dy, 4 * x, 4 * s, 4 * dx),
                                   (4 * (s + 64), 4 * (s + 96), nd))],
            x0, gamma)
        self.check_vjp(
            rows * dim, rows * dim,
            lambda x, y: [(31, 0, (4 * x, 4 * s, 4 * y, 4 * (s + 64)),
                           (nd, 0, 0))],
            lambda dy, x, dx, y: [(32, 0, (4 * dy, 4 * x, 4 * s, 4 * dx),
                                   (4 * (s + 64), nd, 0))],
            x0, gamma)

    def test_rope_is_a_rotation_and_its_backward_the_transpose(self):
        b, s, h, d = 2, 5, 2, 6
        n = b * s * h * d
        dims = (b << 32 | s, h << 32 | d, fx.f32_bits(10000.0))
        x0 = np.random.default_rng(5).standard_normal(n)
        out = self.run_ops(3 * n, [(33, 0, (0, 4 * n, N, N), dims),
                                   (34, 0, (4 * n, 8 * n, N, N), dims)],
                           {0: x0})
        np.testing.assert_allclose(out[2 * n:3 * n], x0, atol=2e-6)
        y = out[n:2 * n].reshape(b, s, h, d)
        np.testing.assert_allclose(y[:, 0], x0.reshape(b, s, h, d)[:, 0],
                                   atol=1e-7)  # position 0: identity
        # pair (0, 1) of position 1 turns by exactly one radian
        a, c = x0.reshape(b, s, h, d)[0, 1, 0, :2]
        np.testing.assert_allclose(
            y[0, 1, 0, :2], [a * math.cos(1) - c * math.sin(1),
                             a * math.sin(1) + c * math.cos(1)], atol=1e-6)

    def test_attention_chain(self):
        b, s, h, d = 1, 4, 2, 4
        n, pn, sc = b * s * h * d, b * h * s * s, self.SCRATCH
        bs, hd = b << 32 | s, h << 32 | d
        rng = np.random.default_rng(11)
        k0, v0 = rng.standard_normal(n), rng.standard_normal(n)
        p, dp, ds = sc + 200, sc + 300, sc + 400

        def fwd_q(x, y):   # y = attention(q = x, k, v)
            return [(35, 0, (4 * x, 4 * sc, 4 * (sc + 64), 4 * y),
                     (4 * p, bs, hd))]

        def chain(dy):
            return [(36, 0, (4 * dy, 4 * (sc + 64), 4 * dp, N), (bs, hd, 0)),
                    (38, 0, (4 * p, 4 * dp, 4 * ds, N),
                     ((b * h * s) << 32 | s, 0, 0))]
        extra = {sc: k0, sc + 64: v0}
        self.check_vjp(n, n, fwd_q,
                       lambda dy, x, dx, y: chain(dy) + [
                           (39, 0, (4 * ds, 4 * sc, 4 * dx, N), (bs, hd, 0))],
                       rng.standard_normal(n), extra)

        def fwd_k(x, y):
            return [(35, 0, (4 * sc, 4 * x, 4 * (sc + 64), 4 * y),
                     (4 * p, bs, hd))]
        self.check_vjp(n, n, fwd_k,
                       lambda dy, x, dx, y: chain(dy) + [
                           (40, 0, (4 * ds, 4 * sc, 4 * dx, N), (bs, hd, 0))],
                       rng.standard_normal(n), extra)

        def fwd_v(x, y):
            return [(35, 0, (4 * sc, 4 * (sc + 64), 4 * x, 4 * y),
                     (4 * p, bs, hd))]
        self.check_vjp(n, n, fwd_v,
                       lambda dy, x, dx, y: [
                           (37, 0, (4 * p, 4 * dy, 4 * dx, N), (bs, hd, 0))],
                       rng.standard_normal(n), extra)
        out = self.run_ops(sc + 600, fwd_q(0, n), {0: k0, sc: k0, sc + 64: v0})
        probs = out[p:p + pn].reshape(b * h, s, s)
        np.testing.assert_allclose(probs.sum(-1), 1.0, atol=1e-6)
        self.assertEqual(float(np.triu(probs, 1).max()), 0.0)  # causal

    def test_losses(self):
        rows, classes, sc = 3, 5, self.SCRATCH
        n = rows * classes
        rng = np.random.default_rng(13)
        labels = np.array([1, 4, 0], "<i4").view("<f4")
        seed = {sc: labels, sc + 8: [1.0]}

        def scalar(op, srcs, outs):
            return lambda x, y: [(op, 0, srcs(x, y), outs)]
        self.check_vjp(          # softmax cross-entropy
            n, 1,
            scalar(11, lambda x, y: (4 * x, 4 * sc, 4 * y, 4 * (sc + 16)),
                   (rows, classes, 0)),
            lambda dy, x, dx, y: [(12, 0, (4 * (sc + 16), 4 * sc, 4 * dy,
                                           4 * dx), (rows, classes, 0))],
            rng.standard_normal(n), seed, tol=5e-3)
        target = {sc: rng.standard_normal(n)}
        self.check_vjp(          # mean squared error
            n, 1,
            scalar(13, lambda x, y: (4 * x, 4 * sc, 4 * y, N), (n, 0, 0)),
            lambda dy, x, dx, y: [(14, 0, (4 * x, 4 * sc, 4 * dy, 4 * dx),
                                   (n, 0, 0))],
            rng.standard_normal(n), target)
        word = fx.f32_bits(4.0) << 32 | fx.f32_bits(2.0)  # T = 2, scale T^2
        nc = rows << 32 | classes
        self.check_vjp(          # KL distillation against a fixed teacher
            n, 1,
            scalar(15, lambda x, y: (4 * x, 4 * sc, 4 * y, 4 * (sc + 32)),
                   (4 * (sc + 64), nc, word)),
            lambda dy, x, dx, y: [(16, 0, (4 * (sc + 32), 4 * (sc + 64),
                                           4 * dy, 4 * dx), (nc, word, 0))],
            rng.standard_normal(n), target, tol=5e-3)

    def test_adamw_matches_the_textbook_recurrence(self):
        g = [0.5, -2.0, 0.0, 1e-9]
        out = self.run_ops(16, [(18, 0, (0, 16, 32, 48), (4, 0, 0))],
                           {0: [1.0, 1.0, 1.0, 1.0], 4: g})
        lr, eps, wd = 1e-3, 1e-8, 0.0
        for i, gi in enumerate(g):  # step 1: m_hat = g, v_hat = g^2
            want = 1.0 - lr * (gi / (abs(gi) + eps) + wd)
            self.assertAlmostEqual(out[i], want, places=6)

    def test_clip_norm(self):
        clip = [(28, 0, (0, fx.f32_bits(1.0), N, N), (2, 0, 0))]
        np.testing.assert_allclose(self.run_ops(4, clip, {0: [3.0, 4.0]})[:2],
                                   [0.6, 0.8], atol=1e-7)
        np.testing.assert_allclose(self.run_ops(4, clip, {0: [0.3, 0.4]})[:2],
                                   [0.3, 0.4], atol=1e-7)
        self.assertEqual(list(self.run_ops(4, clip, {0: [0.0, 0.0]})[:2]),
                         [0.0, 0.0])

    def test_the_opcodes_no_compiler_path_emits(self):
        ops = [(10, 0, (0, 24, N, N), (2, 3, 0)),    # reduce_rows -> [6, 9)
               (20, 0, (24, 36, N, N), (3, 0, 0)),   # copy        -> [9, 12)
               (0, 0, (N, N, N, N), (0, 0, 0))]      # nop
        out = self.run_ops(12, ops, {0: [1, 2, 3, 10, 20, 30]})
        self.assertEqual(list(out[6:12]), [11, 22, 33, 11, 22, 33])

    def test_quantized_and_bf16_weights_widen_exactly(self):
        q = np.array([[1, -2], [3, 127]], np.int8)
        h = np.array([[1.5, -0.25], [4.0, 1024.0]], np.float32)
        bf = (h.view(np.uint32) >> 16).astype("<u2")  # exact for these
        rodata = q.tobytes().ljust(16, b"\0") + bf.tobytes()
        ops = [(29, 0, (0, fx.RODATA_BIT, 16, fx.f32_bits(0.5)), (1, 2, 2)),
               (43, 0, (0, fx.RODATA_BIT | 16, 32, N), (1, 2, 2))]
        plan = fx.Plan(assemble(64, train=ops, rodata=rodata))
        ex = fx.Executor(plan, fx.NumpyBackend("float64"))
        ex.mem.stage(0, np.array([1.0, 2.0], "<f4").tobytes())
        ex.execute("train")
        out = np.frombuffer(ex.mem.export(), "<f4")
        np.testing.assert_allclose(out[4:6], [1, 2] @ (q * 0.5))
        np.testing.assert_allclose(out[8:10], [1, 2] @ h)

    def test_slot_memory_survives_arena_reuse(self):
        """SlotMemory (the GPU backends' model) under the aliasing the arena
        allocator really produces: overlapping writes, sub-range reads."""
        class Slots(fx.NumpyBackend):
            arena_views = False
        ops = [(19, 0, (0, fx.f32_bits(1.0), N, N), (8, 0, 0)),   # [0,8) = 1
               (19, 0, (16, fx.f32_bits(2.0), N, N), (4, 0, 0)),  # [4,8) = 2
               (5, 0, (0, 16, 32, N), (4, 0, 0)),   # [8,12) = [0,4)+[4,8)
               (9, 0, (8, 0, fx.f32_bits(3.0), N), (2, 0, 0))]    # [0,2)=3*[2,4)
        plan = fx.Plan(assemble(64, train=ops))
        outs = []
        for backend in (fx.NumpyBackend("float64"), Slots("float64")):
            ex = fx.Executor(plan, backend)
            ex.execute("train")
            outs.append(np.frombuffer(ex.mem.export(), "<f4").copy())
        np.testing.assert_array_equal(outs[0], outs[1])
        np.testing.assert_array_equal(outs[0][:12],
                                      [3, 3, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])


# --- The differential suite ----------------------------------------------------


def _write_models(out_dir):
    """Two feature models and a token decoder, NumPy only, covering every
    SMF operator: bias, the three activations, LayerNorm, a gated product
    and a residual add (MLP); RMSNorm, RoPE, attention, embedding (decoder)."""
    import export_model as em
    rng = np.random.default_rng(42)

    def mat(*shape):
        return np.ascontiguousarray(rng.standard_normal(shape) * 0.3, "<f4")

    def mlp(path, width, rich):
        b = em._SmfBuilder("x", 16)

        def tensor(name, arr):
            b.add_tensor(name, list(arr.shape), arr)
            return name
        b.add_op(em.OP_MATMUL, "mm0", ["x", tensor("w0", mat(16, width))], "z0")
        b.add_op(em.OP_ADDBIAS, "ab0", ["z0", tensor("b0", mat(width))], "zb0")
        b.add_op(em.OP_GELU if rich else em.OP_RELU, "a0", ["zb0"], "h0")
        prev = "h0"
        if rich:
            b.add_op(em.OP_LAYERNORM, "ln", ["h0", tensor("ln_g", mat(width) + 1),
                                             tensor("ln_b", mat(width))], "n")
            b.add_op(em.OP_MATMUL, "mmg", ["n", tensor("wg", mat(width, width))],
                     "g")
            b.add_op(em.OP_MATMUL, "mmu", ["n", tensor("wu", mat(width, width))],
                     "u")
            b.add_op(em.OP_SILU, "silu", ["g"], "s")
            b.add_op(em.OP_MUL, "gate", ["s", "u"], "m")
            b.add_op(em.OP_ADD, "res", ["m", "n"], "r")
            b.add_op(em.OP_MATMUL, "mm1", ["r", tensor("w1", mat(width, width))],
                     "z1")
            b.add_op(em.OP_ADDBIAS, "ab1", ["z1", tensor("b1", mat(width))],
                     "zb1")
            b.add_op(em.OP_RELU, "a1", ["zb1"], "h1")
            prev = "h1"
        b.add_op(em.OP_MATMUL, "mmo", [prev, tensor("wo", mat(width, 4))], "zo")
        b.add_op(em.OP_ADDBIAS, "abo", ["zo", tensor("bo", mat(4))], "logits")
        with open(path, "wb") as f:
            b.write(f)

    mlp(os.path.join(out_dir, "mlp.smf"), 24, rich=True)
    mlp(os.path.join(out_dir, "teacher.smf"), 32, rich=False)
    x = rng.standard_normal((96, 16)).astype(np.float32)
    y = (x[:, :4].sum(axis=1) > 0).astype(np.int32) + 2 * (x[:, 0] > 0)
    with contextlib.redirect_stdout(io.StringIO()):
        em.export_sds(x, y, os.path.join(out_dir, "class.sds"))
        em.export_sds(x, np.eye(4, dtype=np.float32)[y],
                      os.path.join(out_dir, "dense.sds"), label_kind=2)
        em.export_sds(x, None, os.path.join(out_dir, "none.sds"))
        em._demo_decoder(out_dir, vocab=40, dim=24, heads=4, seq=6,
                         blocks_n=2, samples=48, seed=1)


# (name, model, corpus, data batch, compiler flags)
MATRIX = [
    ("mlp_xent_adamw", "mlp.smf", "class.sds", 8,
     ["--clip-norm", "0.5", "--lr-schedule", "cosine", "--warmup", "2",
      "--min-lr-factor", "0.1", "--steps", "8"]),
    ("mlp_unfused", "mlp.smf", "class.sds", 8, ["--no-fuse-epilogue"]),
    ("mlp_mse_sgd", "mlp.smf", "dense.sds", 8,
     ["--loss", "mse", "--optimizer", "sgd", "--weight-decay", "0.01",
      "--clip-norm", "0.5"]),                       # the fused SGD clip
    ("mlp_kl", "mlp.smf", "none.sds", 8,
     ["--loss", "kl", "--teacher", "teacher.smf", "--temperature", "2"]),
    ("mlp_xent_kl", "mlp.smf", "class.sds", 8,
     ["--loss", "xent+kl", "--teacher", "teacher.smf",
      "--distill-weight", "0.4"]),
    ("mlp_q8", "mlp.smf", "class.sds", 8, ["--quantize-base"]),
    ("mlp_bf16", "mlp.smf", "class.sds", 8, ["--bf16-base"]),
    ("mlp_accum", "mlp.smf", "class.sds", 8,
     ["--grad-accum", "3", "--clip-norm", "1.0"]),
    ("dec_f32", "decoder.smf", "decoder_corpus.sds", 12,
     ["--lora-rank", "4"]),
    ("dec_q8_accum", "decoder.smf", "decoder_corpus.sds", 12,
     ["--quantize-base", "--grad-accum", "2"]),
    ("dec_bf16", "decoder.smf", "decoder_corpus.sds", 12, ["--bf16-base"]),
    # The v11 program shapes plan v12 replaces, still compiled on request:
    # a standalone clip per tensor, RoPE angles recomputed in place.
    ("dec_v11_shape", "decoder.smf", "decoder_corpus.sds", 12,
     ["--clip-norm", "0.5", "--no-fuse-clip", "--no-rope-table",
      "--no-fuse-elementwise"]),
    ("mlp_unfused_maps", "mlp.smf", "class.sds", 8,
     ["--no-fuse-elementwise", "--no-fuse-epilogue"]),
]
# Opcodes no seeml-update-compile invocation produces today: kNop is a
# placeholder, kCopy a utility no pass selects, and kReduceRows the bias
# gradient, which needs-set pruning removes because LoRA freezes every
# bias. They are interpreted, and InterpreterSelfCheck runs them from a
# hand-assembled plan; when a compiler path starts emitting one, this set
# shrinks and the matrix grows.
NEVER_COMPILED = {"nop", "copy", "reduce.rows"}


@unittest.skipIf(np is None, "NumPy not installed")
@unittest.skipUnless(HAVE_CPP, f"no built seeml tools under {BUILD}")
class DifferentialSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="seeml-frontier-test-")
        cls.dir = cls._tmp.name
        _write_models(cls.dir)
        cls.plans = {}
        for name, model, corpus, batch, flags in MATRIX:
            out = os.path.join(cls.dir, name)
            flags = [os.path.join(cls.dir, f) if f.endswith(".smf") else f
                     for f in flags]
            done = subprocess.run(
                [COMPILER, "--source", os.path.join(cls.dir, model), "--out",
                 out, "--data-batch", str(batch), "--no-embed"] + flags,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if done.returncode != 0:
                raise AssertionError(f"{name}: compile failed\n{done.stdout}")
            cls.plans[name] = (os.path.join(out, "update_plan.seeu"),
                               os.path.join(cls.dir, corpus))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def diff(self, name, *extra):
        plan, corpus = self.plans[name]
        report = os.path.join(self.dir, name + ".json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            status = fx.main(["diff", plan, "--corpus", corpus, "--probe",
                              PROBE, "--steps", "3", "--report", report,
                              *extra])
        with open(report) as f:
            return status, json.load(f), buf.getvalue()

    def test_the_matrix_agrees_and_covers_the_instruction_set(self):
        covered = set()
        for name, *_ in MATRIX:
            with self.subTest(plan=name):
                status, report, log = self.diff(name)
                self.assertEqual(status, 0, log)
                self.assertTrue(report["agrees"])
                for opcode, stat in report["opcodes"].items():
                    self.assertGreater(stat["extents"], 0, opcode)
                    self.assertLess(stat["worst_ratio"], 1.0)
                covered |= set(report["opcodes"])
        self.assertEqual(set(fx.OPCODES.values()) - covered, NEVER_COMPILED)

    def test_the_oracle_has_teeth(self):
        """One wrong constant in one opcode is caught, and blamed on it."""
        for opcode, broken in (
                (22, fx._unary(lambda x, t: 0.5 * t * (1.0 + x.tanh(
                    fx.GELU_C * (t + 0.045 * t * t * t))))),   # GELU's cubic
                (18, lambda m, x, ins, step: fx._adamw(
                    m, x, ins, dict(step, eps=2e-8))),):        # AdamW's eps
            original = fx.INTERPRETER[opcode]
            fx.INTERPRETER[opcode] = broken
            try:
                status, report, _ = self.diff("mlp_unfused")
            finally:
                fx.INTERPRETER[opcode] = original
            self.assertEqual(status, 1)
            blamed = {re.search(r"#\d+ (\S+)", line).group(1)
                      for line in report["failures"]}
            self.assertEqual(blamed, {fx.OPCODES[opcode]})

    def test_float32_backends_agree_too(self):
        for backend in ("numpy32", "torch", "mlx"):
            try:
                fx.make_backend(backend)
            except ImportError:
                continue
            for name in ("mlp_xent_kl", "dec_q8_accum"):
                with self.subTest(backend=backend, plan=name):
                    status, _, log = self.diff(name, "--backend", backend)
                    self.assertEqual(status, 0, log)

    def test_the_probe_is_strict_about_arguments_and_paths(self):
        plan, _ = self.plans["mlp_mse_sgd"]
        work = os.path.join(self.dir, "probe")
        os.makedirs(work, exist_ok=True)
        arena = os.path.join(work, "arena.in")
        with open(arena, "wb") as f:
            f.write(fx.Plan.load(plan).initial_arena())
        out = os.path.join(work, "arena.out")

        def probe(*argv):
            done = subprocess.run([PROBE, *argv], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
            return done.returncode, done.stderr
        base = ["--plan", plan, "--section", "merge", "--arena-in", arena]
        self.assertEqual(probe(*base, "--arena-out", out)[0], 0)
        self.assertTrue(os.path.isfile(out))
        self.assertFalse(os.path.exists(out + ".tmp"))  # staged, then renamed
        done = subprocess.run([PROBE, *base, "--arena-out", out, "--profile",
                               "2"], stdout=subprocess.PIPE, text=True,
                              stderr=subprocess.DEVNULL, check=True)
        profile = json.loads(done.stdout)
        self.assertEqual(profile["executions"], 2)
        self.assertTrue(any(row["key"].startswith("op4 M")  # gemm.acc_nn
                            for row in profile["rows"]))

        for argv, code, message in (
                ((*base, "--arena-out", out, "--bogus"), 2, "unknown argument"),
                ((*base, "--arena-out"), 2, "missing its value"),
                ((*base, "--arena-out", out, "--section", "x"), 2, "--section"),
                ((*base, "--arena-out", out, "--time", "0"), 2, "--time"),
                ((*base, "--arena-out", out, "--time", "2", "--trace",
                  out + ".t"), 2, "mutually exclusive"),
                ((*base, "--arena-out", out, "--profile", "2", "--time", "2"),
                 2, "mutually exclusive"),
                ((*base, "--arena-out", out, "--profile", "0"), 2,
                 "--profile"),
                (("--plan", work, "--section", "merge", "--arena-in", arena,
                  "--arena-out", out), 1, "not a regular file"),
                ((*base, "--arena-out", os.path.join(work, "nowhere", "a")),
                 1, "does not exist"),
                ((*base, "--arena-out", work), 1, "not a regular file")):
            status, err = probe(*argv)
            self.assertEqual(status, code, err)
            self.assertIn(message, err)

        # An output that is a symlink is refused, never written through.
        victim = os.path.join(work, "victim")
        with open(victim, "w") as f:
            f.write("untouched")
        link = os.path.join(work, "link.out")
        os.symlink(victim, link)
        for flag in ("--arena-out", "--trace"):
            argv = list(base) + (["--arena-out", link] if flag == "--arena-out"
                                 else ["--arena-out", out, "--trace", link])
            status, err = probe(*argv)
            self.assertEqual(status, 1, err)
            self.assertIn("symlink is refused", err)
        with open(victim) as f:
            self.assertEqual(f.read(), "untouched")

        # A run that fails leaves no output behind to be mistaken for one.
        os.remove(out)
        with open(arena, "ab") as f:
            f.write(b"\0")  # no longer the plan's arena size
        status, err = probe(*base, "--arena-out", out, "--trace", out + ".t")
        self.assertEqual(status, 1)
        self.assertIn("the plan declares", err)
        self.assertEqual(sorted(os.listdir(work)),
                         ["arena.in", "link.out", "victim"])

    def test_run_reproduces_the_emitted_package(self):
        """The whole update — tail split, per-epoch shuffle, schedule, G
        micro-batches, evaluation — against the real model_update binary:
        same loss curve, same validation losses."""
        out = os.path.join(self.dir, "e2e")
        done = subprocess.run(
            [COMPILER, "--source", os.path.join(self.dir, "decoder.smf"),
             "--out", out, "--data-batch", "12", "--grad-accum", "2",
             "--lr-schedule", "cosine", "--warmup", "3", "--steps", "25",
             "--build"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        binary = os.path.join(out, "model_update")
        if done.returncode != 0 or not os.path.exists(binary):
            self.skipTest("no C++ toolchain to build the emitted package")
        corpus = os.path.join(self.dir, "decoder_corpus.sds")
        log = os.path.join(out, "loss.csv")
        trained = subprocess.run(
            [binary, "--model", os.path.join(self.dir, "decoder.smf"),
             "--data", corpus, "--out", os.path.join(out, "updated.smf"),
             "--steps", "25", "--seed", "11", "--val-frac", "0.25",
             "--loss-log", log, "--force"], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
        self.assertEqual(trained.returncode, 0, trained.stdout)
        before, after = re.search(r"validation loss (\S+) -> (\S+)",
                                  trained.stdout).groups()
        with open(log) as f:
            cpp = [float(line.split(",")[1]) for line in f.read().split()[1:]]
        report = os.path.join(out, "run.json")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fx.main([
                "run", os.path.join(out, "update_plan.seeu"), "--corpus",
                corpus, "--steps", "25", "--seed", "11", "--val-fraction",
                "0.25", "--log-every", "0", "--report", report]), 0)
        with open(report) as f:
            ran = json.load(f)
        self.assertEqual(len(cpp), 25)
        np.testing.assert_allclose(ran["loss_curve"], cpp, rtol=2e-5)
        self.assertAlmostEqual(ran["val_initial_loss"], float(before), 4)
        self.assertAlmostEqual(ran["val_final_loss"], float(after), 4)

    def test_run_trains_and_price_reports(self):
        plan, corpus = self.plans["dec_f32"]
        report = os.path.join(self.dir, "run.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(fx.main(["run", plan, "--corpus", corpus,
                                      "--steps", "30", "--report", report,
                                      "--log-every", "0"]), 0)
        with open(report) as f:
            ran = json.load(f)
        self.assertLess(ran["loss_curve"][-1], ran["loss_curve"][0])
        self.assertLess(ran["val_final_loss"], ran["val_initial_loss"])

        with contextlib.redirect_stdout(buf):
            self.assertEqual(fx.main([
                "price", plan, "--backends", "numpy32", "--probe", PROBE,
                "--steps-lo", "2",
                "--steps-hi", "4", "--repeats", "1", "--peak-gflops", "100",
                "--report", report]), 0)
        with open(report) as f:
            priced = json.load(f)
        for entry in (priced["frontier"]["numpy32"], priced["cpp"]["cpu"]):
            self.assertTrue(entry["available"])
            for key in ("step_ms", "rows_per_s", "tokens_per_s", "it_per_s",
                        "gemm_gflops", "mfu"):
                self.assertGreater(entry[key], 0)
        self.assertIn("speedup_vs_cpp_cpu", priced["frontier"]["numpy32"])


if __name__ == "__main__":
    unittest.main()
