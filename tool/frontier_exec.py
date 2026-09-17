#!/usr/bin/env python3
"""Run a compiled .seeu update plan through NumPy, PyTorch or MLX (P4, #78).

A build-host interpreter for the plan instruction set — never shipped on a
device, never imported by the compiler or the runtime. It gives the C++
runtime two things it structurally cannot give itself:

  an oracle   every opcode written a second time, independently, from the
              documented mathematics (source/plan/instruction.h). The
              runtime's own tests compare it bit-for-bit against itself, and
              a bit-exact self-comparison reproduces a kernel bug faithfully;
              this one does not share the bug.
  a price     what Accelerate/AMX (PyTorch CPU) or the unified-memory GPU
              (MLX, PyTorch MPS) would buy ON THIS PLAN'S OWN SHAPES, in the
              fields seeml-bench reports, next to the C++ number.

Usage:
    frontier_exec.py info  plan.seeu
    frontier_exec.py diff  plan.seeu --probe build/seeml-plan-probe
                           [--corpus c.sds] [--steps N] [--backend numpy]
                           [--cpp-backend cpu|metal] [--rtol F] [--atol F]
                           [--report out.json]
    frontier_exec.py run   plan.seeu --corpus c.sds [--steps N] [--seed S]
                           [--val-fraction F] [--backend numpy]
    frontier_exec.py price plan.seeu [--corpus c.sds]
                           [--backends torch,mlx] [--probe PATH]
                           [--steps-lo N --steps-hi N --repeats N]
                           [--peak-gflops F] [--report out.json]

`info` needs only the standard library. Everything else needs NumPy; the
`torch`, `torch-mps` and `mlx` backends need those packages and are
reported as unavailable, not failed, where they are missing.

Backends:
    numpy      float64 compute over the f32 arena — the oracle's default: it
               owes nothing to any BLAS's accumulation order.
    numpy32    float32 compute (what a NumPy user would actually run).
    torch      PyTorch on the CPU (Accelerate/AMX on Apple silicon, MKL or
               OpenBLAS elsewhere), zero-copy over the arena bytes.
    torch-mps  PyTorch on the Apple GPU.
    mlx        MLX on the Apple GPU (lazy; evaluated once per step), with
               MLX_ENABLE_TF32=0: true f32 GEMMs.
    mlx-tf32   MLX as it ships: on recent Apple GPUs its matmul defaults to
               a reduced-precision mode (~1e-3 relative error, measured and
               printed) that is faster and is NOT the arithmetic the plan
               specifies — price it, never use it as the oracle. MLX reads
               the switch once per process, so one price run takes one of
               the two.

`diff` is lockstep at INSTRUCTION granularity. `seeml-plan-probe --trace`
runs a section first and returns every extent every instruction wrote;
this interpreter then replays the section, executing instruction i on
exactly the bytes the C++ runtime had before its instruction i, comparing
its writes against the trace at  |a - b| <= atol + rtol * max|b|  (a
per-tensor scale: a cancellation residue is judged against the tensor it
lives in), and adopting the C++ bytes before instruction i+1. Each opcode
is therefore judged alone: no upstream rounding is amplified into a
downstream verdict (AdamW's first steps divide by |g| + eps, which turns
1e-7 of gradient noise into a visible parameter difference), and a
disagreement names the one instruction that owns it. The train, step
(gradient accumulation), eval and merge sections are all compared. Exit 1 on any disagreement, 2 on a bad command line.
"""
import argparse
import bisect
import json
import math
import os
import resource
import struct
import subprocess
import sys
import tempfile
import time

# --- The .seeu container (source/plan/schema.h, instruction.h) ---------------

SEEU_MAGIC = 0x55454553
SEEU_OLDEST, SEEU_NEWEST = 4, 11
RODATA_BIT = 1 << 63
NULL_REF = (1 << 64) - 1

_HEADER = struct.Struct(
    "<II" "QQ" "QQQQ" "II" "Q" "QQQQQQQQQQ" "fffff" "I" "QQ" "QQ" "QQ"
    "II" "Q" "ff" "II" "Q" "QQ")
_HEADER_FIELDS = (
    "magic version arena_size persistent_size input_ref input_floats "
    "label_ref label_bytes label_kind optimizer_kind loss_ref "
    "train_instr_offset train_instr_count merge_instr_offset "
    "merge_instr_count rodata_offset rodata_size persist_init_offset "
    "persist_init_size emit_table_offset emit_count lr beta1 beta2 eps "
    "weight_decay gemm_tile_k batch default_steps eval_instr_offset "
    "eval_instr_count source_model_hash plan_hash lr_schedule gemm_tile_n "
    "warmup_steps min_lr_factor clip_norm input_kind grad_accum_steps "
    "seq_len step_instr_offset step_instr_count").split()
assert _HEADER.size == 280 and len(_HEADER_FIELDS) == 43
_INSTRUCTION = struct.Struct("<HHI4Q3Q")
assert _INSTRUCTION.size == 64

OPCODES = {
    0: "nop", 1: "gemm.nn", 2: "gemm.nt", 3: "gemm.tn", 4: "gemm.acc_nn",
    5: "add.ew", 6: "add.bias", 7: "relu.fwd", 8: "relu.bwd", 9: "scale",
    10: "reduce_rows", 11: "softmax_xent.fwd", 12: "softmax_xent.bwd",
    13: "mse.fwd", 14: "mse.bwd", 15: "kl_distill.fwd", 16: "kl_distill.bwd",
    17: "sgd.step", 18: "adamw.step", 19: "fill", 20: "copy", 21: "mul.ew",
    22: "gelu.fwd", 23: "gelu.bwd", 24: "silu.fwd", 25: "silu.bwd",
    26: "layer_norm.fwd", 27: "layer_norm.bwd", 28: "clip_norm",
    29: "gemm.nn_q8", 30: "gemm.nt_q8", 31: "rms_norm.fwd",
    32: "rms_norm.bwd", 33: "rope.fwd", 34: "rope.bwd", 35: "attn.fwd",
    36: "attn.dp", 37: "attn.dv", 38: "softmax_rows.bwd", 39: "attn.dq",
    40: "attn.dk", 41: "embed.fwd", 42: "accumulate", 43: "gemm.nn_bf16",
    44: "gemm.nt_bf16",
}
GEMM_OPCODES = (1, 2, 3, 4, 29, 30, 43, 44)
SECTIONS = ("train", "step", "eval", "merge")
NORM_EPS = 1e-5  # runtime/executor/normalization.cc; SMF carries none (#96)
GELU_C, GELU_A = 0.7978845608028654, 0.044715  # the tanh approximation


class PlanError(Exception):
    """A plan, corpus or environment this tool refuses (exit 1)."""


class Instruction:
    __slots__ = ("opcode", "flags", "src", "out")

    def __init__(self, opcode, flags, src, out):
        self.opcode, self.flags, self.src, self.out = opcode, flags, src, out


def bits_f32(word):
    return struct.unpack("<f", struct.pack("<I", word & 0xFFFFFFFF))[0]


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f32(value):
    """`value` rounded to the nearest f32, as a Python float."""
    return bits_f32(f32_bits(value))


def hi_lo(word):
    return word >> 32, word & 0xFFFFFFFF


class Plan:
    """A parsed .seeu blob: the header as attributes, the four instruction
    streams, and the byte sections. Bounds are checked here because nothing
    downstream re-checks them — this is a build-host tool, but it still must
    not index outside a file someone handed it."""

    def __init__(self, blob, path="<memory>"):
        self.path, self.blob = path, blob
        if len(blob) < _HEADER.size:
            raise PlanError(f"{path}: truncated (no plan header)")
        for name, value in zip(_HEADER_FIELDS, _HEADER.unpack_from(blob, 0)):
            setattr(self, name, value)
        if self.magic != SEEU_MAGIC:
            raise PlanError(f"{path}: not a .seeu plan")
        if not SEEU_OLDEST <= self.version <= SEEU_NEWEST:
            raise PlanError(f"{path}: plan version {self.version} is outside "
                            f"[{SEEU_OLDEST}, {SEEU_NEWEST}]")
        self.sections = {}
        for name in SECTIONS:
            off = getattr(self, name + "_instr_offset")
            count = getattr(self, name + "_instr_count")
            self._bounds(name, off, count * _INSTRUCTION.size)
            self.sections[name] = [
                Instruction(op, flags, (a, b, c, d), (x, y, z))
                for op, flags, _, a, b, c, d, x, y, z in
                _INSTRUCTION.iter_unpack(
                    blob[off:off + count * _INSTRUCTION.size])]
        self._bounds("rodata", self.rodata_offset, self.rodata_size)
        self._bounds("persist_init", self.persist_init_offset,
                     self.persist_init_size)
        if self.persist_init_size != self.persistent_size:
            raise PlanError(f"{path}: persist_init_size != persistent_size")
        if self.persistent_size > self.arena_size:
            raise PlanError(f"{path}: persistent segment exceeds the arena")
        self.grad_accum = max(1, self.grad_accum_steps)

    def _bounds(self, what, off, size):
        if off > len(self.blob) or size > len(self.blob) - off:
            raise PlanError(f"{self.path}: the {what} section lies outside "
                            "the file")

    @classmethod
    def load(cls, path):
        try:
            with open(path, "rb") as f:
                # A bytearray: PyTorch's zero-copy frombuffer wants a
                # writable buffer even for tensors it only reads.
                return cls(bytearray(f.read()), path)
        except OSError as e:
            raise PlanError(f"cannot read {path}: {e}")

    def initial_arena(self):
        """The engine's arena at load: zeros, with the persistent prefix
        (adapters, optimizer state, accumulators) set to its initial image."""
        arena = bytearray(self.arena_size)
        o = self.persist_init_offset
        arena[:self.persistent_size] = self.blob[o:o + self.persistent_size]
        return arena

    def effective_lr(self, step):
        """UpdateEngine::EffectiveLr in f32 arithmetic: linear warmup, then
        cosine to lr * min_lr_factor across default_steps, clamped after."""
        base = f32(self.lr)
        if self.lr_schedule != 1:
            return base
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return f32(f32(base * f32(step)) / f32(self.warmup_steps))
        horizon = max(0, self.default_steps - self.warmup_steps)
        floor = f32(base * f32(self.min_lr_factor))
        if horizon == 0 or step - self.warmup_steps >= horizon:
            return floor
        t = f32(f32(step - self.warmup_steps) / f32(horizon))
        cos = f32(math.cos(f32(f32(math.pi) * t)))
        return f32(floor + f32(f32(f32(base - floor) * 0.5) * f32(1.0 + cos)))

    def gemm_flops(self, section="train"):
        """seeml-bench's GemmFlopsPerStep: 2*M*N*K over the GEMM family."""
        return sum(2 * i.out[0] * i.out[1] * i.out[2]
                   for i in self.sections[section]
                   if i.opcode in GEMM_OPCODES)

    def label_classes(self):
        """The narrowest softmax width over the train and eval programs —
        the bound the feeder contract holds class labels to."""
        widths = [i.out[1] for name in ("train", "eval")
                  for i in self.sections[name] if i.opcode == 11]
        return min(widths) if widths else 0

    def vocab(self):
        sizes = [i.out[1] >> 32 for name in ("train", "eval")
                 for i in self.sections[name] if i.opcode == 41]
        return min(sizes) if sizes else 0

    def histogram(self):
        out = {}
        for name in SECTIONS:
            for ins in self.sections[name]:
                key = OPCODES.get(ins.opcode, f"unknown({ins.opcode})")
                out.setdefault(key, {}).setdefault(name, 0)
                out[key][name] += 1
        return out


# --- The SDS corpus and the feeder (runtime/feeder/dataset.cc) ---------------

SDS_MAGIC = 0x31534453
_MASK64 = (1 << 64) - 1


def _splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return state, z ^ (z >> 31)


class Corpus:
    """An .sds corpus served exactly as the C++ feeder serves it: a
    positional tail split, then a fresh Fisher-Yates permutation per epoch
    from one splitmix64 stream. Token corpora move whole records."""

    def __init__(self, np, path):
        self.np = np
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError as e:
            raise PlanError(f"cannot read {path}: {e}")
        if len(raw) < 40:
            raise PlanError(f"{path}: truncated (no SDS header)")
        (magic, version, self.num_samples, self.input_dim, self.label_kind,
         self.input_kind, self.label_dim) = struct.unpack_from("<IIQQIIQ", raw)
        if magic != SDS_MAGIC or version not in (1, 2):
            raise PlanError(f"{path}: not an SDS v1/v2 corpus")
        n, d = self.num_samples, self.input_dim
        if self.input_kind == 1:
            self.record = (d + 1) * 4
            if len(raw) - 40 < n * self.record:
                raise PlanError(f"{path}: truncated records")
            self.tokens = np.frombuffer(raw, "<i4", n * (d + 1), 40).reshape(
                n, d + 1)
        else:
            self.label_nbytes = {0: 0, 1: 4, 2: 4 * self.label_dim}.get(
                self.label_kind)
            if self.label_nbytes is None:
                raise PlanError(f"{path}: unknown label kind")
            self.record = d * 4 + self.label_nbytes
            if len(raw) - 40 < n * self.record:
                raise PlanError(f"{path}: truncated records")
            rows = np.frombuffer(raw, np.uint8, n * self.record, 40).reshape(
                n, self.record)
            self.inputs = np.ascontiguousarray(rows[:, :d * 4])
            self.labels = np.ascontiguousarray(rows[:, d * 4:])
        self.order, self.cursor, self.state = None, 0, 0

    def _view(self, lo, hi):
        other = object.__new__(Corpus)
        other.__dict__.update(self.__dict__)
        other.num_samples = hi - lo
        if self.input_kind == 1:
            other.tokens = self.tokens[lo:hi]
        else:
            other.inputs, other.labels = self.inputs[lo:hi], self.labels[lo:hi]
        other.order, other.cursor = None, 0
        return other

    def split_validation(self, fraction):
        """(train, validation): the tail `fraction` of the samples is held
        out, by position — Dataset::SplitValidation."""
        n = self.num_samples
        val_n = min(max(int(n * fraction), 1), n - 1)
        return self._view(0, n - val_n), self._view(n - val_n, n)

    def enable_shuffle(self, seed):
        self.state, z = _splitmix64(seed & _MASK64)
        self.state = z | 1
        self._reshuffle()

    def _reshuffle(self):
        order = list(range(self.num_samples))
        for i in range(self.num_samples - 1, 0, -1):
            self.state, z = _splitmix64(self.state)
            j = z % (i + 1)
            order[i], order[j] = order[j], order[i]
        self.order = order

    def _next(self):
        i = self.cursor if self.order is None else self.order[self.cursor]
        self.cursor += 1
        if self.cursor == self.num_samples:
            self.cursor = 0
            if self.order is not None:
                self._reshuffle()
        return i

    def batch(self, plan):
        """(input_bytes, label_bytes) for one compiled batch."""
        np = self.np
        if self.input_kind == 1:
            ids = [self._next() for _ in range(plan.batch // self.input_dim)]
            recs = self.tokens[ids]
            return (np.ascontiguousarray(recs[:, :-1]).tobytes(),
                    np.ascontiguousarray(recs[:, 1:]).tobytes())
        ids = [self._next() for _ in range(plan.batch)]
        return self.inputs[ids].tobytes(), self.labels[ids].tobytes()

    def check(self, plan):
        rows = plan.seq_len if plan.input_kind == 1 else 1
        if self.input_kind != plan.input_kind:
            raise PlanError("the corpus and the plan disagree on input kind")
        if plan.input_kind == 1 and self.input_dim != plan.seq_len:
            raise PlanError(f"the corpus holds {self.input_dim}-token "
                            f"records; the plan wants {plan.seq_len}")
        if plan.input_kind == 0 and (
                self.input_dim * plan.batch != plan.input_floats):
            raise PlanError("the corpus input width does not match the plan")
        if plan.batch % rows:
            raise PlanError("the plan batch is not a whole number of records")


class SyntheticCorpus:
    """Seeded batches shaped by the plan alone, for pricing a plan whose
    corpus is not at hand: per-step cost does not depend on the values."""

    def __init__(self, np, plan, seed=0):
        self.np, self.rng = np, np.random.default_rng(seed)
        self.num_samples = 1 << 30

    def batch(self, plan):
        np, rng = self.np, self.rng
        if plan.input_kind == 1:
            tokens = rng.integers(0, max(plan.vocab(), 1), plan.batch + 1,
                                  dtype=np.int32)
            return tokens[:-1].tobytes(), tokens[1:].tobytes()
        x = rng.standard_normal(plan.input_floats, dtype=np.float32)
        if plan.label_kind == 1:
            y = rng.integers(0, max(plan.label_classes(), 1),
                             plan.label_bytes // 4, dtype=np.int32)
        else:
            y = rng.standard_normal(plan.label_bytes // 4, dtype=np.float32)
        return x.tobytes(), y.tobytes()

    def check(self, plan):
        pass


# --- Array backends -----------------------------------------------------------
# One small adapter per framework. The opcode implementations below touch
# tensors only through these methods and through arithmetic operators,
# indexing and .reshape/.T, which NumPy, PyTorch and MLX all spell alike.


class NumpyBackend:
    arena_views = True  # tensors are zero-copy views of the arena bytes

    def __init__(self, compute="float64"):
        import numpy as np
        self.np, self.name = np, "numpy" if compute == "float64" else "numpy32"
        self.compute = np.dtype(compute)
        self.device = f"host CPU, NumPy {np.__version__} ({compute})"

    def view(self, buf, dtype, count, offset):
        return self.np.frombuffer(buf, dtype, count, offset)

    def up(self, t):
        return t.astype(self.compute, copy=False)

    def store(self, view, t):
        view[...] = t.reshape(view.shape)

    def from_numpy(self, a):
        return a

    def to_numpy(self, t):
        return self.np.asarray(t)

    def scalar(self, t):
        return float(self.np.asarray(t).reshape(-1)[0])

    def sync(self, tensors=()):
        pass

    def matmul(self, a, b):
        return a @ b

    def perm(self, t, axes):
        return self.np.transpose(t, axes)

    def sum(self, t, axis, keepdims=False):
        return t.sum(axis=axis, keepdims=keepdims)

    def mean(self, t, axis, keepdims=False):
        return t.mean(axis=axis, keepdims=keepdims)

    def max(self, t, axis):
        return t.max(axis=axis, keepdims=True)

    def stack_last(self, a, b):
        return self.np.stack([a, b], axis=-1)

    def take_rows(self, table, idx):
        return table[idx]

    def one_hot(self, idx, width):
        out = self.np.zeros((idx.shape[0], width), self.compute)
        out[self.np.arange(idx.shape[0]), idx] = 1.0
        return out

    def causal(self, s):
        return self.np.tril(self.np.ones((s, s), self.compute))

    def positions(self, s):
        return self.np.arange(s, dtype=self.compute)

    def full(self, n, value):
        return self.np.full(n, value, self.compute)

    def scalar_tensor(self, value):
        return self.np.full(1, value, self.compute)

    def __getattr__(self, name):  # tanh exp log sqrt cos sin maximum where
        return getattr(self.np, name)


class TorchBackend:
    def __init__(self, device="cpu"):
        import numpy as np
        import torch
        self.np, self.torch = np, torch
        self.name = "torch" if device == "cpu" else "torch-" + device
        if device == "mps" and not torch.backends.mps.is_available():
            raise ImportError("PyTorch reports no MPS device")
        self.dev = torch.device(device)
        self.arena_views = device == "cpu"
        self.compute = torch.float32
        self.device = (f"PyTorch {torch.__version__} on {device}"
                       + (f", {torch.get_num_threads()} threads"
                          if device == "cpu" else ""))

    _DTYPES = {"<f4": "float32", "<i4": "int32", "<i1": "int8",
               "<u2": "int16"}

    def view(self, buf, dtype, count, offset):
        return self.torch.frombuffer(
            buf, dtype=getattr(self.torch, self._DTYPES[dtype]), count=count,
            offset=offset)

    def up(self, t):
        return t if t.dtype == self.compute else t.to(self.compute)

    def store(self, view, t):
        view.copy_(t.reshape(view.shape))

    def from_numpy(self, a):
        return self.torch.from_numpy(self.np.array(a)).to(self.dev)

    def to_numpy(self, t):
        return t.detach().to("cpu").numpy()

    def scalar(self, t):
        return float(t.reshape(-1)[0].item())

    def sync(self, tensors=()):
        if self.dev.type == "mps":
            self.torch.mps.synchronize()

    def matmul(self, a, b):
        return self.torch.matmul(a, b)

    def perm(self, t, axes):
        return t.permute(*axes)

    def sum(self, t, axis, keepdims=False):
        return t.sum(dim=axis, keepdim=keepdims)

    def mean(self, t, axis, keepdims=False):
        return t.mean(dim=axis, keepdim=keepdims)

    def max(self, t, axis):
        return t.max(dim=axis, keepdim=True).values

    def stack_last(self, a, b):
        return self.torch.stack([a, b], dim=-1)

    def take_rows(self, table, idx):
        return table[idx.long()]

    def one_hot(self, idx, width):
        return self.torch.nn.functional.one_hot(idx.long(), width).to(
            self.compute)

    def causal(self, s):
        return self.torch.tril(self.torch.ones(s, s, dtype=self.compute,
                                               device=self.dev))

    def positions(self, s):
        return self.torch.arange(s, dtype=self.compute, device=self.dev)

    def full(self, n, value):
        return self.torch.full((n,), value, dtype=self.compute,
                               device=self.dev)

    def scalar_tensor(self, value):
        return self.full(1, value)

    def maximum(self, t, floor):
        return self.torch.clamp(t, min=floor)

    def __getattr__(self, name):  # tanh exp log sqrt cos sin where
        return getattr(self.torch, name)


class MlxBackend:
    arena_views = False
    _mode = None  # MLX latches MLX_ENABLE_TF32 on first use: one per process

    def __init__(self, tf32=False):
        import numpy as np
        if MlxBackend._mode not in (None, tf32):
            raise ImportError("MLX already runs with TF32 "
                              f"{'on' if MlxBackend._mode else 'off'} in this "
                              "process; price the other mode in its own run")
        if "mlx.core" not in sys.modules:
            os.environ["MLX_ENABLE_TF32"] = "1" if tf32 else "0"
        import mlx.core as mx
        MlxBackend._mode = tf32
        self.np, self.mx = np, mx
        self.name = "mlx-tf32" if tf32 else "mlx"
        self.compute = mx.float32
        # Say what the GEMMs really do rather than what was asked for.
        rng = np.random.default_rng(0)
        a, b = (rng.standard_normal((96, 96)).astype(np.float32)
                for _ in range(2))
        exact = a.astype(np.float64) @ b.astype(np.float64)
        self.matmul_rel_error = float(
            np.abs(np.array(mx.matmul(mx.array(a), mx.array(b))) -
                   exact).max() / np.abs(exact).max())
        if not tf32 and self.matmul_rel_error > 1e-5:
            raise ImportError("MLX's matmul is not f32-exact here (relative "
                              f"error {self.matmul_rel_error:.1e}) — was "
                              "mlx imported before this tool with TF32 on?")
        self.device = (f"MLX {getattr(mx, '__version__', '?')} on "
                       f"{mx.default_device()}, matmul relative error "
                       f"{self.matmul_rel_error:.1e}")

    def up(self, t):
        return t if t.dtype == self.compute else t.astype(self.compute)

    def from_numpy(self, a):
        return self.mx.array(a)

    def to_numpy(self, t):
        return self.np.array(t)

    def scalar(self, t):
        return float(t.reshape(-1)[0].item())

    def sync(self, tensors=()):
        self.mx.eval(*tensors) if tensors else self.mx.synchronize()

    def matmul(self, a, b):
        return self.mx.matmul(a, b)

    def perm(self, t, axes):
        return self.mx.transpose(t, axes)

    def sum(self, t, axis, keepdims=False):
        return self.mx.sum(t, axis=axis, keepdims=keepdims)

    def mean(self, t, axis, keepdims=False):
        return self.mx.mean(t, axis=axis, keepdims=keepdims)

    def max(self, t, axis):
        return self.mx.max(t, axis=axis, keepdims=True)

    def stack_last(self, a, b):
        return self.mx.stack([a, b], axis=-1)

    def take_rows(self, table, idx):
        return table[idx]

    def one_hot(self, idx, width):
        return (self.mx.arange(width)[None, :] == idx[:, None]).astype(
            self.compute)

    def causal(self, s):
        return self.mx.tril(self.mx.ones((s, s), dtype=self.compute))

    def positions(self, s):
        return self.mx.arange(s, dtype=self.compute)

    def full(self, n, value):
        return self.mx.full((n,), value, dtype=self.compute)

    def scalar_tensor(self, value):
        return self.full(1, value)

    def __getattr__(self, name):  # tanh exp log sqrt cos sin maximum where
        return getattr(self.mx, name)


BACKENDS = {
    "numpy": lambda: NumpyBackend("float64"),
    "numpy32": lambda: NumpyBackend("float32"),
    "torch": lambda: TorchBackend("cpu"),
    "torch-mps": lambda: TorchBackend("mps"),
    "mlx": lambda: MlxBackend(tf32=False),
    "mlx-tf32": lambda: MlxBackend(tf32=True),
}


def make_backend(name):
    if name not in BACKENDS:
        raise PlanError(f"unknown backend '{name}' (one of "
                        f"{', '.join(BACKENDS)})")
    return BACKENDS[name]()  # ImportError = unavailable; the caller decides


def require_numpy():
    try:
        import numpy
        return numpy
    except ImportError:
        raise PlanError("this command needs NumPy (tool/requirements.txt)")


# --- Memory: where a tensor reference lives -----------------------------------

_DTYPE_BYTES = {"<f4": 4, "<i4": 4, "<i1": 1, "<u2": 2}


def _count(shape):
    n = 1
    for d in shape:
        n *= d
    return n


class Memory:
    """The two address spaces of a plan. `recorder`, when set, receives
    (arena_offset, f32 ndarray) for every arena write — the diff trace."""

    def __init__(self, plan, backend):
        self.plan, self.x, self.np = plan, backend, backend.np
        self.arena = plan.initial_arena()
        self.recorder = None
        self._frozen = {}

    # Frozen weights: f32 as stored, int8 and bf16 widened once. A frontier
    # framework holds its weights in a dtype it can multiply; the on-the-fly
    # dequantization is the C++ runtime's memory trade, not part of the
    # arithmetic being checked or priced.
    def frozen(self, ref, shape, kind="<f4", scale=1.0):
        key = (ref, shape, kind, scale)
        if key not in self._frozen:
            np, off = self.np, self.plan.rodata_offset + (ref & ~RODATA_BIT)
            n = _count(shape)
            if off + n * _DTYPE_BYTES[kind] > len(self.plan.blob):
                raise PlanError("a rodata operand lies outside the plan")
            raw = np.frombuffer(self.plan.blob, kind, n, off)
            if kind == "<i1":
                host = raw.astype(np.float32) * np.float32(scale)
            elif kind == "<u2":  # bfloat16: f32's top 16 bits, exactly
                host = (raw.astype(np.uint32) << 16).view(np.float32)
            else:
                host = raw
            self._frozen[key] = self.x.up(
                self.x.from_numpy(host.reshape(shape)))
        return self._frozen[key]

    def _check(self, ref, nbytes):
        if ref == NULL_REF or ref & RODATA_BIT:
            raise PlanError(f"operand {ref:#x} is not an arena reference")
        if ref + nbytes > len(self.arena):
            raise PlanError("an arena operand lies outside the arena")

    def read(self, ref, shape, kind="<f4"):
        if ref != NULL_REF and ref & RODATA_BIT:
            return self.frozen(ref, tuple(shape), kind)
        return self._read(ref, tuple(shape), kind)

    def write(self, ref, tensor):
        raise NotImplementedError

    def stage(self, ref, payload):
        raise NotImplementedError

    def loss(self):
        return self.x.scalar(self.read(self.plan.loss_ref, (1,)))


class ArenaMemory(Memory):
    """Tensors are zero-copy views of the arena bytes — the C++ runtime's
    own memory model (NumPy, PyTorch CPU)."""

    def _read(self, ref, shape, kind):
        self._check(ref, _count(shape) * _DTYPE_BYTES[kind])
        t = self.x.view(self.arena, kind, _count(shape), ref).reshape(shape)
        return self.x.up(t) if kind == "<f4" else t

    def write(self, ref, tensor):
        n = _count(tensor.shape)
        self._check(ref, 4 * n)
        self.x.store(self.x.view(self.arena, "<f4", n, ref), tensor)
        if self.recorder:
            self.recorder(ref, self.np.frombuffer(self.arena, "<f4", n,
                                                  ref).copy())

    def stage(self, ref, payload):
        self.arena[ref:ref + len(payload)] = payload

    def sync(self):
        pass

    def export(self):
        return bytes(self.arena)

    def adopt(self, image):
        self.arena[:] = image


class SlotMemory(Memory):
    """Tensors live on the device, keyed by arena offset (MLX, PyTorch MPS).
    The compiler reuses arena bytes across a step, so a write evicts every
    live tensor it overlaps; a read that is not an exact hit (the staged
    batch, the persistent image, a sub-range) is served from the host bytes
    after the live tensors overlapping it have been flushed there."""

    def __init__(self, plan, backend):
        super().__init__(plan, backend)
        self.live, self.starts = {}, []  # off -> (nbytes, kind, tensor)

    def _overlapping(self, off, nbytes):
        hits, i = [], bisect.bisect_left(self.starts, off)
        if i and self.starts[i - 1] + self.live[self.starts[i - 1]][0] > off:
            hits.append(self.starts[i - 1])
        while i < len(self.starts) and self.starts[i] < off + nbytes:
            hits.append(self.starts[i])
            i += 1
        return hits

    def _flush(self, off):
        nbytes, kind, t = self.live[off]
        host = self.x.to_numpy(t).astype(kind, copy=False)
        self.arena[off:off + nbytes] = host.tobytes()

    def _evict(self, off, nbytes, flush):
        for start in self._overlapping(off, nbytes):
            size = self.live[start][0]
            if flush or start < off or start + size > off + nbytes:
                self._flush(start)  # bytes survive outside the new range
            del self.live[start]
            self.starts.pop(bisect.bisect_left(self.starts, start))

    def _insert(self, off, nbytes, kind, t):
        self.live[off] = (nbytes, kind, t)
        bisect.insort(self.starts, off)

    def _read(self, ref, shape, kind):
        nbytes = _count(shape) * _DTYPE_BYTES[kind]
        hit = self.live.get(ref)
        if hit and hit[0] == nbytes and hit[1] == kind:
            return hit[2].reshape(shape)
        self._check(ref, nbytes)
        self._evict(ref, nbytes, flush=True)
        host = self.np.frombuffer(self.arena, kind, _count(shape), ref)
        t = self.x.from_numpy(host)
        t = self.x.up(t) if kind == "<f4" else t
        self._insert(ref, nbytes, kind, t)
        return t.reshape(shape)

    def write(self, ref, tensor):
        nbytes = 4 * _count(tensor.shape)
        self._check(ref, nbytes)
        self._evict(ref, nbytes, flush=False)
        self._insert(ref, nbytes, "<f4", tensor.reshape(-1))
        if self.recorder:
            self.recorder(ref, self.x.to_numpy(tensor).astype(
                "<f4").reshape(-1).copy())

    def stage(self, ref, payload):
        self._evict(ref, len(payload), flush=False)
        self.arena[ref:ref + len(payload)] = payload

    def sync(self):
        self.x.sync([entry[2] for entry in self.live.values()])

    def export(self):
        for off in list(self.live):
            self._flush(off)
        return bytes(self.arena)

    def adopt(self, image):
        self.live, self.starts = {}, []
        self.arena[:] = image


# --- The instruction set --------------------------------------------------------
# Written from the definitions in source/plan/instruction.h and the kernel
# contracts in runtime/executor/update_kernels.h — the mathematics, not the
# loops: whole-tensor expressions, no tiling, no chunked f64 partial sums.
# Where a constant is the runtime's choice rather than mathematics (the norm
# epsilon, the GELU approximation, the KL clamp) it is named above.


def _act(x, kind, t):
    if kind == 1:
        return x.maximum(t, 0.0)
    if kind == 2:
        return 0.5 * t * (1.0 + x.tanh(GELU_C * (t + GELU_A * t * t * t)))
    if kind == 3:
        return t / (1.0 + x.exp(-t))
    return t


def _sigmoid(x, t):
    return 1.0 / (1.0 + x.exp(-t))


def _softmax(x, t):
    e = x.exp(t - x.max(t, -1))
    return e / x.sum(e, -1, keepdims=True)


def _heads(x, t, b, s, h, d):
    """[B*S, H*d] rows with heads interleaved -> [B, H, S, d]."""
    return x.perm(t.reshape((b, s, h, d)), (0, 2, 1, 3))


def _rows(x, t, b, s, h, d):
    """[B, H, S, d] -> [B*S, H*d]."""
    return x.perm(t, (0, 2, 1, 3)).reshape((b * s, h * d))


def _rope(x, t, b, s, h, d, base, sign):
    # angle(s, c) = s * base^(-2c/d) over the interleaved pair (2c, 2c+1).
    freq = x.exp(x.positions(d // 2) * (-2.0 / d * math.log(base)))
    theta = x.positions(s).reshape((s, 1, 1)) * freq.reshape((1, 1, d // 2))
    cos, sin = x.cos(theta), sign * x.sin(theta)
    pairs = t.reshape((b, s, h, d // 2, 2))
    even, odd = pairs[..., 0], pairs[..., 1]
    return x.stack_last(even * cos - odd * sin,
                        even * sin + odd * cos).reshape((b * s, h * d))


def _gemm_operands(m, ins, kind, transposed_b):
    rows, cols, inner = ins.out
    a = m.read(ins.src[0], (rows, inner))
    shape = (cols, inner) if transposed_b else (inner, cols)
    if kind == "<f4":
        b = m.read(ins.src[1], shape)
    else:
        scale = bits_f32(ins.src[3]) if kind == "<i1" else 1.0
        b = m.frozen(ins.src[1], shape, kind, scale)
    return a, (b.T if transposed_b else b)


def _gemm_nn(kind):
    def op(m, x, ins, step):
        a, b = _gemm_operands(m, ins, kind, False)
        c = x.matmul(a, b)
        if ins.flags & 1:  # the fused epilogue: C = act(A@B + bias)
            c = c + m.read(ins.src[3], (ins.out[1],))
        m.write(ins.src[2], _act(x, (ins.flags >> 1) & 3, c))
    return op


def _gemm_nt(kind):
    def op(m, x, ins, step):
        a, b = _gemm_operands(m, ins, kind, True)
        m.write(ins.src[2], x.matmul(a, b))
    return op


def _gemm_tn(m, x, ins, step):
    rows, cols, inner = ins.out
    a, b = m.read(ins.src[0], (inner, rows)), m.read(ins.src[1], (inner, cols))
    m.write(ins.src[2], x.matmul(a.T, b))


def _gemm_acc(m, x, ins, step):
    rows, cols, inner = ins.out
    a, b = m.read(ins.src[0], (rows, inner)), m.read(ins.src[1], (inner, cols))
    c = m.read(ins.src[2], (rows, cols))
    m.write(ins.src[2], c + bits_f32(ins.src[3]) * x.matmul(a, b))


def _binary(fn):
    def op(m, x, ins, step):
        n = (ins.out[0],)
        m.write(ins.src[2], fn(x, m.read(ins.src[0], n), m.read(ins.src[1], n)))
    return op


def _unary(fn):
    def op(m, x, ins, step):
        m.write(ins.src[1], fn(x, m.read(ins.src[0], (ins.out[0],))))
    return op


def _gelu_bwd(x, dy, v):
    u = GELU_C * (v + GELU_A * v * v * v)
    t = x.tanh(u)
    sech2 = 1.0 - t * t
    # Where tanh has saturated the sech^2 term is a true zero; the polynomial
    # beside it may have overflowed, and 0 * inf must not become NaN.
    tail = x.where(sech2 == 0.0, 0.0 * v, 0.5 * v * sech2 * GELU_C *
                   (1.0 + 3.0 * GELU_A * v * v))
    return dy * (0.5 * (1.0 + t) + tail)


def _silu_bwd(x, dy, v):
    s = _sigmoid(x, v)
    return dy * (s * (1.0 + v * (1.0 - s)))


def _add_bias(m, x, ins, step):
    rows, cols = ins.out[0], ins.out[1]
    m.write(ins.src[2], m.read(ins.src[0], (rows, cols)) +
            m.read(ins.src[1], (cols,)))


def _scale(m, x, ins, step):
    m.write(ins.src[1], bits_f32(ins.src[2]) * m.read(ins.src[0],
                                                      (ins.out[0],)))


def _reduce_rows(m, x, ins, step):
    m.write(ins.src[1], x.sum(m.read(ins.src[0], (ins.out[0], ins.out[1])), 0))


def _xent_fwd(m, x, ins, step):
    n, c = ins.out[0], ins.out[1]
    probs = _softmax(x, m.read(ins.src[0], (n, c)))
    hot = x.one_hot(m.read(ins.src[1], (n,), "<i4"), c)
    picked = x.maximum(x.sum(probs * hot, -1), 1e-12)  # the underflow clamp
    m.write(ins.src[3], probs)
    m.write(ins.src[2], (-x.sum(x.log(picked), 0) / n).reshape((1,)))


def _xent_bwd(m, x, ins, step):
    n, c = ins.out[0], ins.out[1]
    hot = x.one_hot(m.read(ins.src[1], (n,), "<i4"), c)
    seed = m.read(ins.src[2], (1,))
    m.write(ins.src[3], seed * (m.read(ins.src[0], (n, c)) - hot) / n)


def _mse_fwd(m, x, ins, step):
    n = (ins.out[0],)
    d = m.read(ins.src[0], n) - m.read(ins.src[1], n)
    m.write(ins.src[2], (x.sum(d * d, 0) / ins.out[0]).reshape((1,)))


def _mse_bwd(m, x, ins, step):
    n = (ins.out[0],)
    d = m.read(ins.src[0], n) - m.read(ins.src[1], n)
    m.write(ins.src[3], m.read(ins.src[2], (1,)) * (2.0 / ins.out[0]) * d)


def _kl_words(word):
    hi, lo = hi_lo(word)
    return bits_f32(lo), (bits_f32(hi) if hi else 1.0)  # pre-v8: no scale


def _kl_fwd(m, x, ins, step):
    n, c = hi_lo(ins.out[1])
    temperature, scale = _kl_words(ins.out[2])
    p_s = _softmax(x, m.read(ins.src[0], (n, c)) / temperature)
    p_t = _softmax(x, m.read(ins.src[1], (n, c)) / temperature)
    terms = x.where(p_t > 0.0, p_t * (x.log(x.maximum(p_t, 1e-30)) -
                                      x.log(x.maximum(p_s, 1e-12))), 0.0 * p_t)
    m.write(ins.src[3], p_s)
    m.write(ins.out[0], p_t)
    m.write(ins.src[2], (scale * x.sum(x.sum(terms, -1), 0) / n).reshape((1,)))


def _kl_bwd(m, x, ins, step):
    n, c = hi_lo(ins.out[0])
    temperature, scale = _kl_words(ins.out[1])
    seed = m.read(ins.src[2], (1,))
    m.write(ins.src[3], seed * (scale / (n * temperature)) *
            (m.read(ins.src[0], (n, c)) - m.read(ins.src[1], (n, c))))


def _sgd(m, x, ins, step):
    n = (ins.out[0],)
    p, g = m.read(ins.src[0], n), m.read(ins.src[1], n)
    m.write(ins.src[0], p - step["lr"] * (g + step["weight_decay"] * p))


def _adamw(m, x, ins, step):
    n = (ins.out[0],)
    p, g = m.read(ins.src[0], n), m.read(ins.src[1], n)
    b1, b2, t = step["beta1"], step["beta2"], step["step"]
    mom = b1 * m.read(ins.src[2], n) + (1.0 - b1) * g
    var = b2 * m.read(ins.src[3], n) + (1.0 - b2) * g * g
    m_hat, v_hat = mom / (1.0 - b1 ** t), var / (1.0 - b2 ** t)
    m.write(ins.src[2], mom)
    m.write(ins.src[3], var)
    m.write(ins.src[0], p - step["lr"] * (
        m_hat / (x.sqrt(v_hat) + step["eps"]) + step["weight_decay"] * p))


def _fill(m, x, ins, step):
    m.write(ins.src[0], x.full(ins.out[0], bits_f32(ins.src[1])))


def _copy(m, x, ins, step):
    m.write(ins.src[1], m.read(ins.src[0], (ins.out[0],)) * 1.0)


def _accumulate(m, x, ins, step):
    n = (ins.out[0],)
    m.write(ins.src[0], m.read(ins.src[0], n) + m.read(ins.src[1], n))


def _clip(m, x, ins, step):
    g = m.read(ins.src[0], (ins.out[0],))
    limit = bits_f32(ins.src[1])
    norm = x.sqrt(x.sum(g * g, 0))
    # g *= min(1, max/||g||), spelled without a host round-trip: the factor
    # is exactly 1 where the norm is already inside the ball (or is zero).
    factor = x.where(norm > limit, limit / x.maximum(norm, 1e-30),
                     1.0 + 0.0 * norm)
    m.write(ins.src[0], g * factor)


def _layer_norm_fwd(m, x, ins, step):
    n, d = hi_lo(ins.out[0])
    t = m.read(ins.src[0], (n, d))
    mean = x.mean(t, -1, keepdims=True)
    centered = t - mean
    rstd = 1.0 / x.sqrt(x.mean(centered * centered, -1, keepdims=True) +
                        NORM_EPS)
    m.write(ins.src[3], centered * rstd * m.read(ins.src[1], (d,)) +
            m.read(ins.src[2], (d,)))
    m.write(ins.out[1], mean.reshape((n,)))
    m.write(ins.out[2], rstd.reshape((n,)))


def _layer_norm_bwd(m, x, ins, step):
    n, d = hi_lo(ins.out[2])
    mean = m.read(ins.out[0], (n,)).reshape((n, 1))
    rstd = m.read(ins.out[1], (n,)).reshape((n, 1))
    xhat = (m.read(ins.src[1], (n, d)) - mean) * rstd
    g = m.read(ins.src[0], (n, d)) * m.read(ins.src[2], (d,))
    m.write(ins.src[3], rstd * (g - x.mean(g, -1, keepdims=True) -
                                xhat * x.mean(g * xhat, -1, keepdims=True)))


def _rms_norm_fwd(m, x, ins, step):
    n, d = hi_lo(ins.out[0])
    t = m.read(ins.src[0], (n, d))
    rstd = 1.0 / x.sqrt(x.mean(t * t, -1, keepdims=True) + NORM_EPS)
    m.write(ins.src[2], t * rstd * m.read(ins.src[1], (d,)))
    m.write(ins.src[3], rstd.reshape((n,)))


def _rms_norm_bwd(m, x, ins, step):
    n, d = hi_lo(ins.out[1])
    t = m.read(ins.src[1], (n, d))
    rstd = m.read(ins.out[0], (n,)).reshape((n, 1))
    g = m.read(ins.src[0], (n, d)) * m.read(ins.src[2], (d,))
    m.write(ins.src[3], rstd * g - t * (rstd * rstd * rstd) *
            x.mean(g * t, -1, keepdims=True))


def _rope_op(sign):
    def op(m, x, ins, step):
        (b, s), (h, d) = hi_lo(ins.out[0]), hi_lo(ins.out[1])
        t = m.read(ins.src[0], (b * s, h * d))
        m.write(ins.src[1], _rope(x, t, b, s, h, d, bits_f32(ins.out[2]),
                                  sign))
    return op


def _attn_fwd(m, x, ins, step):
    (b, s), (h, d) = hi_lo(ins.out[1]), hi_lo(ins.out[2])
    q, k, v = (_heads(x, m.read(ref, (b * s, h * d)), b, s, h, d)
               for ref in ins.src[:3])
    mask = x.causal(s)
    scores = x.matmul(q, x.perm(k, (0, 1, 3, 2))) / math.sqrt(d)
    scores = x.where(mask > 0.0, scores, scores * 0.0 - 1e30)
    probs = _softmax(x, scores) * mask  # masked entries are exactly zero
    m.write(ins.out[0], probs.reshape((b * h * s, s)))
    m.write(ins.src[3], _rows(x, x.matmul(probs, v), b, s, h, d))


def _attn_dp(m, x, ins, step):
    (b, s), (h, d) = hi_lo(ins.out[0]), hi_lo(ins.out[1])
    dout = _heads(x, m.read(ins.src[0], (b * s, h * d)), b, s, h, d)
    v = _heads(x, m.read(ins.src[1], (b * s, h * d)), b, s, h, d)
    # The masked half is a don't-care (P is zero there, so dS is too); the
    # CPU backend writes zeros and the GPU finite values. Zeros here.
    dp = x.matmul(dout, x.perm(v, (0, 1, 3, 2))) * x.causal(s)
    m.write(ins.src[2], dp.reshape((b * h * s, s)))


def _attn_dv(m, x, ins, step):
    (b, s), (h, d) = hi_lo(ins.out[0]), hi_lo(ins.out[1])
    probs = m.read(ins.src[0], (b * h * s, s)).reshape((b, h, s, s))
    dout = _heads(x, m.read(ins.src[1], (b * s, h * d)), b, s, h, d)
    m.write(ins.src[2], _rows(x, x.matmul(x.perm(probs, (0, 1, 3, 2)), dout),
                              b, s, h, d))


def _softmax_rows_bwd(m, x, ins, step):
    shape = hi_lo(ins.out[0])
    p, dp = m.read(ins.src[0], shape), m.read(ins.src[1], shape)
    m.write(ins.src[2], p * (dp - x.sum(dp * p, -1, keepdims=True)))


def _attn_dqk(transpose):
    def op(m, x, ins, step):
        (b, s), (h, d) = hi_lo(ins.out[0]), hi_lo(ins.out[1])
        ds = m.read(ins.src[0], (b * h * s, s)).reshape((b, h, s, s))
        if transpose:
            ds = x.perm(ds, (0, 1, 3, 2))
        other = _heads(x, m.read(ins.src[1], (b * s, h * d)), b, s, h, d)
        m.write(ins.src[2], _rows(x, x.matmul(ds, other) / math.sqrt(d),
                                  b, s, h, d))
    return op


def _embed(m, x, ins, step):
    rows = ins.out[0]
    vocab, dim = hi_lo(ins.out[1])
    table = m.read(ins.src[1], (vocab, dim))
    m.write(ins.src[2], x.take_rows(table, m.read(ins.src[0], (rows,), "<i4")))


def _nop(m, x, ins, step):
    pass


INTERPRETER = {
    0: _nop, 1: _gemm_nn("<f4"), 2: _gemm_nt("<f4"), 3: _gemm_tn,
    4: _gemm_acc, 5: _binary(lambda x, a, b: a + b), 6: _add_bias,
    7: _unary(lambda x, t: x.maximum(t, 0.0)),
    8: _binary(lambda x, dy, v: x.where(v > 0.0, dy, 0.0 * dy)), 9: _scale,
    10: _reduce_rows, 11: _xent_fwd, 12: _xent_bwd, 13: _mse_fwd,
    14: _mse_bwd, 15: _kl_fwd, 16: _kl_bwd, 17: _sgd, 18: _adamw, 19: _fill,
    20: _copy, 21: _binary(lambda x, a, b: a * b),
    22: _unary(lambda x, t: _act(x, 2, t)), 23: _binary(_gelu_bwd),
    24: _unary(lambda x, t: _act(x, 3, t)), 25: _binary(_silu_bwd),
    26: _layer_norm_fwd, 27: _layer_norm_bwd, 28: _clip,
    29: _gemm_nn("<i1"), 30: _gemm_nt("<i1"), 31: _rms_norm_fwd,
    32: _rms_norm_bwd, 33: _rope_op(1.0), 34: _rope_op(-1.0), 35: _attn_fwd,
    36: _attn_dp, 37: _attn_dv, 38: _softmax_rows_bwd, 39: _attn_dqk(False),
    40: _attn_dqk(True), 41: _embed, 42: _accumulate,
    43: _gemm_nn("<u2"), 44: _gemm_nt("<u2"),
}
assert sorted(INTERPRETER) == sorted(OPCODES)


class Executor:
    """One plan bound to one backend: the engine's loop, minus the gate."""

    def __init__(self, plan, backend):
        self.plan, self.x = plan, backend
        mem = ArenaMemory if backend.arena_views else SlotMemory
        self.mem = mem(plan, backend)
        self.step = 0
        unknown = sorted({i.opcode for s in plan.sections.values() for i in s}
                         - set(INTERPRETER))
        if unknown:
            raise PlanError(f"the plan uses opcodes this interpreter does "
                            f"not know: {unknown}")

    def step_scalars(self, step=None):
        p, step = self.plan, self.step if step is None else step
        return {"lr": p.effective_lr(step), "beta1": f32(p.beta1),
                "beta2": f32(p.beta2), "eps": f32(p.eps),
                "weight_decay": f32(p.weight_decay), "step": max(step, 1)}

    def stage(self, batch):
        inputs, labels = batch
        p = self.plan
        if len(inputs) != 4 * p.input_floats:
            raise PlanError("a staged batch does not match input_floats")
        self.mem.stage(p.input_ref, inputs)
        if p.label_kind:
            if len(labels) != p.label_bytes:
                raise PlanError("a staged batch does not match label_bytes")
            self.mem.stage(p.label_ref, labels)

    def execute(self, section, scalars=None, on_instruction=None):
        scalars = scalars or self.step_scalars()
        for index, ins in enumerate(self.plan.sections[section]):
            INTERPRETER[ins.opcode](self.mem, self.x, ins, scalars)
            if on_instruction:
                on_instruction(index, ins)

    def train_step(self, corpus):
        """One optimizer step: G grad executions, then the step program.
        Returns the mean micro-batch loss, as the engine reports it."""
        self.step += 1
        scalars, total = self.step_scalars(), 0.0
        for _ in range(self.plan.grad_accum):
            self.stage(corpus.batch(self.plan))
            self.execute("train", scalars)
            total += self.mem.loss()
        self.execute("step", scalars)
        self.mem.sync()
        return total / self.plan.grad_accum

    def evaluate(self, corpus):
        """Mean eval-program loss over one pass in compiled-batch chunks
        (the final partial batch wraps, as in the engine)."""
        p = self.plan
        per_step = p.batch // (p.seq_len if p.input_kind == 1 else 1)
        batches = max(1, -(-corpus.num_samples // per_step))
        saved = (corpus.cursor, corpus.order, corpus.state)
        corpus.cursor, total = 0, 0.0
        for _ in range(batches):
            self.stage(corpus.batch(p))
            self.execute("eval")
            total += self.mem.loss()
        corpus.cursor, corpus.order, corpus.state = saved
        return total / batches


# --- diff: the oracle ------------------------------------------------------------

TRACE_MAGIC = 0x54504553


def read_trace(np, path):
    """[(index, opcode, [(offset, f32 ndarray)])] from a probe trace."""
    with open(path, "rb") as f:
        raw = f.read()
    magic, version, count = struct.unpack_from("<IIQ", raw, 0)
    if magic != TRACE_MAGIC or version != 1:
        raise PlanError(f"{path}: not a seeml-plan-probe trace")
    pos, out = 16, []
    for _ in range(count):
        index, opcode, extents = struct.unpack_from("<IHH", raw, pos)
        pos += 8
        writes = []
        for _ in range(extents):
            off, nbytes = struct.unpack_from("<QQ", raw, pos)
            pos += 16
            writes.append((off, np.frombuffer(raw, "<f4", nbytes // 4, pos)))
            pos += nbytes
        out.append((index, opcode, writes))
    return out


class Probe:
    """seeml-plan-probe, the C++ side of the comparison."""

    def __init__(self, path, plan_path, backend="cpu", threads=None):
        if not (path and os.path.isfile(path) and os.access(path, os.X_OK)):
            raise PlanError(f"no seeml-plan-probe at {path!r} — build the "
                            "tree (sh build/build.sh) or pass --probe")
        self.base = [path, "--plan", plan_path, "--backend", backend]
        if threads:
            self.base += ["--threads", str(threads)]
        self.tmp = tempfile.TemporaryDirectory(prefix="seeml-frontier-")

    def close(self):
        self.tmp.cleanup()

    def _call(self, section, arena, scalars, extra):
        a_in = os.path.join(self.tmp.name, "arena.in")
        a_out = os.path.join(self.tmp.name, "arena.out")
        with open(a_in, "wb") as f:
            f.write(arena)
        cmd = self.base + ["--section", section, "--arena-in", a_in,
                           "--arena-out", a_out,
                           "--lr-bits", f"{f32_bits(scalars['lr']):08x}",
                           "--step", str(scalars["step"])] + extra
        done = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        if done.returncode != 0:
            raise PlanError(f"seeml-plan-probe failed ({done.returncode}): "
                            f"{done.stderr.strip()}")
        with open(a_out, "rb") as f:
            return f.read(), done.stdout

    def trace(self, np, section, arena, scalars):
        path = os.path.join(self.tmp.name, "trace.bin")
        image, _ = self._call(section, arena, scalars, ["--trace", path])
        return image, read_trace(np, path)

    def time(self, section, arena, scalars, executions):
        _, stdout = self._call(section, arena, scalars,
                               ["--time", str(executions)])
        return json.loads(stdout)


class Differ:
    def __init__(self, np, rtol, atol):
        self.np, self.rtol, self.atol = np, rtol, atol
        self.per_opcode, self.failures, self.compared = {}, [], 0

    def _mask(self, ins, values):
        """kAttnDP's strictly-upper triangle is unspecified (see _attn_dp)."""
        if ins.opcode != 36:
            return values
        (b, s), (h, _) = hi_lo(ins.out[0]), hi_lo(ins.out[1])
        return (values.reshape(b * h, s, s) *
                self.np.tril(self.np.ones((s, s), "<f4"))).reshape(-1)

    def section(self, label, plan, section, ours, theirs):
        np = self.np
        if len(ours) != len(theirs):
            self.failures.append(f"{label}: {len(ours)} instructions here, "
                                 f"{len(theirs)} in the C++ trace")
            return
        for (index, ins, mine), (_, opcode, cpp) in zip(ours, theirs):
            name = OPCODES.get(opcode, str(opcode))
            mine_by_off = dict(mine)
            if sorted(mine_by_off) != sorted(off for off, _ in cpp):
                self.failures.append(
                    f"{label} #{index} {name}: writes {sorted(mine_by_off)} "
                    f"here, {sorted(off for off, _ in cpp)} in C++")
                continue
            for off, ref in cpp:
                got = mine_by_off[off]
                if got.shape != ref.shape:
                    self.failures.append(f"{label} #{index} {name} @{off}: "
                                         f"{got.size} floats vs {ref.size}")
                    continue
                got, ref = self._mask(ins, got), self._mask(ins, ref)
                self.compared += 1
                both_nan = np.isnan(got) & np.isnan(ref)
                delta = np.where(both_nan, 0.0, np.abs(
                    got.astype(np.float64) - ref.astype(np.float64)))
                finite = ref[np.isfinite(ref)]
                scale = float(np.abs(finite).max()) if finite.size else 0.0
                bound = self.atol + self.rtol * scale
                worst = float(np.nanmax(delta)) if delta.size else 0.0
                if np.isnan(delta).any():
                    worst = float("inf")
                stat = self.per_opcode.setdefault(
                    name, {"extents": 0, "worst_ratio": 0.0})
                stat["extents"] += 1
                stat["worst_ratio"] = max(stat["worst_ratio"], worst / bound)
                if not worst <= bound:
                    at = int(np.nanargmax(delta)) if worst != float(
                        "inf") else -1
                    self.failures.append(
                        f"{label} #{index} {name} @{off}: max |delta| "
                        f"{worst:.3e} > {bound:.3e} (element {at}: here "
                        f"{got[at]!r}, C++ {ref[at]!r})")


def cmd_diff(args):
    np = require_numpy()
    plan = Plan.load(args.plan)
    try:
        backend = make_backend(args.backend)
    except ImportError as e:
        raise PlanError(f"backend '{args.backend}' is unavailable: {e}")
    corpus = (Corpus(np, args.corpus) if args.corpus
              else SyntheticCorpus(np, plan, args.seed))
    corpus.check(plan)
    ex = Executor(plan, backend)
    probe = Probe(args.probe, args.plan, args.cpp_backend, args.threads)
    differ = Differ(np, args.rtol, args.atol)

    def both(label, section, scalars):
        """The C++ side runs `section` first; this side replays it one
        instruction at a time from the C++ bytes (see the module docstring),
        so both arrive at the C++ arena for whatever runs next."""
        image, theirs = probe.trace(np, section, ex.mem.export(), scalars)
        if len(theirs) != len(plan.sections[section]):
            differ.failures.append(f"{label}: the C++ trace holds "
                                   f"{len(theirs)} instructions")
            ex.mem.adopt(image)
            return
        ours, writes = [], []
        ex.mem.recorder = lambda off, values: writes.append((off, values))

        def done(index, ins):
            ours.append((index, ins, list(writes)))
            del writes[:]
            for off, values in theirs[index][2]:
                ex.mem.stage(off, values.tobytes())
        ex.execute(section, scalars, done)
        ex.mem.recorder = None
        differ.section(label, plan, section, ours, theirs)
        ex.mem.adopt(image)

    try:
        for step in range(1, args.steps + 1):
            scalars = ex.step_scalars(step)
            for micro in range(plan.grad_accum):
                ex.stage(corpus.batch(plan))
                tag = f"step {step}" + (f".{micro}" if plan.grad_accum > 1
                                        else "")
                both(f"{tag} train", "train", scalars)
            if plan.sections["step"]:
                both(f"step {step} step", "step", scalars)
        if plan.sections["eval"]:
            ex.stage(corpus.batch(plan))
            both("eval", "eval", ex.step_scalars(args.steps))
        if plan.sections["merge"]:
            both("merge", "merge", ex.step_scalars(args.steps))
    finally:
        probe.close()

    used = {OPCODES[i.opcode] for s in plan.sections.values() for i in s}
    report = {
        "plan": args.plan, "plan_version": plan.version,
        "frontier_backend": backend.name, "frontier_device": backend.device,
        "cpp_backend": args.cpp_backend, "steps": args.steps,
        "rtol": args.rtol, "atol": args.atol,
        "extents_compared": differ.compared,
        "opcodes": {k: differ.per_opcode.get(k, {"extents": 0,
                                                 "worst_ratio": 0.0})
                    for k in sorted(used)},
        "failures": differ.failures, "agrees": not differ.failures,
    }
    for name, stat in report["opcodes"].items():
        print(f"frontier_exec: {name:<18} {stat['extents']:>5} extent(s), "
              f"worst |delta|/bound {stat['worst_ratio']:.3g}")
    for line in differ.failures[:40]:
        print(f"frontier_exec: DISAGREE {line}")
    verdict = ("agrees with" if report["agrees"] else
               f"DISAGREES ({len(differ.failures)}) with")
    print(f"frontier_exec: {backend.name} {verdict} the C++ {args.cpp_backend}"
          f" backend on {differ.compared} extent(s) over {args.steps} step(s)")
    write_report(args.report, report)
    return 0 if report["agrees"] else 1


# --- run: a whole update, fed as the device feeds it ----------------------------


def cmd_run(args):
    np = require_numpy()
    plan = Plan.load(args.plan)
    try:
        backend = make_backend(args.backend)
    except ImportError as e:
        raise PlanError(f"backend '{args.backend}' is unavailable: {e}")
    corpus = Corpus(np, args.corpus)
    corpus.check(plan)
    val = None
    if args.val_fraction > 0.0:
        corpus, val = corpus.split_validation(args.val_fraction)
    if not args.no_shuffle:
        corpus.enable_shuffle(args.seed)
    ex = Executor(plan, backend)
    steps = args.steps or plan.default_steps
    report = {"plan": args.plan, "backend": backend.name,
              "device": backend.device, "steps": steps, "loss_curve": []}
    if val is not None and plan.sections["eval"]:
        report["val_initial_loss"] = ex.evaluate(val)
        print(f"frontier_exec: val loss {report['val_initial_loss']:.6f} "
              "(before)")
    for step in range(1, steps + 1):
        loss = ex.train_step(corpus)
        report["loss_curve"].append(loss)
        if not math.isfinite(loss):
            raise PlanError(f"loss became non-finite at step {step}")
        if args.log_every and (step - 1) % args.log_every == 0:
            print(f"frontier_exec: step {step}  loss {loss:.6f}")
    if val is not None and plan.sections["eval"]:
        report["val_final_loss"] = ex.evaluate(val)
        print(f"frontier_exec: val loss {report['val_final_loss']:.6f} "
              "(after)")
    write_report(args.report, report)
    return 0


# --- price: the ceiling --------------------------------------------------------


def median(values):
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def peak_rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def price_backend(np, plan, name, corpus_factory, lo, hi, repeats):
    """seeml-bench's steps-regression: time lo and hi steps, the slope is
    the per-step cost; medians over `repeats`. One warm-up step first (lazy
    kernels, the frozen-weight upload) — a fresh executor per timed run
    would re-pay it."""
    backend = make_backend(name)  # ImportError propagates: "unavailable"
    ex = Executor(plan, backend)
    corpus = corpus_factory()
    ex.train_step(corpus)
    slopes = []
    for _ in range(repeats):
        walls = []
        for steps in (lo, hi):
            ex.mem.sync()
            backend.sync()
            t0 = time.perf_counter()
            for _ in range(steps):
                loss = ex.train_step(corpus)
            ex.mem.sync()
            backend.sync()
            walls.append(time.perf_counter() - t0)
        slopes.append(1000.0 * (walls[1] - walls[0]) / (hi - lo))
    return backend, slopes, loss


def cell(plan, step_ms, slopes, peak_gflops):
    unit = "tokens_per_s" if plan.input_kind == 1 else "samples_per_s"
    rows = 1000.0 * plan.batch * plan.grad_accum / step_ms if step_ms else 0.0
    gflops = (plan.gemm_flops() * plan.grad_accum / (step_ms * 1e6)
              if step_ms else 0.0)
    out = {"step_ms": round(step_ms, 3), "rows_per_s": round(rows),
           unit: round(rows), "it_per_s": round(1000.0 / step_ms, 3)
           if step_ms else 0.0, "gemm_gflops": round(gflops, 1),
           "step_ms_min": round(min(slopes), 3),
           "step_ms_max": round(max(slopes), 3)}
    if peak_gflops:
        out["mfu"] = round(gflops / peak_gflops, 4)
    return out


def cmd_price(args):
    np = require_numpy()
    plan = Plan.load(args.plan)
    if args.steps_hi <= args.steps_lo:
        raise PlanError("--steps-hi must exceed --steps-lo")

    def corpus_factory():
        if args.corpus:
            corpus = Corpus(np, args.corpus)
            corpus.check(plan)
            return corpus
        return SyntheticCorpus(np, plan, args.seed)

    frontier = {}
    for name in args.backends.split(","):
        try:
            backend, slopes, loss = price_backend(
                np, plan, name, corpus_factory, args.steps_lo, args.steps_hi,
                args.repeats)
        except ImportError as e:
            frontier[name] = {"available": False, "reason": str(e)}
            print(f"frontier_exec: {name:<10} unavailable ({e})")
            continue
        entry = cell(plan, median(slopes), slopes, args.peak_gflops)
        entry.update(available=True, device=backend.device,
                     peak_rss_bytes=peak_rss_bytes(), last_loss=loss)
        frontier[name] = entry
        print(f"frontier_exec: {name:<10} {entry['step_ms']:>10.3f} ms/step "
              f"{entry['rows_per_s']:>9} rows/s "
              f"{entry['gemm_gflops']:>8.1f} GEMM GFLOP/s")

    cpp = {}
    if args.probe:
        ex = Executor(plan, make_backend("numpy32"))
        ex.stage(corpus_factory().batch(plan))
        for cpp_backend in args.cpp_backends.split(","):
            probe = Probe(args.probe, args.plan, cpp_backend, args.threads)
            try:
                scalars, image = ex.step_scalars(1), ex.mem.export()
                ms = probe.time("train", image, scalars,
                                args.steps_hi)["ms_median"] * plan.grad_accum
                if plan.sections["step"]:
                    ms += probe.time("step", image, scalars,
                                     args.steps_hi)["ms_median"]
            except PlanError as e:
                cpp[cpp_backend] = {"available": False, "reason": str(e)}
                print(f"frontier_exec: c++/{cpp_backend:<6} unavailable ({e})")
                continue
            finally:
                probe.close()
            cpp[cpp_backend] = dict(cell(plan, ms, [ms], args.peak_gflops),
                                    available=True)
            print(f"frontier_exec: c++/{cpp_backend:<6} {ms:>10.3f} ms/step "
                  f"{cpp[cpp_backend]['rows_per_s']:>9} rows/s "
                  f"{cpp[cpp_backend]['gemm_gflops']:>8.1f} GEMM GFLOP/s")
    ref = cpp.get("cpu", {})
    if ref.get("available"):
        for name, entry in frontier.items():
            if entry.get("available") and entry["step_ms"] > 0:
                entry["speedup_vs_cpp_cpu"] = round(
                    ref["step_ms"] / entry["step_ms"], 3)
                print(f"frontier_exec: {name} would buy "
                      f"{entry['speedup_vs_cpp_cpu']:.2f}x over the C++ CPU "
                      "backend on this plan")
    write_report(args.report, {
        "schema": 1, "plan": args.plan, "plan_version": plan.version,
        "batch_rows": plan.batch, "grad_accum_steps": plan.grad_accum,
        "gemm_flops_per_step": plan.gemm_flops() * plan.grad_accum,
        "config": {"steps_lo": args.steps_lo, "steps_hi": args.steps_hi,
                   "repeats": args.repeats, "peak_gflops": args.peak_gflops,
                   "corpus": args.corpus or "synthetic"},
        "frontier": frontier, "cpp": cpp})
    return 0


# --- info ------------------------------------------------------------------------


def cmd_info(args):
    plan = Plan.load(args.plan)
    print(f"{args.plan}: SEEU v{plan.version}, arena {plan.arena_size} B "
          f"(persistent {plan.persistent_size} B), rodata {plan.rodata_size} "
          f"B, batch {plan.batch}"
          + (f" x {plan.grad_accum} micro-batches" if plan.grad_accum > 1
             else "")
          + (f", {plan.seq_len}-token records" if plan.input_kind else ""))
    print("sections: " + ", ".join(
        f"{name} {len(plan.sections[name])}" for name in SECTIONS))
    print(f"GEMM FLOPs per optimizer step: "
          f"{plan.gemm_flops() * plan.grad_accum:,}")
    for name, where in sorted(plan.histogram().items()):
        print(f"  {name:<18} " + "  ".join(f"{s}={n}"
                                           for s, n in where.items()))
    return 0


def write_report(path, report):
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def default_probe():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "build", "seeml-plan-probe")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="header, sections, opcode histogram")
    info.add_argument("plan")
    info.set_defaults(fn=cmd_info)

    def common(sp):
        sp.add_argument("plan")
        sp.add_argument("--corpus", metavar="SDS", default=None)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--threads", type=int, default=None,
                        help="SEEML thread count for the C++ probe")
        sp.add_argument("--report", metavar="JSON", default=None)

    diff = sub.add_parser("diff", help="compare every write against the "
                                       "C++ runtime")
    common(diff)
    diff.add_argument("--probe", default=default_probe())
    diff.add_argument("--steps", type=int, default=2)
    diff.add_argument("--backend", default="numpy")
    diff.add_argument("--cpp-backend", default="cpu",
                      choices=("cpu", "metal"))
    diff.add_argument("--rtol", type=float, default=1e-5)
    diff.add_argument("--atol", type=float, default=1e-7)
    diff.set_defaults(fn=cmd_diff)

    run = sub.add_parser("run", help="train a plan as the device would")
    common(run)
    run.add_argument("--steps", type=int, default=0,
                     help="optimizer steps (default: the plan's)")
    run.add_argument("--backend", default="numpy")
    run.add_argument("--val-fraction", type=float, default=0.1)
    run.add_argument("--no-shuffle", action="store_true")
    run.add_argument("--log-every", type=int, default=100)
    run.set_defaults(fn=cmd_run)

    price = sub.add_parser("price", help="what would AMX / the GPU buy?")
    common(price)
    price.add_argument("--backends", default="torch,mlx")
    price.add_argument("--probe", default=None,
                       help="seeml-plan-probe, for the C++ numbers")
    price.add_argument("--cpp-backends", default="cpu")
    price.add_argument("--steps-lo", type=int, default=5)
    price.add_argument("--steps-hi", type=int, default=20)
    price.add_argument("--repeats", type=int, default=3)
    price.add_argument("--peak-gflops", type=float, default=0.0)
    price.set_defaults(fn=cmd_price)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run" and not args.corpus:
        print("frontier_exec: ERROR run needs --corpus", file=sys.stderr)
        return 2
    try:
        return args.fn(args)
    except PlanError as e:
        print(f"frontier_exec: ERROR {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
