#!/usr/bin/env python3
"""Export a PyTorch MLP and a dataset into SeeML's SMF / SDS formats.

SMF is the model container consumed by seeml-update-compile (source and
teacher models); SDS is the fixed-shape dataset streamed by the compiled
`model_update` executable on-device.

Usage:
    python3 export_model.py --demo out_dir/
        Writes a demo model.smf, teacher.smf and corpus.sds for a smoke run.

    python3 export_model.py --demo-decoder out_dir/
        Writes a token-native demo decoder.smf (SMF v4) and its
        decoder_corpus.sds (SDS v2, i32 token ids). NumPy only.

    Every demo dimension, seed, and corpus size is a flag — defaults
    reproduce the classic demos byte-for-byte:
        --demo:          --width 32 --depth 1 --samples 2048 --seed 0
                         --corpus-kind class|dense|none  (none = unlabeled,
                         what --loss kl distillation wants)
        --demo-decoder:  --width 32 --heads 4 --seq-len 8 --blocks 2
                         --ffn 64 --vocab 50 --samples 192 --seed 0
    A flag that cannot apply to the requested mode is a hard error (exit 2),
    never silently ignored — the same discipline as seeml-update-compile.

    python3 export_model.py --corpus data.npz out/corpus.sds
        Converts arrays into an SDS corpus without writing Python. A .npz
        with `records` [N, S+1] integer ids -> token corpus (SDS v2); with
        `inputs` [N, D] f32 and optional `labels` (integer -> class, float
        -> dense, absent -> unlabeled) -> feature corpus (SDS v1). A bare
        .npy is treated as `records`.

    from export_model import export_smf, export_sds
        export_smf(torch_sequential, "model.smf")
        export_sds(inputs, labels, "corpus.sds")

    from export_model import export_token_decoder_smf, export_token_sds
        export_token_decoder_smf(embedding, blocks, head, "model.smf",
                                 seq_len=S, num_heads=H)
        export_token_sds(token_records, "corpus.sds")  # [N, S+1] i32

    python3 export_model.py --hf <model_dir> out.smf --seq-len S
        Imports a Llama-class Hugging Face checkpoint directory (llama,
        qwen2, SmolLM2: config.json + safetensors) as a token-native SMF
        decoder — NumPy only. Adds: --text-corpus text.txt out.sds
        (tokenizes with the checkpoint's tokenizer.json; needs the
        `tokenizers` package), --hf-parity (max |Δ logits| of the SeeML
        forward vs transformers; needs torch + transformers), and
        --allow-eps-drift (a checkpoint whose rms_norm_eps is not the
        runtime's 1e-5).

Supported modules inside an nn.Sequential:
    nn.Linear     -> MatMul(x, W[in,out]) + AddBias(b[out])  (W stored
                     transposed from PyTorch's [out, in] layout)
    nn.ReLU       -> Relu
    nn.GELU       -> Gelu (tanh approximation on-device)
    nn.SiLU       -> Silu
    nn.LayerNorm  -> LayerNorm(x, gamma, beta) over the last dim
"""

import argparse
import os
import json
import io
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seeml import formats  # noqa: E402

SMF_MAGIC = formats.SMF_MAGIC
SMF_VERSION = formats.SMF_VERSION
DEFAULT_ROPE_BASE = formats.SMF_DEFAULT_ROPE_BASE  # attr1 == 0 on a Rope op
# The writer emits the LOWEST version that can carry the model: v3 for a
# feature-input model, v4 for a token-native one, v5 only when some Rope op
# carries a non-default base (attr1 != 0). Files that use nothing newer keep
# loading on older compilers, and the classic demos stay byte-for-byte.
SDS_MAGIC = formats.SDS_MAGIC
ALIGN = 64
# Feature corpora stream to disk in row chunks: constant memory at any
# corpus size, one write per chunk instead of two per row.
SDS_CHUNK_ROWS = 1 << 16

_K = formats.SMF_OP_KINDS
(OP_MATMUL, OP_ADDBIAS, OP_RELU, OP_GELU, OP_SILU, OP_MUL, OP_LAYERNORM,
 OP_ADD, OP_RMSNORM, OP_ROPE, OP_ATTENTION) = (
    _K["matmul"], _K["add_bias"], _K["relu"], _K["gelu"], _K["silu"],
    _K["mul"], _K["layer_norm"], _K["add"], _K["rms_norm"], _K["rope"],
    _K["attention"])
OP_EMBEDDING = _K["embedding"]  # SMF v4: gather table rows by the model's i32 token input


def _align(n: int) -> int:
    return (n + ALIGN - 1) & ~(ALIGN - 1)


def _s(name: str) -> bytes:
    b = name.encode()
    return struct.pack("<H", len(b)) + b


def _nbytes(data) -> int:
    return memoryview(data).nbytes


def _flat_bytes(data) -> memoryview:
    """A 1-D byte view of any C-contiguous bytes-like (bytes, a NumPy array,
    a memoryview): what file.write consumes, without copying the payload."""
    mv = memoryview(data)
    if mv.ndim == 1 and mv.format == "B":
        return mv
    if not mv.c_contiguous:
        raise ValueError("tensor payloads must be C-contiguous (pass the "
                         "array through numpy.ascontiguousarray)")
    return mv.cast("B")


class _SmfBuilder:
    def __init__(self, input_name: str, input_dim: int, seq_len: int = 0,
                 token_input: bool = False):
        self.input_name = input_name
        self.output_name = input_name
        self.seq_len = seq_len  # rows per sequence; 0 = non-sequential
        # A token-native model (SMF v4+) declares a rank-1 dynamic input:
        # one i32 token id per row, gathered on-device by kEmbedding.
        self.min_version = 4 if token_input else 3
        dims = [-1] if token_input else [-1, input_dim]
        self.tensors = [dict(name=input_name, dims=dims, const=False, data=b"",
                             nbytes=0)]
        self.ops = []

    def add_tensor(self, name, dims, data, nbytes=None):
        """`data` is any C-contiguous bytes-like — bytes, a memoryview, or a
        NumPy array — written as-is at serialization time, so a weight
        costs no copy between the framework and the file. It may instead be
        a zero-argument callable returning such a bytes-like, with `nbytes`
        its size: the payload is then produced only while it is being
        written, so a model whose weights need a transposed copy each holds
        one such copy at a time, not all of them at once."""
        if callable(data):
            if nbytes is None:
                raise ValueError(f"add_tensor({name!r}): a callable payload "
                                 "needs its nbytes up front")
        else:
            nbytes = _nbytes(data)
        self.tensors.append(dict(name=name, dims=list(dims), const=True,
                                 data=data, nbytes=int(nbytes)))

    def add_op(self, kind, name, inputs, output, attr0: int = 0,
               attr1: int = 0):
        self.ops.append(dict(kind=kind, name=name, inputs=inputs,
                             output=output, attr0=attr0, attr1=attr1))
        self.output_name = output

    @property
    def version(self) -> int:
        needs_v5 = any(op["attr1"] != 0 for op in self.ops)
        return max(self.min_version, 5 if needs_v5 else 3)

    def _meta(self, offsets, version: int) -> bytes:
        out = formats.SMF_PREAMBLE.pack(SMF_MAGIC, version,
                                        len(self.tensors), len(self.ops))
        out += _s(self.input_name) + _s(self.output_name)
        out += struct.pack("<Q", self.seq_len)
        for t in self.tensors:
            out += _s(t["name"])
            out += struct.pack("<BB", len(t["dims"]), 1 if t["const"] else 0)
            for d in t["dims"]:
                out += struct.pack("<q", d)
            out += struct.pack("<QQ", offsets.get(t["name"], 0), t["nbytes"])
        for op in self.ops:
            out += struct.pack("<B", op["kind"]) + _s(op["name"])
            out += struct.pack("<B", len(op["inputs"]))
            for i in op["inputs"]:
                out += _s(i)
            out += _s(op["output"])
            out += struct.pack("<I", op["attr0"])
            if version >= 5:
                out += struct.pack("<I", op["attr1"])
        return out

    def _layout(self):
        """(header bytes, {tensor name: data offset}, total file size): every
        constant tensor at a 64-byte-aligned offset after the header, the
        file padded to alignment after the last one."""
        version = self.version
        cursor, offsets = _align(len(self._meta({}, version))), {}
        for t in self.tensors:
            if not t["const"]:
                continue
            offsets[t["name"]] = cursor
            cursor = _align(cursor + t["nbytes"])
        return self._meta(offsets, version), offsets, cursor

    def write(self, f) -> int:
        """Stream the container to a binary file object and return the byte
        count. The header, each tensor's bytes (straight from the array
        that holds them), and the alignment gaps are written in file order,
        so the process never holds a second copy of the model: a 540 MB
        model exports at ~1x its size, not the ~6x of assembling the whole
        file in memory first. Byte-identical to serialize()."""
        meta, offsets, total = self._layout()
        f.write(meta)
        pos = len(meta)
        for t in self.tensors:
            if not t["const"]:
                continue
            o = offsets[t["name"]]
            if o > pos:
                f.write(b"\x00" * (o - pos))
            view = _flat_bytes(t["data"]() if callable(t["data"])
                               else t["data"])
            if view.nbytes != t["nbytes"]:
                raise ValueError(f"tensor {t['name']!r}: payload is "
                                 f"{view.nbytes} bytes, declared {t['nbytes']}")
            f.write(view)
            pos = o + view.nbytes
        if total > pos:
            f.write(b"\x00" * (total - pos))
        return total

    def serialize(self) -> bytes:
        """The whole container as one bytes object (library convenience;
        the file exporters stream with write() instead)."""
        buf = io.BytesIO()
        self.write(buf)
        return buf.getvalue()


def export_smf(model, path: str, input_name: str = "x"):
    """Export an nn.Sequential of Linear/ReLU/GELU/SiLU/LayerNorm to SMF."""
    import numpy as np
    import torch
    import torch.nn as nn

    linears = [m for m in model if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("export_smf: model contains no nn.Linear layers")

    def tensor(name, t):
        # Produced only while being written: the f32, C-contiguous array
        # shares the parameter's storage when it already is one (a bias),
        # and is a single transient copy when it is not (a transposed
        # weight, or a half-precision one) — so export peaks near the
        # model's own size, and no bytes() copy sits in between.
        t = t.detach()
        b.add_tensor(name, list(t.shape),
                     lambda t=t: np.asarray(t.float().contiguous().numpy(),
                                            dtype="<f4"),
                     nbytes=t.numel() * 4)

    b = _SmfBuilder(input_name, linears[0].in_features)
    prev, idx = input_name, 0
    for pos, m in enumerate(model):
        if isinstance(m, nn.Linear):
            tensor(f"w{idx}", m.weight.t())  # [in, out]
            b.add_op(OP_MATMUL, f"mm{idx}", [prev, f"w{idx}"], f"z{idx}")
            if m.bias is not None:
                tensor(f"b{idx}", m.bias)
                b.add_op(OP_ADDBIAS, f"ab{idx}", [f"z{idx}", f"b{idx}"],
                         f"zb{idx}")
                prev = f"zb{idx}"
            else:
                # nn.Linear(bias=False): the matmul output feeds forward
                # directly — no zero-bias tensor bloating the model.
                prev = f"z{idx}"
            idx += 1
        elif isinstance(m, (nn.ReLU, nn.GELU, nn.SiLU)):
            # Name by module position: consecutive activations must not collide.
            kind = (OP_RELU if isinstance(m, nn.ReLU)
                    else OP_GELU if isinstance(m, nn.GELU) else OP_SILU)
            b.add_op(kind, f"act{pos}", [prev], f"h{pos}")
            prev = f"h{pos}"
        elif isinstance(m, nn.LayerNorm):
            if len(m.normalized_shape) != 1:
                raise ValueError("export_smf: LayerNorm must normalize the "
                                 "last dimension only")
            # elementwise_affine=False has weight/bias of None; the SMF op
            # always takes gamma/beta, so synthesize the identity affine.
            d = m.normalized_shape[0]
            tensor(f"ln_g{pos}",
                   m.weight if m.weight is not None else torch.ones(d))
            tensor(f"ln_b{pos}",
                   m.bias if m.bias is not None else torch.zeros(d))
            b.add_op(OP_LAYERNORM, f"ln{pos}",
                     [prev, f"ln_g{pos}", f"ln_b{pos}"], f"n{pos}")
            prev = f"n{pos}"
        else:
            raise ValueError(f"export_smf: unsupported module {type(m).__name__}")

    with open(path, "wb") as f:
        b.write(f)
    print(f"wrote {path} ({idx} linear layers)")


def _decoder_dim(blocks, head, num_heads: int, caller: str) -> int:
    D = int(blocks[0]["wq"].shape[0]) if blocks else int(head["w_head"].shape[0])
    if D % num_heads != 0 or (D // num_heads) % 2 != 0:
        raise ValueError(f"{caller}: num_heads must divide D and "
                         "leave an even head width for RoPE")
    return D


def _rope_base_bits(rope_base: float) -> int:
    """attr1 encoding of a rotary base: its IEEE-754 single bits, exactly as
    the on-device kernel will read them (0 would mean the 10000 default)."""
    import math
    base = float(rope_base)
    if not math.isfinite(base) or base <= 1.0:
        raise ValueError(f"rope_base must be a finite float > 1, got {rope_base}")
    # Round-trip through f32 so the exporter's notion of θ matches the
    # kernel's: a base that is not exactly representable is rounded once,
    # here, and the rounded value is what gets written — so the checks
    # must hold for the ROUNDED value (1 + 2^-24 rounds to exactly 1.0f;
    # > FLT_MAX rounds to inf), or the compiler's sema would be the first
    # to notice.
    try:
        packed = struct.pack("<f", base)
    except OverflowError:
        raise ValueError(f"rope_base {rope_base} exceeds float32 range")
    rounded = struct.unpack("<f", packed)[0]
    if not math.isfinite(rounded) or rounded <= 1.0:
        raise ValueError(f"rope_base {rope_base} rounds to {rounded} in "
                         "float32, which is not a usable base (> 1)")
    if rounded == DEFAULT_ROPE_BASE:
        return 0  # the format default; keeps such files at SMF v3/v4
    return struct.unpack("<I", packed)[0]


def _emit_decoder_graph(b, blocks, head, num_heads: int, prev: str,
                        rope_base: float = DEFAULT_ROPE_BASE):
    """Append the pre-norm block stack and lm head, reading rows from `prev`."""
    import numpy as np

    base_bits = _rope_base_bits(rope_base)

    def tensor(name, arr):
        # Explicit little-endian f32 — the container's byte order, whatever
        # the host's — and no bytes() copy: the builder writes the array.
        a = np.ascontiguousarray(np.asarray(arr, dtype="<f4"))
        b.add_tensor(name, list(a.shape), a)
        return name

    for i, blk in enumerate(blocks):
        p = f"l{i}."
        tensor(p + "ln1_g", blk["ln1_g"])
        b.add_op(OP_RMSNORM, p + "ln1", [prev, p + "ln1_g"], p + "n1")
        for w in ("wq", "wk", "wv"):
            tensor(p + w, blk[w])
            # Optional projection bias (Qwen2-class attention_bias): a rank-1
            # AddBias after the MatMul; absent for Llama-class blocks.
            bias = blk.get("b" + w[1])
            if bias is None:
                b.add_op(OP_MATMUL, p + "mm_" + w, [p + "n1", p + w], p + w[1])
            else:
                tensor(p + "b" + w[1], bias)
                b.add_op(OP_MATMUL, p + "mm_" + w, [p + "n1", p + w],
                         p + w[1] + "_mm")
                b.add_op(OP_ADDBIAS, p + "ab_" + w, [p + w[1] + "_mm",
                                                     p + "b" + w[1]], p + w[1])
        b.add_op(OP_ROPE, p + "rope_q", [p + "q"], p + "qr", attr0=num_heads,
                 attr1=base_bits)
        b.add_op(OP_ROPE, p + "rope_k", [p + "k"], p + "kr", attr0=num_heads,
                 attr1=base_bits)
        b.add_op(OP_ATTENTION, p + "attn", [p + "qr", p + "kr", p + "v"],
                 p + "a", attr0=num_heads)
        tensor(p + "wo", blk["wo"])
        b.add_op(OP_MATMUL, p + "mm_wo", [p + "a", p + "wo"], p + "o")
        b.add_op(OP_ADD, p + "res1", [prev, p + "o"], p + "x1")
        tensor(p + "ln2_g", blk["ln2_g"])
        b.add_op(OP_RMSNORM, p + "ln2", [p + "x1", p + "ln2_g"], p + "n2")
        tensor(p + "w_gate", blk["w_gate"])
        tensor(p + "w_up", blk["w_up"])
        b.add_op(OP_MATMUL, p + "mm_gate", [p + "n2", p + "w_gate"], p + "g")
        b.add_op(OP_MATMUL, p + "mm_up", [p + "n2", p + "w_up"], p + "u")
        b.add_op(OP_SILU, p + "silu", [p + "g"], p + "s")
        b.add_op(OP_MUL, p + "swiglu", [p + "s", p + "u"], p + "m")
        tensor(p + "w_down", blk["w_down"])
        b.add_op(OP_MATMUL, p + "mm_down", [p + "m", p + "w_down"], p + "d")
        b.add_op(OP_ADD, p + "res2", [p + "x1", p + "d"], p + "x2")
        prev = p + "x2"

    tensor("lnf_g", head["lnf_g"])
    b.add_op(OP_RMSNORM, "lnf", [prev, "lnf_g"], "nf")
    tensor("w_head", head["w_head"])
    b.add_op(OP_MATMUL, "mm_head", ["nf", "w_head"], "logits")


def export_decoder_smf(blocks, head, path: str, seq_len: int, num_heads: int,
                       input_name: str = "x",
                       rope_base: float = DEFAULT_ROPE_BASE):
    """Export a pre-norm causal decoder stack to SMF v5.

    `rope_base` is the rotary θ (HF `rope_theta`): 10000 for GPT-NeoX /
    SmolLM-v1, 100000 for SmolLM2, 500000 for Llama 3.x, 1000000 for Qwen.
    It is written per Rope op (attr1) and lowered into the plan verbatim.

    `blocks` is a list of dicts of float32 numpy arrays, one per layer:
        ln1_g [D]; wq, wk, wv, wo [D, D]; ln2_g [D];
        w_gate, w_up [D, F]; w_down [F, D]
    `head` is {"lnf_g": [D], "w_head": [D, V]}. Rows of the runtime input
    are pre-embedded hidden states [T = B*seq_len, D] (embedding lookup is
    outside the update scope — the corpus carries embedded vectors). For a
    model that consumes raw token ids instead, use export_token_decoder_smf.
    Requires D % num_heads == 0 and (D // num_heads) % 2 == 0 (RoPE pairs).
    """
    D = _decoder_dim(blocks, head, num_heads, "export_decoder_smf")
    b = _SmfBuilder(input_name, D, seq_len=seq_len)
    _emit_decoder_graph(b, blocks, head, num_heads, prev=input_name,
                        rope_base=rope_base)
    with open(path, "wb") as f:
        b.write(f)
    print(f"wrote {path} ({len(blocks)} decoder blocks, seq_len={seq_len}, "
          f"heads={num_heads}, rope_base={float(rope_base):g})")


def export_token_decoder_smf(embedding, blocks, head, path: str, seq_len: int,
                             num_heads: int, input_name: str = "x",
                             rope_base: float = DEFAULT_ROPE_BASE):
    """Export a token-native pre-norm causal decoder to SMF v5.

    Same `blocks` / `head` dictionaries as export_decoder_smf, plus
    `embedding`: a float32 [V, D] table. The exported model's input is a
    rank-1 i32 row of token ids; the frozen table gathers on-device
    (kEmbedding), so the corpus stays token ids (export_token_sds) and
    never carries embedded vectors. The table itself is frozen — LoRA
    adapters attach to the projection matmuls as usual.
    """
    import numpy as np

    emb = np.ascontiguousarray(np.asarray(embedding, dtype="<f4"))
    if emb.ndim != 2:
        raise ValueError("export_token_decoder_smf: embedding must be [V, D]")
    D = _decoder_dim(blocks, head, num_heads, "export_token_decoder_smf")
    if int(emb.shape[1]) != D:
        raise ValueError("export_token_decoder_smf: embedding width "
                         f"{emb.shape[1]} does not match block width {D}")
    b = _SmfBuilder(input_name, D, seq_len=seq_len, token_input=True)
    b.add_tensor("emb", list(emb.shape), emb)
    b.add_op(OP_EMBEDDING, "embed", [input_name, "emb"], "e")
    _emit_decoder_graph(b, blocks, head, num_heads, prev="e",
                        rope_base=rope_base)
    with open(path, "wb") as f:
        b.write(f)
    print(f"wrote {path} ({len(blocks)} decoder blocks, seq_len={seq_len}, "
          f"heads={num_heads}, vocab={emb.shape[0]}, token-native, "
          f"rope_base={float(rope_base):g})")



# --- Hugging Face import (roadmap Project 4, Phase T2) ---------------------
# A Llama-class checkpoint directory (config.json + model.safetensors, or an
# index over shards) becomes the SeeML token-native decoder without torch,
# transformers or safetensors: the container is parsed by hand (it is a
# JSON header plus raw little-endian tensors) and the walk is NumPy. What
# the walk does, weight by weight:
#   [out, in] -> [in, out]      every Linear is transposed for MatMul(x, W)
#   GQA -> MHA                  k/v projections are repeated per query head
#                               (head h reads kv head h // (H / H_kv)) until
#                               the format carries KV heads natively
#   rotate-half -> interleaved  HF rotates pairs (c, c + d/2); SeeML rotates
#                               (2c, 2c+1) at the same frequency base^(-2c/d),
#                               so q and k output features are permuted
#                               within each head: SeeML[2c] = HF[c],
#                               SeeML[2c+1] = HF[c + d/2]. Scores are a dot
#                               product over d, invariant to a permutation
#                               applied to both; v and o are untouched.
#   rope_theta -> rope_base     per Rope op (SMF v5 attr1)
#   tied lm_head                w_head = embedding^T when lm_head is absent
#   RMSNorm eps                 the runtime's norms are fixed at 1e-5 (P7,
#                               #96, adds the attribute); a checkpoint with
#                               another eps is refused unless the drift is
#                               accepted explicitly.
# The parity check (--hf-parity) runs the SeeML-semantics NumPy forward
# below against `transformers` when it is installed — a second, independent
# implementation of every op the compiled plan will execute.

RUNTIME_NORM_EPS = 1e-5  # runtime/executor/normalization.cc; P7 (#96) lifts it

_SAFETENSORS_DTYPES = {"F32": "<f4", "F16": "<f2", "BF16": "<u2", "F64": "<f8"}


def read_safetensors(path: str):
    """{name: float32 array} from one safetensors file (F32/F16/BF16/F64).

    The format is an 8-byte little-endian header length, a JSON header
    mapping tensor names to {dtype, shape, data_offsets}, then the raw
    tensor bytes. BF16 is widened by placing the 16 bits in the top half
    of a float32 — exact, since bf16 is float32 with the low mantissa
    dropped."""
    import numpy as np

    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 8:
        raise ValueError(f"{path}: not a safetensors file (too short)")
    n = struct.unpack("<Q", raw[:8])[0]
    if n > len(raw) - 8:
        raise ValueError(f"{path}: safetensors header runs past the file")
    header = json.loads(raw[8:8 + n].decode("utf-8"))
    base = 8 + n
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = info["dtype"]
        if dtype not in _SAFETENSORS_DTYPES:
            raise ValueError(f"{path}: tensor '{name}' has dtype {dtype}; "
                             "the importer reads F32/F16/BF16/F64 only")
        b, e = info["data_offsets"]
        if b > e or base + e > len(raw):
            raise ValueError(f"{path}: tensor '{name}' offsets run past "
                             "the file")
        arr = np.frombuffer(raw, dtype=_SAFETENSORS_DTYPES[dtype],
                            count=(e - b) // np.dtype(
                                _SAFETENSORS_DTYPES[dtype]).itemsize,
                            offset=base + b)
        if dtype == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        else:
            arr = arr.astype(np.float32)
        out[name] = arr.reshape(info["shape"])
    return out


def load_hf_checkpoint(model_dir: str):
    """(config, {name: float32 array}) from a Hugging Face model directory:
    config.json plus model.safetensors or the shards its index names."""
    if not os.path.isdir(model_dir):
        raise ValueError(f"{model_dir}: not a directory (pass a local "
                         "checkout — `huggingface-cli download <id>` first)")
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise ValueError(f"{model_dir}: no config.json")
    with open(cfg_path) as f:
        config = json.load(f)
    single = os.path.join(model_dir, "model.safetensors")
    index = os.path.join(model_dir, "model.safetensors.index.json")
    tensors = {}
    if os.path.isfile(single):
        tensors.update(read_safetensors(single))
    elif os.path.isfile(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        for shard in shards:
            tensors.update(read_safetensors(os.path.join(model_dir, shard)))
    else:
        raise ValueError(f"{model_dir}: no model.safetensors (nor an index "
                         "over shards); .bin checkpoints are not read")
    return config, tensors


def _rope_interleave_order(d: int):
    """HF feature index for each SeeML feature within a head: SeeML pair
    (2c, 2c+1) <- HF (c, c + d/2)."""
    import numpy as np

    order = np.empty(d, dtype=np.int64)
    order[0::2] = np.arange(d // 2)
    order[1::2] = np.arange(d // 2) + d // 2
    return order


def hf_llama_to_seeml(config: dict, tensors: dict, seq_len: int,
                      allow_eps_drift: bool = False):
    """Convert a Llama-class checkpoint (llama / qwen2 / SmolLM2) into the
    embedding / blocks / head arrays the token-native exporter takes.
    Returns a dict with those plus num_heads, rope_base, and the notes
    printed to the user."""
    import numpy as np

    mt = config.get("model_type")
    if mt not in ("llama", "qwen2"):
        raise ValueError(f"model_type '{mt}' is not a Llama-class decoder "
                         "the importer walks (llama, qwen2)")
    if config.get("hidden_act", "silu") != "silu":
        raise ValueError(f"hidden_act '{config.get('hidden_act')}': the "
                         "SeeML decoder block is SwiGLU (silu) only")
    if config.get("rope_scaling") not in (None, {}):
        raise ValueError("rope_scaling is set; the runtime's RoPE is the "
                         "plain base^(-2c/d) recurrence")
    if config.get("mlp_bias", False):
        raise ValueError("mlp_bias=true: the SeeML MLP has no biases")
    D = int(config["hidden_size"])
    H = int(config["num_attention_heads"])
    Hkv = int(config.get("num_key_value_heads") or H)
    int(config["intermediate_size"])  # required; the widths come from the tensors
    L = int(config["num_hidden_layers"])
    V = int(config["vocab_size"])
    eps = float(config.get("rms_norm_eps", 1e-6))
    theta = float(config.get("rope_theta", DEFAULT_ROPE_BASE))
    max_pos = int(config.get("max_position_embeddings", seq_len))
    if D % H != 0:
        raise ValueError(f"hidden_size {D} is not a multiple of "
                         f"num_attention_heads {H}")
    d = D // H
    head_dim = int(config.get("head_dim") or d)
    if head_dim != d:
        raise ValueError(f"head_dim {head_dim} != hidden_size / heads {d}: "
                         "the SeeML attention geometry is H x (D / H)")
    if d % 2 != 0:
        raise ValueError(f"head width {d} is odd; RoPE needs pairs")
    if H % Hkv != 0:
        raise ValueError(f"num_attention_heads {H} is not a multiple of "
                         f"num_key_value_heads {Hkv}")
    if seq_len < 1:
        raise ValueError("--seq-len must be >= 1")
    if seq_len > max_pos:
        raise ValueError(f"--seq-len {seq_len} exceeds the checkpoint's "
                         f"max_position_embeddings {max_pos}")
    notes = []
    if abs(eps - RUNTIME_NORM_EPS) > 0:
        msg = (f"rms_norm_eps {eps:g} differs from the runtime's fixed "
               f"{RUNTIME_NORM_EPS:g} (P7, #96, adds the attribute)")
        if not allow_eps_drift:
            raise ValueError(msg + "; pass --allow-eps-drift to import "
                             "anyway (step 0 will not equal the source "
                             "model exactly)")
        notes.append("accepted eps drift: " + msg)
    if Hkv != H:
        notes.append(f"GQA: {Hkv} kv heads repeated to {H} query heads "
                     "(the format carries no kv heads yet)")

    def take(name):
        if name not in tensors:
            raise ValueError(f"checkpoint lacks tensor '{name}'")
        return tensors[name]

    order = _rope_interleave_order(d)
    rep = H // Hkv
    kv_of = np.arange(H) // rep  # kv head serving each query head

    def q_cols(w):  # [H*d, D] -> [D, H*d] with the RoPE permutation
        return np.ascontiguousarray(
            w.T.reshape(D, H, d)[:, :, order].reshape(D, H * d))

    def kv_cols(w, permute):  # [Hkv*d, D] -> repeated + (optionally) permuted
        t = w.T.reshape(D, Hkv, d)[:, kv_of, :]
        if permute:
            t = t[:, :, order]
        return np.ascontiguousarray(t.reshape(D, H * d))

    def q_bias(bv):
        return np.ascontiguousarray(bv.reshape(H, d)[:, order].reshape(H * d))

    def kv_bias(bv, permute):
        t = bv.reshape(Hkv, d)[kv_of, :]
        if permute:
            t = t[:, order]
        return np.ascontiguousarray(t.reshape(H * d))

    blocks = []
    for i in range(L):
        p = f"model.layers.{i}."
        blk = {
            "ln1_g": take(p + "input_layernorm.weight"),
            "wq": q_cols(take(p + "self_attn.q_proj.weight")),
            "wk": kv_cols(take(p + "self_attn.k_proj.weight"), True),
            "wv": kv_cols(take(p + "self_attn.v_proj.weight"), False),
            "wo": np.ascontiguousarray(take(p + "self_attn.o_proj.weight").T),
            "ln2_g": take(p + "post_attention_layernorm.weight"),
            "w_gate": np.ascontiguousarray(take(p + "mlp.gate_proj.weight").T),
            "w_up": np.ascontiguousarray(take(p + "mlp.up_proj.weight").T),
            "w_down": np.ascontiguousarray(take(p + "mlp.down_proj.weight").T),
        }
        if p + "self_attn.q_proj.bias" in tensors:  # Qwen2-class
            blk["bq"] = q_bias(take(p + "self_attn.q_proj.bias"))
            blk["bk"] = kv_bias(take(p + "self_attn.k_proj.bias"), True)
            blk["bv"] = kv_bias(take(p + "self_attn.v_proj.bias"), False)
        if p + "self_attn.o_proj.bias" in tensors:
            raise ValueError("o_proj has a bias; the SeeML block has none")
        blocks.append(blk)
    emb = np.ascontiguousarray(take("model.embed_tokens.weight"))
    if emb.shape != (V, D):
        raise ValueError(f"embedding is {emb.shape}, expected {(V, D)}")
    if "lm_head.weight" in tensors:
        w_head = np.ascontiguousarray(take("lm_head.weight").T)
        notes.append("untied lm_head")
    else:
        w_head = np.ascontiguousarray(emb.T)
        notes.append("tied lm_head (w_head = embedding^T; the merged head "
                     "adapter is not written back into the embedding)")
    head = {"lnf_g": take("model.norm.weight"), "w_head": w_head}
    return {"embedding": emb, "blocks": blocks, "head": head,
            "num_heads": H, "rope_base": theta, "seq_len": seq_len,
            "notes": notes, "vocab": V, "dim": D}


def export_hf_decoder(model_dir: str, out_path: str, seq_len: int,
                      allow_eps_drift: bool = False):
    """--hf: import a Llama-class checkpoint and write the SMF."""
    config, tensors = load_hf_checkpoint(model_dir)
    conv = hf_llama_to_seeml(config, tensors, seq_len,
                             allow_eps_drift=allow_eps_drift)
    for note in conv["notes"]:
        print(f"hf import: {note}", file=sys.stderr)
    export_token_decoder_smf(conv["embedding"], conv["blocks"], conv["head"],
                             out_path, seq_len=seq_len,
                             num_heads=conv["num_heads"],
                             rope_base=conv["rope_base"])
    return conv


def reference_decoder_logits(embedding, blocks, head, num_heads: int,
                             rope_base: float, tokens):
    """The SeeML decoder's semantics in NumPy: the forward the compiled
    plan executes (RMSNorm at the runtime's eps, interleaved RoPE with the
    kernel's frequency recurrence, causal softmax, SwiGLU), for parity
    checks. tokens: int array [B, S]; returns float32 logits [B, S, V]."""
    import numpy as np

    tok = np.asarray(tokens)
    B, S = tok.shape
    H = num_heads
    x = np.asarray(embedding, np.float32)[tok]  # [B, S, D]
    D = x.shape[-1]
    d = D // H

    def rmsnorm(v, g):
        rs = 1.0 / np.sqrt(np.mean(v.astype(np.float32) ** 2, axis=-1,
                                   keepdims=True) + RUNTIME_NORM_EPS)
        return (v * rs * g).astype(np.float32)

    # angle(s, c) = s * base^(-2c/d) by the kernel's multiplicative recurrence
    step = np.float32(rope_base) ** np.float32(-2.0 / d)
    freq = np.empty(d // 2, np.float32)
    f = np.float32(1.0)
    for c in range(d // 2):
        freq[c] = f
        f = np.float32(f * step)
    theta = np.arange(S, dtype=np.float32)[:, None] * freq[None, :]  # [S, d/2]
    cs, sn = np.cos(theta), np.sin(theta)

    def rope(v):  # [B, S, H*d]
        v = v.reshape(B, S, H, d)
        a, b = v[..., 0::2], v[..., 1::2]
        c_, s_ = cs[None, :, None, :], sn[None, :, None, :]
        out = np.empty_like(v)
        out[..., 0::2] = a * c_ - b * s_
        out[..., 1::2] = a * s_ + b * c_
        return out.reshape(B, S, H * d)

    mask = np.triu(np.ones((S, S), dtype=bool), 1)
    for blk in blocks:
        n1 = rmsnorm(x, blk["ln1_g"])
        q = n1 @ blk["wq"] + (blk["bq"] if "bq" in blk else 0)
        k = n1 @ blk["wk"] + (blk["bk"] if "bk" in blk else 0)
        v = n1 @ blk["wv"] + (blk["bv"] if "bv" in blk else 0)
        q, k = rope(q), rope(k)
        qh = q.reshape(B, S, H, d).transpose(0, 2, 1, 3)
        kh = k.reshape(B, S, H, d).transpose(0, 2, 1, 3)
        vh = v.reshape(B, S, H, d).transpose(0, 2, 1, 3)
        scores = (qh @ kh.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(d))
        scores = np.where(mask, np.float32(-np.inf), scores)
        scores = scores - scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p = p / p.sum(axis=-1, keepdims=True)
        a = (p @ vh).transpose(0, 2, 1, 3).reshape(B, S, H * d)
        x = x + a @ blk["wo"]
        n2 = rmsnorm(x, blk["ln2_g"])
        g = n2 @ blk["w_gate"]
        u = n2 @ blk["w_up"]
        x = x + ((g / (1.0 + np.exp(-g))) * u) @ blk["w_down"]
    nf = rmsnorm(x, head["lnf_g"])
    return (nf @ head["w_head"]).astype(np.float32)


def hf_parity(model_dir: str, conv: dict, batch: int = 2, seed: int = 0):
    """max |Δ logits| between the SeeML-semantics NumPy forward of the
    converted arrays and `transformers`' forward of the checkpoint on
    seeded random tokens. Needs torch + transformers (tier 2)."""
    import numpy as np
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise RuntimeError("--hf-parity needs torch and transformers "
                           f"({e.name} is not installed)") from e

    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, conv["vocab"], (batch, conv["seq_len"]),
                          dtype=np.int64)
    try:  # transformers >= 5 spells the keyword `dtype`; older, `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                     dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                     torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        want = model(torch.from_numpy(tokens)).logits.float().numpy()
    got = reference_decoder_logits(conv["embedding"], conv["blocks"],
                                   conv["head"], conv["num_heads"],
                                   conv["rope_base"], tokens)
    return float(np.max(np.abs(got - want))), float(np.max(np.abs(want)))


def export_text_corpus(model_dir: str, text_path: str, out_path: str,
                       seq_len: int, vocab: int):
    """--text-corpus: tokenize a text file with the checkpoint's
    tokenizer.json and write consecutive (seq_len + 1)-token records
    (the remainder is dropped). Needs the `tokenizers` package (tier 2)."""
    import numpy as np
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise RuntimeError("--text-corpus needs the `tokenizers` package "
                           "(pip install tokenizers)") from e
    tok_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.isfile(tok_path):
        raise ValueError(f"{model_dir}: no tokenizer.json for --text-corpus")
    with open(text_path, encoding="utf-8") as f:
        text = f.read()
    ids = np.asarray(Tokenizer.from_file(tok_path).encode(text).ids,
                     dtype=np.int64)
    if ids.size and (ids.min() < 0 or ids.max() >= vocab):
        raise ValueError("tokenizer produced ids outside the model's vocab")
    rec = seq_len + 1
    n = ids.size // rec
    if n == 0:
        raise ValueError(f"{text_path}: {ids.size} tokens is fewer than one "
                         f"record of {rec}")
    export_token_sds(ids[:n * rec].reshape(n, rec).astype(np.int32), out_path)
    print(f"text corpus: {ids.size} tokens -> {n} records of {rec}",
          file=sys.stderr)

def export_sds(inputs, labels, path: str, label_kind: int = 1):
    """inputs: float32 array [N, D]; labels: int32 [N] (kind 1),
    float32 [N, L] (kind 2), or None (kind 0, distillation corpora)."""
    import numpy as np

    x = np.ascontiguousarray(np.asarray(inputs, dtype="<f4"))
    if x.ndim != 2:
        raise ValueError(f"export_sds: inputs must be [N, D], got shape "
                         f"{x.shape}")
    n, d = x.shape
    if labels is None:
        label_kind, label_dim, lab = 0, 0, None
    elif label_kind == 1:
        lab = np.asarray(labels)
        if not np.issubdtype(lab.dtype, np.integer):
            raise ValueError(
                "export_sds: label_kind=1 expects integer class labels; "
                "pass label_kind=2 for dense float targets")
        lab = lab.astype("<i4").reshape(n)
        label_dim = 0
    else:
        lab = np.asarray(labels, dtype="<f4").reshape(n, -1)
        label_dim = lab.shape[1]

    # Each record is the input row immediately followed by its label
    # (nothing for kind 0, one i32 for kind 1, `label_dim` f32 for kind 2).
    # A packed structured dtype is exactly that layout, so rows interleave
    # in NumPy and stream out in chunks — the same bytes as a per-row loop,
    # at memcpy speed and constant memory.
    if lab is None:
        record = None
    elif label_kind == 1:
        record = np.dtype([("x", "<f4", (d,)), ("y", "<i4")])
    else:
        record = np.dtype([("x", "<f4", (d,)), ("y", "<f4", (label_dim,))])

    with open(path, "wb") as f:
        f.write(formats.SDS_HEADER.pack(SDS_MAGIC, 1, n, d, label_kind, 0, label_dim))
        for start in range(0, n, SDS_CHUNK_ROWS):
            end = min(n, start + SDS_CHUNK_ROWS)
            if record is None:
                f.write(_flat_bytes(x[start:end]))
            else:
                rows = np.empty(end - start, dtype=record)
                rows["x"] = x[start:end]
                rows["y"] = lab[start:end]
                f.write(_flat_bytes(rows))
    print(f"wrote {path} ({n} samples, input_dim={d}, label_kind={label_kind})")


def export_token_sds(records, path: str):
    """Export a token corpus to SDS v2 (for token-native SMF v4 models).

    `records`: int32 array [N, S + 1] — N training records of S + 1 token
    ids each, where S is the model's compiled seq_len. No labels are
    stored: the runtime derives next-token class labels from the shifted
    view (record[1:] labels record[:-1]).
    """
    import numpy as np

    raw = np.asarray(records)
    if np.issubdtype(raw.dtype, np.floating):
        # A float array here is almost always a mistake (a logits/embedding
        # array handed where ids belong); silently truncating 3.7 → 3 would
        # train on garbage labels. Exact integers in float dtype are fine.
        if not np.all(np.isfinite(raw)) or not np.all(np.mod(raw, 1) == 0):
            raise ValueError("export_token_sds: records must be integer "
                             "token ids (got non-integral float values)")
    elif not (np.issubdtype(raw.dtype, np.integer) or raw.size == 0):
        raise ValueError("export_token_sds: records must be integer token "
                         f"ids (got dtype {raw.dtype})")
    if raw.size and (raw.max() > np.iinfo(np.int32).max or
                     raw.min() < np.iinfo(np.int32).min):
        raise ValueError("export_token_sds: token id does not fit int32")
    a = np.ascontiguousarray(raw.astype("<i4", copy=False))
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError("export_token_sds: records must be [N, seq_len + 1]")
    if a.shape[0] == 0:
        raise ValueError("export_token_sds: no records — an empty corpus "
                         "would only fail later, at on-device load")
    if (a < 0).any():
        raise ValueError("export_token_sds: negative token id")
    n, s = a.shape[0], a.shape[1] - 1
    with open(path, "wb") as f:
        f.write(formats.SDS_HEADER.pack(SDS_MAGIC, 2, n, s, 1, 1, 0))
        f.write(_flat_bytes(a))
    print(f"wrote {path} ({n} records, seq_len={s}, token-native)")


def _demo(out_dir: str, width: int = 32, depth: int = 1, samples: int = 2048,
          seed: int = 0, corpus_kind: str = "class"):
    import numpy as np
    import torch
    import torch.nn as nn

    def mlp(w):
        layers = [nn.Linear(16, w), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(w, w), nn.ReLU()]
        return nn.Sequential(*layers, nn.Linear(w, 4))

    # One seed, student drawn before teacher — the classic defaults
    # (width=32, depth=1, seed=0) reproduce the original demo byte-for-byte.
    torch.manual_seed(seed)
    export_smf(mlp(width), f"{out_dir}/model.smf")
    export_smf(mlp(2 * width), f"{out_dir}/teacher.smf")

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((samples, 16), dtype=np.float32)
    y = (x[:, :4].sum(axis=1) > 0).astype(np.int32) + 2 * (x[:, 0] > 0)
    if corpus_kind == "class":
        export_sds(x, y, f"{out_dir}/corpus.sds")
    elif corpus_kind == "dense":
        export_sds(x, np.eye(4, dtype=np.float32)[y], f"{out_dir}/corpus.sds",
                   label_kind=2)
    else:  # "none": unlabeled — the corpus a --loss kl distillation wants
        export_sds(x, None, f"{out_dir}/corpus.sds")


def _demo_decoder(out_dir: str, vocab: int = 50, dim: int = 32,
                  heads: int = 4, seq: int = 8, ffn: int = 0,
                  blocks_n: int = 2, samples: int = 192, seed: int = 0,
                  rope_base: float = DEFAULT_ROPE_BASE):
    """A tiny token-native decoder plus a corpus it can actually learn:
    a cyclic-successor language where token t is always followed by
    (t + 3) % vocab. Needs NumPy only — no PyTorch."""
    import numpy as np

    rng = np.random.default_rng(seed)
    ffn = ffn or 2 * dim  # classic defaults (dim=32) give the original 64

    def mat(rows, cols):
        return (rng.standard_normal((rows, cols)) * 0.15).astype(np.float32)

    blocks = [{
        "ln1_g": np.ones(dim, np.float32), "wq": mat(dim, dim),
        "wk": mat(dim, dim), "wv": mat(dim, dim), "wo": mat(dim, dim),
        "ln2_g": np.ones(dim, np.float32), "w_gate": mat(dim, ffn),
        "w_up": mat(dim, ffn), "w_down": mat(ffn, dim),
    } for _ in range(blocks_n)]
    head = {"lnf_g": np.ones(dim, np.float32), "w_head": mat(dim, vocab)}
    emb = (rng.standard_normal((vocab, dim)) * 0.5).astype(np.float32)
    export_token_decoder_smf(emb, blocks, head, f"{out_dir}/decoder.smf",
                             seq_len=seq, num_heads=heads, rope_base=rope_base)

    starts = rng.integers(0, vocab, samples)
    records = (starts[:, None] + 3 * np.arange(seq + 1)) % vocab
    export_token_sds(records.astype(np.int32),
                     f"{out_dir}/decoder_corpus.sds")
    print(f"try: seeml-update-compile --source {out_dir}/decoder.smf"
          f" --out {out_dir}/pkg/ --data-batch {4 * seq} --loss xent"
          " --lora-rank 4 --lora-alpha 8 --optimizer adamw --lr 2e-3"
          " --steps 600 --build")


def _export_corpus(data_path: str, out_path: str):
    """Convert a .npz (or a bare .npy of token records) into an SDS corpus."""
    import numpy as np

    if data_path.endswith(".npy"):
        export_token_sds(np.load(data_path), out_path)
        return
    z = np.load(data_path)
    names = set(z.files)
    if "records" in names:
        export_token_sds(z["records"], out_path)
    elif "inputs" in names:
        labels = z["labels"] if "labels" in names else None
        if labels is None:
            export_sds(z["inputs"], None, out_path)
        else:
            kind = 1 if np.issubdtype(labels.dtype, np.integer) else 2
            export_sds(z["inputs"], labels, out_path, label_kind=kind)
    else:
        raise ValueError("--corpus: expected array 'records' (token ids) or "
                         f"'inputs' (+ optional 'labels') in {data_path}, "
                         f"found {sorted(names)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", metavar="OUT_DIR",
                        help="emit a demo model/teacher/corpus into OUT_DIR")
    parser.add_argument("--demo-decoder", metavar="OUT_DIR",
                        help="emit a token-native demo decoder/corpus "
                             "into OUT_DIR (NumPy only, no PyTorch)")
    parser.add_argument("--corpus", nargs=2, metavar=("DATA", "OUT_SDS"),
                        help="convert a .npz/.npy into an SDS corpus "
                             "(see the usage text above)")
    parser.add_argument("--hf", nargs=2, metavar=("MODEL_DIR", "OUT_SMF"),
                        help="import a Llama-class Hugging Face checkpoint "
                             "directory (llama / qwen2 / SmolLM2; config.json "
                             "+ safetensors) as a token-native decoder; needs "
                             "--seq-len; NumPy only")
    parser.add_argument("--allow-eps-drift", action="store_true",
                        help="--hf only: import a checkpoint whose "
                             "rms_norm_eps differs from the runtime's 1e-5")
    parser.add_argument("--hf-parity", action="store_true",
                        help="--hf only: after the export, compare the "
                             "SeeML-semantics NumPy forward against "
                             "transformers (needs torch + transformers)")
    parser.add_argument("--text-corpus", nargs=2,
                        metavar=("TEXT", "OUT_SDS"),
                        help="--hf only: tokenize a UTF-8 text file with the "
                             "checkpoint's tokenizer.json into (seq-len + 1)"
                             "-token records (needs the tokenizers package)")
    parser.add_argument("--seed", type=int, metavar="N",
                        help="demo RNG seed for weights and corpora (0)")
    parser.add_argument("--samples", type=int, metavar="N",
                        help="demo corpus size: samples for --demo (2048), "
                             "records for --demo-decoder (192)")
    parser.add_argument("--width", type=int, metavar="N",
                        help="hidden width for --demo (32), model dim for "
                             "--demo-decoder (32)")
    parser.add_argument("--depth", type=int, metavar="N",
                        help="--demo only: hidden Linear+ReLU layers (1)")
    parser.add_argument("--corpus-kind", choices=("class", "dense", "none"),
                        help="--demo only: labels — int32 classes for xent, "
                             "one-hot floats for mse, none for distillation "
                             "(class)")
    parser.add_argument("--vocab", type=int, metavar="N",
                        help="--demo-decoder only: vocabulary size (50)")
    parser.add_argument("--heads", type=int, metavar="N",
                        help="--demo-decoder only: attention heads (4)")
    parser.add_argument("--seq-len", type=int, metavar="N",
                        help="compiled sequence length: --demo-decoder (8) "
                             "or --hf (required)")
    parser.add_argument("--blocks", type=int, metavar="N",
                        help="--demo-decoder only: decoder blocks (2)")
    parser.add_argument("--ffn", type=int, metavar="N",
                        help="--demo-decoder only: FFN width (2x width)")
    parser.add_argument("--rope-base", type=float, metavar="THETA",
                        help="--demo-decoder only: rotary base θ (HF "
                             "rope_theta; default 10000 — 500000 for Llama 3, "
                             "1000000 for Qwen)")
    args = parser.parse_args()
    if args.rope_base is not None:
        try:
            _rope_base_bits(args.rope_base)
        except ValueError as e:
            parser.error(f"--rope-base: {e}")

    if not (args.demo or args.demo_decoder or args.corpus or args.hf):
        parser.print_help()
        sys.exit(0)

    # Strict-CLI discipline, as in seeml-update-compile: a knob that cannot
    # apply to the requested mode is a hard error (exit 2), never silently
    # ignored — a demo you didn't mean to configure is worse than one that
    # refuses to start.
    def _require(mode_ok, mode_name, **flags):
        for name, value in flags.items():
            if value is not None and not mode_ok:
                parser.error(f"--{name.replace('_', '-')} requires "
                             f"{mode_name}")
    _require(args.demo, "--demo",
             depth=args.depth, corpus_kind=args.corpus_kind)
    _require(args.demo_decoder, "--demo-decoder", vocab=args.vocab,
             heads=args.heads, blocks=args.blocks, ffn=args.ffn,
             rope_base=args.rope_base)
    _require(args.demo_decoder or args.hf, "--demo-decoder or --hf",
             seq_len=args.seq_len)
    _require(args.hf, "--hf",
             allow_eps_drift=args.allow_eps_drift or None,
             hf_parity=args.hf_parity or None, text_corpus=args.text_corpus)
    if args.hf and args.seq_len is None:
        parser.error("--hf requires --seq-len (the compiled sequence length)")
    _require(args.demo or args.demo_decoder, "--demo or --demo-decoder",
             width=args.width, samples=args.samples, seed=args.seed)
    for name in ("samples", "width", "depth", "vocab", "heads", "seq_len",
                 "blocks", "ffn"):
        v = getattr(args, name)
        if v is not None and v < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1, got {v}")

    import os

    if args.demo:
        os.makedirs(args.demo, exist_ok=True)
        _demo(args.demo, width=args.width or 32, depth=args.depth or 1,
              samples=args.samples or 2048,
              seed=args.seed if args.seed is not None else 0,
              corpus_kind=args.corpus_kind or "class")
    if args.demo_decoder:
        dim, heads = args.width or 32, args.heads or 4
        if dim % heads != 0 or (dim // heads) % 2 != 0:
            parser.error(f"--width {dim} with --heads {heads}: width must "
                         "divide into heads with an even head width "
                         "(RoPE pairs)")
        os.makedirs(args.demo_decoder, exist_ok=True)
        _demo_decoder(args.demo_decoder, vocab=args.vocab or 50, dim=dim,
                      heads=heads, seq=args.seq_len or 8, ffn=args.ffn or 0,
                      blocks_n=args.blocks or 2, samples=args.samples or 192,
                      seed=args.seed if args.seed is not None else 0,
                      rope_base=(args.rope_base if args.rope_base is not None
                                 else DEFAULT_ROPE_BASE))
    if args.corpus:
        _export_corpus(args.corpus[0], args.corpus[1])
    if args.hf:
        try:
            conv = export_hf_decoder(args.hf[0], args.hf[1], args.seq_len,
                                     allow_eps_drift=args.allow_eps_drift)
            if args.text_corpus:
                export_text_corpus(args.hf[0], args.text_corpus[0],
                                   args.text_corpus[1], args.seq_len,
                                   conv["vocab"])
            if args.hf_parity:
                delta, scale = hf_parity(args.hf[0], conv)
                print(f"hf parity: max |Δ logits| = {delta:.3g} "
                      f"(max |logits| {scale:.3g}) vs transformers",
                      file=sys.stderr)
        except (ValueError, RuntimeError) as e:
            print(f"export_model.py: --hf: {e}", file=sys.stderr)
            sys.exit(2)
