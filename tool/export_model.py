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

    from export_model import export_smf, export_sds
        export_smf(torch_sequential, "model.smf")
        export_sds(inputs, labels, "corpus.sds")

    from export_model import export_token_decoder_smf, export_token_sds
        export_token_decoder_smf(embedding, blocks, head, "model.smf",
                                 seq_len=S, num_heads=H)
        export_token_sds(token_records, "corpus.sds")  # [N, S+1] i32

Supported modules inside an nn.Sequential:
    nn.Linear     -> MatMul(x, W[in,out]) + AddBias(b[out])  (W stored
                     transposed from PyTorch's [out, in] layout)
    nn.ReLU       -> Relu
    nn.GELU       -> Gelu (tanh approximation on-device)
    nn.SiLU       -> Silu
    nn.LayerNorm  -> LayerNorm(x, gamma, beta) over the last dim
"""

import argparse
import struct
import sys

SMF_MAGIC = 0x31464D53  # "SMF1"
SMF_VERSION = 3         # v3: transformer op kinds, seq_len, per-op attr0
SDS_MAGIC = 0x31534453  # "SDS1"
ALIGN = 64

(OP_MATMUL, OP_ADDBIAS, OP_RELU, OP_GELU, OP_SILU, OP_MUL, OP_LAYERNORM,
 OP_ADD, OP_RMSNORM, OP_ROPE, OP_ATTENTION) = range(11)
OP_EMBEDDING = 11  # SMF v4: gather table rows by the model's i32 token input


def _align(n: int) -> int:
    return (n + ALIGN - 1) & ~(ALIGN - 1)


def _s(name: str) -> bytes:
    b = name.encode()
    return struct.pack("<H", len(b)) + b


class _SmfBuilder:
    def __init__(self, input_name: str, input_dim: int, seq_len: int = 0,
                 token_input: bool = False):
        self.input_name = input_name
        self.output_name = input_name
        self.seq_len = seq_len  # rows per sequence; 0 = non-sequential
        # A token-native model (SMF v4) declares a rank-1 dynamic input:
        # one i32 token id per row, gathered on-device by kEmbedding.
        self.version = 4 if token_input else SMF_VERSION
        dims = [-1] if token_input else [-1, input_dim]
        self.tensors = [dict(name=input_name, dims=dims, const=False, data=b"")]
        self.ops = []

    def add_tensor(self, name, dims, data: bytes):
        self.tensors.append(dict(name=name, dims=list(dims), const=True, data=data))

    def add_op(self, kind, name, inputs, output, attr0: int = 0):
        self.ops.append(dict(kind=kind, name=name, inputs=inputs,
                             output=output, attr0=attr0))
        self.output_name = output

    def serialize(self) -> bytes:
        def meta(offsets):
            out = struct.pack("<IIII", SMF_MAGIC, self.version,
                              len(self.tensors), len(self.ops))
            out += _s(self.input_name) + _s(self.output_name)
            out += struct.pack("<Q", self.seq_len)
            for t in self.tensors:
                out += _s(t["name"])
                out += struct.pack("<BB", len(t["dims"]), 1 if t["const"] else 0)
                for d in t["dims"]:
                    out += struct.pack("<q", d)
                out += struct.pack("<QQ", offsets.get(t["name"], 0), len(t["data"]))
            for op in self.ops:
                out += struct.pack("<B", op["kind"]) + _s(op["name"])
                out += struct.pack("<B", len(op["inputs"]))
                for i in op["inputs"]:
                    out += _s(i)
                out += _s(op["output"])
                out += struct.pack("<I", op["attr0"])
            return out

        meta_size = len(meta({}))
        cursor, offsets = _align(meta_size), {}
        for t in self.tensors:
            if not t["const"]:
                continue
            offsets[t["name"]] = cursor
            cursor = _align(cursor + len(t["data"]))

        blob = bytearray(meta(offsets))
        blob.extend(b"\x00" * (cursor - len(blob)))
        for t in self.tensors:
            if t["const"]:
                o = offsets[t["name"]]
                blob[o : o + len(t["data"])] = t["data"]
        return bytes(blob)


def export_smf(model, path: str, input_name: str = "x"):
    """Export an nn.Sequential of Linear/ReLU/GELU/SiLU/LayerNorm to SMF."""
    import torch
    import torch.nn as nn

    linears = [m for m in model if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("export_smf: model contains no nn.Linear layers")

    b = _SmfBuilder(input_name, linears[0].in_features)
    prev, idx = input_name, 0
    for pos, m in enumerate(model):
        if isinstance(m, nn.Linear):
            w = m.weight.detach().t().contiguous().float()  # [in, out]
            b.add_tensor(f"w{idx}", list(w.shape), w.numpy().tobytes())
            b.add_op(OP_MATMUL, f"mm{idx}", [prev, f"w{idx}"], f"z{idx}")
            if m.bias is not None:
                bias = m.bias.detach().float()
                b.add_tensor(f"b{idx}", [bias.numel()], bias.numpy().tobytes())
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
            gamma = (m.weight.detach().float() if m.weight is not None
                     else torch.ones(d))
            beta = (m.bias.detach().float() if m.bias is not None
                    else torch.zeros(d))
            b.add_tensor(f"ln_g{pos}", [gamma.numel()], gamma.numpy().tobytes())
            b.add_tensor(f"ln_b{pos}", [beta.numel()], beta.numpy().tobytes())
            b.add_op(OP_LAYERNORM, f"ln{pos}",
                     [prev, f"ln_g{pos}", f"ln_b{pos}"], f"n{pos}")
            prev = f"n{pos}"
        else:
            raise ValueError(f"export_smf: unsupported module {type(m).__name__}")

    with open(path, "wb") as f:
        f.write(b.serialize())
    print(f"wrote {path} ({idx} linear layers)")


def _decoder_dim(blocks, head, num_heads: int, caller: str) -> int:
    D = int(blocks[0]["wq"].shape[0]) if blocks else int(head["w_head"].shape[0])
    if D % num_heads != 0 or (D // num_heads) % 2 != 0:
        raise ValueError(f"{caller}: num_heads must divide D and "
                         "leave an even head width for RoPE")
    return D


def _emit_decoder_graph(b, blocks, head, num_heads: int, prev: str):
    """Append the pre-norm block stack and lm head, reading rows from `prev`."""
    import numpy as np

    def tensor(name, arr):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        b.add_tensor(name, list(a.shape), a.tobytes())
        return name

    for i, blk in enumerate(blocks):
        p = f"l{i}."
        tensor(p + "ln1_g", blk["ln1_g"])
        b.add_op(OP_RMSNORM, p + "ln1", [prev, p + "ln1_g"], p + "n1")
        for w in ("wq", "wk", "wv"):
            tensor(p + w, blk[w])
            b.add_op(OP_MATMUL, p + "mm_" + w, [p + "n1", p + w], p + w[1])
        b.add_op(OP_ROPE, p + "rope_q", [p + "q"], p + "qr", attr0=num_heads)
        b.add_op(OP_ROPE, p + "rope_k", [p + "k"], p + "kr", attr0=num_heads)
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
                       input_name: str = "x"):
    """Export a pre-norm causal decoder stack to SMF v3.

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
    _emit_decoder_graph(b, blocks, head, num_heads, prev=input_name)
    with open(path, "wb") as f:
        f.write(b.serialize())
    print(f"wrote {path} ({len(blocks)} decoder blocks, seq_len={seq_len}, "
          f"heads={num_heads})")


def export_token_decoder_smf(embedding, blocks, head, path: str, seq_len: int,
                             num_heads: int, input_name: str = "x"):
    """Export a token-native pre-norm causal decoder to SMF v4.

    Same `blocks` / `head` dictionaries as export_decoder_smf, plus
    `embedding`: a float32 [V, D] table. The exported model's input is a
    rank-1 i32 row of token ids; the frozen table gathers on-device
    (kEmbedding), so the corpus stays token ids (export_token_sds) and
    never carries embedded vectors. The table itself is frozen — LoRA
    adapters attach to the projection matmuls as usual.
    """
    import numpy as np

    emb = np.ascontiguousarray(np.asarray(embedding, dtype=np.float32))
    if emb.ndim != 2:
        raise ValueError("export_token_decoder_smf: embedding must be [V, D]")
    D = _decoder_dim(blocks, head, num_heads, "export_token_decoder_smf")
    if int(emb.shape[1]) != D:
        raise ValueError("export_token_decoder_smf: embedding width "
                         f"{emb.shape[1]} does not match block width {D}")
    b = _SmfBuilder(input_name, D, seq_len=seq_len, token_input=True)
    b.add_tensor("emb", list(emb.shape), emb.tobytes())
    b.add_op(OP_EMBEDDING, "embed", [input_name, "emb"], "e")
    _emit_decoder_graph(b, blocks, head, num_heads, prev="e")
    with open(path, "wb") as f:
        f.write(b.serialize())
    print(f"wrote {path} ({len(blocks)} decoder blocks, seq_len={seq_len}, "
          f"heads={num_heads}, vocab={emb.shape[0]}, token-native)")


def export_sds(inputs, labels, path: str, label_kind: int = 1):
    """inputs: float32 array [N, D]; labels: int32 [N] (kind 1),
    float32 [N, L] (kind 2), or None (kind 0, distillation corpora)."""
    import numpy as np

    x = np.asarray(inputs, dtype=np.float32)
    n, d = x.shape
    if labels is None:
        label_kind, label_dim, lab = 0, 0, None
    elif label_kind == 1:
        lab = np.asarray(labels)
        if not np.issubdtype(lab.dtype, np.integer):
            raise ValueError(
                "export_sds: label_kind=1 expects integer class labels; "
                "pass label_kind=2 for dense float targets")
        lab = lab.astype(np.int32).reshape(n)
        label_dim = 0
    else:
        lab = np.asarray(labels, dtype=np.float32).reshape(n, -1)
        label_dim = lab.shape[1]

    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQIIQ", SDS_MAGIC, 1, n, d, label_kind, 0, label_dim))
        for i in range(n):
            f.write(x[i].tobytes())
            if lab is not None:
                f.write(lab[i].tobytes())
    print(f"wrote {path} ({n} samples, input_dim={d}, label_kind={label_kind})")


def export_token_sds(records, path: str):
    """Export a token corpus to SDS v2 (for token-native SMF v4 models).

    `records`: int32 array [N, S + 1] — N training records of S + 1 token
    ids each, where S is the model's compiled seq_len. No labels are
    stored: the runtime derives next-token class labels from the shifted
    view (record[1:] labels record[:-1]).
    """
    import numpy as np

    a = np.ascontiguousarray(np.asarray(records, dtype=np.int32))
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError("export_token_sds: records must be [N, seq_len + 1]")
    if (a < 0).any():
        raise ValueError("export_token_sds: negative token id")
    n, s = a.shape[0], a.shape[1] - 1
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQIIQ", SDS_MAGIC, 2, n, s, 1, 1, 0))
        f.write(a.tobytes())
    print(f"wrote {path} ({n} records, seq_len={s}, token-native)")


def _demo(out_dir: str):
    import numpy as np
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    student = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    teacher = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4))
    export_smf(student, f"{out_dir}/model.smf")
    export_smf(teacher, f"{out_dir}/teacher.smf")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((2048, 16), dtype=np.float32)
    y = (x[:, :4].sum(axis=1) > 0).astype(np.int32) + 2 * (x[:, 0] > 0)
    export_sds(x, y, f"{out_dir}/corpus.sds")


def _demo_decoder(out_dir: str):
    """A tiny token-native decoder plus a corpus it can actually learn:
    a cyclic-successor language where token t is always followed by
    (t + 3) % vocab. Needs NumPy only — no PyTorch."""
    import numpy as np

    rng = np.random.default_rng(0)
    vocab, dim, heads, seq, ffn = 50, 32, 4, 8, 64

    def mat(rows, cols):
        return (rng.standard_normal((rows, cols)) * 0.15).astype(np.float32)

    blocks = [{
        "ln1_g": np.ones(dim, np.float32), "wq": mat(dim, dim),
        "wk": mat(dim, dim), "wv": mat(dim, dim), "wo": mat(dim, dim),
        "ln2_g": np.ones(dim, np.float32), "w_gate": mat(dim, ffn),
        "w_up": mat(dim, ffn), "w_down": mat(ffn, dim),
    } for _ in range(2)]
    head = {"lnf_g": np.ones(dim, np.float32), "w_head": mat(dim, vocab)}
    emb = (rng.standard_normal((vocab, dim)) * 0.5).astype(np.float32)
    export_token_decoder_smf(emb, blocks, head, f"{out_dir}/decoder.smf",
                             seq_len=seq, num_heads=heads)

    starts = rng.integers(0, vocab, 192)
    records = (starts[:, None] + 3 * np.arange(seq + 1)) % vocab
    export_token_sds(records.astype(np.int32),
                     f"{out_dir}/decoder_corpus.sds")
    print(f"try: seeml-update-compile --source {out_dir}/decoder.smf"
          f" --data-batch {4 * seq} --loss xent --lora-rank 4"
          " --lora-alpha 8 --optimizer adamw --lr 2e-3 --steps 600")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", metavar="OUT_DIR",
                        help="emit a demo model/teacher/corpus into OUT_DIR")
    parser.add_argument("--demo-decoder", metavar="OUT_DIR",
                        help="emit a token-native demo decoder/corpus "
                             "into OUT_DIR (NumPy only, no PyTorch)")
    args = parser.parse_args()
    if not args.demo and not args.demo_decoder:
        parser.print_help()
        sys.exit(0)
    import os

    if args.demo:
        os.makedirs(args.demo, exist_ok=True)
        _demo(args.demo)
    if args.demo_decoder:
        os.makedirs(args.demo_decoder, exist_ok=True)
        _demo_decoder(args.demo_decoder)
