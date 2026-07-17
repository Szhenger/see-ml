#!/usr/bin/env python3
"""Export a PyTorch MLP and a dataset into SeeML's SMF / SDS formats.

SMF is the model container consumed by seeml-update-compile (source and
teacher models); SDS is the fixed-shape dataset streamed by the compiled
`model_update` executable on-device.

Usage:
    python3 export_model.py --demo out_dir/
        Writes a demo model.smf, teacher.smf and corpus.sds for a smoke run.

    from export_model import export_smf, export_sds
        export_smf(torch_sequential, "model.smf")
        export_sds(inputs, labels, "corpus.sds")

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
SMF_VERSION = 2         # v2: Gelu/Silu/Mul/LayerNorm op kinds
SDS_MAGIC = 0x31534453  # "SDS1"
ALIGN = 64

OP_MATMUL, OP_ADDBIAS, OP_RELU, OP_GELU, OP_SILU, OP_MUL, OP_LAYERNORM = range(7)


def _align(n: int) -> int:
    return (n + ALIGN - 1) & ~(ALIGN - 1)


def _s(name: str) -> bytes:
    b = name.encode()
    return struct.pack("<H", len(b)) + b


class _SmfBuilder:
    def __init__(self, input_name: str, input_dim: int):
        self.input_name = input_name
        self.output_name = input_name
        self.tensors = [
            dict(name=input_name, dims=[-1, input_dim], const=False, data=b"")
        ]
        self.ops = []

    def add_tensor(self, name, dims, data: bytes):
        self.tensors.append(dict(name=name, dims=list(dims), const=True, data=data))

    def add_op(self, kind, name, inputs, output):
        self.ops.append(dict(kind=kind, name=name, inputs=inputs, output=output))
        self.output_name = output

    def serialize(self) -> bytes:
        def meta(offsets):
            out = struct.pack("<IIII", SMF_MAGIC, SMF_VERSION,
                              len(self.tensors), len(self.ops))
            out += _s(self.input_name) + _s(self.output_name)
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
    import torch.nn as nn

    linears = [m for m in model if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("export_smf: model contains no nn.Linear layers")

    b = _SmfBuilder(input_name, linears[0].in_features)
    prev, idx = input_name, 0
    for pos, m in enumerate(model):
        if isinstance(m, nn.Linear):
            w = m.weight.detach().t().contiguous().float()  # [in, out]
            bias = m.bias.detach().float()
            b.add_tensor(f"w{idx}", list(w.shape), w.numpy().tobytes())
            b.add_tensor(f"b{idx}", [bias.numel()], bias.numpy().tobytes())
            b.add_op(OP_MATMUL, f"mm{idx}", [prev, f"w{idx}"], f"z{idx}")
            b.add_op(OP_ADDBIAS, f"ab{idx}", [f"z{idx}", f"b{idx}"], f"zb{idx}")
            prev = f"zb{idx}"
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
            gamma = m.weight.detach().float()
            beta = m.bias.detach().float()
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


def export_sds(inputs, labels, path: str, label_kind: int = 1):
    """inputs: float32 array [N, D]; labels: int32 [N] (kind 1),
    float32 [N, L] (kind 2), or None (kind 0, distillation corpora)."""
    import numpy as np

    x = np.asarray(inputs, dtype=np.float32)
    n, d = x.shape
    if labels is None:
        label_kind, label_dim, lab = 0, 0, None
    elif label_kind == 1:
        lab = np.asarray(labels, dtype=np.int32).reshape(n)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", metavar="OUT_DIR",
                        help="emit a demo model/teacher/corpus into OUT_DIR")
    args = parser.parse_args()
    if not args.demo:
        parser.print_help()
        sys.exit(0)
    import os

    os.makedirs(args.demo, exist_ok=True)
    _demo(args.demo)
