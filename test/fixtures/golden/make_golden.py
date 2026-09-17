#!/usr/bin/env python3
"""Writes the golden cross-plane fixtures: files the Python plane produces
and the C++ core must read, value for value.

    python3 test/fixtures/golden/make_golden.py [out_dir]

  mlp.smf      a 4 -> 3 linear layer (MatMul + AddBias), SMF v3
  class.sds    5 feature rows of width 4 with class labels, SDS v1
  tokens.sds   3 token records of 4 + 1 ids, SDS v2

Every value is a small dyadic rational, exact in float32, so the C++ suite
(test/compiler/frontend/golden_fixture_test.cc) can state them as literals.
The committed copies are byte-compared against a fresh run of this script
by test/tool/formats_test.py: an exporter change that moves a byte, or a
reader change that stops accepting one, fails in the plane that moved.
NumPy only — the exporter's corpus writers need it; nothing here needs torch.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tool"))
import export_model as em  # noqa: E402

IN_DIM, OUT_DIM, ROWS = 4, 3, 5
RECORDS, SEQ, VOCAB = 3, 4, 7


def weight(i, j):
    return (i * OUT_DIM + j - 5) / 8.0


BIAS = (0.5, -0.25, 0.125)


def feature(n, d):
    return (n * IN_DIM + d) / 16.0


def token(r, k):
    return (r * 5 + k) % VOCAB


def write_all(out_dir):
    import numpy as np

    b = em._SmfBuilder("x", IN_DIM)
    w = np.array([[weight(i, j) for j in range(OUT_DIM)]
                  for i in range(IN_DIM)], dtype="<f4")
    b.add_tensor("w0", [IN_DIM, OUT_DIM], w)
    b.add_op(em.OP_MATMUL, "mm0", ["x", "w0"], "z0")
    b.add_tensor("b0", [OUT_DIM], np.array(BIAS, dtype="<f4"))
    b.add_op(em.OP_ADDBIAS, "ab0", ["z0", "b0"], "zb0")
    with open(os.path.join(out_dir, "mlp.smf"), "wb") as f:
        b.write(f)

    x = np.array([[feature(n, d) for d in range(IN_DIM)]
                  for n in range(ROWS)], dtype="<f4")
    y = np.array([n % OUT_DIM for n in range(ROWS)], dtype="<i4")
    em.export_sds(x, y, os.path.join(out_dir, "class.sds"))

    records = np.array([[token(r, k) for k in range(SEQ + 1)]
                        for r in range(RECORDS)], dtype="<i4")
    em.export_token_sds(records, os.path.join(out_dir, "tokens.sds"))


if __name__ == "__main__":
    write_all(sys.argv[1] if len(sys.argv) > 1 else HERE)
