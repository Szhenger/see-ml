"""Tests for tool/export_model.py, the on-ramp.

NumPy-only unless noted (the module is skipped entirely without NumPy, so
the bare-interpreter jobs stay green); the two tests that need PyTorch skip
themselves without it. The oracles are the exporter's original per-row and
whole-blob algorithms, kept here verbatim: the streaming, vectorized writers
must produce the same bytes on every host, on NumPy 1.x and 2.x, and on the
oldest and newest interpreters the tools support.

    python3 -m unittest discover -s test/tool -p '*_test.py'
"""

import hashlib
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tool"))

try:
    import numpy as np
except ImportError:  # pragma: no cover — bare interpreter
    np = None

if np is not None:
    import export_model as em

DIGESTS = os.path.join(os.path.dirname(__file__), "demo_digests.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def have_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# --- The oracles: the pre-streaming algorithms, verbatim. --------------------

def ref_sds(x, lab, label_kind, label_dim):
    out = io.BytesIO()
    n, d = x.shape
    out.write(struct.pack("<IIQQIIQ", em.SDS_MAGIC, 1, n, d, label_kind, 0,
                          label_dim))
    for i in range(n):
        out.write(x[i].tobytes())
        if lab is not None:
            out.write(lab[i].tobytes())
    return out.getvalue()


def ref_serialize(b):
    """The original _SmfBuilder.serialize: assemble the whole blob in RAM."""
    version = b.version

    def meta(offsets):
        out = struct.pack("<IIII", em.SMF_MAGIC, version, len(b.tensors),
                          len(b.ops))
        out += em._s(b.input_name) + em._s(b.output_name)
        out += struct.pack("<Q", b.seq_len)
        for t in b.tensors:
            data = bytes(memoryview(t["data"]))
            out += em._s(t["name"])
            out += struct.pack("<BB", len(t["dims"]), 1 if t["const"] else 0)
            for d in t["dims"]:
                out += struct.pack("<q", d)
            out += struct.pack("<QQ", offsets.get(t["name"], 0), len(data))
        for op in b.ops:
            out += struct.pack("<B", op["kind"]) + em._s(op["name"])
            out += struct.pack("<B", len(op["inputs"]))
            for i in op["inputs"]:
                out += em._s(i)
            out += em._s(op["output"])
            out += struct.pack("<I", op["attr0"])
            if version >= 5:
                out += struct.pack("<I", op["attr1"])
        return out

    cursor, offsets = em._align(len(meta({}))), {}
    for t in b.tensors:
        if not t["const"]:
            continue
        offsets[t["name"]] = cursor
        cursor = em._align(cursor + memoryview(t["data"]).nbytes)
    blob = bytearray(meta(offsets))
    blob.extend(b"\x00" * (cursor - len(blob)))
    for t in b.tensors:
        if t["const"]:
            data = bytes(memoryview(t["data"]))
            o = offsets[t["name"]]
            blob[o:o + len(data)] = data
    return bytes(blob)


def demo_blocks(rng, dim=8, ffn=16, n=2):
    def mat(r, c):
        return (rng.standard_normal((r, c)) * 0.1).astype(np.float32)
    blocks = [{
        "ln1_g": np.ones(dim, np.float32), "wq": mat(dim, dim),
        "wk": mat(dim, dim), "wv": mat(dim, dim), "wo": mat(dim, dim),
        "ln2_g": np.ones(dim, np.float32), "w_gate": mat(dim, ffn),
        "w_up": mat(dim, ffn), "w_down": mat(ffn, dim)} for _ in range(n)]
    head = {"lnf_g": np.ones(dim, np.float32), "w_head": mat(dim, 11)}
    return blocks, head


@unittest.skipIf(np is None, "NumPy not installed")
class FeatureCorpusTest(unittest.TestCase):
    """export_sds streams chunks of a packed structured array; the oracle
    writes row by row. Sizes straddle the chunk boundary."""

    def check(self, n, d, kind, label_dim=0):
        rng = np.random.default_rng(n * 7 + d)
        x = rng.standard_normal((n, d), dtype=np.float32)
        if kind == 0:
            lab_in, lab_ref = None, None
        elif kind == 1:
            lab_in = (rng.integers(0, 5, n)).astype(np.int64)  # int64 in
            lab_ref = lab_in.astype(np.int32)
        else:
            lab_in = rng.standard_normal((n, label_dim)).astype(np.float32)
            lab_ref = lab_in
        d_ = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d_, True)
        path = os.path.join(d_, "c.sds")
        em.export_sds(x, lab_in, path, label_kind=kind)
        with open(path, "rb") as f:
            got = f.read()
        self.assertEqual(got, ref_sds(x, lab_ref, kind, label_dim))

    def test_class_labels_across_chunk_sizes(self):
        for n in (1, 7, em.SDS_CHUNK_ROWS - 1, em.SDS_CHUNK_ROWS,
                  em.SDS_CHUNK_ROWS + 3, 3 * em.SDS_CHUNK_ROWS + 5):
            self.check(n, 3, 1)

    def test_dense_labels(self):
        for n in (1, 5, em.SDS_CHUNK_ROWS + 1):
            self.check(n, 4, 2, label_dim=2)
        self.check(9, 4, 2, label_dim=1)

    def test_unlabeled(self):
        for n in (1, em.SDS_CHUNK_ROWS + 2):
            self.check(n, 16, 0)

    def test_non_contiguous_and_float64_inputs_are_converted(self):
        rng = np.random.default_rng(3)
        wide = rng.standard_normal((50, 8))          # float64
        x = wide[:, ::2]                              # non-contiguous view
        d_ = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d_, True)
        path = os.path.join(d_, "c.sds")
        em.export_sds(x, np.arange(50), path)
        with open(path, "rb") as f:
            got = f.read()
        self.assertEqual(got, ref_sds(np.ascontiguousarray(x, dtype=np.float32),
                                      np.arange(50, dtype=np.int32), 1, 0))

    def test_rejects_the_wrong_rank_and_float_class_labels(self):
        d_ = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d_, True)
        with self.assertRaisesRegex(ValueError, r"\[N, D\]"):
            em.export_sds(np.zeros(5, np.float32), None, os.path.join(d_, "x"))
        with self.assertRaisesRegex(ValueError, "integer class labels"):
            em.export_sds(np.zeros((5, 2), np.float32),
                          np.zeros(5, np.float32), os.path.join(d_, "x"))


@unittest.skipIf(np is None, "NumPy not installed")
class TokenCorpusTest(unittest.TestCase):
    def test_matches_the_flat_int32_layout(self):
        recs = (np.arange(9 * 70).reshape(70, 9) % 50).astype(np.int64)
        d_ = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d_, True)
        path = os.path.join(d_, "t.sds")
        em.export_token_sds(recs, path)
        with open(path, "rb") as f:
            got = f.read()
        want = struct.pack("<IIQQIIQ", em.SDS_MAGIC, 2, 70, 8, 1, 1, 0)
        want += recs.astype(np.int32).tobytes()
        self.assertEqual(got, want)


@unittest.skipIf(np is None, "NumPy not installed")
class ContainerTest(unittest.TestCase):
    """The streaming writer against the whole-blob oracle, for a token
    decoder (SMF v4, then v5 with a non-default RoPE base)."""

    def build(self, rope_base):
        blocks, head = demo_blocks(np.random.default_rng(1))
        emb = np.random.default_rng(2).standard_normal((11, 8)).astype(np.float32)
        d_ = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d_, True)
        path = os.path.join(d_, "m.smf")
        em.export_token_decoder_smf(emb, blocks, head, path, seq_len=4,
                                    num_heads=2, rope_base=rope_base)
        with open(path, "rb") as f:
            return f.read(), emb, blocks, head

    def rebuild(self, emb, blocks, head, rope_base):
        b = em._SmfBuilder("x", 8, seq_len=4, token_input=True)
        b.add_tensor("emb", list(emb.shape), emb)
        b.add_op(em.OP_EMBEDDING, "embed", ["x", "emb"], "e")
        em._emit_decoder_graph(b, blocks, head, 2, prev="e", rope_base=rope_base)
        return b

    def test_streamed_file_equals_the_oracle_blob_and_serialize(self):
        for base, version in ((10000.0, 4), (500000.0, 5)):
            got, emb, blocks, head = self.build(base)
            b = self.rebuild(emb, blocks, head, base)
            self.assertEqual(got, ref_serialize(b))
            self.assertEqual(got, b.serialize())
            self.assertEqual(struct.unpack_from("<II", got)[1], version)
            # Every tensor lands 64-byte aligned and the file is padded.
            self.assertEqual(len(got) % em.ALIGN, 0)
            _, offsets, total = b._layout()
            self.assertEqual(total, len(got))
            self.assertTrue(all(o % em.ALIGN == 0 for o in offsets.values()))

    def test_non_contiguous_payloads_are_refused_clearly(self):
        b = em._SmfBuilder("x", 4)
        b.add_tensor("w", [4, 2], np.zeros((4, 4), "<f4")[:, ::2])
        with self.assertRaisesRegex(ValueError, "C-contiguous"):
            b.serialize()

    def test_callable_payloads_must_match_their_declared_size(self):
        b = em._SmfBuilder("x", 4)
        b.add_tensor("w", [2], lambda: np.zeros(3, "<f4"), nbytes=8)
        with self.assertRaisesRegex(ValueError, "declared 8"):
            b.serialize()
        with self.assertRaisesRegex(ValueError, "nbytes"):
            b.add_tensor("v", [2], lambda: np.zeros(2, "<f4"))

    def test_arrays_are_written_without_a_copy(self):
        b = em._SmfBuilder("x", 4)
        a = np.arange(8, dtype="<f4").reshape(2, 4)
        b.add_tensor("w", [2, 4], a)
        b.add_op(em.OP_MATMUL, "mm", ["x", "w"], "z")
        first = b.serialize()
        a[0, 0] = 42.0  # the builder holds the array itself, not bytes of it
        self.assertNotEqual(first, b.serialize())
        self.assertEqual(struct.unpack_from("<f", b.serialize(),
                                            b._layout()[1]["w"])[0], 42.0)


@unittest.skipIf(np is None, "NumPy not installed")
class GoldenDemoTest(unittest.TestCase):
    """The docs promise the default demos byte-for-byte; the digests in
    demo_digests.json make that a test. The NumPy-only demo is pinned on
    every host; the PyTorch demo's weights come from torch's CPU RNG, so
    they are pinned per torch release (skipped on any other)."""

    def run_export(self, *args):
        d_ = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d_, True)
        subprocess.run([sys.executable, os.path.join(REPO, "tool", "export_model.py"),
                        *args, d_], check=True, capture_output=True)
        return d_

    def test_demo_decoder_defaults(self):
        with open(DIGESTS) as f:
            want = json.load(f)["--demo-decoder"]
        out = self.run_export("--demo-decoder")
        for name, digest in want.items():
            self.assertEqual(sha256(os.path.join(out, name)), digest, name)

    def test_demo_corpus_default(self):
        """corpus.sds is NumPy-only even in the torch demo, but the demo
        needs torch to write the models first."""
        if not have_torch():
            self.skipTest("PyTorch not installed")
        with open(DIGESTS) as f:
            want = json.load(f)["--demo"]
        out = self.run_export("--demo")
        self.assertEqual(sha256(os.path.join(out, "corpus.sds")),
                         want["corpus.sds"])
        import torch
        release = torch.__version__.split("+")[0]  # "2.14.0+cpu" -> "2.14.0"
        pinned = want["models"].get(release)
        if pinned is None:
            self.skipTest(f"model digests not pinned for torch {release}")
        for name, digest in pinned.items():
            self.assertEqual(sha256(os.path.join(out, name)), digest, name)


@unittest.skipIf(np is None or not have_torch(), "PyTorch not installed")
class TorchExportTest(unittest.TestCase):
    def test_sequential_export_matches_the_oracle_and_runs_under_no_grad(self):
        import torch
        import torch.nn as nn
        torch.manual_seed(5)
        model = nn.Sequential(nn.Linear(6, 5), nn.GELU(), nn.LayerNorm(5),
                              nn.Linear(5, 3, bias=False), nn.SiLU(),
                              nn.LayerNorm(3, elementwise_affine=False),
                              nn.Linear(3, 2))
        d_ = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d_, True)
        path = os.path.join(d_, "m.smf")
        em.export_smf(model, path)
        with open(path, "rb") as f:
            got = f.read()
        # Rebuild by hand with the original bytes() path as the oracle.
        b = em._SmfBuilder("x", 6)
        prev, idx = "x", 0
        for pos, m in enumerate(model):
            if isinstance(m, nn.Linear):
                w = m.weight.detach().t().contiguous().float()
                b.add_tensor(f"w{idx}", list(w.shape), w.numpy().tobytes())
                b.add_op(em.OP_MATMUL, f"mm{idx}", [prev, f"w{idx}"], f"z{idx}")
                if m.bias is not None:
                    bias = m.bias.detach().float()
                    b.add_tensor(f"b{idx}", [bias.numel()], bias.numpy().tobytes())
                    b.add_op(em.OP_ADDBIAS, f"ab{idx}", [f"z{idx}", f"b{idx}"], f"zb{idx}")
                    prev = f"zb{idx}"
                else:
                    prev = f"z{idx}"
                idx += 1
            elif isinstance(m, (nn.ReLU, nn.GELU, nn.SiLU)):
                kind = (em.OP_RELU if isinstance(m, nn.ReLU)
                        else em.OP_GELU if isinstance(m, nn.GELU) else em.OP_SILU)
                b.add_op(kind, f"act{pos}", [prev], f"h{pos}")
                prev = f"h{pos}"
            else:
                d = m.normalized_shape[0]
                gamma = m.weight.detach().float() if m.weight is not None else torch.ones(d)
                beta = m.bias.detach().float() if m.bias is not None else torch.zeros(d)
                b.add_tensor(f"ln_g{pos}", [d], gamma.numpy().tobytes())
                b.add_tensor(f"ln_b{pos}", [d], beta.numpy().tobytes())
                b.add_op(em.OP_LAYERNORM, f"ln{pos}", [prev, f"ln_g{pos}", f"ln_b{pos}"], f"n{pos}")
                prev = f"n{pos}"
        self.assertEqual(got, ref_serialize(b))

    def test_half_precision_weights_export_as_f32(self):
        import torch
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(4, 3)).to(torch.bfloat16)
        d_ = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d_, True)
        path = os.path.join(d_, "m.smf")
        em.export_smf(model, path)
        w = model[0].weight.detach().float().t().contiguous().numpy().tobytes()
        with open(path, "rb") as f:
            self.assertIn(w, f.read())


if __name__ == "__main__":
    unittest.main()
