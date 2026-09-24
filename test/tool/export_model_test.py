"""Tests for tool/export_model.py, the on-ramp.

NumPy-only unless noted (the module is skipped entirely without NumPy, so
the bare-interpreter jobs stay green); the two tests that need PyTorch skip
themselves without it. The oracles are the exporter's original per-row and
whole-blob algorithms, kept here verbatim: the streaming, vectorized writers
must produce the same bytes on every host, on NumPy 1.x and 2.x, and on the
oldest and newest interpreters the tools support.

    python3 -m unittest discover -s test/tool -p '*_test.py'
"""

import contextlib
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



# --- Hugging Face import (T2, #69) ----------------------------------------

def hf_tiny_config(model_type="llama", D=16, H=4, Hkv=2, F=24, L=2, V=20,
                   eps=1e-5, theta=10000.0, tied=True, max_pos=64):
    return {"model_type": model_type, "architectures": ["LlamaForCausalLM"],
            "hidden_size": D, "num_attention_heads": H,
            "num_key_value_heads": Hkv, "intermediate_size": F,
            "num_hidden_layers": L, "vocab_size": V, "rms_norm_eps": eps,
            "rope_theta": theta, "tie_word_embeddings": tied,
            "max_position_embeddings": max_pos, "hidden_act": "silu",
            "attention_bias": model_type == "qwen2", "mlp_bias": False,
            "rope_scaling": None, "torch_dtype": "float32"}


def hf_tiny_tensors(cfg, seed=0, scale=0.2):
    """A random checkpoint in Hugging Face's [out, in] layout and names."""
    rng = np.random.default_rng(seed)
    D, H, Hkv, F, L, V = (cfg[k] for k in ("hidden_size", "num_attention_heads",
                                           "num_key_value_heads",
                                           "intermediate_size",
                                           "num_hidden_layers", "vocab_size"))
    d = D // H
    m = lambda *shape: (rng.standard_normal(shape) * scale).astype(np.float32)
    t = {"model.embed_tokens.weight": m(V, D),
         "model.norm.weight": (1 + m(D)).astype(np.float32)}
    if not cfg["tie_word_embeddings"]:
        t["lm_head.weight"] = m(V, D)
    for i in range(L):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = (1 + m(D)).astype(np.float32)
        t[p + "post_attention_layernorm.weight"] = (1 + m(D)).astype(np.float32)
        t[p + "self_attn.q_proj.weight"] = m(H * d, D)
        t[p + "self_attn.k_proj.weight"] = m(Hkv * d, D)
        t[p + "self_attn.v_proj.weight"] = m(Hkv * d, D)
        t[p + "self_attn.o_proj.weight"] = m(D, H * d)
        if cfg.get("attention_bias"):
            t[p + "self_attn.q_proj.bias"] = m(H * d)
            t[p + "self_attn.k_proj.bias"] = m(Hkv * d)
            t[p + "self_attn.v_proj.bias"] = m(Hkv * d)
        t[p + "mlp.gate_proj.weight"] = m(F, D)
        t[p + "mlp.up_proj.weight"] = m(F, D)
        t[p + "mlp.down_proj.weight"] = m(D, F)
    return t


def hf_reference_logits(cfg, t, tokens):
    """Hugging Face Llama semantics, written independently of the importer:
    [out, in] Linears, rotate-half RoPE on (c, c + d/2), GQA by repeat_kv,
    causal softmax, SwiGLU, RMSNorm at the checkpoint's eps."""
    D, H, Hkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    d = D // H
    B, S = tokens.shape
    eps = cfg["rms_norm_eps"]
    x = t["model.embed_tokens.weight"][tokens]

    def rms(v, g):
        return v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + eps) * g

    inv = cfg["rope_theta"] ** (-np.arange(0, d, 2, dtype=np.float64) / d)
    ang = np.arange(S)[:, None] * inv[None, :]           # [S, d/2]
    cos = np.cos(np.concatenate([ang, ang], -1)).astype(np.float32)  # [S, d]
    sin = np.sin(np.concatenate([ang, ang], -1)).astype(np.float32)

    def rot_half(v):  # [B, h, S, d]
        v1, v2 = v[..., : d // 2], v[..., d // 2:]
        return np.concatenate([-v2, v1], -1)

    mask = np.triu(np.ones((S, S), bool), 1)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        n1 = rms(x, t[p + "input_layernorm.weight"])
        lin = lambda name, v: v @ t[p + name + ".weight"].T + t.get(p + name + ".bias", 0)
        q = lin("self_attn.q_proj", n1).reshape(B, S, H, d).transpose(0, 2, 1, 3)
        k = lin("self_attn.k_proj", n1).reshape(B, S, Hkv, d).transpose(0, 2, 1, 3)
        v = lin("self_attn.v_proj", n1).reshape(B, S, Hkv, d).transpose(0, 2, 1, 3)
        q = q * cos + rot_half(q) * sin
        k = k * cos + rot_half(k) * sin
        k = np.repeat(k, H // Hkv, axis=1)
        v = np.repeat(v, H // Hkv, axis=1)
        sc = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d)
        sc = np.where(mask, -np.inf, sc)
        sc = np.exp(sc - sc.max(-1, keepdims=True))
        pr = sc / sc.sum(-1, keepdims=True)
        a = (pr @ v).transpose(0, 2, 1, 3).reshape(B, S, H * d)
        x = x + a @ t[p + "self_attn.o_proj.weight"].T
        n2 = rms(x, t[p + "post_attention_layernorm.weight"])
        g = n2 @ t[p + "mlp.gate_proj.weight"].T
        u = n2 @ t[p + "mlp.up_proj.weight"].T
        x = x + ((g / (1 + np.exp(-g))) * u) @ t[p + "mlp.down_proj.weight"].T
    nf = rms(x, t["model.norm.weight"])
    head = t.get("lm_head.weight", t["model.embed_tokens.weight"])
    return (nf @ head.T).astype(np.float32)


def write_safetensors(path, tensors, dtype="F32"):
    """A minimal safetensors writer (the format the importer parses)."""
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        if dtype == "BF16":
            raw = (a.view(np.uint32) >> 16).astype("<u2").tobytes()
        elif dtype == "F16":
            raw = a.astype("<f2").tobytes()
        else:
            raw = a.astype("<f4").tobytes()
        header[name] = {"dtype": dtype, "shape": list(a.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


def write_hf_dir(d, cfg, tensors, dtype="F32"):
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    write_safetensors(os.path.join(d, "model.safetensors"), tensors, dtype)


@unittest.skipIf(np is None, "NumPy not installed")
class HfImportTest(unittest.TestCase):
    def test_safetensors_reader_widens_every_dtype(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        rng = np.random.default_rng(3)
        want = {"a": rng.standard_normal((3, 5)).astype(np.float32),
                "b": np.arange(7, dtype=np.float32)}
        for dtype in ("F32", "BF16", "F16"):
            path = os.path.join(d, dtype + ".safetensors")
            write_safetensors(path, want, dtype)
            got = em.read_safetensors(path)
            self.assertEqual(set(got), set(want))
            for k in want:
                self.assertEqual(got[k].dtype, np.float32)
                self.assertEqual(got[k].shape, want[k].shape)
                tol = {"F32": 0.0, "BF16": 1e-2, "F16": 1e-3}[dtype]
                np.testing.assert_allclose(got[k], want[k], rtol=tol, atol=tol)
        # BF16 round trip of bf16-representable values is exact.
        exact = {"c": (np.arange(-8, 8, dtype=np.float32) * 0.25)}
        write_safetensors(os.path.join(d, "x.safetensors"), exact, "BF16")
        np.testing.assert_array_equal(
            em.read_safetensors(os.path.join(d, "x.safetensors"))["c"],
            exact["c"])

    def _check_mapping(self, cfg, seed):
        t = hf_tiny_tensors(cfg, seed)
        conv = em.hf_llama_to_seeml(cfg, t, seq_len=8)
        rng = np.random.default_rng(seed + 100)
        tokens = rng.integers(0, cfg["vocab_size"], (2, 8))
        want = hf_reference_logits(cfg, t, tokens)
        got = em.reference_decoder_logits(conv["embedding"], conv["blocks"],
                                          conv["head"], conv["num_heads"],
                                          conv["rope_base"], tokens,
                                          norm_eps=conv["norm_eps"])
        self.assertEqual(got.shape, want.shape)
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)
        return conv

    def test_gqa_repetition_and_rope_permutation_reproduce_hf_logits(self):
        # GQA (4 heads over 2 kv heads), tied head, theta 10000.
        conv = self._check_mapping(hf_tiny_config(), 1)
        self.assertTrue(any("GQA" in n for n in conv["notes"]))
        self.assertTrue(any("tied" in n for n in conv["notes"]))
        # MHA (H == Hkv), untied head, a Llama-3-style base.
        conv = self._check_mapping(hf_tiny_config(H=4, Hkv=4, tied=False,
                                                  theta=500000.0), 2)
        self.assertFalse(any("GQA" in n for n in conv["notes"]))
        self.assertEqual(conv["rope_base"], 500000.0)
        # Qwen2-class: q/k/v biases ride the permutation and the repetition.
        conv = self._check_mapping(hf_tiny_config(model_type="qwen2"), 3)
        self.assertIn("bq", conv["blocks"][0])
        self.assertIn("bk", conv["blocks"][0])

    def test_refuses_what_the_runtime_cannot_compute(self):
        cfg = hf_tiny_config()
        t = hf_tiny_tensors(cfg)
        # A Qwen2-class epsilon is carried now (P7, #96), not refused, and
        # the SeeAI forward at that epsilon is the checkpoint's forward.
        conv = self._check_mapping(hf_tiny_config(eps=1e-6), 4)
        self.assertEqual(conv["norm_eps"], 1e-6)
        self.assertTrue(any("carried" in n for n in conv["notes"]))
        with self.assertRaisesRegex(ValueError, "epsilon"):
            em.hf_llama_to_seeml(hf_tiny_config(eps=0.0), t, 8)
        with self.assertRaisesRegex(ValueError, "max_position_embeddings"):
            em.hf_llama_to_seeml(cfg, t, 65)
        with self.assertRaisesRegex(ValueError, "hidden_act"):
            em.hf_llama_to_seeml(dict(cfg, hidden_act="gelu"), t, 8)
        with self.assertRaisesRegex(ValueError, "rope_scaling"):
            em.hf_llama_to_seeml(dict(cfg, rope_scaling={"type": "linear"}), t, 8)
        with self.assertRaisesRegex(ValueError, "model_type"):
            em.hf_llama_to_seeml(dict(cfg, model_type="gpt2"), t, 8)
        with self.assertRaisesRegex(ValueError, "num_key_value_heads"):
            em.hf_llama_to_seeml(dict(cfg, num_key_value_heads=3), t, 8)
        with self.assertRaisesRegex(ValueError, "lacks tensor"):
            em.hf_llama_to_seeml(cfg, {k: v for k, v in t.items()
                                       if "o_proj" not in k}, 8)

    def test_cli_imports_a_checkpoint_directory(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        cfg = hf_tiny_config()
        write_hf_dir(d, cfg, hf_tiny_tensors(cfg, 5), "BF16")
        script = os.path.join(REPO, "tool", "export_model.py")
        out = os.path.join(d, "m.smf")
        # --hf needs --seq-len; --rope-base cannot apply to --hf.
        r = subprocess.run([sys.executable, script, "--hf", d, out],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)
        r = subprocess.run([sys.executable, script, "--hf", d, out,
                            "--seq-len", "8", "--rope-base", "1"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)
        r = subprocess.run([sys.executable, script, "--hf", d, out,
                            "--seq-len", "8"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("token-native", r.stdout)
        self.assertIn("GQA", r.stderr)
        digest = sha256(out)
        # Deterministic: the same checkpoint exports the same bytes.
        subprocess.run([sys.executable, script, "--hf", d, out,
                        "--seq-len", "8"], check=True, capture_output=True)
        self.assertEqual(sha256(out), digest)
        # The SMF is one the compiler accepts (when a build is present).
        compiler = os.path.join(REPO, "build", "seeml-update-compile")
        if os.path.exists(compiler):
            r = subprocess.run([compiler, "--source", out, "--out",
                                os.path.join(d, "pkg"), "--data-batch", "16",
                                "--steps", "2"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
        # A 1e-6 epsilon exports as SMF v6 with the epsilon on every norm
        # (P7, #96) — no flag needed; the old flag is accepted and ignored.
        write_hf_dir(d, hf_tiny_config(eps=1e-6), hf_tiny_tensors(cfg, 5))
        r = subprocess.run([sys.executable, script, "--hf", d, out,
                            "--seq-len", "8"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("rms_norm_eps", r.stderr)
        with open(out, "rb") as f:
            self.assertEqual(struct.unpack_from("<I", f.read(8), 4)[0], 6)
        r = subprocess.run([sys.executable, script, "--hf", d, out,
                            "--seq-len", "8", "--allow-eps-drift"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


def have_transformers():
    try:
        import transformers  # noqa: F401
        return have_torch()
    except ImportError:
        return False


@unittest.skipIf(np is None or not have_transformers(),
                 "transformers not installed")
class HfParityTest(unittest.TestCase):
    def test_numpy_reference_matches_transformers_on_a_synthetic_llama(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        cfg = hf_tiny_config(D=32, H=4, Hkv=2, F=48, L=2, V=40)
        write_hf_dir(d, cfg, hf_tiny_tensors(cfg, 9))
        config, tensors = em.load_hf_checkpoint(d)
        conv = em.hf_llama_to_seeml(config, tensors, seq_len=16)
        delta, scale = em.hf_parity(d, conv)
        self.assertLess(delta, 1e-4 * (1.0 + scale))


@unittest.skipIf(np is None or not have_torch(), "PyTorch not installed")
class TorchExportTest(unittest.TestCase):
    def test_sequential_export_matches_the_oracle_and_runs_under_no_grad(self):
        import torch
        import torch.nn as nn
        torch.manual_seed(5)
        model = nn.Sequential(nn.Linear(6, 5), nn.GELU(approximate="tanh"),
                              nn.LayerNorm(5),
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

    def test_an_erf_gelu_is_refused_and_a_tanh_one_is_the_same_bytes(self):
        # P7 (#96): the device's GELU is the tanh approximation. torch's
        # default nn.GELU() is erf — exporting it silently would make step
        # 0 a different function in every MLP layer.
        import torch
        import torch.nn as nn
        d_ = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d_, True)

        def model(gelu):
            torch.manual_seed(9)
            return nn.Sequential(nn.Linear(4, 6), gelu, nn.Linear(6, 3))

        path = os.path.join(d_, "m.smf")
        with self.assertRaisesRegex(ValueError,
                                    r"approximate='tanh'.*gelu_tanh_ok"):
            em.export_smf(model(nn.GELU()), path)
        self.assertFalse(os.path.exists(path))
        em.export_smf(model(nn.GELU(approximate="tanh")), path)
        with open(path, "rb") as f:
            tanh = f.read()
        accepted = os.path.join(d_, "accepted.smf")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            em.export_smf(model(nn.GELU()), accepted, gelu_tanh_ok=True)
        self.assertIn("drift", err.getvalue())
        with open(accepted, "rb") as f:
            self.assertEqual(f.read(), tanh)  # the same file either way
        self.assertEqual(struct.unpack_from("<I", tanh, 4)[0], 3)
        # The documented bound is the real one.
        x = torch.linspace(-12, 12, 200001, dtype=torch.float64)
        drift = (torch.nn.functional.gelu(x) -
                 torch.nn.functional.gelu(x, approximate="tanh")).abs().max()
        self.assertLessEqual(float(drift), em.GELU_TANH_MAX_DRIFT)

    def test_a_layer_norm_epsilon_rides_attr2(self):
        # P7 (#96): nn.LayerNorm.eps is the model's; 1e-5 keeps the file
        # at v3 byte for byte, anything else writes SMF v6 with attr2.
        import torch
        import torch.nn as nn
        d_ = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d_, True)
        files = {}
        for eps in (1e-5, 1e-6):
            torch.manual_seed(3)
            m = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4, eps=eps),
                              nn.Linear(4, 2))
            path = os.path.join(d_, f"ln{eps}.smf")
            em.export_smf(m, path)
            with open(path, "rb") as f:
                files[eps] = f.read()
        self.assertEqual(struct.unpack_from("<I", files[1e-5], 4)[0], 3)
        self.assertEqual(struct.unpack_from("<I", files[1e-6], 4)[0], 6)
        bits = struct.unpack("<I", struct.pack("<f", 1e-6))[0]
        self.assertIn(struct.pack("<I", bits), files[1e-6])

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
