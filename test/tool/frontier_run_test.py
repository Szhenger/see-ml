"""tool/frontier_run.py — the frontier harness (F1, #129).

What is held here without a framework or a checkpoint (tier 0 / 1):
  * the adapted-set rule: the adapter parameter count every row must show,
    for the frameworks and for SeeML (whose k/v repeat differs), and that a
    wrong rank is refused loudly;
  * the parity checks: different records, different tokens per step, a
    step-0 validation loss that disagrees among the f32 / bf16 rows — each a
    refusal — while an int8-base row is exempt from the loss rule;
  * the split rule, byte for byte the feeder's (tail floor(0.1 n), clamped);
  * the report's shape and the CLI's strictness.
The records rule is checked against a real tokenizer when `tokenizers` is
installed (tier 2) and skipped otherwise.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tool"))

import frontier_run as fr  # noqa: E402

try:
    import numpy as np
except ImportError:  # the bare-interpreter tier
    np = None

SMOLLM = {"layers": 30, "dim": 576, "heads": 9, "kv_heads": 3, "ffn": 1536,
          "vocab": 49152, "tied": True}
QWEN = {"layers": 24, "dim": 896, "heads": 14, "kv_heads": 2, "ffn": 4864,
        "vocab": 151936, "tied": True}


def a_row(system, precision, **over):
    r = {"system": system, "backend": "gpu", "precision": precision,
         "records_digest": "abc", "tokens_per_step": 512,
         "val_loss": [4.5744, 4.0], "adapter_params": None}
    r.update(over)
    return r


class AdaptedSetTest(unittest.TestCase):
    def test_the_counts_the_2026_09_22_harness_measured(self):
        # SmolLM-135M at r8: torch / MLX 2,840,064; SeeML 3,024,384 (k/v
        # repeated 3 -> 9 heads). Qwen2.5-0.5B: 5,621,760 / 5,916,672.
        self.assertEqual(fr.expected_adapter_params(SMOLLM, 8), 2840064)
        self.assertEqual(fr.expected_adapter_params(SMOLLM, 8, seeml=True), 3024384)
        self.assertEqual(fr.expected_adapter_params(QWEN, 8), 5621760)
        self.assertEqual(fr.expected_adapter_params(QWEN, 8, seeml=True), 5916672)

    def test_a_wrong_rank_is_refused_by_name(self):
        rows = [a_row("torch.compile", "f32", adapter_params=2840064),
                a_row("SeeML", "f32 math, int8 base",
                      adapter_params=fr.expected_adapter_params(SMOLLM, 4, True))]
        problems = fr.parity_problems(rows, SMOLLM, 8)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("SeeML", problems[0])
        self.assertIn("expected 3024384 at rank 8", problems[0])
        self.assertEqual(fr.parity_problems(rows[:1], SMOLLM, 8), [])


class ParityTest(unittest.TestCase):
    def test_one_comparison_passes(self):
        rows = [a_row("torch.compile", "f32"), a_row("MLX-LM LoRA", "bf16",
                                                     val_loss=[4.5570, 4.04]),
                a_row("SeeML", "f32 math, int8 base", val_loss=[4.7024, 4.06])]
        self.assertEqual(fr.parity_problems(rows), [])

    def test_each_rule_refuses_alone(self):
        base = [a_row("torch.compile", "f32")]
        self.assertTrue(any("different records" in p for p in fr.parity_problems(
            base + [a_row("MLX-LM LoRA", "bf16", records_digest="xyz")])))
        self.assertTrue(any("tokens per step" in p for p in fr.parity_problems(
            base + [a_row("MLX-LM LoRA", "bf16", tokens_per_step=516)])))
        loss = fr.parity_problems(base + [a_row("MLX-LM LoRA", "f32", val_loss=[4.61, 4.0])])
        self.assertEqual(len(loss), 1, loss)
        self.assertIn("step-0 validation loss 4.6100", loss[0])
        # An int8 base sits above the f32 rows by design: labelled, exempt.
        self.assertEqual(fr.parity_problems(
            base + [a_row("SeeML", "f32 math, int8 base", val_loss=[4.7024, 4.0])]), [])


class SplitTest(unittest.TestCase):
    @unittest.skipIf(np is None, "needs NumPy")
    def test_tail_split_is_the_feeders(self):
        # dataset.cc: val_n = floor(n * 0.1), at least 1, at most n - 1.
        for n, want in ((501, 50), (467, 46), (5, 1), (2, 1)):
            recs = np.arange(n * 3).reshape(n, 3)
            train, val = fr.split_tail(recs, 0.1)
            self.assertEqual(len(val), want)
            self.assertEqual(len(train), n - want)
            self.assertTrue((val == recs[n - want:]).all())


class RecordsTest(unittest.TestCase):
    def test_records_match_the_exporters_rule(self):
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers
        except ImportError:
            self.skipTest("needs tokenizers (tier 2)")
        # A whitespace tokenizer with a tiny vocabulary, written as the
        # tokenizer.json a checkpoint directory carries.
        vocab = {w: i for i, w in enumerate(["[UNK]", "a", "b", "c", "d"])}
        tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        with tempfile.TemporaryDirectory() as d:
            tok.save(os.path.join(d, "tokenizer.json"))
            text = os.path.join(d, "corpus.txt")
            with open(text, "w") as f:
                f.write("a b c d " * 8)  # 32 ids -> 7 records of 4 + remainder
            recs = fr.load_records(d, text, 3)
            self.assertEqual(recs.shape, (8, 4))
            self.assertEqual(recs[0].tolist(), [1, 2, 3, 4])
            self.assertEqual(recs.dtype, np.int64)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(fr.main(["records", d, text, "--seq-len", "3"]), 0)
            self.assertIn("8 records of 4 ids (7 train, 1 held out)", out.getvalue())


class CliTest(unittest.TestCase):
    def test_check_reads_a_matrix_report_and_refuses_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            rep = os.path.join(d, "report.json")
            rows = [a_row("torch.compile", "f32"),
                    a_row("MLX-LM LoRA", "f32", records_digest="other")]
            with open(rep, "w") as f:
                json.dump({"schema": fr.REPORT_SCHEMA, "rows": rows}, f)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(fr.main(["check", rep]), 3)
            self.assertIn("REFUSED", out.getvalue())
            with open(rep, "w") as f:
                json.dump({"schema": fr.REPORT_SCHEMA, "rows": rows[:1]}, f)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(fr.main(["check", rep]), 0)

    def test_unknown_subcommands_and_flags_are_hard_errors(self):
        for argv in (["frontier"], ["torch", "--model-dir", "x"], ["check"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as e:
                    fr.main(argv)
            self.assertEqual(e.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
