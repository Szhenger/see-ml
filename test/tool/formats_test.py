#!/usr/bin/env python3
"""The byte seam between the planes, held by machine (P6, #86).

  * tool/seeml/formats.py agrees with the committed C++ manifest
    (tool/seeml/abi.json), field by field;
  * the committed manifest is what the built `seeml-abi` prints today — a
    header edit that skipped the regenerate step fails here and in CI;
  * the golden fixtures are what the exporter writes today, byte for byte
    (the C++ half reads the same files: golden_fixture_test.cc);
  * the versions the documentation states are the manifest's;
  * the C++ decode of a plan (`seeml-seeu-dump --json`) and the Python
    decode (frontier_exec.Plan) are the same decode, for every header
    field, instruction and emit entry.

Run: python3 -m unittest discover -s test/tool -p '*_test.py'
"""

import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tool"))
sys.path.insert(0, os.path.join(REPO, "test", "fixtures", "golden"))

from seeml import formats  # noqa: E402
import frontier_exec as fx  # noqa: E402

try:
    import numpy as np
except ImportError:  # the stdlib tier still checks the manifest
    np = None

BUILD = os.environ.get("SEEML_BUILD_DIR", os.path.join(REPO, "build"))
ABI_TOOL = os.path.join(BUILD, "seeml-abi")
COMPILER = os.path.join(BUILD, "seeml-update-compile")
DUMP = os.path.join(BUILD, "seeml-seeu-dump")
GOLDEN = os.path.join(REPO, "test", "fixtures", "golden")


class ManifestAgreement(unittest.TestCase):
    def test_python_formats_match_the_committed_manifest(self):
        self.assertEqual(formats.check_against(formats.load_abi()), [])

    def test_the_known_sizes(self):
        self.assertEqual(formats.PLAN_HEADER.size, 280)
        self.assertEqual(formats.INSTRUCTION.size, 64)
        self.assertEqual(formats.EMIT_ENTRY.size, 24)
        self.assertEqual(formats.SDS_HEADER.size, 40)
        self.assertEqual(formats.CKPT_HEADER.size, 40)

    def test_every_kind_of_drift_is_named(self):
        abi = formats.load_abi()
        cases = []
        moved = copy.deepcopy(abi)
        moved["seeu"]["structs"]["PlanHeader"]["fields"][5]["offset"] += 4
        cases.append((moved, "PlanHeader.fields"))
        renamed = copy.deepcopy(abi)
        renamed["seeu"]["opcodes"]["reduce_rows"] = renamed["seeu"][
            "opcodes"].pop("reduce.rows")
        cases.append((renamed, "seeu.opcodes"))
        bumped = copy.deepcopy(abi)
        bumped["sekp"]["version"] += 1
        cases.append((bumped, "sekp.version"))
        grown = copy.deepcopy(abi)
        grown["smf"]["op_kinds"]["conv"] = 12
        cases.append((grown, "smf.op_kinds"))
        for doctored, where in cases:
            problems = formats.check_against(doctored)
            self.assertEqual(len(problems), 1, problems)
            self.assertTrue(problems[0].startswith(where), problems[0])

    def test_the_interpreter_covers_the_opcode_space(self):
        self.assertEqual(sorted(fx.INTERPRETER), sorted(formats.OPCODES))
        self.assertIs(fx.OPCODES, formats.OPCODES)

    @unittest.skipUnless(os.path.exists(ABI_TOOL), "seeml-abi not built")
    def test_the_committed_manifest_is_current(self):
        done = subprocess.run([ABI_TOOL], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        with open(formats.ABI_PATH, "r", encoding="utf-8") as f:
            committed = f.read()
        self.assertEqual(
            done.stdout, committed,
            "tool/seeml/abi.json is stale: run "
            "`build/seeml-abi > tool/seeml/abi.json` and update "
            "tool/seeml/formats.py to match")


class DocumentedVersions(unittest.TestCase):
    """The prose states format versions in two places; both follow the
    manifest (the spec table said SEEU v7 / SEKP v3 six versions late)."""

    def read(self, *parts):
        with open(os.path.join(REPO, *parts), "r", encoding="utf-8") as f:
            return f.read()

    def test_the_format_reference_headings(self):
        text = self.read("docs", "formats.md")
        for heading in (
                f"(`.smf`, v{formats.SMF_VERSION})",
                f"(`.sds`, v{formats.SDS_VERSION})",
                f"(`.seeu`, v{formats.SEEU_VERSION})",
                f"(`SEKP`, v{formats.SEKP_VERSION})"):
            self.assertTrue(heading in text,
                            f"docs/formats.md has no heading {heading!r}")

    def test_the_specification_table(self):
        text = self.read("SPECIFICATION.md")
        for row in (
                f'| `"SMF1"` | v{formats.SMF_VERSION} ',
                f'| `"SEEU"` | v{formats.SEEU_VERSION}, oldest-readable '
                f'v{formats.SEEU_OLDEST_READABLE} |',
                f'| `"SEKP"` | v{formats.SEKP_VERSION}, oldest-readable '
                f'v{formats.SEKP_OLDEST_READABLE} |'):
            self.assertTrue(row in text,
                            f"SPECIFICATION.md has no table row {row!r}")


@unittest.skipIf(np is None, "NumPy not installed")
class GoldenFixtures(unittest.TestCase):
    def test_the_exporter_still_writes_the_committed_bytes(self):
        import make_golden
        with tempfile.TemporaryDirectory(prefix="seeml-golden-") as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                make_golden.write_all(tmp)
            for name in ("mlp.smf", "class.sds", "tokens.sds"):
                with open(os.path.join(tmp, name), "rb") as f:
                    fresh = f.read()
                with open(os.path.join(GOLDEN, name), "rb") as f:
                    committed = f.read()
                self.assertEqual(fresh, committed, name)

    def test_the_corpus_reader_serves_the_golden_rows(self):
        import make_golden as g
        corpus = fx.Corpus(np, os.path.join(GOLDEN, "class.sds"))
        self.assertEqual((corpus.num_samples, corpus.input_dim,
                          corpus.label_kind), (g.ROWS, g.IN_DIM, 1))
        tokens = fx.Corpus(np, os.path.join(GOLDEN, "tokens.sds"))
        self.assertEqual(tokens.tokens.tolist(),
                         [[g.token(r, k) for k in range(g.SEQ + 1)]
                          for r in range(g.RECORDS)])


PLAN_FLAGS = [
    ("sgd_const", ["--optimizer", "sgd", "--loss", "xent"]),
    ("adamw_cosine", ["--lr-schedule", "cosine", "--warmup", "2",
                      "--steps", "12", "--clip-norm", "0.5",
                      "--grad-accum", "2"]),
    ("q8", ["--quantize-base"]),
]


@unittest.skipUnless(os.path.exists(COMPILER) and os.path.exists(DUMP),
                     f"no built seeml tools under {BUILD}")
class TwoDecodersOnePlan(unittest.TestCase):
    """The compiler writes the plan; C++ and Python each decode it."""

    def decode_both(self, flags):
        with tempfile.TemporaryDirectory(prefix="seeml-seam-") as tmp:
            done = subprocess.run(
                [COMPILER, "--source", os.path.join(GOLDEN, "mlp.smf"),
                 "--out", tmp, "--data-batch", "4", "--no-embed"] + flags,
                capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            path = os.path.join(tmp, "update_plan.seeu")
            dumped = subprocess.run([DUMP, path, "--json"],
                                    capture_output=True, text=True)
            self.assertEqual(dumped.returncode, 0, dumped.stderr)
            with open(path, "rb") as f:
                return json.loads(dumped.stdout), fx.Plan(f.read(), path)

    def test_header_instructions_and_emit_table_agree(self):
        for name, flags in PLAN_FLAGS:
            with self.subTest(plan=name):
                cpp, plan = self.decode_both(flags)
                floats = 0
                for field, code in zip(formats.PLAN_HEADER.names,
                                       formats.PLAN_HEADER.codes):
                    mine = getattr(plan, field)
                    if code == "f":
                        floats += 1
                        self.assertEqual(fx.f32_bits(mine),
                                         cpp["header"][field + "_bits"], field)
                    else:
                        self.assertEqual(mine, cpp["header"][field], field)
                self.assertEqual(floats, 7)
                self.assertEqual(sorted(cpp["sections"]),
                                 sorted(fx.SECTIONS))
                for section in fx.SECTIONS:
                    theirs = cpp["sections"][section]
                    ours = plan.sections[section]
                    self.assertEqual(len(ours), len(theirs), section)
                    for a, b in zip(ours, theirs):
                        self.assertEqual(
                            (a.opcode, a.flags, list(a.src), list(a.out),
                             formats.OPCODES[a.opcode]),
                            (b["opcode"], b["flags"], b["in"], b["out"],
                             b["name"]))
                self.assertEqual(len(cpp["emit"]), plan.emit_count)
                for i, entry in enumerate(cpp["emit"]):
                    got = formats.EMIT_ENTRY.unpack_from(
                        plan.blob, plan.emit_table_offset +
                        i * formats.EMIT_ENTRY.size)
                    self.assertEqual(
                        dict(zip(formats.EMIT_ENTRY.names, got)), entry)

    def test_a_broken_seal_yields_no_decode(self):
        with tempfile.TemporaryDirectory(prefix="seeml-seam-") as tmp:
            done = subprocess.run(
                [COMPILER, "--source", os.path.join(GOLDEN, "mlp.smf"),
                 "--out", tmp, "--data-batch", "4", "--no-embed"], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            path = os.path.join(tmp, "update_plan.seeu")
            with open(path, "r+b") as f:
                f.seek(formats.PLAN_HEADER.offsets["batch"])
                f.write((5).to_bytes(8, "little"))
            dumped = subprocess.run([DUMP, path, "--json"],
                                    capture_output=True, text=True)
            # Fail closed: no decode of bytes the seal does not vouch for.
            self.assertEqual(dumped.returncode, 1)
            self.assertEqual(dumped.stdout, "")


if __name__ == "__main__":
    unittest.main()
