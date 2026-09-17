"""Every byte layout and constant the Python plane shares with the C++ core.

The build-host tools read and write four containers the C++ side owns — SMF
models, SDS corpora, SEEU plans, SEKP checkpoints — plus the probe's trace
file. Before this module each script kept its own copy of whichever magics
and struct strings it needed, and nothing but review held them to the
headers. Now there is one Python statement of the seam, and it is checked
by machine: `seeml-abi` prints the same facts from the C++ headers
(sizeof/offsetof, the enums, the version constants), the result is committed
as `abi.json` beside this file, CI regenerates it and fails on a diff, and
test/tool/formats_test.py holds every name below to that manifest.

So a format change is a three-step edit that cannot be half done: the C++
header, this file, and `build/seeml-abi > tool/seeml/abi.json`.

Standard library only — this is the floor of the dependency tiers.
"""

import json
import os
import struct
from typing import Any, Dict, List, Sequence, Tuple

ABI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "abi.json")


def load_abi(path: str = ABI_PATH) -> Dict[str, Any]:
    """The committed C++ manifest (what `seeml-abi` printed)."""
    with open(path, "r", encoding="utf-8") as f:
        manifest: Dict[str, Any] = json.load(f)
    return manifest


class Layout:
    """A packed little-endian record: field names bound to a struct string.

    `fields` is a sequence of (name, code) pairs where `code` is a struct
    format fragment (e.g. "I", "Q", "f", "4Q"). Offsets and sizes fall out of
    the codes, so the same declaration both packs bytes and is comparable,
    field by field, with the C++ offsetof table.
    """

    def __init__(self, name: str, fields: Sequence[Tuple[str, str]]) -> None:
        self.name = name
        self.names = tuple(n for n, _ in fields)
        self.codes = tuple(c for _, c in fields)
        self.struct = struct.Struct("<" + "".join(self.codes))
        self.size = self.struct.size
        self.offsets: Dict[str, int] = {}
        self.sizes: Dict[str, int] = {}
        pos = 0
        for n, c in fields:
            width = struct.calcsize("<" + c)
            self.offsets[n], self.sizes[n] = pos, width
            pos += width
        assert pos == self.size, name  # "<" never pads

    def unpack_from(self, raw: Any, offset: int = 0) -> Tuple[Any, ...]:
        return self.struct.unpack_from(raw, offset)

    def pack(self, *values: Any) -> bytes:
        return self.struct.pack(*values)

    def table(self) -> List[Tuple[str, int, int]]:
        """[(name, offset, size)] — the shape `seeml-abi` prints."""
        return [(n, self.offsets[n], self.sizes[n]) for n in self.names]


# --- SMF: the model container (source/language/model_format.h) ---------------

SMF_MAGIC = 0x31464D53  # "SMF1"
SMF_VERSION = 5         # v5: per-op attr1 (RoPE base as f32 bits) after attr0
SMF_MIN_VERSION = 1
SMF_DEFAULT_ROPE_BASE = 10000.0  # what attr1 == 0 means on a Rope op
SMF_OP_KINDS = {
    "matmul": 0, "add_bias": 1, "relu": 2, "gelu": 3, "silu": 4, "mul": 5,
    "layer_norm": 6, "add": 7, "rms_norm": 8, "rope": 9, "attention": 10,
    "embedding": 11,
}
SMF_PREAMBLE = Layout("SmfPreamble", [
    ("magic", "I"), ("version", "I"), ("tensor_count", "I"),
    ("op_count", "I")])

# --- SDS: the corpus container (runtime/feeder/dataset.h) --------------------

SDS_MAGIC = 0x31534453  # "SDS1"
SDS_VERSION = 2         # v2: token records (input_kind 1)
SDS_MIN_VERSION = 1
SDS_LABEL_KIND_MAX = 2  # 0 none, 1 class index (i32), 2 float vector
SDS_HEADER = Layout("SdsHeader", [
    ("magic", "I"), ("version", "I"), ("num_samples", "Q"),
    ("input_dim", "Q"), ("label_kind", "I"), ("input_kind", "I"),
    ("label_dim", "Q")])
SDS_HEADER_BYTES = SDS_HEADER.size

# --- SEEU: the update plan (source/plan/schema.h, instruction.h) -------------

SEEU_MAGIC = 0x55454553  # "SEEU"
SEEU_VERSION = 13
SEEU_OLDEST_READABLE = 4
RODATA_BIT = 1 << 63
NULL_REF = (1 << 64) - 1
RODATA_ALIGNMENT = 16384
GEMM_PANEL_FLOATS = 8192

PLAN_HEADER = Layout("PlanHeader", [
    ("magic", "I"), ("version", "I"),
    ("arena_size", "Q"), ("persistent_size", "Q"),
    ("input_ref", "Q"), ("input_floats", "Q"),
    ("label_ref", "Q"), ("label_bytes", "Q"),
    ("label_kind", "I"), ("optimizer_kind", "I"),
    ("loss_ref", "Q"),
    ("train_instr_offset", "Q"), ("train_instr_count", "Q"),
    ("merge_instr_offset", "Q"), ("merge_instr_count", "Q"),
    ("rodata_offset", "Q"), ("rodata_size", "Q"),
    ("persist_init_offset", "Q"), ("persist_init_size", "Q"),
    ("emit_table_offset", "Q"), ("emit_count", "Q"),
    ("lr", "f"), ("beta1", "f"), ("beta2", "f"), ("eps", "f"),
    ("weight_decay", "f"),
    ("gemm_tile_k", "I"),
    ("batch", "Q"), ("default_steps", "Q"),
    ("eval_instr_offset", "Q"), ("eval_instr_count", "Q"),
    ("source_model_hash", "Q"), ("plan_hash", "Q"),
    ("lr_schedule", "I"), ("gemm_tile_n", "I"),
    ("warmup_steps", "Q"),
    ("min_lr_factor", "f"), ("clip_norm", "f"),
    ("input_kind", "I"), ("grad_accum_steps", "I"),
    ("seq_len", "Q"),
    ("step_instr_offset", "Q"), ("step_instr_count", "Q")])

# `pad` is written as zero and never read; `in`/`out` are operand words.
INSTRUCTION = Layout("UpdateInstruction", [
    ("opcode", "H"), ("flags", "H"), ("pad", "I"), ("in", "4Q"),
    ("out", "3Q")])
EMIT_ENTRY = Layout("EmitEntry", [
    ("smf_data_offset", "Q"), ("byte_size", "Q"), ("arena_offset", "Q")])

# Opcode -> the name seeml-seeu-dump prints (source/plan/opcode_names.h).
OPCODES = {
    0: "nop", 1: "gemm.nn", 2: "gemm.nt", 3: "gemm.tn", 4: "gemm.acc_nn",
    5: "add.ew", 6: "add.bias", 7: "relu.fwd", 8: "relu.bwd", 9: "scale",
    10: "reduce.rows", 11: "softmax_xent.fwd", 12: "softmax_xent.bwd",
    13: "mse.fwd", 14: "mse.bwd", 15: "kl_distill.fwd", 16: "kl_distill.bwd",
    17: "sgd.step", 18: "adamw.step", 19: "fill", 20: "copy", 21: "mul.ew",
    22: "gelu.fwd", 23: "gelu.bwd", 24: "silu.fwd", 25: "silu.bwd",
    26: "layer_norm.fwd", 27: "layer_norm.bwd", 28: "clip.norm",
    29: "gemm.nn.q8", 30: "gemm.nt.q8", 31: "rms_norm.fwd",
    32: "rms_norm.bwd", 33: "rope.fwd", 34: "rope.bwd", 35: "attn.fwd",
    36: "attn.dp", 37: "attn.dv", 38: "softmax_rows.bwd", 39: "attn.dq",
    40: "attn.dk", 41: "embed.fwd", 42: "accumulate", 43: "gemm.nn.bf16",
    44: "gemm.nt.bf16", 45: "rope.table", 46: "fused.map",
}

# Instruction flags on the GEMM family: the fused epilogue.
FLAG_EPILOGUE_BIAS = 1
FLAG_EPILOGUE_ACT_SHIFT = 1
FLAG_EPILOGUE_ACT_MASK = 6

# kFusedMap micro-program: one byte per stage, packed into an operand word.
FUSED_STAGES = {"end": 0, "add": 1, "mul": 2, "scale": 3, "relu": 4,
                "gelu": 5, "silu": 6}
FUSED_KIND_MASK = 15
FUSED_ARG_SHIFT = 4
FUSED_ARG_MASK = 3
FUSED_RUN_IS_RIGHT = 128
FUSED_MAX_STAGES = 4

# --- SEKP: the checkpoint (runtime/custodian/checkpoint_format.h) ------------

SEKP_MAGIC = 0x504B4553  # "SEKP"
SEKP_VERSION = 4         # v4: the run's horizon follows the v3 header
SEKP_OLDEST_READABLE = 3
CKPT_HEADER = Layout("CkptHeader", [
    ("magic", "I"), ("version", "I"), ("plan_hash", "Q"), ("step", "Q"),
    ("persistent_size", "Q"), ("payload_hash", "Q")])
CKPT_HEADER_V4_TAIL = Layout("CkptHeaderV4Tail", [("horizon_steps", "Q")])

# --- The probe trace (tool/probe_trace.h) -------------------------------------

PROBE_TRACE_MAGIC = 0x54504553  # "SEPT"
PROBE_TRACE_VERSION = 1
PROBE_TRACE_PREAMBLE = struct.Struct("<IIQ")  # magic, version, instructions
PROBE_TRACE_RECORD = struct.Struct("<IHH")    # index, opcode, extents
PROBE_TRACE_EXTENT = struct.Struct("<QQ")     # arena offset, bytes

STRUCTS = {
    "seeu": (PLAN_HEADER, INSTRUCTION, EMIT_ENTRY),
    "sekp": (CKPT_HEADER, CKPT_HEADER_V4_TAIL),
}


def check_against(abi: Dict[str, Any]) -> List[str]:
    """Every disagreement between this module and a `seeml-abi` manifest, as
    a list of sentences; empty means the two planes describe one seam."""
    bad: List[str] = []

    def same(what: str, mine: Any, theirs: Any) -> None:
        if mine != theirs:
            bad.append(f"{what}: Python says {mine!r}, C++ says {theirs!r}")

    smf, sds, seeu, sekp = abi["smf"], abi["sds"], abi["seeu"], abi["sekp"]
    same("smf.magic", SMF_MAGIC, smf["magic"])
    same("smf.version", SMF_VERSION, smf["version"])
    same("smf.min_version", SMF_MIN_VERSION, smf["min_version"])
    same("smf.default_rope_base", SMF_DEFAULT_ROPE_BASE,
         float(smf["default_rope_base"]))
    same("smf.op_kinds", SMF_OP_KINDS, smf["op_kinds"])
    same("sds.magic", SDS_MAGIC, sds["magic"])
    same("sds.version", SDS_VERSION, sds["version"])
    same("sds.min_version", SDS_MIN_VERSION, sds["min_version"])
    same("sds.header_bytes", SDS_HEADER_BYTES, sds["header_bytes"])
    same("sds.label_kind_max", SDS_LABEL_KIND_MAX, sds["label_kind_max"])
    same("seeu.magic", SEEU_MAGIC, seeu["magic"])
    same("seeu.version", SEEU_VERSION, seeu["version"])
    same("seeu.oldest_readable", SEEU_OLDEST_READABLE,
         seeu["oldest_readable"])
    same("seeu.rodata_bit", RODATA_BIT, 1 << seeu["rodata_bit"])
    same("seeu.rodata_alignment", RODATA_ALIGNMENT, seeu["rodata_alignment"])
    same("seeu.gemm_panel_floats", GEMM_PANEL_FLOATS,
         seeu["gemm_panel_floats"])
    same("seeu.opcodes", {n: k for k, n in OPCODES.items()}, seeu["opcodes"])
    same("seeu.flags", {"epilogue_bias": FLAG_EPILOGUE_BIAS,
                        "epilogue_act_shift": FLAG_EPILOGUE_ACT_SHIFT,
                        "epilogue_act_mask": FLAG_EPILOGUE_ACT_MASK},
         seeu["flags"])
    same("seeu.fused_stages", FUSED_STAGES, seeu["fused_stages"])
    same("seeu.fused_stage", {"kind_mask": FUSED_KIND_MASK,
                              "arg_shift": FUSED_ARG_SHIFT,
                              "arg_mask": FUSED_ARG_MASK,
                              "run_is_right": FUSED_RUN_IS_RIGHT,
                              "max_stages": FUSED_MAX_STAGES},
         seeu["fused_stage"])
    same("sekp.magic", SEKP_MAGIC, sekp["magic"])
    same("sekp.version", SEKP_VERSION, sekp["version"])
    same("sekp.oldest_readable", SEKP_OLDEST_READABLE,
         sekp["oldest_readable"])
    same("probe_trace.magic", PROBE_TRACE_MAGIC, abi["probe_trace"]["magic"])
    same("probe_trace.version", PROBE_TRACE_VERSION,
         abi["probe_trace"]["version"])
    for section, layouts in STRUCTS.items():
        for layout in layouts:
            theirs = abi[section]["structs"].get(layout.name)
            if theirs is None:
                bad.append(f"{section}.{layout.name}: not in the manifest")
                continue
            same(f"{layout.name}.size", layout.size, theirs["size"])
            same(f"{layout.name}.fields", layout.table(),
                 [(f["name"], f["offset"], f["size"])
                  for f in theirs["fields"]])
    return bad


if __name__ == "__main__":
    import sys
    problems = check_against(load_abi(*sys.argv[1:2]))
    for line in problems:
        print(f"formats: {line}", file=sys.stderr)
    sys.exit(1 if problems else 0)
