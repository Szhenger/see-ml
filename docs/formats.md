# SeeML Binary Formats

All formats are little-endian; loaders reject big-endian hosts at compile
time. Every multi-byte integer is packed without padding unless a struct is
shown (structs are `#pragma pack(1)` and part of the ABI — the application
binary interface, the byte contract the compiler and runtime share).
Integrity hashing comes from `source/identity/hash.h` — the deterministic
parallel `ContentHash64` for whole-artifact identity, and its sibling
`PlanSelfHash` for a blob whose hash field lives inside the sealed bytes,
both chunked folds of 64-bit FNV-1a (Fowler–Noll–Vo). Hashing is a
corruption/mismatch detector, not a signature; authenticate plans in your
update transport.

## SMF — SeeML Model Format (`.smf`, v2)

The dependency-free model container consumed by `seeml-update-compile`
(source and teacher models). Produced by `tool/export_model.py`.

```
u32 magic  "SMF1" (0x31464D53)
u32 version         1 or 2 accepted; writer emits 2
u32 num_tensors
u32 num_ops
str input_name      (str = u16 length + bytes, no terminator)
str output_name
tensors[num_tensors]:
  str  name
  u8   rank
  u8   flags        bit0 = constant (weight); else graph I/O
  i64  dims[rank]   -1 = dynamic batch (non-const tensors only)
  u64  data_offset  absolute file offset of the f32 blob (0 if not constant)
  u64  byte_size    must equal volume * 4 for constant tensors
data section: each constant tensor's f32 blob at its 64-aligned offset
ops[num_ops] (topologically ordered):
  u8   kind         0 MatMul  1 AddBias  2 Relu
                    3 Gelu    4 Silu     5 Mul    6 LayerNorm   (v2)
  str  name
  u8   num_inputs
  str  inputs[num_inputs]
  str  output
```

Op signatures: `MatMul(x, W)`, `AddBias(x, b)`, unary activations `(x)`,
`Mul(x, y)` (same shape), `LayerNorm(x, gamma, beta)` over the last dim.

The absolute `data_offset` of every weight is preserved through compilation:
it is how the emit table addresses the byte ranges that commit patches.
`LoadSmf` records the whole file's `ContentHash64` as the model's identity.
`SaveSmf` refuses any value that would not fit its length/count field
(names over 64 KiB, ranks or op-input counts over 255) rather than writing
a truncated prefix ahead of a full payload, and reports success only after
the stream has flushed and closed cleanly.

## SDS — SeeML Dataset (`.sds`, v1)

```
u32 magic "SDS1"; u32 version = 1
u64 num_samples; u64 input_dim
u32 label_kind    0 none | 1 class index (i32) | 2 dense (f32[label_dim])
u32 pad; u64 label_dim
records[num_samples]: f32 input[input_dim], then the label
```

## SEEU — Update Plan (`.seeu`, v5)

**Version negotiation.** The runtime accepts every plan version in
`[kSeeuOldestReadable, kSeeuVersion]` (`source/plan/schema.h`), not only the
version it was built at. Additive format changes — new header fields carved
out of `reserved`, zero meaning "feature absent" — bump `kSeeuVersion` only,
so already-deployed runtimes keep reading newer-compiled-but-compatible
plans' predecessors; semantic breaks (a field changing meaning or layout)
raise `kSeeuOldestReadable`, because misreading an old plan is worse than
rejecting it. Plans newer than the runtime are always rejected.

The fully ahead-of-time (AOT) compiled update: three instruction streams (train / eval /
merge), the frozen weights, the persistent segment's initial image, and the
emit table, addressed by a single `PlanHeader` (authoritative definition:
`source/plan/schema.h`).

Key header fields:

| field | meaning |
|---|---|
| `plan_hash` | `PlanSelfHash` of the blob with this field zeroed; verified on load |
| `source_model_hash` | `ContentHash64` of the source `.smf`; commit refuses other files (0 = unbound) |
| `eval_instr_offset/count` | forward+loss program for validation gating |
| `lr_schedule, warmup_steps, min_lr_factor` | runtime LR schedule (0 = constant) |
| `clip_norm` | informational; clip instructions are baked into the stream |

Version history: v2 added the eval program, integrity hashes, LR schedule,
and int8 rodata opcodes; v3 moved `source_model_hash` to `ContentHash64`;
v4 moved `plan_hash` to the chunked-parallel `PlanSelfHash`; v5 gave the
instruction's `flags` word meaning (epilogues fused into the GEMM —
general matrix–matrix multiply — instructions; below). The version gate
rejects plans below the readable floor — recompile.

Loaders additionally prove `batch` nonzero and every I/O slot and
instruction operand ref element-aligned; misaligned refs are load errors,
not UB at dispatch.

Tensor references are 64-bit words: bit 63 selects the address space
(0 = mutable arena, 1 = read-only rodata), bits 0..62 are a byte offset.
Instructions are exactly 64 bytes (`UpdateInstruction`): opcode, a `flags`
word, four input refs, three dim/aux words. Frozen weights selected by
`--quantize-base` are stored in rodata as per-tensor symmetric int8 with the
dequant scale carried in the GEMM instruction (`kGemmNNQ8` / `kGemmNTQ8`).

**Epilogue flags (v5).** On the forward GEMMs, `flags` fuses the layer's
epilogue into the C write-back: bit 0 = add a bias (`in[3]` carries the bias
ref — f32 GEMM only, since the q8 GEMM's `in[3]` is its scale), bits 1–2
select an activation (relu / gelu / silu). `C = act(A@B + bias)` evaluates
per element exactly what the standalone `kAddBias` + activation instructions
would, so fusion changes memory traffic, never bits. The validator rejects
unknown flag bits from v5 on, flags on any other opcode, and any nonzero
flags in a pre-v5 plan.

The emit table (`EmitEntry[]`) maps each LoRA (Low-Rank Adaptation)
adapter's **delta** (`Δ = (α/r)·A@B`, the adapter pair's product scaled by
α over rank r, materialized by the merge program) to the f32 byte range of its weight inside
the source `.smf`. Commit applies `W' = W + Δ` onto the file's pristine
weights — a quantized plan never bakes quantization error into the committed
model.

## Checkpoint (`SEKP`, v3)

```
u32 magic "SEKP"; u32 version = 3
u64 plan_hash        must match the plan's PlanHeader::plan_hash
u64 step             1-indexed AdamW timestep at save
u64 persistent_size  payload length
u64 payload_hash     ContentHash64 of the payload (v3; v2 used serial FNV-1a)
payload              the arena's persistent segment (adapters + moments)
```

A checkpoint resumes only under the exact plan that produced it; a foreign or
bit-flipped checkpoint is rejected before any byte reaches the arena.

## Durability

Model commits and checkpoints are written as `fsync`'d sidecar files followed
by an atomic `rename` and a best-effort directory `fsync` — a power cut leaves
either the old file or the new file, never a torn one. Both platform branches
honor the same contract: POSIX uses raw `write`/`fsync`/`rename`; Windows
uses checked `WriteFile` calls, `FlushFileBuffers` (the fsync equivalent),
and `MoveFileEx(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)`, which —
unlike CRT `rename` — can replace an existing file, so repeated checkpoints
to one path work.
