# SeeML Binary Formats

All formats are little-endian; loaders reject big-endian hosts at compile
time. Every multi-byte integer is packed without padding unless a struct is
shown (structs are `#pragma pack(1)` and part of the ABI). Integrity hashing
is 64-bit FNV-1a (`source/hash.h`) — a corruption/mismatch detector, not a
signature; authenticate plans in your update transport.

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
`LoadSmf` records the whole file's FNV-1a as the model's identity.

## SDS — SeeML Dataset (`.sds`, v1)

```
u32 magic "SDS1"; u32 version = 1
u64 num_samples; u64 input_dim
u32 label_kind    0 none | 1 class index (i32) | 2 dense (f32[label_dim])
u32 pad; u64 label_dim
records[num_samples]: f32 input[input_dim], then the label
```

## SEEU — Update Plan (`.seeu`, v2)

The fully AOT-compiled update: three instruction streams (train / eval /
merge), the frozen weights, the persistent segment's initial image, and the
emit table, addressed by a single `PlanHeader` (authoritative definition:
`compiler/backend/update_types.h`).

Key v2 header fields:

| field | meaning |
|---|---|
| `plan_hash` | FNV-1a of the blob with this field zeroed; verified on load |
| `source_model_hash` | FNV-1a of the source `.smf`; commit refuses other files (0 = unbound) |
| `eval_instr_offset/count` | forward+loss program for validation gating |
| `lr_schedule, warmup_steps, min_lr_factor` | runtime LR schedule (0 = constant) |
| `clip_norm` | informational; clip instructions are baked into the stream |

Tensor references are 64-bit words: bit 63 selects the address space
(0 = mutable arena, 1 = read-only rodata), bits 0..62 are a byte offset.
Instructions are exactly 64 bytes (`UpdateInstruction`): opcode, four input
refs, three dim/aux words. Frozen weights selected by `--quantize-base` are
stored in rodata as per-tensor symmetric int8 with the dequant scale carried
in the GEMM instruction (`kGemmNNQ8` / `kGemmNTQ8`).

The emit table (`EmitEntry[]`) maps each adapter's **delta** (`Δ = (α/r)·A@B`,
materialized by the merge program) to the f32 byte range of its weight inside
the source `.smf`. Commit applies `W' = W + Δ` onto the file's pristine
weights — a quantized plan never bakes quantization error into the committed
model.

## Checkpoint (`SEKP`, v2)

```
u32 magic "SEKP"; u32 version = 2
u64 plan_hash        must match the plan's PlanHeader::plan_hash
u64 step             1-indexed AdamW timestep at save
u64 persistent_size  payload length
u64 payload_hash     FNV-1a of the payload
payload              the arena's persistent segment (adapters + moments)
```

A checkpoint resumes only under the exact plan that produced it; a foreign or
bit-flipped checkpoint is rejected before any byte reaches the arena.

## Durability

Model commits and checkpoints are written as `fsync`'d sidecar files followed
by an atomic `rename` and a best-effort directory `fsync` — a power cut leaves
either the old file or the new file, never a torn one.
