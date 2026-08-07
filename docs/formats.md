# SeeML Binary Formats

## Why bytes, and how to read this document

Every artifact SeeML produces or consumes — models, corpora, plans, checkpoints — is a binary file with a fixed, documented layout. Why binary, when JSON exists? Because these files are mostly *tensors* — n-dimensional arrays of floats, millions of them — and because the device-side loader must be tiny, allocation-conscious, and paranoid. A format you can parse with bounds-checked `memcpy` is a format you can *prove* things about.

Before the individual formats, three conventions that apply everywhere, each worth understanding once:

- **Little-endian, always.** A multi-byte integer like `0x31464D53` is stored least-significant byte first. Rather than swap bytes on big-endian machines, the loaders simply refuse to compile there (`static_assert`) — a deliberate simplification: every target SeeML cares about is little-endian, so byte-swapping code would be untested dead weight.
- **Packed layouts.** Every multi-byte integer is packed without padding; where a C struct is shown, it is `#pragma pack(1)` and part of the ABI. Normally compilers insert invisible padding between struct fields for alignment; packing turns the struct into an exact byte-for-byte contract, so `sizeof` is the wire size and a `static_assert` can pin it forever.
- **Magic numbers.** Each format opens with a four-byte signature — read the little-endian `u32` as ASCII and you get the name back (`"SMF1"`, `"SDS1"`, `"SEEU"`, `"SEKP"`). It's the file introducing itself, and it means a mix-up (feeding a corpus where a model belongs) dies on byte 0 with a clear message, not on byte 40,000 with a weird one.

## First, a word about hashes

Several formats below carry 64-bit hashes, so let's establish what they are and — just as important — what they are *not*.

The workhorse is **FNV-1a**, a classic non-cryptographic hash chosen for being almost embarrassingly simple:

```
h = 0xcbf29ce484222325                    (the "offset basis")
for each byte b:  h = (h XOR b) × 0x100000001b3   (the "FNV prime")
```

Two constants, one XOR, one multiply per byte. Any single flipped bit avalanches through the multiplications and changes the digest — which is exactly the property needed to detect *corruption*.

For whole-model identity there's a faster sibling, **`ContentHash64`** (`source/identity/hash.h`), which fixes FNV's one weakness — it's inherently serial, one byte after another — with two layers of parallelism:

1. **Striping (instruction-level):** run 8 independent FNV lanes, byte i feeding lane i mod 8, each lane seeded differently (the offset basis XOR a golden-ratio multiple, so lanes never collide). Eight independent multiply chains keep a modern core's pipeline full — roughly 8× the throughput of the serial loop.
2. **Chunking (thread-level):** split the input into 1 MiB chunks, hash each chunk (with the striped kernel) in parallel, then fold the per-chunk digests together *in chunk order*, finishing with the total length. Because chunk boundaries depend only on the input *size* — never the thread count — the digest is bitwise-identical on one core or eight, in keeping with SeeML's determinism rule ([runtime.md](runtime.md)).

Note that `ContentHash64` of some bytes deliberately does **not** equal plain `Fnv1a64` of the same bytes — they are distinct identity contracts, and the format version bumps below track which one a field uses.

Now, the caveat, stated as bluntly as possible: **these hashes detect accidents, not adversaries.** FNV is trivially forgeable by anyone who wants to; there are no keys and no signatures here. SeeML's hashes answer "is this the same file, uncorrupted?" — never "do I trust whoever sent this?" Authenticate plans in your update transport (TLS, signed manifests — whatever your deployment already uses for software updates).

## SMF — SeeML Model Format (`.smf`, v2)

The dependency-free model container consumed by `seeml-update-compile` (source and teacher models), produced by `tool/export_model.py`. It answers exactly two questions: *what are the tensors?* and *what is the computation graph over them?*

```
u32 magic  "SMF1" (0x31464D53)
u32 version         1..4 accepted; writer emits 4
u32 num_tensors
u32 num_ops
str input_name      (str = u16 length + bytes, no terminator)
str output_name
u64 seq_len         (v3 only) rows per sequence; 0 = non-sequential
tensors[num_tensors]:
  str  name
  u8   rank
  u8   flags        bit0 = constant (weight); else graph I/O
  i64  dims[rank]   -1 = dynamic batch (non-const tensors only)
  u64  data_offset  absolute file offset of the f32 blob (0 if not constant)
  u64  byte_size    must equal volume × 4 for constant tensors
data section: each constant tensor's f32 blob at its 64-aligned offset
ops[num_ops] (topologically ordered):
  u8   kind         0 MatMul  1 AddBias  2 Relu
                    3 Gelu    4 Silu     5 Mul    6 LayerNorm    (v2)
                    7 Add     8 RmsNorm  9 Rope   10 Attention   (v3)
                    11 Embedding                                  (v4)
  str  name
  u8   num_inputs
  str  inputs[num_inputs]
  str  output
  u32  attr0        (v3 only) num_heads for Rope/Attention, else 0
```

Op signatures: `MatMul(x, W)`, `AddBias(x, b)`, unary activations `(x)`, `Mul(x, y)` / `Add(x, y)` (same shape), `LayerNorm(x, gamma, beta)` and `RmsNorm(x, gamma)` over the last dim, `Rope(x)` (rotary position embedding), and causal `Attention(q, k, v)` (v3, with model-level `seq_len`), plus `Embedding(tokens, table)` (v4). For a token-native model, the graph input is a rank-1 dynamic (`{-1}`) non-const tensor meaning i32 token ids, consumed *only* by embedding ops that gather rows of a constant `[vocab, dim]` table.

A few design choices worth noticing:

- **Strings are length-prefixed** (u16 + bytes, no NUL terminator). A parser reading length-prefixed strings can bounds-check *before* reading; a parser scanning for terminators can run off the end. This is the safer of the two classic conventions, chosen on purpose.
- **Ops must arrive topologically ordered** — every input produced before it's consumed. This moves a whole class of work (dependency resolution) out of every loader and into the single writer, and turns the loader's check into a linear scan ([compiler.md](compiler.md)).
- **Data offsets are absolute and 64-byte aligned.** Aligned so a mapped tensor starts on a cache line; *absolute* for a deeper reason: the offset of every weight is preserved through the entire compilation and lands in the plan's emit table — it is literally the patch address that commit uses to apply `W′ = W + Δ` to this very file. The format is designed for being *updated in place*, not just read.

`LoadSmf` records the whole file's `ContentHash64` as the model's identity; the compiled plan carries it, and commit refuses any file that doesn't match.

## SDS — SeeML Dataset (`.sds`, v2)

The corpus container — deliberately the simplest format in the family, because a dataset is just samples:

```
u32 magic "SDS1"; u32 version (1 or 2)
u64 num_samples; u64 input_dim
u32 label_kind    0 none | 1 class index (i32) | 2 dense (f32[label_dim])
u32 input_kind    v2: 0 = f32 feature rows | 1 = i32 token records
                  (this word was padding in v1, always 0)
u64 label_dim
records[num_samples]:
  input_kind 0: f32 input[input_dim], then the label
  input_kind 1: i32 tokens[input_dim + 1] and NO stored label — one
                sequence per record; inputs are tokens[0..S) and the
                next-token class labels are the shifted view tokens[1..S],
                derived at serving time (label_kind must be 1)
```

A 40-byte header, then fixed-size records — which means sample k lives at a *computable* offset, no index needed. `label_kind` covers the three training modes: `1` (a class index) for cross-entropy, `2` (a dense vector) for regression/MSE — predicting continuous values rather than classes, `0` (no label at all) for distillation, where the teacher model provides the target. The `u32` after `label_kind` was header padding in v1; v2 gives it meaning as `input_kind`, so old files (always zero there) still read correctly. Token corpora (input_kind 1) shuffle, split, and replay at record (sequence) granularity, so no sequence is ever cut or mixed.

## SEEU — Update Plan (`.seeu`, v7)

The star of the show: the fully AOT-compiled update. One file containing three instruction streams (train / eval / merge), the frozen weights, the persistent segment's initial image, and the emit table — every section addressed by a single `PlanHeader` at offset 0 (authoritative definition: `source/plan/schema.h`; every section 64-byte aligned). Versioning is additive: see the version history below (currently v7).

Key header fields:

| field | meaning |
|---|---|
| `plan_hash` | `PlanSelfHash` (chunked-parallel FNV-1a) of the whole blob with this field zeroed; verified on load |
| `source_model_hash` | `ContentHash64` of the source `.smf`; commit refuses other files (0 = unbound) |
| `arena_size`, `persistent_size` | the single device allocation, and its checkpointable prefix |
| `input_ref/floats`, `label_ref/bytes/kind`, `loss_ref` | the I/O slots — where the feeder writes and the engine reads |
| `train/eval/merge_instr_offset/count` | the three programs; eval is the forward+loss program for validation gating |
| `rodata_offset/size`, `persist_init_offset/size` | frozen weights; initial adapter/moment image |
| `emit_table_offset`, `emit_count` | the patch map (below) |
| `lr, beta1, beta2, eps, weight_decay` | optimizer hyperparameters, read at dispatch |
| `lr_schedule, warmup_steps, min_lr_factor` | runtime LR schedule (0 = constant) |
| `clip_norm` | informational; clip instructions are baked into the stream |

Notice the "zeroed field" trick in `plan_hash`: you can't hash a file that contains its own hash (the act of writing the digest would change it), so the digest is computed with that one field held at zero, then patched in. The verifier replays the same convention.

Version history: v2 added the eval program, integrity hashes, and LR schedule; v3 moved `source_model_hash` to `ContentHash64`; v4 moved `plan_hash` to the chunked-parallel `PlanSelfHash`; v5 gave the instruction `flags` word meaning (fused GEMM epilogues); v6 added the transformer opcode family (RMSNorm, RoPE, causal attention and its backward primitives); v7 added token-native input — `input_kind` and `seq_len` carved from `reserved`, plus the `kEmbedFwd` gather over a rodata-only table. Each new opcode is version-gated: a plan carrying one below its introducing version is corruption, not forward compatibility. The gather's *index* bound is the feeder contract's runtime job — every token id is proven inside both the narrowest embedding table and the narrowest softmax width before anything executes, exactly how class labels are bounded.

Why do hyperparameters live in the *header* while clip lives in the *stream*? Because a learning rate is a number the runtime consults, but clipping changes which instructions exist — structure belongs to the program, parameters to the header, and each fact has exactly one home ([compiler.md](compiler.md)).

**Tensor references** are 64-bit words: bit 63 selects the address space (0 = mutable arena, 1 = read-only rodata), bits 0..62 are a byte offset. Two flat address spaces and an offset — the entire memory model, and the reason the validator can prove write-safety with a single bit test.

**Instructions** are exactly 64 bytes — one cache line — laid out as:

```
u16 opcode | u16 flags | u32 pad | u64 in[4] | u64 out[3]
```

with 31 opcodes (`kNop` through `kGemmNTQ8` — the full enum with per-opcode operand conventions is in `source/plan/instruction.h`). The `in[]` slots hold tensor refs; scalars (a GEMM's α, a fill value, clip's max-norm) are f32 *bit-cast* into a spare slot; the `out[]` words carry dimensions (a GEMM's M, N, K; LayerNorm packs `(rows << 32) | cols`). Inspect any plan's streams with `seeml-seeu-dump --instrs` ([usage.md](usage.md)).

Frozen weights selected by `--quantize-base` are stored in rodata as per-tensor symmetric int8 (scale = max|w|/127) with the dequant scale carried *in the GEMM instruction itself* (`kGemmNNQ8` / `kGemmNTQ8`) — the kernel folds it into its existing multiply, so dequantization is free.

**The emit table** (`EmitEntry[]`, 24 bytes each: `smf_data_offset`, `byte_size`, `arena_offset`) is the bridge back to the model file: it maps each adapter's **delta** (`Δ = (α/r)·A@B`, materialized by the merge program at `arena_offset`) to the f32 byte range of its weight inside the source `.smf`. Commit applies `W′ = W + Δ` onto the file's pristine weights — which is why a quantized plan never bakes quantization error into the committed model: the int8 copy trains, but the original floats get patched.

## Checkpoint (`SEKP`, v3)

Training state you can power-cycle through:

```
u32 magic "SEKP"; u32 version = 3
u64 plan_hash        must match the plan's PlanHeader::plan_hash
u64 step             1-indexed AdamW timestep at save
u64 persistent_size  payload length
u64 payload_hash     ContentHash64 of the payload (v3; v2 used serial FNV-1a)
payload              the arena's persistent segment (adapters + moments)
```

The payload is a raw byte-copy of the arena's persistent segment — possible only because the compiler put everything resumable (LoRA parameters *and* AdamW moments) contiguously at arena offset 0. Three fields guard the restore, each against a different failure: `plan_hash` against the wrong plan (offsets into someone else's arena layout would be garbage), `persistent_size` against a layout drift, `payload_hash` against bit rot. A foreign or bit-flipped checkpoint is rejected before any byte reaches the arena. Saving `step` matters more than it looks: AdamW's bias correction depends on t, so resuming at the wrong step would silently distort the next updates.

## Durability

Model commits and checkpoints are written as `fsync`'d sidecar files followed by an atomic `rename` and a best-effort directory `fsync` — the full liturgy, and why each step exists, is in [runtime.md](runtime.md). The contract to remember: a power cut leaves either the old file or the new file, never a torn one.

## To recap

- Four formats, one philosophy: fixed layouts you can bounds-check before trusting, magic numbers so mix-ups die at byte 0, and hashes so corruption dies at load — while authentication remains the transport's job.
- SMF is a model designed to be *patched* (absolute weight offsets survive compilation into the emit table); SDS is samples at computable offsets; SEEU is the entire training job as data; SEKP is the resumable slice of the arena, triple-guarded.
- Every artifact is verified before its first byte is acted upon — the same boundary discipline as the code that reads them.
