# SeeML Binary Formats

## Why bytes, and how to read this document

Every artifact SeeML produces or consumes — models, corpora, plans, checkpoints — is a binary file with a fixed, documented layout. Why binary, when JSON exists? Because these files are mostly *tensors* — n-dimensional arrays of floats, millions of them — and because the device-side loader must be tiny, allocation-conscious, and paranoid. A format you can parse with bounds-checked `memcpy` is a format you can *prove* things about.

Before the individual formats, three conventions that apply everywhere, each worth understanding once:

- **Little-endian, always.** A multi-byte integer like `0x31464D53` is stored least-significant byte first. Rather than swap bytes on big-endian machines, the loaders simply refuse to compile there (`static_assert`) — a deliberate simplification: every target SeeML cares about is little-endian, so byte-swapping code would be untested dead weight.
- **Packed layouts.** Every multi-byte integer is packed without padding; where a C struct is shown, it is `#pragma pack(1)` and part of the ABI. Normally compilers insert invisible padding between struct fields for alignment; packing turns the struct into an exact byte-for-byte contract, so `sizeof` is the wire size and a `static_assert` can pin it forever.
- **Magic numbers.** Each format opens with a four-byte signature — read the little-endian `u32` as ASCII and you get the name back (`"SMF1"`, `"SDS1"`, `"SEEU"`, `"SEKP"`). It's the file introducing itself, and it means a mix-up (feeding a corpus where a model belongs) dies on byte 0 with a clear message, not on byte 40,000 with a weird one.

## One statement per plane, checked by machine

Every layout below is declared twice — once in the C++ headers, once in `tool/seeml/formats.py` for the build-host tools — and never a third time. The two are held together without anyone remembering to: `seeml-abi` prints the C++ side's magics, versions, enums and packed-struct offsets as JSON; that output is committed as `tool/seeml/abi.json`; CI regenerates it and fails on a diff; and `test/tool/formats_test.py` compares every Python declaration with it. Changing a format is therefore a three-step edit that cannot be half done: the header, `formats.py`, and `build/seeml-abi > tool/seeml/abi.json`. `seeml-seeu-dump --json` is the same idea for whole plans: the C++ decode as data, compared field for field with the Python reader on freshly compiled plans.

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

## SMF — SeeML Model Format (`.smf`, v5)

The dependency-free model container consumed by `seeml-update-compile` (source and teacher models), produced by `tool/export_model.py`. It answers exactly two questions: *what are the tensors?* and *what is the computation graph over them?*

```
u32 magic  "SMF1" (0x31464D53)
u32 version         1..5 accepted; the C++ writer emits 5, the exporter the
                    lowest version that can carry the model (3, 4 or 5)
u32 num_tensors
u32 num_ops
str input_name      (str = u16 length + bytes, no terminator)
str output_name
u64 seq_len         (v3+) rows per sequence; 0 = non-sequential
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
  u32  attr0        (v3+) num_heads for Rope/Attention, else 0
  u32  attr1        (v5+) Rope: rotary base θ as IEEE-754 f32 bits (0 = 10000); else 0
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

## SEEU — Update Plan (`.seeu`, v15)

The star of the show: the fully AOT-compiled update. One file containing three instruction streams (train / eval / merge) — four under gradient accumulation, when the train section is the grad program and a step section holds the optimizer program — the frozen weights, the persistent segment's initial image, and the emit table — every section addressed by a single `PlanHeader` at offset 0 (authoritative definition: `source/plan/schema.h`; every section 64-byte aligned). Versioning is additive: see the version history below (currently v15).

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
| `gemm_tile_k`, `gemm_tile_n` | the CPU blocked-GEMM tile geometry (v11) the compiler decided from the offline tuner's host-keyed table or `--gemm-tiles`; 0 = the runtime's compiled-in default; throughput only, never bits (the K tile must be a multiple of 4) |

Notice the "zeroed field" trick in `plan_hash`: you can't hash a file that contains its own hash (the act of writing the digest would change it), so the digest is computed with that one field held at zero, then patched in. The verifier replays the same convention.

Version history: v2 added the eval program, integrity hashes, and LR schedule; v3 moved `source_model_hash` to `ContentHash64`; v4 moved `plan_hash` to the chunked-parallel `PlanSelfHash`; v5 gave the instruction `flags` word meaning (fused GEMM epilogues); v6 added the transformer opcode family (RMSNorm, RoPE, causal attention and its backward primitives); v7 added token-native input — `input_kind` and `seq_len` carved from `reserved`, plus the `kEmbedFwd` gather over a rodata-only table; v8 gave the high half of the `kKLDistill{Fwd,Bwd}` temperature word meaning as the distillation loss scale (`T²`, so the soft-target gradient stays commensurate with a hard-label term) — zero there, as in every pre-v8 plan, reads as 1.0; v9 added gradient accumulation — `grad_accum_steps` (from a pad word) and the step program's offset and count (from `reserved`), plus the `kAccumulate` opcode (`dst += src`), version-gated: when `grad_accum_steps > 1` the train section is the grad program and the step section the optimizer program, and the two must appear together; v10 added bf16 frozen weights — `kGemmNNBF16` / `kGemmNTBF16`, whose B operand is bfloat16 rodata (2 bytes per element, rodata-pinned like int8, widened exactly in the kernel), version-gated; v11 gave the header's two pad words meaning as the CPU GEMM tile geometry, `gemm_tile_k` and `gemm_tile_n` — the compiler's kernel-policy decision (`tool/autotune.py`'s table, `--gemm-tiles`, or zero for the runtime default), proven on load (the K tile on the kernel's 4-wide unroll) and handed to the CPU backend; it changes throughput only, never bits. v12 is the bitwise-safe kernel batch (E3), carried entirely in instruction words that were `kNullRef` / zero in every earlier plan, so the header is unchanged: one opcode, `kRopeTable` (`[S, d/2, 2]` cos/sin of the RoPE angles, built on the device with the recurrence the rotation kernels run), an optional third operand on `kRopeFwd` / `kRopeBwd` naming its result, and an optional per-tensor clip threshold in `out[1]` of `kSgdStep` / `kAdamWStep` that folds the preceding `kClipNorm` into the step (the gradient is consumed clipped and never rewritten; a non-finite norm refuses that tensor's step). All three are rejected below v12, and a v12 program computes the same bits as the v11 program it replaces (`--no-rope-table`, `--no-fuse-clip` compile the old shapes for comparison). v13 added `kFusedMap` (E4): a chain of up to four elementwise stages — add, mul, scale, relu, gelu, silu — as one instruction, the stage bytes in `out[1]` (kind in the low nibble; operand slot or immediate index in bits 4–5; bit 7 = the running value is the right operand), two scale immediates in `out[2]`, operands `in[0]` = x, `in[1..2]` = the binary stages' tensors, `in[3]` = out; the validator proves the program (known kinds, zero-terminated, every named slot present and every unnamed one absent, finite immediates) and the runtime runs each stage as its own loop over a cache-resident block, so the result is bit-identical to the instruction sequence it replaces. v14 added the GEMM addend (E10): `kFlagGemmAddend` (flag bit 3) on `kGemmNN` / `kGemmNT` / `kGemmTN` makes the instruction `C = D + A@B`, with `D`'s ref (`[M, N]` f32, read, disjoint from `C`) in the otherwise-free `in[3]` — every LoRA site's forward `C + ts@B` and backward `dC@Wᵀ + dt@Aᵀ` without the activation-sized `kAddEW` and the transient it read. Each element is `d + s` over the complete dot product, so the fold is bit-identical to the GEMM + add pair; the bit excludes the bias / activation epilogue (one slot, one tenant), is refused on the int8 / bf16 GEMMs and every other opcode, and is corruption below v14. v15 added the tiled attention family (E11): `kAttnFwdTiled` (47), `kAttnDQTiled` (48), `kAttnDKTiled` (49), `kAttnDVTiled` (50). Instead of the `[B·H·S, S]` probability cache the cached family keeps from forward to backward, they keep a stats row of four floats per query row (`kAttnStatsWidth`: the score max, the inverse softmax denominator, the softmax-backward rowsum delta — written by the dQ pass, read by the dK pass, so a stream runs dQ first — and a pad), and every backward pass recomputes the probabilities it needs with the cached kernels' own expressions, so the two families compute identical bits. The forward keeps the cached forward's slots (stats where the cache was); the backward takes q, k, v, dO in `in[0..3]`, the stats row in `out[0]`, its result in `out[1]` and the geometry packed 16 bits a field in `out[2]` (`B<<48 | S<<32 | H<<16 | d`). All four are corruption below v15. The header has no spare word left: the next additive header field needs a larger header. Layout note, no version: since G1b-2 the rodata section starts on a 16 KiB boundary and the blob is padded to a 16 KiB multiple (`kSeeuRodataAlignment`), so a page-aligned embedded plan lets the Metal backend wrap the frozen weights zero-copy — the header's explicit offsets already describe it, and older runtimes read such plans unchanged. Each new opcode is version-gated: a plan carrying one below its introducing version is corruption, not forward compatibility. The gather's *index* bound is the feeder contract's runtime job — every token id is proven inside both the narrowest embedding table and the narrowest softmax width before anything executes, exactly how class labels are bounded.

Why do hyperparameters live in the *header* while clip lives in the *stream*? Because a learning rate is a number the runtime consults, but clipping changes which instructions exist — structure belongs to the program, parameters to the header, and each fact has exactly one home ([compiler.md](compiler.md)).

**Tensor references** are 64-bit words: bit 63 selects the address space (0 = mutable arena, 1 = read-only rodata), bits 0..62 are a byte offset. Two flat address spaces and an offset — the entire memory model, and the reason the validator can prove write-safety with a single bit test.

**Instructions** are exactly 64 bytes — one cache line — laid out as:

```
u16 opcode | u16 flags | u32 pad | u64 in[4] | u64 out[3]
```

with 45 opcodes (`kNop` through `kGemmNTBF16`, contiguous from 0). The `in[]` slots hold tensor refs; scalars (a GEMM's α, a fill value, clip's max-norm) are f32 *bit-cast* into a spare slot; the `out[]` words carry dimensions (a GEMM's M, N, K; LayerNorm packs `(rows << 32) | cols`). Inspect any plan's streams with `seeml-seeu-dump --instrs` ([usage.md](usage.md)).

Frozen weights selected by `--quantize-base` are stored in rodata as per-tensor symmetric int8 (scale = max|w|/127) with the dequant scale carried *in the GEMM instruction itself* (`kGemmNNQ8` / `kGemmNTQ8`) — the kernel folds it into its existing multiply, so dequantization is free.

**The emit table** (`EmitEntry[]`, 24 bytes each: `smf_data_offset`, `byte_size`, `arena_offset`) is the bridge back to the model file: it maps each adapter's **delta** (`Δ = (α/r)·A@B`, materialized by the merge program at `arena_offset`) to the f32 byte range of its weight inside the source `.smf`. Commit applies `W′ = W + Δ` onto the file's pristine weights — which is why a quantized plan never bakes quantization error into the committed model: the int8 copy trains, but the original floats get patched.

## Checkpoint (`SEKP`, v5)

Training state you can power-cycle through:

```
u32 magic "SEKP"; u32 version = 5      (v3 and v4 files are still read)
u64 plan_hash        must match the plan's PlanHeader::plan_hash
u64 step             1-indexed AdamW timestep at save
u64 persistent_size  payload length
u64 payload_hash     ContentHash64 of the payload (v3; v2 used serial FNV-1a)
u64 horizon_steps    v4: the LR-schedule horizon of the run that saved it
v5 tail (64 bytes):
  u64 shuffle_origin       the training set's shuffle stream (0 = sequential)
  u64 train_samples        the split: training samples ...
  u64 val_samples          ... and validation samples (0 = no split)
  u64 best_step            the step of the best evaluated state
  u64 best_payload_hash    ContentHash64 of the best payload, when present
  u32 flags                1 val_initial present, 2 accuracy present,
                           4 best payload follows, 8 best_* are live
  u32 val_initial_loss     f32 bits: the SOURCE model's validation loss
  u32 val_initial_accuracy f32 bits
  u32 best_loss            f32 bits
  u32 best_accuracy        f32 bits
  u32 stale_evals          evaluations since the best (patience)
payload              the arena's persistent segment (adapters + moments)
best payload         v5, flag 4: the best evaluated segment, same length
```

The payload is a raw byte-copy of the arena's persistent segment — possible only because the compiler put everything resumable (LoRA parameters *and* AdamW moments) contiguously at arena offset 0. Three fields guard the restore, each against a different failure: `plan_hash` against the wrong plan (offsets into someone else's arena layout would be garbage), `persistent_size` against a layout drift, `payload_hash` against bit rot. A foreign or bit-flipped checkpoint is rejected before any byte reaches the arena. Saving `step` matters more than it looks: AdamW's bias correction depends on t, so resuming at the wrong step would silently distort the next updates. The **horizon** (v4) is the same kind of fact for the schedule: the step count the interrupted run was annealing over. A resume with no step count trains exactly the remainder to it, so the resumed run's learning rates — and its bits — are the uninterrupted run's; a v3 file has none and resumes on the plan's compiled budget, as it always did. The **v5 tail** makes a resume honest in two more ways. The run binding — which shuffle stream, and how many samples on each side of the train/validation split — lets a resume *refuse* a different `--seed` or `--val-frac`: those would silently move the boundary and train on the first run's validation rows. The stored `val_initial_*` is the source model's score, measured once by the first run, so the gate keeps comparing the whole update against it instead of scoring the resumed adapter as its own "before". And the **best payload** is the state the update will commit — the best evaluated one, not the last — which must survive an interruption exactly as the last one does; it is written only while it differs from the main payload (a checkpoint taken at a new best holds it once).

## Durability

Model commits and checkpoints are written as `fsync`'d sidecar files followed by an atomic `rename` and a best-effort directory `fsync` — the full liturgy, and why each step exists, is in [runtime.md](runtime.md). The contract to remember: a power cut leaves either the old file or the new file, never a torn one.

## To recap

- Four formats, one philosophy: fixed layouts you can bounds-check before trusting, magic numbers so mix-ups die at byte 0, and hashes so corruption dies at load — while authentication remains the transport's job.
- SMF is a model designed to be *patched* (absolute weight offsets survive compilation into the emit table); SDS is samples at computable offsets; SEEU is the entire training job as data; SEKP is the resumable slice of the arena, triple-guarded.
- Every artifact is verified before its first byte is acted upon — the same boundary discipline as the code that reads them.
