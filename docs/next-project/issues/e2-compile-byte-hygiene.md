---
title: "Core plane E2: compile-side byte hygiene — single weight residency, streaming emission, drop the zero-fills"
number: 81
labels: enhancement,efficiency,core-plane
plane: Core plane
origin: Performance audit
priority: P0
---
## Algorithmic root cause

The compile pipeline holds **three simultaneous copies of the weights** at
plan assembly — the loaded `SmfModel`, the packed `rodata` vector, and the
assembled plan blob (`update_compiler.cc:510-522`) — **≈84 GB peak for a
7B-f32 model**, and the emitter then peaks at ≈11× plan size. This is the
difference between compiling a 7B model and an OOM on the very devices
SeeAI targets. None of it is doctrinal; it is copy hygiene
(Performance Audit finding #2, the highest-value compile-side item).

## Design — all bitwise-safe

- **Reader/writer zero-fills:** `std::vector(n)` value-initializes GBs
  immediately overwritten by the read (`model_reader.cc:76`); the writer
  zero-fills its whole data section when only alignment gaps need it
  (`model_writer.cc:120`). Drop both.
- **Chunk the parallel per-tensor copy by bytes, not tensor index**
  (`model_reader.cc:219-224`) — an embedding-dominated model currently
  copies its largest tensor on one thread.
- **`reserve` + pack-in-place for rodata:** `binding.rodata` grows by
  repeated `resize` with no `reserve` (`arena_binder.cc:147,162`) —
  geometric reallocation copies a multi-GB vector 1–2× extra and
  zero-fills bytes the next line memcpys over; the total is exactly
  computable up front. Then pack rodata **directly into the plan buffer**
  so assembly holds ~1 copy, not 3.
- **Stream emission and free chunks as consumed**
  (`native_emitter.cc:46,65-69,87`): 11× → ~2× peak RSS. Coordinate with
  **P1** — the Python packer removes most of the chunk-string machinery
  outright; do not build streaming infrastructure P1 deletes.
- **Model the LoRA-delta segment in the step-0 memory gate**
  (`update_compiler.cc:354-369`, `merge_builder.cc:56`): with a quantized
  base the full-size f32 deltas can be 4× the weights they patch, and
  only the *final* gate sees them today (finding #8 — honesty, not speed).

## Acceptance

- Plan-assembly peak RSS ≤ ~1.2× (model + plan residency collapsed);
  emission ≤ ~2× plan size; a 7B-f32 compile fits a 96 GB build host with
  headroom, and 135M compiles show 2–3 fewer full-bandwidth passes.
- Byte-identical `.seeu` output on all fixtures.
- Step-0 estimate includes delta segments; the gate's number matches the
  field-audit RSS discipline (1.00–1.13×).

Refs: Performance Audit §01–§03 (findings #2, #8).
