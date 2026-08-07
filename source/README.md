# The SeeML Source Language & Substrate

## Why isn't this under `compiler/`?

Because it isn't a stage of compilation. `source/` holds the abstractions of
the *source language itself* — the model container, the plan ABI — and the
low-level substrate both halves of the product share. The
[compiler](../compiler/README.md) consumes these; the
[runtime](../runtime/README.md) consumes these; neither *owns* them. Putting
them here, outside both, is what lets the two halves agree on every byte
without depending on each other.

```
source/
  language/               the SMF model container (SeeML Model Format)
    model_format          the parsed structs + on-disk layout constants
  plan/                   the compiler <-> runtime ABI (façade: update_types.h)
    config                the compilation request (UpdateConfig)
    instruction           the 64-byte instruction word + opcode vocabulary
    schema                the .seeu container header + emit table
  parallel/               deterministic chunked data-parallelism
    parallel_for          the leaked worker pool, chunk geometry by shape
  identity/               content hashing for integrity
    hash                  ContentHash64 + the in-blob PlanSelfHash seal
```

## The four things everyone shares

**`language/`** defines what a model *is* on disk — tensors, ops, the
byte layout — so the exporter, the compiler's loader, and the tools all
speak one format. **`plan/`** is the crucial seam: it is the entire
vocabulary the compiler and runtime share — the instruction set the compiler
*writes* and the runtime *reads*, plus the container header addressing every
section. Change it in one place and both halves move together.

**`parallel/`** and **`identity/`** are the substrate that makes the
product's headline promises possible. `parallel_for` chunks work by problem
shape and never by thread count — the single reason results are
**bitwise-identical** whether one core ran or eight. `hash` is the
integrity layer: a parallel FNV-1a variant that fingerprints a model (so
commit refuses the wrong file) and seals a plan (so a flipped bit is caught
at load, not as silent garbage on-device).

The discipline to notice: `plan/` is an *ABI*, so its structs are
`#pragma pack(1)` with `static_assert`s on their sizes — a layout change is
a versioned event, negotiated by the runtime's version policy, never an
accident.

## Where to go next

The byte-level formats — every field, offset, and magic number, and *why*
each is there — are specified in **[docs/formats.md](../docs/formats.md)**.
The parallelism and hashing machinery is taught where it matters most, in
[docs/runtime.md](../docs/runtime.md). How the compiler consumes all of this
is in [docs/compiler.md](../docs/compiler.md). The substrate's own suites
live under [test/source/](../test/README.md) (`parallel_for`, `hash`); the
`language/` and `plan/` structs are exercised throughout the compiler and
runtime suites.
