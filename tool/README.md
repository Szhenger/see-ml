# The SeeML Tools

## The human-facing edge of the pipeline

Everything else in the repository is a library. `tool/` is where a person
actually stands: it's how a model *gets into* SeeML, how the update plan is
*produced*, and how you *look inside* one when something seems wrong. Three
programs, each a thin, strict shell around the libraries the rest of the
tree provides.

```
tool/
  export_model.py         PyTorch model + data  ->  SMF / SDS files
  seeml_update_compile.cc the compiler CLI       ->  a .seeu update package
  pack_update.py          the package assembler  ->  .incbin-embedded package
  seeml_seeu_dump.cc      the plan disassembler   (inspect any .seeu)
  seeml_bench.cc          the benchmark harness  ->  one JSON per run
  bench_compare.py        the nightly Tier A regression gate over two runs
```

## What each one is for

**`export_model.py`** is the on-ramp: with `--hf` it imports a Llama-class Hugging Face checkpoint directory (NumPy only — the safetensors container is parsed by hand; `--hf-parity` and `--text-corpus` are optional tier-2 extras on torch + transformers / tokenizers), and it converts a PyTorch `nn.Sequential`
(or a decoder stack — pre-embedded via `export_decoder_smf`, or token-native
SMF v4 via `export_token_decoder_smf` + `export_token_sds`) into the SMF
model container and turns arrays into an SDS corpus — the only place
PyTorch and NumPy are needed (`pip install -r tool/requirements-pinned.txt`
for the stack it is gated on, CPython 3.14.7 with torch 2.14 and NumPy 2.5;
`tool/requirements.txt` states the floors it still runs on, down to Python
3.9; the token-native path and `--demo-decoder` need NumPy only). It
streams: the container is written header-first with each tensor produced
straight from the framework's array as it goes out, and corpora go out as
chunked packed records, so export memory is about the model's own size and
corpus size does not matter. Its demo generators are fully
parameterizable from the command line (`--width`, `--depth`, `--vocab`,
`--seq-len`, `--blocks`, `--seed`, `--samples`, `--corpus-kind`), and
`--corpus` converts saved NumPy arrays into an SDS corpus without writing
Python; a flag that cannot apply to the requested mode is a hard error,
matching the compiler CLI's discipline. Everything that reaches the device
is dependency-free C++; the build host may additionally run the Python
packer below, which leaves no trace in the package.

**`seeml_update_compile.cc`** is the compiler itself as a command: source
model in, `.seeu` plan (and optional self-contained native package) out. Its
argument parsing is deliberately *strict* — an unknown flag, a flag missing
its value, or a numeric with trailing garbage is a hard error, never a
silent default — because a fine-tune you didn't mean to configure is worse
than one that refuses to start.

**`pack_update.py`** is the package assembler, the first Python-plane
subsystem of the Two-Plane Overhaul (`docs/next-project/`): given a
directory the compiler emitted with `--no-embed`, it embeds the plan as an
`.incbin` assembly stub (`update_plan_embedded.S`, page-aligned) in place
of the decimal C-array TU, and with `--build` runs the package's own
`build.sh`. The host compiler never parses weight bytes, so a 135M-parameter
package builds in seconds instead of minutes and the plan lands on disk
once. Standard library only (it sits on the compile path), strict CLI
(exit 2 on any unknown flag), atomic writes. `build.sh` prefers the stub
when present, so it also upgrades unmodified packages from older compilers.

**`seeml_seeu_dump.cc`** is the disassembler: point it at any plan and it
verifies the integrity seal and prints the header and instruction streams in
human-readable form — the tool you reach for when a plan misbehaves and you
need to see what the compiler actually emitted.

**`seeml_bench.cc`** is the metric program of
[docs/benchmarks.md](../docs/benchmarks.md) as a command: it compiles the
standard seeded fixture set in-process, measures per-step latency by
steps-regression at every requested thread width (with the engine's
fwd/bwd/optimizer split when built with `-DSEEML_BENCH=ON` or
`SEEML_BENCH=1 sh build/build.sh`), and emits one JSON per run.
`bench_compare.py` diffs two such runs and fails on a >10% Tier A
regression — the nightly `bench` job's gate.

## Where to go next

The full workflow — export, compile, run on-device — with **every flag
explained and every exit code**, is in
**[docs/usage.md](../docs/usage.md)**; start there. What the compile step
does internally is in [docs/compiler.md](../docs/compiler.md); the formats
these tools read and write are in [docs/formats.md](../docs/formats.md). The
compile CLI's argument discipline is verified alongside the driver suites
under [test/compiler/driver/](../test/README.md); the packer's suite is
[test/tool/pack_update_test.py](../test/tool/pack_update_test.py)
(`python3 -m unittest discover -s test/tool -p '*_test.py'`).
