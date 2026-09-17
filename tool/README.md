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
  autotune.py             the offline tuner      ->  host-keyed kernel-policy table
  frontier_exec.py        the frontier executor  ->  runs a .seeu through NumPy / PyTorch / MLX
  seeml_plan_probe.cc     its C++ oracle half     (one plan section, every write traced)
  certify_numerics.py     the numerics certifier ->  numerics_certificate.json, or a refusal
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
regression — the nightly `bench` job's gate. It is runner-normalized (each
report carries the rate of a frozen calibration kernel, and the comparer
divides the two runners' ratio out of every delta) and, with `--state`,
two-consecutive-red: a key's first regression warns, its second fails
(P5, #79). Standard library only.

**`frontier_exec.py`** is the off-ramp to the frontier frameworks, the
fourth Python-plane subsystem (P4): a build-host interpreter for the whole
plan instruction set, written a second time from the documented mathematics
rather than ported from the kernels. `diff` replays a plan against
`seeml-plan-probe --trace` (the probe also times a section, `--time`, and attributes it per opcode and GEMM shape, `--profile`) one instruction at a time — each opcode executes
on exactly the bytes the C++ runtime had and is judged on its own writes —
so it catches the kernel bug a bit-exact self-comparison would reproduce
faithfully, and names the instruction that owns it; it runs against the CPU
and the Metal backend alike. `price` answers "what would Accelerate/AMX or
the GPU buy on THIS plan" in `seeml-bench`'s own fields (`step_ms`,
`tokens_per_s`, `it_per_s`, `gemm_gflops`, `mfu`) beside the C++ numbers
from `seeml-plan-probe --time`; `run` trains a plan with the feeder's exact
split and shuffle. NumPy is the only requirement; `torch`, `torch-mps`,
`mlx` and `mlx-tf32` are reported unavailable, not failed, where absent.
The priced number is a floor on the frontier, not the frontier: the
interpreter dispatches op by op, with no graph compile or fusion. Never
shipped, never imported by the compiler or the runtime.

**`certify_numerics.py`** is the gate in front of any relaxed-reduction
plan (P3): *no certificate, no relaxed plan*. It trains the plan twice
through the frontier executor on the plan's own operands (its frozen
weights, the real corpus, the adapters as they move) — once with every
reduction exact, once with every reduction inside a float64-reducing opcode
(norms, losses, softmaxes, attention scores, the clip norm) recomputed in
float32 over W strided lanes — and records the worst per-site output error
on identical inputs, the rounding bound that holds for *any* summation
order of those operands (so the certificate survives whatever schedule a
kernel author picks), and the free-run loss and validation deviation with
the update gate's verdict under both contracts. It grants or refuses
against `--max-site-error` / `--max-loss-deviation`, binds the result to
the plan and corpus by SHA-256, and `verify` holds a measured run (a
`frontier_exec.py diff` report) to the certified tolerances.
`pack_update.py` refuses to package a plan beside a certificate that does
not vouch for it, so a recompile voids a certificate loudly. The relaxed
opcode family itself is core-plane work that does not exist yet: today the
certificate is the measured answer to "what would relaxing cost this
plan", and the contract that family will be admitted under.

**`autotune.py`** is the offline autotuner, the second Python-plane
subsystem of the overhaul (P2): it sweeps CPU GEMM tile arms through
`seeml-bench --gemm-tiles` — round-robin over rounds, medians of medians,
each arm scored by the geometric mean of its rows/s relative to the
kernel defaults, which are always in the sweep alongside the compiler's
analytic tiling — and writes the winner (or "the defaults are best", when
nothing clears `--min-gain`) into a JSON table keyed on the host key the
bench printed. `seeml-update-compile --kernel-policy table.json` then
looks the compile host up and writes the tiles into the plan header (v11),
where the runtime proves and runs them; every geometry computes the same
bits, so the table only ever chooses among equivalent schedules. Standard
library only, strict CLI, atomic table writes, `show` to read a table
back. The measurement never happens inside the compiler: no table means
the defaults and an unchanged compile time.

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
