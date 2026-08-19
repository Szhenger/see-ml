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
  seeml_seeu_dump.cc      the plan disassembler   (inspect any .seeu)
```

## What each one is for

**`export_model.py`** is the on-ramp: it converts a PyTorch `nn.Sequential`
(or a decoder stack — pre-embedded via `export_decoder_smf`, or token-native
SMF v4 via `export_token_decoder_smf` + `export_token_sds`) into the SMF
model container and turns arrays into an SDS corpus — the only Python in
the product, and the only place PyTorch and NumPy are needed
(`pip install -r tool/requirements.txt`; the token-native path and
`--demo-decoder` need NumPy only). Its demo generators are fully
parameterizable from the command line (`--width`, `--depth`, `--vocab`,
`--seq-len`, `--blocks`, `--seed`, `--samples`, `--corpus-kind`), and
`--corpus` converts saved NumPy arrays into an SDS corpus without writing
Python; a flag that cannot apply to the requested mode is a hard error,
matching the compiler CLI's discipline. Everything downstream is
dependency-free C++.

**`seeml_update_compile.cc`** is the compiler itself as a command: source
model in, `.seeu` plan (and optional self-contained native package) out. Its
argument parsing is deliberately *strict* — an unknown flag, a flag missing
its value, or a numeric with trailing garbage is a hard error, never a
silent default — because a fine-tune you didn't mean to configure is worse
than one that refuses to start.

**`seeml_seeu_dump.cc`** is the disassembler: point it at any plan and it
verifies the integrity seal and prints the header and instruction streams in
human-readable form — the tool you reach for when a plan misbehaves and you
need to see what the compiler actually emitted.

## Where to go next

The full workflow — export, compile, run on-device — with **every flag
explained and every exit code**, is in
**[docs/usage.md](../docs/usage.md)**; start there. What the compile step
does internally is in [docs/compiler.md](../docs/compiler.md); the formats
these tools read and write are in [docs/formats.md](../docs/formats.md). The
compile CLI's argument discipline is verified alongside the driver suites
under [test/compiler/driver/](../test/README.md).
