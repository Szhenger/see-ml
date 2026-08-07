# Building SeeML

## What does it even mean to "build" a program?

You've written source code — text files ending in `.cc` and `.h`. A computer
can't run text; it runs machine code. **Building** is the process of turning
one into the other, and it happens in two distinct steps that are worth
keeping separate in your head:

1. **Compile.** The compiler (`clang++` or `g++`) reads *one* `.cc` file at a
   time and translates it into an **object file** (`.o`) — machine code, but
   *incomplete*: it still has holes where it calls functions defined in other
   files. Sixty source files means sixty object files, each compiled on its
   own, in ignorance of the others.
2. **Link.** The linker takes all those object files and stitches them
   together — filling every hole with the address of the function it needs —
   into a single runnable **executable**.

So the whole story is: *many `.cc` → many `.o` (compile) → one program
(link).* Every build system, no matter how fancy, is just automating those
two steps.

## Why not just type the compiler commands yourself?

You could. For one file, `clang++ hello.cc -o hello` is the whole build. But
SeeML has ~90 source files and two dozen test programs. Typing ninety compile
commands and then a link command by hand, every time you change one line,
would be miserable and error-prone. Worse, if you change *one* file, you'd
want to recompile only *that* file and re-link — not redo all ninety.

A **build system** is a program that knows the list of files, the commands,
and (sometimes) which outputs are stale, so `build` becomes one word instead
of ninety commands. SeeML gives you two, and they produce the identical
result.

## What's actually in this folder

Almost nothing here is source code. This directory is mostly the *output* of
a build — the `.o` object files and the `seeml_*` executables land here — and
all of that is git-ignored, because it's regenerated, never edited. **The one
file you'd ever open is `build.sh`.** (The other build system, CMake, is
driven by `CMakeLists.txt` up in the repository root.)

```
build/
  build.sh              the CMake-free build (the only tracked file here)
  *.o                   compiled object files      ── generated, git-ignored
  seeml-update-compile  the compiler CLI           ── generated
  seeml-seeu-dump       the plan disassembler      ── generated
  seeml_*_test          one executable per suite   ── generated
```

## The two ways to build

**Option A — `build.sh` (no tools required).** A plain shell script. It needs
nothing but a C++ compiler — no build tool installed at all. Read it top to
bottom and you see *every* command it runs: it compiles each `.cc` into a
`.o`, groups the object files into the three libraries the project is made of,
then links the tools and every test suite. It is deliberately transparent —
when a build breaks on some unfamiliar machine, the entire build is one file
you can read.

```bash
sh build/build.sh
```

**Option B — CMake (the industry standard).** CMake reads `CMakeLists.txt`,
figures out the compile/link commands for *your* platform and compiler, and
writes them out for a tool like `make` or `ninja` to run. It does the things a
hand-written script won't: rebuild only what changed, integrate with IDEs and
debuggers, run the test suite through `ctest`, and switch on the sanitizer and
fuzzing builds used in CI.

```bash
cmake -S . -B build-cmake      # -S = source dir, -B = where to put the build
cmake --build build-cmake -j   # -j = compile files in parallel
ctest --test-dir build-cmake   # run every test suite
```

(Note the CMake build goes in a *separate* `build-cmake/` directory. Keeping
generated files out of your source tree — an "out-of-source build" — is a
convention worth adopting: you can delete the whole thing to start clean, and
never accidentally commit a build artifact.)

## Reading a compile command

Every compile in `build.sh` uses the same flags. They're worth knowing:

| flag | what it does |
|---|---|
| `-std=c++23` | use the 2023 C++ standard (the codebase relies on `std::expected`) |
| `-I.` | look for `#include "..."` files starting at the repository root |
| `-O2` | optimize the machine code (level 2 — a good speed/compile-time balance) |
| `-Wall -Wextra` | turn on warnings — the compiler telling you about likely mistakes |
| `-Werror` | treat every warning as a hard error, so none can be ignored |
| `-pthread` | enable threads (the runtime parallelizes its kernels) |
| `-c` | *compile only* — stop after making the `.o`, don't link yet |
| `-o NAME` | name the output file |

## After it builds

Every test suite is its own executable; run one directly:

```bash
./build/seeml_updater_test
```

The two tools are here too — `./build/seeml-update-compile` and
`./build/seeml-seeu-dump`.

## Where to go next

This is *building the compiler*. A different, later build appears once the
compiler runs: the update package it emits contains *its own* generated
`build.sh` that compiles the vendored runtime on the device — that flow, and
every command-line flag, is in **[docs/usage.md](../docs/usage.md)**. The two
build systems here mirror each other file-for-file; the test tree they compile
is mapped in [test/README.md](../test/README.md).
