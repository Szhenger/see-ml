# Workflows — An Introduction from First Principles

*This is SeeML's continuous integration. Let's take it from the top.*

You've just changed some code. It compiles on your machine. The tests pass
on your machine. You open a pull request, someone merges it, and three days
later a teammate on Linux discovers the build is broken, a kernel produces
different bits at eight threads than at one, and a fuzzer-shaped user has
crashed the model loader with a file your tests never imagined.

Every one of those failures was *detectable at the moment of the merge*.
Nobody detected them, because detection was a human's job, and humans
forget. The entire idea of this document is one sentence:

> **A workflow is a program whose job is to run your other programs'
> checks, every time, so that no human has to remember to.**

That's it. Everything else is mechanics. But the mechanics matter, so
let's build them up from nothing.

---

## Part 0 — What problem are we actually solving?

SeeML makes three promises, stated all over its documentation:

1. **Correctness** — a compiled update plan does what the math says: the
   gradients match calculus, the bounds checks hold against hostile
   bytes, the committed model is the trained model.
2. **Determinism** — the same plan, data, and seed produce the *same
   bits* at any thread count, on any run. Not "approximately the same" —
   bitwise identical.
3. **Boundedness** — where behavior genuinely varies (two writers racing
   to a file, a fuzzer throwing garbage, threads interleaving), the
   *envelope* of outcomes is still guaranteed: one complete file wins,
   the loader rejects instead of crashing, races never corrupt.

Here's the thing about promises like these: they decay. Every diff is an
opportunity to silently break one. The test suites in `test/` are the
*assertions* of these promises — 240 of them at last count — but a test
that nobody runs is documentation, not verification.

So we need a machine that runs them. On every change. On computers that
are *not* your laptop, because your laptop is one operating system, one
compiler, one CPU count, and the promises are supposed to hold everywhere.

That machine is GitHub Actions, and the programs you write for it are
called **workflows**.

---

## Part 1 — What is a workflow, mechanically?

A workflow is a YAML file in the magic directory `.github/workflows/`.
When certain **events** happen in the repository — someone opens a pull
request, pushes to `main`, or a scheduled time arrives — GitHub reads
these files and executes them. SeeML has three:

```
.github/workflows/
  ci.yml        runs on every pull request and push to main
  nightly.yml   runs every night, and on demand
  codeql.yml    runs on every pull request, push, and weekly
```

Let's dissect the anatomy with a real (simplified) excerpt from `ci.yml`:

```yaml
name: CI                        # what the badge and the checks tab call it

on:                             # WHEN does this run?
  pull_request:                 #   every time a PR is opened or updated
  push:
    branches: [main]            #   and every push to main

jobs:                           # WHAT runs? A set of named jobs...
  build-and-test:               #   ...this one is called build-and-test
    runs-on: ubuntu-latest      #   on a fresh Linux virtual machine
    steps:                      #   as a sequence of steps:
      - uses: actions/checkout@v4      # step 1: clone the repo
      - name: Build
        run: sh build/build.sh         # step 2: run a shell command
      - name: Run every suite
        run: |
          for t in build/seeml_*_test; do "$t" || exit 1; done
```

Five concepts, and you know them all already in other forms:

- An **event** (`on:`) is a trigger — the moment the machine wakes up.
  Think of an interrupt handler, or a database trigger.
- A **job** is an independent unit of work that gets its own **runner** —
  a disposable virtual machine, born clean, destroyed after. Jobs run *in
  parallel* with each other by default. That's why SeeML's CI is split
  into six jobs instead of one long script: six machines working at once,
  and when one fails you know *which promise* broke from the job's name
  alone.
- A **step** is one command (or one reusable **action**, like
  `actions/checkout` — someone else's packaged steps) inside a job.
  Steps run *in sequence*, and the job stops at the first failure.
- A **matrix** (you'll see it below) is a for-loop over jobs: write the
  job once, run it per operating system × compiler combination.
- The **exit code** is the entire verdict. Zero means the step passed;
  anything else fails the step, the job, and the green checkmark. This
  is why every test binary, every `cmp`, every `grep -q` in these
  workflows matters — each one is a proposition, and its exit code is
  the proof or the counterexample.

One more idea and the mechanics are done. This line appears near the top
of `ci.yml`:

```yaml
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
```

If you push twice in quick succession, the first run is now testing dead
code. This says: cancel it, test the newest. The `${{ ... }}` syntax is
the workflow's variable interpolation — here, the branch name — so each
branch gets its own cancellation group.

That's the whole language. Now the interesting part: *what do we choose
to check?*

---

## Part 2 — Mapping promises to jobs

Recall the three promises. Here is the entire CI surface of SeeML, one
row per job, and — this is the design — **every row exists to guard a
named promise**:

| workflow | job | promise guarded |
|---|---|---|
| ci.yml | `build-and-test` (×3 toolchains) | correctness, everywhere |
| ci.yml | `determinism` (widths 1/3/8) | determinism |
| ci.yml | `asan-ubsan` | correctness (memory / UB) |
| ci.yml | `fuzz-smoke` | boundedness (hostile input) |
| ci.yml | `e2e-package` | correctness + determinism of the *product* |
| ci.yml | `exporter-syntax` | correctness (the Python edge) |
| nightly.yml | `tsan` | boundedness (races) |
| nightly.yml | `fuzz-extended` | boundedness (hostile input, deeper) |
| codeql.yml | `analyze` | correctness (paths the tests never run) |

If a job goes red, the *name of the job* tells you which promise your
diff endangered. That is not an accident; that is the specification.

Let's take them one at a time.

---

## Part 3 — `ci.yml`: the per-diff gate

### 3.1 `build-and-test` — does it even work, everywhere?

```yaml
strategy:
  matrix:
    include:
      - { os: ubuntu-latest, cxx: g++ }
      - { os: ubuntu-latest, cxx: clang++ }
      - { os: macos-latest, cxx: clang++ }
```

The README promises "a C++23 compiler (clang or gcc)". A promise about
*two compilers and two operating systems* cannot be kept by testing one
of each — so the matrix expands this job into three, each on its own
fresh machine, each running the canonical build (`build/build.sh`, plain
`sh`, no CMake required) and then all 25 test suites.

Why does multi-compiler matter for *correctness*, not just convenience?
Because compilers disagree about exactly the things that hide bugs:
which warnings fire (the build uses `-Werror`, so a warning *is* a
failure), how aggressively undefined behavior is exploited, how floats
are contracted. Code that is correct C++ passes both; code that merely
*happens to work* on clang often does not survive gcc, and you want to
learn that at the pull request, not in a user's bug report.

The macOS cell earns its keep twice: it also compiles the Objective-C++
Metal GPU harness (`runtime/executor/metal_gemm.mm`), which exists only
on Apple platforms. One subtlety worth internalizing: GitHub's macOS
runners are virtual machines that may expose **no Metal device**. The
test suite was written for exactly this: with no device it prints a note
and passes vacuously. A check that cannot run *must* say so and step
aside — a skipped check that pretends to be a passed check is how CI
lies to you.

### 3.2 `determinism` — same bits at every width

Here is the job in one breath: build once, then run **every suite three
times** — under `SEEML_THREADS=1`, `=3`, and `=8`.

To see why, you need one fact about SeeML's design (the full story is in
[runtime.md](runtime.md)): parallel work is chunked by *problem shape*,
never by thread count. The thread count decides who computes each chunk,
never what the chunks are — so every result is supposed to be bitwise
identical whether one thread ran or eight did.

Many of the 240 tests assert exactly that ("compile this plan twice,
compare bytes"; "train serially and widely, compare persistent
segments"). But an assertion inside a program can only test the
configurations the program was run in. This job supplies the
configurations: fully serial (width 1 — no worker thread is ever
created, the degenerate case), an *odd* width (3 — chunk counts that
don't divide evenly, the awkward case), and wide (8 — real contention).
A diff that sneaks any scheduling-dependent numeric into a kernel —
an accumulation order that depends on which thread got there first —
trips at least one width.

Ask yourself: why not width 2 and 4? Because 3 is the interesting one.
Even widths tend to divide chunk counts neatly; oddness exercises the
remainders. When you design a matrix, you are choosing *which* points of
a space to sample — sample the awkward points.

### 3.3 `asan-ubsan` — the bugs that don't crash

Some bugs are polite enough to crash. The dangerous ones aren't: a read
one byte past a buffer usually *works*; casting `NaN` to `int8_t` is
undefined behavior that usually *produces some number*. Your tests pass.
The corruption ships.

**Sanitizers** are compiler modes that plant tripwires: AddressSanitizer
(ASan) shadows every allocation and faults on the first out-of-bounds
touch; UndefinedBehaviorSanitizer (UBSan) instruments every operation
the C++ standard leaves undefined and reports the violation at the
moment it happens. The cost is a few× slowdown, which is why this is its
own job rather than the default build.

This job builds the whole tree with both (`-DSEEML_SANITIZE=
"address;undefined"`, plumbing the repo's CMake already carries) and
runs all suites. The environment line `UBSAN_OPTIONS: halt_on_error=1`
is load-bearing: by default UBSan *warns and continues*, and a warning
nobody reads is that lying-CI problem again. Halting makes the exit code
carry the verdict.

Historical note, because it's instructive: a code review of this project
found a real `int8_t(std::clamp(NaN, ...))` — silent corruption on
every compiler, invisible to every test, and *precisely* the species
UBSan converts into a red build. This job exists so that class of bug
can never return quietly.

### 3.4 `fuzz-smoke` — the inputs you didn't think of

Every test you write is an input you thought of. The model loader, the
plan loader, and the dataset loader parse *files* — and files come from
outside, where nobody thinks the way you do.

A **fuzzer** generates inputs by mutation, guided by coverage: it
watches which branches each input reaches and mutates the inputs that
reached new ones. Over time it discovers, mechanically, the malformed
files that your hand-written rejection tests missed. SeeML's harness
(`test/fuzz/binary_formats.cc`) exposes four attack surfaces — the SMF
model reader, the SEEU plan loader, the SDS dataset reader, and the
compiler running on loader-accepted models — and the contract under
test is absolute: *arbitrary bytes may be rejected but must never
crash, overflow, or over-read.*

Per pull request this job runs **90 seconds** of fuzzing. Ninety seconds
is not a security audit — it is a smoke test that catches *shallow*
regressions (an unguarded length field, a reordered bounds check) while
staying affordable on every diff. Depth is the nightly's job (Part 4).
If it ever fails, the crashing input itself is uploaded as an artifact,
so the counterexample lands in your hands, reproducible.

### 3.5 `e2e-package` — the product contract, literally

Everything so far tests the *repository*. But SeeML's actual product is
a thing the repository emits: a self-contained package that must build
and run **with no access to this repository at all** (that's the
vendoring contract in [runtime.md](runtime.md)). Until this job, that
claim was tested structurally — "the files are present" — never
operationally.

This job performs the entire documented user journey, for real:

1. `pip install torch` (CPU build) and export the demo model + corpus
   with `tool/export_model.py --demo` — the Python/C++ format seam,
   exercised end to end.
2. Compile the plan and emit the package; `--build` then compiles the
   package *using its own generated `build.sh`*, exactly as a user
   would on a device.
3. Verify the plan's integrity seal with the disassembler
   (`grep -q "(verified)"` — remember, exit codes are propositions).
4. Run `model_update`: load, validate, train, gate, merge, commit.
5. Run it **again**, fully serial (`SEEML_THREADS=1`), and `cmp` the two
   committed models. They must be *bitwise identical* — the determinism
   promise, asserted not on a kernel or a test fixture but on the final
   artifact a user would deploy.

One design decision deserves a highlight. The demo's regression gate
might (deterministically) decide the training didn't improve and refuse
to commit — exit code 3, which is *correct behavior*, not a bug. The
job distinguishes it: exit 0 proceeds, exit 3 retries with `--force`
(because the invariant under test is build/run/commit/determinism, not
the demo's convergence), and anything else fails loudly. When you write
checks, decide up front which outcomes are failures and which are
merely *answers* — conflating them produces flaky CI, and flaky CI
trains humans to ignore red, which destroys the entire enterprise.

### 3.6 `exporter-syntax` — the cheapest possible tripwire

`python3 -m py_compile tool/export_model.py`. Five seconds. It catches
exactly one class of regression — the exporter no longer parses — and
costs nearly nothing. Not every check needs to be deep; it needs to be
*proportionate*. (Deliberately absent: installing PyTorch here — the
e2e job already exercises the exporter for real.)

---

## Part 4 — `nightly.yml`: what's too slow for every diff

Two checks are worth running but too slow (or too statistical) to make
every contributor wait for. They run on a schedule —
`cron: "17 7 * * *"`, which is cron notation for "07:17 UTC daily" —
plus `workflow_dispatch`, meaning you can click "run now."

**`tsan`** rebuilds everything under **ThreadSanitizer**, which watches
the ordering of every memory access across threads and reports *data
races* — two threads touching the same location, at least one writing,
with no synchronization between them. Races are the ultimate
"works on my machine" bug: their outcome depends on scheduling luck.
SeeML's test tree already contains genuine race *scenarios* run in
anger — two engines training concurrently on the shared worker pool,
two writers racing one durable file, a batch pipeline torn down
mid-stream — and TSan runs *underneath* those tests, converting "the
assertions held this time" into "no race existed on any interleaving
TSan observed." That's the boundedness promise, proven rather than
sampled. At ~10× slowdown, it's a nightly.

**`fuzz-extended`** is the same fuzzer as 3.4 with a budget of fifteen
minutes — and one crucial upgrade: the **corpus** (the set of
interesting inputs discovered so far) is cached between runs via
`actions/cache`. Tonight's fuzzing starts where last night's stopped.
Coverage compounds. This is the difference between poking at a lock for
90 seconds and hiring someone to pick it a little further every night.

A nightly failure has no pull request attached — it means **main
itself** regressed (or a latent bug finally surfaced). Treat it with
red-PR severity; the only thing worse than a broken main is a broken
main nobody is looking at.

---

## Part 5 — `codeql.yml`: checking the paths nothing executes

Everything above is **dynamic** analysis: run the code, observe it.
Dynamic analysis has one blind spot by construction — it only sees the
paths that actually execute. The error path with the off-by-one, the
integer conversion in a branch no fixture reaches: invisible.

**CodeQL** is the complement: **static** analysis. It compiles the code
(tracing the same `build/build.sh` as everything else, so the analyzed
tree is exactly the shipped tree), builds a database of every function,
call, and dataflow edge, and runs queries against it — "does any
attacker-influenced length reach an allocation without a bounds
check?", "is any signed value narrowed where it could wrap?" — across
*all* paths, executed or not. Findings appear in the repository's
Security tab with the dataflow spelled out step by step.

It runs per pull request, per push to main, and — because query suites
themselves improve over time — on a weekly schedule, so the *same* code
gets re-examined by *smarter* queries as they ship.

Neither static nor dynamic analysis subsumes the other: sanitizers
prove facts about real executions; CodeQL reasons about hypothetical
ones. You want both, which is why both exist here.

---

## Part 6 — Reading a red build

The practical skill. A check failed on your PR — now what?

1. **The job name is the diagnosis.** `determinism` red? You introduced
   a scheduling-dependent numeric. `asan-ubsan` red but `build-and-test`
   green? Memory or UB bug that happens not to crash. `e2e-package` red
   alone? You broke the emitted package or the CLI seam, not the
   library. Only gcc red? You wrote clang-flavored C++.
2. **Reproduce locally with the same knob.** Every job is a wrapper
   around commands you can run yourself: `SEEML_THREADS=3
   ./build/seeml_kernels_test`, or the sanitizer build via
   `cmake -DSEEML_SANITIZE="address;undefined"`. The workflows
   deliberately contain no logic that exists only in CI.
3. **Artifacts are your counterexample.** A fuzz failure uploads the
   crashing file; feed it back to the harness locally and you have a
   deterministic reproducer.
4. **Never retry your way to green.** Every check here is seeded and
   deterministic *by design* (that's promise #2 doing double duty: it
   makes CI itself non-flaky). If a rerun changes the verdict, that
   contradiction is itself a determinism bug — file it as one.

---

## Part 7 — Takeaways

- A workflow is just a program that runs your checks on every change,
  because humans forget and machines don't.
- Structure follows promises: **one job per invariant**, so a red check
  names what broke before you read a single log line.
- Exit codes are propositions. Design every step so its exit code
  *means* something, and never let a check that couldn't run
  impersonate a check that passed.
- Sample the awkward points: odd thread widths, hostile bytes, missing
  devices, both compilers.
- Split by cost: cheap and deterministic gates every diff; slow and
  statistical runs nightly; static analysis covers what dynamic
  execution can't reach.
- Determinism isn't only a product feature — it's what makes the CI
  itself trustworthy. Flaky checks train people to ignore red, and an
  ignored red is worse than no check at all.

Where to go next: the invariants themselves are specified in
[compiler.md](compiler.md) and [runtime.md](runtime.md); the test tree
these workflows execute is mapped in [test/README.md](../test/README.md);
the binary formats the fuzzer attacks are in [formats.md](formats.md).

*This was workflows.*
