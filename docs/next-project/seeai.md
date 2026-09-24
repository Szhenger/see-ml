# SeeAI — the design (2026-09-24)

*The source of truth for the **SeeAI v1.0.0.B** milestone. The issue
bodies in [`issues/`](issues/) (`s*.md`, `cr*.md`) carry a root cause and an
acceptance each and cite a section here; they do not repeat it. The
frontier-parity milestone SeeAI v1.0.0.A (F1–F9) is unchanged by this
document and is referenced where the two meet.*

## §1 Mission

SeeAI exists because engineers are being disconnected from the ML systems
and fine-tuning techniques underneath them by complexity. The compiler keeps
its three subsystems — frontend, analysis, backend — and each one prints,
from first principles, the quantitative statistics of what it just did.
**SeeAI == no leaky abstractions.** The functional purpose is fine-tuning an
open-weight language model to the user's own data, on a consumer device.
Security is applied at every layer, as in OS design: every input is assumed
malformed or ill-intentioned, and every check the build host runs is re-run
on the device.

Scope: **language models** on consumer devices. Convolution lowering, the
mean-squared-error loss family and feature-row datasets leave the product;
the MLP fixtures stay only as test canaries (a subset of the LM op surface).

## §2 Statistics screens (the doctrine)

Every subsystem produces one statistics struct; the console output is a
rendering of the same struct the compile report serializes (one source of
truth, the P6 seam rule applied to statistics). One screen per subsystem by
default, per-tensor detail behind a flag. Every number carries its unit and
its derivation in one clause; FLOPs are summed from the plan's own GEMM
instructions, bytes from the arena binder's segments, the loss floor from
the corpus. Golden statistic blocks for the fixtures are tested. Screens
print aggregates, hashes and losses — never record contents. The backend
screen ends with a predicted step time at the host's peak; the runtime
prints the measured step beside it, and the ratio is the utilization.

## §3 Frontend — six sub-subsystems on two paths

| stage | role | today | artifact |
|---|---|---|---|
| **ingressor** | bounded SMF reader, weight copy, content hash | `frontend/ingressor/model_reader.cc` | decoded model + hash |
| **accountant** | weights + activations lower bound vs the budget; corpus statistics before and after sanitization; the identity manifest | `ingressor/resource_analyzer.cc` | estimate + manifest |
| **topology** | well-formedness: dependency order, unique names, output exists, every shape agrees with its op; cross-input checks (student↔teacher, corpus↔model, adapter↔weight, tokenizer vocab == embedding rows == head width) | `parser/sema.cc` (per-op shape checks are interleaved in the build loop today and must be lifted out so topology completes first) | verdict + inferred shapes |
| **computation** | mechanical translation of the verified op list into the SIR computation graph, then `Block::verify()`; teacher prefixed `t::` | `parser/parser.cc`, `value_resolver.cc`, `graph_build.h` | the SIR forward graph |
| **tokenizer** | the training subsystem: raw text in; normalize → sanitize/filter → deduplicate (exact, by hash) → encode (byte-level BPE, zero-dependency) → pack (concatenate with EOS, cut fixed S+1 windows, never pad) | new | canonical SDS + manifest of drops + hash |
| **validation** | "what counts": the held-out split (seeded partition after dedup, or `--val-data` through the same tokenizer; seed, indices and hashes into the plan) and the loss mask (chat template from the SMF tokenizer section; response positions are targets) | new; today a runtime `--val-frac` | two canonical corpora with masks |

Shared infrastructure beside the six, not stages: `representation/` (SIR),
`operator/` (OpBuilder, also used by analysis). `model_writer` is egress and
moves out of `ingressor/`.

Frontend doctrine lines:
- The accountant's footprint is a **lower bound**; the backend's arena is
  exact; the screen prints the estimate as an estimate and later the ratio.
- The plan binds to the corpus **contract** and the **tokenizer hash**, not
  to corpus bytes: the device may train on data the build host never saw.
  The corpus hash is provenance in the manifest; an opt-in flag pins it.
- The tokenizer definition comes **only** from a new SMF tokenizer section
  (vocabulary, merges, pre-tokenizer pattern, normalization, special tokens,
  chat template), written by the exporter and proven bit-exact against the
  reference tokenizer on a golden adversarial corpus at export. Its hash
  binds every corpus on both layers (plan header, SDS header). Byte-level
  BPE first; SentencePiece refused loudly until implemented.
- Sanitizer, contract check and tokenizer are pure deterministic C++
  functions in `source/`, vendored into the package so the feeder re-runs
  them on device data. The contract function leaves `runtime/engine/`.
- Threat model, stated on the screen: the sanitizer covers malformed bytes,
  out-of-contract records, degenerate records and exact duplicates. It does
  **not** cover well-formed poisoned data; the gate and a held-out set the
  training source cannot influence are that defense.
- The recipe is a declarative file with a schema; flags override fields; the
  resolved recipe is hashed into the manifest with the model, tokenizer and
  corpus hashes. The exporter stays in Python and SMF stays canonical; the
  frontend **verifies** the exporter's output (hash, tokenizer parity,
  logits parity in the manifest). Native safetensors ingress is a follow-up.

## §4 Analysis — five directories named for the mathematics

Boundary: the frontend knows what the model *is*; analysis knows what
*training* is; the backend knows what the *machine* is and cannot tell an
adapter from an activation. Analysis owns which ops happen, in what order,
at what precision; the runtime owns the arithmetic of each op.

| directory | field | holds |
|---|---|---|
| **calculus** | differentiation | autodiff (seed handed in as a parameter), the masked cross-entropy VJP, rematerialization machinery |
| **algebra** | value-preserving rewrites | LoRA graft and merge, the three fusers, the RoPE table; a fusion is an algebraic identity whose payoff is fewer arena round trips — the screen prints bytes removed per fusion; the attention segment mask for packed windows |
| **statistics** | decisions from measured numbers | quantization review, attention tiling, precision assignment (`kFlagRelaxed`), the adapter policy with per-site parameters and FLOP share |
| **topology** | the graph's structure | DCE (reachability), verify-after-every-pass, the reduction-order ledger: per pass, bit-identical to its input or not, and under which certificate |
| **optimization** | minimization, not compiler optimization | SGD/AdamW synthesis, the schedule and horizon, clipping, the grad-accumulation 1/G seed, 8-bit moments; screen: effective-LR curve, steps at the floor, epochs covered, predicted update-to-weight ratio at step 1 |

The pass manager sits at the analysis root; `updater/` and `reviewer/`
disappear; conv lowering is deleted. The analysis screen: trainable
fraction, adjoints, saved-activation bytes, FLOPs forward/backward/optimizer,
bytes removed per fusion, dead ops, the ledger.

**Best-practice defaults.** A default may change the *algorithm*, never the
*arithmetic*: anything that reorders reductions (flash attention, bf16
compute) is a default only inside the certified opt-in family. The memory
row (base storage, moments, recompute) is a function of the accountant's
numbers, not a constant. Every default is a measured claim: no flip without
a frontier row (F1) at the same setting; defaults live in the versioned
recipe, each citing its field report. Audit of `source/plan/config.h`
against frontier LoRA practice: AdamW already; learning rate 1e-3 constant →
1e-4–2e-4 with cosine decay and 3–10 % warmup; clipping off → norm 1.0;
α/r → α/√r; masked loss; the rest are F2, F3, F5.

## §5 Backend — six phases

| phase | does | today |
|---|---|---|
| **architecture** | target description (ISA, SIMD, cores, caches, host key; a GPU half: device family, cores, Metal 4 capability, peak); analytic tiling; reads the measured kernel-policy table | `architecture/host_arch.cc` + `tuner/kernel_policy_table.cc` |
| **allocation** | arena binder: three segments, SSA liveness, first-fit, rodata layout | `trainer/arena_binder.cc` |
| **selection** | SIR op → one 64-byte instruction; ISA loses MSE and conv, gains masked xent and the attention segment operand | `trainer/instruction_lowering.cc` |
| **scheduling** (candidate) | instruction order for the target; earns a directory only if F8 (#139) measures the step time outside kernels as real; schedule before allocate | none |
| **assembly** | plan layout: header with the identity manifest, kernel policy and reduction-order ledger; streams; rodata written once; seeded persistent image; emit table; seal | the driver's persist-init / assemble / seal phases (misfiled) |
| **packaging** | the self-contained package: embedded plan, generated main, vendored runtime **plus the shared `source/` contract check, sanitizer and tokenizer**, build script | `trainer/native_emitter.cc`; Python twin `tool/pack_update.py` |

`tuner/` folds into architecture; `kernel_emitter.cc` (tests only since the
runtime owns the kernel library) retires or moves its GPU tiling clamp under
architecture. The compiler emits data, the runtime executes code: a kernel
is vendored, never generated, and every compile-time GPU decision is a
permission (the `kFlagRelaxed` model), never a requirement, because the
build host is not the device.

## §6 Runtime — nine roles, in lifecycle order

**host** (package entry: generated main, flags, exit codes, cooperative
cancellation; thread and affinity policy) → **loader** (map the plan, seal
hash, header, source-model hash, resource contract vs the device, the one
arena allocation; manifest, tokenizer hash and ledger checked here) →
**verifier** (load-time proof of every instruction) → **pipeline** (input
pipeline: decode, shuffle, prefetch; runs the vendored contract check,
sanitizer and tokenizer on device data) → **dispatcher** (the VM dispatch
loop over the train/eval/merge programs) on the **executor** (kernel library
and backends behind the executor seam) → **gating** (evaluate before/after,
best state, patience, exit 0 or 3) → **storage** (checkpoints, the durable
atomic commit) — **profiler** throughout (step split, per-kernel spans, peak
RSS, energy per token; measured beside predicted) and **diagnostics**
cross-cutting. Today `engine/` holds loader, dispatcher, gating and profiler
together; `validator/`, `feeder/`, `custodian/` are verifier, pipeline,
storage. Shared infrastructure outside: `source/parallel` (the deterministic
pool), `source/plan`, `source/identity`, and a `source/platform/` seam to be
created.

## §7 Python's five roles

Python **ingests** (the data format zoo into one canonical text container),
**proves** (per-pass oracles against torch and f64 NumPy; tokenizer parity),
**measures** (policy tables keyed on the host: kernel tiles today, a fusion
policy next), **searches** (adapter policy, schedule → a recipe) and
**certifies** (relaxed precision today, quantization error next). Never in
the default compile path, never on the device, never a second copy of a
contract. Python subsystems keep the core's footprint discipline.

## §8 The 2026-09-24 code review, by root cause

96 confirmed findings (4 high) at main e023b8b; full text in the "SeeML Code
Review" page. Owners: (1) no non-finite policy → `cr1`; (2) contracts that
skip a program or operand → `cr2`; (3) silent acceptance of CLI input →
`s4`; (4) certification and packaging holes → F2 #130; (5) tests and CI that
cannot fail → `cr3`; (6) durability and egress edges → `cr4`; (7) platform
and project tooling → fixed in the PR that lands this document
(`create_next_project.sh` identity by number, portable mktemp, no SIGPIPE).
Review artifacts never filed until now: the Style Audit's seven items →
`s8`; the Two-Plane Port Review's doc-drift list → `s8`.

## §9 Follow-ups deliberately not in this milestone

Structured pruning at the on-ramp (one-shot importance scoring, recovery by
an ordinary fine-tune; unstructured pruning out until the runtime skips
zeros); preference tuning (DPO: reference-model log-probabilities = the
teacher path with a pairwise VJP); native safetensors ingress; a CUDA
backend behind the executor seam with Triton as ahead-of-time authoring only
(no JIT on device), for the NVIDIA consumer target after Apple silicon and
phones and only with a Windows story (WSL2 first; only storage's atomic
commit proof truly needs a Win32 port); warm-start adapter import.
