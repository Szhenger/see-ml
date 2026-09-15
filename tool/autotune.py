#!/usr/bin/env python3
"""The offline autotuner: measure kernel-policy arms on this host, persist
the winner in a host-keyed table the compiler consumes (stdlib only).

Usage:
    python3 tool/autotune.py tune --bench build/seeml-bench --out table.json
        [--fixtures a,b,...] [--threads 1,8] [--backend cpu]
        [--arms 64x256,128x512,...] [--rounds 3] [--repeats 3]
        [--steps-lo 20] [--steps-hi 80] [--min-gain 0.03] [--keep-runs DIR]
    python3 tool/autotune.py show table.json [--host KEY]

Why this is a build-host tool and not a compiler pass: tuning is
measurement, and a measurement made inside the compiler is a measurement
the compiler cannot reproduce or explain. The Two-Plane Overhaul
(docs/next-project/README.md, P2) moves it here, offline, where it can run
for minutes, keep every number it saw, and hand the compiler one decided
fact per host. The compiler never measures: `seeml-update-compile
--kernel-policy table.json` looks this host up and writes the tiles into
the plan header (v11); no table, no entry, or no flag means the runtime's
defaults, and compile time is unchanged either way.

What an arm is: a CPU GEMM tile geometry `KxN` (the K panel a pass over C
folds in, the N sweep width — runtime/executor/kernel_policy.h). Every arm
computes bit-identical results (the K tile must be a multiple of 4, which
the bench and the compiler both refuse otherwise), so the table only ever
picks among equivalent schedules and determinism is untouched. The sweep
always includes two arms explicitly, whatever `--arms` says: the kernel
defaults (the arm to beat — a table that cannot beat it records that) and
the compiler's analytic tiling `SuggestGemmTiling` (a hypothesis worth
measuring, never trusted: it ran 1.3–3.3x slower than the defaults on the
unpacked kernels, #90).

How it measures, per docs/benchmarks.md: every arm is one `seeml-bench`
run (medians of `--repeats` per fixture and thread width, by
steps-regression); arms are visited round-robin for `--rounds` rounds so
thermal drift lands on every arm alike; each arm's per-key result is the
median over rounds — medians of medians. An arm's score is the geometric
mean, over every (fixture, threads) key, of its rows/s relative to the
default arm's, so a fixture never outvotes another by being larger. The
winner is the highest score, and it is written as the host's policy only
when it beats the default by at least `--min-gain` (3 %, the "signal, not
noise" line of the benchmarks doc); otherwise the default is recorded,
with every arm's numbers beside it, so the decision is measured either way.

Table schema (version 1), the contract with
compiler/backend/tuner/kernel_policy_table.cc, which reads `hosts.<key>.cpu`
and ignores the rest:

    {"schema": 1,
     "hosts": {"<host key>": {"cpu": {"gemm_tile_k": K, "gemm_tile_n": N},
                              "tuned": {...every measurement...}}}}

The host key is the string the bench printed (`host_key` in its JSON,
`HostKey()` in host_arch.h): ISA, CPU model, cores, L1d, L2, SIMD width —
this tool copies it, never computes it, so the compiler and the bench
cannot disagree about who a host is. Exit codes: 0 tuned (or default
recorded), 1 a bench run or the table failed, 2 usage.
"""
import argparse
import datetime
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile

SCHEMA = 1
DEFAULT_GRID_K = (32, 64, 128, 256)
DEFAULT_GRID_N = (64, 128, 256, 512)


class TuneError(Exception):
    """A failure that ends the run with exit 1 and the message."""


# --- Arms ------------------------------------------------------------------

def parse_arm(text):
    """'KxN' -> (K, N); K a positive multiple of 4, N positive."""
    parts = text.lower().split("x")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise argparse.ArgumentTypeError(
            f"arm '{text}' is not KxN (two whole numbers)")
    k, n = int(parts[0]), int(parts[1])
    if k <= 0 or k % 4 or n <= 0:
        raise argparse.ArgumentTypeError(
            f"arm '{text}': K must be a positive multiple of 4, N positive")
    return (k, n)


def parse_arms(text):
    return [parse_arm(a) for a in text.split(",") if a]


def arm_name(arm):
    return f"{arm[0]}x{arm[1]}"


def default_grid():
    return [(k, n) for k in DEFAULT_GRID_K for n in DEFAULT_GRID_N]


# --- Driving the bench -----------------------------------------------------

def bench_command(args, out_path, arm):
    cmd = [args.bench, "--out", out_path, "--threads", args.threads,
           "--steps-lo", str(args.steps_lo), "--steps-hi", str(args.steps_hi),
           "--repeats", str(args.repeats), "--backend", args.backend]
    if args.fixtures:
        cmd += ["--fixtures", args.fixtures]
    if arm is not None:
        cmd += ["--gemm-tiles", f"{arm[0]},{arm[1]}"]
    return cmd


def run_bench(args, arm, run_index, keep_dir):
    """One seeml-bench run; returns its parsed JSON. `arm` None = the
    defaults (no --gemm-tiles), which the report then names."""
    if keep_dir:
        out_path = os.path.join(
            keep_dir, f"run{run_index:03d}_{arm_name(arm) if arm else 'default'}.json")
        holder = None
    else:
        holder = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        holder.close()
        out_path = holder.name
    cmd = bench_command(args, out_path, arm)
    print("autotune: " + " ".join(cmd), file=sys.stderr, flush=True)
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
    except OSError as e:
        raise TuneError(f"cannot run the bench '{args.bench}': {e}")
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise TuneError(f"bench exited {proc.returncode} for arm "
                        f"{arm_name(arm) if arm else 'default'}")
    try:
        with open(out_path) as f:
            report = json.load(f)
    except (OSError, ValueError) as e:
        raise TuneError(f"cannot read the bench report {out_path}: {e}")
    finally:
        if holder is not None:
            os.unlink(out_path)
    for key in ("host_key", "kernel_policy", "fixtures"):
        if key not in report:
            raise TuneError(f"bench report lacks '{key}' — the harness "
                            "predates schema 3 (rebuild it)")
    return report


def tier_a(report):
    """{'fixture@threads': rows_per_s} for one run."""
    out = {}
    for name, fx in report["fixtures"].items():
        for t, m in fx.get("threads", {}).items():
            out[f"{name}@{t}"] = float(m["rows_per_s"])
    return out


def policy_of(report):
    kp = report["kernel_policy"]
    return (int(kp["gemm_tile_k"]), int(kp["gemm_tile_n"]))


# --- Scoring ---------------------------------------------------------------

def geometric_mean(values):
    if not values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def score_arms(per_arm_runs, default_arm):
    """per_arm_runs: {arm: [ {key: rows_per_s}, ... ]} over rounds.
    Returns {arm: {'rows_per_s': {key: median}, 'speedup': {key: ratio},
    'score': gmean}} with ratios relative to the default arm's medians."""
    medians = {}
    for arm, runs in per_arm_runs.items():
        keys = set(runs[0])
        for r in runs:
            if set(r) != keys:
                raise TuneError(f"arm {arm_name(arm)}: rounds disagree on the "
                                "fixture/thread keys measured")
        medians[arm] = {k: statistics.median(r[k] for r in runs) for k in keys}
    base = medians[default_arm]
    scored = {}
    for arm, med in medians.items():
        ratios = {}
        for k, v in med.items():
            if k not in base or base[k] <= 0:
                raise TuneError(f"default arm carries no usable rows/s for {k}")
            ratios[k] = v / base[k]
        scored[arm] = {"rows_per_s": med, "speedup": ratios,
                       "score": geometric_mean(ratios.values())}
    return scored


def choose(scored, default_arm, min_gain):
    """(chosen_arm, decision string)."""
    best = max(scored, key=lambda a: (scored[a]["score"], a == default_arm))
    gain = scored[best]["score"] - 1.0
    if best == default_arm:
        return default_arm, "default-is-best"
    if gain < min_gain:
        return default_arm, (f"default-within-margin: best arm "
                             f"{arm_name(best)} gains {gain:.1%}, below "
                             f"--min-gain {min_gain:.1%}")
    return best, f"tuned: {arm_name(best)} gains {gain:.1%} over the default"


# --- The table --------------------------------------------------------------

def load_table(path):
    """The existing table, or a fresh one. A file that exists but is not a
    schema-1 table is an error: silently replacing someone's table is the
    kind of default this tool must not have."""
    if not os.path.exists(path):
        return {"schema": SCHEMA, "hosts": {}}
    try:
        with open(path) as f:
            table = json.load(f)
    except (OSError, ValueError) as e:
        raise TuneError(f"existing table {path} cannot be read: {e}")
    if (not isinstance(table, dict) or table.get("schema") != SCHEMA
            or not isinstance(table.get("hosts"), dict)):
        raise TuneError(f"existing table {path} is not a schema-{SCHEMA} "
                        "kernel-policy table")
    return table


def write_table(path, table):
    """Atomic: a crash mid-write leaves the previous table intact."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".kernel_policy.", suffix=".tmp",
                               dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(table, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except OSError as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise TuneError(f"cannot write the table {path}: {e}")


# --- Commands ---------------------------------------------------------------

def cmd_tune(args):
    if args.rounds < 1 or args.repeats < 1:
        raise TuneError("--rounds and --repeats must be >= 1")
    if args.steps_hi <= args.steps_lo:
        raise TuneError("--steps-hi must exceed --steps-lo")
    if args.min_gain < 0:
        raise TuneError("--min-gain must be >= 0")
    keep_dir = args.keep_runs
    if keep_dir:
        os.makedirs(keep_dir, exist_ok=True)

    # Round 1 opens with the default arm: its report names the host, the
    # defaults it ran, and the analytic tiling — the two arms every sweep
    # carries — so the arm list is fixed by measurement, not by guessing.
    runs_done = 0
    first = run_bench(args, None, runs_done, keep_dir)
    runs_done += 1
    host_key = first["host_key"]
    default_arm = policy_of(first)
    if first["kernel_policy"].get("source") != "default":
        raise TuneError("the bench did not run the defaults when asked to; "
                        "is a kernel-policy table leaking in?")
    analytic = first.get("analytic_gemm_tiles")
    analytic_arm = ((int(analytic["k"]), int(analytic["n"]))
                    if analytic and int(analytic["k"]) % 4 == 0 else None)

    requested = args.arms if args.arms is not None else default_grid()
    arms = [default_arm]
    if analytic_arm and analytic_arm not in arms:
        arms.append(analytic_arm)
    for a in requested:
        if a not in arms:
            arms.append(a)
    labels = {default_arm: "default"}
    if analytic_arm:
        labels.setdefault(analytic_arm, "analytic")

    per_arm_runs = {a: [] for a in arms}
    per_arm_runs[default_arm].append(tier_a(first))
    for rnd in range(args.rounds):
        for arm in arms:
            if rnd == 0 and arm == default_arm:
                continue  # measured above
            report = run_bench(args, arm, runs_done, keep_dir)
            runs_done += 1
            if report["host_key"] != host_key:
                raise TuneError("the host key changed between runs "
                                f"({host_key!r} vs {report['host_key']!r})")
            if policy_of(report) != arm:
                raise TuneError(f"the bench ran {policy_of(report)} when asked "
                                f"for {arm}")
            per_arm_runs[arm].append(tier_a(report))
    for arm, runs in per_arm_runs.items():
        if len(runs) != args.rounds:
            raise TuneError(f"arm {arm_name(arm)} has {len(runs)} rounds, "
                            f"expected {args.rounds}")

    scored = score_arms(per_arm_runs, default_arm)
    chosen, decision = choose(scored, default_arm, args.min_gain)

    ordered = sorted(arms, key=lambda a: -scored[a]["score"])
    print(f"autotune: host {host_key}", file=sys.stderr)
    for a in ordered:
        print(f"autotune:   {arm_name(a):>9}  score {scored[a]['score']:.4f}"
              f"  {labels.get(a, 'grid')}", file=sys.stderr)
    print(f"autotune: {decision}", file=sys.stderr)

    entry = {
        "cpu": {"gemm_tile_k": chosen[0], "gemm_tile_n": chosen[1]},
        "tuned": {
            "date": datetime.datetime.now(datetime.timezone.utc)
                        .replace(microsecond=0).isoformat(),
            "seeml_version": first.get("seeml_version"),
            "host": first.get("host"),
            "host_arch": first.get("host_arch"),
            "backend": first.get("backend"),
            "bench": os.path.abspath(args.bench),
            "fixtures": sorted({k.split("@")[0] for k in tier_a(first)}),
            "threads": args.threads,
            "rounds": args.rounds,
            "repeats": args.repeats,
            "steps_lo": args.steps_lo,
            "steps_hi": args.steps_hi,
            "min_gain": args.min_gain,
            "decision": decision,
            "default_arm": arm_name(default_arm),
            "analytic_arm": arm_name(analytic_arm) if analytic_arm else None,
            "arms": [
                {"gemm_tile_k": a[0], "gemm_tile_n": a[1],
                 "label": labels.get(a, "grid"),
                 "score": round(scored[a]["score"], 6),
                 "rows_per_s": {k: round(v, 3)
                                for k, v in sorted(scored[a]["rows_per_s"].items())},
                 "speedup": {k: round(v, 6)
                             for k, v in sorted(scored[a]["speedup"].items())}}
                for a in ordered],
        },
    }
    table = load_table(args.out)
    table["hosts"][host_key] = entry
    write_table(args.out, table)
    print(f"autotune: wrote {args.out}: {host_key} -> "
          f"{arm_name(chosen)} ({decision.split(':')[0]})", file=sys.stderr)
    return 0


def cmd_show(args):
    table = load_table(args.table)
    hosts = table["hosts"]
    if args.host is not None:
        if args.host not in hosts:
            raise TuneError(f"no entry for host {args.host!r}")
        hosts = {args.host: hosts[args.host]}
    if not hosts:
        print("(empty table)")
    for key, entry in hosts.items():
        cpu = entry.get("cpu", {})
        tuned = entry.get("tuned", {})
        print(f"{key}")
        print(f"  cpu: gemm tiles K {cpu.get('gemm_tile_k')} "
              f"N {cpu.get('gemm_tile_n')}")
        if tuned:
            print(f"  tuned {tuned.get('date')} on {tuned.get('host')}: "
                  f"{tuned.get('decision')}")
            for arm in tuned.get("arms", []):
                print(f"    {arm['gemm_tile_k']}x{arm['gemm_tile_n']:<6} "
                      f"score {arm['score']:.4f}  {arm.get('label', '')}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="autotune.py",
        description="Measure CPU GEMM tile arms with seeml-bench and persist "
                    "the winner in a host-keyed kernel-policy table.")
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tune", help="sweep arms on this host and update the table")
    t.add_argument("--bench", required=True, help="path to seeml-bench")
    t.add_argument("--out", required=True, help="the kernel-policy table to "
                   "create or update (this host's entry is replaced)")
    t.add_argument("--fixtures", default="", help="comma-separated bench "
                   "fixtures (default: the bench's default set)")
    t.add_argument("--threads", default="1,8", help="bench --threads (1,8)")
    t.add_argument("--backend", default="cpu", help="bench --backend (cpu)")
    t.add_argument("--arms", type=parse_arms, default=None,
                   help="comma-separated KxN arms; the defaults and the "
                        "analytic tiling are always added (default: a "
                        "K in {32,64,128,256} x N in {64,128,256,512} grid)")
    t.add_argument("--rounds", type=int, default=3,
                   help="round-robin passes over the arms (3)")
    t.add_argument("--repeats", type=int, default=3, help="bench --repeats (3)")
    t.add_argument("--steps-lo", type=int, default=20, help="bench --steps-lo")
    t.add_argument("--steps-hi", type=int, default=80, help="bench --steps-hi")
    t.add_argument("--min-gain", type=float, default=0.03,
                   help="minimum geometric-mean speedup over the default "
                        "arm for a tuned entry (0.03)")
    t.add_argument("--keep-runs", metavar="DIR", default=None,
                   help="keep every bench report under DIR")
    t.set_defaults(func=cmd_tune)
    s = sub.add_parser("show", help="print a table")
    s.add_argument("table")
    s.add_argument("--host", default=None, help="only this host key")
    s.set_defaults(func=cmd_show)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except TuneError as e:
        print(f"autotune: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
