#!/usr/bin/env python3
"""Gate a seeml-bench run against stored baselines (stdlib only).

Usage:
    python3 bench_compare.py baseline.json current.json
        [--max-regression F] [--epoch-baseline epoch.json]
        [--max-epoch-regression F] [--state gate_state.json] [--promote]

Compares the Tier A throughput metric — rows_per_s, per fixture per thread
width — and exits 1 if any pair regressed by more than --max-regression
(rows_per_s is MLX-LM's "Tokens/sec" under its own definition — every row
of a SeeML batch is a loss target — and schema-2 reports also carry it as
tokens_per_s / samples_per_s; the gate keeps the schema-1 key so a baseline
stored before the rename still compares)
(default 0.10, the >10% gate of docs/benchmarks.md). A missing baseline
file exits 0 with a note: the first run of a new host seeds the baseline
rather than failing it. Fixtures or thread widths present on only one side
are reported and skipped — adding a fixture must not fail the night it
lands — but a comparison with NO overlapping keys is an error (exit 1),
not a pass: after a key rename the gate would otherwise be blind forever
while printing "no regression".

The rolling baseline (last green night) catches step changes; a slow
compounding drift of <10%/night would never trip it. --epoch-baseline names
a second, pinned report that is never rolled forward — the gate also diffs
against it (--max-epoch-regression, default 0.15) so drift accumulates
against a fixed reference. A missing epoch file is a note, not a failure
(the workflow seeds it once).

Everything is medians-of-medians upstream, so a >10% delta is signal, not
noise — on ONE machine. The nightly gate compares machines of one runner
class, which differ by 15-35% on identical work: one fast runner sets a
high-water baseline and every ordinary runner after it fails (Nightly
#20-#23, and 09-13/09-14 on the commit that was green on 09-12), while a
real regression is indistinguishable from the noise. Two defences (P5, #79):

Calibration. A schema-4 report carries `calibration`: the rate of a frozen
single-threaded reference kernel that shares no code with the runtime,
timed before and after the fixtures. Each Tier A delta is taken after
dividing the current rows/s by (current calibration / reference
calibration), so a uniformly slow or fast runner cancels and a slow SeeML
kernel does not. A reference that predates schema 4 compares un-normalized,
with a note (the migration pattern of the rows_per_s key above); a schema-4
report WITHOUT a usable calibration block, two different calibration
kernels, or a ratio outside [1/3, 3] (not the same host class) are errors,
never silent raw comparisons.

Two-consecutive-red. With --state PATH a key that regresses for the first
time is a WARNING (exit 0; a `::warning::` annotation under GitHub Actions)
and is recorded; the gate fails only when the same key regresses against
the same reference on the next run too. A key that recovers is forgotten.
Without --state every red fails at once, as before. --promote makes this
tool the one that rolls the baseline forward: the current report replaces
the baseline only on a fully green gate (or when no baseline exists), so a
warned run is never what the second night is compared against; a missing
--epoch-baseline is seeded under the same condition and never rewritten.

Schema-3 reports name the kernel policy they ran under (the CPU GEMM
tiles the compiler wrote into every plan); two reports with different
policies are never compared — that delta is what tool/autotune.py
measures, not a regression — and the gate exits 1 telling you to re-seed.
"""
import argparse
import json
import os
import sys


def tier_a(report):
    """{(fixture, threads, backend): rows_per_s} from a seeml-bench JSON
    object. The backend joins the key (per-backend determinism doctrine:
    a metal row is never compared against a cpu row); reports that predate
    the field are cpu, the only backend that existed."""
    out = {}
    backend = report.get("backend", "cpu")
    for name, fx in report.get("fixtures", {}).items():
        for t, m in fx.get("threads", {}).items():
            out[(name, t, backend)] = float(m["rows_per_s"])
    return out


def kernel_policy(report):
    """The kernel policy a run measured under: None for the runtime's
    compiled-in defaults (every report that predates schema 3, and every
    schema-3 report whose policy source is "default"), else the explicit
    (gemm_tile_k, gemm_tile_n). A tuned run (tool/autotune.py's table, or
    --gemm-tiles) is never gated against a default baseline, and vice
    versa: the delta would be the tiling, not the commit."""
    kp = report.get("kernel_policy")
    if not kp or kp.get("source", "default") == "default":
        return None
    return (int(kp["gemm_tile_k"]), int(kp["gemm_tile_n"]))


class GateError(Exception):
    """A comparison the gate refuses to make (fail closed, exit 1)."""


CALIBRATION_SCHEMA = 4      # the first seeml-bench schema that calibrates
MAX_CALIBRATION_RATIO = 3.0  # beyond this the two hosts are not one class


def calibration(report, path):
    """(kernel, rows_per_s) of a report's calibration block, or None for a
    report that predates schema 4. A schema-4 report whose block is absent
    or unusable is a GateError: the harness that wrote it always
    calibrates, so its absence is a truncated or edited report."""
    block = report.get("calibration")
    if int(report.get("schema", 1)) < CALIBRATION_SCHEMA and block is None:
        return None
    try:
        kernel, rate = str(block["kernel"]), float(block["calib_rows_per_s"])
    except (TypeError, KeyError, ValueError):
        raise GateError(f"{path} is a schema-{report.get('schema')} report "
                        "without a usable calibration block")
    if not kernel or not rate > 0.0 or rate == float("inf"):
        raise GateError(f"{path} carries a non-positive or non-finite "
                        "calibration rate")
    return kernel, rate


class Report:
    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)
        self.path = path
        self.tier_a = tier_a(raw)
        self.policy = kernel_policy(raw)
        self.calibration = calibration(raw, path)


def runner_ratio(label, ref, cur):
    """current-runner speed over the reference runner's, from the two
    calibration rates; 1.0 (un-normalized, with a note) when either side
    predates calibration."""
    if ref.calibration is None or cur.calibration is None:
        old = label if ref.calibration is None else "current report"
        print(f"bench_compare: NOTE the {old} predates schema "
              f"{CALIBRATION_SCHEMA} (no calibration) — comparing "
              f"un-normalized against the {label}")
        return 1.0
    (ref_kernel, ref_rate), (cur_kernel, cur_rate) = (ref.calibration,
                                                      cur.calibration)
    if ref_kernel != cur_kernel:
        raise GateError(f"the {label} calibrated with '{ref_kernel}' but the "
                        f"current report with '{cur_kernel}'; the rates are "
                        "not comparable — re-seed the baseline deliberately")
    ratio = cur_rate / ref_rate
    if not 1.0 / MAX_CALIBRATION_RATIO <= ratio <= MAX_CALIBRATION_RATIO:
        raise GateError(f"the current runner calibrates at {ratio:.2f}x the "
                        f"{label}'s — not the same host class; re-seed the "
                        "baseline deliberately")
    return ratio


def key_name(key):
    return f"{key[0]}@{key[1]}t/{key[2]}"


def compare(label, ref, cur, max_regression):
    """Prints per-key lines; returns ({key_name: line} of regressed keys,
    compared_count). Deltas are runner-normalized (see runner_ratio)."""
    base, now = ref.tier_a, cur.tier_a
    ratio = runner_ratio(label, ref, cur)
    shared = [k for k in base if k in now and base[k] > 0]
    if ratio != 1.0 or (ref.calibration and cur.calibration):
        # The median raw key ratio is printed beside the calibration ratio
        # as a diagnostic only: normalizing by it would hide a regression
        # that slows every fixture at once.
        raw = sorted(now[k] / base[k] for k in shared)
        mid = (f", median raw key ratio {raw[len(raw) // 2]:.3f}x"
               if raw else "")
        print(f"bench_compare: NOTE runner calibrates at {ratio:.3f}x the "
              f"{label}'s{mid}; deltas below are normalized")
    failures, skipped, compared = {}, [], 0
    for key, base_v in sorted(base.items()):
        if key not in now:
            skipped.append(f"{key_name(key)}: in {label} only")
            continue
        compared += 1
        cur_v = now[key]
        norm_v = cur_v / ratio
        delta = (norm_v - base_v) / base_v if base_v > 0 else 0.0
        shown = (f"{cur_v:.0f}" if ratio == 1.0
                 else f"{cur_v:.0f} (normalized {norm_v:.0f})")
        line = (f"{key_name(key)}: {base_v:.0f} -> {shown} rows/s "
                f"({delta:+.1%}) vs {label}")
        if delta < -max_regression:
            failures[key_name(key)] = line
        else:
            print(f"bench_compare: OK   {line}")
    for key in sorted(now.keys() - base.keys()):
        skipped.append(f"{key_name(key)}: new, not in {label}")
    for s in skipped:
        print(f"bench_compare: SKIP {s}")
    return failures, compared


STATE_SCHEMA = 1


def load_state(path):
    """{label: [key_name, ...]} — the keys that were red on the previous
    run. A missing file is an empty state; an unreadable one is an error
    (a gate that forgets its reds silently never fails)."""
    if path is None or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            state = json.load(f)
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError(f"schema {state.get('schema')!r}")
        return {label: list(keys) for label, keys in state["red"].items()}
    except (OSError, ValueError, KeyError, AttributeError, TypeError) as e:
        raise GateError(f"cannot read the gate state {path}: {e}")


def write_atomic(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def judge(label, failures, previous, use_state):
    """Splits one reference's regressed keys into (fatal, warned) lines and
    prints them. Without --state every red is fatal."""
    fatal, warned = [], []
    for name, line in sorted(failures.items()):
        if not use_state or name in previous.get(label, []):
            fatal.append(line)
            suffix = " (second consecutive red)" if use_state else ""
            print(f"bench_compare: FAIL {line}{suffix}")
        else:
            warned.append(line)
            print(f"bench_compare: WARN {line} (first red — fails if it "
                  "repeats next run)")
            if os.environ.get("GITHUB_ACTIONS") == "true":
                print(f"::warning title=bench gate, first red::{line}")
    return fatal, warned


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline")
    p.add_argument("current")
    p.add_argument("--max-regression", type=float, default=0.10,
                   help="fractional Tier A loss vs the rolling baseline "
                        "that fails the gate (0.10)")
    p.add_argument("--epoch-baseline", metavar="PATH", default=None,
                   help="pinned report that is never rolled forward; "
                        "guards against compounding drift")
    p.add_argument("--max-epoch-regression", type=float, default=0.15,
                   help="fractional Tier A loss vs the epoch baseline that "
                        "fails the gate (0.15)")
    p.add_argument("--state", metavar="PATH", default=None,
                   help="gate state carried between runs: a key's first red "
                        "is a warning, its second consecutive red fails")
    p.add_argument("--promote", action="store_true",
                   help="on a fully green gate (or a missing baseline) "
                        "replace the baseline with the current report and "
                        "seed a missing epoch baseline")
    args = p.parse_args()
    try:
        return gate(args)
    except GateError as e:
        print(f"bench_compare: ERROR {e}")
        return 1


def gate(args):
    cur = Report(args.current)
    if not cur.tier_a:
        raise GateError(f"current report {args.current} carries no Tier A "
                        "metrics")

    def check_policy(label, ref):
        if ref.policy == cur.policy:
            return
        # Fail closed: a delta between two tilings is a tuning result, not a
        # regression signal — re-seed the baseline with the policy in use.
        def name(p):
            return "the runtime defaults" if p is None else f"GEMM tiles {p}"
        raise GateError(f"the {label} was measured with {name(ref.policy)} "
                        f"but the current report with {name(cur.policy)}; "
                        "the gate compares like with like — re-seed the "
                        "baseline deliberately")

    previous = load_state(args.state)
    use_state = args.state is not None
    red, fatal, warned = {}, [], []

    def against(label, path, max_regression, what):
        ref = Report(path)
        check_policy(label, ref)
        failures, compared = compare(label.split()[0], ref, cur,
                                     max_regression)
        if compared == 0:
            # Fail closed: zero overlap means the gate measured nothing.
            raise GateError("no Tier A metric is present in both the "
                            f"{label} and the current report — the gate "
                            "would be blind; re-seed the baseline "
                            "deliberately if the keys were renamed")
        f, w = judge(label.split()[0], failures, previous, use_state)
        if f:
            print(f"bench_compare: {len(f)} Tier A metric(s) regressed more "
                  f"than {max_regression:.0%} vs {what}")
        red[label.split()[0]] = sorted(failures)
        fatal.extend(f)
        warned.extend(w)

    seeded = not os.path.exists(args.baseline)
    if seeded:
        print(f"bench_compare: no baseline at {args.baseline} — "
              "this run seeds it")
    else:
        against("baseline", args.baseline, args.max_regression,
                "the rolling baseline")

    if args.epoch_baseline is not None:
        if not os.path.exists(args.epoch_baseline):
            print(f"bench_compare: no epoch baseline at "
                  f"{args.epoch_baseline} — this run seeds it")
        else:
            against("epoch baseline", args.epoch_baseline,
                    args.max_epoch_regression,
                    "the pinned epoch baseline (compounding drift)")

    if use_state:
        write_atomic(args.state, json.dumps(
            {"schema": STATE_SCHEMA, "red": red}, indent=2, sort_keys=True)
            + "\n")
    if fatal:
        return 1
    if warned:
        print(f"bench_compare: {len(warned)} Tier A metric(s) red for the "
              "first time — warning only; the baseline is not rolled "
              "forward")
        return 0
    print("bench_compare: no Tier A regression")
    if args.promote:
        with open(args.current) as f:
            text = f.read()
        write_atomic(args.baseline, text)
        print(f"bench_compare: promoted {args.current} to {args.baseline}")
        if (args.epoch_baseline is not None
                and not os.path.exists(args.epoch_baseline)):
            write_atomic(args.epoch_baseline, text)
            print(f"bench_compare: seeded the epoch {args.epoch_baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
