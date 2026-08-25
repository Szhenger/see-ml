#!/usr/bin/env python3
"""Gate a seeml-bench run against stored baselines (stdlib only).

Usage:
    python3 bench_compare.py baseline.json current.json
        [--max-regression F] [--epoch-baseline epoch.json]
        [--max-epoch-regression F]

Compares the Tier A throughput metric — rows_per_s, per fixture per thread
width — and exits 1 if any pair regressed by more than --max-regression
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
noise.
"""
import argparse
import json
import os
import sys


def tier_a(report):
    """{(fixture, threads): rows_per_s} from a seeml-bench JSON object."""
    out = {}
    for name, fx in report.get("fixtures", {}).items():
        for t, m in fx.get("threads", {}).items():
            out[(name, t)] = float(m["rows_per_s"])
    return out


def load(path):
    with open(path) as f:
        return tier_a(json.load(f))


def compare(label, base, cur, max_regression):
    """Prints per-key lines; returns (failures, compared_count)."""
    failures, skipped, compared = [], [], 0
    for key, base_v in sorted(base.items()):
        if key not in cur:
            skipped.append(f"{key[0]}@{key[1]}t: in {label} only")
            continue
        compared += 1
        cur_v = cur[key]
        delta = (cur_v - base_v) / base_v if base_v > 0 else 0.0
        line = (f"{key[0]}@{key[1]}t: {base_v:.0f} -> {cur_v:.0f} rows/s "
                f"({delta:+.1%}) vs {label}")
        if delta < -max_regression:
            failures.append(line)
        else:
            print(f"bench_compare: OK   {line}")
    for key in sorted(cur.keys() - base.keys()):
        skipped.append(f"{key[0]}@{key[1]}t: new, not in {label}")
    for s in skipped:
        print(f"bench_compare: SKIP {s}")
    for f_ in failures:
        print(f"bench_compare: FAIL {f_}")
    return failures, compared


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
    args = p.parse_args()

    cur = load(args.current)
    if not cur:
        print(f"bench_compare: ERROR current report {args.current} carries "
              "no Tier A metrics")
        return 1

    status = 0
    if not os.path.exists(args.baseline):
        print(f"bench_compare: no baseline at {args.baseline} — "
              "this run seeds it")
    else:
        base = load(args.baseline)
        failures, compared = compare("baseline", base, cur,
                                     args.max_regression)
        if compared == 0:
            # Fail closed: zero overlap means the gate measured nothing.
            print("bench_compare: ERROR no Tier A metric is present in both "
                  "the baseline and the current report — the gate would be "
                  "blind; re-seed the baseline deliberately if the keys "
                  "were renamed")
            return 1
        if failures:
            print(f"bench_compare: {len(failures)} Tier A metric(s) regressed "
                  f"more than {args.max_regression:.0%} vs the rolling "
                  "baseline")
            status = 1

    if args.epoch_baseline is not None:
        if not os.path.exists(args.epoch_baseline):
            print(f"bench_compare: no epoch baseline at "
                  f"{args.epoch_baseline} — this run seeds it")
        else:
            epoch = load(args.epoch_baseline)
            failures, compared = compare("epoch", epoch, cur,
                                         args.max_epoch_regression)
            if compared == 0:
                print("bench_compare: ERROR no Tier A metric is present in "
                      "both the epoch baseline and the current report")
                return 1
            if failures:
                print(f"bench_compare: {len(failures)} Tier A metric(s) "
                      f"regressed more than {args.max_epoch_regression:.0%} "
                      "vs the pinned epoch baseline (compounding drift)")
                status = 1

    if status == 0:
        print("bench_compare: no Tier A regression")
    return status


if __name__ == "__main__":
    sys.exit(main())
