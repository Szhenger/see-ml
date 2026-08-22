#!/usr/bin/env python3
"""Gate a seeml-bench run against a stored baseline (stdlib only).

Usage:
    python3 bench_compare.py baseline.json current.json [--max-regression F]

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
lands. Everything is medians-of-medians upstream, so a >10% delta is
signal, not noise.
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline")
    p.add_argument("current")
    p.add_argument("--max-regression", type=float, default=0.10,
                   help="fractional Tier A loss that fails the gate (0.10)")
    args = p.parse_args()

    if not os.path.exists(args.baseline):
        print(f"bench_compare: no baseline at {args.baseline} — "
              "this run seeds it")
        return 0
    with open(args.baseline) as f:
        base = tier_a(json.load(f))
    with open(args.current) as f:
        cur = tier_a(json.load(f))

    failures, skipped = [], []
    for key, base_v in sorted(base.items()):
        if key not in cur:
            skipped.append(f"{key[0]}@{key[1]}t: in baseline only")
            continue
        cur_v = cur[key]
        delta = (cur_v - base_v) / base_v if base_v > 0 else 0.0
        line = (f"{key[0]}@{key[1]}t: {base_v:.0f} -> {cur_v:.0f} rows/s "
                f"({delta:+.1%})")
        if delta < -args.max_regression:
            failures.append(line)
        else:
            print(f"bench_compare: OK   {line}")
    for key in sorted(cur.keys() - base.keys()):
        skipped.append(f"{key[0]}@{key[1]}t: new, no baseline")
    for s in skipped:
        print(f"bench_compare: SKIP {s}")
    for f_ in failures:
        print(f"bench_compare: FAIL {f_}")
    if failures:
        print(f"bench_compare: {len(failures)} Tier A metric(s) regressed "
              f"more than {args.max_regression:.0%}")
        return 1
    print("bench_compare: no Tier A regression")
    return 0


if __name__ == "__main__":
    sys.exit(main())
