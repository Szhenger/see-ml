#!/usr/bin/env python3
"""Certify a plan for relaxed reductions, or refuse to (P3, #77).

The runtime's bitwise-determinism contract makes every reduction accumulate
in float64, in a fixed chunk order — half the SIMD width and a convert per
element on every norm, loss, softmax and attention score. A relaxed-reduction
opcode family (vectorized float32 accumulation, reassociated within one
backend's fixed schedule) would buy that back, at the price of cross-schedule
bit-equality. That price is spent consciously or not at all:

    no certificate, no relaxed plan.

This tool writes the certificate. It runs the plan twice over ITS OWN
operands — the frozen weights in the plan, batches from the real corpus,
the adapters as they actually train — through tool/frontier_exec.py's
interpreter:

  reference  every reduction exact (float64), the contract of today;
  relaxed    the same interpreter with every reduction inside a reduction
             site recomputed in float32 over W independent lanes (strided
             sequential accumulators, combined by a fixed pairwise tree —
             the shape of a SIMD reduction), everything else identical.

and records three kinds of evidence:

  lockstep   per site opcode, on identical inputs: the worst observed
             output error, relative to the output tensor's scale;
  a priori   per reduction, the rounding-error bound that holds for ANY
             summation order of those operands (Higham, Accuracy and
             Stability of Numerical Algorithms, s.3.1 and s.4.2:
             |fl(sum) - sum| <= gamma_n * sum|x_i|, gamma_n = nu/(1 - nu)),
             so the certificate does not depend on the modelled schedule
             being the one a kernel author eventually picks;
  free run   both arms trained for --steps from the same initial state on
             the same batches: the loss-curve deviation, both validation
             losses, and whether the update gate (validation loss must
             improve) holds under either contract.

The certificate is GRANTED only if every site's observed error is within
--max-site-error, the free-run deviations are within --max-loss-deviation,
and the gate verdict is the same in both arms. It binds to the plan by
SHA-256 and by the plan's own sealed hash, and to the corpus by SHA-256; it
carries a digest of itself. `verify` re-checks all of that against a plan,
and, given the measured deviations of a real relaxed run (--observed, a
frontier_exec.py diff report), that they are inside the certified
tolerances. tool/pack_update.py refuses to package a plan beside a
certificate that does not verify against it.

Usage:
    certify_numerics.py certify PLAN_OR_PACKAGE --corpus c.sds
        [--steps N] [--lanes W] [--seed S] [--val-fraction F]
        [--max-site-error F] [--max-loss-deviation F] [--out cert.json]
    certify_numerics.py verify cert.json --plan plan.seeu
        [--corpus c.sds] [--observed diff.json]

Exit codes: 0 granted / verified; 3 refused / not verified; 1 unusable
input; 2 bad arguments. Needs NumPy for `certify`; `verify` is standard
library only.
"""
import argparse
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frontier_exec as fx  # noqa: E402

CERT_SCHEMA = 1
CERT_KIND = "seeml-numerics-certificate"
CERT_FILE = "numerics_certificate.json"
PLAN_FILE = "update_plan.seeu"
FAMILY = "relaxed-reductions-f32"
UNIT_ROUNDOFF = 2.0 ** -24  # float32, round to nearest

# The opcodes whose C++ kernels reduce in float64 today — the only places a
# relaxed family would change arithmetic (runtime/executor: attention.cc,
# normalization.cc, loss.cc, optimizer.cc, elementwise.cc). The GEMMs are
# float32 accumulations already and are not sites.
SITES = {
    10: "reduce.rows", 11: "softmax_xent.fwd", 13: "mse.fwd",
    15: "kl_distill.fwd", 26: "layer_norm.fwd", 27: "layer_norm.bwd",
    28: "clip.norm", 31: "rms_norm.fwd", 32: "rms_norm.bwd",
    35: "attn.fwd", 36: "attn.dp", 38: "softmax_rows.bwd",
    # The tiled family (v15) recomputes the same double reductions.
    47: "attn.fwd.tiled", 48: "attn.dq.tiled", 49: "attn.dk.tiled",
    50: "attn.dv.tiled",
}
# The optimizer steps reduce only when they carry a fused clip (plan v12,
# threshold bits in out[1]): then the gradient norm is theirs.
FUSED_CLIP_SITES = {17: "sgd.step", 18: "adamw.step"}
SITES.update(FUSED_CLIP_SITES)
assert all(fx.OPCODES[k] == v for k, v in SITES.items())


def is_site(ins):
    return ins.opcode in SITES and (ins.opcode not in FUSED_CLIP_SITES or
                                    ins.out[1] != 0)


def gamma(n):
    """Higham's gamma_n for float32; inf where the bound is vacuous."""
    nu = n * UNIT_ROUNDOFF
    return nu / (1.0 - nu) if nu < 1.0 else math.inf


class LaneBackend(fx.NumpyBackend):
    """float64 everywhere except reductions, which run in float32 over
    `lanes` strided accumulators. Tracks, per reduction, the a-priori bound
    relative to the result's scale (worst over the current instruction)."""

    def __init__(self, lanes):
        super().__init__("float64")
        if lanes < 1 or lanes & (lanes - 1):
            raise fx.PlanError("--lanes must be a power of two")
        self.lanes, self.name = lanes, f"relaxed-f32x{lanes}"
        self.bound_rel = 0.0

    def _note(self, bound, result):
        scale = float(self.np.abs(result).max()) if result.size else 0.0
        if scale > 0.0:
            self.bound_rel = max(self.bound_rel,
                                 float(self.np.max(bound)) / scale)

    def _lane_sum(self, terms, axis):
        """sum over `axis` of float32 `terms`: `lanes` sequential float32
        accumulators over strided elements, then a pairwise tree."""
        np = self.np
        v = np.moveaxis(terms, axis, -1)
        n = v.shape[-1]
        pad = -n % self.lanes
        if pad:
            v = np.concatenate(
                [v, np.zeros(v.shape[:-1] + (pad,), np.float32)], -1)
        v = v.reshape(v.shape[:-1] + (-1, self.lanes))
        acc = np.cumsum(v, axis=-2, dtype=np.float32)[..., -1, :]
        while acc.shape[-1] > 1:
            acc = acc[..., 0::2] + acc[..., 1::2]
        return acc[..., 0]

    def sum(self, t, axis, keepdims=False):
        np = self.np
        terms = np.asarray(t).astype(np.float32)
        out = self._lane_sum(terms, axis)
        n = terms.shape[axis]
        self._note(gamma(n) * np.abs(terms.astype(np.float64)).sum(axis), out)
        out = out.astype(np.float64)
        return np.expand_dims(out, axis) if keepdims else out

    def mean(self, t, axis, keepdims=False):
        n = self.np.asarray(t).shape[axis]
        return self.sum(t, axis, keepdims) / n

    def matmul(self, a, b):
        """a @ b as float32 dot products, each lane-accumulated: the
        products are rounded once (hence gamma_{K+1}), then summed."""
        np = self.np
        a32 = np.asarray(a).astype(np.float32)
        b32 = np.swapaxes(np.asarray(b), -1, -2).astype(np.float32)
        products = a32[..., :, None, :] * b32[..., None, :, :]
        out = self._lane_sum(products, -1)
        k = a32.shape[-1]
        self._note(gamma(k + 1) * (np.abs(a32.astype(np.float64)) @
                                   np.abs(np.swapaxes(b32, -1, -2).astype(
                                       np.float64))), out)
        return out.astype(np.float64)


class DryMemory:
    """Reads through to a real Memory; captures writes instead of storing
    them — one instruction's relaxed result on the reference arm's inputs."""

    def __init__(self, mem):
        self.mem, self.writes = mem, []

    def read(self, ref, shape, kind="<f4"):
        return self.mem.read(ref, shape, kind)

    def frozen(self, *args):
        return self.mem.frozen(*args)

    def write(self, ref, tensor):
        np = self.mem.np
        self.writes.append((ref, np.asarray(tensor, np.float64).astype(
            np.float32).reshape(-1)))


class RelaxedExecutor(fx.Executor):
    """The relaxed arm: site opcodes through the lane backend."""

    def __init__(self, plan, lanes):
        super().__init__(plan, fx.NumpyBackend("float64"))
        self.lane = LaneBackend(lanes)

    def execute(self, section, scalars=None, on_instruction=None):
        scalars = scalars or self.step_scalars()
        for index, ins in enumerate(self.plan.sections[section]):
            x = self.lane if is_site(ins) else self.x
            fx.INTERPRETER[ins.opcode](self.mem, x, ins, scalars)
            if on_instruction:
                on_instruction(index, ins)


class LockstepExecutor(fx.Executor):
    """The reference arm, measuring each site instruction's relaxed result
    against its exact one on the same inputs."""

    def __init__(self, plan, lanes):
        super().__init__(plan, fx.NumpyBackend("float64"))
        self.lane, self.sites = LaneBackend(lanes), {}

    def execute(self, section, scalars=None, on_instruction=None):
        np = self.x.np
        scalars = scalars or self.step_scalars()
        for ins in self.plan.sections[section]:
            op = fx.INTERPRETER[ins.opcode]
            if not is_site(ins):
                op(self.mem, self.x, ins, scalars)
                continue
            dry = DryMemory(self.mem)
            self.lane.bound_rel = 0.0
            op(dry, self.lane, ins, scalars)
            exact = []
            self.mem.recorder = lambda off, values: exact.append((off, values))
            op(self.mem, self.x, ins, scalars)
            self.mem.recorder = None
            stat = self.sites.setdefault(SITES[ins.opcode], {
                "instances": 0, "observed_rel": 0.0, "bound_rel": 0.0})
            stat["instances"] += 1
            stat["bound_rel"] = max(stat["bound_rel"], self.lane.bound_rel)
            for (_, got), (_, want) in zip(dry.writes, exact):
                scale = float(np.abs(want).max()) if want.size else 0.0
                if scale == 0.0:
                    continue
                worst = float(np.abs(got.astype(np.float64) -
                                     want.astype(np.float64)).max()) / scale
                stat["observed_rel"] = max(stat["observed_rel"], worst)


def sha256_file(path):
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
    except OSError as e:
        raise fx.PlanError(f"cannot read {path}: {e}")
    return digest.hexdigest()


def self_digest(cert):
    body = {k: v for k, v in cert.items() if k != "digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def resolve_plan(path):
    return os.path.join(path, PLAN_FILE) if os.path.isdir(path) else path


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def cmd_certify(args):
    np = fx.require_numpy()
    plan_path = resolve_plan(args.target)
    plan = fx.Plan.load(plan_path)

    def feed():
        corpus = fx.Corpus(np, args.corpus)
        corpus.check(plan)
        val = None
        if args.val_fraction > 0.0:
            corpus, val = corpus.split_validation(args.val_fraction)
        corpus.enable_shuffle(args.seed)
        return corpus, val

    # Lockstep evidence rides the reference arm of the free run: the
    # operands every site sees are the ones real training produces.
    arms = {"reference": LockstepExecutor(plan, args.lanes),
            "relaxed": RelaxedExecutor(plan, args.lanes)}
    runs = {}
    for name, ex in arms.items():
        corpus, val = feed()
        can_eval = val is not None and bool(plan.sections["eval"])
        run = {"loss_curve": []}
        if can_eval:
            run["val_initial"] = ex.evaluate(val)
        for _ in range(args.steps):
            run["loss_curve"].append(ex.train_step(corpus))
        if can_eval:
            run["val_final"] = ex.evaluate(val)
            run["gate_improved"] = run["val_final"] < run["val_initial"]
        else:  # the engine's fallback: the training-loss trend
            run["gate_improved"] = run["loss_curve"][-1] < run["loss_curve"][0]
        runs[name] = run

    ref, rlx = runs["reference"], runs["relaxed"]
    finite = all(math.isfinite(v) for r in runs.values()
                 for v in r["loss_curve"])
    free_run = {
        "loss_max_rel_dev": max(rel(a, b) for a, b in zip(
            rlx["loss_curve"], ref["loss_curve"])) if finite else math.inf,
        "loss_final_reference": ref["loss_curve"][-1],
        "loss_final_relaxed": rlx["loss_curve"][-1],
        "gate_improved_reference": ref["gate_improved"],
        "gate_improved_relaxed": rlx["gate_improved"],
    }
    if "val_final" in ref:
        free_run.update(val_initial=ref["val_initial"],
                        val_final_reference=ref["val_final"],
                        val_final_relaxed=rlx["val_final"],
                        val_rel_dev=rel(rlx["val_final"], ref["val_final"]))

    sites = arms["reference"].sites
    reasons = []
    if not sites:
        reasons.append("the plan has no reduction site to relax")
    for name, stat in sorted(sites.items()):
        if not stat["observed_rel"] <= args.max_site_error:
            reasons.append(f"{name}: observed relative error "
                           f"{stat['observed_rel']:.3e} exceeds "
                           f"{args.max_site_error:.1e}")
    for key in ("loss_max_rel_dev", "val_rel_dev"):
        if key in free_run and not free_run[key] <= args.max_loss_deviation:
            reasons.append(f"{key} {free_run[key]:.3e} exceeds "
                           f"{args.max_loss_deviation:.1e}")
    if ref["gate_improved"] != rlx["gate_improved"]:
        reasons.append("the update gate decides differently under the two "
                       "contracts")

    cert = {
        "schema": CERT_SCHEMA, "kind": CERT_KIND,
        "seeml_tool": "certify_numerics",
        "plan": {"file": os.path.basename(plan_path),
                 "sha256": sha256_file(plan_path),
                 "plan_hash": f"{plan.plan_hash:016x}",
                 "version": plan.version},
        "corpus": {"file": os.path.basename(args.corpus),
                   "sha256": sha256_file(args.corpus)},
        "contract": {"family": FAMILY, "model_lanes": args.lanes,
                     "unit_roundoff": UNIT_ROUNDOFF,
                     "sites": sorted(sites)},
        "evidence": {"steps": args.steps, "seed": args.seed,
                     "val_fraction": args.val_fraction,
                     "sites": {k: sites[k] for k in sorted(sites)},
                     "free_run": free_run},
        "tolerances": {"site_rel": args.max_site_error,
                       "loss_rel": args.max_loss_deviation},
        "verdict": "refused" if reasons else "granted",
        "reasons": reasons,
    }
    cert["digest"] = self_digest(cert)

    for name, stat in cert["evidence"]["sites"].items():
        print(f"certify_numerics: {name:<18} {stat['instances']:>5} "
              f"instance(s)  observed {stat['observed_rel']:.2e}  "
              f"any-order bound {stat['bound_rel']:.2e}")
    print(f"certify_numerics: free run over {args.steps} step(s): loss "
          f"deviation {free_run['loss_max_rel_dev']:.2e}"
          + (f", validation {free_run['val_final_reference']:.6f} vs "
             f"{free_run['val_final_relaxed']:.6f} "
             f"({free_run['val_rel_dev']:.2e})" if "val_rel_dev" in free_run
             else "")
          + f"; gate improved: reference {ref['gate_improved']}, relaxed "
            f"{rlx['gate_improved']}")
    for reason in reasons:
        print(f"certify_numerics: REFUSED {reason}")
    out = args.out or os.path.join(os.path.dirname(plan_path) or ".",
                                   CERT_FILE)
    fx.write_report(out, cert)
    print(f"certify_numerics: {cert['verdict']} — wrote {out}")
    return 3 if reasons else 0


def check_certificate(cert, plan_path, corpus_path=None):
    """Every reason `cert` does not vouch for the plan at `plan_path`
    (empty = it does). Standard library only: tool/pack_update.py calls
    this on the build host's tier-0 Python."""
    problems = []
    if (not isinstance(cert, dict) or cert.get("kind") != CERT_KIND
            or cert.get("schema") != CERT_SCHEMA):
        return [f"not a schema-{CERT_SCHEMA} {CERT_KIND}"]
    try:
        if cert.get("digest") != self_digest(cert):
            problems.append("the certificate was edited after it was "
                            "written (digest mismatch)")
        if cert["plan"]["sha256"] != sha256_file(plan_path):
            problems.append("the certificate was issued for a different "
                            "plan (SHA-256 mismatch) — re-certify after "
                            "recompiling")
        if corpus_path and cert["corpus"]["sha256"] != sha256_file(
                corpus_path):
            problems.append("the certificate was issued over a different "
                            "corpus")
        if cert["verdict"] != "granted":
            problems.append("the certificate is a refusal: " +
                            "; ".join(cert.get("reasons", [])))
        if cert["contract"]["family"] != FAMILY:
            problems.append(f"unknown contract family "
                            f"{cert['contract']['family']!r}")
    except (KeyError, TypeError) as e:
        problems.append(f"malformed certificate ({e!r})")
    return problems


def check_observed(cert, observed):
    """A real relaxed run's measured deviations against the certified
    tolerances. `observed` is a frontier_exec.py diff report: its
    per-opcode worst |delta|/bound ratios, with the report's own rtol, give
    each site's relative error."""
    problems = []
    limit = cert["tolerances"]["site_rel"]
    rtol = float(observed.get("rtol", 0.0))
    for name, stat in sorted(observed.get("opcodes", {}).items()):
        # diff judges |a-b| <= atol + rtol*scale; ratio*rtol bounds the
        # relative error from above for any tensor with scale >= atol/rtol.
        measured = float(stat.get("worst_ratio", 0.0)) * rtol
        if name in cert["contract"]["sites"]:
            if measured > limit:
                problems.append(f"{name}: measured relative error "
                                f"{measured:.3e} exceeds the certified "
                                f"{limit:.1e}")
        elif not stat.get("worst_ratio", 0.0) <= 1.0:
            problems.append(f"{name} is not a certified site and must "
                            "agree with the reference at the diff tolerance")
    return problems


def cmd_verify(args):
    try:
        with open(args.certificate) as f:
            cert = json.load(f)
    except (OSError, ValueError) as e:
        raise fx.PlanError(f"cannot read {args.certificate}: {e}")
    problems = check_certificate(cert, resolve_plan(args.plan), args.corpus)
    if not problems and args.observed:
        try:
            with open(args.observed) as f:
                problems = check_observed(cert, json.load(f))
        except (OSError, ValueError) as e:
            raise fx.PlanError(f"cannot read {args.observed}: {e}")
    for problem in problems:
        print(f"certify_numerics: NOT VERIFIED {problem}")
    if not problems:
        print(f"certify_numerics: verified — {cert['contract']['family']} "
              f"over {len(cert['contract']['sites'])} site opcode(s), site "
              f"tolerance {cert['tolerances']['site_rel']:.1e}")
    return 3 if problems else 0


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("certify")
    c.add_argument("target", help="a .seeu plan or an emitted package dir")
    c.add_argument("--corpus", required=True)
    c.add_argument("--steps", type=int, default=20)
    c.add_argument("--lanes", type=int, default=8)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--val-fraction", type=float, default=0.1)
    c.add_argument("--max-site-error", type=float, default=1e-5)
    c.add_argument("--max-loss-deviation", type=float, default=1e-4)
    c.add_argument("--out", default=None)
    c.set_defaults(fn=cmd_certify)
    v = sub.add_parser("verify")
    v.add_argument("certificate")
    v.add_argument("--plan", required=True)
    v.add_argument("--corpus", default=None)
    v.add_argument("--observed", default=None)
    v.set_defaults(fn=cmd_verify)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "certify" and args.steps < 1:
        print("certify_numerics: ERROR --steps must be positive",
              file=sys.stderr)
        return 2
    try:
        return args.fn(args)
    except fx.PlanError as e:
        print(f"certify_numerics: ERROR {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
