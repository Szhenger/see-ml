#!/usr/bin/env python3
"""The frontier harness (SeeRL F1, #129): real `torch.compile` and MLX-LM LoRA
runs on the same host, model, corpus, split, adapted set and tokens per step
as SeeML's emitted package — the measurement the Frontier Outlook's
asterisks became on 2026-09-22 — as one tool with the parity rules pinned.

Subcommands (each writes one JSON report in seeml-bench's units):

  records  MODEL_DIR TEXT --seq-len S        the exact records every row trains on
  torch    --model-dir D --corpus TEXT ...    a PyTorch LoRA loop, eager or
                                             torch.compile, cpu or mps, f32 or bf16
  mlx      --model-dir D --corpus TEXT ...    mlx_lm's own LoRA trainer on the same
                                             records, bf16 or true f32
  seeml    --package PKG --model SMF --data SDS --backend cpu|metal
                                             the emitted model_update, timed by
                                             steps-regression, then a quality run
  matrix   PLAN.json --out REPORT.json       the rows above, strictly serial, then
                                             the parity checks
  check    REPORT.json                       the parity checks on a report (exit 3
                                             on a refusal)

The parity rules — each one a comparison that was silently wrong before it
was written down (the 2026-09-22 Frontier Harness report, §6):

  1. Records: TEXT tokenized with the checkpoint's tokenizer.json, no BOS /
     EOS, consecutive (S + 1)-id records, the remainder dropped — exactly
     `export_model.py --text-corpus`; the tail floor(0.1 n) records held out
     — exactly `dataset.cc`. Same ids in every stack, checked by hash.
  2. Adapted set: the seven projections of every layer AND the LM head —
     SeeML's `--hf` importer writes the tied head as its own weight and
     adapts it. Rank and alpha/r are the plan's. The adapter parameter
     count of every row must equal the SeeML report's up to the k/v
     expansion SeeML's GQA-to-MHA repeat implies (`expected_adapter_params`).
  3. Optimizer: AdamW with BIAS CORRECTION (mlx.optimizers.AdamW defaults to
     none, which triples the first update), weight decay 0.01, constant LR,
     no clipping (SeeML clips per tensor, so any clip breaks parity).
  4. Precision named on every row and cast EXPLICITLY: SmolLM-135M's
     safetensors are stored F32 under a bfloat16 config, and mlx_lm loads
     what is stored; an MLX "f32" row is true f32 only under
     MLX_ENABLE_TF32=0 (the default matmul is TF32-class); a torch bf16 row
     keeps f32 adapters.
  5. Step-0 validation loss over the SAME records must agree across the
     f32 / bf16-base rows (B = 0, so every stack starts from the checkpoint);
     an int8-base row is expected to sit above them and is labelled so.
     mlx_lm's own `evaluate` drops the ragged tail of the validation set, so
     the harness scores the validation set itself, over every record.
  6. Timing: torch's median step after a stated warm-up (the compile is
     reported, not hidden); MLX's `It/sec` from its own trainer; SeeML's
     steps-regression slope (hi - lo and hi - quality) on the package binary;
     `torch.set_num_threads` pinned to SEEML_THREADS's value.

Tier 2 (torch / mlx / transformers / tokenizers) for the framework rows;
the seeml and check subcommands are standard library. Nothing here is
imported by the compiler or the runtime, and nothing here ships.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import time

REPORT_SCHEMA = 1
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
           "down_proj")
MLX_KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
            "mlp.down_proj"]


# --- Rule 1: the records ------------------------------------------------------

def load_records(model_dir, text_path, seq_len):
    """np.int64 [n, seq_len + 1] — export_model.py --text-corpus's records."""
    import numpy as np
    from tokenizers import Tokenizer
    ids = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json")).encode(
        open(text_path, encoding="utf-8").read()).ids
    rec = seq_len + 1
    n = len(ids) // rec
    return np.asarray(ids[: n * rec], dtype=np.int64).reshape(n, rec)


def split_tail(records, val_frac=0.1):
    """dataset.cc: val_n = floor(n * frac) clamped to [1, n - 1]; the TAIL."""
    n = len(records)
    val_n = min(max(int(n * val_frac), 1), n - 1)
    return records[: n - val_n], records[n - val_n:]


def records_digest(records):
    return hashlib.sha256(records.astype("<i4").tobytes()).hexdigest()[:16]


# --- Rule 2: the adapted set -------------------------------------------------

def hf_geometry(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    return {"layers": cfg["num_hidden_layers"], "dim": cfg["hidden_size"],
            "heads": cfg["num_attention_heads"],
            "kv_heads": cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
            "ffn": cfg["intermediate_size"], "vocab": cfg["vocab_size"],
            "tied": bool(cfg.get("tie_word_embeddings", True))}


def expected_adapter_params(geom, rank, seeml=False):
    """LoRA parameters r*(K + M) over the seven projections and the head.
    SeeML repeats grouped k/v heads to one per query head, so its k and v
    projections are D -> D where the frameworks' are D -> D_kv."""
    d, h, kvh, ffn, v, layers = (geom["dim"], geom["heads"], geom["kv_heads"],
                                 geom["ffn"], geom["vocab"], geom["layers"])
    head_dim = d // h
    d_kv = d if seeml else kvh * head_dim
    per_layer = (rank * (d + d) + 2 * rank * (d + d_kv) + rank * (d + d)
                 + 2 * rank * (d + ffn) + rank * (ffn + d))
    return layers * per_layer + rank * (d + v)


# --- The report ----------------------------------------------------------------

def host_info():
    out = {"platform": platform.platform(), "machine": platform.machine(),
           "python": platform.python_version(),
           "seeml_threads": os.environ.get("SEEML_THREADS")}
    try:
        out["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                    capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    return out


def row(system, backend, precision, **fields):
    r = {"schema": REPORT_SCHEMA, "system": system, "backend": backend,
         "precision": precision, "host": host_info()}
    r.update(fields)
    return r


def write_report(path, report):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    os.replace(tmp, path)


# --- torch --------------------------------------------------------------------

def cmd_torch(args):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    class LoRALinear(nn.Module):
        def __init__(self, base, r, alpha, gen):
            super().__init__()
            self.base, self.scale = base, alpha / r
            k, m = base.in_features, base.out_features
            self.A = nn.Parameter(torch.randn(k, r, generator=gen) / math.sqrt(k))
            self.B = nn.Parameter(torch.zeros(r, m))

        def forward(self, x):
            y = self.base(x)
            return y + (self.scale * ((x.to(self.A.dtype) @ self.A) @ self.B)).to(y.dtype)

    threads = int(os.environ.get("SEEML_THREADS", args.threads))
    torch.set_num_threads(threads)
    torch.manual_seed(args.seed)
    dtype = torch.float32 if args.dtype == "f32" else torch.bfloat16
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype,
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    for prm in model.parameters():
        prm.requires_grad_(False)
    gen = torch.Generator().manual_seed(args.lora_seed)
    adapters = 0
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, mod in list(parent.named_children()):
                if name in TARGETS and isinstance(mod, nn.Linear):
                    setattr(parent, name, LoRALinear(mod, args.rank, args.alpha, gen))
                    adapters += 1
    model.lm_head = LoRALinear(model.lm_head, args.rank, args.alpha, gen)  # rule 2
    adapters += 1
    model.to(args.device).train()
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=args.weight_decay)  # rule 3
    load_s = time.perf_counter() - t0
    records = load_records(args.model_dir, args.corpus, args.seq_len)
    train_rec, val_rec = split_tail(records, args.val_frac)
    eager = model
    run_model = (torch.compile(model, mode=args.compile_mode, dynamic=False)
                 if args.compile else model)

    def sync():
        if args.device == "mps":
            torch.mps.synchronize()

    @torch.no_grad()
    def evaluate():
        eager.eval()
        loss = correct = tot = 0
        for i in range(0, len(val_rec), args.batch):
            rec = torch.from_numpy(val_rec[i:i + args.batch]).to(args.device)
            x, y = rec[:, :-1], rec[:, 1:]
            logits = eager(input_ids=x, use_cache=False).logits.float()
            loss += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                    y.reshape(-1), reduction="sum").item()
            correct += (logits.argmax(-1) == y).sum().item()
            tot += y.numel()
        eager.train()
        return loss / tot, correct / tot

    val0 = evaluate()
    sgen = torch.Generator().manual_seed(args.seed)
    perm, cursor = torch.randperm(len(train_rec), generator=sgen).numpy(), 0
    curve, wall, fwd, bwd, optt = [], [], [], [], []
    peak_dev = 0
    for _ in range(args.steps):
        if cursor + args.batch > len(train_rec):
            perm, cursor = torch.randperm(len(train_rec), generator=sgen).numpy(), 0
        rec = torch.from_numpy(train_rec[perm[cursor:cursor + args.batch]]).to(args.device)
        cursor += args.batch
        x, y = rec[:, :-1], rec[:, 1:]
        sync()
        t0 = time.perf_counter()
        logits = run_model(input_ids=x, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
        sync()
        t1 = time.perf_counter()
        loss.backward()
        sync()
        t2 = time.perf_counter()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        t3 = time.perf_counter()
        curve.append(loss.item())
        wall.append(t3 - t0)
        fwd.append(t1 - t0)
        bwd.append(t2 - t1)
        optt.append(t3 - t2)
        if args.device == "mps":
            peak_dev = max(peak_dev, torch.mps.driver_allocated_memory())
    val1 = evaluate()
    w = args.warmup
    step_s = statistics.median(wall[w:])
    tokens = args.batch * args.seq_len
    report = row(
        "torch.compile" if args.compile else "torch eager", args.device,
        "f32" if args.dtype == "f32" else "bf16 base, f32 adapters",
        model_dir=args.model_dir, torch=torch.__version__, compile=bool(args.compile),
        threads=threads, records_digest=records_digest(records),
        records={"total": len(records), "train": len(train_rec), "val": len(val_rec)},
        config=dict(rank=args.rank, alpha=args.alpha, scale=args.alpha / args.rank,
                    lr=args.lr, weight_decay=args.weight_decay, optimizer="adamw",
                    bias_correction=True, schedule="const", clip=0, batch_records=args.batch,
                    seq_len=args.seq_len, tokens_per_step=tokens, steps=args.steps,
                    targets=list(TARGETS) + ["lm_head"], lora_seed=args.lora_seed,
                    shuffle_seed=args.seed),
        adapters=adapters, adapter_params=sum(q.numel() for q in params),
        load_s=load_s, warmup_steps=w, warmup_s=sum(wall[:w]), first_step_s=wall[0],
        step_ms=step_s * 1000, step_ms_min=min(wall[w:]) * 1000,
        step_ms_max=max(wall[w:]) * 1000, fwd_ms=statistics.median(fwd[w:]) * 1000,
        bwd_ms=statistics.median(bwd[w:]) * 1000, opt_ms=statistics.median(optt[w:]) * 1000,
        tokens_per_s=tokens / step_s, it_per_s=1 / step_s, tokens_per_step=tokens,
        train_loss=[float(np.mean(curve[:10])), float(np.mean(curve[-10:]))],
        val_loss=[val0[0], val1[0]], val_acc=[val0[1], val1[1]],
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_device_bytes=peak_dev, loss_curve=curve)
    write_report(args.report, report)
    print(f"frontier_run torch: {args.device} compile={bool(args.compile)} {args.dtype}: "
          f"{report['step_ms']:.1f} ms/step, {report['tokens_per_s']:.0f} tok/s, warm-up "
          f"{report['warmup_s']:.1f} s, val {val0[0]:.4f} -> {val1[0]:.4f}")
    return 0


# --- mlx ----------------------------------------------------------------------

def cmd_mlx(args):
    # Rule 4: the TF32 switch is latched at import, so the mode must be in
    # the environment before mlx loads — re-execute if it is not.
    want = "0" if args.dtype == "f32" and not args.tf32 else None
    have = os.environ.get("MLX_ENABLE_TF32")
    if want != have and not os.environ.get("_FRONTIER_RUN_REEXEC"):
        env = dict(os.environ, _FRONTIER_RUN_REEXEC="1")
        if want is None:
            env.pop("MLX_ENABLE_TF32", None)
        else:
            env["MLX_ENABLE_TF32"] = want
        return subprocess.call([sys.executable] + sys.argv, env=env)
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import TrainingArgs, TrainingCallback, train
    from mlx_lm.tuner.utils import linear_to_lora_layers

    class TokenRecords:
        def __init__(self, recs):
            self.recs = [[int(t) for t in r] for r in recs]

        def __getitem__(self, i):
            return self.recs[i]

        def __len__(self):
            return len(self.recs)

        def process(self, d):
            return (d, 0)

    class LoRAHead(nn.Module):  # rule 2: the tied head's own adapter
        def __init__(self, emb, r, scale):
            super().__init__()
            self.emb, self.scale = emb, scale
            v, d = emb.weight.shape
            self.lora_a = mx.random.uniform(low=-1 / d ** 0.5, high=1 / d ** 0.5, shape=(d, r))
            self.lora_b = mx.zeros((r, v))

        def __call__(self, x):
            return self.emb(x)

        def as_linear(self, x):
            return self.emb.as_linear(x) + (self.scale * ((x @ self.lora_a) @ self.lora_b)).astype(x.dtype)

    class Capture(TrainingCallback):
        def __init__(self):
            self.train, self.val = [], []

        def on_train_loss_report(self, info):
            self.train.append(dict(info))

        def on_val_loss_report(self, info):
            self.val.append(dict(info))

    mx.random.seed(args.seed)
    t0 = time.perf_counter()
    model, _ = load(args.model_dir)
    model.set_dtype(mx.float32 if args.dtype == "f32" else mx.bfloat16)  # rule 4
    model.freeze()
    scale = args.alpha / args.rank
    linear_to_lora_layers(model, len(model.layers),
                          {"rank": args.rank, "scale": scale, "dropout": 0.0, "keys": MLX_KEYS})
    head = LoRAHead(model.model.embed_tokens, args.rank, scale)
    head.freeze()
    head.unfreeze(keys=["lora_a", "lora_b"])
    model.model.embed_tokens = head
    n_adapter = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    load_s = time.perf_counter() - t0
    records = load_records(args.model_dir, args.corpus, args.seq_len)
    train_rec, val_rec = split_tail(records, args.val_frac)

    def evaluate():  # rule 5: every record, not mlx_lm's batched subset
        model.eval()
        loss = correct = tot = 0
        for i in range(0, len(val_rec), args.batch):
            b = mx.array(val_rec[i:i + args.batch])
            x, y = b[:, :-1], b[:, 1:]
            logits = model(x).astype(mx.float32)
            ce = nn.losses.cross_entropy(logits, y, reduction="none")
            loss += float(ce.sum().item())
            correct += int((logits.argmax(-1) == y).sum().item())
            tot += y.size
        model.train()
        return loss / tot, correct / tot

    val0 = evaluate()
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay,
                      bias_correction=True)  # rule 3
    cb = Capture()
    targs = TrainingArgs(batch_size=args.batch, iters=args.steps, val_batches=-1,
                         steps_per_report=10, steps_per_eval=10 ** 9, steps_per_save=10 ** 9,
                         max_seq_length=args.seq_len + 1,  # no pad target: 512 per step
                         adapter_file=os.path.join(os.path.dirname(os.path.abspath(args.report)),
                                                   "mlx_adapters.safetensors"))
    mx.reset_peak_memory()
    t1 = time.perf_counter()
    train(model, opt, CacheDataset(TokenRecords(train_rec)), CacheDataset(TokenRecords(val_rec)),
          args=targs, training_callback=cb)
    train_wall = time.perf_counter() - t1
    val1 = evaluate()
    reports = cb.train[1:] or cb.train
    it_s = statistics.median(r["iterations_per_second"] for r in reports)
    tokens = args.batch * args.seq_len
    precision = ("bf16" if args.dtype == "bf16"
                 else "f32" if not args.tf32 else "f32 weights, TF32-class matmul")
    report = row(
        "MLX-LM LoRA", "gpu", precision, model_dir=args.model_dir,
        mlx=mx.__version__, mlx_lm=__import__("mlx_lm").__version__,
        mlx_enable_tf32=os.environ.get("MLX_ENABLE_TF32"),
        records_digest=records_digest(records),
        records={"total": len(records), "train": len(train_rec), "val": len(val_rec)},
        config=dict(rank=args.rank, alpha=args.alpha, scale=scale, lr=args.lr,
                    weight_decay=args.weight_decay, optimizer="adamw", bias_correction=True,
                    schedule="const", clip=0, batch_records=args.batch, seq_len=args.seq_len,
                    tokens_per_step=tokens, steps=args.steps, targets=MLX_KEYS + ["lm_head"],
                    seed=args.seed),
        adapter_params=int(n_adapter), load_s=load_s, step_ms=1000 / it_s,
        tokens_per_s=it_s * tokens, it_per_s=it_s, tokens_per_step=tokens,
        tokens_per_step_mlx_count=cb.train[-1]["trained_tokens"] / args.steps,
        train_wall_s=train_wall,
        train_loss=[cb.train[0]["train_loss"], cb.train[-1]["train_loss"]],
        val_loss=[val0[0], val1[0]], val_acc=[val0[1], val1[1]],
        peak_device_bytes=int(mx.get_peak_memory()),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_report(args.report, report)
    print(f"frontier_run mlx: {precision}: {report['step_ms']:.1f} ms/step, "
          f"{report['tokens_per_s']:.0f} tok/s, val {val0[0]:.4f} -> {val1[0]:.4f}")
    return 0


# --- seeml ---------------------------------------------------------------------

def _timed(cmd, env=None):
    """/usr/bin/time -l on macOS (GNU time -v elsewhere): wall seconds and
    the OS's peak RSS of the child."""
    e = dict(os.environ)
    e.update(env or {})
    timer = ["/usr/bin/time", "-l"] if sys.platform == "darwin" else ["/usr/bin/time", "-v"]
    t0 = time.perf_counter()
    done = subprocess.run(timer + cmd, capture_output=True, text=True, env=e)
    wall = time.perf_counter() - t0
    out = done.stdout + done.stderr
    m = re.search(r"(\d+)\s+maximum resident set size", out) or \
        re.search(r"Maximum resident set size \(kbytes\): (\d+)", out)
    rss = int(m.group(1)) * (1 if sys.platform == "darwin" else 1024) if m else 0
    real = re.search(r"([\d.]+) real", out)
    return done.returncode, out, float(real.group(1)) if real else wall, rss


def cmd_seeml(args):
    binary = os.path.join(args.package, "model_update")
    if not os.access(binary, os.X_OK):
        raise SystemExit(f"frontier_run: no model_update in {args.package} (pack_update.py --build first)")
    out_dir = os.path.dirname(os.path.abspath(args.report)) or "."
    env = {"SEEML_THREADS": str(args.threads)}

    def run(steps, tag):
        model_out = os.path.join(out_dir, f"{tag}.smf")
        rep = os.path.join(out_dir, f"{tag}.json")
        for f in (model_out, rep):
            if os.path.exists(f):
                os.remove(f)
        rc, out, wall, rss = _timed(
            [binary, "--model", args.model, "--data", args.data, "--out", model_out,
             "--val-frac", str(args.val_frac), "--seed", str(args.seed), "--steps", str(steps),
             "--eval-every", "0", "--backend", args.backend, "--report", rep], env)
        report = json.load(open(rep)) if os.path.exists(rep) else {}
        sha = None
        if os.path.exists(model_out):
            h = hashlib.sha256()
            with open(model_out, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            sha = h.hexdigest()
            os.remove(model_out)
        device = re.search(r"backend \w+ \(([^\n]*)\)", out)
        return {"steps": steps, "exit": rc, "wall_s": wall, "peak_rss_bytes": rss,
                "report": report, "committed_sha256": sha,
                "device": device.group(1) if device else None}

    slopes, runs = [], []
    for _ in range(args.repeats):
        lo, hi = run(args.lo, "lo"), run(args.hi, "hi")
        runs += [lo, hi]
        slopes.append((hi["wall_s"] - lo["wall_s"]) / (args.hi - args.lo))
    quality = run(args.quality_steps, "quality") if args.quality_steps else None
    if quality and quality["steps"] > args.hi:
        hi_wall = statistics.median(r["wall_s"] for r in runs if r["steps"] == args.hi)
        step_s = (quality["wall_s"] - hi_wall) / (quality["steps"] - args.hi)
    else:
        step_s = statistics.median(slopes)
    # A short lo run on a cold file cache (the source model is hashed on
    # load) can outrun the hi run: a non-positive slope is not a number.
    suspect = step_s <= 0 or min(slopes) <= 0
    if suspect:
        print("frontier_run: WARNING the steps-regression slope is not positive "
              f"({[round(s * 1000, 1) for s in slopes]} ms): raise --lo / --hi or "
              "add --quality-steps; the row is marked suspect")
        step_s = abs(step_s) or 1e-9
    compile_report = json.load(open(os.path.join(args.package, "report.json")))
    tokens = compile_report["effective_batch"]
    r0 = quality["report"] if quality else {}
    report = row(
        "SeeML", args.backend,
        "f32 math, int8 base" if compile_report["quantized_base"]
        else "f32 math, bf16 base" if compile_report["bf16_base"] else "f32",
        package=args.package, seeml_version=compile_report["seeml_version"],
        plan_version=compile_report["plan_version"], plan_hash=compile_report["plan_hash"],
        precision_flag=compile_report.get("precision", "f32"),
        relaxed_gemms=compile_report.get("relaxed_gemms", 0),
        device=(quality or runs[0])["device"], threads=args.threads,
        config=dict(steps=args.quality_steps, tokens_per_step=tokens,
                    lo=args.lo, hi=args.hi, repeats=args.repeats),
        adapters=len(compile_report["adapters"]),
        adapter_params=sum(a["rank"] * (a["k"] + a["m"]) for a in compile_report["adapters"]),
        arena_bytes=compile_report["arena_bytes"], plan_bytes=compile_report["plan_bytes"],
        step_ms=step_s * 1000, step_ms_lohi=statistics.median(slopes) * 1000,
        step_ms_min=min(slopes) * 1000, step_ms_max=max(slopes) * 1000,
        tokens_per_s=tokens / step_s, it_per_s=1 / step_s, tokens_per_step=tokens,
        intercept_s=statistics.median(r["wall_s"] for r in runs if r["steps"] == args.lo)
        - statistics.median(slopes) * args.lo,
        val_loss=r0.get("validation_loss"), val_acc=r0.get("validation_accuracy"),
        train_loss=r0.get("train_loss"), verdict=r0.get("verdict"),
        committed=r0.get("committed"),
        committed_sha256=quality["committed_sha256"] if quality else None,
        suspect=suspect,
        wall_quality_s=quality["wall_s"] if quality else None,
        peak_rss_bytes=max(r["peak_rss_bytes"] for r in runs + ([quality] if quality else [])),
        runs=runs + ([quality] if quality else []))
    write_report(args.report, report)
    print(f"frontier_run seeml: {args.backend}: {report['step_ms']:.1f} ms/step, "
          f"{report['tokens_per_s']:.0f} tok/s"
          + (f", val {r0['validation_loss'][0]:.4f} -> {r0['validation_loss'][1]:.4f}"
             if r0.get("validation_loss") else ""))
    return 0


# --- the parity checks ---------------------------------------------------------

def parity_problems(rows, geom=None, rank=None, val0_tol=0.02):
    """Every reason the rows are not one comparison (empty = they are)."""
    problems = []
    digests = {r.get("records_digest") for r in rows if r.get("records_digest")}
    if len(digests) > 1:
        problems.append(f"rows train on different records: {sorted(digests)}")
    tokens = {r.get("tokens_per_step") for r in rows}
    if len(tokens) > 1:
        problems.append(f"rows train different tokens per step: {sorted(tokens)}")
    if geom and rank:
        for r in rows:
            got = r.get("adapter_params")
            want = expected_adapter_params(geom, rank, seeml=r["system"] == "SeeML")
            if got is not None and got != want:
                problems.append(f"{r['system']} / {r['backend']} / {r['precision']}: "
                                f"{got} adapter parameters, expected {want} at rank {rank}")
    ref = [r for r in rows if r.get("val_loss") and "int8" not in r["precision"]]
    if len(ref) > 1:
        base = ref[0]["val_loss"][0]
        for r in ref[1:]:
            if abs(r["val_loss"][0] - base) > val0_tol:
                problems.append(f"{r['system']} / {r['backend']} / {r['precision']}: "
                                f"step-0 validation loss {r['val_loss'][0]:.4f} vs "
                                f"{ref[0]['system']}'s {base:.4f} (tolerance {val0_tol})")
    return problems


def cmd_check(args):
    report = json.load(open(args.report))
    rows = report["rows"] if isinstance(report, dict) and "rows" in report else [report]
    geom = hf_geometry(args.model_dir) if args.model_dir else None
    problems = parity_problems(rows, geom, args.rank, args.val0_tolerance)
    for p in problems:
        print(f"frontier_run: REFUSED {p}")
    if not problems:
        print(f"frontier_run: {len(rows)} row(s) are one comparison")
    return 3 if problems else 0


def cmd_matrix(args):
    plan = json.load(open(args.plan))
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    rows = []
    for i, job in enumerate(plan["rows"]):
        rep = os.path.join(out_dir, f"row_{i:02d}.json")
        if os.path.exists(rep) and not args.force:
            rows.append(json.load(open(rep)))
            continue
        cmd = [sys.executable, os.path.abspath(__file__), job["kind"]] + job["args"] + ["--report", rep]
        print(f"frontier_run: [{i}] {' '.join(cmd[2:])}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"frontier_run: row {i} failed with {rc}")
            return 1
        rows.append(json.load(open(rep)))
    geom = hf_geometry(plan["model_dir"]) if plan.get("model_dir") else None
    problems = parity_problems(rows, geom, plan.get("rank"), plan.get("val0_tolerance", 0.02))
    write_report(args.out, {"schema": REPORT_SCHEMA, "plan": args.plan, "rows": rows,
                            "parity_problems": problems, "host": host_info()})
    for p in problems:
        print(f"frontier_run: REFUSED {p}")
    print(f"frontier_run: wrote {args.out} ({len(rows)} rows)")
    return 3 if problems else 0


def cmd_records(args):
    records = load_records(args.model_dir, args.text, args.seq_len)
    train_rec, val_rec = split_tail(records, args.val_frac)
    print(f"{len(records)} records of {args.seq_len + 1} ids ({len(train_rec)} train, "
          f"{len(val_rec)} held out), digest {records_digest(records)}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--model-dir", required=True)
        sp.add_argument("--corpus", required=True, help="a UTF-8 text file")
        sp.add_argument("--seq-len", type=int, default=128)
        sp.add_argument("--batch", type=int, default=4)
        sp.add_argument("--steps", type=int, default=300)
        sp.add_argument("--lr", type=float, default=1e-4)
        sp.add_argument("--weight-decay", type=float, default=0.01)
        sp.add_argument("--rank", type=int, default=8)
        sp.add_argument("--alpha", type=float, default=16.0)
        sp.add_argument("--seed", type=int, default=7)
        sp.add_argument("--val-frac", type=float, default=0.1)
        sp.add_argument("--report", required=True)

    t = sub.add_parser("torch")
    common(t)
    t.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    t.add_argument("--compile", type=int, default=1)
    t.add_argument("--compile-mode", default="default")
    t.add_argument("--dtype", default="f32", choices=("f32", "bf16"))
    t.add_argument("--warmup", type=int, default=5)
    t.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    t.add_argument("--lora-seed", type=int, default=42)
    t.set_defaults(fn=cmd_torch)

    m = sub.add_parser("mlx")
    common(m)
    m.add_argument("--dtype", default="bf16", choices=("bf16", "f32"))
    m.add_argument("--tf32", type=int, default=0,
                   help="f32 rows: 1 keeps MLX's default (TF32-class) matmul")
    m.set_defaults(fn=cmd_mlx)

    s = sub.add_parser("seeml")
    s.add_argument("--package", required=True, help="an emitted, built package dir")
    s.add_argument("--model", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--backend", default="cpu", choices=("cpu", "metal"))
    s.add_argument("--lo", type=int, default=40)
    s.add_argument("--hi", type=int, default=160)
    s.add_argument("--repeats", type=int, default=2)
    s.add_argument("--quality-steps", type=int, default=300)
    s.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--val-frac", type=float, default=0.1)
    s.add_argument("--report", required=True)
    s.set_defaults(fn=cmd_seeml)

    x = sub.add_parser("matrix")
    x.add_argument("plan")
    x.add_argument("--out", required=True)
    x.add_argument("--force", action="store_true")
    x.set_defaults(fn=cmd_matrix)

    c = sub.add_parser("check")
    c.add_argument("report")
    c.add_argument("--model-dir", default=None)
    c.add_argument("--rank", type=int, default=None)
    c.add_argument("--val0-tolerance", type=float, default=0.02)
    c.set_defaults(fn=cmd_check)

    r = sub.add_parser("records")
    r.add_argument("model_dir")
    r.add_argument("text")
    r.add_argument("--seq-len", type=int, default=128)
    r.add_argument("--val-frac", type=float, default=0.1)
    r.set_defaults(fn=cmd_records)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
