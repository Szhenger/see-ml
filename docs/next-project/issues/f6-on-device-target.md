---
title: "SeeRL F6: on-device target — iPhone/iPad package runs, energy per token as a Tier A metric, adapter-only update packages"
labels: enhancement,frontier-parity
plane: Gates & docs
priority: P2
---
## Why

SeeML's claim is on-device training with no Python, but every measurement
so far is from a MacBook. The frontier frameworks are not designed to run
on a phone; this is where SeeML's doctrine (one arena, validated plans,
gates that score what ships, #95) should turn into a product advantage —
and it is unmeasured.

## Design

1. iOS/iPadOS: build the emitted package for arm64 iOS (Metal backend
   vendored, `.incbin` plan), a minimal host app that runs `model_update`
   on a bundled corpus; thermal-throttling behaviour recorded
   (the M5 laptop already shows two regimes).
2. **Energy per token** (J/token) as a Tier A metric: powermetrics on
   macOS, MetricKit / Xcode energy gauges on iOS; recorded next to tok/s in
   `seeml-bench` and in F1's frontier rows where the frontier can run.
3. **Adapter-only update packages** (roadmap pillar D2): ship the LoRA
   delta and the plan, not the model; the device commits onto its own
   copy, hash-checked (`BindSourceModel` already verifies the source).
   Headline: package bytes from model-scale to adapter-scale.

## Acceptance

- SmolLM-135M update runs on a current iPhone and iPad: tok/s, J/token,
  peak memory, gate result, committed model hash.
- Adapter-only package for SmolLM-135M r8 measured in bytes vs today's
  package.
