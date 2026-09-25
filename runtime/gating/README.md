# runtime/gating — the accept-or-refuse decision

Evaluate before and after, track the best evaluated state, honour patience
(also on resume — the 2026-09-24 review's N12), and decide exit 0 or 3
(`docs/next-project/seeai.md` §6). It is the product's decision point and
the last defense against poisoned training data, scoring the held-out set
the validation subsystem (S3 #149) chose at compile time. Today it lives
inside `dispatcher/update_engine.cc`; S7 (#153) extracts it. Until then this
README is the contract.
