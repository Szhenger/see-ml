# runtime/profiler — the fourth reporter

Instrumentation and the runtime screen (`docs/next-project/seeai.md` §2,
§6): the step split by phase, per-kernel spans on Metal, peak resident
memory, energy per token where the platform reports it, and the
measured-versus-predicted line — measured step beside the backend's
predicted one, resident set beside the arena, loss beside the accountant's
loss floor. Today these are fifteen sites behind `SEEML_STEP_TIMING` and
`SEEML_METAL_PROFILE` in `dispatcher/update_engine.cc` and
`executor/metal_backend.mm`; S7 (#153) gathers them here. Until then this
README is the contract.
