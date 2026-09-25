# runtime/loader — the security boundary

Plan load (`docs/next-project/seeai.md` §6): map the bytes, check the seal
hash, read the header, verify the source model's hash, check the resource
contract against the device, allocate the one arena. Everything after it
trusts the plan; the verifier proves instructions, the loader proves the
container and the environment. The identity manifest, the tokenizer hash
and the reduction-order ledger (S4, S2, S5) are checked here. Today this is
`LoadFromMemory`, `VerifySourceModel` and the resource contract inside
`dispatcher/update_engine.cc`; S7 (#153) extracts them. Until then this
README is the contract.
