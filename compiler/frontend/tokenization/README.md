# frontend/tokenization — the training-data ingress

The frontend's second input path (`docs/next-project/seeai.md` §3). The
model path is `ingressor → accountant → topology → computation`; the data
path is this directory, and the two meet at the accountant.

Stages, in order, each a pure deterministic function so the package can
vendor them and the device's pipeline re-runs the same checks on its own
data (security at every layer):

1. **ingress** — a bounded reader over raw text (or a pre-tokenized SDS
   whose header carries the tokenizer hash).
2. **normalize** — Unicode normalization as the model's tokenizer section
   specifies.
3. **sanitize / filter** — invalid UTF-8, control characters, bidirectional
   overrides, empty and degenerate records; every drop recorded by index and
   reason in the manifest.
4. **deduplicate** — exact, by record hash (near-duplicates are the Python
   plane's job).
5. **encode** — byte-level BPE from the SMF tokenizer section only (no
   `--tokenizer` flag exists); proven bit-exact against the reference
   tokenizer at export.
6. **pack** — concatenate documents with EOS and cut fixed S+1 windows;
   never pad, because every position is a loss target.

Artifact: a canonical SDS, its manifest, and its hash. The tokenizer
definition's hash binds every corpus on both layers (plan header, SDS
header). What this directory does **not** defend against, and says so on
its screen: well-formed poisoned data — the gate and a held-out set the
training source cannot influence are that defense.

The code lands with **S2 (#148)**; until then this README is the contract.
