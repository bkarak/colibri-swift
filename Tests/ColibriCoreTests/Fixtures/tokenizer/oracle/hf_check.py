# Second, independent check of ../qwen3_encode*.json: HF `tokenizers` (Rust, onig regex)
# built from the GGUF's own vocab/merges/token types with Qwen2's tokenizer.json pipeline
# (NFC normalizer, the Split regex, ByteLevel without regex, BPE, added tokens unnormalized).
# Third-party packages, so run it only in a throwaway container (files piped in, since
# /private/tmp is not shared with Docker):
#
#   tar --no-xattrs -C <dir with this file and the two json files> -cf - . | \
#     docker run --rm -i -v ~/colibri-models/qwen3-0.6b:/model:ro python:3.12-slim sh -c \
#     "mkdir /data && tar -C /data -xf - && pip install -q 'gguf==0.10.0' sentencepiece 'numpy<2' \
#      'tokenizers==0.20.3' && python -I /data/hf_check.py"
#
# 2026-11-14: 45/45 and 200/200 identical.
import json, sys
from gguf import GGUFReader
from tokenizers import Tokenizer, Regex, AddedToken, models, normalizers, pre_tokenizers, decoders
r = GGUFReader("/model/Qwen3-0.6B-Q8_0.gguf")
def arr(key):
    f = r.fields[key]
    return [f.parts[i] for i in f.data]
toks = [bytes(p).decode("utf-8") for p in arr("tokenizer.ggml.tokens")]
types = [int(p[0]) for p in arr("tokenizer.ggml.token_type")]
merges = [bytes(p).decode("utf-8") for p in arr("tokenizer.ggml.merges")]
vocab = {t: i for i, t in enumerate(toks)}
bpe = models.BPE(vocab=vocab, merges=[tuple(m.split(" ", 1)) for m in merges], ignore_merges=False)
tok = Tokenizer(bpe)
tok.normalizer = normalizers.NFC()
pat = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
tok.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(Regex(pat), behavior="isolated", invert=False),
    pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)])
tok.decoder = decoders.ByteLevel()
added = [AddedToken(toks[i], special=(types[i] == 3), normalized=False) for i in range(len(toks)) if types[i] in (3, 4)]
tok.add_tokens(added)
for i in range(len(toks)):
    if types[i] in (3, 4): assert tok.token_to_id(toks[i]) == i, (i, toks[i])
total = 0
for fn in ["qwen3_encode.json", "qwen3_encode_fuzz.json"]:
    d = json.load(open("/data/" + fn))
    bad = [c["name"] for c in d["cases"] if tok.encode(c["text"], add_special_tokens=False).ids != c["ids"]]
    total += len(d["cases"])
    print(fn, len(d["cases"]), "cases,", len(bad), "differ from HF tokenizers", bad[:20])
