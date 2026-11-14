#!/usr/bin/env python3
"""Regenerates the tokenizer fixtures from upstream colibri's own code (standard library
only; needs cc):

    python3 -I make_fixtures.py <colibri>/c ~/colibri-models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf

It copies <colibri>/c to a temporary directory, compiles the two harnesses next to this
script there, and writes:

- ../qwen3_encode.json, ../qwen3_encode_fuzz.json: token ids of a fixed corpus and of 200
  seeded random strings from c/qwen36.c load_tokenizer_gguf + encode_text (tok_harness.c).
  `ids` are from upstream's exact mode (EXACT=1: g_tok_exact, full Unicode classes and NFC,
  which is HF's Qwen2 tokenizer and what the port implements). `ids_upstream_default` is
  recorded only where upstream's default mode (approximate character ranges, no NFC)
  gives something else.
- ../gpt2_split.json: the pieces c/tok.h pretok_chunk_gpt2 (OLMoE's pre-tokenizer) cuts
  the same texts into (split_harness.c, built against a copy of tok.h whose bpe_piece
  records piece ends instead of encoding).
"""
import json, os, random, shutil, subprocess, sys, tempfile

CASES = [
    ("ascii", "Hello, world!"),
    ("sentence", "The quick brown fox jumps over the lazy dog."),
    ("greek", "Καλημέρα κόσμε! Τι κάνεις σήμερα;"),
    ("greek_polytonic", "Ἀθῆναι καὶ Σπάρτη"),
    ("chinese", "你好，世界！今天天气很好。"),
    ("japanese", "こんにちは、世界。カタカナもテスト。"),
    ("korean", "안녕하세요 세계"),
    ("emoji", "I love 🍕 and 🐍! 👨‍👩‍👧‍👦 family 🇬🇷"),
    ("code_python", "def foo(x):\n    return x**2 + 1\n"),
    ("code_c", "for (int i = 0; i < n; i++) {\n\tsum += a[i];\n}"),
    ("space_runs", "a  b   c    d"),
    ("leading_trailing", "   leading and trailing   "),
    ("newlines", "line1\n\nline2\r\nline3\n   indented"),
    ("tabs", "\t\ttabs\tand spaces  \t mix"),
    ("numbers", "3.14159 2026-11-14 1,234,567 0x1F 1e-9"),
    ("long_number", "12345678901234567890"),
    ("contractions", "I'm sure they'll say it's what we've done, isn't it? I'D SAY YOU'RE RIGHT"),
    ("chatml_system", "<|im_start|>system\nYou are helpful.<|im_end|>\n"),
    ("special_after_punct", "X.<|im_end|>"),
    ("think_block", "<think>\nhmm\n</think>\n\nanswer"),
    ("tool_call", "<tool_call>{\"name\": \"f\"}</tool_call>"),
    ("endoftext_twice", "<|endoftext|><|endoftext|>"),
    ("partial_specials", "<|im_start <|im_end| <|im_end|>> <<think>>"),
    ("accented", "naïve café résumé façade Zürich"),
    ("decomposed_nfc", "école café"),
    ("combining_marks", "x̃̄y"),
    ("cyrillic", "Привет, как дела? экскурсии удовольствием"),
    ("arabic", "مرحبا بالعالم ١٢٣"),
    ("hindi", "नमस्ते दुनिया"),
    ("thai", "สวัสดีครับ"),
    ("url_punct", "https://example.com/path?q=1&r=2#frag — “quotes” … ellipsis"),
    ("markdown", "## Title\n\n- item **bold** `code`\n\n```swift\nlet x = 1\n```"),
    ("json", "{\"key\": [1, 2, 3], \"nested\": {\"a\": null}}"),
    ("single_space", " "),
    ("single_newline", "\n"),
    ("mixed_scripts", "Swift和Python都很好用😀OK"),
    ("repeated_letters", "a" * 64),
    ("repeated_equals", "=" * 40),
    ("unicode_spaces", "a b　c d"),
    ("fullwidth", "１２３ＡＢＣ"),
    ("control_chars", "bell\u0007 esc\u001b[0m"),
    ("golden_prompt", "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ("split_edges", "(foo) .bar a  \tb x.\n\ny I'M here we'll end  "),
    ("ws_before_newlines", "  \n\n  y\r\n\r\n\tz \n"),
    ("long_mixed", "In 2024, the model scored 87.5% on MMLU (5-shot) — impressive!\n\n"
                   "Ελληνικά: Η γρήγορη καφέ αλεπού.\n中文：快速的棕色狐狸。\n"
                   "    indented code();  // comment\n\n\n\tend"),
]


# Random strings over a pool that exercises every rule of the split regex, the added
# tokens and NFC; fixed seed, so the file is reproducible.
POOL = (list("abcXYZ") * 3 + list("0123456789") + list("     ") + ["\n", "\n", "\r", "\t"]
        + list(".,;:!?'\"()[]{}<>|/\\-_=+*&^%$#@~`") + ["'s", "'LL", "'re", "'d"]
        + list("αβγΩЖжé中文字日本") + ["😀", "👍🏽", "🇬🇷", "\u0301", "\u0303", "\u00a0", "\u3000", "\u2003", "１", "٣", "ⅷ"]
        + ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>", "<|im", "|>"])


def fuzz_cases(n=200, seed=20261114):
    rnd = random.Random(seed)
    return [(f"fuzz_{i:03d}", "".join(rnd.choice(POOL) for _ in range(rnd.randint(1, 40))))
            for i in range(n)]


def run(harness, model, files, exact):
    env = dict(os.environ)
    env.pop("EXACT", None)
    if exact:
        env["EXACT"] = "1"
    out = subprocess.run([harness, model] + files, env=env, check=True,
                         capture_output=True, text=True).stdout.splitlines()
    assert len(out) == len(files), (len(out), len(files))
    return [[int(t) for t in line.split()] for line in out]


BPE_PIECE = "static void bpe_piece(Tok *T, const unsigned char *p, int a, int b, int *out, int *no, int max){"
HOOK = ("static void bpe_piece(Tok *T, const unsigned char *p, int a, int b, int *out, int *no, int max){\n"
        "    (void)T; (void)p; (void)a; if(*no<max) out[(*no)++]=b;   /* split oracle: the piece end */\n"
        "}\n" + BPE_PIECE.replace("bpe_piece", "bpe_piece_unused"))


def build(colibri_c, work):
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(work, "c")
    shutil.copytree(colibri_c, src)
    for f in ("tok_harness.c", "split_harness.c"):
        shutil.copy(os.path.join(here, f), src)
    with open(os.path.join(src, "tok.h")) as f:
        tok = f.read()
    assert tok.count(BPE_PIECE) == 1
    with open(os.path.join(src, "tok_split.h"), "w") as f:
        f.write(tok.replace(BPE_PIECE, HOOK))
    cc = ["cc", "-O2", "-pthread", "-w"]
    subprocess.run(cc + ["tok_harness.c", "-o", "tok_harness", "-lm"], cwd=src, check=True)
    subprocess.run(cc + ["split_harness.c", "-o", "split_harness", "-lm"], cwd=src, check=True)
    return os.path.join(src, "tok_harness"), os.path.join(src, "split_harness")


def main():
    colibri_c, model = sys.argv[1], sys.argv[2]
    with tempfile.TemporaryDirectory() as work:
        harness, splitter = build(colibri_c, work)
        write(harness, model, CASES, "qwen3_encode.json")
        write(harness, model, fuzz_cases(), "qwen3_encode_fuzz.json")
        write_split(splitter, CASES + fuzz_cases(), "gpt2_split.json")


def write_split(harness, cases, filename):
    with tempfile.TemporaryDirectory() as d:
        files = []
        for i, (_, text) in enumerate(cases):
            p = os.path.join(d, f"{i:03d}.txt")
            with open(p, "wb") as f:
                f.write(text.encode("utf-8"))
            files.append(p)
        lines = subprocess.run([harness] + files, check=True, capture_output=True, text=True).stdout.splitlines()
    assert len(lines) == len(cases)
    out = []
    for (name, text), line in zip(cases, lines):
        b = text.encode("utf-8")
        ends = [int(t) for t in line.split()]
        starts = [0] + ends[:-1]
        out.append({"name": name, "text": text, "pieces": [b[s:e].decode("utf-8") for s, e in zip(starts, ends)]})
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", filename), "w", encoding="utf-8") as f:
        json.dump({"oracle": "colibri 0a5504a c/tok.h pretok_chunk_gpt2 (split_harness.c)", "cases": out},
                  f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{filename}: {len(out)} cases")


def write(harness, model, CASES, filename):
    with tempfile.TemporaryDirectory() as d:
        files = []
        for i, (_, text) in enumerate(CASES):
            p = os.path.join(d, f"{i:03d}.txt")
            with open(p, "wb") as f:
                f.write(text.encode("utf-8"))
            files.append(p)
        exact = run(harness, model, files, True)
        default = run(harness, model, files, False)
    cases = []
    for (name, text), ids, dflt in zip(CASES, exact, default):
        c = {"name": name, "text": text, "ids": ids}
        if dflt != ids:
            c["ids_upstream_default"] = dflt
        cases.append(c)
    doc = {
        "model": "Qwen3-0.6B-Q8_0.gguf",
        "oracle": "colibri 0a5504a c/qwen36.c encode_text, g_tok_exact=1 (tok_harness.c)",
        "cases": cases,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", filename), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")
    diff = sum(1 for c in cases if "ids_upstream_default" in c)
    print(f"{filename}: {len(cases)} cases, {diff} differ between upstream exact and default modes")


if __name__ == "__main__":
    main()
