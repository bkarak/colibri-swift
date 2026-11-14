#!/usr/bin/env python3
"""Record upstream colibri's greedy outputs as fixtures for the Swift port.

Runs the upstream C engines (built by build-upstream.sh) on
  * the real Qwen3-0.6B-Q8_0 GGUF, 5 fixed prompts      -> Fixtures/oracle/qwen3-0.6b-q8_0.json
  * the synthetic tiny GGUFs written by TinyModel        -> Fixtures/oracle/tiny/<case>.json
and stores, per prompt: the prompt ids, the greedy ids with the float reference matmul
(GQ_EXACT=1, what an f32-dequantising port should reproduce) and with upstream's default
int8-activation path, the top-5 logits behind every greedy step (so a divergence can be
told apart from a near tie), and the prefill logits (all of them for the tiny models,
the top 20 for Qwen3-0.6B).

Python standard library only. Run with `python3 -I`:

  python3 -I scripts/oracle/record.py                 # everything
  python3 -I scripts/oracle/record.py --tiny --skip-generate --cases olmoe_f32
  python3 -I scripts/oracle/record.py --real

Recipe and flags: docs/port/models.md section 10 and scripts/oracle/README.md.
"""
import argparse, array, concurrent.futures as cf, json, os, shutil, subprocess, sys, tempfile, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES = os.path.join(REPO, 'Tests', 'ColibriCoreTests', 'Fixtures', 'oracle')
ORACLE_DIR = os.environ.get('ORACLE_DIR', os.path.join(os.environ.get('TMPDIR', '/tmp'), 'colibri-oracle'))
REAL_GGUF = os.environ.get('COLIBRI_QWEN3_GGUF',
                           os.path.expanduser('~/colibri-models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf'))
UPSTREAM_COMMIT = '0a5504a'

TINY_NEW = 24
REAL_NEW = 32
STEP_TOP = 5
REAL_PREFILL_TOP = 20

# Prompts for the tiny models (byte vocabulary: one token per byte, control tokens matched).
TINY_PROMPTS = {
    'qwen': [
        'Hello, world!',
        'The quick brown fox jumps over the lazy dog.\n',
        'def f(x):\n    return x * 2  # 42\n',
        '<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n',
    ],
    'olmoe': [
        'Hello, world!',
        'The quick brown fox jumps over the lazy dog.\n',
        'def f(x):\n    return x * 2  # 42\n',
        '|||IP_ADDRESS|||<|user|>\nHi there\n<|assistant|>\n',
    ],
}

THINK_OFF = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
REAL_PROMPTS = [
    ('chat_france', '<|im_start|>user\nWhat is the capital of France?<|im_end|>\n' + THINK_OFF),
    ('chat_haiku', '<|im_start|>user\nWrite a haiku about the sea.<|im_end|>\n' + THINK_OFF),
    ('raw_fox', 'The quick brown fox jumps over the lazy dog. The quick brown fox'),
    ('raw_code', 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n'),
    ('chat_greek_system', '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
                          '<|im_start|>user\nΠώς λέγεται η πρωτεύουσα της Ελλάδας; Απάντησε με μία λέξη.<|im_end|>\n'
                          + THINK_OFF),
]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def run(cmd, env, cwd=None):
    e = dict(os.environ)
    e.update(env)
    p = subprocess.run(cmd, env=e, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout, p.stderr


def parse_engine_ids(out, err, what):
    i = out.find(b'C engine :')
    if i < 0:
        raise RuntimeError('%s: no "C engine :" line\nstdout: %s\nstderr: %s' % (what, out[-2000:], err[-2000:]))
    rest = out[i + len(b'C engine :'):]
    ids = []
    for x in rest.split():    # "C engine : 1 2 3 Text : ..." (qwen36) or "... 3\nMatching tokens" (olmoe)
        if not x.isdigit():
            break
        ids.append(int(x))
    return ids


def read_f32(path):
    a = array.array('f')
    with open(path, 'rb') as f:
        a.frombytes(f.read())
    if sys.byteorder != 'little':
        a.byteswap()
    return a


def top(logits, k):
    idx = sorted(range(len(logits)), key=lambda i: (-logits[i], i))[:k]
    return [[i, f32(logits[i])] for i in idx]


def f32(x):
    """A float32 as the shortest decimal that round-trips through float32."""
    for digits in range(6, 10):
        s = '%.*g' % (digits, x)
        if array.array('f', [float(s)])[0] == x:
            return float(s)
    return float(x)


def argmax_first(logits):
    best = 0
    for i in range(1, len(logits)):
        if logits[i] > logits[best]:
            best = i
    return best


class Upstream:
    def __init__(self, oracle_dir):
        self.dir = oracle_dir
        self.qwen36 = os.path.join(oracle_dir, 'c', 'qwen36')
        self.olmoe = os.path.join(oracle_dir, 'c', 'olmoe')
        self.importer = os.path.join(oracle_dir, 'c', 'tools', 'gguf_import.py')
        self.tmp = tempfile.mkdtemp(prefix='oracle-', dir=oracle_dir)

    def tokenize(self, gguf, text):
        """qwen36 text mode with TOK_ONLY=1 (GGUF tokenizer, specials matched, no BOS)."""
        fd, path = tempfile.mkstemp(suffix='.txt', dir=self.tmp)
        with os.fdopen(fd, 'wb') as f:
            f.write(text.encode('utf-8'))
        rc, out, err = run([self.qwen36, '0', '4', path], {'SNAP': gguf, 'TOK_ONLY': '1'})
        if rc != 0:
            raise RuntimeError('TOK_ONLY failed: %s' % err[-2000:])
        return [int(x) for x in out.split()]

    def _ref(self, prompt, n):
        fd, path = tempfile.mkstemp(suffix='.json', dir=self.tmp)
        with os.fdopen(fd, 'w') as f:
            json.dump({'prompt_ids': prompt, 'full_ids': prompt + [0] * n}, f)
        return path

    def import_olmoe(self, gguf):
        d = gguf[:-5] + '.import'
        shutil.rmtree(d, ignore_errors=True)
        rc, out, err = run([sys.executable, '-I', self.importer, gguf, d], {})
        if rc != 0:
            raise RuntimeError('gguf_import failed: %s' % err[-2000:])
        return d

    def generate(self, engine, model, prompt, n, exact, dump=False, experts=0):
        """Greedy ids (ref.json oracle mode: float32 KV, never stops at EOS).
        Returns (ids, logits) where logits is the DUMP file (or None)."""
        ref = self._ref(prompt, n)
        env = {'SNAP': model, 'COLI_LOOKUP': '0'}
        if exact:
            env['GQ_EXACT'] = '1'
        dump_path = None
        if dump:
            fd, dump_path = tempfile.mkstemp(suffix='.f32', dir=self.tmp)
            os.close(fd)
            env['DUMP'] = dump_path
        if engine == 'qwen36':
            cmd = [self.qwen36, '0', '4', ref]
        else:
            cmd = [self.olmoe, str(max(experts, 8)), '8', ref]
        rc, out, err = run(cmd, env)
        ids = parse_engine_ids(out, err, '%s %s' % (engine, model))
        if len(ids) != n:
            raise RuntimeError('%s: %d ids, expected %d' % (model, len(ids), n))
        logits = read_f32(dump_path) if dump_path else None
        return ids, logits

    def steps(self, engine, model, prompt, n, vocab, experts=0, pool=None):
        """Greedy ids with GQ_EXACT=1 plus, for every step, the logits that chose the token.
        olmoe dumps every step in one run; qwen36 dumps only the last step, so it is run once
        per step count (deterministic: run k's ids are a prefix of run n's)."""
        if engine == 'olmoe':
            ids, lg = self.generate(engine, model, prompt, n, exact=True, dump=True, experts=experts)
            if len(lg) != n * vocab:
                raise RuntimeError('olmoe DUMP: %d floats, expected %d' % (len(lg), n * vocab))
            rows = [lg[s * vocab:(s + 1) * vocab] for s in range(n)]
            return ids, rows
        jobs = [(k,) for k in range(1, n + 1)]

        def one(k):
            ids_k, lg = self.generate(engine, model, prompt, k, exact=True, dump=True)
            if len(lg) != vocab:
                raise RuntimeError('qwen36 DUMP: %d floats, expected %d' % (len(lg), vocab))
            return k, ids_k, lg
        results = list((pool.map(lambda j: one(*j), jobs)) if pool else map(lambda j: one(*j), jobs))
        results.sort()
        ids = results[-1][1]
        rows = []
        for k, ids_k, lg in results:
            if ids_k != ids[:k]:
                raise RuntimeError('%s: run with %d steps gave %s, not a prefix of %s' % (model, k, ids_k, ids))
            rows.append(lg)
        return ids, rows


def step_records(ids, rows):
    recs = []
    for s, (tok, lg) in enumerate(zip(ids, rows)):
        t = top(lg, STEP_TOP)
        if argmax_first(lg) != tok:
            raise RuntimeError('step %d: dumped argmax %d != emitted %d' % (s, argmax_first(lg), tok))
        recs.append({'top': t, 'gap': f32(t[0][1] - t[1][1])})
    return recs


def byte_ids(text, specials):
    """Byte vocabulary: added tokens (ids 256..) matched longest first, other bytes are their id."""
    data, out, i = text.encode('utf-8'), [], 0
    sp = sorted(((s.encode('utf-8'), 256 + n) for n, s in enumerate(specials)), key=lambda x: -len(x[0]))
    while i < len(data):
        for s, tid in sp:
            if data.startswith(s, i):
                out.append(tid)
                i += len(s)
                break
        else:
            out.append(data[i])
            i += 1
    return out


OLMOE_SPECIALS = ['|||IP_ADDRESS|||', '<|endoftext|>', '<|padding|>']


def generate_tiny_models(out_dir, scratch):
    log('[record] writing tiny models with TinyModelTests/writeOracleModels ...')
    env = dict(os.environ, COLIBRI_TINY_OUT=out_dir)
    p = subprocess.run(['swift', 'test', '--scratch-path', scratch, '--filter', 'TinyModelTests/writeOracleModels'],
                       cwd=REPO, env=env)
    if p.returncode != 0:
        sys.exit('swift test (model writer) failed; fix the build or pass --skip-generate with --models')


def record_tiny(up, models_dir, only, pool):
    with open(os.path.join(models_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    os.makedirs(os.path.join(FIXTURES, 'tiny'), exist_ok=True)
    for name in sorted(manifest):
        if only and name not in only:
            continue
        m = manifest[name]
        t0 = time.time()
        gguf = os.path.join(models_dir, name + '.gguf')
        arch, vocab = m['architecture'], m['vocab_size']
        engine = 'olmoe' if arch == 'olmoe' else 'qwen36'
        model = up.import_olmoe(gguf) if engine == 'olmoe' else gguf
        qwenvocab = m['vocabulary'] != 'bytes'
        texts = TINY_PROMPTS['qwen' if (arch != 'olmoe' or qwenvocab) else 'olmoe']
        prompts = []
        for text in texts:
            if arch == 'olmoe' and not qwenvocab:
                pids = byte_ids(text, OLMOE_SPECIALS)
            else:
                pids = up.tokenize(REAL_GGUF if qwenvocab else gguf, text)
            ids, rows = up.steps(engine, model, pids, TINY_NEW, vocab, experts=m['experts'], pool=pool)
            ids_default, _ = up.generate(engine, model, pids, TINY_NEW, exact=False, experts=m['experts'])
            pre = rows[0]
            rec = {'text': text, 'prompt_ids': pids, 'greedy_exact': ids, 'greedy_default': ids_default,
                   'steps_exact': step_records(ids, rows)}
            if vocab <= 4096:
                rec['prefill_logits_exact'] = [f32(x) for x in pre]
            else:
                rec['prefill_top_exact'] = top(pre, REAL_PREFILL_TOP)
            prompts.append(rec)
        fixture = {
            'case': name, 'model': m,
            'upstream': {'commit': UPSTREAM_COMMIT, 'engine': engine,
                         'command': ('SNAP=<gguf_import dir> GQ_EXACT=1 DUMP=<f32> ./olmoe %d 8 ref.json' % max(m['experts'], 8))
                         if engine == 'olmoe' else 'SNAP=<gguf> COLI_LOOKUP=0 GQ_EXACT=1 DUMP=<f32> ./qwen36 0 4 ref.json',
                         'kv_cache': 'f32'},
            'n_new': TINY_NEW, 'prompts': prompts,
        }
        path = os.path.join(FIXTURES, 'tiny', name + '.json')
        with open(path, 'w') as f:
            json.dump(fixture, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')
        log('[record] %-28s %5.1fs  %s' % (name, time.time() - t0, ' | '.join(' '.join(map(str, p['greedy_exact'][:8])) for p in prompts)))


def record_real(up, pool):
    if not os.path.exists(REAL_GGUF):
        log('[record] no %s, skipping the real model' % REAL_GGUF)
        return
    vocab = 151936
    prompts = []
    for name, text in REAL_PROMPTS:
        t0 = time.time()
        pids = up.tokenize(REAL_GGUF, text)
        ids, rows = up.steps('qwen36', REAL_GGUF, pids, REAL_NEW, vocab, pool=pool)
        ids_default, pre_default = up.generate('qwen36', REAL_GGUF, pids, 1, exact=False, dump=True)
        ids_default, _ = up.generate('qwen36', REAL_GGUF, pids, REAL_NEW, exact=False)
        prompts.append({'name': name, 'text': text, 'prompt_ids': pids,
                        'greedy_exact': ids, 'greedy_default': ids_default,
                        'steps_exact': step_records(ids, rows),
                        'prefill_top_exact': top(rows[0], REAL_PREFILL_TOP),
                        'prefill_top_default': top(pre_default, REAL_PREFILL_TOP)})
        log('[record] real %-18s %5.1fs  %d prompt ids -> %s' % (name, time.time() - t0, len(pids), ' '.join(map(str, ids))))
    st = os.stat(REAL_GGUF)
    fixture = {
        'model': {'file': os.path.basename(REAL_GGUF), 'bytes': st.st_size, 'architecture': 'qwen3', 'vocab_size': vocab},
        'upstream': {'commit': UPSTREAM_COMMIT, 'engine': 'qwen36',
                     'command': 'SNAP=<gguf> COLI_LOOKUP=0 GQ_EXACT=1 DUMP=<f32> ./qwen36 0 4 ref.json',
                     'tokenize': 'SNAP=<gguf> TOK_ONLY=1 ./qwen36 0 4 prompt.txt', 'kv_cache': 'f32'},
        'n_new': REAL_NEW, 'prompts': prompts,
    }
    os.makedirs(FIXTURES, exist_ok=True)
    with open(os.path.join(FIXTURES, 'qwen3-0.6b-q8_0.json'), 'w') as f:
        json.dump(fixture, f, ensure_ascii=False, indent=1)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiny', action='store_true', help='only the synthetic models')
    ap.add_argument('--real', action='store_true', help='only Qwen3-0.6B')
    ap.add_argument('--cases', default='', help='comma-separated tiny case names')
    ap.add_argument('--models', default=os.path.join(ORACLE_DIR, 'tiny'), help='tiny model directory')
    ap.add_argument('--skip-generate', action='store_true', help='reuse the tiny models already in --models')
    ap.add_argument('--scratch', default=os.path.join(REPO, '.build-oracle'), help='SwiftPM scratch path')
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    both = not a.tiny and not a.real
    subprocess.run([os.path.join(REPO, 'scripts', 'oracle', 'build-upstream.sh')], check=True,
                   env=dict(os.environ, ORACLE_DIR=ORACLE_DIR))
    up = Upstream(ORACLE_DIR)
    try:
        with cf.ThreadPoolExecutor(a.jobs) as pool:
            if a.tiny or both:
                if not a.skip_generate:
                    generate_tiny_models(a.models, a.scratch)
                record_tiny(up, a.models, set(filter(None, a.cases.split(','))), pool)
            if a.real or both:
                record_real(up, pool)
    finally:
        shutil.rmtree(up.tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
