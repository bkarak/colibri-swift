#!/usr/bin/env python3
"""Run coli-swift on the oracle fixtures' inputs and report the first divergence per case.

Called by compare.sh (which builds coli-swift and writes the tiny models); it can also be
run directly. Python standard library only; run with `python3 -I`.

For every fixture prompt it runs

  coli-swift generate --model <gguf> --backend <b> --raw --prompt <text> --temp 0
                      --max-tokens <n> --ignore-eos --kv f32 --dump [extra args]

(or `--prompt-ids` instead of `--prompt` with --ids), reads `prompt_ids (n): ...` and
`generated_ids (n): ...` from stderr, and compares them with upstream's recorded prompt
ids and greedy ids. A divergence at a step where upstream's own top-2 logit gap is below
--near-tie is reported as a near tie (summation order can flip it) and does not fail the
run; any other divergence, a prompt-id mismatch or a crash does.

Exit status: 0 when every case matches (near ties allowed), 1 otherwise, 2 on usage errors.
"""
import argparse, concurrent.futures as cf, glob, json, os, re, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES = os.path.join(REPO, 'Tests', 'ColibriCoreTests', 'Fixtures', 'oracle')
REAL_GGUF = os.environ.get('COLIBRI_QWEN3_GGUF',
                           os.path.expanduser('~/colibri-models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf'))


def parse_ids(stderr, label):
    m = re.search(r'^%s \((\d+)\): ?(.*)$' % label, stderr, re.M)
    if not m:
        return None
    ids = [int(x) for x in m.group(2).split()]
    if len(ids) != int(m.group(1)):
        return None
    return ids


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


class Job:
    def __init__(self, label, gguf, prompt, fixture_n, ref_ids, steps):
        self.label, self.gguf, self.prompt, self.n = label, gguf, prompt, fixture_n
        self.ref_ids, self.steps = ref_ids, steps


def run_job(j, a):
    cmd = [a.coli_swift, 'generate', '--model', j.gguf, '--backend', a.backend, '--temp', '0',
           '--max-tokens', str(j.n), '--ignore-eos', '--kv', 'f32', '--dump', '--raw']
    if a.ids:
        cmd += ['--prompt-ids', ' '.join(map(str, j.prompt['prompt_ids']))]
    else:
        cmd += ['--prompt', j.prompt['text']]
    if a.backend == 'cpu' and a.threads:
        cmd += ['--threads', str(a.threads)]
    if a.reference == 'default':
        cmd += ['--int8']
    cmd += a.extra
    t0 = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    dt = time.time() - t0
    err = p.stderr.decode('utf-8', 'replace')
    r = {'label': j.label, 'seconds': dt, 'cmd': cmd}
    if p.returncode != 0:
        r['status'] = 'ERROR'
        r['detail'] = 'exit %d: %s' % (p.returncode, err.strip().splitlines()[-1] if err.strip() else '')
        return r
    pids = parse_ids(err, 'prompt_ids')
    gids = parse_ids(err, 'generated_ids')
    if pids is None or gids is None:
        r['status'] = 'ERROR'
        r['detail'] = 'no "prompt_ids (n): ..." / "generated_ids (n): ..." lines on stderr (needs --dump)'
        return r
    want_p = j.prompt['prompt_ids']
    d = first_diff(pids, want_p)
    if d is not None:
        r['status'] = 'TOKENIZER'
        r['detail'] = 'prompt ids differ at %d: swift %s, upstream %s' % (
            d, pids[d] if d < len(pids) else 'end', want_p[d] if d < len(want_p) else 'end')
        return r
    d = first_diff(gids, j.ref_ids)
    if d is None:
        r['status'] = 'MATCH'
        r['detail'] = '%d/%d ids' % (len(gids), len(j.ref_ids))
        return r
    s = j.steps[d] if j.steps and d < len(j.steps) else None
    sw = gids[d] if d < len(gids) else 'end'
    detail = 'step %d: swift %s, upstream %s' % (d, sw, j.ref_ids[d] if d < len(j.ref_ids) else 'end')
    if s:
        top = ' '.join('%d:%g' % (int(t[0]), t[1]) for t in s['top'][:3])
        detail += ' (upstream top-2 gap %g; top %s)' % (s['gap'], top)
        if isinstance(sw, int):
            ranks = [int(t[0]) for t in s['top']]
            if sw in ranks:
                detail += '; swift picked upstream rank %d' % (ranks.index(sw) + 1)
    near = s is not None and s['gap'] < a.near_tie
    r['status'] = 'NEAR-TIE' if near else 'DIVERGED'
    r['detail'] = detail + ('; %d ids matched before' % d)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--coli-swift', required=True, help='path to the coli-swift binary')
    ap.add_argument('--models', required=True, help='directory with the tiny <case>.gguf files and manifest.json')
    ap.add_argument('--backend', default='cpu', choices=['cpu', 'metal', 'auto'])
    ap.add_argument('--reference', default='exact', choices=['exact', 'default'],
                    help='exact: upstream GQ_EXACT=1 ids (float math); default: upstream int8-activation ids, '
                         'compared with coli-swift --int8 (cpu only)')
    ap.add_argument('--cases', default='', help='comma-separated tiny case names (default: all)')
    ap.add_argument('--tiny-only', action='store_true')
    ap.add_argument('--real-only', action='store_true')
    ap.add_argument('--ids', action='store_true', help='feed upstream prompt ids (--prompt-ids) instead of text')
    ap.add_argument('--near-tie', type=float, default=0.01)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--threads', type=int, default=0, help='cpu: --threads per run (default: cores / jobs)')
    ap.add_argument('--verbose', action='store_true', help='print each command')
    ap.add_argument('extra', nargs='*', help='extra coli-swift arguments (after --)')
    a = ap.parse_args()
    if a.reference == 'default' and a.backend != 'cpu':
        ap.error('--reference default needs --backend cpu (int8 activations are a CPU path)')
    if a.backend == 'cpu' and not a.threads:
        a.threads = max(1, (os.cpu_count() or 4) // max(1, a.jobs))
    key = 'greedy_exact' if a.reference == 'exact' else 'greedy_default'
    only = set(filter(None, a.cases.split(',')))

    jobs, problems = [], []
    if not a.real_only:
        manifest_path = os.path.join(a.models, 'manifest.json')
        manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
        for path in sorted(glob.glob(os.path.join(FIXTURES, 'tiny', '*.json'))):
            fx = json.load(open(path))
            name = fx['case']
            if only and name not in only:
                continue
            gguf = os.path.join(a.models, name + '.gguf')
            m = manifest.get(name)
            if not os.path.exists(gguf) or m is None:
                problems.append({'label': name, 'status': 'MISSING', 'detail': 'no %s in %s' % (name + '.gguf', a.models)})
                continue
            if m['fingerprint'] != fx['model']['fingerprint']:
                problems.append({'label': name, 'status': 'STALE', 'detail': 'model fingerprint %s, fixture %s: '
                                 're-record (record.py --tiny) or regenerate the models' % (m['fingerprint'], fx['model']['fingerprint'])})
                continue
            for i, p in enumerate(fx['prompts']):
                steps = p['steps_exact'] if a.reference == 'exact' else None
                jobs.append(Job('%s#%d' % (name, i), gguf, p, fx['n_new'], p[key], steps))
    if not a.tiny_only and not only:
        real = os.path.join(FIXTURES, 'qwen3-0.6b-q8_0.json')
        if not os.path.exists(REAL_GGUF):
            print('note: no %s, skipping the real model' % REAL_GGUF)
        else:
            fx = json.load(open(real))
            if os.path.getsize(REAL_GGUF) != fx['model']['bytes']:
                problems.append({'label': 'qwen3-0.6b', 'status': 'STALE', 'detail': 'file size differs from the fixture'})
            else:
                for p in fx['prompts']:
                    steps = p['steps_exact'] if a.reference == 'exact' else None
                    jobs.append(Job('qwen3-0.6b/' + p['name'], REAL_GGUF, p, fx['n_new'], p[key], steps))

    print('coli-swift %s, backend %s, reference upstream %s (%s), %d runs, %d jobs'
          % (a.coli_swift, a.backend, a.reference, 'GQ_EXACT=1' if a.reference == 'exact' else 'int8 activations',
             len(jobs), a.jobs))
    # Runs on the same model go one after another: coli-swift saves the routing history
    # next to the model when it exits, and concurrent saves of one file can collide.
    groups = {}
    for j in jobs:
        groups.setdefault(j.gguf, []).append(j)
    results = list(problems)
    with cf.ThreadPoolExecutor(a.jobs) as pool:
        for group in pool.map(lambda g: [run_job(j, a) for j in g], groups.values()):
            for r in group:
                results.append(r)
                sec = ' %5.1fs' % r['seconds'] if 'seconds' in r else ''
                print('%-9s %-34s%s  %s' % (r['status'], r['label'], sec, r['detail']), flush=True)
                if a.verbose and 'cmd' in r:
                    print('          ' + ' '.join(repr(c) if ' ' in c or '\n' in c else c for c in r['cmd']))
    for r in problems:
        print('%-9s %-34s  %s' % (r['status'], r['label'], r['detail']))

    counts = {}
    for r in results:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    print('summary: ' + ', '.join('%s %d' % kv for kv in sorted(counts.items())))
    bad = [r for r in results if r['status'] not in ('MATCH', 'NEAR-TIE')]
    sys.exit(1 if bad or not results else 0)


if __name__ == '__main__':
    main()
