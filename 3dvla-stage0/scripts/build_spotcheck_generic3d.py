#!/usr/bin/env python3
"""generic3d-slice reviewer (VSI + vlm_3r after metric-consistency fix).

Stratified sampling over (source file x scannet/scannetpp x question type) so
every cell is visible, incl. the rare scannetpp leftovers. While scanning the
FULL 268,702 records it also audits the metric-consistency principle: any
metric-numeric answer without K is a violation (must be zero after the
scannetpp drop). Multi-frame records render a contact-sheet grid of ALL frames.
"""
import hashlib, json, random, re, shutil
from pathlib import Path

import cv2
import numpy as np

DATA = Path('/workspace/tingting/3dvla-data')
SFT = DATA / 'sft'
ROOT = DATA / 'vlaser/spatial_data'
OUT = Path('/workspace/tingting/3dvla-stage0/out')
IMG = OUT / 'spotcheck_generic3d_imgs'
shutil.rmtree(IMG, ignore_errors=True)
IMG.mkdir()
random.seed(11)

METRIC_Q = re.compile(r'\bin (meters|centimeters)\b')
NUM_A = re.compile(r'^-?\d+\.?\d*$')
MCQ = re.compile(r'\b[A-D]\.\s')
FILES = ['vlaser_vsi_100k_merged_all', 'vlaser_vlm_3r_merged_all']
PER_BUCKET = 6

buckets = {}          # (src, ds, qtype) -> [(file, line_offset)]
violations = 0        # metric-numeric answer with K=None (must stay 0)
totals = {}
for src in FILES:
    f = SFT / f'{src}.jsonl'
    n = 0
    with open(f) as fh:
        off = fh.tell(); l = fh.readline()
        while l:
            d = json.loads(l)
            ds = 'scannetpp' if any('scannetpp' in i for i in d['images']) else 'scannet'
            q = d['conversations'][0]['value']
            a = d['conversations'][1]['value'].strip()
            metric = bool(METRIC_Q.search(q)) and bool(NUM_A.match(a))
            qt = 'metric_num' if metric else ('mcq' if MCQ.search(q) else 'other')
            if metric and d['K'] is None:
                violations += 1
            buckets.setdefault((src, ds, qt), []).append(off)
            n += 1
            off = fh.tell(); l = fh.readline()
    totals[src] = n

print(f'full-scan audit: {sum(totals.values()):,} records, '
      f'metric-without-K violations = {violations} (must be 0)')
for k, v in sorted(buckets.items()):
    print(f'  {k[0].replace("vlaser_","")[:14]:<14} {k[1]:<10} {k[2]:<10} {len(v):>7,}')

TOK = hashlib.md5(json.dumps({k[0]+k[1]+k[2]: len(v)
                              for k, v in buckets.items()}, sort_keys=True).encode()).hexdigest()[:6]
items = []
for (src, ds, qt), offs in sorted(buckets.items()):
    for off in random.sample(offs, min(PER_BUCKET, len(offs))):
        with open(SFT / f'{src}.jsonl') as fh:
            fh.seek(off)
            d = json.loads(fh.readline())
        mats = [cv2.imread(str(ROOT / i)) for i in d['images']]
        if any(m is None for m in mats):
            continue
        if len(mats) == 1:
            canvas = mats[0]
        else:
            tiles = []
            for i, m in enumerate(mats):
                t = cv2.resize(m, (300, 225))
                cv2.putText(t, f'{i+1}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                tiles.append(t)
            per_row = 4
            rows = [np.concatenate(tiles[i:i+per_row], 1) for i in range(0, len(tiles), per_row)]
            Wm = max(r.shape[1] for r in rows)
            rows = [np.pad(r, ((0, 4), (0, Wm - r.shape[1]), (0, 0)), constant_values=30) for r in rows]
            canvas = np.concatenate(rows, 0)
        conv = d['conversations']
        q = re.sub(r'(Image-\d+:\s*)?<image>\s*', '', conv[0]['value']).strip()
        a = conv[1]['value']
        k_str = 'K=None' if d['K'] is None else 'K=(%.0f,%.0f,%.0f,%.0f)' % tuple(d['K'])
        bad = (qt == 'metric_num' and d['K'] is None)
        qid = 'g_' + hashlib.md5((src + str(d['images']) + q[:40]).encode()).hexdigest()[:12]
        cv2.imwrite(str(IMG / f'{TOK}_{qid}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
        items.append({'qid': qid,
                      't': ('!!VIOLATION ' if bad else '') + f'[{src.replace("vlaser_", "").replace("_merged_all", "")}] {ds} · {qt}',
                      'ep': d['images'][0][-44:], 'frame': 0,
                      'view': f'{len(mats)} imgs · {k_str}',
                      'q': q[:600], 'a': a[:450]})

tmpl = (OUT / 'spotcheck.html').read_text()
page = tmpl.replace('spotcheck_imgs/', 'spotcheck_generic3d_imgs/')
page = re.sub(r"const TOKEN = '[a-f0-9]*'", f"const TOKEN = '{TOK}'", page)
page = page.replace("const KEY = 'spotcheck_m1_v2';", "const KEY = 'spotcheck_generic3d_v1';")
page = page.replace('<title>Benchmark GT 人工抽检</title>', '<title>generic3d 切片抽检</title>')
page = page.replace('<h1>GT 人工抽检</h1>',
                    f'<h1>generic3d 抽检 — 分层：源×scannet/scannetpp×题型; 多帧显示全部帧; '
                    f'全量审计 度量无K违例={violations}/268,702</h1>')
page = page.replace("a.download = 'spotcheck_verdicts.csv'",
                    "a.download = 'spotcheck_generic3d_verdicts.csv'")
page = re.sub(r'const ITEMS = \[.*?\];\n',
              lambda m: 'const ITEMS = ' + json.dumps(items, ensure_ascii=False) + ';\n',
              page, count=1, flags=re.S)
(OUT / 'spotcheck_generic3d.html').write_text(page)
print(f'{len(items)} items (token {TOK}) -> localhost:8737/spotcheck_generic3d.html')
