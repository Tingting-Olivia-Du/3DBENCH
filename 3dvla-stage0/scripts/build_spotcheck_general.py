#!/usr/bin/env python3
"""general-slice reviewer (LLaVA-COCO + ShareGPT4V-COCO, anti-forgetting, 40%).

Dialect contract for this slice (two-dialect policy): NATIVE RefCOCO-style
[0,1] floats, NO [0,2000) preamble. While building, full-scans both files to
audit: preamble contamination == 0 and all coords within [0,1].

Strata: llava coord-bearing (grounding/region-VQA) / llava plain / sharegpt4v
dense captions. Coord cards render SIDE-BY-SIDE original | annotated copy
(user-requested layout): red = answer coords, blue = question coords, drawn
for ONE coord-bearing QA pair so each card is a verifiable unit.
"""
import ast, hashlib, json, random, re, shutil
from pathlib import Path

import cv2
import numpy as np

DATA = Path('/workspace/tingting/3dvla-data')
SFT = DATA / 'sft'
ROOT = DATA / 'general'
OUT = Path('/workspace/tingting/3dvla-stage0/out')
IMG = OUT / 'spotcheck_general_imgs'
shutil.rmtree(IMG, ignore_errors=True)
IMG.mkdir()
random.seed(7)

FLOAT_LIST = re.compile(r'\[\s*(?:-?\d+\.\d+\s*,\s*){1,3}-?\d+\.\d+\s*\]')
FILES = ['general_llava_coco.jsonl', 'general_sharegpt4v_coco.jsonl']


def coords_of(text):
    out = []
    for m in FLOAT_LIST.finditer(text):
        try:
            v = ast.literal_eval(m.group(0))
        except Exception:
            continue
        if len(v) in (2, 4):
            out.append(v)
    return out


def draw(img, rows, color):
    H, W = img.shape[:2]
    for r in rows:
        if len(r) == 2:
            cv2.drawMarker(img, (int(r[0]*W), int(r[1]*H)), color, cv2.MARKER_CROSS, 18, 2)
        else:
            cv2.rectangle(img, (int(r[0]*W), int(r[1]*H)), (int(r[2]*W), int(r[3]*H)), color, 2)


# ---- full-scan audit + bucket offsets -------------------------------------
buckets = {}   # stratum -> [(file, offset)]
audit = {'preamble_contam': 0, 'coord_out_of_range': 0, 'coord_records': 0, 'records': 0}
for f in FILES:
    p = SFT / f
    with open(p) as fh:
        off = fh.tell(); l = fh.readline()
        while l:
            d = json.loads(l)
            text = ' '.join(c['value'] for c in d['conversations'])
            audit['records'] += 1
            if '[0,2000)' in text or '[0, 2000)' in text:
                audit['preamble_contam'] += 1
            cs = coords_of(text)
            if cs:
                audit['coord_records'] += 1
                vals = [x for r in cs for x in r]
                if max(vals) > 1.0 or min(vals) < 0.0:
                    audit['coord_out_of_range'] += 1
                stratum = 'llava_coord'
            else:
                stratum = 'llava_plain' if 'llava' in f else 'sharegpt4v'
            buckets.setdefault(stratum, []).append((f, off))
            off = fh.tell(); l = fh.readline()

print(f"full-scan audit over {audit['records']:,}: preamble_contam={audit['preamble_contam']} "
      f"coord_out_of_range={audit['coord_out_of_range']} coord_records={audit['coord_records']:,}")
for k, v in sorted(buckets.items()):
    print(f'  {k:<12} {len(v):>9,}')

PER = {'llava_coord': 14, 'llava_plain': 8, 'sharegpt4v': 8}
TOK = hashlib.md5(json.dumps({k: len(v) for k, v in buckets.items()},
                             sort_keys=True).encode()).hexdigest()[:6]
items = []
for stratum, offs in sorted(buckets.items()):
    for f, off in random.sample(offs, min(PER[stratum], len(offs))):
        with open(SFT / f) as fh:
            fh.seek(off)
            d = json.loads(fh.readline())
        img = cv2.imread(str(ROOT / d['images'][0]))
        if img is None:
            continue
        conv = d['conversations']
        pairs = [(conv[i]['value'], conv[i+1]['value']) for i in range(0, len(conv)-1, 2)]
        # pick the QA pair to display: first coord-bearing pair, else first pair
        sel = next((i for i, (q, a) in enumerate(pairs) if coords_of(q) or coords_of(a)), 0)
        q_raw, a_raw = pairs[sel]
        q = re.sub(r'<image>\s*', '', q_raw).strip()
        qc, ac = coords_of(q_raw), coords_of(a_raw)
        if qc or ac:
            anno = img.copy()
            draw(anno, ac, (46, 46, 230))    # red: answer coords
            draw(anno, qc, (230, 120, 30))   # blue: question coords
            div = np.full((img.shape[0], 6, 3), 255, np.uint8)
            canvas = np.concatenate([img, div, anno], 1)
        else:
            canvas = img
        qid = 'n_' + hashlib.md5((f + str(off)).encode()).hexdigest()[:12]
        cv2.imwrite(str(IMG / f'{TOK}_{qid}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
        items.append({'qid': qid, 't': f'[general] {stratum} · 轮{sel+1}/{len(pairs)}',
                      'ep': d['images'][0][-30:], 'frame': 0,
                      'view': (f'红=答案坐标{len(ac)} 蓝=问题坐标{len(qc)} (原生[0,1], 左原图右标注)'
                               if (qc or ac) else '无坐标 (纯对话/长描述)'),
                      'q': q[:500], 'a': a_raw[:500]})

tmpl = (OUT / 'spotcheck.html').read_text()
page = tmpl.replace('spotcheck_imgs/', 'spotcheck_general_imgs/')
page = re.sub(r"const TOKEN = '[a-f0-9]*'", f"const TOKEN = '{TOK}'", page)
page = page.replace("const KEY = 'spotcheck_m1_v2';", "const KEY = 'spotcheck_general_v2';")
page = page.replace('<title>Benchmark GT 人工抽检</title>', '<title>general 切片抽检</title>')
page = page.replace('<h1>GT 人工抽检</h1>',
                    f'<h1>general 抽检 — 原生[0,1]方言(无前言); 全量审计: 前言污染='
                    f'{audit["preamble_contam"]}, 坐标越界={audit["coord_out_of_range"]}'
                    f'/{audit["coord_records"]:,}坐标记录</h1>')
page = page.replace("a.download = 'spotcheck_verdicts.csv'",
                    "a.download = 'spotcheck_general_verdicts.csv'")
page = re.sub(r'const ITEMS = \[.*?\];\n',
              lambda m: 'const ITEMS = ' + json.dumps(items, ensure_ascii=False) + ';\n',
              page, count=1, flags=re.S)
(OUT / 'spotcheck_general.html').write_text(page)
print(f'{len(items)} items (token {TOK}) -> localhost:8737/spotcheck_general.html')
