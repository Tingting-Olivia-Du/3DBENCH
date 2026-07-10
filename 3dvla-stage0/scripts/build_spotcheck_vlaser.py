#!/usr/bin/env python3
"""Vlaser-slice reviewer. Coordinate sources: converted coords drawn back on the
image (x2 verification). Multi-image records (video QA, up to 32 frames): render
a contact-sheet GRID of ALL frames (truncating to 2 frames previously caused
false image-text-mismatch impressions — user-caught)."""
import ast, hashlib, json, random, re, shutil
from pathlib import Path

import cv2
import numpy as np

DATA = Path('/workspace/tingting/3dvla-data')
SFT = DATA / 'sft'
OUT = Path('/workspace/tingting/3dvla-stage0/out')
IMG = OUT / 'spotcheck_vlaser_imgs'
shutil.rmtree(IMG, ignore_errors=True)
IMG.mkdir()
NUMS = re.compile(r'\[(?:\s*-?\d+\.?\d*\s*,)+\s*-?\d+\.?\d*\s*\]')
random.seed(4)
roots = json.load(open(SFT / 'vlaser_image_roots.json'))
roots['vlaser_bounding_box_data_single'] = {'root': str(DATA / 'vlaser/grounding_data')}

def draw(img, text):
    H, W = img.shape[:2]
    n = 0
    for m in NUMS.finditer(text):
        try:
            v = ast.literal_eval(m.group(0))
        except Exception:
            continue
        rows = v if isinstance(v[0], list) else [v]
        for r in rows:
            if len(r) == 2:
                cv2.drawMarker(img, (int(r[0]/2000*W), int(r[1]/2000*H)), (46,46,230),
                               cv2.MARKER_CROSS, 16, 2); n += 1
            elif len(r) == 4:
                cv2.rectangle(img, (int(r[0]/2000*W), int(r[1]/2000*H)),
                              (int(r[2]/2000*W), int(r[3]/2000*H)), (46,46,230), 2); n += 1
    return n

TOK = hashlib.md5(str(sorted(roots)).encode()).hexdigest()[:6]
items = []
for src, info in sorted(roots.items()):
    f = SFT / f'{src}.jsonl'
    if not f.exists():
        continue
    per = 4 if ('vsi' in src or 'vlm_3r' in src) else 8
    lines = open(f).readlines()
    for l in random.sample(lines, min(per, len(lines))):
        d = json.loads(l)
        imgs = d['images']
        mats = [cv2.imread(str(Path(info['root']) / i)) for i in imgs]
        if any(m is None for m in mats):
            continue
        conv = d['conversations']
        q = re.sub(r'(Image-\d+:\s*)?<image>\s*', '', conv[0]['value']).strip()
        a = conv[1]['value']
        n = draw(mats[0], a) + draw(mats[0], q)
        if len(mats) == 1:
            canvas = mats[0]
        else:  # contact-sheet grid of ALL frames
            tiles = []
            for i, m in enumerate(mats):
                t = cv2.resize(m, (300, 225))
                cv2.putText(t, f'{i+1}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                tiles.append(t)
            per_row = 4
            rows = [np.concatenate(tiles[i:i+per_row], 1) for i in range(0, len(tiles), per_row)]
            Wm = max(r.shape[1] for r in rows)
            rows = [np.pad(r, ((0,4),(0,Wm-r.shape[1]),(0,0)), constant_values=30) for r in rows]
            canvas = np.concatenate(rows, 0)
        qid = 'v_' + hashlib.md5((src + str(imgs) + q[:40]).encode()).hexdigest()[:12]
        cv2.imwrite(str(IMG / f'{TOK}_{qid}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
        items.append({'qid': qid, 't': ('[generic3d] ' if ('vsi' in src or 'vlm_3r' in src) else '[embodied] ') + src.replace('vlaser_', ''), 'ep': imgs[0][-38:],
                      'frame': 0, 'view': f'{len(imgs)} imgs, {n} coord marks',
                      'q': q[:450], 'a': a[:450]})

tmpl = (OUT / 'spotcheck.html').read_text()
page = tmpl.replace('spotcheck_imgs/', 'spotcheck_vlaser_imgs/')
page = re.sub(r"const TOKEN = '[a-f0-9]*'", f"const TOKEN = '{TOK}'", page)
page = page.replace("const KEY = 'spotcheck_m1_v2';", "const KEY = 'spotcheck_vlaser_v2';")
page = page.replace('<title>Benchmark GT 人工抽检</title>', '<title>Vlaser 切片 GT 抽检</title>')
page = page.replace('<h1>GT 人工抽检</h1>',
                    '<h1>Vlaser 抽检 — 红=×2换算坐标; 多图记录显示全部帧(视频QA需看整个sheet)</h1>')
page = page.replace("a.download = 'spotcheck_verdicts.csv'", "a.download = 'spotcheck_vlaser_verdicts.csv'")
page = re.sub(r'const ITEMS = \[.*?\];\n',
              lambda m: 'const ITEMS = ' + json.dumps(items, ensure_ascii=False) + ';\n',
              page, count=1, flags=re.S)
(OUT / 'spotcheck_vlaser.html').write_text(page)
print(f'{len(items)} items (token {TOK}) -> localhost:8737/spotcheck_vlaser.html')
