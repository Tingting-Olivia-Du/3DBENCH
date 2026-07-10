#!/usr/bin/env python3
"""Final data-freeze audit across all five slices + benchmark separation.
Programmatic re-verification of every invariant in DATA_SPEC.md:

  A. counts: every mixture_manifest file recounted
  B. fine3d: K non-null (full), train chunks exclude bench chunk-001 (full),
     coord range/preamble + paths (sampled)
  C. anchor: full scan — K/Ks, coord range, demo ids <= 44 (bench uses 45-49)
  D. embodied: recount vs vlaser_pipeline_report + sampled re-check
  E. generic3d: full scan — metric-numeric must carry K == K_SCANNET
  F. general: full scan — zero preamble contamination, coords within [0,1]
  G. variants: template filters subset of templates actually present in fine3d
  H. benchmark: m1_bench all chunk-001; libero_bench all demo>=45

Output: sft/final_audit_report.json + console summary. Exit 1 on any FAIL.
"""
import glob, json, os, random, re, sys
from pathlib import Path

PREAMBLE = "Pixel coordinates in this conversation are normalized to [0,2000)"
DATA = Path('/workspace/tingting/3dvla-data')
SFT = DATA / 'sft'
LIST_RE = re.compile(r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)'
                     r'(?:\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*))?\s*\]')
METRIC_Q = re.compile(r'\bin (meters|centimeters)\b')
NUM_A = re.compile(r'^-?\d+\.?\d*$')
K_SCANNET = [1170.19, 1170.19, 647.75, 483.75]
random.seed(0)
R = {}
fails = []


def gpt_coords(d):
    txt = ' '.join(c['value'] for c in d['conversations'] if c['from'] == 'gpt')
    return [float(x) for m in LIST_RE.finditer(txt) for x in m.groups() if x]


def check(name, cond, detail):
    R[name] = {'ok': bool(cond), 'detail': detail}
    print(f"  [{'OK' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        fails.append(name)


# ---- A. manifest recount ---------------------------------------------------
print('== A. manifest recount')
man = json.load(open(SFT / 'mixture_manifest.json'))
mismatch = []
totals = {}
for sl, spec in man['slices'].items():
    t = 0
    for e in spec['files']:
        p = SFT / e['file']
        n = sum(1 for _ in open(p))
        t += n
        if n != e['convs']:
            mismatch.append((e['file'], e['convs'], n))
    totals[sl] = t
check('A_counts', not mismatch, f'totals={totals}' + (f' mismatch={mismatch}' if mismatch else ''))

# ---- B. fine3d --------------------------------------------------------------
print('== B. fine3d')
packs = sorted(glob.glob(str(SFT / 'm1_chunk*_pack.jsonl')))
k_null = 0
chunks = set()
templates_seen = set()
coord_bad = pre_miss = coord_rec = path_miss = 0
for pk in packs:
    lines = open(pk).readlines()
    for i, l in enumerate(lines):
        d = json.loads(l)
        if d['K'] is None:
            k_null += 1
        chunks.add(re.search(r'/frames/(chunk-\d+)/', d['images'][0]).group(1))
        templates_seen.update(d.get('templates', []))
        if i < 300:  # sampled content checks
            vals = gpt_coords(d)
            if vals:
                coord_rec += 1
                if max(vals) >= 2000 or min(vals) < 0:
                    coord_bad += 1
                if PREAMBLE not in d['conversations'][0]['value']:
                    pre_miss += 1
    for l in random.sample(lines, min(3, len(lines))):
        if not os.path.exists(json.loads(l)['images'][0]):
            path_miss += 1
check('B_K_nonnull', k_null == 0, f'{k_null} null-K over {totals["fine3d"]:,}')
check('B_bench_chunk_excluded', 'chunk-001' not in chunks,
      f'{len(chunks)} train chunks, chunk-001 excluded={("chunk-001" not in chunks)}')
check('B_coords_preamble', coord_bad == 0 and pre_miss == 0,
      f'sampled {coord_rec:,} coord-recs: out-of-range={coord_bad} preamble-missing={pre_miss}')
check('B_paths', path_miss == 0, f'{len(packs)*3} sampled paths, missing={path_miss}')

# ---- C. anchor ---------------------------------------------------------------
print('== C. anchor')
k_null = coord_bad = pre_miss = coord_rec = path_miss = 0
demos = set()
n = 0
for f in ['libero_anchor_pack.jsonl', 'libero_anchor_t5b_pack.jsonl']:
    for l in open(SFT / f):
        d = json.loads(l)
        n += 1
        if d['K'] is None:
            k_null += 1
        m = re.search(r'demo(\d+)/', d['images'][0])
        demos.add(int(m.group(1)))
        vals = gpt_coords(d)
        if vals:
            coord_rec += 1
            if max(vals) >= 2000 or min(vals) < 0:
                coord_bad += 1
            if PREAMBLE not in d['conversations'][0]['value']:
                pre_miss += 1
        if random.random() < 0.005 and not os.path.exists(d['images'][0]):
            path_miss += 1
check('C_K_nonnull', k_null == 0, f'{k_null} null-K over {n:,}')
check('C_demo_holdout', max(demos) <= 44, f'train demos {min(demos)}..{max(demos)} (bench=45-49)')
check('C_coords_preamble', coord_bad == 0 and pre_miss == 0,
      f'{coord_rec:,} coord-recs (full): out-of-range={coord_bad} preamble-missing={pre_miss}')
check('C_paths', path_miss == 0, f'~0.5% sampled, missing={path_miss}')

# ---- D. embodied -------------------------------------------------------------
print('== D. embodied')
rep = json.load(open(SFT / 'vlaser_pipeline_report.json'))
emb_files = ['vlaser_refspatial_3d', 'vlaser_refspatial_simulator',
             'vlaser_bounding_box_data_single', 'vlaser_affordance',
             'vlaser_trajectory', 'vlaser_paco_lvis_v1_train']
rec_match = all(rep['sources'][s]['validation']['records']
                == sum(1 for _ in open(SFT / f'{s}.jsonl')) for s in emb_files)
tag_bad = coord_bad = 0
for s in emb_files:
    lines = open(SFT / f'{s}.jsonl').readlines()
    for l in random.sample(lines, min(400, len(lines))):
        d = json.loads(l)
        txt = ' '.join(c['value'] for c in d['conversations'])
        if re.search(r'</?(point|box|ref)>', txt):
            tag_bad += 1
        vals = gpt_coords(d)
        if vals and (max(vals) >= 2001 or min(vals) < -1):
            coord_bad += 1
check('D_pipeline_report', rep['all_ok'] and rec_match,
      f'report all_ok={rep["all_ok"]}, recount match={rec_match}')
check('D_sampled_recheck', tag_bad == 0 and coord_bad == 0,
      f'2,400 sampled: residual-tags={tag_bad} coord-out-of-range={coord_bad}')

# ---- E. generic3d ------------------------------------------------------------
print('== E. generic3d')
viol = k_wrong = 0
n = 0
for s in ['vlaser_vsi_100k_merged_all', 'vlaser_vlm_3r_merged_all']:
    for l in open(SFT / f'{s}.jsonl'):
        d = json.loads(l)
        n += 1
        pp = any('scannetpp' in i for i in d['images'])
        q = d['conversations'][0]['value']
        a = d['conversations'][1]['value'].strip()
        if METRIC_Q.search(q) and NUM_A.match(a) and d['K'] is None:
            viol += 1
        if (d['K'] is None) != pp or (not pp and d['K'] != K_SCANNET):
            k_wrong += 1
check('E_metric_needs_K', viol == 0, f'{viol} metric-without-K over {n:,}')
check('E_K_values', k_wrong == 0, f'{k_wrong} wrong K assignment (scannet=K_SCANNET, scannetpp=None)')

# ---- F. general --------------------------------------------------------------
print('== F. general')
contam = oob = coord_rec = 0
n = 0
FL = re.compile(r'\[\s*(?:-?\d+\.\d+\s*,\s*){1,3}-?\d+\.\d+\s*\]')
for s in ['general_llava_coco', 'general_sharegpt4v_coco']:
    for l in open(SFT / f'{s}.jsonl'):
        d = json.loads(l)
        n += 1
        txt = ' '.join(c['value'] for c in d['conversations'])
        if PREAMBLE in txt:
            contam += 1
        for m in FL.finditer(txt):
            coord_rec += 1
            vs = [float(x) for x in re.findall(r'-?\d+\.\d+', m.group(0))]
            if max(vs) > 1.0 or min(vs) < 0.0:
                oob += 1
            break
check('F_native_dialect', contam == 0 and oob == 0,
      f'over {n:,}: preamble-contam={contam} out-of-[0,1]={oob} ({coord_rec:,} coord-recs)')

# ---- G. variants -------------------------------------------------------------
print('== G. variants')
bad = []
vfiles = sorted(glob.glob(str(SFT / 'variants/*.json')))
for vf in vfiles:
    v = json.load(open(vf))
    filt = (v.get('slices', {}).get('fine3d', {}) or {}).get('template_filter')
    for t in (filt or []):
        if t not in templates_seen:
            bad.append((Path(vf).name, t))
check('G_variant_templates', len(vfiles) >= 6 and not bad,
      f'{len(vfiles)} variant specs; unknown templates={bad or "none"}; '
      f'templates in packs={sorted(templates_seen)}')

# ---- H. benchmark ------------------------------------------------------------
print('== H. benchmark separation')
m1_chunks = {re.search(r'(chunk-\d+)', json.loads(l)['image']).group(1)
             for l in open(DATA / 'benchmark/m1_bench.jsonl')}
lb_demos = {int(re.search(r'demo(\d+)', json.loads(l)['image']).group(1))
            for l in open(DATA / 'benchmark/libero_bench.jsonl')}
check('H_m1_bench', m1_chunks == {'chunk-001'}, f'm1_bench chunks={sorted(m1_chunks)}')
check('H_libero_bench', min(lb_demos) >= 45, f'libero_bench demos={sorted(lb_demos)}')

json.dump(R, open(SFT / 'final_audit_report.json', 'w'), indent=1)
print(f"\n== {'ALL OK' if not fails else 'FAILURES: ' + ', '.join(fails)} "
      f"-> sft/final_audit_report.json")
sys.exit(1 if fails else 0)
