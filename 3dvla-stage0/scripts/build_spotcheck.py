#!/usr/bin/env python3
"""Build the local spot-check reviewer: pre-render GT overlays for every item in
spotcheck.csv and emit a single-page review UI (out/spotcheck.html + out/spotcheck_imgs/).

Serve via the existing http.server at localhost:8737 -> open /spotcheck.html.
Verdicts persist in localStorage; export button downloads spotcheck_verdicts.csv.
"""
import ast, csv, html, json, pickle, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from m1_calib import K_PIXEL_FIXED, world_to_cam_fixed, project
from qa_gen import uid_boxes, uid_center3d

DATA = Path("/workspace/tingting/3dvla-data")
OUT = Path("/workspace/tingting/3dvla-stage0/out")
IMG_DIR = OUT / "spotcheck_imgs"
import shutil
shutil.rmtree(IMG_DIR, ignore_errors=True)  # stale overlays from older qid numbering caused image/text mismatches
IMG_DIR.mkdir(exist_ok=True)
W, H, NORM = 640, 480, 2000.0
RED, BLU, YEL = (46, 46, 230), (220, 200, 40), (60, 200, 240)


def d(v, ax):
    return int(v / NORM * (W if ax == "x" else H))


def dbox(b):
    return d(b[0], "x"), d(b[1], "y"), d(b[2], "x"), d(b[3], "y")


bench = {json.loads(l)["qid"]: json.loads(l) for l in open(DATA / "benchmark" / "m1_bench.jsonl")}
rows = list(csv.DictReader(open(DATA / "benchmark" / "spotcheck.csv")))
recs = {}


def rec_of(ep):
    if ep not in recs:
        recs[ep] = pickle.load(open(DATA / "ann" / "chunk-001" / f"{ep}.pkl", "rb"))
    return recs[ep]


import hashlib as _h
TOK = _h.md5(open(DATA / "benchmark" / "spotcheck.csv","rb").read()).hexdigest()[:6]
items = []
for r in rows:
    q = bench[r["qid"]]
    t, ep, f = q["template"], q["episode"], q["frame"]
    rec = rec_of(ep)
    img = cv2.imread(q["image"])
    if t == "T1_grounding":
        b = ast.literal_eval(q["answer"])
        cv2.rectangle(img, dbox(b)[:2], dbox(b)[2:], RED, 2)
    elif t == "T5_where_to_act":
        p = ast.literal_eval(q["answer"])
        cv2.drawMarker(img, (d(p[0], "x"), d(p[1], "y")), RED, cv2.MARKER_CROSS, 20, 2)
    elif t == "T4_tcp_trace":
        pts = [(d(p[0], "x"), d(p[1], "y")) for p in ast.literal_eval(q["answer"])]
        for i, p in enumerate(pts):
            cv2.circle(img, p, 4, RED, -1)
            if i:
                cv2.line(img, pts[i - 1], p, BLU, 1)
    elif t in ("T2_metric3d_dist", "T3_depth_obj"):
        view = q["view"]
        boxes = uid_boxes(rec, f, view)
        R_cv, C = world_to_cam_fixed(view)
        uids = [rec["pick_obj_uid"]] + ([rec["place_obj_uid"]] if t == "T2_metric3d_dist" else [])
        cts = []
        for uid in uids:
            if uid in boxes:
                cv2.rectangle(img, tuple(boxes[uid][:2]), tuple(boxes[uid][2:]), RED, 2)
            c3, _ = uid_center3d(rec, f, uid)
            if c3 is not None:
                uv, vis = project(c3, R_cv, C, K_PIXEL_FIXED)
                if vis[0]:
                    cts.append((int(uv[0, 0]), int(uv[0, 1])))
        if t == "T2_metric3d_dist" and len(cts) == 2:
            cv2.line(img, cts[0], cts[1], BLU, 2)
            mid = ((cts[0][0] + cts[1][0]) // 2, (cts[0][1] + cts[1][1]) // 2)
            cv2.putText(img, f"{float(q['answer']):.0f}cm", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elif t == "T3_depth_tcp":
        uv = q["meta"]["tcp_uv"]
        cv2.drawMarker(img, (d(uv[0], "x"), d(uv[1], "y")), RED, cv2.MARKER_CROSS, 20, 2)
    elif t == "T6_cross_view":
        img2 = cv2.imread(q["image2"])
        qb = ast.literal_eval(q["question"].split("is at ")[1].split(" in the first")[0])
        ab = ast.literal_eval(q["answer"])
        cv2.rectangle(img, dbox(qb)[:2], dbox(qb)[2:], YEL, 2)
        cv2.rectangle(img2, dbox(ab)[:2], dbox(ab)[2:], RED, 2)
        img = np.concatenate([img, img2], 1)
    elif t == "T7_ego_motion":
        img2 = cv2.imread(q["image2"])
        cv2.putText(img, "t", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, YEL, 2)
        cv2.putText(img2, f"t+{q['meta']['gap']}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, YEL, 2)
        img = np.concatenate([img, img2], 1)
    cv2.imwrite(str(IMG_DIR / f"{TOK}_{q['qid']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    items.append({"qid": q["qid"], "t": t, "ep": ep, "frame": f, "view": str(q["view"]),
                  "q": q["question"], "a": q["answer"]})

payload = json.dumps(items, ensure_ascii=False)
page = """<meta charset="utf-8">
<title>Benchmark GT 人工抽检</title>
<style>
:root { --bg:#f4f5f4; --panel:#fff; --ink:#1c201e; --sub:#5b6461; --line:#dde1df;
  --accent:#0e7490; --ok:#15803d; --bad:#b42318; --mono:ui-monospace,'SF Mono',Consolas,monospace; }
@media (prefers-color-scheme: dark) { :root { --bg:#15181a; --panel:#1d2124; --ink:#e6e9e7;
  --sub:#98a29e; --line:#31383b; --accent:#39bcd8; --ok:#4ade80; --bad:#f87171; } }
:root[data-theme="dark"] { --bg:#15181a; --panel:#1d2124; --ink:#e6e9e7; --sub:#98a29e; --line:#31383b; --accent:#39bcd8; --ok:#4ade80; --bad:#f87171; }
:root[data-theme="light"] { --bg:#f4f5f4; --panel:#fff; --ink:#1c201e; --sub:#5b6461; --line:#dde1df; --accent:#0e7490; --ok:#15803d; --bad:#b42318; }
body { background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system,'Segoe UI','Noto Sans SC',sans-serif;
  max-width:1160px; margin:0 auto; padding:20px; }
header { display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin-bottom:12px; }
h1 { font-size:18px; margin:0; }
.bar { flex:1; height:8px; background:var(--line); border-radius:4px; overflow:hidden; min-width:160px; }
.bar i { display:block; height:100%; background:var(--accent); }
select,button,input { font:inherit; color:var(--ink); background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:6px 10px; }
button { cursor:pointer; }
button:focus-visible { outline:2px solid var(--accent); }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; }
.meta { font:12px var(--mono); color:var(--sub); }
img { width:100%; border-radius:6px; margin:10px 0; }
.q { margin:4px 0; }
.a { font:14px var(--mono); color:var(--accent); overflow-wrap:anywhere; }
.verdicts { display:flex; gap:10px; margin:14px 0 6px; flex-wrap:wrap; }
.verdicts button { font-size:15px; padding:9px 22px; border-width:2px; }
.v-ok.active { border-color:var(--ok); color:var(--ok); font-weight:700; }
.v-bad.active { border-color:var(--bad); color:var(--bad); font-weight:700; }
.v-un.active { border-color:var(--accent); color:var(--accent); font-weight:700; }
.note { width:100%; margin-top:6px; }
.hint { color:var(--sub); font-size:12.5px; margin-top:10px; }
.count { font:13px var(--mono); color:var(--sub); }
</style>
<header>
  <h1>GT 人工抽检</h1>
  <select id="tpl"></select>
  <div class="bar"><i id="prog"></i></div>
  <span class="count" id="count"></span>
  <button id="export">导出 CSV</button>
  <button id="submit" style="border-color:var(--accent);color:var(--accent)">提交给Claude</button>
</header>
<div class="card">
  <div class="meta" id="meta"></div>
  <img id="im" alt="sample">
  <p class="q" id="q"></p>
  <p class="a" id="a"></p>
  <div class="verdicts">
    <button class="v-ok" id="b-ok">对 (1)</button>
    <button class="v-bad" id="b-bad">错 (2)</button>
    <button class="v-un" id="b-un">存疑 (3)</button>
    <span style="flex:1"></span>
    <button id="prev">← 上一条</button>
    <button id="next">下一条 →</button>
  </div>
  <input class="note" id="note" placeholder="备注（可选，发现什么问题）">
  <p class="hint">快捷键：1 对 · 2 错 · 3 存疑（判完自动下一条）· ←/→ 翻页 · 进度自动保存在浏览器</p>
</div>
<script>
const ITEMS = __PAYLOAD__;
const TOKEN = '__TOK__';
const KEY = 'spotcheck_m1_v2';
let store = JSON.parse(localStorage.getItem(KEY) || '{}');
let filter = '';
let idx = 0;
const tpls = [...new Set(ITEMS.map(x=>x.t))].sort();
const sel = document.getElementById('tpl');
sel.innerHTML = '<option value="">全部模板</option>' + tpls.map(t=>`<option>${t}</option>`).join('');
sel.onchange = () => { filter = sel.value; idx = 0; render(); };
function list(){ return filter ? ITEMS.filter(x=>x.t===filter) : ITEMS; }
function render(){
  const L = list(); if(!L.length) return;
  idx = Math.max(0, Math.min(idx, L.length-1));
  const it = L[idx], v = store[it.qid] || {};
  document.getElementById('meta').textContent = `${idx+1}/${L.length} · ${it.t} · ${it.ep} · f${it.frame} · ${it.view} · ${it.qid}`;
  document.getElementById('im').src = 'spotcheck_imgs/' + TOKEN + '_' + it.qid + '.jpg';
  document.getElementById('q').textContent = it.q;
  document.getElementById('a').textContent = 'GT: ' + it.a;
  document.getElementById('note').value = v.note || '';
  for(const [id,val] of [['b-ok','ok'],['b-bad','bad'],['b-un','unsure']])
    document.getElementById(id).classList.toggle('active', v.verdict===val);
  const done = list().filter(x=>store[x.qid]&&store[x.qid].verdict).length;
  document.getElementById('prog').style.width = (100*done/L.length)+'%';
  const okN = L.filter(x=>store[x.qid]?.verdict==='ok').length,
        badN = L.filter(x=>store[x.qid]?.verdict==='bad').length,
        unN = L.filter(x=>store[x.qid]?.verdict==='unsure').length;
  document.getElementById('count').textContent = `已审 ${done} · 对 ${okN} 错 ${badN} 疑 ${unN}`;
}
function setV(val){
  const it = list()[idx];
  store[it.qid] = {...(store[it.qid]||{}), verdict:val, t:it.t};
  localStorage.setItem(KEY, JSON.stringify(store));
  if(idx < list().length-1){ idx++; } render();
}
document.getElementById('b-ok').onclick = ()=>setV('ok');
document.getElementById('b-bad').onclick = ()=>setV('bad');
document.getElementById('b-un').onclick = ()=>setV('unsure');
document.getElementById('prev').onclick = ()=>{ idx--; render(); };
document.getElementById('next').onclick = ()=>{ idx++; render(); };
document.getElementById('note').onchange = e=>{
  const it = list()[idx];
  store[it.qid] = {...(store[it.qid]||{}), note:e.target.value, t:it.t};
  localStorage.setItem(KEY, JSON.stringify(store));
};
document.addEventListener('keydown', e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT') return;
  if(e.key==='1') setV('ok'); else if(e.key==='2') setV('bad');
  else if(e.key==='3') setV('unsure');
  else if(e.key==='ArrowLeft'){ idx--; render(); }
  else if(e.key==='ArrowRight'){ idx++; render(); }
});
document.getElementById('submit').onclick = async ()=>{
  try{
    const r = await fetch('/submit',{method:'POST',headers:{'Content-Type':'application/json'},
      body: JSON.stringify({page:KEY, when:new Date().toISOString(), store})});
    document.getElementById('count').textContent += r.ok ? ' · 已提交✓' : ' · 提交失败';
  }catch(e){ document.getElementById('count').textContent += ' · 提交失败'; }
};
document.getElementById('export').onclick = ()=>{
  let csv = 'qid,template,verdict,note\\n';
  for(const it of ITEMS){ const v = store[it.qid]||{};
    csv += `${it.qid},${it.t},${v.verdict||''},"${(v.note||'').replace(/"/g,'""')}"\\n`; }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
  a.download = 'spotcheck_verdicts.csv'; a.click();
};
render();
</script>
"""
(OUT / "spotcheck.html").write_text(page.replace("__PAYLOAD__", payload).replace("__TOK__", TOK))
print(f"rendered {len(items)} overlays -> {IMG_DIR}")
print(f"reviewer -> {OUT/'spotcheck.html'}  (serve: localhost:8737/spotcheck.html)")
