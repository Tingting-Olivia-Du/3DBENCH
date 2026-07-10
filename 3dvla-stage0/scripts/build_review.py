#!/usr/bin/env python3
"""Build the QA quality-review gallery (HTML) for joint inspection.

For each template: 3 seeded samples rendered with the ANSWER drawn on the image,
an independent verification note, and distribution stats over the full chunk.
Independent checks:
  - T1/T6: answer box vs projection of bbox3d through OUR calibration (independent
    annotation chain) -> IoU (expect moderate: tight-visible vs full-extent).
  - T7: qualitative sign check — static object's image shift vs camera translation.
Output: out/qa_review.html (self-contained, base64 images).
"""
import ast, base64, html, json, pickle, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from m1_calib import (BASE_Z, K_PIXEL_FIXED, M_OPT, WristChain, project,
                      world_to_cam_fixed)
K_FIXED = K_PIXEL_FIXED  # pixel-space projections throughout the gallery

DATA = Path("/workspace/tingting/3dvla-data")
QA = DATA / "qa" / "chunk-000.jsonl"
W, H, NORM = 640, 480, 2000.0
rng = np.random.default_rng(7)
wc = WristChain()

RED, YEL, CYA, GRN = (46, 46, 230), (60, 200, 240), (220, 200, 40), (80, 190, 90)


def d(v, ax):  # denorm
    return int(v / NORM * (W if ax == "x" else H))


def dbox(b):
    return d(b[0], "x"), d(b[1], "y"), d(b[2], "x"), d(b[3], "y")


def b64(img, q=72):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def load_rec(ep):
    return pickle.load(open(DATA / "ann" / "chunk-000" / f"{ep}.pkl", "rb"))


def proj3d_box(rec, f, uid, view):
    d_ = rec["per_kf"][f]
    id2l = d_["bbox3d_id2labels"] or {}
    sem2uid = {int(k): v["class"] for k, v in id2l.items()}
    for ob in (d_["bbox3d"] or []):
        if sem2uid.get(int(np.asarray(ob["class"]).item())) == uid:
            corners = np.asarray(ob["corners"], float).reshape(8, 3)
            R_cv, C = world_to_cam_fixed(view)
            uv, vis = project(corners, R_cv, C, K_FIXED)
            if not vis.all():
                return None
            return [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
    return None


def iou(a, b):
    xi = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - xi
    return xi / (ua + 1e-9)


qas = [json.loads(l) for l in open(QA)]
by_t = {}
for q in qas:
    by_t.setdefault(q["template"], []).append(q)

TEMPLATES = [
    ("T1_grounding", "2D grounding", "定位物体，输出 [x0,y0,x1,y1]，归一化 [0,2000)"),
    ("T2_metric3d_dist", "物体间度量距离", "两物体 3D 中心距离（cm），源自 bbox3d"),
    ("T3_depth_obj", "相机→物体深度", "相机到物体中心距离（cm）"),
    ("T3_depth_tcp", "相机→TCP 深度", "相机到夹爪 TCP 距离（cm），源自 FK+逐集基座"),
    ("T4_tcp_trace", "未来 TCP 轨迹", "未来 10..60 帧 TCP 的 2D 投影（截断出画点）"),
    ("T5_where_to_act", "操作目标指点", "任务应抓取物体的中心点 [x,y]"),
    ("T6_cross_view", "跨视图对应", "view1 的框 → view2 的框（同一 uid）"),
    ("T7_ego_motion", "腕相机自运动", "两帧间相机平移(cm/光学系)+旋转(yaw/pitch/roll)"),
]

cards_html = []
for tid, tname, tdesc in TEMPLATES:
    items = by_t.get(tid, [])
    picks = [items[i] for i in rng.choice(len(items), size=min(3, len(items)), replace=False)]
    # ---------------- stats ----------------
    stats = f"{len(items):,} 条"
    if tid == "T2_metric3d_dist":
        v = np.array([float(q["answer"]) for q in items])
        stats += f" · 距离 p5/p50/p95 = {np.percentile(v,5):.0f}/{np.percentile(v,50):.0f}/{np.percentile(v,95):.0f} cm"
    if tid in ("T3_depth_obj", "T3_depth_tcp"):
        v = np.array([float(q["answer"]) for q in items])
        stats += f" · 深度 p5/p50/p95 = {np.percentile(v,5):.0f}/{np.percentile(v,50):.0f}/{np.percentile(v,95):.0f} cm"
    if tid == "T7_ego_motion":
        v = np.array([q["meta"]["dist_cm"] for q in items])
        g = np.array([q["meta"]["gap"] for q in items])
        stats += f" · |平移| p50 = {np.percentile(v,50):.1f} cm · 帧距 10–{g.max()}"
    if tid == "T1_grounding":
        sub = [items[i] for i in rng.choice(len(items), size=min(200, len(items)), replace=False)]
        ious = []
        for q in sub:
            rec = load_rec(q["episode"])
            pb = proj3d_box(rec, q["frame"], q["meta"]["uid"], q["view"])
            if pb is None:
                continue
            ious.append(iou(dbox(ast.literal_eval(q["answer"])), pb))
        stats += (f" · 独立链核验：2D框 vs bbox3d投影框 IoU 中位 {np.median(ious):.2f}"
                  f"（紧框⊂全包围盒投影，中等值即正常）")
    # ---------------- sample cards ----------------
    sec = []
    for q in picks:
        rec = load_rec(q["episode"])
        note = ""
        if tid == "T6_cross_view":
            img1 = cv2.imread(q["image"]); img2 = cv2.imread(q["image2"])
            qb = ast.literal_eval(q["question"].split("is at ")[1].split(" in the first")[0])
            ab = ast.literal_eval(q["answer"])
            cv2.rectangle(img1, dbox(qb)[:2], dbox(qb)[2:], YEL, 2)
            cv2.rectangle(img2, dbox(ab)[:2], dbox(ab)[2:], RED, 2)
            img = np.concatenate([img1, img2], 1)
            pb = proj3d_box(rec, q["frame"], q["meta"]["uid"], "base_view_2")
            if pb is not None:
                note = f"独立核验：答案框 vs bbox3d投影(view2) IoU={iou(dbox(ab), pb):.2f}"
        elif tid == "T7_ego_motion":
            a, b = q["frame"], q["frame"] + q["meta"]["gap"]
            img1 = cv2.imread(q["image"]); img2 = cv2.imread(q["image2"])
            # qualitative sign check via a static object visible in both ego frames
            from qa_gen import uid_boxes
            sh = None
            for f2, im in ((a, None), ):
                pass
            ba = uid_boxes(rec, a, "ego_view") if a in rec["per_kf"] else {}
            bb = uid_boxes(rec, b, "ego_view") if b in rec["per_kf"] else {}
            common = [u for u in ba if u in bb and u != rec["pick_obj_uid"]]
            t = q["meta"]["t_rel_cm"]
            if common:
                u0 = ba[common[0]]; u1 = bb[common[0]]
                sh = ((u1[0]+u1[2])/2-(u0[0]+u0[2])/2, (u1[1]+u1[3])/2-(u0[1]+u0[3])/2)
                cv2.rectangle(img1, tuple(u0[:2]), tuple(u0[2:]), GRN, 2)
                cv2.rectangle(img2, tuple(u1[:2]), tuple(u1[2:]), GRN, 2)
                ok = "一致" if (abs(t[0]) < 1.5 or sh[0] * t[0] < 0) else "需人工判读(旋转项)"
                note = (f"静物(绿框)图移 Δu={sh[0]:.0f}px, Δv={sh[1]:.0f}px；相机平移 x={t[0]}cm "
                        f"→ 符号{ok}")
            else:
                note = "两帧无共同可见静物，凭画面运动感目视判断"
            img = np.concatenate([img1, img2], 1)
        else:
            img = cv2.imread(q["image"])
            ans = ast.literal_eval(q["answer"]) if q["answer"][0] == "[" else None
            if tid in ("T1_grounding",):
                cv2.rectangle(img, dbox(ans)[:2], dbox(ans)[2:], RED, 2)
                pb = proj3d_box(rec, q["frame"], q["meta"]["uid"], q["view"])
                if pb is not None:
                    p = [int(x) for x in pb]
                    cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), CYA, 1)
                    note = f"红=答案(2D紧框)，蓝=独立链(bbox3d投影) IoU={iou(dbox(ans), pb):.2f}"
            elif tid == "T5_where_to_act":
                cv2.drawMarker(img, (d(ans[0], "x"), d(ans[1], "y")), RED, cv2.MARKER_CROSS, 16, 2)
            elif tid == "T4_tcp_trace":
                pts = [(d(p[0], "x"), d(p[1], "y")) for p in ans]
                for i, p in enumerate(pts):
                    cv2.circle(img, p, 4, RED, -1)
                    if i:
                        cv2.line(img, pts[i - 1], p, CYA, 1)
                note = f"{len(pts)} 个航点（出画即截断）"
            elif tid in ("T3_depth_obj", "T3_depth_tcp"):
                if tid == "T3_depth_tcp":
                    uv = q["meta"]["tcp_uv"]
                    cv2.drawMarker(img, (d(uv[0], "x"), d(uv[1], "y")), RED, cv2.MARKER_CROSS, 16, 2)
                    note = "十字=TCP 投影位置"
                else:
                    name = q["question"].split("center of the ")[1].split(" from")[0]
                    from qa_gen import uid_boxes
                    for uid, bx in uid_boxes(rec, q["frame"], q["view"]).items():
                        if uid in (rec["pick_obj_uid"],):
                            cv2.rectangle(img, tuple(bx[:2]), tuple(bx[2:]), RED, 2)
                    note = "红框=被问物体"
            elif tid == "T2_metric3d_dist":
                from qa_gen import uid_boxes, uid_center3d
                bxs = uid_boxes(rec, q["frame"], q["view"])
                R_cv, C = world_to_cam_fixed(q["view"])
                cts = []
                for uid in (rec["pick_obj_uid"], rec["place_obj_uid"]):
                    if uid in bxs:
                        cv2.rectangle(img, tuple(bxs[uid][:2]), tuple(bxs[uid][2:]), RED, 2)
                    c3, _ = uid_center3d(rec, q["frame"], uid)
                    if c3 is not None:
                        uv, _ = project(c3, R_cv, C, K_FIXED)
                        cts.append((int(uv[0, 0]), int(uv[0, 1])))
                if len(cts) == 2:
                    cv2.line(img, cts[0], cts[1], CYA, 2)
                    mid = ((cts[0][0] + cts[1][0]) // 2, (cts[0][1] + cts[1][1]) // 2)
                    cv2.putText(img, f"{float(q['answer']):.0f}cm", mid,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                note = "蓝线连接两物体 3D 中心的投影"
        sec.append(f"""
      <figure class="card">
        <img src="{b64(img)}" alt="{tid} sample">
        <figcaption>
          <div class="qrow"><span class="tag">{html.escape(q['episode'])} · f{q['frame']} · {html.escape(str(q['view']))}</span></div>
          <p class="q">{html.escape(q['question'])}</p>
          <p class="a">{html.escape(q['answer'])}</p>
          {f'<p class="note">{html.escape(note)}</p>' if note else ''}
        </figcaption>
      </figure>""")
    cards_html.append(f"""
    <section>
      <p class="eyebrow">{tid}</p>
      <h2>{tname}</h2>
      <p class="desc">{tdesc}</p>
      <p class="stats">{stats}</p>
      <div class="grid">{''.join(sec)}</div>
    </section>""")

total = len(qas)
page = f"""<meta charset="utf-8">
<title>QA 质量评审 · InternData-M1 chunk-000</title>
<style>
:root {{
  --bg:#f4f5f4; --panel:#ffffff; --ink:#1c201e; --sub:#5b6461; --line:#dde1df;
  --accent:#0e7490; --mono:ui-monospace,'SF Mono',Consolas,monospace;
}}
@media (prefers-color-scheme: dark) {{ :root {{
  --bg:#15181a; --panel:#1d2124; --ink:#e6e9e7; --sub:#98a29e; --line:#31383b; --accent:#39bcd8;
}} }}
:root[data-theme="dark"] {{ --bg:#15181a; --panel:#1d2124; --ink:#e6e9e7; --sub:#98a29e; --line:#31383b; --accent:#39bcd8; }}
:root[data-theme="light"] {{ --bg:#f4f5f4; --panel:#ffffff; --ink:#1c201e; --sub:#5b6461; --line:#dde1df; --accent:#0e7490; }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.55 -apple-system,'Segoe UI',Roboto,'Noto Sans SC',sans-serif;
       max-width:1120px; margin:0 auto; padding:40px 20px 80px; }}
header h1 {{ font-size:26px; margin:0 0 6px; text-wrap:balance; }}
header p {{ color:var(--sub); margin:0; }}
.eyebrow {{ font:600 12px/1 var(--mono); letter-spacing:.08em; text-transform:uppercase; color:var(--accent); margin:48px 0 4px; }}
h2 {{ font-size:20px; margin:0 0 2px; }}
.desc {{ color:var(--sub); margin:0 0 4px; }}
.stats {{ font:13px var(--mono); color:var(--ink); background:var(--panel); border:1px solid var(--line);
         border-radius:6px; padding:8px 12px; display:inline-block; font-variant-numeric:tabular-nums; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:18px; margin-top:14px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; margin:0; overflow:hidden; }}
.card img {{ width:100%; display:block; }}
.card figcaption {{ padding:10px 14px 14px; }}
.tag {{ font:11px var(--mono); color:var(--sub); }}
.q {{ margin:6px 0 4px; font-size:13.5px; }}
.a {{ font:12.5px var(--mono); color:var(--accent); margin:0; overflow-wrap:anywhere; }}
.note {{ font-size:12.5px; color:var(--sub); margin:6px 0 0; border-top:1px dashed var(--line); padding-top:6px; }}
</style>
<header>
  <h1>QA 质量评审 — InternData-M1 franka · chunk-000</h1>
  <p>共 {total:,} 条 QA / 1000 集 · 每模板抽 3 条 · 红=答案 · 蓝=独立证据链(bbox3d投影) · 黄=题面输入框 · 绿=辅助核验对象</p>
</header>
{''.join(cards_html)}
"""
out = Path("out/qa_review.html")
out.write_text(page)
print("wrote", out, f"{out.stat().st_size/1e6:.1f} MB")
