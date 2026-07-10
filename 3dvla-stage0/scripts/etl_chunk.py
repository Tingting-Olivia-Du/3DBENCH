#!/usr/bin/env python3
"""Production ETL for InternData-M1 franka subset — one chunk end-to-end.

Per chunk:
  1. ensure data/chunk-XXX.tar (download if missing), extract parquets
  2. per episode (parallel): parse parquet -> solve per-episode base (m1_calib)
     -> select <=KF keyframes -> write annotation slice (.pkl)
  3. ensure videos/chunk-XXX.tar, extract mp4s
  4. per episode (parallel): decode ONLY manifest frames from 3 views -> JPEG
  5. delete tars + extracted mp4s (unless --keep-raw)

Output layout (default /workspace/tingting/3dvla-data):
  ann/chunk-XXX/episode_XXXXXX.pkl
  frames/chunk-XXX/episode_XXXXXX/{base_view,base_view_2,ego_view}_f{NNNN}.jpg
  manifest/chunk-XXX.json     (per-episode: keyframes, base_xy, rms, task, counts)
"""
import argparse, json, os, pickle, shutil, sys, tarfile, time, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

REPO = "InternRobotics/InternData-M1"
VIEWS = ("base_view", "base_view_2", "ego_view")


# ---------------- keyframe selection ----------------

def select_keyframes(eef_pos, gripper, max_kf=8, min_gap=12):
    """State-change keyframes: episode ends, gripper events, EE-speed minima."""
    T = len(eef_pos)
    v = np.linalg.norm(np.diff(eef_pos, axis=0), axis=1)
    if len(v) >= 5:
        k = np.ones(5) / 5
        v = np.convolve(v, k, mode="same")
    cand = {0: 3.0, T - 1: 3.0}  # frame -> priority
    g = np.asarray(gripper, float)
    dg = np.abs(np.diff(g))
    thr = max(1e-4, dg.max() * 0.2)
    ev = np.where(dg > thr)[0]
    # compress runs of gripper motion into single events (start of run)
    last = -10**9
    for f in ev:
        if f - last > 5:
            cand[int(f)] = 4.0  # grasp/release: highest priority
        last = f
    # EE speed local minima (hover / contact)
    if T > 10:
        lo = np.percentile(v, 25)
        for f in range(2, T - 3):
            if v[f] <= lo and v[f] <= v[f - 1] and v[f] <= v[f + 1]:
                cand.setdefault(int(f), 1.0)
    # motion midpoints for coverage
    for f in range(T // 4, T, max(T // 4, 1)):
        cand.setdefault(int(f), 0.5)
    # greedy pick by priority with min gap
    chosen = []
    for f, _ in sorted(cand.items(), key=lambda kv: -kv[1]):
        if all(abs(f - c) >= min_gap for c in chosen):
            chosen.append(f)
        if len(chosen) >= max_kf:
            break
    return sorted(chosen)


# ---------------- per-episode annotation ----------------

_WC = None  # per-process WristChain


def _init_worker():
    global _WC
    from m1_calib import WristChain
    _WC = WristChain()


def process_episode_ann(args):
    fp, out_pkl, max_kf = args
    import pyarrow.parquet as pq
    global _WC
    try:
        t = pq.read_table(fp)
        T = t.num_rows
        col = lambda n: t.column(n)
        unpkl = lambda n, r: (lambda b: None if b is None else pickle.loads(b))(col(n)[r].as_py())

        joints = np.array([col("states.joint.position")[r].as_py() for r in range(T)])
        eef_pos = np.array([col("states.effector.position")[r].as_py() for r in range(T)])
        eef_orn = np.array([col("states.effector.orientation")[r].as_py() for r in range(T)])
        grip = np.array([col("actions.gripper.position")[r].as_py()[0] for r in range(T)])

        glyph = None
        J, UV = [], []
        for r in range(0, T, 6):
            b3, b2 = unpkl("annotation.tcp_3d_trace", r), unpkl("annotation.ego_view.tcp_2d_trace", r)
            if b3 is None or b2 is None:
                continue
            if glyph is None:
                glyph = np.asarray(b3, float)
            J.append(joints[r])
            UV.append(np.asarray(b2, float))
        base_xy, rms = _WC.solve_base(J, UV, glyph)

        kfs = select_keyframes(eef_pos, grip, max_kf=max_kf)
        per_kf = {}
        for f in kfs:
            d = {}
            for view in VIEWS:
                d[f"{view}.bbox2d_tight"] = unpkl(f"annotation.{view}.bbox2d_tight", f)
                d[f"{view}.bbox2d_tight_id2labels"] = unpkl(f"annotation.{view}.bbox2d_tight_id2labels", f)
            d["bbox3d"] = unpkl("annotation.bbox3d", f)
            d["bbox3d_id2labels"] = unpkl("annotation.bbox3d_id2labels", f)
            per_kf[f] = d

        rec = {
            "episode": Path(fp).stem,
            "length": T,
            "task_index": int(col("task_index")[0].as_py()),
            "pick_obj_uid": unpkl("annotation.pick_obj_uid", 0),
            "place_obj_uid": unpkl("annotation.place_obj_uid", 0),
            "diverse_instructions": unpkl("annotation.diverse_instructions", 0),
            "base_xy": np.asarray(base_xy, float),
            "base_solve_rms_px": float(rms),
            "glyph": glyph,
            "keyframes": kfs,
            "per_kf": per_kf,
            "joints": joints.astype(np.float32),
            "eef_pos_base": eef_pos.astype(np.float32),
            "eef_orn_base": eef_orn.astype(np.float32),
            "gripper_cmd": grip.astype(np.float32),
        }
        with open(out_pkl, "wb") as f:
            pickle.dump(rec, f, protocol=4)
        return (rec["episode"], kfs, float(rms), None)
    except Exception:
        return (Path(fp).stem, [], -1.0, traceback.format_exc(limit=3))


# ---------------- per-episode frame decode ----------------

def decode_episode(args):
    ep, mp4_paths, kfs, out_dir, jpeg_q = args
    import av, cv2
    try:
        os.makedirs(out_dir, exist_ok=True)
        want = set(kfs)
        n = 0
        for view, path in mp4_paths.items():
            c = av.open(path)
            for i, fr in enumerate(c.decode(video=0)):
                if i in want:
                    img = fr.to_ndarray(format="bgr24")
                    cv2.imwrite(f"{out_dir}/{view}_f{i:04d}.jpg", img,
                                [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
                    n += 1
                if i >= max(want):
                    break
            c.close()
        return (ep, n, None)
    except Exception:
        return (ep, 0, traceback.format_exc(limit=3))


# ---------------- chunk driver ----------------

def ensure_file(rel, raw_dir):
    p = Path(raw_dir) / rel
    if p.exists():
        return str(p)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, rel, repo_type="dataset", local_dir=raw_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--out", default="/workspace/tingting/3dvla-data")
    ap.add_argument("--raw", default="/workspace/tingting/3dvla-stage0/raw")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-kf", type=int, default=8)
    ap.add_argument("--jpeg-q", type=int, default=85)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N episodes")
    args = ap.parse_args()

    cid = f"chunk-{args.chunk:03d}"
    out = Path(args.out)
    ann_dir = out / "ann" / cid
    frm_dir = out / "frames" / cid
    man_path = out / "manifest" / f"{cid}.json"
    for d in (ann_dir, frm_dir, man_path.parent):
        d.mkdir(parents=True, exist_ok=True)
    if man_path.exists():
        print(f"[{cid}] manifest exists — already done, skipping"); return
    t0 = time.time()

    # --- annotations ---
    data_tar = ensure_file(f"simulated/franka/data/{cid}.tar", args.raw)
    scratch = Path(args.raw) / f"scratch_{cid}"
    scratch.mkdir(exist_ok=True)
    with tarfile.open(data_tar) as tf:
        tf.extractall(scratch)
    pq_dir = scratch / cid
    files = sorted(pq_dir.glob("episode_*.parquet"))
    if args.limit:
        files = files[: args.limit]
    print(f"[{cid}] {len(files)} episodes; solving base + keyframes ...")

    jobs = [(str(fp), str(ann_dir / f"{fp.stem}.pkl"), args.max_kf) for fp in files]
    manifest, errs = {}, []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        for ep, kfs, rms, err in ex.map(process_episode_ann, jobs, chunksize=8):
            if err:
                errs.append((ep, err)); continue
            manifest[ep] = {"keyframes": kfs, "rms_px": rms}
    bad_rms = [e for e, m in manifest.items() if m["rms_px"] > 0.5]
    print(f"[{cid}] ann done: {len(manifest)} ok, {len(errs)} failed, {len(bad_rms)} rms>0.5px "
          f"({time.time()-t0:.0f}s)")

    # --- frames ---
    vid_tar = ensure_file(f"simulated/franka/videos/{cid}.tar", args.raw)
    with tarfile.open(vid_tar) as tf:
        tf.extractall(scratch)
    vjobs = []
    for ep, m in manifest.items():
        mp4s = {v: str(scratch / cid / f"images.rgb.{v}" / f"{ep}.mp4") for v in VIEWS}
        if not all(os.path.exists(p) for p in mp4s.values()):
            errs.append((ep, "missing mp4")); continue
        vjobs.append((ep, mp4s, m["keyframes"], str(frm_dir / ep), args.jpeg_q))
    t1 = time.time()
    ok_frames = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for ep, n, err in ex.map(decode_episode, vjobs, chunksize=4):
            if err:
                errs.append((ep, err)); continue
            manifest[ep]["n_frames"] = n
            ok_frames += n
    print(f"[{cid}] frames done: {ok_frames} jpegs ({time.time()-t1:.0f}s)")

    # --- finalize ---
    with open(man_path, "w") as f:
        json.dump({"chunk": cid, "episodes": manifest,
                   "errors": [(e, s.splitlines()[-1] if s else s) for e, s in errs]}, f)
    if not args.keep_raw:
        shutil.rmtree(scratch, ignore_errors=True)
        for p in (data_tar, vid_tar):
            if str(args.raw) in str(p):
                os.remove(p)
    du = sum(f.stat().st_size for f in frm_dir.rglob("*.jpg")) / 1e9
    da = sum(f.stat().st_size for f in ann_dir.rglob("*.pkl")) / 1e9
    print(f"[{cid}] DONE in {time.time()-t0:.0f}s — frames {du:.2f} GB, ann {da:.2f} GB, "
          f"errors {len(errs)}")


if __name__ == "__main__":
    main()
