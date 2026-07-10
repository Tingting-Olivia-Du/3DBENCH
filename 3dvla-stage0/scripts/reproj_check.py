#!/usr/bin/env python3
"""S3.5 convention check: grid-search {quat order} x {camera convention} x {tcp_3d frame}
by reprojecting annotation.tcp_3d_trace against per-view annotation.*.tcp_2d_trace.

Fixed views only (obs_camera / obs_camera_2). Wrist chain is a separate script.
"""
import argparse, glob, json, pickle
import numpy as np
import pyarrow.parquet as pq
import yaml

YML = "/workspace/tingting/GenManip/configs/cameras/fixed_camera_robotiq_s2r_3L_align_twoObs.yml"
# Franka base world pose per HF discussion #4 (Axi404, 2025-11-13), quat scalar-first
BASE_POS = np.array([-0.41623, -0.00135, 0.99931])
BASE_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])

VIEW2CAM = {"base_view": "obs_camera", "base_view_2": "obs_camera_2"}


def quat_to_R(q, order):
    if order == "wxyz":
        w, x, y, z = q
    else:  # xyzw
        x, y, z, w = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


# USD camera looks down -Z with +Y up; OpenCV looks down +Z with +Y down.
USD2CV = np.diag([1.0, -1.0, -1.0])


def project(points_w, cam_pos, cam_R_wc, K, usd):
    """points_w: (N,3) world; cam pose = position + rotation (camera->world)."""
    p_cam = (points_w - cam_pos) @ cam_R_wc  # world -> camera axes (R^T @ v)
    if usd:
        p_cam = p_cam @ USD2CV.T
    fx, fy, cx, cy = K
    z = p_cam[:, 2]
    valid = z > 1e-6
    u = fx * p_cam[:, 0] / z + cx
    v = fy * p_cam[:, 1] / z + cy
    return np.stack([u, v], 1), valid


def load_cams():
    with open(YML) as f:
        y = yaml.safe_load(f)
    cams = {}
    for name in ("obs_camera", "obs_camera_2"):
        c = y[name]
        cams[name] = {
            "pos": np.array(c["position"], dtype=float),
            "quat": np.array(c["orientation"], dtype=float),
            "K": np.array(c["camera_params"][:4], dtype=float),
        }
    return cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk_dir", default="raw/chunk-000")
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    cams = load_cams()
    files = sorted(glob.glob(f"{args.chunk_dir}/episode_*.parquet"))[: args.n_episodes]

    R_base = quat_to_R(BASE_QUAT_WXYZ, "wxyz")

    # accumulate per (view, quat_order, cam_conv, frame_hyp)
    errs = {}
    n_pts = 0
    for fp in files:
        t = pq.read_table(fp, columns=[
            "annotation.tcp_3d_trace",
            "annotation.base_view.tcp_2d_trace",
            "annotation.base_view_2.tcp_2d_trace",
        ])
        for r in range(0, t.num_rows, args.stride):
            b3 = t.column("annotation.tcp_3d_trace")[r].as_py()
            if b3 is None:
                continue
            p3 = np.asarray(pickle.loads(b3), dtype=float)  # (6,3)
            hyp_points = {
                "world": p3,
                "base": p3 @ R_base.T + BASE_POS,  # base-frame -> world
            }
            for view, camname in VIEW2CAM.items():
                b2 = t.column(f"annotation.{view}.tcp_2d_trace")[r].as_py()
                if b2 is None:
                    continue
                p2 = np.asarray(pickle.loads(b2), dtype=float)  # (6,2)
                cam = cams[camname]
                for qo in ("wxyz", "xyzw"):
                    R = quat_to_R(cam["quat"], qo)
                    for conv in ("usd", "opencv"):
                        for fh, pw in hyp_points.items():
                            uv, valid = project(pw, cam["pos"], R, cam["K"], usd=(conv == "usd"))
                            if valid.sum() == 0:
                                continue
                            e = np.linalg.norm(uv[valid] - p2[valid], axis=1)
                            errs.setdefault((view, qo, conv, fh), []).append(e)
            n_pts += 1

    print(f"# frames sampled: {n_pts}\n")
    print(f"{'view':<12} {'quat':<5} {'conv':<7} {'frame':<6} {'mean_px':>9} {'median_px':>10} {'p90_px':>8}")
    rows = []
    for k, v in errs.items():
        e = np.concatenate(v)
        rows.append((k, e.mean(), np.median(e), np.percentile(e, 90)))
    rows.sort(key=lambda x: x[1])
    for (view, qo, conv, fh), m, md, p90 in rows:
        print(f"{view:<12} {qo:<5} {conv:<7} {fh:<6} {m:9.2f} {md:10.2f} {p90:8.2f}")

    best = {}
    for (view, qo, conv, fh), m, md, p90 in rows:
        if view not in best:
            best[view] = ((qo, conv, fh), m, md, p90)
    print("\nBEST per view:")
    for view, (combo, m, md, p90) in best.items():
        print(f"  {view}: {combo}  mean={m:.3f}px median={md:.3f}px p90={p90:.3f}px")
    with open("out/reproj_fixed_views.json", "w") as f:
        json.dump({f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": {
            "mean": float(np.concatenate(v).mean()),
            "median": float(np.median(np.concatenate(v))),
            "p90": float(np.percentile(np.concatenate(v), 90)),
            "n": int(sum(len(x) for x in v)),
        } for k, v in errs.items()}, f, indent=1)


if __name__ == "__main__":
    main()
