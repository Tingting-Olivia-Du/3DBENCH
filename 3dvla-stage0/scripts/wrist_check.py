#!/usr/bin/env python3
"""S3.5 wrist-chain check: per-frame wrist cam pose = T_world_base @ FK(q)->link8 @ mount,
world->cv via the pinned convention (quat wxyz, body x-forward frame, M_OPT permutation).
Validates FK params + mount transform; fits true wrist K given FK extrinsics.
"""
import glob, pickle
import numpy as np
import pyarrow.parquet as pq
import yaml, yourdfpy

YML = "/workspace/tingting/GenManip/configs/cameras/fixed_camera_robotiq_s2r_3L_align_twoObs.yml"
URDF = "/workspace/tingting/GenManip/assets/robots/panda/panda_v2.urdf"
BASE_POS = np.array([-0.41623, -0.00135, 0.99931])  # discussion #4, quat wxyz identity
M_OPT = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def quat_to_R_wxyz(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


with open(YML) as f:
    y = yaml.safe_load(f)
rs = y["realsense"]
T_mount = np.eye(4)
T_mount[:3, :3] = quat_to_R_wxyz(np.array(rs["orientation"], float))
T_mount[:3, 3] = np.array(rs["position"], float)
K_CANDS = {
    "yml_camera_params": np.array(rs["camera_params"][:4], float),
    "fixedcam_style_606.74_center": np.array([606.74, 606.74, 320.0, 240.0]),
    "aperture_288": np.array([288.0, 288.0, 320.0, 240.0]),
}

urdf = yourdfpy.URDF.load(URDF, load_meshes=False)
T_world_base = np.eye(4)
T_world_base[:3, 3] = BASE_POS


def wrist_pose(joints):
    cfg = {f"panda_joint{i+1}": joints[i] for i in range(7)}
    urdf.update_cfg(cfg)
    T_b_l8 = urdf.get_transform("panda_link8", "panda_link0")
    return T_world_base @ T_b_l8 @ T_mount


def collect(n_eps=5, stride=5):
    files = sorted(glob.glob("raw/chunk-000/episode_*.parquet"))[:n_eps]
    samples = []  # (p_cam_cv (N,3), p2 (N,2))
    for fp in files:
        t = pq.read_table(fp, columns=[
            "states.joint.position", "annotation.tcp_3d_trace", "annotation.ego_view.tcp_2d_trace"])
        for r in range(0, t.num_rows, stride):
            b3 = t.column("annotation.tcp_3d_trace")[r].as_py()
            b2 = t.column("annotation.ego_view.tcp_2d_trace")[r].as_py()
            j = t.column("states.joint.position")[r].as_py()
            if b3 is None or b2 is None or j is None:
                continue
            p3 = np.asarray(pickle.loads(b3), float)
            p2 = np.asarray(pickle.loads(b2), float)
            T_w_cam = wrist_pose(np.asarray(j, float))
            R_w2cv = M_OPT @ T_w_cam[:3, :3].T
            p_cv = (p3 - T_w_cam[:3, 3]) @ R_w2cv.T
            keep = p_cv[:, 2] > 1e-6
            if keep.sum() < 3:
                continue
            samples.append((p_cv[keep], p2[keep]))
    return samples


def eval_K(samples, K):
    fx, fy, cx, cy = K
    errs = []
    for p_cv, p2 in samples:
        u = fx * p_cv[:, 0] / p_cv[:, 2] + cx
        v = fy * p_cv[:, 1] / p_cv[:, 2] + cy
        errs.append(np.linalg.norm(np.stack([u, v], 1) - p2, axis=1))
    e = np.concatenate(errs)
    return e.mean(), np.median(e), np.percentile(e, 90)


def fit_K(samples):
    # linear LS for fx, fy, cx, cy given normalized coords
    rows, rhs = [], []
    for p_cv, p2 in samples:
        xn = p_cv[:, 0] / p_cv[:, 2]
        yn = p_cv[:, 1] / p_cv[:, 2]
        for i in range(len(xn)):
            rows.append([xn[i], 0, 1, 0]); rhs.append(p2[i, 0])
            rows.append([0, yn[i], 0, 1]); rhs.append(p2[i, 1])
    A = np.array(rows); b = np.array(rhs)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol  # fx, fy, cx, cy


samples = collect()
n = sum(len(s[0]) for s in samples)
print(f"samples: {len(samples)} frames, {n} points")
for name, K in K_CANDS.items():
    m, md, p90 = eval_K(samples, K)
    print(f"  K={name:<32} mean={m:9.3f}px median={md:9.3f}px p90={p90:9.3f}px")
Kfit = fit_K(samples)
m, md, p90 = eval_K(samples, Kfit)
print(f"  K=FITTED fx={Kfit[0]:.2f} fy={Kfit[1]:.2f} cx={Kfit[2]:.2f} cy={Kfit[3]:.2f}"
      f"  mean={m:9.3f}px median={md:9.3f}px p90={p90:9.3f}px")
