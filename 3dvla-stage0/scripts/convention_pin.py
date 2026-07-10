#!/usr/bin/env python3
"""Pin the exact yml-quat -> OpenCV world->cam formula by comparing against the
DLT-recovered rotation. Tests quat order x USD/CV flip x transpose direction.
"""
import numpy as np, yaml, pickle, glob
import pyarrow.parquet as pq
import cv2

YML = "/workspace/tingting/GenManip/configs/cameras/fixed_camera_robotiq_s2r_3L_align_twoObs.yml"
USD2CV = np.diag([1.0, -1.0, -1.0])


def quat_to_R(q, order):
    if order == "wxyz":
        w, x, y, z = q
    else:
        x, y, z, w = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def dlt_R(view):
    fp = sorted(glob.glob("raw/chunk-000/episode_*.parquet"))[1]
    t = pq.read_table(fp, columns=["annotation.tcp_3d_trace", f"annotation.{view}.tcp_2d_trace"])
    P3, P2 = [], []
    for r in range(t.num_rows):
        b3 = t.column("annotation.tcp_3d_trace")[r].as_py()
        b2 = t.column(f"annotation.{view}.tcp_2d_trace")[r].as_py()
        if b3 is None or b2 is None:
            continue
        P3.append(np.asarray(pickle.loads(b3), float))
        P2.append(np.asarray(pickle.loads(b2), float))
    P3, P2 = np.concatenate(P3), np.concatenate(P2)
    n = len(P3)
    A = np.zeros((2 * n, 12))
    X = np.concatenate([P3, np.ones((n, 1))], 1)
    A[0::2, 0:4] = X
    A[0::2, 8:12] = -P2[:, 0:1] * X
    A[1::2, 4:8] = X
    A[1::2, 8:12] = -P2[:, 1:2] * X
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    K, R, t4, *_ = cv2.decomposeProjectionMatrix(P)
    # normalize: want fx,fy > 0 and proper rotation
    S = np.diag(np.sign(np.diag(K)))  # flips columns of K to positive
    K = K @ S
    R = S @ R
    if np.linalg.det(R) < 0:
        R = -R
    return K / K[2, 2], R


with open(YML) as f:
    y = yaml.safe_load(f)

for view, cam in (("base_view", "obs_camera"), ("base_view_2", "obs_camera_2")):
    Kd, Rd = dlt_R(view)
    q = np.array(y[cam]["orientation"], float)
    print(f"=== {view} ({cam})  true K: fx={Kd[0,0]:.2f} fy={Kd[1,1]:.2f} cx={Kd[0,2]:.2f} cy={Kd[1,2]:.2f}")
    best = None
    for qo in ("wxyz", "xyzw"):
        Rq = quat_to_R(q, qo)  # hypothesis: camera->world in USD axes
        for name, Rcand in (
            (f"{qo} | R(q)@USD2CV then transpose", (Rq @ USD2CV).T),
            (f"{qo} | R(q) transpose (no flip)", Rq.T),
            (f"{qo} | USD2CV@R(q).T", USD2CV @ Rq.T),
            (f"{qo} | R(q)@USD2CV no transpose", Rq @ USD2CV),
        ):
            d = np.abs(Rcand - Rd).max()
            if best is None or d < best[0]:
                best = (d, name, Rcand)
            print(f"  {name:<40} max|dR|={d:.6f}" + ("   <-- MATCH" if d < 1e-3 else ""))
    print()
