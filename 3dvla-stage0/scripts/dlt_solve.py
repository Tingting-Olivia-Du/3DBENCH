#!/usr/bin/env python3
"""Solve the projection matrix P = K[R|t] per view per episode by DLT from
tcp_3d_trace <-> tcp_2d_trace correspondences, then decompose to reveal the
true K, camera pose, and the reference frame of tcp_3d_trace.
"""
import glob, pickle, sys
import numpy as np
import pyarrow.parquet as pq
import cv2


def collect(fp, view):
    t = pq.read_table(fp, columns=["annotation.tcp_3d_trace", f"annotation.{view}.tcp_2d_trace"])
    P3, P2 = [], []
    for r in range(t.num_rows):
        b3 = t.column("annotation.tcp_3d_trace")[r].as_py()
        b2 = t.column(f"annotation.{view}.tcp_2d_trace")[r].as_py()
        if b3 is None or b2 is None:
            continue
        p3 = np.asarray(pickle.loads(b3), float)
        p2 = np.asarray(pickle.loads(b2), float)
        if p3.shape[0] != p2.shape[0]:
            continue
        P3.append(p3)
        P2.append(p2)
    return np.concatenate(P3), np.concatenate(P2)


def dlt(P3, P2):
    n = len(P3)
    A = np.zeros((2 * n, 12))
    X = np.concatenate([P3, np.ones((n, 1))], 1)
    A[0::2, 0:4] = X
    A[0::2, 8:12] = -P2[:, 0:1] * X
    A[1::2, 4:8] = X
    A[1::2, 8:12] = -P2[:, 1:2] * X
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    # reprojection error
    uvw = X @ P.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    err = np.linalg.norm(uv - P2, axis=1)
    return P, err


def decompose(P):
    K, R, t, *_ = cv2.decomposeProjectionMatrix(P)
    K = K / K[2, 2]
    C = (t[:3] / t[3]).ravel()  # camera center in the 3D-point frame
    return K, R, C


def main():
    files = sorted(glob.glob("raw/chunk-000/episode_*.parquet"))
    eps = [files[i] for i in (0, 1, 2, 50, 100)]
    for view in ("base_view", "base_view_2", "ego_view"):
        print(f"=== {view}")
        for fp in eps:
            P3, P2 = collect(fp, view)
            P, err = dlt(P3, P2)
            K, R, C = decompose(P)
            flag = "FIXED-CAM-OK" if err.mean() < 2 else ""
            print(f"  {fp.split('/')[-1]}: n={len(P3)} reproj mean={err.mean():8.3f}px p90={np.percentile(err,90):8.3f}px {flag}")
            if err.mean() < 50:
                print(f"    K: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
                print(f"    cam center C (in tcp3d frame): {np.round(C,4)}")
                print(f"    R (world->cam rows):\n{np.round(R,4)}")


if __name__ == "__main__":
    main()
