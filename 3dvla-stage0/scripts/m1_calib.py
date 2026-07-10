#!/usr/bin/env python3
"""InternData-M1 franka-subset geometry module — conventions pinned by DLT + joint fit
(2026-07-07 Stage-0 verification; see out/calibration_report.md).

Pinned facts:
- yml quats are wxyz; orientation gives the camera BODY frame (x-forward/y-left/z-up)
  pose; OpenCV optical axes = M_OPT @ body^T.
- Fixed cams: true K = (606.74, 606.74, 320, 240); positions exactly as yml;
  constant across episodes. base_view<->obs_camera, base_view_2<->obs_camera_2.
- Wrist (ego_view): mount = yml realsense values EXACTLY (link8 -> camera body);
  true K = (609.36, 609.36, 320, 240).
- Robot base: z=1.011, yaw=0 fixed; x,y RANDOMIZED PER EPISODE (+-4cm around
  [-0.44, 0.01]). Solve per episode with solve_base() using the static glyph.
- annotation.tcp_3d_trace = STATIC 6-point gripper icon at world
  [-0.4577,-0.0014,1.011] (identical across dataset) - a calibration target,
  NOT the TCP. Its per-view 2d traces are exact projections (0.000 px).
"""
import numpy as np
import cv2
import yourdfpy

M_OPT = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])

# DUAL INTRINSICS (discovered 2026-07-07 during QA review): the dataset has TWO
# self-consistent K per camera, sharing extrinsics/world:
#  - K_ANNOT_*: idealized pinhole used to GENERATE glyph/tcp_2d_trace annotations
#    (recovered by DLT at 0.000 px). Use ONLY for glyph-based solving (solve_base).
#  - K_PIXEL_*: the yml `camera_params` — what the RGB was RENDERED with (and what
#    bbox2d/bbox3d annotators saw). Use for EVERYTHING that touches pixels.
#    Evidence: projecting bbox3d centers with K_PIXEL matches tight-box centers to
#    median (+0.5,+0.2) px; with K_ANNOT the v-offset is exactly cy_diff = -20.8 px.
K_ANNOT_FIXED = np.array([606.74, 606.74, 320.0, 240.0])   # obs_camera & obs_camera_2
K_ANNOT_WRIST = np.array([609.36, 609.36, 320.0, 240.0])   # realsense (ego_view)
K_PIXEL_FIXED = np.array([606.99, 606.49, 321.73, 260.76])
K_PIXEL_WRIST = np.array([609.79, 608.92, 323.59, 236.77])
K_FIXED = K_ANNOT_FIXED  # legacy aliases (annotation-space)
K_WRIST = K_ANNOT_WRIST
GLYPH_ANCHOR = np.array([-0.4577, -0.0014, 1.011])
BASE_Z, BASE_YAW = 1.011, 0.0

OBS_CAMERA = {  # from fixed_camera_robotiq_s2r_3L_align_twoObs.yml (positions verified exact)
    "base_view":   {"pos": np.array([-0.54772, -0.79264, 1.64032]),
                    "quat_wxyz": np.array([0.827804, -0.118445, 0.240595, 0.492774])},
    "base_view_2": {"pos": np.array([0.65, 0.0, 1.85]),
                    "quat_wxyz": np.array([0.0, -0.42261829, 0.0, 0.90630777])},
}
MOUNT_POS = np.array([0.07405, -0.032, 0.02149])          # link8 -> camera body (verified exact)
MOUNT_QUAT_WXYZ = np.array([0.0, 0.572857, 0.0, 0.819655])

URDF_PATH = "/workspace/tingting/GenManip/assets/robots/panda/panda_v2.urdf"


def quat_to_R_wxyz(q):
    w, x, y, z = q
    s = 2.0 / (q @ q)
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def world_to_cam_fixed(view):
    """Return (R_world->cv, camera_center) for a fixed view."""
    c = OBS_CAMERA[view]
    return M_OPT @ quat_to_R_wxyz(c["quat_wxyz"]).T, c["pos"]


def project(points_w, R_w2cv, C, K):
    pc = (np.atleast_2d(points_w) - C) @ R_w2cv.T
    z = pc[:, 2]
    return np.stack([K[0] * pc[:, 0] / z + K[2], K[1] * pc[:, 1] / z + K[3]], 1), z > 1e-6


class WristChain:
    def __init__(self, urdf_path=URDF_PATH):
        self.urdf = yourdfpy.URDF.load(urdf_path, load_meshes=False)
        self.T_mount = np.eye(4)
        self.T_mount[:3, :3] = quat_to_R_wxyz(MOUNT_QUAT_WXYZ)
        self.T_mount[:3, 3] = MOUNT_POS

    def fk_link8(self, joints7):
        self.urdf.update_cfg({f"panda_joint{i+1}": joints7[i] for i in range(7)})
        return self.urdf.get_transform("panda_link8", "panda_link0").copy()

    def cam_pose_world(self, joints7, base_xy):
        """T_world<-camera_body given per-episode base (x, y)."""
        T_wb = np.eye(4)
        T_wb[:3, 3] = [base_xy[0], base_xy[1], BASE_Z]
        return T_wb @ self.fk_link8(joints7) @ self.T_mount

    def solve_base(self, joints_list, ego_uv_list, glyph6, uv_bound=(-1280, 1920)):
        """Recover the episode's base (x, y) from the static glyph seen by the
        moving wrist cam. joints_list: (T,7); ego_uv_list: (T,6,2).

        Frames whose annotated uv falls outside uv_bound are dropped: when the
        glyph grazes the camera plane the projection explodes (|uv| ~ 1e5,
        seen in episode 10) and poisons the least squares. Multi-start guards
        against the remaining local minima; returns the best (xy, rms_px).
        """
        from scipy.optimize import least_squares
        keep_j, keep_uv = [], []
        for j, uv in zip(joints_list, ego_uv_list):
            if (uv >= uv_bound[0]).all() and (uv <= uv_bound[1]).all():
                keep_j.append(j)
                keep_uv.append(uv)

        def res(xy):
            out = []
            for j, uv in zip(keep_j, keep_uv):
                T = self.cam_pose_world(j, xy)
                R_cv = M_OPT @ T[:3, :3].T
                p, _ = project(glyph6, R_cv, T[:3, 3], K_WRIST)
                out.append((p - uv).ravel())
            return np.concatenate(out)

        best = None
        for x0 in ([-0.44, 0.01], [-0.48, -0.03], [-0.42, 0.04], [-0.50, 0.03]):
            r = least_squares(res, x0=x0, method="trf", loss="soft_l1", f_scale=5.0)
            rms = float(np.sqrt(np.mean(r.fun ** 2)))
            if best is None or rms < best[1]:
                best = (np.array(r.x), rms)
            if rms < 0.1:
                break
        return best
