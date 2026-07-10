#!/usr/bin/env python3
"""Focal-length unification (plan §3.1, VLM³ recipe): resize an image so its
focal length becomes F_TARGET px (600), then reach the training resolution by
CROP/PAD ONLY (never a second resize — that would silently re-break the
convention; user-signed-off design decision).

Key property: [0,2000)-normalized coordinates are invariant under the uniform
resize but NOT under crop/pad — so this module also remaps normalized coords
into the final canvas. Metric answers (cm) are untouched.

DEFAULT MODE (canvas=None): pure uniform resize to f=1000 — Qwen3-VL takes
native dynamic resolutions, so no fixed canvas is needed and [0,2000)-normalized
coordinates are EXACTLY invariant (no text remapping at load time). The crop/pad
canvas mode remains available for fixed-resolution backbones only.

Sources without K (general/Vlaser 2D QA) skip unification entirely (their
supervision is resize-invariant 2D; documented scope).
"""
import numpy as np
import cv2

F_TARGET = 600.0  # near source-median f (312..610); any CONSTANT works — 1000 would upscale wrist x3.2 and collide with processor max_pixels (silent re-resize = convention break)
CANVAS = (1056, 800)  # W, H — fits 640x480@f~607 scaled ~1.65x with small pad


def unify(img, K, canvas=None, pad_value=127):
    """img: HxW BGR; K: (fx, fy, cx, cy) in pixel units of img.
    Returns (canvas_img, info) — info carries the affine for coordinate remap."""
    H, W = img.shape[:2]
    fx = float(K[0])
    s = F_TARGET / fx  # uniform scale (fx≈fy for all our sources; assert loosely)
    assert abs(float(K[1]) / fx - 1.0) < 0.02, "anisotropic focal not supported"
    newW, newH = int(round(W * s)), int(round(H * s))
    resized = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_CUBIC)
    if canvas is None:  # dynamic-resolution backbone: done, coords invariant
        info = {"scale": s, "src_wh": (W, H), "new_wh": (newW, newH),
                "canvas_wh": (newW, newH), "sx0": 0, "sy0": 0, "dx0": 0, "dy0": 0}
        return resized, info
    CW, CH = canvas
    # center-crop or pad each axis independently
    out = np.full((CH, CW, 3), pad_value, np.uint8)
    # source rect in resized image, dest rect in canvas
    sx0 = max(0, (newW - CW) // 2)
    sy0 = max(0, (newH - CH) // 2)
    dx0 = max(0, (CW - newW) // 2)
    dy0 = max(0, (CH - newH) // 2)
    w = min(newW, CW)
    h = min(newH, CH)
    out[dy0:dy0 + h, dx0:dx0 + w] = resized[sy0:sy0 + h, sx0:sx0 + w]
    info = {"scale": s, "src_wh": (W, H), "new_wh": (newW, newH),
            "canvas_wh": (CW, CH), "sx0": sx0, "sy0": sy0, "dx0": dx0, "dy0": dy0}
    return out, info


def remap_norm(u_norm, v_norm, info):
    """[0,2000)-normalized coords in the ORIGINAL image -> canvas frame."""
    W, H = info["src_wh"]
    CW, CH = info["canvas_wh"]
    x = u_norm / 2000.0 * W * info["scale"] - info["sx0"] + info["dx0"]
    y = v_norm / 2000.0 * H * info["scale"] - info["sy0"] + info["dy0"]
    return x / CW * 2000.0, y / CH * 2000.0


def selftest():
    # synthetic: a marker at known pixel must land where remap says
    K = (606.99, 606.49, 321.73, 260.76)
    img = np.zeros((480, 640, 3), np.uint8)
    px, py = 500, 130
    img[py - 2:py + 3, px - 2:px + 3] = 255
    out_dyn, info_dyn = unify(img, K)   # dynamic mode: coords must be invariant
    un, vn = px / 640 * 2000, py / 480 * 2000
    un_d, vn_d = remap_norm(un, vn, info_dyn)
    assert abs(un_d - un) < 2 and abs(vn_d - vn) < 2, "dynamic mode must be coord-invariant (2 norm units ~ 0.6px rounding)"
    canvas, info = unify(img, K, canvas=CANVAS)
    un, vn = px / 640 * 2000, py / 480 * 2000
    un2, vn2 = remap_norm(un, vn, info)
    cx = int(un2 / 2000 * info["canvas_wh"][0])
    cy = int(vn2 / 2000 * info["canvas_wh"][1])
    patch = canvas[max(0, cy - 4):cy + 5, max(0, cx - 4):cx + 5]
    assert patch.max() == 255, "marker not found at remapped location"
    # effective focal in canvas: fx * scale == 1000
    assert abs(K[0] * info["scale"] - F_TARGET) < 1.0
    print(f"selftest OK: scale={info['scale']:.4f}, marker remap verified, f'={K[0]*info['scale']:.1f}px")


if __name__ == "__main__":
    selftest()
