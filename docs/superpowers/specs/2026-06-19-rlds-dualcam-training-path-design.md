# RLDS Dual-Camera Training Path — Design Spec

**Date:** 2026-06-19
**Author:** brainstorming session (ang.li@aimai.ai)
**Status:** approved design, pre-implementation

## Goal

Stand up an **RLDS-backed training path** for the LIBERO VLA ablation that is a
**byte-for-byte known-good reference** to the current HDF5 path, so we can
isolate whether the persistent eval≈0 result is a **data-pipeline** problem or a
**model** problem.

The current production run uses
`3DBENCH/configs/vla_ablation/e0_full_dualcam_bs256.json`, which trains
`RoboQwen25VL` on **dual-camera** LIBERO (agent + wrist) via the HDF5 loader
(`LiberoActionPredictionDataset`). openvla's native pipeline is RLDS
(`OpenVLADataset`). Both already exist in the repo and are designed for parity;
this spec covers the env, data, and the **two code gaps** that prevent the RLDS
path from matching the dual-camera HDF5 path today.

Related memory: `[[vla-libero-train-eval-mismatches]]`,
`[[vla-loss-type-and-crop-changes]]`.

## Key finding: the pipeline is already built for HDF5↔RLDS parity

Both dataset classes inherit `ActionPredictionDataset` and emit the **same
5-field contract** through the **same** `batch_transform`:

```
(task_description, action, episode_mask, images, gripper_images)
```

- `VLM4VLA/vlm4vla/data/openvla_action_prediction_dataset.py` — `OpenVLADataset` (RLDS, IterableDataset)
- `VLM4VLA/vlm4vla/data/libero_action_prediction_dataset.py` — `LiberoActionPredictionDataset` (HDF5, map-style)

`GRDataModule` (`datamodule/gr_datamodule.py`) already dispatches RLDS vs HDF5
automatically on `issubclass(dataset_cls, IterableDataset)`. **Switching paths is
a config `type` change — no model/trainer code changes.**

Therefore the work is **environmental + two small data-path edits**, not a
rewrite.

## Obstacles (all confirmed)

### 1. Environment conflict — separate env is mandatory

`vlmbench` env: `torch 2.7.1+cu126`, `numpy 2.2.6`, Python 3.12,
`transformers 5.8.0`, **no tensorflow / tensorflow_datasets / dlimp**.

openvla RLDS stack pins (`openvla/pyproject.toml`):
`tensorflow==2.15.0`, `tensorflow_datasets==4.9.3`,
`tensorflow_graphics==2021.12.3` → require **numpy<2** and have **no Python 3.12
wheels**. These **cannot** be added to `vlmbench` without breaking numpy/torch.

**Decision:** build a separate env `vlmbench-rlds`. To keep the data-vs-model
comparison clean (not confounded by framework versions), the new env keeps
**torch / transformers / lightning identical to `vlmbench`** and adds *only* the
TF/tfds/dlimp stack + the forced `numpy<2` + Python 3.11.

### 2. No RLDS-format data exists locally

Local LIBERO is **HDF5 only** (`/workspace/tingting/LIBERO/libero/datasets/`).
There are **zero** tfrecords / `dataset_info.json` on disk.

**Decision:** download the prebuilt `modified_libero_rlds` TFDS datasets
(`libero_10_no_noops`, `libero_goal_no_noops`, `libero_object_no_noops`,
`libero_spatial_no_noops`). These **already contain a `wrist_image` stream**
(`openvla/prismatic/vla/datasets/rlds/oxe/configs.py`:
`"wrist": "wrist_image"`).

## The two code gaps to close

### Gap 1 — RLDS path drops the wrist view

The dual-cam config needs agent **and** wrist. The RLDS data has the wrist
stream, but the loader is hardcoded to single-view:

- `vlm4vla/data/base_openvla_dataset.py:79` — `load_camera_views=("primary",)`
  → must become `("primary", "wrist")`.
- `vlm4vla/data/openvla_action_prediction_dataset.py` — `__iter__` hardcodes
  `gripper_images=None` → must become
  `gripper_images=rlds_batch["observation"]["image_wrist"]`.

After the RLDS frame transform, the wrist frames are available under
`observation["image_wrist"]` (`openvla/.../rlds/dataset.py:68-70`).

**Why this is sufficient (no model/collater changes):** in
`base_action_prediction_dataset.py:256`, `dual_image` is keyed **purely on
whether wrist tensors exist** (`gripper_image_tensors[0] is not None`), NOT on
`use_hand_rgb`. The HDF5 loader already produces wrist images
unconditionally, which is why the current dual-cam run works. Emitting
`image_wrist` from the RLDS path reproduces exactly that condition → Qwen emits
two image_pad blocks and trains/evals dual-view, identical to HDF5.

`use_hand_rgb: true` stays in the config (it gates the trainer-side hand_rgb
tensor); the dual-view behavior itself comes from the wrist tensors existing.

**Gating:** make the camera-view selection follow `use_hand_rgb` so the
single-view RLDS path is unaffected. When `use_hand_rgb` is true →
`("primary","wrist")` + emit `image_wrist`; else → `("primary",)` + `None`.

### Gap 2 — Image orientation parity (silent-failure risk)

This is the highest-risk item — an orientation mismatch is exactly the kind of
bug that yields eval≈0.

- **HDF5 path:** raw LIBERO frames are stored **upside-down** (robosuite/OpenGL
  framebuffer). The loader rotates 180° (`agent[:, ::-1, ::-1]`,
  `libero_hdf5_dataset.py:177-178`) → **upright**, matching the live eval env.
- **RLDS path:** `modified_libero_rlds` frames are **already upright** (fixed at
  TFDS build time by the openvla converter).

**Parity = both end upright**, which means the RLDS path must **NOT** add any
flip. A naive copy of the HDF5 flip into RLDS would make images upside-down.

**Requirement:** verify orientation **empirically**, not by assumption — dump one
agent + one wrist frame from BOTH paths to PNG and visually confirm both are
upright and depict comparable scenes. This is a **hard gate** before any
training run.

## Components

| # | Component | Detail |
|---|---|---|
| A | env `vlmbench-rlds` | py3.11, tf==2.15.0, tfds==4.9.3, tensorflow_graphics==2021.12.3, numpy<2, dlimp (from `openvla/dlimp_openvla/`, editable), `pip install -e openvla/`. Same torch/transformers/lightning as `vlmbench`. Followed by an import + numpy-compat smoke test. |
| B | RLDS data | Download `modified_libero_rlds` (4 `libero_*_no_noops`) into a new `data_root_dir`, e.g. `/workspace/tingting/LIBERO_rlds/`. Verify with `tfds.builder_from_directory`. |
| C | RLDS dual-cam code edits | Gap 1 edits in `base_openvla_dataset.py` + `openvla_action_prediction_dataset.py`, gated on `use_hand_rgb`. |
| D | Config | `3DBENCH/configs/vla_ablation_rlds/e0_full_dualcam_bs256.json` — copy of the current dual-cam config; only `train_dataset`/`val_dataset` blocks swapped to `type: OpenVLADataset`, `data_root_dir` → RLDS root, `data_mix: libero_10_no_noops`. Everything else identical (lr 8e-5, bs, loss_type l1_unified, use_hand_rgb true). |
| E | Parity smoke test | Pull one batch from each path; assert both carry agent **and** wrist tensors, matching shapes/dtypes, aligned action ranges, matching gripper convention; **dump PNGs for visual orientation check**. |

**Launcher:** reuse `3DBENCH/scripts/run_vla_ablation.sh` unchanged via
`CONFIG_DIR=configs/vla_ablation_rlds` (already supported).

## Data flow

```
modified_libero_rlds (TFDS, upright, has wrist_image)
        │
        ▼
OpenVLADataset (RLDS, IterableDataset)  ──┐
   load_camera_views=("primary","wrist")  │
   images=image_primary                   │  batch_transform (SHARED)
   gripper_images=image_wrist             ├──►  5-field contract ─► GRDataModule ─► RoboQwen25VL ─► trainer
                                          │
HDF5 LIBERO (raw upside-down → 180° flip) │
LiberoActionPredictionDataset ───────────┘
   images=agent, gripper_images=wrist
```

Both paths converge on the identical `batch_transform` and downstream stack.

## Scope

**In scope**
- Build `vlmbench-rlds` env (riskiest step → first, with verification gate).
- Download `modified_libero_rlds`.
- Two RLDS code edits (dual-cam + confirm no-flip), gated on `use_hand_rgb`.
- One dual-cam RLDS config mirroring `e0_full_dualcam_bs256.json`.
- Parity smoke test with mandatory image dump.

**Out of scope (YAGNI)**
- Converting HDF5 → RLDS ourselves (we download instead).
- Any model/trainer code changes (none required).
- Multi-suite RLDS mixtures — expand to all 4 suites only after `e0` (libero_10)
  parity is proven.
- Changing the HDF5 path.

## Success criteria

1. `vlmbench-rlds` imports torch + tensorflow + tfds + prismatic + dlimp with no
   numpy/ABI errors.
2. `tfds.builder_from_directory` loads all 4 LIBERO RLDS builds.
3. Parity smoke test passes: RLDS batch carries agent **and** wrist, shapes/dtypes
   match HDF5, action ranges align, gripper convention matches.
4. Dumped PNGs confirm **both** paths produce **upright** agent + wrist frames.
5. RLDS `e0_full_dualcam_bs256` trains end-to-end (loss decreases) using the
   unchanged launcher + trainer.

## Open risk to watch

A new env has different TF/torch build resolution than `vlmbench`. If RLDS trains
fine but HDF5 evals ≈0, the gap could be the **data path** *or* the
**env/version delta**. Mitigation: keep torch/transformers/lightning pinned
identical to `vlmbench`; if numpy<2 forces a torch rebuild, record the exact
versions so the confound is documented, not hidden. (Implemented: env keeps
torch 2.7.1 / transformers 5.8.0 / lightning 2.6.5, and flash-attn 2.8.3 matches
`vlmbench` so both runs use the same `flash_attention_2`.)

## Known RLDS↔HDF5 differences (post-implementation, verified)

These are the *only* divergences between the two paths besides the data source
itself — all intended or benign for the train-time data-vs-model comparison:

- **Val-split crop asymmetry.** The openvla RLDS pipeline gates *all* image
  augmentation (incl. resized-crop) on `train` (`rlds/dataset.py:498`), so the
  RLDS **val** split does decode+resize only — **no crop**. The HDF5 val split
  (`libero_hdf5_dataset.py:192-194`, `image_aug=True`) applies a center 0.9
  crop. Both configs set `val_dataset.image_aug: true`. This is **val-only**,
  and val is already decoupled from sim eval, so it does **not** confound the
  train-time comparison. Noted, not fixed.
- **Train path parity holds:** q99 action normalization, gripper convention
  (`+1=open, 0=close`), resized-crop, and image orientation (both upright; RLDS
  adds no flip) all match. The dual-cam wiring is `use_hand_rgb`-gated with no
  KeyError/None path; the single-view default path is byte-for-byte unchanged.
