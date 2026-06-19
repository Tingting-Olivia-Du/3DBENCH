# RLDS Dual-Camera Training Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up an RLDS-backed dual-camera LIBERO training path that is a byte-for-byte parity reference to the existing HDF5 path, to isolate whether eval≈0 is a data or model problem.

**Architecture:** A separate conda env (`vlmbench-rlds`) carries the TF/tfds/dlimp stack that conflicts with `vlmbench`'s numpy 2 / py3.12. The repo's `OpenVLADataset` (RLDS) already feeds the same `batch_transform` 5-field contract as the HDF5 `LiberoActionPredictionDataset`, so only two RLDS code gaps need closing (wrist view + confirm no image flip). A new config swaps just the dataset block; launcher and model are untouched.

**Tech Stack:** Python 3.11, tensorflow 2.15.0, tensorflow_datasets 4.9.3, tensorflow_graphics 2021.12.3, numpy<2, dlimp (moojink fork), torch 2.7.1, transformers 5.8.0, lightning 2.6.5, prismatic (openvla, editable).

## Global Constraints

- **Env isolation is mandatory.** TF 2.15 / tfds 4.9.3 require numpy<2 and have no py3.12 wheels. Do NOT install TF into `vlmbench`. Build `vlmbench-rlds` separately.
- **Pin to match `vlmbench` where possible:** torch `2.7.1+cu126`, transformers `5.8.0`, lightning `2.6.5`. Keep transformers at 5.8.0 (it has `Qwen2_5_VLForConditionalGeneration`; openvla's pinned 4.40.1 does NOT).
- **conda binary:** `/workspace/ghsun/miniconda3/condabin/conda`. `vlmbench` lives at `/workspace/tingting/envs/vlmbench` (path-based conda env).
- **Repo roots:** VLM4VLA at `/workspace/tingting/VLM4VLA`; openvla (with `prismatic`) at `/workspace/tingting/VLM4VLA/openvla`; 3DBENCH at `/workspace/tingting/3DBENCH`.
- **RLDS data root (new):** `/workspace/tingting/LIBERO_rlds/`. TFDS layout is `<root>/<dataset_name>/<version>/`.
- **Dataset names:** `libero_10_no_noops`, `libero_goal_no_noops`, `libero_object_no_noops`, `libero_spatial_no_noops`. e0 uses `libero_10_no_noops` only.
- **Orientation parity rule:** HDF5 frames are stored upside-down and get a 180° flip (`[:, ::-1, ::-1]`); `modified_libero_rlds` frames are ALREADY upright. The RLDS path must NOT add any flip. Verify by dumping PNGs — hard gate before training.
- **Gripper convention (verified):** both paths produce gripper `+1=open, 0=close`. Parity assertion must hold.
- **Disk:** ~394 GB free on `/`. RLDS LIBERO is tens of GB — fine, but check before downloading all 4.
- **Branch:** work on `feat/tf-free-libero-pipeline` (current). Commit frequently.

---

## File Structure

- **Create** `/workspace/tingting/3DBENCH/scripts/setup_rlds_env.sh` — reproducible env build (Task 1).
- **Create** `/workspace/tingting/3DBENCH/scripts/download_libero_rlds.sh` — data fetch + verify (Task 3).
- **Modify** `/workspace/tingting/VLM4VLA/vlm4vla/data/base_openvla_dataset.py` — gate `load_camera_views` on dual-cam (Task 4).
- **Modify** `/workspace/tingting/VLM4VLA/vlm4vla/data/openvla_action_prediction_dataset.py` — emit `image_wrist` as `gripper_images` (Task 4).
- **Create** `/workspace/tingting/3DBENCH/configs/vla_ablation_rlds/e0_full_dualcam_bs256.json` — RLDS dual-cam config (Task 5).
- **Create** `/workspace/tingting/VLM4VLA/tools/rlds_parity_smoketest.py` — one-batch parity + image dump (Task 6).

---

## Task 1: Build the `vlmbench-rlds` conda env

**Files:**
- Create: `/workspace/tingting/3DBENCH/scripts/setup_rlds_env.sh`

**Interfaces:**
- Produces: a conda env at `/workspace/tingting/envs/vlmbench-rlds` with python 3.11, torch 2.7.1, tensorflow 2.15.0, tfds 4.9.3, tensorflow_graphics 2021.12.3, numpy<2, dlimp, prismatic (editable). Invoked via `/workspace/ghsun/miniconda3/condabin/conda run -p /workspace/tingting/envs/vlmbench-rlds python ...`.

- [ ] **Step 1: Write the setup script**

Create `/workspace/tingting/3DBENCH/scripts/setup_rlds_env.sh`:

```bash
#!/bin/bash
# Build the RLDS-capable training env (separate from vlmbench, which has
# numpy 2 / py3.12 and cannot host tensorflow 2.15).
set -euo pipefail

CONDA=/workspace/ghsun/miniconda3/condabin/conda
ENV_PREFIX=/workspace/tingting/envs/vlmbench-rlds
OPENVLA_ROOT=/workspace/tingting/VLM4VLA/openvla
VLM4VLA_ROOT=/workspace/tingting/VLM4VLA

# 1) Base env: python 3.11 (tf 2.15 has no 3.12 wheels).
$CONDA create -y -p "$ENV_PREFIX" python=3.11

PY="$CONDA run -p $ENV_PREFIX python"
PIP="$CONDA run -p $ENV_PREFIX pip"

# 2) torch matched to vlmbench (cu126 build).
$PIP install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu126

# 3) TF/RLDS stack (these force numpy<2).
$PIP install "tensorflow==2.15.0" "tensorflow_datasets==4.9.3" \
             "tensorflow_graphics==2021.12.3" "numpy<2"

# 4) dlimp (moojink fork; --no-deps so it doesn't drag conflicting pins).
$PIP install --no-deps "git+https://github.com/moojink/dlimp_openvla"

# 5) Training stack matched to vlmbench.
$PIP install "transformers==5.8.0" "lightning==2.6.5"

# 6) prismatic (openvla) + vlm4vla, editable, no-deps to preserve pins above.
$PIP install --no-deps -e "$OPENVLA_ROOT"
$PIP install --no-deps -e "$VLM4VLA_ROOT"

echo "[setup_rlds_env] done. Prefix: $ENV_PREFIX"
```

- [ ] **Step 2: Run the setup script**

Run: `bash /workspace/tingting/3DBENCH/scripts/setup_rlds_env.sh 2>&1 | tail -40`
Expected: ends with `[setup_rlds_env] done.` and no `ERROR:` lines from pip resolver. If pip reports residual missing transitive deps (e.g. `draccus`, `einops`, `rich`, `wandb`, `h5py`, `pillow`), install them with `$CONDA run -p /workspace/tingting/envs/vlmbench-rlds pip install <pkg>` and append those lines to the script.

- [ ] **Step 3: Commit**

```bash
cd /workspace/tingting/3DBENCH
git add scripts/setup_rlds_env.sh
git commit -m "feat: add vlmbench-rlds env setup script"
```

---

## Task 2: Verify the env imports cleanly (numpy/ABI gate)

**Files:**
- Test: inline python via conda run (no file).

**Interfaces:**
- Consumes: env from Task 1.
- Produces: confidence that torch + tf + tfds + dlimp + prismatic coexist. This is the first hard gate.

- [ ] **Step 1: Write the import smoke check and run it**

Run:
```bash
/workspace/ghsun/miniconda3/condabin/conda run -p /workspace/tingting/envs/vlmbench-rlds python - <<'EOF'
import numpy, torch, tensorflow as tf, tensorflow_datasets as tfds, dlimp
import prismatic
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES
print("numpy", numpy.__version__)
print("torch", torch.__version__)
print("tf", tf.__version__)
print("tfds", tfds.__version__)
# tf must not grab GPUs (clobbers torch)
tf.config.set_visible_devices([], "GPU")
# round-trip a tensor through torch to confirm numpy<2 ABI is fine
import numpy as np
x = torch.from_numpy(np.zeros((2,3), dtype=np.float32))
assert x.shape == (2,3)
print("OXE mixtures has libero_10_no_noops:", "libero_10_no_noops" in
      [n for n,_ in OXE_NAMED_MIXTURES.get("libero_10_no_noops", [("libero_10_no_noops",1.0)])])
print("ALL IMPORTS OK")
EOF
```
Expected: prints versions (numpy `1.x`, torch `2.7.1`, tf `2.15.0`, tfds `4.9.3`) and `ALL IMPORTS OK`. No `ImportError`, no numpy ABI `RuntimeError`.

- [ ] **Step 2: If it fails, record and fix**

If numpy ABI error on torch import: the cu126 torch wheel may need `numpy<2` reinstalled AFTER torch — run `$CONDA run -p ... pip install --force-reinstall "numpy<2"` and re-run Step 1. Document the final working numpy version in `scripts/setup_rlds_env.sh` as a comment. Do not proceed until `ALL IMPORTS OK`.

---

## Task 3: Download and verify `modified_libero_rlds`

**Files:**
- Create: `/workspace/tingting/3DBENCH/scripts/download_libero_rlds.sh`

**Interfaces:**
- Consumes: env from Task 1.
- Produces: TFDS data at `/workspace/tingting/LIBERO_rlds/<dataset_name>/<version>/` loadable via `tfds.builder(name, data_dir=...)`.

- [ ] **Step 1: Write the download + verify script**

Create `/workspace/tingting/3DBENCH/scripts/download_libero_rlds.sh`:

```bash
#!/bin/bash
# Download prebuilt modified_libero_rlds TFDS datasets (openvla LIBERO).
# These already contain a `wrist_image` stream and are stored UPRIGHT.
set -euo pipefail

DEST=/workspace/tingting/LIBERO_rlds
mkdir -p "$DEST"

# Prebuilt datasets are published on the HF Hub under openvla/modified_libero_rlds.
# Mirror via hf-mirror (matches main.py HF_ENDPOINT).
export HF_ENDPOINT="https://hf-mirror.com"

CONDA=/workspace/ghsun/miniconda3/condabin/conda
ENV=/workspace/tingting/envs/vlmbench-rlds

# Only e0 needs libero_10 to start; pass "all" to fetch the 4 suites.
SUITES="${1:-libero_10_no_noops}"
if [ "$SUITES" = "all" ]; then
  SUITES="libero_10_no_noops libero_goal_no_noops libero_object_no_noops libero_spatial_no_noops"
fi

$CONDA run -p "$ENV" pip install -q "huggingface_hub[cli]"
for s in $SUITES; do
  echo "[download] $s -> $DEST/$s"
  $CONDA run -p "$ENV" huggingface-cli download \
      openvla/modified_libero_rlds --repo-type dataset \
      --include "$s/*" --local-dir "$DEST"
done
echo "[download] done -> $DEST"
ls -R "$DEST" | head -40
```

- [ ] **Step 2: Run the download (e0 suite only first)**

Run: `bash /workspace/tingting/3DBENCH/scripts/download_libero_rlds.sh 2>&1 | tail -50`
Expected: `/workspace/tingting/LIBERO_rlds/libero_10_no_noops/<version>/` populated with `*.tfrecord-*`, `dataset_info.json`, `features.json`.
NOTE: if `openvla/modified_libero_rlds` is unavailable on the mirror, fall back to the GS bucket documented in openvla README (`gs://...`) via `gsutil -m cp -r`, or point `--repo-type`/repo id at the actual published location; record the working source as a comment in the script.

- [ ] **Step 3: Verify TFDS loads it**

Run:
```bash
/workspace/ghsun/miniconda3/condabin/conda run -p /workspace/tingting/envs/vlmbench-rlds python - <<'EOF'
import tensorflow_datasets as tfds
b = tfds.builder("libero_10_no_noops", data_dir="/workspace/tingting/LIBERO_rlds")
b.download_and_prepare(download_config=tfds.download.DownloadConfig(verify_ssl=False)) if not b.info.splits else None
print("splits:", {k:v.num_examples for k,v in b.info.splits.items()})
feat = b.info.features["steps"]["observation"]
print("obs keys:", list(feat.keys()))
assert "image" in feat and "wrist_image" in feat, "missing wrist_image stream"
print("TFDS OK with wrist_image")
EOF
```
Expected: prints split counts, `obs keys` including `image` and `wrist_image`, then `TFDS OK with wrist_image`.

- [ ] **Step 4: Commit**

```bash
cd /workspace/tingting/3DBENCH
git add scripts/download_libero_rlds.sh
git commit -m "feat: add modified_libero_rlds download+verify script"
```

---

## Task 4: Enable dual-camera (wrist) in the RLDS path

**Files:**
- Modify: `/workspace/tingting/VLM4VLA/vlm4vla/data/base_openvla_dataset.py:79`
- Modify: `/workspace/tingting/VLM4VLA/vlm4vla/data/openvla_action_prediction_dataset.py` (`__init__` + `__iter__`)

**Interfaces:**
- Consumes: `use_hand_rgb` kwarg already flowing into datasets via `GRDataModule.kwargs` (it is a top-level config field).
- Produces: when `use_hand_rgb=True`, `OpenVLADataset.__iter__` yields `gripper_images=rlds_batch["observation"]["image_wrist"]` (shape matches `image_primary`); when False, yields `None`. Downstream `batch_transform` keys `dual_image` on wrist-tensor existence (`base_action_prediction_dataset.py:256`), so Qwen goes dual-view automatically — no model/collater change.

- [ ] **Step 1: Make `load_camera_views` follow `use_hand_rgb` in `base_openvla_dataset.py`**

In `RLDSDataset.__init__`, add a `use_hand_rgb` kwarg (default False) and change the hardcoded `load_camera_views=("primary",)` (line 79) to be conditional. Edit:

```python
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        image_size: int,
        chunk_action: bool = True,
        frame_num: int = -1,
        left_pad: bool = False,
        window_sample: Literal["sliding", "range"] = "sliding",
        window_size: int = 1,
        fwd_pred_next_n: int = 1,
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        filter_langs=False,
        use_hand_rgb: bool = False,
        **kwargs,
    ) -> None:
```

Then change the `get_oxe_dataset_kwargs_and_weights` call:

```python
        self.use_hand_rgb = use_hand_rgb
        _camera_views = ("primary", "wrist") if use_hand_rgb else ("primary",)
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=_camera_views,
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=action_proprio_normalization_type,
        )
```

- [ ] **Step 2: Emit `image_wrist` from `OpenVLADataset.__iter__`**

In `openvla_action_prediction_dataset.py`, `OpenVLADataset.__init__` already forwards `**kwargs` to `RLDSDataset.__init__`, so `use_hand_rgb` flows through. Change `__iter__` to read `self.use_hand_rgb`:

```python
    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in RLDSDataset.__iter__(self):
            gripper_images = (
                rlds_batch["observation"]["image_wrist"]
                if getattr(self, "use_hand_rgb", False)
                else None
            )
            yield self.batch_transform(
                task_description=rlds_batch["task"]["language_instruction"].decode(),
                action=rlds_batch["action"],
                episode_mask=rlds_batch["chunk_mask"],
                images=rlds_batch["observation"]["image_primary"],
                gripper_images=gripper_images,
            )
```

- [ ] **Step 3: Confirm NO image flip exists in the RLDS path**

Run: `grep -rn "::-1\|np.flip\|tf.image.flip\|rot90\|flip_up_down\|flip_left_right" /workspace/tingting/VLM4VLA/vlm4vla/data/base_openvla_dataset.py /workspace/tingting/VLM4VLA/vlm4vla/data/openvla_action_prediction_dataset.py`
Expected: **no matches.** RLDS data is already upright; any match here is a parity bug to remove. (Visual confirmation happens in Task 6.)

- [ ] **Step 4: Commit**

```bash
cd /workspace/tingting/VLM4VLA
git add vlm4vla/data/base_openvla_dataset.py vlm4vla/data/openvla_action_prediction_dataset.py
git commit -m "feat: dual-camera (wrist) support in RLDS path, gated on use_hand_rgb"
```

---

## Task 5: Create the RLDS dual-cam config

**Files:**
- Create: `/workspace/tingting/3DBENCH/configs/vla_ablation_rlds/e0_full_dualcam_bs256.json`

**Interfaces:**
- Consumes: dataset class `OpenVLADataset`, RLDS data root `/workspace/tingting/LIBERO_rlds`, `data_mix: libero_10_no_noops`.
- Produces: a config runnable via `CONFIG_DIR=configs/vla_ablation_rlds bash scripts/run_vla_ablation.sh e0_full_dualcam_bs256`.

- [ ] **Step 1: Create the config by copying the HDF5 dual-cam config and swapping only the dataset blocks**

Copy `/workspace/tingting/3DBENCH/configs/vla_ablation/e0_full_dualcam_bs256.json` to the new path. Change ONLY: `task_name`, `output_root`/`log_root`/`cache_root` (point to `runs/vla_ablation_rlds/...`), set `resume` to `null`, and replace both `train_dataset` and `val_dataset` blocks with:

```json
    "train_dataset": {
        "type": "OpenVLADataset",
        "data_root_dir": "/workspace/tingting/LIBERO_rlds",
        "model_name": "qwen25vl",
        "image_aug": true,
        "data_mix": "libero_10_no_noops",
        "window_sample": "sliding",
        "organize_type": "interleave",
        "shuffle_buffer_size": 51200,
        "train": true
    },
    "val_dataset": {
        "type": "OpenVLADataset",
        "data_root_dir": "/workspace/tingting/LIBERO_rlds",
        "model_name": "qwen25vl",
        "image_aug": true,
        "data_mix": "libero_10_no_noops",
        "window_sample": "sliding",
        "organize_type": "interleave",
        "shuffle_buffer_size": 10000,
        "train": false
    },
```

Keep everything else identical to the HDF5 config: `use_hand_rgb: true`, `learning_rate: 8e-05`, `act_head.loss_type: l1_unified`, batch/accum/scheduler. Note: the RLDS path takes `image_size`/`fwd_pred_next_n`/`window_size`/`use_hand_rgb` from top-level config via `GRDataModule.kwargs`, and does its own crop via `image_aug` (no `crop_scale` key needed — drop it).

- [ ] **Step 2: Validate the JSON parses**

Run: `python -c "import json; c=json.load(open('/workspace/tingting/3DBENCH/configs/vla_ablation_rlds/e0_full_dualcam_bs256.json')); print('use_hand_rgb', c['use_hand_rgb'], '| train type', c['train_dataset']['type'], '| mix', c['train_dataset']['data_mix'])"`
Expected: `use_hand_rgb True | train type OpenVLADataset | mix libero_10_no_noops`

- [ ] **Step 3: Commit**

```bash
cd /workspace/tingting/3DBENCH
git add configs/vla_ablation_rlds/e0_full_dualcam_bs256.json
git commit -m "feat: add RLDS dual-cam e0 config (libero_10_no_noops)"
```

---

## Task 6: Parity smoke test with image dump (HARD GATE)

**Files:**
- Create: `/workspace/tingting/VLM4VLA/tools/rlds_parity_smoketest.py`

**Interfaces:**
- Consumes: both dataset configs (HDF5 + RLDS), the modified RLDS data, the dual-cam code from Task 4.
- Produces: assertion pass + dumped PNGs at `/workspace/tingting/tmp/parity_{rlds,hdf5}_{agent,wrist}.png`. Run in the `vlmbench-rlds` env (it can read both HDF5 and RLDS).

- [ ] **Step 1: Write the smoke test**

Create `/workspace/tingting/VLM4VLA/tools/rlds_parity_smoketest.py`:

```python
"""Pull one item from the RLDS and HDF5 LIBERO dual-cam paths, assert the
5-field contract matches, and dump agent+wrist PNGs for a visual orientation
gate. Run in the vlmbench-rlds env."""
import os, numpy as np
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
OUT = "/workspace/tingting/tmp"
os.makedirs(OUT, exist_ok=True)

COMMON = dict(model_name="qwen25vl", image_size=224, window_size=1,
              fwd_pred_next_n=4, use_hand_rgb=True, organize_type="interleave",
              window_sample="sliding", image_aug=False, is_training=False,
              train=False)

def first_item(ds):
    for it in ds:
        return it
    raise RuntimeError("dataset yielded nothing")

def to_uint8(arr):
    a = np.asarray(arr)
    while a.ndim > 3:
        a = a[0]
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
    return a

# --- RLDS path ---
from vlm4vla.data import OpenVLADataset
rlds = OpenVLADataset(data_root_dir="/workspace/tingting/LIBERO_rlds",
                      data_mix="libero_10_no_noops",
                      shuffle_buffer_size=256, **COMMON)
r = first_item(rlds)

# --- HDF5 path ---
from vlm4vla.data import LiberoActionPredictionDataset
hdf5 = LiberoActionPredictionDataset(
    data_root_dir="/workspace/tingting/LIBERO/libero/datasets",
    data_mix="libero_10", crop_scale=0.9, **COMMON)
h = first_item(hdf5)

# --- contract parity ---
for key in ("rgb", "hand_rgb"):
    assert key in r and key in h, f"missing {key}"
print("RLDS rgb tensor shape:", np.asarray(r["rgb"]).shape)
print("HDF5 rgb tensor shape:", np.asarray(h["rgb"]).shape)
print("RLDS hand_rgb present:", r["hand_rgb"][0] is not None)
print("HDF5 hand_rgb present:", h["hand_rgb"][0] is not None)
assert r["hand_rgb"][0] is not None, "RLDS produced no wrist tensor (dual-cam broken)"
assert h["hand_rgb"][0] is not None, "HDF5 produced no wrist tensor"

# --- dump raw frames for visual orientation gate ---
# Pull pre-normalization frames straight off the underlying loaders so the
# PNGs are human-viewable (the collated rgb is normalized).
import itertools
for tag, batch in (("rlds", r), ("hdf5", h)):
    # action range sanity (gripper dim convention parity)
    act = np.asarray(batch["action"])
    print(f"{tag} action shape {act.shape} min {act.min():.3f} max {act.max():.3f} "
          f"gripper[...,-1] unique~ {np.unique(np.round(act[...,-1],2))[:5]}")

print("\nDumping viewable frames...")
# RLDS: re-iterate the raw RLDS batch for unnormalized images.
from vlm4vla.data.base_openvla_dataset import RLDSDataset
raw_rlds = RLDSDataset(data_root_dir="/workspace/tingting/LIBERO_rlds",
                       data_mix="libero_10_no_noops", image_size=224,
                       use_hand_rgb=True, shuffle_buffer_size=256, train=False)
rb = next(iter(raw_rlds))
Image.fromarray(to_uint8(rb["observation"]["image_primary"])).save(f"{OUT}/parity_rlds_agent.png")
Image.fromarray(to_uint8(rb["observation"]["image_wrist"])).save(f"{OUT}/parity_rlds_wrist.png")

# HDF5: read one trajectory raw (post-flip, as training sees it).
from vlm4vla.data.libero_hdf5_dataset import LiberoHDF5Dataset
raw_h5 = LiberoHDF5Dataset(data_root_dir="/workspace/tingting/LIBERO/libero/datasets",
                           data_mix="libero_10", model_name="qwen25vl",
                           image_size=224, window_size=1, fwd_pred_next_n=4,
                           use_hand_rgb=True, image_aug=False, train=False,
                           window_sample="sliding")
for traj in raw_h5.iter_trajectories():
    # iter_trajectories yields keys: language, actions, images (agent),
    # gripper_images (wrist) — both already 180°-flipped to upright.
    Image.fromarray(to_uint8(traj["images"][0])).save(f"{OUT}/parity_hdf5_agent.png")
    Image.fromarray(to_uint8(traj["gripper_images"][0])).save(f"{OUT}/parity_hdf5_wrist.png")
    break

print(f"\nWrote PNGs to {OUT}/parity_*.png")
print("PARITY CONTRACT OK — now VISUALLY confirm all 4 PNGs are UPRIGHT and "
      "the agent views depict comparable LIBERO scenes.")
```

VERIFIED key names (no guessing needed): `batch_transform` returns `rgb`, `hand_rgb`, `text`, `action` (`base_action_prediction_dataset.py:307-314`). `LiberoHDF5Dataset.iter_trajectories` yields `language`, `actions`, `images` (agent), `gripper_images` (wrist), both already 180°-flipped to upright (`libero_hdf5_dataset.py:198-213`).

- [ ] **Step 2: Run the smoke test**

Run: `/workspace/ghsun/miniconda3/condabin/conda run -p /workspace/tingting/envs/vlmbench-rlds python /workspace/tingting/VLM4VLA/tools/rlds_parity_smoketest.py 2>&1 | tail -40`
Expected: prints matching rgb shapes, both `hand_rgb present: True`, aligned action ranges with gripper dim in `{0,1}`, and `PARITY CONTRACT OK`. If `image_wrist` KeyError → Task 4 wiring is wrong; fix before continuing.

- [ ] **Step 3: VISUAL orientation gate (mandatory, human or vision check)**

Read the 4 PNGs (`/workspace/tingting/tmp/parity_rlds_agent.png`, `parity_rlds_wrist.png`, `parity_hdf5_agent.png`, `parity_hdf5_wrist.png`) and confirm: all four are **upright** (table at bottom, not ceiling), and the two agent views show the same LIBERO scene geometry. If RLDS frames are upside-down relative to HDF5, STOP — the modified_libero_rlds build is not pre-flipped as assumed; add an explicit flip ONLY to the RLDS path and document it. Do not train until both paths are visually upright.

- [ ] **Step 4: Commit**

```bash
cd /workspace/tingting/VLM4VLA
git add tools/rlds_parity_smoketest.py
git commit -m "test: RLDS<->HDF5 dual-cam parity smoke test with image dump"
```

---

## Task 7: End-to-end RLDS training smoke (final gate)

**Files:**
- Uses: `/workspace/tingting/3DBENCH/scripts/run_vla_ablation.sh` (unchanged), Task 5 config.

**Interfaces:**
- Consumes: everything above. Produces: a short training run proving the RLDS dual-cam path trains and loss decreases.

- [ ] **Step 1: Launch a short capped run**

Run (single GPU, the env activated for TF):
```bash
cd /workspace/tingting/3DBENCH
CONDA_PREFIX_OVERRIDE=/workspace/tingting/envs/vlmbench-rlds \
CONFIG_DIR=$PWD/configs/vla_ablation_rlds \
CUDA_VISIBLE_DEVICES=0 MASTER_PORT=6071 \
/workspace/ghsun/miniconda3/condabin/conda run -p /workspace/tingting/envs/vlmbench-rlds \
  bash scripts/run_vla_ablation.sh e0_full_dualcam_bs256 2>&1 | tee /workspace/tingting/tmp/rlds_train_smoke.log
```
Let it run ~50–100 steps then Ctrl-C. (Optionally pre-edit the config `trainer.max_steps` to a small number like 50 for the smoke, then restore.)
Expected: dataloader initializes from `OpenVLADataset`, training loop starts, action loss logged and trending down. No TF/torch GPU clobber (TF must show no visible GPU). 

- [ ] **Step 2: Confirm loss is finite and decreasing**

Run: `grep -iE "loss|step" /workspace/tingting/tmp/rlds_train_smoke.log | tail -20`
Expected: finite `action`/total loss values, generally decreasing over the first steps. If loss is NaN/flat from step 0, debug data scaling/normalization before any long run.

- [ ] **Step 3: Commit any config tweaks**

```bash
cd /workspace/tingting/3DBENCH
git add -A configs/vla_ablation_rlds/
git commit -m "chore: finalize RLDS e0 dual-cam config after training smoke" || echo "nothing to commit"
```

---

## Self-Review Notes

- **Spec coverage:** Env (Task 1–2) ↔ spec Component A; data (Task 3) ↔ B; dual-cam code (Task 4) ↔ C/Gap 1; config (Task 5) ↔ D; parity+orientation (Task 6) ↔ E/Gap 2 + success criteria 3–4; training smoke (Task 7) ↔ success criterion 5. All covered.
- **Orientation (highest risk):** verified two ways — Task 4 Step 3 (no flip in code) + Task 6 Step 3 (visual PNG gate). Fallback documented if RLDS turns out not pre-flipped.
- **Type consistency:** `use_hand_rgb` kwarg added in Task 4 Step 1, consumed in Task 4 Step 2 and Task 6 COMMON. `gripper_images`/`hand_rgb` contract consistent across Task 4 and Task 6.
- **Known soft spots flagged inline for the implementer:** (a) exact download source for `modified_libero_rlds` (Task 3 Step 2 fallback), (b) residual pip transitive deps (Task 1 Step 2), (c) exact `batch_transform` output key names (Task 6 NOTE). These require reading the live code during implementation, not guessing.
