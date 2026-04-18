# 3DBENCH: VLM Spatial Reasoning Benchmark in LIBERO

Fine-grained zero-shot evaluation of VLMs on 3D spatial reasoning tasks
within the [LIBERO](https://libero-project.github.io) robot manipulation environment.

---

## What This Benchmark Tests

Each sample is a single RGB image from a LIBERO scene. The VLM is asked four
spatial questions, answered in a defined robot-base coordinate frame:

| # | Question | Ground Truth Source | Metric |
|---|---|---|---|
| Q1 | Target object 3D position [x,y,z] | `sim.data.get_body_xpos(obj) - base_pos` | MAE/RMSE per axis |
| Q2 | Gripper 3D position [x,y,z] | `obs["robot0_eef_pos"] - base_pos` | MAE/RMSE per axis |
| Q3 | Can gripper close to grasp now? | `dist(EE, object) < 0.04 m` | Accuracy, F1 |
| Q4 | Next movement direction [dx,dy,dz] | Unit vector EE → target object | Cosine similarity |

### Coordinate Frame

```
Z (up)
│
│    X (forward, into workspace)
│   /
│  /
└──────── Y (left)

Origin: robot arm base center
Units: meters
```

Workspace bounds (approximate): X ∈ [0.05, 0.45], Y ∈ [−0.30, 0.30], Z ∈ [0.80, 1.05]

---

## Quick Start

### 1. Install Dependencies

```bash
# Install LIBERO (choose one)
pip install hf-libero
# OR clone and add to PYTHONPATH:
# git clone https://github.com/Lifelong-Robot-Learning/LIBERO && export PYTHONPATH=$PWD/LIBERO:$PYTHONPATH

# Install benchmark dependencies
pip install -r requirements.txt
```

### 2. Extract Ground Truth

```bash
python scripts/01_extract_gt.py \
    --suite libero_spatial \
    --n_states 5 \
    --out_dir data/gt
```

This produces `data/gt/sample_NNNN.png` and `data/gt/sample_NNNN.json` for
each sample, plus a `manifest.json` index.

### 3. Run VLM Evaluation

```bash
# All three models (requires GPU for Qwen and InternVL2)
python scripts/02_run_vlm_eval.py

# Random baseline only (no GPU)
python scripts/02_run_vlm_eval.py --models random

# Single model
python scripts/02_run_vlm_eval.py --models qwen2.5-vl-7b --device cuda:0
```

### 4. Compute Metrics

```bash
python scripts/03_compute_metrics.py
```

Example output:

```
SPATIAL REASONING BENCHMARK RESULTS
------------------------------------------------------------------------------------------
Model                           Parse%   Q1 MAE(m)   Q2 MAE(m)     Q3 Acc     Q3 F1   Q4 CosSim
------------------------------------------------------------------------------------------
qwen2.5-vl-7b                   87.3%      0.1832      0.1247     0.5600    0.1034      0.2341
internvl2-8b                    91.2%      0.2014      0.1589     0.5200    0.0823      0.1876
random                         100.0%      0.1654      0.1629     0.5000    0.0000      0.0012
------------------------------------------------------------------------------------------
```

---

## File Structure

```
3DBENCH/
├── data/                         # gitignored — generated outputs
│   ├── gt/
│   │   ├── sample_0000.png
│   │   ├── sample_0000.json
│   │   └── manifest.json
│   ├── responses/
│   │   ├── qwen2.5-vl-7b/
│   │   ├── internvl2-8b/
│   │   └── random/
│   └── results.json
├── scripts/
│   ├── 01_extract_gt.py          # Extract GT from LIBERO sim
│   ├── 02_run_vlm_eval.py        # Query VLMs with spatial prompts
│   └── 03_compute_metrics.py     # Parse responses and compute metrics
├── src/bench/
│   ├── gt_extractor.py           # MuJoCo sim state helpers
│   ├── prompt_builder.py         # Build structured VLM prompts
│   ├── output_parser.py          # JSON response parser
│   └── metrics.py                # MAE, RMSE, F1, cosine similarity
├── prompts/
│   └── spatial_qa.yaml           # Prompt templates
├── lerobot/                      # Reference: LeRobot repo (read-only)
├── requirements.txt
└── README.md
```

---

## Adding New Models

Subclass `VLMBase` in `scripts/02_run_vlm_eval.py` and register it in
`_MODEL_REGISTRY`:

```python
class MyModel(VLMBase):
    slug = "my-model"

    def generate(self, image_path, system_prompt, user_prompt, max_new_tokens=512) -> str:
        ...  # return raw text response

_MODEL_REGISTRY["my-model"] = MyModel
```

---

## Notes

- **GT at task reset**: All samples are extracted immediately after environment reset
  and physics settling. At this point, the gripper is far from any object, so Q3
  ground truth is almost always `False`. This tests whether VLMs understand
  "gripper not yet at object" vs. "gripper already grasping".

- **Target object detection**: Keyword matching between task description and MuJoCo
  body names. If matching fails, the first detected object is used. Check
  `01_extract_gt.py` output for `target_object_name` to verify correctness.

- **Object positions**: Only bodies that pass `is_scene_object_body()` filtering are
  included. If an expected object is missing, adjust `_ROBOT_KEYWORDS`/`_ENV_KEYWORDS`
  in `src/bench/gt_extractor.py`.
