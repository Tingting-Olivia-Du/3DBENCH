# 3DBENCH Pipeline — 端到端调用手册

整个研究流程分 6 个阶段。每阶段给出**完整调用命令**、**预期输出**与**前置条件**。所有命令默认在仓库根目录 `/umd-datapool/tingting/3DBENCH` 下执行；激活环境：

```bash
cd 3DBENCH

conda activate /workspace/tingting/envs/vlmbench

conda deactivate
conda activate /workspace/tingting/envs/vlmbench-rlds

ENV_PREFIX=/workspace/tingting/envs/vlmbench-rlds

cd /workspace/tingting/3DBENCH


cd /workspace/tingting/3DBENCH
CONFIG_DIR=$PWD/configs/vla_ablation_rlds 

CUDA_VISIBLE_DEVICES=4 \
bash scripts/run_vla_ablation.sh e0_full_dualcam_bs256


CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS_PER_NODE=4 \
  bash scripts/run_vla_two_stage_fm.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_NODE=4 \
  STAGE=stage2 \
  bash scripts/run_vla_two_stage_fm.sh



CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS_PER_NODE=4 MASTER_PORT=6072 \
bash scripts/run_vla_ablation.sh e0_full_dualcam_bs256


CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS_PER_NODE=4 MASTER_PORT=6057 \
  bash scripts/run_vla_ablation.sh e0_full_dualcam_bs256



CUDA_DEVICE=1 \
CKPT_PATH=/workspace/tingting/VLM4VLA/runs/vla_two_stage_fm/stage2_lora/checkpoints/qwen25vl/vla_two_stage_fm_stage2/2026-06-25/vla_two_stage_fm_stage2_Qwen2.5-VL-3B-Instruct-bs256-lr8e-05-ws1-FMDecoder-latent1/best-val-2-4001-0.2181.ckpt \
USE_WANDB=0 \
NUM_TRIALS=10 \
TASK_IDS=2,3 \
EXECUTE_STEP=4 \
bash scripts/run_vla_ablation_eval.sh fm_dualcam_stage2_lora libero_10

CKPT_PATH="/workspace/tingting/VLM4VLA/runs/vla_two_stage_fm/stage2/checkpoints/qwen25vl/vla_two_stage_fm_stage2/2026-06-24/vla_two_stage_fm_stage2_Qwen2.5-VL-3B-Instruct-bs256-lr8e-05-ws1-FMDecoder-latent1/epoch=15-step=24000.ckpt" \
CUDA_DEVICE=2 \
bash scripts/run_vla_ablation_eval.sh fm_dualcam_stage2_lora libero_10



CKPT="/workspace/tingting/VLM4VLA/runs/vla_ablation_rlds/e0_full_dualcam_bs256/checkpoints/qwen25vl/vla_ablation_e0_full_dualcam_bs256_rlds/2026-06-20/vla_ablation_e0_full_dualcam_bs256_rlds_Qwen2.5-VL-3B-Instruct-bs256-lr8e-05-ws1-FCDecoder-latent1/epoch=31-step=50000.ckpt"
LIBERO_USE_TRAIN_INIT=1 EXECUTE_STEP=1 NUM_TRIALS=5 TASK_IDS=0 USE_WANDB=0 CUDA_DEVICE=3 MUJOCO_GL=osmesa \
CKPT_PATH="$CKPT" \
  bash scripts/run_vla_ablation_eval.sh e0_full_dualcam_bs256 libero_10 

cd /workspace/tingting/3DBENCH
CUDA_VISIBLE_DEVICES=7 MASTER_PORT=6056 bash scripts/run_vla_ablation.sh e0_full_dualcam



cd /workspace/tingting/3DBENCH
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS_PER_NODE=4 MASTER_PORT=6055 \
  bash scripts/run_vla_ablation.sh e0_full_bs256



cd /workspace/tingting/3DBENCH
CUDA_VISIBLE_DEVICES=6,7 GPUS_PER_NODE=2 MASTER_PORT=6054 \
  bash scripts/run_vla_ablation.sh e0_full_bs256



cd /workspace/tingting/3DBENCH
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS_PER_NODE=4 MASTER_PORT=6053 \
  bash scripts/run_vla_ablation.sh e0_full_bs256_4gpu


cd /workspace/tingting/3DBENCH
CUDA_VISIBLE_DEVICES=6,7 GPUS_PER_NODE=2 MASTER_PORT=6052 \
  bash scripts/run_vla_ablation.sh e0_full_bs128_2gpu





CONFIG_DIR=$PWD/configs/vla_ablation_fm \
CUDA_VISIBLE_DEVICES=1 MASTER_PORT=6051 \
bash ./scripts/run_vla_ablation.sh e1_full

```




---

## 数据流总览

```
  阶段 1                  阶段 2                  阶段 3                 阶段 4
  ┌──────────┐           ┌──────────────┐        ┌──────────┐          ┌──────────────┐
  │ LIBERO   │ → GT 抽取 │ data/gt-q6-mv│ → 评测 │data/runs-mv│ ──┐    │data/finetune-│
  │ sim      │           │ (1878样本,   │        │(7模型×    │   │    │mv (6 dim ×   │
  │ (4 suite)│           │ agent+wrist) │        │ baseline) │   │    │ 1504/189/185)│
  └──────────┘           └──────────────┘        └──────────┘   ↓    └──────────────┘
                                ↓                                ↓             ↓
                                ↓                          阶段3准备数据    阶段4训练
                                ↓                                            ↓
                                ↓                              ┌──────────────────────────┐
                                ↓                              │ models/qwen2.5-vl-3b-mv- │
                                ↓                              │ lora/{q1..q6}/final/     │
                                ↓                              └──────────────────────────┘
                                ↓                                            ↓
                                ↓                                            ↓ 阶段6 加载 LoRA
                                ↓                                            ↓
                          阶段5: CALVIN                                  阶段6评测
                          ┌──────────────────┐         ┌──────────────────────┐
                          │ data/calvin_data │ → 抽取  │data/gt-q6-mv-calvin  │
                          │ /task_D_D/       │         │ (~500-800样本)        │
                          └──────────────────┘         └──────────────────────┘
                                                                  ↓
                                                          阶段6 LoRA × CALVIN
                                                          ┌────────────────────┐
                                                          │ data/runs-mv-calvin│
                                                          └────────────────────┘
```

---

## 阶段 1：双视角 GT 抽取（LIBERO）

**前置**：LIBERO 已通过 conda 安装在 `vlmbench` 环境中（已就绪）。

```bash
# 全量抽取 4 suite × ~1878 样本，每条样本 agent + wrist 双 PNG
python scripts/01_extract_gt.py \
    --suite all \
    --n_states 5 \
    --n_traj_frames 8 \
    --n_close 3 \
    --out_dir data/gt-q6-mv
```

**参数**：
- `--n_states 5`：每 task 5 个 init state
- `--n_traj_frames 8`：每 init state 沿 P-controller 轨迹采 8 帧
- `--n_close 3`：每 task 额外 3 个 close 帧（gripper 离 target < 4 cm）
- `--suite all`：4 个 suite 全跑

**预期输出**：

```
data/gt-q6-mv/
  manifest.json                                   # 顶层（合并后）
  libero_10/{manifest.json, task_<NN>/...}        # 466 样本
  libero_goal/...                                 # 463
  libero_object/...                               # 479
  libero_spatial/...                              # 470
  TOTAL: 1878 样本 × 2 PNG = 3756 张图
```

**耗时**：约 30–45 min（CPU + 简单渲染，不需要 GPU）。

### （可选）补抓 close 帧

部分 task（articulation / 容器内目标）原始 P-controller 抓不到 close 帧。如需补抓 pick_and_place 任务的缺口：

```bash
# 先 dry-run 看缺多少
python scripts/01c_supplement_close_frames.py \
    --gt_root data/gt-q6-mv --target_close 5 --dry_run

# 实际补抓（更激进：6 attempt × gain=0.5 × 600 步）
python scripts/01c_supplement_close_frames.py \
    --gt_root data/gt-q6-mv \
    --target_close 5 \
    --max_init_attempts 6 \
    --close_max_steps 600 \
    --approach_gain 0.5
```

注意：articulation 任务（drawer/stove）和容器内目标（top drawer / on top of cabinet）从几何上无法靠 P-controller 满足 dist<4cm 阈值，这部分 task 不会有 close 帧——这是 GT 定义的内在限制，不是 bug。

### 合并顶层 manifest（如果只跑了 per-suite）

```bash
python -c "
import json, glob
out = []
for mf in sorted(glob.glob('data/gt-q6-mv/*/manifest.json')):
    out.extend(json.load(open(mf)))
for i, rec in enumerate(out):
    rec['sample_id'] = i
json.dump(out, open('data/gt-q6-mv/manifest.json','w'), indent=2)
print('merged', len(out), 'samples')
"
```

---

## 阶段 2：双视角 baseline 评测（11 模型）

**前置**：阶段 1 完成；本地模型在 `/umd-datapool/tingting/models/`（6 个 Qwen 已下好；PaliGemma/Kosmos-2/InternVL2 需要 `snapshot_download`，本次先跳过）。

### 推荐：单次调用，串行跑所有模型

```bash
mkdir -p logs
nohup python scripts/02_run_vlm_eval.py \
    --manifest data/gt-q6-mv/manifest.json \
    --out_dir data/runs-mv \
    --models qwen2.5-vl-3b qwen2.5-vl-7b qwen3-vl-2b qwen3-vl-4b qwen3-vl-8b qwen3-vl-30b-a3b random \
    --device cuda:7 --resume \
    > logs/baseline_mv_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "started PID $!"
# 看进度： tail -f logs/baseline_mv_*.log
```

每个模型加载一次跑完所有 1878 样本，再 `_free_model()` 清显存换下一个。

### 多 GPU 并行（更快）

```bash
# 大模型 → cuda:7
nohup python scripts/02_run_vlm_eval.py \
    --manifest data/gt-q6-mv/manifest.json --out_dir data/runs-mv --resume \
    --models qwen2.5-vl-7b qwen3-vl-8b qwen3-vl-30b-a3b \
    --device cuda:7 > logs/baseline_big.log 2>&1 &

# 小模型 → cuda:1 (并行)
nohup python scripts/02_run_vlm_eval.py \
    --manifest data/gt-q6-mv/manifest.json --out_dir data/runs-mv --resume \
    --models qwen2.5-vl-3b qwen3-vl-2b qwen3-vl-4b random \
    --device cuda:1 > logs/baseline_small.log 2>&1 &
```

`--resume` 让两边互不冲突（每个模型各自的目录）。

### 计算 metrics

```bash
python scripts/03_compute_metrics.py \
    --manifest data/gt-q6-mv/manifest.json \
    --responses_dir data/runs-mv \
    --out data/results-mv-all.json \
    --report data/runs-mv-results.md

python scripts/04_aggregate_results.py \
    --results data/results-mv-all.json \
    --report data/results-mv-summary.md
```

**预期输出**：
- `data/runs-mv/<model>/<timestamp>/<suite>/sample_NNNN.json` — 每个模型每条样本一份响应 JSON
- `data/results-mv.json` — per-model × per-question 的 MAE/F1/cosine
- `data/results-mv-summary.md` — 排版好的对比表

**耗时**（单卡 cuda:7 串行）：
- 3B/2B/4B：每模型 ~2.5 h
- 7B/8B：每模型 ~4 h
- 30B-A3B：~9 h
- random：< 1 min
- **总计：~25–28 h** （多 GPU 并行可砍半）

---

## 阶段 3：分维度训练数据准备

**前置**：阶段 1 的 `data/gt-q6-mv/<suite>/manifest.json` 已生成。

```bash
python scripts/05_prepare_finetune_data.py \
    --manifest_glob 'data/gt-q6-mv/*/manifest.json' \
    --out_dir data/finetune-mv \
    --val_task 8 \
    --test_task 9
```

**参数**：
- `--val_task 8 / --test_task 9`：每 suite 留 task_id=8 做 val、task_id=9 做 test。其余 task 进 train。这种切分避免 init/traj/close 帧泄漏。
- `--float_decimals_m 3 / --float_decimals_unit 2`：assistant 答案的浮点精度。

**预期输出**：

```
data/finetune-mv/
  q1/{train,val,test}.jsonl    # 1504 / 189 / 185
  q2/{train,val,test}.jsonl
  q3/{train,val,test}.jsonl
  q4/{train,val,test}.jsonl
  q5/{train,val,test}.jsonl
  q6/{train,val,test}.jsonl
  split_info.json              # 记录每 suite 的 train/val/test task 分配
```

每 dim 的 train.jsonl 每行：
```json
{"sample_id": 30, "suite": "libero_10", "task_id": 0,
 "image_paths": {"agent": "...", "wrist": "..."},
 "system": "<dim-specific stripped system prompt>",
 "user": "<dim-specific user prompt with task description>",
 "assistant": "<JSON answer for this dim only>"}
```

---

## 阶段 4：分维度 LoRA 微调

**前置**：peft 已装入 `vlmbench`（peft 0.19.1）；阶段 3 的 jsonl 已生成；GPU 显存 ≥ 35 GB（单 L20 够）。

### 训练所有 6 个 adapter

```bash
# 串行训 q1..q6（默认）
CUDA_VISIBLE_DEVICES=0 bash run_finetune_all.sh

# 训子集
DIMS="q3 q6" CUDA_VISIBLE_DEVICES=0 bash run_finetune_all.sh

# 改 epochs / lr
EPOCHS=2 LR=5e-5 CUDA_VISIBLE_DEVICES=0 bash run_finetune_all.sh
```

环境变量（可选）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `DIMS` | `q1 q2 q3 q4 q5 q6` | 训哪些 dim |
| `EPOCHS` | `3` | 每 dim epoch 数 |
| `LR` | `1e-4` | LoRA learning rate |
| `RANK` | `16` | LoRA rank |
| `ALPHA` | `32` | LoRA alpha |
| `GRAD_ACCUM` | `8` | 梯度累积 |
| `PER_DEV_BATCH` | `1` | per-GPU batch |
| `BASE_MODEL` | `/umd-datapool/tingting/models/Qwen2.5-VL-3B-Instruct` | 基座路径 |

### 单 dim 训练（细粒度控制）

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/06_finetune_qwen.py \
    --dim q3 \
    --train_jsonl data/finetune-mv/q3/train.jsonl \
    --val_jsonl   data/finetune-mv/q3/val.jsonl \
    --data_root   data \
    --base_model  /umd-datapool/tingting/models/Qwen2.5-VL-3B-Instruct \
    --output_dir  models/qwen2.5-vl-3b-mv-lora/q3 \
    --epochs 3 --lr 1e-4 --lora_rank 16 \
    --per_device_batch_size 1 --grad_accum 8
```

### Smoke test（3 步）

```bash
python scripts/06_finetune_qwen.py --dim q3 \
    --train_jsonl /tmp/finetune_mini/q3/train.jsonl \
    --val_jsonl   /tmp/finetune_mini/q3/val.jsonl \
    --output_dir  /tmp/lora_smoke_q3 \
    --max_steps 3 --grad_accum 1
```

**预期输出**：

```
models/qwen2.5-vl-3b-mv-lora/
  q1/final/
    adapter_config.json
    adapter_model.safetensors    (~149 MB, rank=16, bf16)
    adapter_meta.json            { "dim": "q1", ... }
    chat_template.jinja, tokenizer.json, ...
  q2/final/
  ...
  q6/final/
```

**耗时**：约 16 min/dim × 6 dim = **~1.5 GPU-hour 总计**（单 L20，bf16，effective batch=8，3 epoch × 188 步）。

**冻结部分**：vision encoder + merger 全冻；LoRA 仅作用于 LLM 的 q/k/v/o/gate/up/down 投影。可训参数 ≈ 37 M（约 1% 总参数）。

---

## 阶段 5：CALVIN 双视角 QA 抽取

**前置**：CALVIN `task_D_D.zip` 已下载并解压到 `/umd-datapool/tingting/calvin_data/task_D_D/`。

### 下载 CALVIN

```bash
mkdir -p /umd-datapool/tingting/calvin_data
cd /umd-datapool/tingting/calvin_data

# 177 GB，建议后台下载
nohup wget -c "http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip" \
    > /tmp/calvin_download.log 2>&1 &

# 解压（解压后 ~250 GB）
unzip task_D_D.zip
# 解压后目录结构：
#   task_D_D/training/episode_NNNNNNN.npz + lang_annotations/
#   task_D_D/validation/episode_NNNNNNN.npz + lang_annotations/
```

### 抽取 QA 数据

```bash
# Smoke：5 个标注窗口
python scripts/01b_extract_gt_calvin.py \
    --data_root /umd-datapool/tingting/calvin_data/task_D_D \
    --split validation \
    --max_windows 5 \
    --out_dir /tmp/calvin_smoke

# 全量（验证集所有标注窗口，约 200 个 → 500–800 样本）
python scripts/01b_extract_gt_calvin.py \
    --data_root /umd-datapool/tingting/calvin_data/task_D_D \
    --split validation \
    --out_dir data/gt-q6-mv-calvin \
    --n_traj_frames 3 \
    --include_close
```

**参数**：
- `--n_traj_frames 3`：每标注窗口采 3 个 traj 帧
- `--include_close`：取窗口末帧作 close 候选（与 LIBERO 一致）
- `--max_windows`：smoke 时限制窗口数

**预期输出**（mirror LIBERO 结构）：

```
data/gt-q6-mv-calvin/
  manifest.json
  calvin_D/
    manifest.json
    task_<NN>/                  # task class index, 0..~26
      window_<MMMM>/
        sample_NNNN_tNN_wMMMM_init_agent.png  + _wrist.png + .json
        sample_NNNN_tNN_wMMMM_traj_001of003_*.png
        ...
        sample_NNNN_tNN_wMMMM_close_*.png
```

**与 LIBERO 的差异**：
- 坐标系：CALVIN 用世界系（不是 base-relative），prompt 也对应替换为 `prompts/spatial_qa_calvin.yaml`
- Articulation 目标位置来自硬编码（`src/bench/calvin_taxonomy.py` 中 `ARTICULATED_POSITIONS`）
- 多义任务（"stack the blocks"）通过 deny-list 排除

---

## 阶段 6：交叉评测（base / LoRA × LIBERO / CALVIN）

**前置**：阶段 4 的 6 个 LoRA adapter；阶段 5 的 CALVIN 数据。

### Base 模型在两个数据集上跑

```bash
# LIBERO test 集（阶段 3 切分的 task 9 holdout）

python scripts/02_run_vlm_eval.py \
    --manifest data/gt-q6-mv-test/manifest.json \
    --out_dir data/base-libero-test-0508 \
    --models qwen2.5-vl-3b \
    --device cuda:3 \
    --resume


# CALVIN（用 calvin 专属 prompt）
python scripts/02_run_vlm_eval.py \
    --manifest data/gt-q6-mv-calvin/manifest.json \
    --out_dir data/runs-mv-calvin \
    --models qwen2.5-vl-3b \
    --prompt_yaml prompts/spatial_qa_calvin.yaml \
    --device cuda:0 --resume


# base × CALVIN（用修正后的全 6 题 prompt）
python scripts/02_run_vlm_eval.py \
    --manifest data/gt-q6-mv-calvin/manifest.json \
    --out_dir  data/runs-mv-base-calvin-0509-fixed \
    --models   qwen2.5-vl-3b \
    --prompt_yaml prompts/spatial_qa_calvin.yaml \
    --device cuda:3


```

### LoRA-augmented 模型（每个 dim 一次）

```bash
# 在 LIBERO 上
for dim in q1 q2 q3 q4 q5 q6; do
    python scripts/02_run_vlm_eval.py \
        --manifest data/gt-q6-mv-test/manifest.json \
        --out_dir data/runs-mv-lora-libero-allsuits \
        --models qwen2.5-vl-3b-mv-lora \
        --lora_dir models/$dim/final \
        --lora_dim $dim \
        --device cuda:3 --resume
done



# 在 CALVIN 上

# LoRA × CALVIN（用修正后的 per-dim CALVIN prompt）

for dim in q1 q2 q3 q4 q5 q6; do
    python scripts/02_run_vlm_eval.py \
        --manifest data/gt-q6-mv-calvin/manifest.json \
        --out_dir  data/runs-mv-lora-calvin-0509-fixed \
        --models   qwen2.5-vl-3b-mv-lora \
        --lora_dir models/$dim/final \
        --lora_dim $dim \
        --prompt_yaml prompts/spatial_qa_per_dim_calvin.yaml \
        --device cuda:3
done


```

# 重新跑








# 重生成对比报告
python scripts/07_compare_base_vs_lora_calvin.py


LoRA 模式下脚本自动：
- 加载 base + adapter，调用 `merge_and_unload()` 合并
- 用 `PerDimPromptBuilder(dim)` 替换默认 PromptBuilder（短 prompt，只问该 dim）
- 输出 JSON 只含该 dim 的 keys（如 `{"q3": {"can_close": "no"}}`）

### 计算指标 + 汇总

```bash
# LIBERO 端
python scripts/03_compute_metrics.py \
    --manifest data/gt-q6-mv/manifest.json \
    --responses_root data/runs-mv-lora \
    --out_file data/results-mv-lora.json

# CALVIN 端
python scripts/03_compute_metrics.py \
    --manifest data/gt-q6-mv-calvin/manifest.json \
    --responses_root data/runs-mv-lora-calvin \
    --out_file data/results-mv-lora-calvin.json

python scripts/04_aggregate_results.py \
    --results data/results-mv.json data/results-mv-lora.json data/results-mv-calvin.json data/results-mv-lora-calvin.json \
    --out_file data/results-mv-final-summary.md
```

最终对比矩阵：

| 模型 | LIBERO test (task 9 × 4 suite ≈ 185) | CALVIN gt-q6-mv-calvin (~500-800) |
|---|---|---|
| Qwen2.5-VL-3B base | ✓ | ✓ |
| + Q1 LoRA | ✓ (Q1/Q1_dest) | ✓ |
| + Q2 LoRA | ✓ (Q2) | ✓ |
| ... | ... | ... |
| + Q6 LoRA | ✓ (Q6) | ✓ |

迁移率 = (CALVIN gain) / (LIBERO gain)。

---

## 验证（端到端 smoke 链路）

按这个顺序逐级验证，可在 5 分钟内确认全 pipeline 跑通：

```bash
# 1) GT 抽取（单 task 单 init）
python scripts/01_extract_gt.py --suite libero_10 --task_ids 0 --n_states 1 \
    --n_traj_frames 0 --n_close 0 --out_dir /tmp/gt_smoke

# 2) 双图 eval (random + qwen2.5-vl-3b)
python scripts/02_run_vlm_eval.py --models random qwen2.5-vl-3b \
    --manifest /tmp/gt_smoke/libero_10/manifest.json --out_dir /tmp/runs_smoke \
    --device cuda:0

# 3) Finetune 数据准备
python scripts/05_prepare_finetune_data.py \
    --manifest_glob '/tmp/gt_smoke/libero_10/manifest.json' \
    --out_dir /tmp/ft_smoke

# 4) LoRA 训练 (3 步)
python scripts/06_finetune_qwen.py --dim q3 \
    --train_jsonl /tmp/ft_smoke/q3/train.jsonl \
    --val_jsonl   /tmp/ft_smoke/q3/val.jsonl \
    --output_dir  /tmp/lora_smoke --max_steps 3 --grad_accum 1

# 5) LoRA eval
python scripts/02_run_vlm_eval.py \
    --manifest /tmp/gt_smoke/libero_10/manifest.json \
    --out_dir /tmp/runs_lora_smoke \
    --models qwen2.5-vl-3b-mv-lora \
    --lora_dir /tmp/lora_smoke/final --device cuda:0




python scripts/07_compare_base_vs_lora_calvin.py \
    --base_root  data/runs-mv-base-calvin \
    --lora_root  data/runs-mv-lora-calvin \
    --manifest   data/gt-q6-mv-calvin/manifest.json \
    --out_md     data/results-base-vs-lora-calvin.md \
    --out_json   data/results-base-vs-lora-calvin.json
    
```

---

## 关键路径与文件清单

### 修改的文件

| 文件 | 用途 |
|---|---|
| [scripts/01_extract_gt.py](../scripts/01_extract_gt.py) | LIBERO GT 抽取（双视角） |
| [scripts/02_run_vlm_eval.py](../scripts/02_run_vlm_eval.py) | VLM 评测（双图 + LoRA） |
| [src/bench/gt_extractor.py](../src/bench/gt_extractor.py) | + `get_wrist_image` / `get_views` |
| [src/bench/prompt_builder.py](../src/bench/prompt_builder.py) | 全 6 题 prompt（双视角说明） |
| [prompts/spatial_qa.yaml](../prompts/spatial_qa.yaml) | + 双视角 system prompt 段 |

### 新增的文件

| 文件 | 阶段 | 作用 |
|---|---|---|
| [scripts/01b_extract_gt_calvin.py](../scripts/01b_extract_gt_calvin.py) | 5 | CALVIN 双视角 GT 抽取 |
| [scripts/01c_supplement_close_frames.py](../scripts/01c_supplement_close_frames.py) | 1 | 补抓 close 帧 |
| [scripts/05_prepare_finetune_data.py](../scripts/05_prepare_finetune_data.py) | 3 | manifest → 6× JSONL |
| [scripts/06_finetune_qwen.py](../scripts/06_finetune_qwen.py) | 4 | 单 dim LoRA 训练 |
| [run_finetune_all.sh](../run_finetune_all.sh) | 4 | 6 dim 串行 driver |
| [src/bench/per_dim_prompt_builder.py](../src/bench/per_dim_prompt_builder.py) | 3 | dim-aware prompt |
| [src/bench/calvin_loader.py](../src/bench/calvin_loader.py) | 5 | CALVIN npz / lang 加载 |
| [src/bench/calvin_taxonomy.py](../src/bench/calvin_taxonomy.py) | 5 | 任务 → 目标查表 |
| [src/bench/calvin_gt_extractor.py](../src/bench/calvin_gt_extractor.py) | 5 | CALVIN 6 维标签计算 |
| [prompts/spatial_qa_per_dim.yaml](../prompts/spatial_qa_per_dim.yaml) | 3 | 6 个 stripped per-dim prompt |
| [prompts/spatial_qa_calvin.yaml](../prompts/spatial_qa_calvin.yaml) | 5 | CALVIN 世界系 prompt |

### 数据目录

| 路径 | 内容 | 大小 |
|---|---|---|
| `data/gt-q6-mv/` | LIBERO 4 suite 双视角 GT | 1878 样本 × 2 PNG ≈ 数百 MB |
| `data/finetune-mv/` | 6 dim × {train,val,test}.jsonl | ~50 MB |
| `models/qwen2.5-vl-3b-mv-lora/` | 6 个 LoRA adapter | 6 × 149 MB ≈ 900 MB |
| `data/gt-q6-mv-calvin/` | CALVIN QA 评测集 | ~500–800 样本 ≈ 100 MB |
| `data/runs-mv/` | baseline 响应 | 7 模型 × 1878 样本 |
| `data/runs-mv-lora/` | LoRA 响应 | 6 dim × 1878 样本 |

### 配置

- 环境：`/umd-datapool/tingting/envs/vlmbench`（conda）
- 模型缓存：`/umd-datapool/tingting/models/`（6 个 Qwen 已下）
- HF cache：`/umd-datapool/tingting/hf-home/`
- CALVIN 数据：`/umd-datapool/tingting/calvin_data/task_D_D/`

---

## 常见问题

### Q: GPU OOM？
A: 双图 + 30B-A3B 显存吃紧。降 `--max_new_tokens 256` 或换更大显存卡（cuda:1/7 通常空）。LoRA 训练时降 `RANK=8` 或 `GRAD_ACCUM=16`。

### Q: 多个模型在同一卡跑会冲突？
A: `_free_model()` 在每模型结束时清显存，串行跑安全。多卡并行需要不同 `--device cuda:N` + 不同输出目录（或 `--resume` 兼容）。

### Q: 修改了 prompt 想重跑某个模型？
A: 删除对应的 `data/runs-mv/<model>/<timestamp>/` 子目录或 `data/runs-mv/<model>/latest` 软链接，再用 `--resume` 跑（resume 只跳过已存在的 sample 文件）。

### Q: LIBERO env reset 不确定，重抽 GT 数值跟旧版差别大？
A: 现在的 `data/gt-q6-mv/` 跟旧 `data/gt-q6/` 在 GT 数值上应该完全一致（LIBERO env 是 deterministic）。close 帧数偶有差异是因为 P-controller 步数限制，不影响 init/traj 帧的数值。

### Q: 怎么把 base 和 LoRA 的响应放进同一个 `03_compute_metrics.py` 报告？
A: `03_compute_metrics.py` 接受 `--responses_root`，下面会枚举所有子目录（即所有 model slug）。把 LoRA 响应也存进 `data/runs-mv/qwen2.5-vl-3b-mv-lora-q3/...` 即可一次跑出对比。


