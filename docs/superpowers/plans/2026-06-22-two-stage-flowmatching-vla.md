# 两阶段 Flow-Matching VLA 训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 RLDS dual-cam 单阶段全量微调流程之上，新增 Evo-1 式「两阶段训练 + Flow-Matching action head」的可复现方案（stage1 冻 VLM 只训 FMDecoder，stage2 加载 stage1 权重并重置 optimizer 后全量微调）。

**Architecture:** 复用现有 `RoboQwen25VL` + 已实现的 `FMDecoder`，模型架构零改动。新增物：2 个 RLDS-dualcam FM 配置（stage1/stage2）、`main.py` 的 `resume_pretrain` 校验+日志、`FMDecoder` 内部 gripper {0,1}↔{-1,+1} 同质化、一个双阶段编排脚本。stage1→stage2 衔接靠「`model_load_path`=stage1 ckpt + `resume=null`」让 Lightning 重置 optimizer/scheduler。

**Tech Stack:** PyTorch + PyTorch Lightning（`Trainer.fit`）、torchrun DDP、Qwen2.5-VL-3B backbone、RLDS（TFDS）数据、conda env `vlmbench-rlds`。注：环境中无 `jq`，配置注入统一用 conda env 的 python（`json` 模块）。

## Global Constraints

- 工作目录：训练代码在 `/workspace/tingting/VLM4VLA`，配置/脚本/文档在 `/workspace/tingting/3DBENCH`。
- conda env：`conda run -p /workspace/tingting/envs/vlmbench-rlds`。
- 数据：RLDS dual-cam，`data_root_dir=/workspace/tingting/LIBERO_rlds`，`data_mix=libero_10_no_noops`，`use_hand_rgb=true`。
- 归一化：`norm_action: false`（RLDS 上游 BOUNDS_Q99 已做）；`norm_min/max` 为死参数仅作占位。
- 动作维度：7 维 `[dx,dy,dz,drx,dry,drz,gripper]`，`fwd_pred_next_n=4`。
- `arm_gripper_loss_ratio` 对 FM 路径无效（FMDecoder.loss 是全 7 维 velocity MSE，`loss_gripper≡0`）；配置中保留默认 `1`，**不依赖、不调**。
- gripper 同质化只在 `FMDecoder` 内部做，**禁止改动 FCDecoder 路径**。
- 冒烟测试用 GPU 4,5,6,7（0-3 被他人占用）。
- FCDecoder loss/forward 不得改动。
- 提交频繁；当前工作分支 `feat/two-stage-flowmatching-vla`（spec 已在此分支）。

---

## File Structure

新增：
- `VLM4VLA/vlm4vla/model/policy_head/fm_decoder.py` — 修改：加 gripper 映射 helper + 3 处调用
- `VLM4VLA/main.py` — 修改：`experiment()` 加 `resume_pretrain` 校验+日志
- `3DBENCH/configs/vla_ablation_rlds/fm_dualcam_stage1.json` — 新建
- `3DBENCH/configs/vla_ablation_rlds/fm_dualcam_stage2.json` — 新建
- `3DBENCH/scripts/run_vla_two_stage_fm.sh` — 新建
- `VLM4VLA/tests/test_fm_gripper_mapping.py` — 新建（单元测试）
- `3DBENCH/tests/test_two_stage_fm_configs.py` — 新建（配置校验测试）

---

## Task 1: FMDecoder gripper 同质化（{0,1}↔{-1,+1}）

让 FMDecoder 内部 7 维全为 [-1,1] 连续量，训练时 gripper {0,1}→{-1,+1}，推理输出 [-1,1]→[0,1]。FCDecoder 不受影响。

**Files:**
- Modify: `/workspace/tingting/VLM4VLA/vlm4vla/model/policy_head/fm_decoder.py`（forward `:165-168`、get_labels `:243-246`、predict return `:311`）
- Test: `/workspace/tingting/VLM4VLA/tests/test_fm_gripper_mapping.py`

**Interfaces:**
- Produces:
  - `FMDecoder._gripper_to_continuous(g: Tensor) -> Tensor`（静态方法，`{0,1}->{-1,+1}`，公式 `g*2-1`）
  - `FMDecoder._gripper_to_unit(g: Tensor) -> Tensor`（静态方法，`[-1,1]->[0,1]`，公式 `(g+1)*0.5`）
  - forward / get_labels：concat 前对 gripper 调 `_gripper_to_continuous`
  - predict：return 前对 gripper 维调 `_gripper_to_unit`

- [ ] **Step 1: 写失败测试**

创建 `/workspace/tingting/VLM4VLA/tests/test_fm_gripper_mapping.py`：

```python
import torch
from vlm4vla.model.policy_head.fm_decoder import FMDecoder


def test_gripper_to_continuous_maps_0_1_to_neg1_pos1():
    g = torch.tensor([0.0, 1.0, 0.0, 1.0])
    out = FMDecoder._gripper_to_continuous(g)
    assert torch.allclose(out, torch.tensor([-1.0, 1.0, -1.0, 1.0]))


def test_gripper_to_unit_maps_neg1_pos1_to_0_1():
    g = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    out = FMDecoder._gripper_to_unit(g)
    assert torch.allclose(out, torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_gripper_roundtrip_is_identity():
    g = torch.tensor([0.0, 1.0, 0.0, 1.0])
    back = FMDecoder._gripper_to_unit(FMDecoder._gripper_to_continuous(g))
    assert torch.allclose(back, g)
```

- [ ] **Step 2: 运行测试确认失败**

Run:
```bash
cd /workspace/tingting/VLM4VLA && conda run -p /workspace/tingting/envs/vlmbench-rlds \
  python -m pytest tests/test_fm_gripper_mapping.py -v
```
Expected: FAIL，`AttributeError: ... has no attribute '_gripper_to_continuous'`

- [ ] **Step 3: 加两个静态 helper**

在 `fm_decoder.py` 的 `FMDecoder` 类内（紧跟 `__init__` 之后、`sample_time` 之前，约 `:95` 后）插入：

```python
    @staticmethod
    def _gripper_to_continuous(g: torch.Tensor) -> torch.Tensor:
        """Map gripper {0,1} -> {-1,+1} so all 7 dims share the [-1,1] scale."""
        return g * 2.0 - 1.0

    @staticmethod
    def _gripper_to_unit(g: torch.Tensor) -> torch.Tensor:
        """Map continuous gripper [-1,1] -> [0,1] for the downstream interface."""
        return (g + 1.0) * 0.5
```

- [ ] **Step 4: 运行测试确认通过**

Run:
```bash
cd /workspace/tingting/VLM4VLA && conda run -p /workspace/tingting/envs/vlmbench-rlds \
  python -m pytest tests/test_fm_gripper_mapping.py -v
```
Expected: 3 passed

- [ ] **Step 5: 在 forward 中映射 gripper（concat 前）**

把 `fm_decoder.py:165-168` 由：

```python
        arm, gripper = actions  # shapes: (bs, seq, T, 6), (bs, seq, T)
        if gripper.ndim == arm.ndim:
            gripper = gripper[..., -1]
        full_actions = torch.cat([arm, gripper.unsqueeze(-1)], dim=-1)
```

改为：

```python
        arm, gripper = actions  # shapes: (bs, seq, T, 6), (bs, seq, T)
        if gripper.ndim == arm.ndim:
            gripper = gripper[..., -1]
        gripper = self._gripper_to_continuous(gripper)  # {0,1} -> {-1,+1} (7-dim homogeneous)
        full_actions = torch.cat([arm, gripper.unsqueeze(-1)], dim=-1)
```

- [ ] **Step 6: 在 get_labels 中映射 gripper（concat 前）**

把 `fm_decoder.py:243-246` 由：

```python
        arm, gripper = labels
        if gripper.ndim == arm.ndim:
            gripper = gripper[..., -1]
        full_actions = torch.cat([arm, gripper.unsqueeze(-1)], dim=-1)
```

改为：

```python
        arm, gripper = labels
        if gripper.ndim == arm.ndim:
            gripper = gripper[..., -1]
        gripper = self._gripper_to_continuous(gripper)  # {0,1} -> {-1,+1}, match forward
        full_actions = torch.cat([arm, gripper.unsqueeze(-1)], dim=-1)
```

- [ ] **Step 7: 在 predict 返回前反映射 gripper**

把 `fm_decoder.py:311` 由：

```python
        return actions[...,:6],actions[...,-1]  # Return (arm, gripper)
```

改为：

```python
        gripper_unit = self._gripper_to_unit(actions[..., -1])  # [-1,1] -> [0,1]
        return actions[..., :6], gripper_unit  # Return (arm, gripper)
```

- [ ] **Step 8: 回归测试 helper + import 无破坏**

Run:
```bash
cd /workspace/tingting/VLM4VLA && conda run -p /workspace/tingting/envs/vlmbench-rlds \
  python -m pytest tests/test_fm_gripper_mapping.py -v && \
  conda run -p /workspace/tingting/envs/vlmbench-rlds \
  python -c "from vlm4vla.model.policy_head.fm_decoder import FMDecoder; print('import OK')"
```
Expected: 3 passed；`import OK`

- [ ] **Step 9: 提交**

```bash
cd /workspace/tingting/VLM4VLA && git add vlm4vla/model/policy_head/fm_decoder.py tests/test_fm_gripper_mapping.py
git commit -m "feat(fm): homogenize gripper {0,1}<->{-1,+1} inside FMDecoder

Map gripper to [-1,1] before flow-matching concat (forward + get_labels)
and back to [0,1] at predict output, so all 7 action dims share one scale
for a fair velocity MSE. FCDecoder path untouched.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `main.py` 的 `resume_pretrain` 校验+日志

让 stage2「只取 stage1 权重、重置 optimizer/scheduler」的语义显式化。不改加载逻辑（仍走现成 `from_checkpoint`），只加校验+日志。

**Files:**
- Modify: `/workspace/tingting/VLM4VLA/main.py:164`（`experiment()` 内，`model_load_path` 赋值之后）

**Interfaces:**
- Consumes: config 键 `resume_pretrain: bool`、`model_load_path: str|null`、`resume: str|null`、`model_load_source: str`
- Produces: 运行时校验（断言）+ 一行日志；无新函数签名

- [ ] **Step 1: 插入校验+日志**

在 `main.py` 的 `experiment()` 中，把 `:164`：

```python
    model_load_path = variant.get("model_load_path", None)
```

改为：

```python
    model_load_path = variant.get("model_load_path", None)

    # Evo-1-style stage1->stage2 handoff: load weights only, reset optimizer/scheduler.
    resume_pretrain = variant.get("resume_pretrain", False)
    if resume_pretrain:
        assert model_load_path, \
            "resume_pretrain=True requires model_load_path pointing to the stage1 checkpoint"
        assert variant.get("resume") is None, \
            "resume_pretrain is mutually exclusive with Lightning resume (which restores optimizer/scheduler)"
        print(f"[resume_pretrain] loading weights only from {model_load_path}; "
              f"optimizer/scheduler reset, warmup restarts from step 0")
```

> 说明：实际加载在 `:193` 的 `BaseTrainer.from_checkpoint(model_load_path, variant.get("model_load_source", "torch"), variant)` 完成（`strict=False`，含 head）；`trainer.fit` 因 `resume=null` 自然新建 optimizer+scheduler。本 Task 不改这两处。

- [ ] **Step 2: 校验 resume_pretrain=False 时不报错（语法+逻辑冒烟）**

Run:
```bash
cd /workspace/tingting/VLM4VLA && conda run -p /workspace/tingting/envs/vlmbench-rlds \
  python -c "
import ast; ast.parse(open('main.py').read()); print('syntax OK')
"
```
Expected: `syntax OK`

- [ ] **Step 3: 用最小变体函数验证断言逻辑**

Run（内联模拟 `resume_pretrain` 校验分支，避免拉起整个训练）：
```bash
cd /workspace/tingting/VLM4VLA && conda run -p /workspace/tingting/envs/vlmbench-rlds \
  python -c "
def check(variant):
    model_load_path = variant.get('model_load_path', None)
    if variant.get('resume_pretrain', False):
        assert model_load_path, 'need model_load_path'
        assert variant.get('resume') is None, 'mutually exclusive'
    return True
# 1) off -> ok
assert check({'resume_pretrain': False})
# 2) on without path -> AssertionError
try:
    check({'resume_pretrain': True}); raise SystemExit('should have failed')
except AssertionError: pass
# 3) on with path + resume set -> AssertionError
try:
    check({'resume_pretrain': True, 'model_load_path': 'x', 'resume': 'y'}); raise SystemExit('should have failed')
except AssertionError: pass
# 4) on with path, no resume -> ok
assert check({'resume_pretrain': True, 'model_load_path': 'x'})
print('resume_pretrain validation logic OK')
"
```
Expected: `resume_pretrain validation logic OK`

- [ ] **Step 4: 提交**

```bash
cd /workspace/tingting/VLM4VLA && git add main.py
git commit -m "feat: explicit resume_pretrain validation + log in experiment()

Make the stage1->stage2 handoff explicit: when resume_pretrain=True,
require model_load_path and forbid Lightning resume (which would restore
the optimizer). Loading still flows through from_checkpoint(strict=False);
fit() with resume=null rebuilds optimizer/scheduler and restarts warmup.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: stage1 / stage2 配置文件

以 `e0_full_dualcam_bs256.json` 为模板（保持 RLDS dualcam + Q99 归一化），`act_head` 换 FMDecoder（DiTConfig 取自 `vla_ablation_fm/e0_full.json`）。

**Files:**
- Create: `/workspace/tingting/3DBENCH/configs/vla_ablation_rlds/fm_dualcam_stage1.json`
- Create: `/workspace/tingting/3DBENCH/configs/vla_ablation_rlds/fm_dualcam_stage2.json`
- Test: `/workspace/tingting/3DBENCH/tests/test_two_stage_fm_configs.py`

**Interfaces:**
- Consumes: 无（独立 JSON）
- Produces:
  - stage1：`freeze_backbone=true, train_vision=false, train_text_embedding=false`，`max_steps=5000`，`lr=2e-5`，`warmup_steps=1000`，`resume_pretrain=false`，`model_load_path=null`
  - stage2：`freeze_backbone=false, train_vision=true, train_text_embedding=true`，`max_steps=50000`，`lr=8e-5`，`warmup_steps=800`，`resume_pretrain=true`，`model_load_source=lightning`，`resume=null`，`model_load_path` 占位（脚本运行时填）

- [ ] **Step 1: 写配置校验失败测试**

创建 `/workspace/tingting/3DBENCH/tests/test_two_stage_fm_configs.py`：

```python
import json
import os

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "vla_ablation_rlds")


def _load(name):
    with open(os.path.join(CFG_DIR, name)) as f:
        return json.load(f)


def test_stage1_freezes_vlm_and_trains_fmdecoder():
    c = _load("fm_dualcam_stage1.json")
    ts = c["train_setup"]
    assert ts["freeze_backbone"] is True
    assert ts["train_vision"] is False
    assert ts["train_text_embedding"] is False
    assert c["act_head"]["type"] == "FMDecoder"
    assert c["trainer"]["max_steps"] == 5000
    assert c["learning_rate"] == 2e-05
    assert c["warmup_steps"] == 1000
    assert c["resume_pretrain"] is False
    assert c["model_load_path"] is None
    assert c["norm_action"] is False


def test_stage2_full_finetune_with_resume_pretrain():
    c = _load("fm_dualcam_stage2.json")
    ts = c["train_setup"]
    assert ts["freeze_backbone"] is False
    assert ts["train_vision"] is True
    assert ts["train_text_embedding"] is True
    assert c["act_head"]["type"] == "FMDecoder"
    assert c["trainer"]["max_steps"] == 50000
    assert c["learning_rate"] == 8e-05
    assert c["warmup_steps"] == 800
    assert c["resume_pretrain"] is True
    assert c["model_load_source"] == "lightning"
    assert c["resume"] is None
    assert c["norm_action"] is False


def test_stage1_stage2_act_head_identical():
    a = _load("fm_dualcam_stage1.json")["act_head"]
    b = _load("fm_dualcam_stage2.json")["act_head"]
    assert a == b, "stage1/stage2 must share the exact same FMDecoder/DiTConfig"


def test_dualcam_and_data_match_baseline():
    for name in ("fm_dualcam_stage1.json", "fm_dualcam_stage2.json"):
        c = _load(name)
        assert c["use_hand_rgb"] is True
        assert c["train_dataset"]["type"] == "OpenVLADataset"
        assert c["train_dataset"]["data_mix"] == "libero_10_no_noops"
        assert c["fwd_pred_next_n"] == 4
```

- [ ] **Step 2: 运行测试确认失败**

Run:
```bash
cd /workspace/tingting/3DBENCH && conda run -p /workspace/tingting/envs/vlmbench-rlds \
  python -m pytest tests/test_two_stage_fm_configs.py -v
```
Expected: FAIL（`FileNotFoundError`：配置不存在）

- [ ] **Step 3: 用脚本从模板生成两个配置**

Run（基于模板派生，确保数据/dualcam/归一化与基线一致，只覆盖差异键；act_head 取自 fm 模板）：
```bash
cd /workspace/tingting/3DBENCH && conda run -p /workspace/tingting/envs/vlmbench-rlds python3 - <<'PY'
import json, copy
base = json.load(open("configs/vla_ablation_rlds/e0_full_dualcam_bs256.json"))
fm_act_head = json.load(open("configs/vla_ablation_fm/e0_full.json"))["act_head"]

def derive(stage):
    c = copy.deepcopy(base)
    c["act_head"] = copy.deepcopy(fm_act_head)   # FMDecoder + DiTConfig
    c["arm_gripper_loss_ratio"] = 1               # no-op for FM; keep default
    if stage == 1:
        c["task_name"] = "vla_two_stage_fm_stage1"
        c["learning_rate"] = 2e-05
        c["warmup_steps"] = 1000
        c["train_setup"]["freeze_backbone"] = True
        c["train_setup"]["train_vision"] = False
        c["train_setup"]["train_text_embedding"] = False
        c["model_load_path"] = None
        c["model_load_source"] = "torch"
        c["resume"] = None
        c["resume_pretrain"] = False
        c["trainer"]["max_steps"] = 5000
        root = "runs/vla_two_stage_fm/stage1"
    else:
        c["task_name"] = "vla_two_stage_fm_stage2"
        c["learning_rate"] = 8e-05
        c["warmup_steps"] = 800
        c["train_setup"]["freeze_backbone"] = False
        c["train_setup"]["train_vision"] = True
        c["train_setup"]["train_text_embedding"] = True
        c["model_load_path"] = "RUNTIME_FILLED_BY_SCRIPT"  # script overwrites with stage1 ckpt
        c["model_load_source"] = "lightning"
        c["resume"] = None
        c["resume_pretrain"] = True
        c["trainer"]["max_steps"] = 50000
        root = "runs/vla_two_stage_fm/stage2"
    c["output_root"] = root + "/checkpoints"
    c["log_root"] = root + "/logs"
    c["cache_root"] = root + "/cache"
    return c

json.dump(derive(1), open("configs/vla_ablation_rlds/fm_dualcam_stage1.json", "w"), indent=4, ensure_ascii=False)
json.dump(derive(2), open("configs/vla_ablation_rlds/fm_dualcam_stage2.json", "w"), indent=4, ensure_ascii=False)
print("wrote fm_dualcam_stage1.json and fm_dualcam_stage2.json")
PY
```
Expected: `wrote fm_dualcam_stage1.json and fm_dualcam_stage2.json`

- [ ] **Step 4: 运行测试确认通过**

Run:
```bash
cd /workspace/tingting/3DBENCH && conda run -p /workspace/tingting/envs/vlmbench-rlds \
  python -m pytest tests/test_two_stage_fm_configs.py -v
```
Expected: 4 passed

> 注：`test_stage2_full_finetune_with_resume_pretrain` 不检查 `model_load_path` 的具体值（运行时由脚本填 `RUNTIME_FILLED_BY_SCRIPT` 占位之外的真实路径）。

- [ ] **Step 5: 提交**

```bash
cd /workspace/tingting/3DBENCH && git add configs/vla_ablation_rlds/fm_dualcam_stage1.json configs/vla_ablation_rlds/fm_dualcam_stage2.json tests/test_two_stage_fm_configs.py
git commit -m "feat(config): two-stage FM dualcam configs (stage1 head-only, stage2 full)

Derive from e0_full_dualcam_bs256 (RLDS dualcam, Q99 norm) with FMDecoder
act_head. stage1: freeze VLM, 5k steps, lr=2e-5. stage2: full finetune,
50k steps, lr=8e-5, resume_pretrain=true (lightning source, resume=null).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: 双阶段编排脚本

一条命令跑 stage1 → 抓最新 ckpt → 生成临时 stage2 config 填入 ckpt 路径 → 跑 stage2。

**Files:**
- Create: `/workspace/tingting/3DBENCH/scripts/run_vla_two_stage_fm.sh`

**Interfaces:**
- Consumes: `configs/vla_ablation_rlds/fm_dualcam_stage1.json`、`fm_dualcam_stage2.json`；env 旁路 `STAGE`、`STAGE1_CKPT`、`DRY_RUN`、`GPUS_PER_NODE`、`MASTER_PORT`、`WANDB_PROJECT`、`LOSS_TYPE`
- Produces: 终态 ckpt 于 `runs/vla_two_stage_fm/stage2/checkpoints/`

- [ ] **Step 1: 写脚本**

创建 `/workspace/tingting/3DBENCH/scripts/run_vla_two_stage_fm.sh`：

```bash
#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# Evo-1-style two-stage Flow-Matching VLA training (RLDS dualcam baseline)
#
# Usage:
#   bash scripts/run_vla_two_stage_fm.sh                 # run stage1 then stage2
#   STAGE=stage1 bash scripts/run_vla_two_stage_fm.sh    # only stage1
#   STAGE1_CKPT=/path/to.ckpt STAGE=stage2 bash scripts/run_vla_two_stage_fm.sh  # skip stage1
#   DRY_RUN=1 bash scripts/run_vla_two_stage_fm.sh       # print commands only
#   CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS_PER_NODE=4 bash scripts/run_vla_two_stage_fm.sh
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"               # 3DBENCH/
VLM4VLA_ROOT="$(dirname "$PROJECT_ROOT")/VLM4VLA"     # VLM4VLA/
CONFIG_DIR="$PROJECT_ROOT/configs/vla_ablation_rlds"

STAGE1_CFG="$CONFIG_DIR/fm_dualcam_stage1.json"
STAGE2_CFG="$CONFIG_DIR/fm_dualcam_stage2.json"
STAGE1_CKPT_DIR="$VLM4VLA_ROOT/runs/vla_two_stage_fm/stage1/checkpoints"

GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NUM_NODES=${NUM_NODES:-1}
MASTER_PORT=${MASTER_PORT:-6044}
STAGE=${STAGE:-all}
LOSS_TYPE=${LOSS_TYPE:-l1_unified}

export TMPDIR="${TMPDIR:-/workspace/tingting/.tmp}"
mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export WANDB_PROJECT=${WANDB_PROJECT:-vla_two_stage_fm}

run_main() {
    local config_path="$1"
    local cmd="cd $VLM4VLA_ROOT && torchrun \
        --nnodes $NUM_NODES --node_rank 0 --nproc_per_node $GPUS_PER_NODE \
        --master_addr 127.0.0.1 --master_port $MASTER_PORT \
        main.py $config_path --gpus $GPUS_PER_NODE --num_nodes $NUM_NODES --loss_type $LOSS_TYPE"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY RUN] $cmd"
    else
        eval "$cmd"
    fi
}

latest_ckpt() {
    ls -t "$STAGE1_CKPT_DIR"/*.ckpt 2>/dev/null | head -n 1
}

run_stage1() {
    echo "================ STAGE 1 (freeze VLM, train FMDecoder) ================"
    run_main "$STAGE1_CFG"
}

run_stage2() {
    local ckpt="${STAGE1_CKPT:-}"
    if [ -z "$ckpt" ]; then
        ckpt="$(latest_ckpt || true)"
    fi
    if [ "${DRY_RUN:-0}" != "1" ] && [ -z "$ckpt" ]; then
        echo "[ERROR] No stage1 checkpoint found in $STAGE1_CKPT_DIR (set STAGE1_CKPT=...)"
        exit 1
    fi
    echo "================ STAGE 2 (full finetune, resume_pretrain) ============="
    echo "  stage1 ckpt: ${ckpt:-<dry-run-unknown>}"

    # Generate a temp stage2 config with model_load_path filled in (do not mutate the original).
    local tmp_cfg="$TMPDIR/fm_dualcam_stage2.filled.$$.json"
    if [ "${DRY_RUN:-0}" = "1" ] && [ -z "$ckpt" ]; then
        ckpt="DRY_RUN_PLACEHOLDER.ckpt"
    fi
    # No jq in this env; inject model_load_path via the conda env python.
    CKPT="$ckpt" SRC="$STAGE2_CFG" DST="$tmp_cfg" \
      conda run -p /workspace/tingting/envs/vlmbench-rlds python3 -c "
import json, os
c = json.load(open(os.environ['SRC']))
c['model_load_path'] = os.environ['CKPT']
json.dump(c, open(os.environ['DST'], 'w'), indent=4, ensure_ascii=False)
"
    echo "  filled config: $tmp_cfg"
    run_main "$tmp_cfg"
    rm -f "$tmp_cfg"
}

case "$STAGE" in
    stage1) run_stage1 ;;
    stage2) run_stage2 ;;
    all)    run_stage1; run_stage2 ;;
    *) echo "Unknown STAGE=$STAGE (use stage1|stage2|all)"; exit 1 ;;
esac

echo ""
echo "Done."
```

- [ ] **Step 2: 加可执行权限 + 校验 bash 语法**

Run:
```bash
cd /workspace/tingting/3DBENCH && chmod +x scripts/run_vla_two_stage_fm.sh && \
  bash -n scripts/run_vla_two_stage_fm.sh && echo "bash syntax OK"
```
Expected: `bash syntax OK`

- [ ] **Step 3: DRY_RUN 全程打印不执行**

Run:
```bash
cd /workspace/tingting/3DBENCH && DRY_RUN=1 STAGE=all bash scripts/run_vla_two_stage_fm.sh
```
Expected: 打印 stage1 与 stage2 两条 `[DRY RUN] ... torchrun ... main.py ...` 命令（stage2 用临时 filled config），不真正训练。

- [ ] **Step 4: 校验 python 注入确实写入 model_load_path**

Run:
```bash
cd /workspace/tingting/3DBENCH && CKPT="/tmp/fake/step.ckpt" \
  conda run -p /workspace/tingting/envs/vlmbench-rlds python3 -c "
import json, os
c = json.load(open('configs/vla_ablation_rlds/fm_dualcam_stage2.json'))
c['model_load_path'] = os.environ['CKPT']
print(c['model_load_path'])
"
```
Expected: 输出 `/tmp/fake/step.ckpt`

- [ ] **Step 5: 提交**

```bash
cd /workspace/tingting/3DBENCH && git add scripts/run_vla_two_stage_fm.sh
git commit -m "feat(script): two-stage FM orchestration (stage1 -> grab ckpt -> stage2)

run_vla_two_stage_fm.sh runs stage1, finds the latest stage1 .ckpt, fills
it into a temp stage2 config via jq (no mutation of the original), then
runs stage2. Supports STAGE=stage1|stage2|all, STAGE1_CKPT override, and
DRY_RUN. Reuses TMPDIR/WANDB/torchrun env from run_vla_ablation.sh.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: GPU 冒烟测试（端到端真实跑通两阶段）

这是整套方案的真正集成验证门：用 GPU 4-7、极小步数跑通 stage1→ckpt→stage2，确认冻结/加载/loss/gripper 正确。

**Files:**
- 临时使用：`/tmp` 下的缩水 config（不提交）；不改任何已提交文件

**Interfaces:**
- Consumes: Task 1-4 的全部产物
- Produces: 冒烟通过的证据（参数统计、加载日志、loss 曲线非 NaN）

- [ ] **Step 1: 生成缩水冒烟配置（stage1/stage2，各 20 步）**

Run:
```bash
cd /workspace/tingting/3DBENCH && conda run -p /workspace/tingting/envs/vlmbench-rlds python3 - <<'PY'
import json, copy
for stage in ("stage1", "stage2"):
    c = json.load(open(f"configs/vla_ablation_rlds/fm_dualcam_{stage}.json"))
    c["batch_size"] = 2
    c["trainer"]["max_steps"] = 20
    c["trainer"]["accumulate_grad_batches"] = 1
    c["trainer"]["val_check_interval"] = 10000
    c["warmup_steps"] = 5
    c["train_dataset"]["shuffle_buffer_size"] = 256
    c["val_dataset"]["shuffle_buffer_size"] = 256
    root = f"/workspace/tingting/.tmp/smoke_fm/{stage}"
    c["output_root"] = root + "/checkpoints"
    c["log_root"] = root + "/logs"
    c["cache_root"] = root + "/cache"
    c["trainer"]["logger"] = []  # disable wandb for smoke
    json.dump(c, open(f"/workspace/tingting/.tmp/smoke_{stage}.json", "w"), indent=4, ensure_ascii=False)
print("wrote /workspace/tingting/.tmp/smoke_stage1.json and smoke_stage2.json")
PY
```
Expected: `wrote ... smoke_stage1.json and smoke_stage2.json`

- [ ] **Step 2: 跑 stage1 冒烟，确认只有 FMDecoder 可训 + loss 非 NaN**

Run:
```bash
cd /workspace/tingting/VLM4VLA && CUDA_VISIBLE_DEVICES=4 conda run -p /workspace/tingting/envs/vlmbench-rlds \
  torchrun --nnodes 1 --node_rank 0 --nproc_per_node 1 --master_addr 127.0.0.1 --master_port 6071 \
  main.py /workspace/tingting/.tmp/smoke_stage1.json --gpus 1 --num_nodes 1 --loss_type l1_unified 2>&1 | tee /workspace/tingting/.tmp/smoke_stage1.log | tail -40
```
Expected: 训练跑到 step 20；日志中可见参数统计显示 backbone/vision 冻结、act_head 可训；loss 为有限值（非 NaN/inf）。
> 验证冻结：`grep -i "trainable\|requires_grad\|frozen\|act_head" /workspace/tingting/.tmp/smoke_stage1.log | head`。若现有代码未打印参数统计，改用下方 Step 2b 显式探测。

- [ ] **Step 2b（若 Step 2 无参数统计日志）：显式断言 stage1 冻结正确**

Run:
```bash
cd /workspace/tingting/VLM4VLA && CUDA_VISIBLE_DEVICES=4 conda run -p /workspace/tingting/envs/vlmbench-rlds python3 - <<'PY'
import json
from vlm4vla.train.base_trainer import BaseTrainer
cfg = json.load(open("/workspace/tingting/.tmp/smoke_stage1.json"))
model = BaseTrainer.from_checkpoint(None, "torch", cfg)  # build only
bb = sum(p.requires_grad for n,p in model.named_parameters() if "act_head" not in n)
ah = sum(p.requires_grad for n,p in model.named_parameters() if "act_head" in n)
print(f"non-act_head trainable params: {bb}; act_head trainable params: {ah}")
assert bb == 0, "stage1 must freeze entire VLM (no non-act_head trainable params)"
assert ah > 0, "stage1 must train FMDecoder"
print("stage1 freeze assertion OK")
PY
```
Expected: `non-act_head trainable params: 0; act_head trainable params: <N>`；`stage1 freeze assertion OK`

- [ ] **Step 3: 确认 stage1 产出 ckpt**

Run:
```bash
ls -t /workspace/tingting/.tmp/smoke_fm/stage1/checkpoints/*.ckpt | head -n 1
```
Expected: 打印一个 `.ckpt` 路径（非空）

- [ ] **Step 4: 跑 stage2 冒烟，填入 stage1 ckpt，确认 resume_pretrain 加载成功 + 全量可训**

Run:
```bash
cd /workspace/tingting/3DBENCH && \
  CKPT=$(ls -t /workspace/tingting/.tmp/smoke_fm/stage1/checkpoints/*.ckpt | head -n 1) && \
  CKPT="$CKPT" conda run -p /workspace/tingting/envs/vlmbench-rlds python3 -c "
import json, os
c = json.load(open('/workspace/tingting/.tmp/smoke_stage2.json'))
c['model_load_path'] = os.environ['CKPT']
json.dump(c, open('/workspace/tingting/.tmp/smoke_stage2.filled.json','w'), indent=4)
print('filled', os.environ['CKPT'])
" && \
  cd /workspace/tingting/VLM4VLA && CUDA_VISIBLE_DEVICES=4 conda run -p /workspace/tingting/envs/vlmbench-rlds \
  torchrun --nnodes 1 --node_rank 0 --nproc_per_node 1 --master_addr 127.0.0.1 --master_port 6072 \
  main.py /workspace/tingting/.tmp/smoke_stage2.filled.json --gpus 1 --num_nodes 1 --loss_type l1_unified 2>&1 | tee /workspace/tingting/.tmp/smoke_stage2.log | tail -50
```
Expected: 日志含 `[resume_pretrain] loading weights only from ...`；`load_state_dict` 的 missing/unexpected keys 为空或仅预期项；训练跑到 step 20；loss 有限。
> 验证加载：`grep -i "resume_pretrain\|missing key\|unexpected key\|_IncompatibleKeys" /workspace/tingting/.tmp/smoke_stage2.log`。

- [ ] **Step 5: 记录冒烟结论（不提交临时文件）**

Run:
```bash
cd /workspace/tingting/3DBENCH && {
  echo "## smoke (GPU4) $(date -u +%FT%TZ)"
  echo "stage1 freeze: $(grep -c 'freeze assertion OK' /workspace/tingting/.tmp/smoke_stage1.log 2>/dev/null || echo 'see log')"
  echo "stage2 resume_pretrain: $(grep -c 'resume_pretrain' /workspace/tingting/.tmp/smoke_stage2.log)"
} >> docs/superpowers/plans/2026-06-22-two-stage-flowmatching-vla.md
git add docs/superpowers/plans/2026-06-22-two-stage-flowmatching-vla.md
git commit -m "test: record two-stage FM smoke results (GPU4)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

> 临时文件 `/workspace/tingting/.tmp/smoke_*` 不入库（`.tmp` 已是 untracked）。

---

## 最终交付

两阶段正式训练（GPU 4-7，4 卡，等价 eff_batch 与你的基线对齐）：

```bash
cd /workspace/tingting/3DBENCH && CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS_PER_NODE=4 \
  conda run -p /workspace/tingting/envs/vlmbench-rlds \
  bash scripts/run_vla_two_stage_fm.sh
```

终态 ckpt：`/workspace/tingting/VLM4VLA/runs/vla_two_stage_fm/stage2/checkpoints/`。

---

## Self-Review

**Spec coverage：**
- §3/§4 配置 → Task 3 ✓
- §5 resume_pretrain → Task 2 ✓
- §6 gripper 同质化 → Task 1 ✓
- §6.1 FM loss / ratio no-op → Task 3（ratio=1 + 不依赖）+ Global Constraints ✓
- §7 归一化（norm_action:false）→ Task 3 配置 + 测试断言 ✓
- §8 编排脚本 → Task 4 ✓
- §9 验证（冻结/加载/loss/gripper，GPU4-7，不碰 sim eval）→ Task 1 单测 + Task 5 冒烟 ✓
- §10 不做的事 → 计划未触碰 FCDecoder / DeepSpeed / sim eval ✓

**Placeholder scan：** stage2 配置的 `model_load_path` 在 Task 3 显式置为 `RUNTIME_FILLED_BY_SCRIPT` 占位、Task 4 脚本运行时用 jq 填真实路径、Task 5 冒烟用 jq 填——三处一致，非 plan 占位。无 TODO/TBD。

**Type consistency：** `_gripper_to_continuous`/`_gripper_to_unit` 在 Task 1 定义并在 forward/get_labels/predict 调用，名称一致。`resume_pretrain`/`model_load_path`/`model_load_source=lightning`/`resume=null` 在 Task 2(校验)、Task 3(配置)、Task 4(脚本填)、Task 5(冒烟)中名称一致。
