# 两阶段 Flow-Matching VLA 训练（RLDS dualcam 基线）设计文档

- 日期：2026-06-22
- 状态：待 review
- 目标：参考 Evo-1，在现有 RLDS dual-cam 单阶段全量微调流程之上，新增「两阶段训练 + Flow-Matching action head」的一套可复现训练方案。

## 1. 背景与现状

当前训练方式（单阶段全量微调）：

```bash
conda run -p /workspace/tingting/envs/vlmbench-rlds \
  bash scripts/run_vla_ablation.sh e0_full_dualcam_bs256
```

- 入口：`/workspace/tingting/VLM4VLA/main.py`（PyTorch Lightning `Trainer.fit`）
- 模型：`RoboQwen25VL`（Qwen2.5-VL-3B-Instruct backbone）+ `FCDecoder`（回归式 MLP，L1 / split-BCE）
- 数据：RLDS dual-cam（`OpenVLADataset`，`data_root_dir=/workspace/tingting/LIBERO_rlds`，`use_hand_rgb=true`，`data_mix=libero_10_no_noops`）
- 配置：`/workspace/tingting/3DBENCH/configs/vla_ablation_rlds/e0_full_dualcam_bs256.json`，`max_steps=50000`

参考目标 Evo-1（`/workspace/tingting/Evo-1`）的两阶段范式：
- Stage 1：冻结整个 VLM，只训 action head（含融合模块），少量步数预热。
- Stage 2：加载 Stage 1 权重，**重置 optimizer + LR scheduler**，全量解冻端到端微调。
- Stage 2 的核心 trick 是 `--resume_pretrain`：只取权重、重置优化器/调度器、重新 warmup。

### 现有可复用资产（已验证）

- `FMDecoder` 已实现且端到端打通：`/workspace/tingting/VLM4VLA/vlm4vla/model/policy_head/fm_decoder.py`（DiT + layer-wise cross-attention，`t~Beta(1.5,1.0)`，velocity 目标，20 步 Euler 推理）。
- backbone 已自动检测 `isinstance(act_head, FMDecoder)` 并用 hook 收集 per-layer hidden states 供 `vl_embs_list`：`roboqwen25vl.py:219-276`。
- 冻结机制已具备：`base_backbone.py:363-410` 的 `_trainable_params_setup`，由 `freeze_backbone` / `train_vision` / `train_text_embedding` 控制；`act_head` 永远可训。
- 已有一套 FM 配置（但走旧 HDF5 数据）：`/workspace/tingting/3DBENCH/configs/vla_ablation_fm/`。

### 关键缺口

1. 没有把 stage1 → stage2 串起来的机制（`_head` 与 `_full` 是孤立 ablation，不是「加载权重 + 重置 optimizer」）。
2. FM 配置用的是旧 HDF5 dataset（`LiberoActionPredictionDataset`），不是当前 RLDS dual-cam pipeline。
3. 没有两阶段编排脚本。
4. FMDecoder 对 7 维（含 gripper）统一做 MSE velocity，但 gripper 在 RLDS 里是 {0,1} 离散，与其余 [-1,1] 连续维不同质。

## 2. 设计决策汇总

| 项 | 决策 |
|---|---|
| 基线 | RLDS dualcam（`OpenVLADataset`, `LIBERO_rlds`, dual-cam），与现单阶段一致 |
| action head | `FMDecoder`（flow matching，DiT 12 层 cross-attn），架构零改动 |
| stage1 | 冻整个 VLM 只训 FMDecoder，5k 步，lr=2e-5，warmup 1000 |
| stage2 | 全量解冻，50k 步，lr=8e-5，warmup 800 |
| 衔接 | 显式 `resume_pretrain` 开关：载 stage1 全部权重（含 head）、重置 optimizer+scheduler、`resume=null` |
| 归一化 | `norm_action: false`（RLDS 上游 Q99 已做，`norm_min/max` 为死参数，与基线一致） |
| gripper | FMDecoder 内部 {0,1}↔{-1,+1} 映射，7 维同质，贴 Evo-1 |
| 编排 | `scripts/run_vla_two_stage_fm.sh`（跑1→抓ckpt→填2→跑2，支持 STAGE/DRY_RUN/STAGE1_CKPT） |
| 验证 | GPU 4-7 冒烟：参数冻结/加载校验 + loss 非 NaN，不碰 sim eval |

## 3. 总体架构

复用现有所有组件，**不改模型架构**。新增物只有：2 个配置 + 1 个 `resume_pretrain` 开关（含 gripper 映射） + 1 个双阶段编排脚本。

```
stage1 config ──► main.py (Trainer.fit, resume=null)
                    │  freeze VLM, train FMDecoder only, 5k steps
                    ▼
            runs/vla_two_stage_fm/stage1/checkpoints/*.ckpt
                    │  (脚本抓取最新 ckpt 路径)
                    ▼
stage2 config (model_load_path=stage1 ckpt, resume_pretrain=true, resume=null)
                    │  load all weights (incl head) strict=False
                    │  fresh AdamW + scheduler, full finetune, 50k steps
                    ▼
            runs/vla_two_stage_fm/stage2/checkpoints/*.ckpt  (最终模型)
```

## 4. 配置文件

新增于 `configs/vla_ablation_rlds/`，以 `e0_full_dualcam_bs256.json` 为模板（保持数据/dualcam/Q99 归一化与基线一致），`act_head` 换成 FMDecoder（DiTConfig 取自 `vla_ablation_fm/e0_full.json`）。

### `fm_dualcam_stage1.json`（stage1：冻 VLM 只训 head）

相对模板的差异：

```jsonc
"task_name": "vla_two_stage_fm_stage1",
"learning_rate": 2e-05,
"warmup_steps": 1000,
"arm_gripper_loss_ratio": 1,            // 对 FM 路径无效（见 §6.1）；保留默认值，不写 0.01
"train_setup": {
  "freeze_backbone": true,
  "train_vision": false,
  "train_text_embedding": false
},
"act_head": {
  "type": "FMDecoder",
  // ... 完整 FMDecoder + DiTConfig，照搬 vla_ablation_fm/e0_full.json 的 act_head ...
  "fwd_pred_next_n": 4
},
"norm_action": false,                   // RLDS Q99 已在上游做（关键，勿改 true）
"norm_min": -0.65, "norm_max": 0.65,    // 死参数，RLDS 路径不读，仅保留格式一致
"model_load_path": null,
"model_load_source": "torch",
"resume": null,
"resume_pretrain": false,
"trainer": { "max_steps": 5000, /* 其余同基线 */ },
"output_root": "runs/vla_two_stage_fm/stage1/checkpoints",
"log_root": "runs/vla_two_stage_fm/stage1/logs",
"cache_root": "runs/vla_two_stage_fm/stage1/cache"
```

### `fm_dualcam_stage2.json`（stage2：全量）

```jsonc
"task_name": "vla_two_stage_fm_stage2",
"learning_rate": 8e-05,
"warmup_steps": 800,
"arm_gripper_loss_ratio": 1,            // 同 stage1，对 FM 路径无效（见 §6.1）
"train_setup": {
  "freeze_backbone": false,
  "train_vision": true,
  "train_text_embedding": true
},
"act_head": { /* 与 stage1 完全相同的 FMDecoder + DiTConfig */ },
"norm_action": false, "norm_min": -0.65, "norm_max": 0.65,
"model_load_path": "<由脚本填入 stage1 最新 .ckpt>",
"model_load_source": "lightning",       // stage1 存的是 Lightning .ckpt
"resume": null,                          // 关键：不走 Lightning 续训
"resume_pretrain": true,                 // 只取权重，重置 optimizer/scheduler
"trainer": { "max_steps": 50000, /* 其余同基线 */ },
"output_root": "runs/vla_two_stage_fm/stage2/checkpoints",
"log_root": "runs/vla_two_stage_fm/stage2/logs",
"cache_root": "runs/vla_two_stage_fm/stage2/cache"
```

## 5. `resume_pretrain` 开关语义

### 原理

经代码确认：
- `trainer.fit(ckpt_path=variant["resume"])`：`resume` 非空时 Lightning 恢复 optimizer+scheduler+step；为 null 时全新开始。
- `from_checkpoint(model_load_path, source, variant)`：只 `load_state_dict(strict=False)` 载权重，不动 optimizer。
- 因此 Evo-1 的 `resume_pretrain` == `model_load_path=stage1_ckpt` + `resume=null`：权重载入（含 head），optimizer/scheduler 重置，从 step 0 重新 warmup。

### 改动（最小化）

在 `main.py` 的 `experiment()` 中加几行校验 + 日志，使语义显式：

```python
resume_pretrain = variant.get("resume_pretrain", False)
if resume_pretrain:
    assert variant.get("model_load_path"), \
        "resume_pretrain=True 需要 model_load_path 指向 stage1 ckpt"
    assert variant.get("resume") is None, \
        "resume_pretrain 与 Lightning resume 互斥（后者会恢复 optimizer/scheduler）"
    print(f"[resume_pretrain] 从 {variant['model_load_path']} 仅加载权重，重置 optimizer/scheduler")
```

加载仍走现成 `from_checkpoint(model_load_path, "lightning", variant)`（`strict=False`，含 head）。开关本身**不改加载逻辑**，只做校验 + 日志，保证语义正确且可读。

### 风险确认

stage1 的 `.ckpt` 由 `freeze_backbone=true` 训练，Lightning 仍保存完整 `state_dict`（冻结参数也在内，只是无梯度），故 stage2 的 `load_state_dict(strict=False)` 能完整恢复 VLM + head。无缺键问题。

## 6. Gripper 同质化（7 维全连续，贴 Evo-1）

### 流向（已确认）

dataloader 输出 7 维动作 → `_process_batch` 拆成 `arm_action_chunck`（dims 0-5）与 `gripper_action_chunck`（dim 6，原始 {0,1}）→ 模型把 `(arm, gripper)` 作为 `actions` 传给 head → FMDecoder 内部 concat 回 7 维。

### 设计：映射点全封在 FMDecoder 内部，FCDecoder 路径零影响

FCDecoder 的 gripper 走 sigmoid/BCE，期望 {0,1}，**不能动**。FMDecoder 内部映射：

- 训练 forward（`fm_decoder.py:165-168` concat 前）：`g' = 2g - 1`，{0,1} → {-1,+1}，拼成 7 维做 flow matching。
- `get_labels`（`:243-246`）：同样映射，保证 label 同质。
- inference `predict`（`:311` 输出前）：连续 `ĝ' ∈ [-1,1]` 反映射 `ĝ = (ĝ'+1)/2`，再由下游阈值化到 {0,1}，对外接口不变。

用两个静态 helper 收口：

```python
@staticmethod
def _gripper_to_continuous(g):   # {0,1} -> {-1,+1}
    return g * 2.0 - 1.0
@staticmethod
def _gripper_to_unit(g):         # [-1,1] -> [0,1]
    return (g + 1.0) * 0.5
```

结果：FMDecoder 内部 7 维全为 [-1,1] 连续量，velocity 场同质；对外接口（dataloader 给 {0,1}、推理返回 {0,1}）完全不变。

### 6.1 FMDecoder loss 算法与 `arm_gripper_loss_ratio` 失效说明

FMDecoder 是 flow-matching，loss 与 arm/gripper 的拆分无关（`fm_decoder.py:249-263`）：

```python
def loss(self, pred_action, labels, attention_mask=None):
    target_bt = self._last_velocity            # forward 时缓存的 velocity = actions - noise
    loss = ((pred_bt - target_bt) ** 2).mean() # 对全 7 维一起做 MSE
    return {"loss_arm": loss,                  # 整个 loss 塞进 loss_arm（复用接线）
            "loss_gripper": torch.tensor(0.0), # 恒为 0，纯占位
            "acc_gripper": -1.0}
```

即：单一 velocity MSE，在全部 7 维上同时算（arm 6 维 + gripper 1 维不分家）。

经 trainer `_get_loss`（`base_trainer.py:278-284`）确认 `arm_gripper_loss_ratio` 对 FM 路径**无效**：

| `loss_type` | 实际计算 | ratio 是否生效 |
|---|---|---|
| `l1_unified`（启动脚本默认 `LOSS_TYPE=l1_unified`） | `loss = loss_arm_act`（FM 的 7 维 MSE） | 否，整条分支不读 ratio |
| `split_bce` | `loss = loss_arm_act + loss_gripper_act × ratio` | 读，但 `loss_gripper_act ≡ 0`，乘任何数恒为 0 |

因此本设计**不依赖也不调** `arm_gripper_loss_ratio`，配置中保留为默认 `1`（无害）。

**为何选项 A（无 per-dim 加权）成立**：FM 是全 7 维无差别 MSE，gripper 维必须与 arm 维同尺度，否则其在总 MSE 中的权重失衡。§6 的 gripper {0,1}→{-1,+1} 同质化正是让 gripper 落到与 arm 相同的 [-1,1] 区间，使无差别 MSE 公平。同质化之后无需再做逐维加权。若日后发现 gripper 学习不足，正确旋钮是在 `FMDecoder.loss` 内做逐维加权（而非死参数 `arm_gripper_loss_ratio`）——本次不实现（YAGNI）。

## 7. 归一化说明（澄清，无需新增逻辑）

RLDS 路径在 TFDS/prismatic 上游用 `NormalizationType.BOUNDS_Q99` 把动作逐维映射到 [-1,1]（`base_openvla_dataset.py:75`，`libero_constants.py:26-37`），gripper 维单独 `1-clip(g,0,1)`。dataset 层的 `norm_action`/`norm_min`/`norm_max`（`data_utils.py:577` 的 `normalize_action`）只服务 CALVIN/HDF5，RLDS 路径不读。故 stage1/stage2 均设 `norm_action: false`，与现单阶段基线一致；`norm_min/max` 仅作格式占位。

> 注：memory 中 "flip + 0.65 norm" 指 HDF5 实验批次，与本 RLDS dualcam 方案无关。

## 8. 编排脚本 `scripts/run_vla_two_stage_fm.sh`

一条命令跑完两阶段：

```bash
bash scripts/run_vla_two_stage_fm.sh
```

逻辑：

1. 跑 stage1：`torchrun ... main.py configs/vla_ablation_rlds/fm_dualcam_stage1.json ...`
2. 从 `runs/vla_two_stage_fm/stage1/checkpoints/` 找最新 `*.ckpt`（`ModelCheckpoint(save_top_k=1, every_n_train_steps=2000)`）。
3. 用 `jq` 把该路径写入 stage2 config 的 `model_load_path`（生成一份临时 stage2 config，避免污染原文件），保持 `resume_pretrain=true`、`model_load_source=lightning`。
4. 跑 stage2。

复用现有 `run_vla_ablation.sh` 的环境设置（`TMPDIR`、`WANDB_PROJECT`、`MASTER_PORT`、`GPUS_PER_NODE`、`LOSS_TYPE`）。

旁路开关：
- `STAGE=stage1|stage2|all`（默认 all）
- `STAGE1_CKPT=/path/to.ckpt`：跳过 stage1，直接用指定 ckpt 跑 stage2
- `DRY_RUN=1`：只打印命令不执行

## 9. 验证策略

- **冒烟测试**（GPU 4-7，遵循 memory 中 GPU 偏好）：各阶段 `max_steps=20`、`batch_size=2`、`shuffle_buffer_size` 调小，跑通 stage1 → 抓 ckpt → stage2 加载。确认：
  1. stage1 仅 FMDecoder 参数 `requires_grad=True`（打印 trainable/frozen 参数统计）；
  2. stage2 全部可训且成功加载 stage1 权重（`load_state_dict` missing/unexpected keys 为空或仅预期项）；
  3. loss 正常、不 NaN；
  4. gripper 映射往返正确（单元级断言 `_gripper_to_unit(_gripper_to_continuous(g)) == g`）。
- **不做** sim eval（独立闭环，存在 train/eval 解耦的已知问题，超出本次范围）。

## 10. 不做的事（YAGNI）

- 不改模型架构、不改 backbone 的 `vl_embs_list` 逻辑。
- 不动 FCDecoder 及其 loss。
- 不引入 DeepSpeed 流程改动（沿用现有 Lightning DDP）。
- 不做 sim eval 集成。
- 不为 HDF5 路径出第二套配置（本次只 RLDS）。

## 11. 受影响文件清单

新增：
- `configs/vla_ablation_rlds/fm_dualcam_stage1.json`
- `configs/vla_ablation_rlds/fm_dualcam_stage2.json`
- `scripts/run_vla_two_stage_fm.sh`

修改：
- `VLM4VLA/main.py`：`experiment()` 加 `resume_pretrain` 校验 + 日志（约 6 行）
- `VLM4VLA/vlm4vla/model/policy_head/fm_decoder.py`：gripper 映射 helper + 在 forward/get_labels/predict 三处调用
