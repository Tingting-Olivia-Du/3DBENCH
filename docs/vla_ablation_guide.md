# VLA Ablation Experiment Guide

## 1. Overview

本实验旨在回答一个核心研究问题：**3DBENCH 各维度的空间推理能力增强（通过 LoRA 微调）是否能提升下游 VLA 机器人操控性能？哪些空间能力最重要？**

### 实验设计

- **Backbone**: Qwen2.5-VL-3B-Instruct 基座模型
- **空间增强**: 将 3DBENCH 各维度（Q1-Q8）的 LoRA 适配器 merge 进基座模型
- **VLA 框架**: 基于 [VLM4VLA](../VLM4VLA/) 框架，使用 FCDecoder action head
- **评估环境**: LIBERO 机器人操控基准（4 个 task suite）
- **消融维度**: 10 种 backbone 初始化 x 2 种训练策略 = 20 个实验

### 实验矩阵

| 实验 ID | Backbone LoRA | 描述 | 优先级 |
|---------|--------------|------|--------|
| E0 | 无 (vanilla) | 基线 | 1 |
| E1 | Q1 (物体3D位置) | 物体定位能力 | 3 |
| E2 | Q2 (夹爪3D位置) | 自身定位能力 | 6 |
| E3 | Q3 (偏移向量) | 相对定位能力 | 4 |
| E4 | Q4 (空间关系) | 空间语言理解 | 9 |
| E5 | Q5 (两物体距离) | 距离估计能力 | 10 |
| E6 | Q6 (末端朝向) | 旋转感知能力 | 8 |
| E7 | Q7 (夹爪开合) | 夹爪状态感知 | 7 |
| E8 | Q8 (7D动作) | 完整动作预测 | 2 |
| E9 | All-Q (全维度) | 所有维度联合 | 5 |

每个实验有两种训练策略：
- **full**: 全模型微调（backbone + vision + action head 全部训练）
- **head**: 仅训练 action head（冻结 backbone，完整保留 LoRA 空间推理能力）

---

## 2. 文件结构

```
3DBENCH/
├── scripts/
│   ├── merge_lora_adapters.py        # Step 1: 合并 LoRA 到基座模型
│   ├── generate_ablation_configs.py  # Step 2: 生成实验配置文件
│   ├── run_vla_ablation.sh           # Step 3: 训练启动脚本
│   ├── run_vla_ablation_eval.sh      # Step 4: 评估启动脚本
│   └── analyze_vla_ablation.py       # Step 5: 结果分析脚本
├── configs/
│   └── vla_ablation/
│       ├── README.md                 # 配置文件索引
│       ├── e0_full.json              # 基线 - 全微调
│       ├── e0_head.json              # 基线 - 仅训练 head
│       ├── e1_full.json ~ e9_full.json
│       └── e1_head.json ~ e9_head.json
├── results/
│   └── vla_ablation/                 # 评估结果（运行后生成）
│       ├── *.log                     # 每个实验的评估日志
│       ├── summary.md                # 汇总表格
│       └── correlation.csv           # 相关性分析数据
└── docs/
    └── vla_ablation_guide.md         # 本文档
```

---

## 3. 完整调用流程

### Step 1: 合并 LoRA 适配器

将 3DBENCH 各维度微调的 LoRA 权重合并进 Qwen2.5-VL-3B 基座模型，生成独立的 HuggingFace checkpoint。

**为什么要 merge 而不是保留 adapter？**
- VLM4VLA 的 `build_vlm()` 直接加载 HuggingFace 模型，不支持 PEFT 包装
- merge 后是标准模型格式，无需修改 VLM4VLA 框架代码

**输入**:
- 基座模型: `/workspace/tingting/models/Qwen2.5-VL-3B-Instruct`
- LoRA 适配器: `/workspace/tingting/3DBENCH/models/0515/qwen2.5-vl-3b-mv-lora/{q1..q8,all}/`

**输出**:
- 合并模型: `/workspace/tingting/models/merged/Qwen2.5-VL-3B-{dim}-Merged/`

**命令**:
```bash
cd /workspace/tingting/3DBENCH

# 查看可用的适配器（不实际执行）
python scripts/merge_lora_adapters.py --dry-run

# 合并全部（约 30 分钟，需要 GPU）
python scripts/merge_lora_adapters.py

# 只合并特定维度
python scripts/merge_lora_adapters.py --dims q1 q8 all
```

**脚本工作流程**:
1. 以 FP32 精度加载基座模型（保证合并精度）
2. 加载 LoRA adapter（PEFT `PeftModel.from_pretrained`）
3. 调用 `merge_and_unload()` 将 LoRA 增量权重合并进基座参数
4. 转换为 BF16 存储
5. 保存模型 + 复制 tokenizer/processor 文件
6. 生成 `merge_info.json` 元数据

**适配器映射**（部分维度无 final checkpoint，使用最佳中间 checkpoint）:

| 维度 | 使用的 Checkpoint |
|------|------------------|
| q1, q2, q7, q8, all | `final/` |
| q3, q4, q5, q6 | `checkpoint-4000/` |

---

### Step 2: 准备 LIBERO 训练数据

VLM4VLA 使用 RLDS（TFRecord）格式的数据。从 HuggingFace 下载预转换好的数据集：

```bash
cd /workspace/tingting
git lfs install
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds
```

该数据集包含 4 个 LIBERO task suite（spatial, object, goal, 10），已经过滤了 no-op 动作，图像为 256x256。

> 如果下载不便，也可以自行从 HDF5 转换。参考 `VLM4VLA/openvla/experiments/robot/libero/regenerate_libero_dataset.py`。

---

### Step 3: 生成实验配置

基于 VLM4VLA 的 LIBERO 配置模板，自动生成 20 个实验配置文件。

**命令**:
```bash
cd /workspace/tingting/3DBENCH

# 生成全部配置（默认 RLDS 数据路径）
python scripts/generate_ablation_configs.py

# 指定数据路径
python scripts/generate_ablation_configs.py --data-root /path/to/modified_libero_rlds

# 只生成 full-finetune 策略的配置
python scripts/generate_ablation_configs.py --strategies full
```

**脚本工作流程**:
1. 加载 VLM4VLA 模板: `VLM4VLA/configs/oxe_training/libero10/finetune_qwen25vl-3b_libero10.json`
2. 对每个实验 x 策略组合：
   - 设置 `model_path` 指向对应的 merge 后模型（或基座模型）
   - 设置 `task_name` 用于 W&B 跟踪
   - 设置训练策略（`freeze_backbone`, `train_vision`, `train_text_embedding`）
   - 设置独立的输出/日志路径
3. 写入 JSON 到 `configs/vla_ablation/`
4. 生成 `README.md` 索引文件

**配置文件中的关键差异**:

| 字段 | full 策略 | head 策略 |
|------|----------|----------|
| `freeze_backbone` | `false` | `true` |
| `train_vision` | `true` | `false` |
| `train_text_embedding` | `true` | `false` |

**统一超参数**（所有实验相同）:

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| 学习率 | 2e-5 |
| 批量大小 | 8 x 梯度累积 8 = 有效 64 |
| 最大步数 | 50,000 |
| 精度 | BF16 |
| 策略 | DeepSpeed Stage 2 |
| 动作归一化 | [-0.65, 0.65] |
| Action Head | FCDecoder (hidden=1024, action_dim=7) |

---

### Step 4: 训练

通过 `torchrun` 分布式启动 VLM4VLA 训练。

**命令**:
```bash
cd /workspace/tingting/3DBENCH

# 运行单个实验
bash scripts/run_vla_ablation.sh e0_full

# 按优先级运行全部实验
bash scripts/run_vla_ablation.sh all

# 只运行 full-finetune 策略
bash scripts/run_vla_ablation.sh full

# 只运行 head-only 策略
bash scripts/run_vla_ablation.sh head

# 干跑模式（仅打印命令）
DRY_RUN=1 bash scripts/run_vla_ablation.sh all

# 多 GPU 训练
GPUS_PER_NODE=4 bash scripts/run_vla_ablation.sh e0_full
```

**脚本工作流程**:
1. 设置环境变量（`PYTHONUNBUFFERED`, `OMP_NUM_THREADS`）
2. 定位实验配置文件: `configs/vla_ablation/{exp_name}.json`
3. 在 VLM4VLA 目录下执行 `torchrun main.py` 启动分布式训练
4. W&B 自动记录训练指标

**训练过程中的 VLA 架构**:

```
Image (224x224) + Language instruction
  │
  ▼
Qwen2.5-VL-3B (vanilla 或 LoRA-merged)
  ├── Vision Tower (ViT) → image_embeds
  ├── Text Embedding → text_embeds
  └── LLM Transformer → 全序列注意力
        │
        + 可学习 action_token (nn.Parameter, dim=2048) 追加到序列末尾
        │
        ▼
    hidden_states[-1] → 提取 action token → (B, 1, 2048)
        │
        ▼
    FCDecoder Action Head:
      ├── MLP: Linear(2048,1024) → ReLU → Linear(1024,1024)
      ├── Arm Head (MLPTanhHead): → 6D 位姿 (Tanh, [-1,1])
      └── Gripper Head (MLPSigmoidHead): → 1D 开合
```

**损失函数**:
- Arm: Huber Loss（对异常值鲁棒）
- Gripper: Binary Cross-Entropy with Logits
- 总损失 = arm_loss + 0.01 × gripper_loss

**预估时间**: 每个 full-finetune 实验约 8-12 小时（1xA100），head-only 约 4-6 小时。

---

### Step 5: 评估

在 LIBERO 仿真环境中评估训练好的 VLA 模型。

**命令**:
```bash
cd /workspace/tingting/3DBENCH

# 评估单个实验（所有 LIBERO suite）
bash scripts/run_vla_ablation_eval.sh e0_full

# 评估单个实验，指定 suite
bash scripts/run_vla_ablation_eval.sh e0_full libero_spatial

# 评估所有已完成的实验
bash scripts/run_vla_ablation_eval.sh all

# 指定 GPU
CUDA_DEVICE=0 bash scripts/run_vla_ablation_eval.sh e0_full

# 干跑模式
DRY_RUN=1 bash scripts/run_vla_ablation_eval.sh all
```

**评估 Task Suite**:

| Suite | 任务数 | 关注能力 | 预期相关维度 |
|-------|-------|---------|------------|
| libero_spatial | 10 | 空间操控 | Q1, Q3, Q4, Q5 |
| libero_object | 10 | 物体操控 | Q1, Q7 |
| libero_goal | 10 | 目标导向 | Q3, Q8 |
| libero_10 | 10 | 长时域复杂任务 | Q8, All |

**评估指标**: 每个任务 50 次试验的 success rate (%)。

**输出**: 日志文件保存到 `results/vla_ablation/{exp_name}_{suite}.log`。

---

### Step 6: 结果分析

汇总评估结果，生成表格和相关性数据。

**命令**:
```bash
cd /workspace/tingting/3DBENCH

python scripts/analyze_vla_ablation.py

# 指定结果目录
python scripts/analyze_vla_ablation.py --results-dir results/vla_ablation
```

**脚本工作流程**:
1. 扫描 `results/vla_ablation/*.log` 文件
2. 正则解析 success rate（支持多种输出格式）
3. 生成 `summary.md`：20 行 x 4 列的 success rate 表格
4. 生成 `correlation.csv`：3DBENCH 改进幅度 vs VLA success rate
5. 包含 3DBENCH baseline/LoRA 对照表作为参考

**输出文件**:
- `results/vla_ablation/summary.md` — 主结果表格
- `results/vla_ablation/correlation.csv` — 相关性分析数据（可用于绘图）

---

## 4. 端到端流程总结

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: merge_lora_adapters.py                             │
│  输入: 基座模型 + Q1-Q8 LoRA adapters                        │
│  输出: 9 个 merged 模型 checkpoint                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: 下载 LIBERO RLDS 数据                               │
│  来源: huggingface.co/datasets/openvla/modified_libero_rlds  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: generate_ablation_configs.py                       │
│  输入: VLM4VLA 模板 + merged 模型路径 + 数据路径              │
│  输出: 20 个 JSON 配置文件 (10 backbone x 2 策略)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: run_vla_ablation.sh                                │
│  调用: VLM4VLA/main.py (torchrun 分布式训练)                  │
│  输出: 模型 checkpoint (runs/vla_ablation/e*_*/checkpoints/) │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: run_vla_ablation_eval.sh                           │
│  调用: VLM4VLA/eval/libero/run_libero_eval.py               │
│  输出: 评估日志 (results/vla_ablation/*.log)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 6: analyze_vla_ablation.py                            │
│  输出: summary.md (汇总表格) + correlation.csv (相关性数据)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 关键依赖

本实验依赖以下外部组件，**不修改其代码**，仅通过配置驱动：

| 组件 | 路径 | 作用 |
|------|------|------|
| VLM4VLA | `/workspace/tingting/VLM4VLA/` | VLA 训练框架 |
| `main.py` | `VLM4VLA/main.py` | 训练入口 |
| `BaseRoboVLM` | `VLM4VLA/vlm4vla/model/backbone/base_backbone.py` | 模型架构（action token 注入） |
| `RoboQwen25VL` | `VLM4VLA/vlm4vla/model/backbone/roboqwen25vl.py` | Qwen2.5-VL backbone 集成 |
| `FCDecoder` | `VLM4VLA/vlm4vla/model/policy_head/base_policy.py` | Action Head（MLP + Tanh/Sigmoid） |
| `run_libero_eval.py` | `VLM4VLA/eval/libero/run_libero_eval.py` | LIBERO 评估脚本 |
| LIBERO | `/workspace/tingting/LIBERO/` | 机器人操控仿真环境 |

---

## 6. 预期交付物

1. **主结果表**: 20 行（10 backbone x 2 策略）x 4 列（LIBERO suite）的 success rate
2. **相关性图**: 3DBENCH MAE 降低幅度 vs VLA success rate 提升
3. **核心对比**: full-finetune vs head-only —— 全量训练是否会冲掉 LoRA 的空间推理能力？
4. **分维度分析**: 哪些空间维度对哪类操控任务贡献最大？
