# Stage-1 训练管线技术文档（starVLA 集成 + 三扩展）

日期：2026-07-09 · 状态：CPU 侧全部验证通过，GPU 冒烟待权重就绪
关联：`3dvla-data/DATA_SPEC.md`（数据规格 v2.1）、`plan/fintune-vlm-en.md` §3
本文覆盖：从冻结数据到 `train_starvlm.py` 可训练之间的全部新增/修改代码，及其验证记录。

---

## 1. 总览与设计原则

Stage-1 = 在 5 切片混合数据上微调 **Qwen3-VL-4B-Instruct**（原版，非 -Action 扩表版），产出按能力轴受控的变体检查点（full/none/…），供 Stage-2 VLA 训练与双域 benchmark 评测。框架用 starVLA（用户决策：消除 Stage-1→Stage-2 的接缝风险）。

四条设计原则贯穿实现：

1. **离线物化，而非训练侧采样器**——混合/加权/过滤全部离线完成，产出单一 jsonl；模型看到的就是这份可审计的文件，全部既有数据审计工具可直接复用。训练侧零采样逻辑 = 零采样 bug 面。
2. **加载时统一，磁盘保持原图**——focal 统一在 dataloader worker 里做（PIL 等比缩放，毫秒级），不物化 60 万张统一副本。
3. **契约用断言兑现，不靠约定**——"processor 不得二次缩放"不是注释，是启动断言 + 逐图运行期断言。
4. **护栏 + 计数器**——所有"理论上不该发生"的情形（零监督样本、占位符错配、超预算）要么 raise 要么进审计计数器，不允许静默。

### 数据流

```
sft/variants_tokcal/<v>.json          (token 校准规格, token_calibrate.py)
        │
        ▼
scripts/materialize_variant_data.py   (采样/重复/模板手术/绝对路径/自检)
        │
        ▼
sft/materialized/<v>.jsonl (+.manifest.json)   ← 唯一训练输入，可审计
        │
        ▼
starVLA/dataloader/vlm_datasets_3d.py (focal统一 → chat template → 标签解掩码
        │                              → rope index → 护栏/计数器)
        ▼
DataCollatorForSupervisedDataset      (右 pad、截断、position_ids pad)
        │
        ▼
train_starvlm.py → QwenFast.qwen_vl_interface(**batch) → Qwen3-VL forward loss
```

---

## 2. 训练路径选型（接口核验结论）

| 决策点 | 选择 | 依据 |
|---|---|---|
| 入口 | `starVLA/training/train_starvlm.py` | 显式 PyTorch 循环 + Accelerate/DeepSpeed zero2；`_train_step` 只调 `qwen_vl_interface(**batch)`，即纯 VLM forward |
| 框架名 | `QwenFast` | 全部注册框架都强制构造 action_model，但 QwenFast 的是 **无参数** 的 FAST tokenizer 包装（`Fast_Action_Tokenizer` 仅含 AutoProcessor）→ 可训练图 == 原版 Qwen3-VL，无死参数 |
| base_vlm | `Qwen/Qwen3-VL-4B-Instruct` | 原版；`get_vlm_model` 按字符串含 "Qwen3-VL" 路由到 `_QWen3_VL_Interface`；`-Action` 分支（扩表 FAST 词表）跳过 |
| model_type | `qwen3vl` | 选 `get_rope_index_3`（Qwen3 时间戳式 MRoPE，非 2.5 的绝对时间位置） |
| dataset_py | `vlm_datasets_3d`（新增分支） | `build_dataloader` 是字面量分支而非动态 import；加了一个 elif |

**顺手修掉的上游 bug**：`model/modules/vlm/QWen3.py` 里 `attn_implementation = "sdpa"` 硬编码覆盖配置值（调试残留）——正是 [[starvla-flashattn-abi-slow]] 里"配置写 flash_attention_2 实际跑 sdpa"的另一半根因。已删，配置值生效（现环境 flash_attn 2.7.4.post1 ABI 健康）。

**运行期依赖补丁**：`Fast_Action_Tokenizer` 默认路径 `playground/Pretrained_models/fast` 是空目录 → 已建符号链接指向 HF 缓存快照（`/root/.cache/huggingface/hub/models--physical-intelligence--fast/snapshots/<hash>`）。

---

## 3. 扩展 1：变体物化器

文件：`3dvla-stage0/scripts/materialize_variant_data.py`

```
python scripts/materialize_variant_data.py --variant full \
    [--spec-dir variants_tokcal] [--limit 5000] [--seed 0] [--tag _tokcal]
→ sft/materialized/full[_tokcal][_smoke5000].jsonl + .manifest.json
```

### 3.1 采样语义
- **无放回 + 显式重复**：`take > avail` 时先整份复制 ⌊take/avail⌋ 次、余数随机采样（`sample_indices`）——重复永远是显式的，不靠有放回采样掩盖。
- 两遍式 IO：pass1 收集行偏移（过滤时才解析 JSON），pass2 按 `(file, offset)` seek 读取——fine3d 4M 行池不整载内存。
- `--limit N` 按比例缩放各切片 take（冒烟切片与正式配比同构）。

### 3.2 模板手术（轴向变体的轮级过滤）
前提（已在 3,000 条抽样上验证）：pack 里 `templates[i]` 与第 i 个 QA 对严格一一对应；首轮格式固定为 `"<image>\n"(+"<image>\n") + PREAMBLE + question`（`pack_sharegpt.py:88-101`）。

手术规则（`filter_conv`）：
1. 保留 `template ∈ filter` 的 QA 对；全不匹配 → 整条丢弃；
2. 若第 0 对被切，**前缀（全部 `<image>` 占位符 + PREAMBLE）迁移到新首问**——按 PREAMBLE 常量（从 pack_sharegpt import，单一事实源）定位切分点；
3. `templates` 列表同步裁剪。

conv 数不足规格时，缺口计入 `shortfall` → embodied 切片增补（embodied 排在处理顺序最后）。

### 3.3 输出契约
- 键名 `image`（starVLA `_build_messages` 读 `item["image"]`；我们源文件是 `images`，在此转换）；
- 相对路径 → 绝对（embodied/generic3d/general 三个根；fine3d/anchor 本就绝对）；
- 透传 `K/Ks/source/templates` + 新增 `slice`——K 是 focal 钩子的输入，slice 是审计维度；
- 全局 shuffle（固定 seed）后写盘；**自检**：逐条 `<image>` 占位符数 == 图数、抽 300 条验路径存在，任一失败 exit 1。
- sidecar manifest：各切片 take/emitted、seed、md5 前 12 位。

### 3.4 验证记录
- `full --limit 5000`：1000/500/500/2000/1000（严格 20/10/10/40/20），自检 OK；
- `trace_only --limit 3000`：600 条 fine3d 全部纯 T4、前缀完整、对齐无误。

---

## 4. 扩展 2：focal 统一加载钩子

文件：`starVLA/starVLA/dataloader/vlm_datasets_3d.py`（类 `Stage1MixtureDataset`，继承 `LazySupervisedDataset` 但不调用其 `__init__`）

### 4.1 四象限缩放政策

| 记录类型 | 判定 | 缩放 | 理由 |
|---|---|---|---|
| 单图带 K（fine3d/anchor） | ≤4 图 且 K≠null | `s = 600/fx`，**放大也执行** | F=600 是度量监督的全局不变量 |
| 多帧带 K（generic3d scannet） | ≥5 图 且 K≠null | `s = 280/fx` | 视频子制式 F_VIDEO=280：32 帧每帧 ~70 token，整条 ~2.2K（F=600 会到 1.3 万 token 爆 seq）；子制式内 F 恒定 → 度量一致性在制式内保持（与坐标双方言同构的分层论证） |
| 单图无 K（general/embodied） | ≤4 图 且 K=null | `s = min(1, √(378000/wh))` **自行封顶** | 无度量监督 + 归一化坐标缩放不变 → 缩小无害；自行封顶是因为 max_pixels 为腕视角抬高后，处理器不再约束 960×540 这类图，token 成本会静默上涨 50% |
| 多帧无 K（generic3d scannetpp） | ≥5 图 且 K=null | 对齐 72,000 px/帧 | 与 scannet 帧同面积 → 视频制式内分辨率一致 |

`Ks` 逐图对齐（长度断言），缺失时以扁平 `K` 广播——兑现 DATA_SPEC 加载器契约。

### 4.2 像素预算契约（断言，两级）

- **启动断言**：`max_pixels ≥ REQUIRED_MAX_PIXELS = 1,254,400`。
  最坏例推导：LIBERO 腕 fx=312.77 @640×480 → s=1.918 → 1228×921；Qwen3-VL **patch=16**（非 2.5 的 14）、merge=2 → smart_resize 取整到 32 的倍数 → 1232×924 = **1,138,368 px**。低于此值处理器会静默二次缩放，F=600 即告失真——这正是本项目最初选 F=600（而非 1000）要避免的失败模式，现在用断言锁死。
- **逐图断言**：统一后面积 > `max_pixels/1.06`（smart_resize 上取整余量）时，带 K 图 hard-fail（罪证信息含尺寸/tag/K），无 K 图仅计数（`img_*_procresize`）放行。

### 4.3 标签解掩码（重写，修上游 bug）

上游 `preprocess_qwen_visual` 以**裸 token 77091（"assistant"）**扫描开启监督段——用户文本中出现独立 "assistant" 一词即误开监督段。重写为：

- 锚定 **二元组 `<|im_start|> + "assistant"`**，内容从 +3 开始（im_start / assistant / \n），到 `<|im_end|>`（含）截止；
- token id 全部从 tokenizer 运行期解析（`convert_tokens_to_ids`/`encode`），不硬编码；
- 无任何 assistant 段 → raise（模板漂移的哨兵）。

**验证**：干跑对 400 条（后扩至 5,000 条，exit 0）把解掩码 span 逐一 decode，与源记录 gpt 轮文本**逐字比对**——400/400 完全一致（含 `<|im_end|>` 剥离后）。这同时实证了 `+3` 偏移对 Qwen3-VL 模板成立。

### 4.4 护栏与计数器
- **零监督护栏**：labels 在 `model_max_length` 窗口内全为 -100 → raise（基类 `__getitem__` 重试逻辑自动跳到相邻样本），杜绝全 ignore batch 的 NaN loss；
- 计数器（每 worker，`AUDIT_EVERY=500` 打印）：各象限图数、各切片 conv/token/监督 token、截断数、跳过数。

### 4.5 Dataloader 纪律
- `shuffle=False` + 物化文件预打乱：**各 rank 必须同序**，accelerate 按 batch 切分才不重不漏（上游 `random.shuffle` 每 rank 不同序，是踩过的坑的变体）；
- `drop_last=True`；`padding_side` 与 collator 的 `pad_sequence`（右 pad）一致。

---

## 5. 扩展 3：CPU 干跑审计

文件：`3dvla-stage0/scripts/stage1_dryrun_audit.py`

不占 GPU、不载权重，把物化文件喂进**真实训练数据类**，逐样本验证：

1. 标签保真（§4.3 的逐字比对）；
2. 截断计数（>8192）；
3. **统一尺寸抵达处理器**：由 `image_grid_thw` 反算像素数，与 `原图 × (600/fx)²` 比对（容差 10%，patch=16）——证明缩放没有被链路上任何环节吃掉；
4. 分切片 token 统计（喂 token 校准）；
5. 方言样例打印（人工目检五切片各一）；
6. 真 collator 过 batch 的形状检查。

**结果（n=5000 全量）**：标签保真全过、0 截断、全部网格吻合；分切片 token 均值 general 448 / embodied 658 / fine3d 802 / anchor 1423 / generic3d 2509（max 2588 ≪ 8192）。

干跑抓出的真问题（时间线）：
| 问题 | 处置 |
|---|---|
| embodied 960×540 无 K 图撞预算 | 确立"无 K 单图自行封顶"政策（§4.1 第三象限） |
| **LIBERO 腕 F=600 = 1.92× 放大 → 1228×921** | max_pixels 契约值提至 1,254,400；预算断言分级（K hard-fail / 无 K 计数） |
| 审计公式误用 patch=14 | Qwen3-VL patch=16，公式从 processor 读取 |

---

## 6. Token 校准

文件：`3dvla-stage0/scripts/token_calibrate.py` → `sft/variants_tokcal/*.json` + `sft/token_stats.json`

**动机**：对话数配比 40/20/20/10/10 下，实测 token 份额歪成 20.7/15.2/18.6/16.5/**29.0**%——generic3d（32 帧 ≈2.5K token/条）吃掉近三成算力，general 防遗忘只剩一半。

**公式**：保持对话总预算 N=1M 的量纲，令 token 份额=计划权重：
`n_i = w_i · T / m_i`，`T = N / Σ(w_i/m_i)`，m 为实测均值。

**关键不变量**：**全部变体锚定 full 变体的 T=642.5M token**（受控对照的暴露量必须相等；对话数浮动，如 none 变体 1,035K 条）。初版各变体独立解 T 会差 3.4%，已修。

结果（full）：general 573K(×1.39) / embodied 195K(×0.37) / fine3d 160K(×0.04) / anchor 45K(×0.85) / generic3d 25K(×0.10)。

**两个已记录的口径决策**（用户可否决，见 spec `caveat` 字段）：
1. anchor 在 token 口径下自然不再 ×1.89 重复（腕图贵，10% token 只需 45K 条）；
2. 轴向变体过滤后 fine3d 对话变短 → 其 token 份额偏低，跑轴向变体前需"实测-回填"一步；headline 的 full/none 不受影响。

---

## 7. 配置与启动

### 7.1 `3dvla-stage0/train/stage1.yaml` 非显然值速查

| 键 | 值 | 理由 |
|---|---|---|
| `framework.action_model.action_horizon` | 16 | VLM-only 不用，但 `QwenFast.__init__` 必读 |
| `datasets.vlm_data.max_pixels` | 1,254,400 | §4.2 契约（腕视角最坏例） |
| `model_max_length` | 8192 | 实测 p100=2,588，3× 余量、零截断 |
| `eval_interval` | 10^9 | `eval_action_model` 对 VLM-only 是 no-op |
| `trainer.learning_rate` | 仅 `base: 1e-5` | `build_param_lr_groups`：其余键是模块路径分组，全参数单组即可 |
| `freeze_modules` | `''` | 已核验：空串为假值 → 不冻结（曾疑虑 `''.split(',')` 匹配一切，实际有 truthiness 守卫） |
| `save_format` | safetensors | zero2 rank0 聚合整模 ~9GB/份 |

### 7.2 `run_stage1_smoke.sh` 封装的坑（均有前科）
- `main_process_port` 用运行时探测的空闲端口——**不许 0**（NCCL rendezvous 全 rank 死等）也不用 29500（常被占）；
- checkpoint/日志路由大盘 `/workspace/tingting/3dvla-checkpoints`（/tmp 只有 30G，训练中 ENOSPC 有前科）；
- `HF_HOME` 保持 profile 的 `/root/angli/hf_cache`（4B 权重在此）；FAST tokenizer 走本地符号链接不依赖缓存；
- `WANDB_MODE=offline` 默认；`--num_processes` 显式传（accelerate 配置文件里写死 8）。

用法：
```bash
CUDA_VISIBLE_DEVICES=1,2 NPROC=2 bash 3dvla-stage0/train/run_stage1_smoke.sh
# 变体/步数覆盖：追加 --datasets.vlm_data.annotation_path ... --trainer.max_train_steps ...
```

### 7.3 冒烟检查单（GPU 阶段待执行）
1. `print_trainable_parameters` ≈ 4B 全量（防误冻结）；
2. loss 前 100 步单调下降趋势；
3. 显存峰值（zero2、bs=2/卡、8K pad 上限）；
4. 真实吞吐——读 model_times，**不信 tqdm 累计 s/it**（warmup 假象有前科）；
5. 日志无 `[WARNING] flash_attn not installed`（证明 sdpa 修复生效）；
6. 审计计数器：`skip_zero_supervision`=0、`truncated`=0、四象限图数比例合理。

---

## 8. 改动文件清单

| 文件 | 性质 | 内容 |
|---|---|---|
| `starVLA/starVLA/dataloader/vlm_datasets_3d.py` | 新增 | §4 全部 |
| `starVLA/starVLA/dataloader/__init__.py` | +4 行 | `dataset_py == "vlm_datasets_3d"` 分支 |
| `starVLA/starVLA/model/modules/vlm/QWen3.py` | −1 行 | 删 sdpa 硬覆盖（留注释说明） |
| `starVLA/playground/Pretrained_models/fast` | 符号链接 | → HF 缓存 FAST 快照 |
| `3dvla-stage0/scripts/materialize_variant_data.py` | 新增 | §3 |
| `3dvla-stage0/scripts/stage1_dryrun_audit.py` | 新增 | §5 |
| `3dvla-stage0/scripts/token_calibrate.py` | 新增 | §6 |
| `3dvla-stage0/train/stage1.yaml` / `run_stage1_smoke.sh` | 新增 | §7 |

## 9. 已知限制 / 待办

- ~~热重启不衔接~~ → **已修（2026-07-10）：全状态无缝续训**。`train_starvlm.py` 三处扩展：
  ① `_save_checkpoint` 现同时做 `accelerator.save_state`（全 rank 各存 ZeRO 优化器分片 + 已注册的 lr 调度器 + RNG + `resume_meta.json`，旧快照自动轮转只留最新，~60G/份，可用 `trainer.save_resume_state=false` 关闭）；
  ② `trainer.resume_state=<state_dir>` 触发续训：载入优化器动量/调度器位置/步数计数（**不再有动量清零 + lr 回峰的冲击**，即 full 跑 T7 83→56→88 瞬态的根因）；
  ③ 数据流用 `accelerate.skip_first_batches` 索引级快进到已消费位置（**不再需要手工 tail 切文件**，直接用原物化文件）。
  用法：`--trainer.resume_state .../checkpoints/state_30000`（同 RUN_ID、同 annotation_path、同 world size=4；不要再设 pretrained_checkpoint；wandb 续曲线可加 `WANDB_RUN_ID=<原id> WANDB_RESUME=allow`）。
  **⚠ 测试门未过**：实现完成但 GPU 被 none 跑占用，未做中断-续训等价性冒烟（30 步存档→杀→续→与不间断参照比对 loss/lr 连续性）。none 结束后补测；补测通过前不要依赖。
- **单 epoch 设计**：物化文件顺序读一遍即一个 epoch；多 epoch 需换 seed 重物化（有意为之：epoch 重洗器是 starVLA 里没有的东西，宁缺毋滥）；
- 轴向变体 token 回填（§6 决策 2）；
- generic3d 走 image 路径而非 Qwen video 路径（时间戳 MRoPE 差异未启用；F_VIDEO 政策下 token 成本已可控，暂无必要）；
- 冒烟后：真实训练 full+none 双跑 → 训练中期 checkpoint 过 benchmark（strict/oracle 双轨）→ 遗忘审计（RefCOCO 原生方言）+ 格式回归（双方言串台率）。
