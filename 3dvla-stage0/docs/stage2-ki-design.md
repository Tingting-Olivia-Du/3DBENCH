# Stage-2 KI 装配设计（地面工作记录，2026-07-10）

目标：把 Stage-1 变体骨干（full/none/…）在**知识绝缘（KI）**下适配成 VLA——骨干经离散
FAST token 的 NTP 学动作，flow expert 在 stop-grad 后面学连续动作；Qwen 词表不动。
计划依据：plan §Stage-2 + finetune-vlm-3d-plan-v2（LabVLA 嵌入模式）。

## 1. starVLA 现状盘点（本次读码结论，含精确锚点）

| 组件 | 位置 | 现状 |
|---|---|---|
| FAST NTP 框架 | `starVLA/model/framework/VLM4A/QwenFast.py` | 可用但**非 KI**：`<robot_action_i>` 走标准 LM head 的 `qwenvl_outputs.loss`（`forward()` L164-176） |
| 动作 token 机制 | `starVLA/model/modules/vlm/QWen3.py` L21-22 | `_ACTION_TOKEN_MIN=151669, MAX=153716`（2048 个 id）；依赖 **"-Action" 检查点**（`add_qwen_special_tokens` 工具扩表：267 个空行 + resize 到 153717，新行 normal 初始化） |
| FAST 编解码 | `modules/action_model/fast_ActionHeader.py` | `encoder_action2fastoken` / `fast_tokenizer.decode`；tokenizer 本地 symlink `playground/Pretrained_models/fast` |
| GR00T 流头框架 | `framework/VLM4A/QwenGR00T.py` | `last_hidden = hidden_states[-1]`（L190）直接喂 `FlowmatchingActionHead`（L216）——**无 detach，梯度会回流骨干** |
| 流头实现 | `modules/action_model/GR00T_ActionHeader.py` | FlowmatchingActionHead，可直接复用（expert 不是研究变量） |
| LIBERO 数据 | `dataloader/gr00t_lerobot/` | lerobot 格式装载器已有 |

## 2. 为什么不用 starVLA 默认的扩表方案（决策记录）

"-Action" 检查点把 embedding/lm_head resize 到 153717 行：
1. **softmax 绝缘被破坏**——2048 个新行进入每个文本 token 的 softmax 分母，
   VL 行为被整体轻微扰动，Stage-1 能力画像不再 byte-等价（我们要做 post-Stage-2
   能力保留审计，这个扰动会污染归因）；
2. Stage-1 权重是从 **plain** Qwen3-VL-4B（151936 行）训的，grafting 时要么重训新行
   要么随机初始化——与 Stage-1 冻结变体逐字节可比的原则冲突。

→ 采用 **LabVLA KI 模式**：动作 token 走**独立小表**，Qwen 词表零改动。

## 3. KI 设计（新框架 `QwenFastGR00T_KI.py`）

输入侧：`action_embed = nn.Embedding(2048, H)`；构造 `inputs_embeds` 后，把动作
token 位置的 embedding 替换为 `action_embed(fast_id)`（动作位置在输入 ids 里可以用
占位 id 标记，或直接传位置掩码——不给 tokenizer 加任何真 token）。

输出侧：`action_head_lm = nn.Linear(H, 2048, bias=False)`；动作位置的 hidden state
过这个头算 CE（labels = fast ids），**不过** Qwen lm_head；文本位置照常（co-train
VL 数据时用原 loss）。

流头侧：`self.action_model(last_hidden.detach(), ...)`——一行 detach 即 stop-grad。

损失：`L = w_fast · CE_action + w_flow · flow_loss (+ w_vl · CE_text co-train)`。
主臂 co-train 只用 general VL 数据（计划：uniform spatial co-train 会让 none 在适配期
追平、冲掉干预效应）。

推理：动作 token 的自回归生成不走 `model.generate`（词表里没有动作 id），改为
KV-cache 手写循环：每步 hidden → `action_head_lm` → argmax fast id → `action_embed`
喂下一步；或者直接只用 flow expert 出动作（KI 的部署形态，π0.5 同款）。

## 4. 单元测试规格 —— ✅ 2026-07-10 全部通过（GPU5 实跑）

实现：框架 `starVLA/model/framework/VLM4A/QwenFastGR00T_KI.py`（注册名
`QwenFastGR00T_KI`），测试 `3dvla-stage0/stage2/test_ki_insulation.py`。
两次实跑修出的坑：① FAST tokenizer 工厂用仓库相对路径且无视 config →
框架直构 + `ki.fast_tokenizer_path` 开关；② Qwen3-VL-4B 是 tied embeddings →
"lm_head 零梯度"不可断言，改为"loss 对词表 logits 无 autograd 路径 +
占位符行零梯度"（绑定无关）；③ 位等价测试的参照模型必须用与框架相同的
attention 实现（FA2 vs sdpa 的 bf16 kernel 差可达 logits 3.9）——先比权重
再比 logits 的分离断言已加入。cotrain 循环接口核对兼容（`forward(batch_vla,
where=…)` 签名匹配；vlm 协同流走原生 lm_head，与绝缘不冲突）。

1. **stop-grad 泄漏**：只 backward flow loss → 断言骨干所有参数 grad 为 None/0，
   flow head grad 非零。（计划原文承诺"unit-tested to leak zero gradient"）
2. **词表绝缘**：只 backward CE_action → 断言 `embed_tokens.weight.grad` 与
   `lm_head.weight.grad` 为 None/0；`action_embed`/`action_head_lm` grad 非零。
3. **Stage-1 等价**：KI 框架载入 Stage-1 权重后，对一条 bench 题的文本前向 logits
   与 `eval_bench.py --model stage1` 路径逐 bit 相同（证明 grafting 未扰动 VLM）。
4. **FAST 往返**：action → fast ids → 注入/读出 → decode → action，L∞ < 量化步长。
5. **co-train 掩码**：混合 batch 中文本样本不产生 action CE、动作样本不产生文本 CE
   （各自位置掩码互斥）。

## 5. 已决与待办

- **低数据 demo 子集（已拍板 2026-07-10，选项 A）**：从 demos 0-44 采样，嵌套
  p10=5 ⊂ p25=12（25% 取 floor(12.5)），每任务确定性种子，一次冻结全臂全种子共享。
  文件 `3dvla-data/stage2/lowdata_subsets_v1.json`（md5 `3f686f6c28e81b64c2b20e568814a710`，
  脚本 `scripts/freeze_lowdata_subsets.py` 位级可复现）。暴露 caveat 已写入论文
  §Setup（中英）：关键帧级、零动作、变体间常数可差分；暴露鲁棒读数 =
  LIBERO-Plus OOD + Bridge 第二域（"A + 双评测"定稿）。彩蛋：
  `libero_10/KITCHEN_SCENE4_put_the_black_bowl...` 被 anchor 生成跳过 →
  该任务图像**完全零暴露**，可作套件内无暴露 spot check。
- **LIBERO-Plus 相机扰动 × 焦距统一（必须处理）**：相机 FOV/视角扰动改变有效 fx，
  继续用固定 fx（agentview 579.41 / wrist 312.77）做 F=600 统一 = 系统性错标定，
  会把"预处理错了"误读成"能力不鲁棒"。评测代码须**逐变体读真实 K 再统一**
  （K 来源：env 运行时读取，或 `Sylvest/libero_plus_camparam_rlds`）。
- **lerobot demo↔episode 映射 —— ✅ 已解决（2026-07-10）**：
  发现：`*_no_noops_lerobot` 丢弃了回放失败 demo（spatial 432/object 454/
  goal 428/libero_10 379，皆 <500），episode 顺序非任务优先，无原始 demo id。
  方案落地：`stage2/map_lerobot_episodes.py` 以**动作序列指纹**匹配（臂 6 维
  向量扫描锚定 + 全序列子序列验证），四套件 **100% 唯一匹配、零歧义**。
  两次失败教训记录：① 首帧状态指纹不可用（t=0 是复位臂姿，差异在物体布局）；
  ② 舍入哈希在遥操作数值的舍入刀口上失效（0.13124999 vs 0.13125）→ 容差向量扫描；
  ③ gripper 是仿射映射 demo=1−2×ep（值域 {0,1}↔{−1,1}），不是翻号。
  产物：`stage2/lerobot_episode_map_v1.json`（含各套件 bench demos 45-49 的
  episode 禁用清单：44/47/42/36 条）；子集 **re-freeze v2**
  `3dvla-data/stage2/lowdata_subsets_v2.json`（md5 33b25e46...，池 =
  lerobot-present∩0-44，最小池 26≥12，直接携带 p10/p25_episodes 供训练过滤器
  消费；v1 已删除）。过滤器经 `lerobot_datasets.py` 的 `make_dataset` 工厂钩子接入。
- Stage-2 训练循环：cotrain 入口已核对兼容（见 §4）；grad-accum 仍走 ds_config
  JSON（display bug 已知）。
- LIBERO-Plus OOD 行：仓库已克隆 `/workspace/tingting/LIBERO-plus`（替换式 libero 包，
  **必须独立 conda 环境**）；每扰动变体 1 trial 协议；`starVLA/examples/LIBERO-plus`
  已有官方示例可参照。第二域用 SimplerEnv-Bridge（真机数据，零暴露）；RoboCasa
  留 Tier-3 备胎（`examples/Robocasa_365` 基建现成）。
