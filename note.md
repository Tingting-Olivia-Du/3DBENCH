
conda activate /umd-datapool/tingting/envs/vlmbench

# 完整跑一遍（提取GT + 评测 + 算指标）
bash run_benchmark.sh --device cuda:0 --models "qwen2.5-vl-7b internvl2-8b random" 

# GT 已经有了，只跑新模型
bash run_benchmark.sh --skip_gt --models "qwen2.5-vl-7b internvl2-8b random" --device cuda:1
bash run_benchmark.sh --skip_gt --resume --models "qwen2.5-vl-7b internvl2-8b random" --device cuda:1

# 断点续跑（跳过已完成的样本）
bash run_benchmark.sh --skip_gt --resume --models "qwen2.5-vl-7b"

# 只重算指标（两步都跳过）
bash run_benchmark.sh --skip_gt --skip_eval

# 采轨迹帧（每个 init state 沿接近路径均匀采 8 帧）
bash run_benchmark.sh --n_traj_frames 8 --n_close 0

# 丰富数据集：5 个 init 帧 + 每个 init state 采 8 帧轨迹 + 3 帧 can_close
bash run_benchmark.sh --n_states 5 --n_traj_frames 8 --n_close 3

# 自定义输出目录（不同实验互不干扰）
bash run_benchmark.sh \
    --gt_dir   data/gt-spatial-10 \
    --resp_dir data/responses-0416 \
    --results  data/results-0416.json \
    --n_states 10 \
    --device   cuda:0
