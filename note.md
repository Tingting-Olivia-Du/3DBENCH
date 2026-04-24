
conda activate /umd-datapool/tingting/envs/vlmbench

ALL_MODELS="qwen2.5-vl-3b qwen2.5-vl-7b qwen3-vl-2b qwen3-vl-4b qwen3-vl-8b qwen3-vl-30b-a3b paligemma-1 paligemma-2 kosmos-2 internvl2-8b random"

# 提前下载所有模型到本地（必须在 vlmbench 环境里跑，否则跑 benchmark 时会卡在网络请求）
# 模型保存到 /umd-datapool/tingting/models/，benchmark 脚本会自动优先使用本地路径
python -c "
from huggingface_hub import snapshot_download
models = [
    ('Qwen/Qwen2.5-VL-3B-Instruct',          'Qwen2.5-VL-3B-Instruct'),
    ('Qwen/Qwen2.5-VL-7B-Instruct',          'Qwen2.5-VL-7B-Instruct'),
    ('Qwen/Qwen3-VL-2B-Instruct',            'Qwen3-VL-2B-Instruct'),
    ('Qwen/Qwen3-VL-4B-Instruct',            'Qwen3-VL-4B-Instruct'),
    ('Qwen/Qwen3-VL-8B-Instruct',            'Qwen3-VL-8B-Instruct'),
    ('Qwen/Qwen3-VL-30B-A3B-Instruct',       'Qwen3-VL-30B-A3B-Instruct'),
    ('google/paligemma-3b-mix-448',           'paligemma-3b-mix-448'),
    ('google/paligemma2-3b-mix-448',          'paligemma2-3b-mix-448'),
    ('microsoft/kosmos-2-patch14-224',        'kosmos-2-patch14-224'),
    ('OpenGVLab/InternVL2-8B',               'InternVL2-8B'),
]
root = '/umd-datapool/tingting/models'
for repo, name in models:
    local = f'{root}/{name}'
    print(f'Downloading {repo} ...')
    snapshot_download(repo_id=repo, local_dir=local)
    print(f'  -> {local}')
"

python -c "
from huggingface_hub import snapshot_download
models = [
    ('Qwen/Qwen3-VL-30B-A3B-Instruct',       'Qwen3-VL-30B-A3B-Instruct'),
    ('google/paligemma-3b-mix-448',           'paligemma-3b-mix-448'),
    ('google/paligemma2-3b-mix-448',          'paligemma2-3b-mix-448'),
    ('microsoft/kosmos-2-patch14-224',        'kosmos-2-patch14-224'),
    ('OpenGVLab/InternVL2-8B',               'InternVL2-8B'),
]
root = '/umd-datapool/tingting/models'
for repo, name in models:
    local = f'{root}/{name}'
    print(f'Downloading {repo} ...')
    snapshot_download(repo_id=repo, local_dir=local)
    print(f'  -> {local}')
"

# 完整跑一遍（提取GT + 评测 + 算指标）
bash run_benchmark.sh --device cuda:0 --models "$ALL_MODELS"

# GT 已经有了，只跑新模型
bash run_benchmark.sh --skip_gt --models "$ALL_MODELS" --device cuda:1

bash run_benchmark.sh --skip_gt --resume --models "$ALL_MODELS" --device cuda:1

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


 

# 激活环境
source /umd-datapool/tingting/envs/vlmbench/bin/activate
# 或者
/umd-datapool/tingting/envs/vlmbench/bin/python 对应 python

# 直接跑（libero 已经装好，不需要 --libero_path）

bash run_benchmark.sh \
    --suite libero_object \
    --skip_gt
    --gt_dir data/gt-q6 \
    --device cuda:0 \
    --models "qwen3-vl-2b qwen3-vl-4b qwen3-vl-8b qwen2.5-vl-3b qwen2.5-vl-7b"


bash run_benchmark.sh \
    --suite libero_goal \
    --n_states 5 \
    --n_traj_frames 8 \
    --n_close 5 \
    --gt_dir data/gt-q6 \
    --device cuda:1 \
    --models "qwen3-vl-2b qwen3-vl-4b qwen3-vl-8b qwen2.5-vl-3b qwen2.5-vl-7b"


bash run_benchmark.sh \
    --suite libero_spatial \
    --n_states 5 \
    --n_traj_frames 8 \
    --n_close 5 \
    --gt_dir data/gt-q6 \
    --device cuda:2 \
    --models "qwen3-vl-2b qwen3-vl-4b qwen3-vl-8b qwen2.5-vl-3b qwen2.5-vl-7b"


bash run_benchmark.sh \
    --suite libero_10 \
    --n_states 5 \
    --n_traj_frames 8 \
    --n_close 5 \
    --gt_dir data/gt-q6 \
    --device cuda:3 \
    --models "qwen3-vl-2b qwen3-vl-4b qwen3-vl-8b qwen2.5-vl-3b qwen2.5-vl-7b"


bash run_benchmark.sh --suite "libero_10,libero_goal,libero_spatial,libero_object" --skip_gt --models "qwen3-vl-2b" --device cuda:4

bash run_benchmark.sh --suite "libero_10,libero_goal,libero_spatial,libero_object" --skip_gt --models "qwen3-vl-4b" --device cuda:5


bash run_benchmark.sh --suite "libero_10,libero_goal,libero_spatial,libero_object" --skip_gt --models "qwen3-vl-8b" --device cuda:6




bash run_benchmark.sh --suite "libero_10,libero_goal,libero_spatial,libero_object" --skip_gt --models "qwen2.5-vl-3b" --device cuda:7


bash run_benchmark.sh --suite "libero_10,libero_goal,libero_spatial,libero_object" --skip_gt --models "qwen2.5-vl-7b" --device cuda:0


bash run_benchmark.sh --suite "libero_10,libero_goal,libero_spatial,libero_object" --skip_gt --models "qwen3-vl-30b-a3b" --device cuda:2


conda activate /umd-datapool/tingting/envs/vlmbench
