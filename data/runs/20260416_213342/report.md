# 3DBENCH Run — 2026-04-16 21:33:42

**Suite:** `libero_spatial`  |  **Init states/task:** 5  |  **Models:** `qwen2.5-vl-7b internvl2-8b random`  |  **Device:** `cuda:1`

```
============================================================
[21:33:42] 3DBENCH  –  VLM Spatial Reasoning Benchmark
============================================================
[21:33:42] Suite:       libero_spatial  |  Init states/task: 5  |  Traj frames: 5  |  Close frames: 3
[21:33:42] Models:      qwen2.5-vl-7b internvl2-8b random
[21:33:42] Device:      cuda:1
[21:33:42] GT dir:      data/gt
[21:33:42] Run folder:  data/runs/20260416_213342
[21:33:42]   results.json  → data/runs/20260416_213342/results.json
[21:33:42]   report.md     → data/runs/20260416_213342/report.md
[21:33:42] (skip_gt)   using existing GT
============================================================
[21:33:42] STEP 1 skipped  (using existing data/gt/manifest.json)
============================================================
[21:33:42] STEP 2/3  VLM evaluation
Loaded 319 samples from data/gt/manifest.json
Run timestamp: 20260416_213343
Loading Qwen/Qwen2.5-VL-7B-Instruct ...
Fetching 5 files:   0%|          | 0/5 [00:00<?, ?it/s]Fetching 5 files: 100%|██████████| 5/5 [00:00<00:00, 43151.28it/s]
Loading weights:   0%|          | 0/729 [00:00<?, ?it/s]Loading weights:   0%|          | 1/729 [00:00<01:51,  6.52it/s]Loading weights:   0%|          | 2/729 [00:00<01:54,  6.36it/s]Loading weights:   2%|▏         | 16/729 [00:00<00:13, 51.17it/s]Loading weights:   5%|▌         | 40/729 [00:00<00:06, 106.73it/s]Loading weights:   7%|▋         | 54/729 [00:00<00:05, 113.50it/s]Loading weights:  11%|█         | 77/729 [00:00<00:04, 142.23it/s]Loading weights:  14%|█▎        | 100/729 [00:00<00:03, 160.27it/s]Loading weights:  16%|█▌        | 117/729 [00:00<00:03, 161.23it/s]Loading weights:  19%|█▊        | 136/729 [00:01<00:03, 158.84it/s]Loading weights:  22%|██▏       | 160/729 [00:01<00:03, 164.95it/s]Loading weights:  25%|██▍       | 180/729 [00:01<00:03, 173.43it/s]Loading weights:  27%|██▋       | 198/729 [00:01<00:03, 164.30it/s]Loading weights:  30%|███       | 221/729 [00:01<00:02, 174.34it/s]Loading weights:  33%|███▎      | 239/729 [00:01<00:02, 175.66it/s]Loading weights:  35%|███▌      | 257/729 [00:01<00:02, 169.71it/s]Loading weights:  38%|███▊      | 280/729 [00:01<00:02, 179.30it/s]Loading weights:  41%|████      | 298/729 [00:02<00:02, 171.82it/s]Loading weights:  43%|████▎     | 317/729 [00:02<00:02, 168.03it/s]Loading weights:  47%|████▋     | 343/729 [00:02<00:02, 192.80it/s]Loading weights:  74%|███████▎  | 537/729 [00:02<00:00, 678.21it/s]Loading weights: 100%|█████████▉| 726/729 [00:02<00:00, 1012.46it/s]Loading weights: 100%|██████████| 729/729 [00:02<00:00, 296.18it/s] 
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. 

============================================================
Model: qwen2.5-vl-7b  |  Run: 20260416_213343  |  Output: data/runs/20260416_213342/qwen2.5-vl-7b/20260416_213343
============================================================
qwen2.5-vl-7b:   0%|          | 0/319 [00:00<?, ?it/s]qwen2.5-vl-7b:   1%|          | 2/319 [00:00<00:15, 19.97it/s]qwen2.5-vl-7b:   8%|▊         | 25/319 [00:00<00:02, 141.11it/s]qwen2.5-vl-7b:  15%|█▌        | 49/319 [00:00<00:01, 185.50it/s]  sample 0000 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0000.png'
  sample 0001 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0001.png'
  sample 0002 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0002.png'
  sample 0003 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0003.png'
  sample 0004 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0004.png'
  sample 0005 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0005.png'
  sample 0006 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0006.png'
  sample 0007 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0007.png'
  sample 0008 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0008.png'
  sample 0009 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0009.png'
  sample 0010 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0010.png'
  sample 0011 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0011.png'
  sample 0012 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0012.png'
  sample 0013 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0013.png'
  sample 0014 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0014.png'
  sample 0015 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0015.png'
  sample 0016 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0016.png'
  sample 0017 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0017.png'
  sample 0018 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0018.png'
  sample 0019 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0019.png'
  sample 0020 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0020.png'
  sample 0021 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0021.png'
  sample 0022 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0022.png'
  sample 0023 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0023.png'
  sample 0024 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0024.png'
  sample 0025 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0025.png'
  sample 0026 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0026.png'
  sample 0027 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0027.png'
  sample 0028 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0028.png'
  sample 0029 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0029.png'
  sample 0030 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0030.png'
  sample 0031 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0031.png'
  sample 0032 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0032.png'
  sample 0033 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0033.png'
  sample 0034 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0034.png'
  sample 0035 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0035.png'
  sample 0036 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0036.png'
  sample 0037 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0037.png'
  sample 0038 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0038.png'
  sample 0039 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0039.png'
  sample 0040 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0040.png'
  sample 0041 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0041.png'
  sample 0042 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0042.png'
  sample 0043 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0043.png'
  sample 0044 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0044.png'
  sample 0045 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0045.png'
  sample 0046 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0046.png'
  sample 0047 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0047.png'
  sample 0048 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0048.png'
  sample 0049 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0049.png'
  sample 0050 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0050.png'
  sample 0051 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0051.png'
  sample 0052 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0052.png'
  sample 0053 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0053.png'
  sample 0054 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0054.png'
  sample 0055 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0055.png'
  sample 0056 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0056.png'
  sample 0057 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0057.png'
  sample 0058 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0058.png'
  sample 0059 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0059.png'
  sample 0060 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0060.png'
  sample 0061 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0061.png'
  sample 0062 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0062.png'
  sample 0063 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0063.png'qwen2.5-vl-7b:  22%|██▏       | 69/319 [00:00<00:01, 191.18it/s]qwen2.5-vl-7b:  29%|██▉       | 92/319 [00:00<00:01, 202.54it/s]qwen2.5-vl-7b:  36%|███▌      | 115/319 [00:00<00:00, 211.56it/s]
  sample 0064 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0064.png'
  sample 0065 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0065.png'
  sample 0066 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0066.png'
  sample 0067 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0067.png'
  sample 0068 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0068.png'
  sample 0069 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0069.png'
  sample 0070 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0070.png'
  sample 0071 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0071.png'
  sample 0072 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0072.png'
  sample 0073 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0073.png'
  sample 0074 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0074.png'
  sample 0075 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0075.png'
  sample 0076 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0076.png'
  sample 0077 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0077.png'
  sample 0078 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0078.png'
  sample 0079 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0079.png'
  sample 0080 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0080.png'
  sample 0081 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0081.png'
  sample 0082 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0082.png'
  sample 0083 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0083.png'
  sample 0084 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0084.png'
  sample 0085 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0085.png'
  sample 0086 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0086.png'
  sample 0087 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0087.png'
  sample 0088 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0088.png'
  sample 0089 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0089.png'
  sample 0090 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0090.png'
  sample 0091 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0091.png'
  sample 0092 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0092.png'
  sample 0093 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0093.png'
  sample 0094 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0094.png'
  sample 0095 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0095.png'
  sample 0096 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0096.png'
  sample 0097 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0097.png'
  sample 0098 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_02/sample_0098.png'
  sample 0099 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0099.png'
  sample 0100 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0100.png'
  sample 0101 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0101.png'
  sample 0102 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0102.png'
  sample 0103 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0103.png'
  sample 0104 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0104.png'
  sample 0105 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0105.png'
  sample 0106 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0106.png'
  sample 0107 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0107.png'
  sample 0108 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0108.png'
  sample 0109 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0109.png'
  sample 0110 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0110.png'
  sample 0111 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0111.png'
  sample 0112 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0112.png'
  sample 0113 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0113.png'
  sample 0114 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0114.png'
  sample 0115 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0115.png'
  sample 0116 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0116.png'
  sample 0117 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0117.png'
  sample 0118 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0118.png'
  sample 0119 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0119.png'
  sample 0120 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0120.png'
  sample 0121 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0121.png'
  sample 0122 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0122.png'
  sample 0123 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0123.png'
  sample 0124 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0124.png'
  sample 0125 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0125.png'
  sample 0126 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0126.png'
  sample 0127 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0127.png'qwen2.5-vl-7b:  44%|████▎     | 139/319 [00:00<00:00, 218.88it/s]qwen2.5-vl-7b:  50%|█████     | 161/319 [00:00<00:00, 218.64it/s]qwen2.5-vl-7b:  58%|█████▊    | 184/319 [00:00<00:00, 219.72it/s]
  sample 0128 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0128.png'
  sample 0129 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0129.png'
  sample 0130 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0130.png'
  sample 0131 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0131.png'
  sample 0132 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0132.png'
  sample 0133 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0133.png'
  sample 0134 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0134.png'
  sample 0135 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0135.png'
  sample 0136 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0136.png'
  sample 0137 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0137.png'
  sample 0138 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0138.png'
  sample 0139 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0139.png'
  sample 0140 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0140.png'
  sample 0141 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0141.png'
  sample 0142 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0142.png'
  sample 0143 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0143.png'
  sample 0144 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0144.png'
  sample 0145 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0145.png'
  sample 0146 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0146.png'
  sample 0147 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0147.png'
  sample 0148 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0148.png'
  sample 0149 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0149.png'
  sample 0150 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0150.png'
  sample 0151 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0151.png'
  sample 0152 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0152.png'
  sample 0153 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0153.png'
  sample 0154 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0154.png'
  sample 0155 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0155.png'
  sample 0156 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0156.png'
  sample 0157 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0157.png'
  sample 0158 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0158.png'
  sample 0159 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0159.png'
  sample 0160 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0160.png'
  sample 0161 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_04/sample_0161.png'
  sample 0162 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0162.png'
  sample 0163 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0163.png'
  sample 0164 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0164.png'
  sample 0165 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0165.png'
  sample 0166 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0166.png'
  sample 0167 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0167.png'
  sample 0168 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0168.png'
  sample 0169 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0169.png'
  sample 0170 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0170.png'
  sample 0171 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0171.png'
  sample 0172 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0172.png'
  sample 0173 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0173.png'
  sample 0174 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0174.png'
  sample 0175 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0175.png'
  sample 0176 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0176.png'
  sample 0177 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0177.png'
  sample 0178 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0178.png'
  sample 0179 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0179.png'
  sample 0180 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0180.png'
  sample 0181 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0181.png'
  sample 0182 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0182.png'
  sample 0183 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0183.png'
  sample 0184 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0184.png'
  sample 0185 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0185.png'
  sample 0186 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0186.png'
  sample 0187 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0187.png'
  sample 0188 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0188.png'
  sample 0189 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0189.png'
  sample 0190 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0190.png'
  sample 0191 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0191.png'qwen2.5-vl-7b:  65%|██████▍   | 207/319 [00:01<00:00, 220.97it/s]qwen2.5-vl-7b:  72%|███████▏  | 230/319 [00:01<00:00, 217.71it/s]qwen2.5-vl-7b:  79%|███████▉  | 252/319 [00:01<00:00, 201.50it/s]
  sample 0192 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0192.png'
  sample 0193 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0193.png'
  sample 0194 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0194.png'
  sample 0195 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0195.png'
  sample 0196 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0196.png'
  sample 0197 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0197.png'
  sample 0198 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0198.png'
  sample 0199 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0199.png'
  sample 0200 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0200.png'
  sample 0201 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0201.png'
  sample 0202 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0202.png'
  sample 0203 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0203.png'
  sample 0204 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0204.png'
  sample 0205 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0205.png'
  sample 0206 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0206.png'
  sample 0207 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0207.png'
  sample 0208 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0208.png'
  sample 0209 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0209.png'
  sample 0210 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0210.png'
  sample 0211 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0211.png'
  sample 0212 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0212.png'
  sample 0213 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0213.png'
  sample 0214 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0214.png'
  sample 0215 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0215.png'
  sample 0216 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0216.png'
  sample 0217 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0217.png'
  sample 0218 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0218.png'
  sample 0219 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0219.png'
  sample 0220 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0220.png'
  sample 0221 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0221.png'
  sample 0222 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0222.png'
  sample 0223 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0223.png'
  sample 0224 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_06/sample_0224.png'
  sample 0225 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0225.png'
  sample 0226 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0226.png'
  sample 0227 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0227.png'
  sample 0228 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0228.png'
  sample 0229 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0229.png'
  sample 0230 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0230.png'
  sample 0231 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0231.png'
  sample 0232 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0232.png'
  sample 0233 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0233.png'
  sample 0234 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0234.png'
  sample 0235 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0235.png'
  sample 0236 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0236.png'
  sample 0237 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0237.png'
  sample 0238 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0238.png'
  sample 0239 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0239.png'
  sample 0240 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0240.png'
  sample 0241 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0241.png'
  sample 0242 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0242.png'
  sample 0243 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0243.png'
  sample 0244 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0244.png'
  sample 0245 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0245.png'
  sample 0246 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0246.png'
  sample 0247 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0247.png'
  sample 0248 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0248.png'
  sample 0249 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0249.png'
  sample 0250 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0250.png'
  sample 0251 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0251.png'
  sample 0252 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0252.png'
  sample 0253 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0253.png'
  sample 0254 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0254.png'
  sample 0255 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0255.png'qwen2.5-vl-7b:  86%|████████▌ | 273/319 [00:01<00:00, 203.23it/s]qwen2.5-vl-7b:  92%|█████████▏| 295/319 [00:01<00:00, 207.46it/s]qwen2.5-vl-7b: 100%|██████████| 319/319 [00:01<00:00, 215.32it/s]qwen2.5-vl-7b: 100%|██████████| 319/319 [00:01<00:00, 204.58it/s]

  sample 0256 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0256.png'
  sample 0257 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0257.png'
  sample 0258 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0258.png'
  sample 0259 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0259.png'
  sample 0260 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0260.png'
  sample 0261 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0261.png'
  sample 0262 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0262.png'
  sample 0263 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0263.png'
  sample 0264 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0264.png'
  sample 0265 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0265.png'
  sample 0266 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0266.png'
  sample 0267 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0267.png'
  sample 0268 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0268.png'
  sample 0269 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0269.png'
  sample 0270 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0270.png'
  sample 0271 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0271.png'
  sample 0272 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0272.png'
  sample 0273 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0273.png'
  sample 0274 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0274.png'
  sample 0275 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0275.png'
  sample 0276 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0276.png'
  sample 0277 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0277.png'
  sample 0278 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0278.png'
  sample 0279 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0279.png'
  sample 0280 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0280.png'
  sample 0281 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0281.png'
  sample 0282 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0282.png'
  sample 0283 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0283.png'
  sample 0284 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0284.png'
  sample 0285 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0285.png'
  sample 0286 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0286.png'
  sample 0287 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0287.png'
  sample 0288 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_08/sample_0288.png'
  sample 0289 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0289.png'
  sample 0290 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0290.png'
  sample 0291 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0291.png'
  sample 0292 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0292.png'
  sample 0293 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0293.png'
  sample 0294 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0294.png'
  sample 0295 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0295.png'
  sample 0296 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0296.png'
  sample 0297 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0297.png'
  sample 0298 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0298.png'
  sample 0299 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0299.png'
  sample 0300 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0300.png'
  sample 0301 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0301.png'
  sample 0302 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0302.png'
  sample 0303 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0303.png'
  sample 0304 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0304.png'
  sample 0305 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0305.png'
  sample 0306 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0306.png'
  sample 0307 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0307.png'
  sample 0308 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0308.png'
  sample 0309 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0309.png'
  sample 0310 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0310.png'
  sample 0311 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0311.png'
  sample 0312 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0312.png'
  sample 0313 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0313.png'
  sample 0314 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0314.png'
  sample 0315 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0315.png'
  sample 0316 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0316.png'
  sample 0317 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0317.png'
  sample 0318 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_09/sample_0318.png'
Done: 0 ok  |  319 failed  |  0 skipped
GPU cache cleared.
Loading OpenGVLab/InternVL2-8B ...`torch_dtype` is deprecated! Use `dtype` instead!
Unrecognized keys in `rope_parameters` for 'rope_type'='dynamic': {'rope_theta'}
/umd-datapool/tingting/envs/vlmbench/lib/python3.12/site-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)

Fetching 4 files:   0%|          | 0/4 [00:00<?, ?it/s]