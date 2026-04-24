# 3DBENCH Run — 2026-04-16 11:03:26

**Suite:** `libero_spatial`  |  **Init states/task:** 5  |  **Models:** `qwen2.5-vl-7b`  |  **Device:** `cuda:0`

```
============================================================
[11:03:26] 3DBENCH  –  VLM Spatial Reasoning Benchmark
============================================================
[11:03:26] Suite:       libero_spatial  |  Init states/task: 5  |  Traj frames: 5  |  Close frames: 3
[11:03:26] Models:      qwen2.5-vl-7b
[11:03:26] Device:      cuda:0
[11:03:26] GT dir:      data/gt
[11:03:26] Run folder:  data/runs/20260416_110326
[11:03:26]   results.json  → data/runs/20260416_110326/results.json
[11:03:26]   report.md     → data/runs/20260416_110326/report.md
============================================================
[11:03:26] STEP 1/3  Extract ground truth from LIBERO sim
[robosuite WARNING] No private macro file found! (__init__.py:7)
[robosuite WARNING] It is recommended to use a private macro file (__init__.py:8)
[robosuite WARNING] To setup, run: python /data/miniconda3/envs/vlmbench/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (__init__.py:9)
Suites to extract: ['libero_spatial']
[info] using task orders [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

============================================================
Suite: libero_spatial  |  Tasks: 10/10  |  Init states/task: 5  |  Output: data/gt

[Task 00] pick up the black bowl between the plate and the ramekin and place it on the plate
Local assets not found. Downloading from HuggingFace Hub...
Assets already downloaded at /root/.cache/libero/assets
  Task 0 (init):   0%|          | 0/5 [00:00<?, ?it/s]  Task 0 (init):  20%|██        | 1/5 [00:00<00:03,  1.10it/s]  Task 0 (init):  40%|████      | 2/5 [00:01<00:02,  1.11it/s]  Task 0 (init):  60%|██████    | 3/5 [00:02<00:01,  1.07it/s]  Task 0 (init):  80%|████████  | 4/5 [00:03<00:00,  1.09it/s]  Task 0 (init): 100%|██████████| 5/5 [00:04<00:00,  1.09it/s]                                                                                  sample 0000 | state 00 | EE=[0.449,-0.011,0.262] | target=akita_black_bowl_1_main dist=0.379m can_close=False
                    sample 0001 | state 12 | EE=[0.453,-0.005,0.263] | target=akita_black_bowl_1_main dist=0.370m can_close=False
                    sample 0002 | state 24 | EE=[0.460,0.006,0.265] | target=akita_black_bowl_1_main dist=0.359m can_close=False
                    sample 0003 | state 36 | EE=[0.447,-0.003,0.254] | target=akita_black_bowl_1_main dist=0.379m can_close=False
                    sample 0004 | state 49 | EE=[0.447,0.005,0.251] | target=akita_black_bowl_1_main dist=0.356m can_close=False
  Task 0 (traj):   0%|          | 0/5 [00:00<?, ?it/s]  Task 0 (traj):  20%|██        | 1/5 [00:01<00:06,  1.52s/it]  Task 0 (traj):  40%|████      | 2/5 [00:03<00:04,  1.51s/it]  Task 0 (traj):  60%|██████    | 3/5 [00:04<00:02,  1.50s/it]  Task 0 (traj):  80%|████████  | 4/5 [00:06<00:01,  1.55s/it]  Task 0 (traj): 100%|██████████| 5/5 [00:07<00:00,  1.52s/it]                                                                  [TRAJ 000/034] sample 0005 | state 00 | EE=[0.449,-0.011,0.262] | target=akita_black_bowl_1_main dist=0.379m can_close=False
    [TRAJ 008/034] sample 0006 | state 00 | EE=[0.473,0.031,0.202] | target=akita_black_bowl_1_main dist=0.302m can_close=False
    [TRAJ 016/034] sample 0007 | state 00 | EE=[0.504,0.079,0.139] | target=akita_black_bowl_1_main dist=0.217m can_close=False
    [TRAJ 024/034] sample 0008 | state 00 | EE=[0.536,0.126,0.077] | target=akita_black_bowl_1_main dist=0.133m can_close=False
    [TRAJ 033/034] sample 0009 | state 00 | EE=[0.576,0.180,0.010] | target=akita_black_bowl_1_main dist=0.038m can_close=True
    [TRAJ 000/033] sample 0010 | state 12 | EE=[0.453,-0.005,0.263] | target=akita_black_bowl_1_main dist=0.370m can_close=False
    [TRAJ 008/033] sample 0011 | state 12 | EE=[0.479,0.034,0.202] | target=akita_black_bowl_1_main dist=0.293m can_close=False
    [TRAJ 016/033] sample 0012 | state 12 | EE=[0.511,0.078,0.136] | target=akita_black_bowl_1_main dist=0.208m can_close=False
    [TRAJ 024/033] sample 0013 | state 12 | EE=[0.545,0.122,0.073] | target=akita_black_bowl_1_main dist=0.124m can_close=False
    [TRAJ 032/033] sample 0014 | state 12 | EE=[0.582,0.167,0.012] | target=akita_black_bowl_1_main dist=0.039m can_close=True
    [TRAJ 000/032] sample 0015 | state 24 | EE=[0.460,0.006,0.265] | target=akita_black_bowl_1_main dist=0.359m can_close=False
    [TRAJ 007/032] sample 0016 | state 24 | EE=[0.476,0.041,0.211] | target=akita_black_bowl_1_main dist=0.293m can_close=False
    [TRAJ 015/032] sample 0017 | state 24 | EE=[0.499,0.088,0.143] | target=akita_black_bowl_1_main dist=0.208m can_close=False
    [TRAJ 023/032] sample 0018 | state 24 | EE=[0.524,0.134,0.076] | target=akita_black_bowl_1_main dist=0.123m can_close=False
    [TRAJ 031/032] sample 0019 | state 24 | EE=[0.552,0.181,0.012] | target=akita_black_bowl_1_main dist=0.039m can_close=True
    [TRAJ 000/034] sample 0020 | state 36 | EE=[0.447,-0.003,0.254] | target=akita_black_bowl_1_main dist=0.379m can_close=False
    [TRAJ 008/034] sample 0021 | state 36 | EE=[0.473,0.040,0.195] | target=akita_black_bowl_1_main dist=0.302m can_close=False
    [TRAJ 016/034] sample 0022 | state 36 | EE=[0.506,0.088,0.133] | target=akita_black_bowl_1_main dist=0.218m can_close=False
    [TRAJ 024/034] sample 0023 | state 36 | EE=[0.541,0.135,0.074] | target=akita_black_bowl_1_main dist=0.134m can_close=False
    [TRAJ 033/034] sample 0024 | state 36 | EE=[0.583,0.189,0.010] | target=akita_black_bowl_1_main dist=0.039m can_close=True
    [TRAJ 000/032] sample 0025 | state 49 | EE=[0.447,0.005,0.251] | target=akita_black_bowl_1_main dist=0.356m can_close=False
    [TRAJ 007/032] sample 0026 | state 49 | EE=[0.469,0.038,0.197] | target=akita_black_bowl_1_main dist=0.290m can_close=False
    [TRAJ 015/032] sample 0027 | state 49 | EE=[0.502,0.081,0.132] | target=akita_black_bowl_1_main dist=0.205m can_close=False
    [TRAJ 023/032] sample 0028 | state 49 | EE=[0.539,0.124,0.069] | target=akita_black_bowl_1_main dist=0.121m can_close=False
    [TRAJ 031/032] sample 0029 | state 49 | EE=[0.578,0.167,0.010] | target=akita_black_bowl_1_main dist=0.036m can_close=True
  Task 0 (close):   0%|          | 0/3 [00:00<?, ?it/s]  Task 0 (close):  33%|███▎      | 1/3 [00:01<00:02,  1.33s/it]  Task 0 (close):  67%|██████▋   | 2/3 [00:02<00:01,  1.31s/it]  Task 0 (close): 100%|██████████| 3/3 [00:03<00:00,  1.32s/it]                                                                   [CLOSE]         sample 0030 | state 00 | EE=[0.576,0.180,0.010] | target=akita_black_bowl_1_main dist=0.038m can_close=True
    [CLOSE]         sample 0031 | state 24 | EE=[0.552,0.181,0.012] | target=akita_black_bowl_1_main dist=0.039m can_close=True
    [CLOSE]         sample 0032 | state 49 | EE=[0.578,0.167,0.010] | target=akita_black_bowl_1_main dist=0.036m can_close=True

[Task 01] pick up the black bowl next to the ramekin and place it on the plate
  Task 1 (init):   0%|          | 0/5 [00:00<?, ?it/s]  Task 1 (init):  20%|██        | 1/5 [00:00<00:03,  1.15it/s]  Task 1 (init):  40%|████      | 2/5 [00:01<00:02,  1.14it/s]  Task 1 (init):  60%|██████    | 3/5 [00:02<00:01,  1.14it/s]  Task 1 (init):  80%|████████  | 4/5 [00:03<00:00,  1.14it/s]  Task 1 (init): 100%|██████████| 5/5 [00:04<00:00,  1.15it/s]                                                                                  sample 0033 | state 00 | EE=[0.451,0.005,0.273] | target=akita_black_bowl_1_main dist=0.422m can_close=False
                    sample 0034 | state 12 | EE=[0.453,0.003,0.258] | target=akita_black_bowl_1_main dist=0.430m can_close=False
                    sample 0035 | state 24 | EE=[0.453,-0.004,0.268] | target=akita_black_bowl_1_main dist=0.436m can_close=False
                    sample 0036 | state 36 | EE=[0.459,-0.000,0.277] | target=akita_black_bowl_1_main dist=0.440m can_close=False
                    sample 0037 | state 49 | EE=[0.457,0.000,0.261] | target=akita_black_bowl_1_main dist=0.420m can_close=False
  Task 1 (traj):   0%|          | 0/5 [00:00<?, ?it/s]  Task 1 (traj):  20%|██        | 1/5 [00:01<00:06,  1.53s/it]  Task 1 (traj):  40%|████      | 2/5 [00:03<00:04,  1.55s/it]  Task 1 (traj):  60%|██████    | 3/5 [00:04<00:03,  1.59s/it]  Task 1 (traj):  80%|████████  | 4/5 [00:06<00:01,  1.61s/it]  Task 1 (traj): 100%|██████████| 5/5 [00:07<00:00,  1.59s/it]                                                                  [TRAJ 000/039] sample 0038 | state 00 | EE=[0.451,0.005,0.273] | target=akita_black_bowl_1_main dist=0.422m can_close=False
    [TRAJ 009/039] sample 0039 | state 00 | EE=[0.454,0.068,0.214] | target=akita_black_bowl_1_main dist=0.336m can_close=False
    [TRAJ 019/039] sample 0040 | state 00 | EE=[0.452,0.144,0.140] | target=akita_black_bowl_1_main dist=0.230m can_close=False
    [TRAJ 028/039] sample 0041 | state 00 | EE=[0.453,0.212,0.074] | target=akita_black_bowl_1_main dist=0.136m can_close=False
    [TRAJ 038/039] sample 0042 | state 00 | EE=[0.458,0.286,0.006] | target=akita_black_bowl_1_main dist=0.036m can_close=True
    [TRAJ 000/040] sample 0043 | state 12 | EE=[0.453,0.003,0.258] | target=akita_black_bowl_1_main dist=0.430m can_close=False
    [TRAJ 009/040] sample 0044 | state 12 | EE=[0.460,0.068,0.202] | target=akita_black_bowl_1_main dist=0.343m can_close=False
    [TRAJ 019/040] sample 0045 | state 12 | EE=[0.465,0.149,0.133] | target=akita_black_bowl_1_main dist=0.238m can_close=False
    [TRAJ 029/040] sample 0046 | state 12 | EE=[0.474,0.228,0.065] | target=akita_black_bowl_1_main dist=0.133m can_close=False
    [TRAJ 039/040] sample 0047 | state 12 | EE=[0.487,0.300,0.006] | target=akita_black_bowl_1_main dist=0.040m can_close=True
    [TRAJ 000/043] sample 0048 | state 24 | EE=[0.453,-0.004,0.268] | target=akita_black_bowl_1_main dist=0.436m can_close=False
    [TRAJ 010/043] sample 0049 | state 24 | EE=[0.459,0.069,0.204] | target=akita_black_bowl_1_main dist=0.339m can_close=False
    [TRAJ 021/043] sample 0050 | state 24 | EE=[0.463,0.156,0.127] | target=akita_black_bowl_1_main dist=0.223m can_close=False
    [TRAJ 031/043] sample 0051 | state 24 | EE=[0.471,0.234,0.058] | target=akita_black_bowl_1_main dist=0.118m can_close=False
    [TRAJ 042/043] sample 0052 | state 24 | EE=[0.488,0.299,0.011] | target=akita_black_bowl_1_main dist=0.037m can_close=True
    [TRAJ 000/043] sample 0053 | state 36 | EE=[0.459,-0.000,0.277] | target=akita_black_bowl_1_main dist=0.440m can_close=False
    [TRAJ 010/043] sample 0054 | state 36 | EE=[0.463,0.071,0.212] | target=akita_black_bowl_1_main dist=0.343m can_close=False
    [TRAJ 021/043] sample 0055 | state 36 | EE=[0.464,0.157,0.133] | target=akita_black_bowl_1_main dist=0.227m can_close=False
    [TRAJ 031/043] sample 0056 | state 36 | EE=[0.468,0.234,0.062] | target=akita_black_bowl_1_main dist=0.122m can_close=False
    [TRAJ 042/043] sample 0057 | state 36 | EE=[0.481,0.299,0.010] | target=akita_black_bowl_1_main dist=0.038m can_close=True
    [TRAJ 000/039] sample 0058 | state 49 | EE=[0.457,0.000,0.261] | target=akita_black_bowl_1_main dist=0.420m can_close=False
    [TRAJ 009/039] sample 0059 | state 49 | EE=[0.462,0.064,0.203] | target=akita_black_bowl_1_main dist=0.334m can_close=False
    [TRAJ 019/039] sample 0060 | state 49 | EE=[0.465,0.143,0.132] | target=akita_black_bowl_1_main dist=0.228m can_close=False
    [TRAJ 028/039] sample 0061 | state 49 | EE=[0.471,0.213,0.069] | target=akita_black_bowl_1_main dist=0.134m can_close=False
    [TRAJ 038/039] sample 0062 | state 49 | EE=[0.482,0.287,0.007] | target=akita_black_bowl_1_main dist=0.037m can_close=True
  Task 1 (close):   0%|          | 0/3 [00:00<?, ?it/s]  Task 1 (close):  33%|███▎      | 1/3 [00:01<00:02,  1.44s/it]  Task 1 (close):  67%|██████▋   | 2/3 [00:02<00:01,  1.47s/it]  Task 1 (close): 100%|██████████| 3/3 [00:04<00:00,  1.45s/it]                                                                   [CLOSE]         sample 0063 | state 00 | EE=[0.458,0.286,0.006] | target=akita_black_bowl_1_main dist=0.036m can_close=True
    [CLOSE]         sample 0064 | state 24 | EE=[0.488,0.299,0.011] | target=akita_black_bowl_1_main dist=0.037m can_close=True
    [CLOSE]         sample 0065 | state 49 | EE=[0.482,0.287,0.007] | target=akita_black_bowl_1_main dist=0.037m can_close=True

[Task 02] pick up the black bowl from table center and place it on the plate
  Task 2 (init):   0%|          | 0/5 [00:00<?, ?it/s]  Task 2 (init):  20%|██        | 1/5 [00:00<00:03,  1.19it/s]  Task 2 (init):  40%|████      | 2/5 [00:01<00:02,  1.20it/s]  Task 2 (init):  60%|██████    | 3/5 [00:02<00:01,  1.21it/s]  Task 2 (init):  80%|████████  | 4/5 [00:03<00:00,  1.21it/s]  Task 2 (init): 100%|██████████| 5/5 [00:04<00:00,  1.21it/s]                                                                                  sample 0066 | state 00 | EE=[0.451,-0.004,0.273] | target=akita_black_bowl_1_main dist=0.317m can_close=False
                    sample 0067 | state 12 | EE=[0.453,0.004,0.266] | target=akita_black_bowl_1_main dist=0.309m can_close=False
                    sample 0068 | state 24 | EE=[0.456,0.012,0.274] | target=akita_black_bowl_1_main dist=0.315m can_close=False
                    sample 0069 | state 36 | EE=[0.457,-0.001,0.274] | target=akita_black_bowl_1_main dist=0.315m can_close=False
                    sample 0070 | state 49 | EE=[0.456,-0.007,0.265] | target=akita_black_bowl_1_main dist=0.307m can_close=False
  Task 2 (traj):   0%|          | 0/5 [00:00<?, ?it/s]  Task 2 (traj):  20%|██        | 1/5 [00:01<00:06,  1.72s/it]  Task 2 (traj):  40%|████      | 2/5 [00:05<00:08,  2.72s/it]  Task 2 (traj):  60%|██████    | 3/5 [00:08<00:05,  2.92s/it]  Task 2 (traj):  80%|████████  | 4/5 [00:10<00:02,  2.70s/it]  Task 2 (traj): 100%|██████████| 5/5 [00:13<00:00,  2.86s/it]                                                                  [TRAJ 000/045] sample 0071 | state 00 | EE=[0.451,-0.004,0.273] | target=akita_black_bowl_1_main dist=0.317m can_close=False
    [TRAJ 011/045] sample 0072 | state 00 | EE=[0.490,0.002,0.170] | target=akita_black_bowl_1_main dist=0.207m can_close=False
    [TRAJ 022/045] sample 0073 | state 00 | EE=[0.539,0.009,0.063] | target=akita_black_bowl_1_main dist=0.089m can_close=False
    [TRAJ 033/045] sample 0074 | state 00 | EE=[0.594,0.017,0.041] | target=akita_black_bowl_1_main dist=0.047m can_close=False
    [TRAJ 044/045] sample 0075 | state 00 | EE=[0.615,0.015,0.021] | target=akita_black_bowl_1_main dist=0.039m can_close=True
    [TRAJ 000/115] sample 0076 | state 12 | EE=[0.453,0.004,0.266] | target=akita_black_bowl_1_main dist=0.309m can_close=False
    [TRAJ 028/115] sample 0077 | state 12 | EE=[0.556,-0.000,0.042] | target=akita_black_bowl_1_main dist=0.065m can_close=False
    [TRAJ 057/115] sample 0078 | state 12 | EE=[0.774,-0.002,0.045] | target=akita_black_bowl_1_main dist=0.062m can_close=False
    [TRAJ 085/115] sample 0079 | state 12 | EE=[0.796,-0.002,0.042] | target=akita_black_bowl_1_main dist=0.059m can_close=False
    [TRAJ 114/115] sample 0080 | state 12 | EE=[0.812,0.003,0.025] | target=akita_black_bowl_1_main dist=0.038m can_close=True
    [TRAJ 000/104] sample 0081 | state 24 | EE=[0.456,0.012,0.274] | target=akita_black_bowl_1_main dist=0.315m can_close=False
    [TRAJ 025/104] sample 0082 | state 24 | EE=[0.552,0.010,0.039] | target=akita_black_bowl_1_main dist=0.065m can_close=False
    [TRAJ 051/104] sample 0083 | state 24 | EE=[0.740,0.012,0.047] | target=akita_black_bowl_1_main dist=0.062m can_close=False
    [TRAJ 077/104] sample 0084 | state 24 | EE=[0.792,0.012,0.043] | target=akita_black_bowl_1_main dist=0.059m can_close=False
    [TRAJ 103/104] sample 0085 | state 24 | EE=[0.807,0.017,0.026] | target=akita_black_bowl_1_main dist=0.039m can_close=True
    [TRAJ 000/077] sample 0086 | state 36 | EE=[0.457,-0.001,0.274] | target=akita_black_bowl_1_main dist=0.315m can_close=False
    [TRAJ 019/077] sample 0087 | state 36 | EE=[0.528,-0.001,0.091] | target=akita_black_bowl_1_main dist=0.119m can_close=False
    [TRAJ 038/077] sample 0088 | state 36 | EE=[0.649,-0.003,0.056] | target=akita_black_bowl_1_main dist=0.063m can_close=False
    [TRAJ 057/077] sample 0089 | state 36 | EE=[0.653,-0.004,0.031] | target=akita_black_bowl_1_main dist=0.048m can_close=False
    [TRAJ 076/077] sample 0090 | state 36 | EE=[0.649,-0.002,0.023] | target=akita_black_bowl_1_main dist=0.040m can_close=True
    [TRAJ 000/106] sample 0091 | state 49 | EE=[0.456,-0.007,0.265] | target=akita_black_bowl_1_main dist=0.307m can_close=False
    [TRAJ 026/106] sample 0092 | state 49 | EE=[0.553,-0.008,0.040] | target=akita_black_bowl_1_main dist=0.065m can_close=False
    [TRAJ 052/106] sample 0093 | state 49 | EE=[0.752,-0.012,0.046] | target=akita_black_bowl_1_main dist=0.062m can_close=False
    [TRAJ 078/106] sample 0094 | state 49 | EE=[0.798,-0.013,0.046] | target=akita_black_bowl_1_main dist=0.060m can_close=False
    [TRAJ 105/106] sample 0095 | state 49 | EE=[0.807,-0.010,0.023] | target=akita_black_bowl_1_main dist=0.037m can_close=True
  Task 2 (close):   0%|          | 0/3 [00:00<?, ?it/s]  Task 2 (close):  33%|███▎      | 1/3 [00:01<00:03,  1.56s/it]  Task 2 (close):  67%|██████▋   | 2/3 [00:04<00:02,  2.38s/it]  Task 2 (close): 100%|██████████| 3/3 [00:07<00:00,  2.64s/it]                                                                   [CLOSE]         sample 0096 | state 00 | EE=[0.615,0.015,0.021] | target=akita_black_bowl_1_main dist=0.039m can_close=True
    [CLOSE]         sample 0097 | state 24 | EE=[0.807,0.017,0.026] | target=akita_black_bowl_1_main dist=0.039m can_close=True
    [CLOSE]         sample 0098 | state 49 | EE=[0.807,-0.010,0.023] | target=akita_black_bowl_1_main dist=0.037m can_close=True

[Task 03] pick up the black bowl on the cookie box and place it on the plate
  Task 3 (init):   0%|          | 0/5 [00:00<?, ?it/s]  Task 3 (init):  20%|██        | 1/5 [00:00<00:03,  1.18it/s]  Task 3 (init):  40%|████      | 2/5 [00:01<00:02,  1.19it/s]  Task 3 (init):  60%|██████    | 3/5 [00:02<00:01,  1.19it/s]  Task 3 (init):  80%|████████  | 4/5 [00:03<00:00,  1.19it/s]  Task 3 (init): 100%|██████████| 5/5 [00:04<00:00,  1.19it/s]                                                                                  sample 0099 | state 00 | EE=[0.454,-0.008,0.262] | target=akita_black_bowl_1_main dist=0.377m can_close=False
                    sample 0100 | state 12 | EE=[0.447,-0.006,0.262] | target=akita_black_bowl_1_main dist=0.390m can_close=False
                    sample 0101 | state 24 | EE=[0.452,-0.008,0.267] | target=akita_black_bowl_1_main dist=0.389m can_close=False
                    sample 0102 | state 36 | EE=[0.446,-0.004,0.258] | target=akita_black_bowl_1_main dist=0.371m can_close=False
                    sample 0103 | state 49 | EE=[0.457,-0.004,0.259] | target=akita_black_bowl_1_main dist=0.370m can_close=False
  Task 3 (traj):   0%|          | 0/5 [00:00<?, ?it/s]  Task 3 (traj):  20%|██        | 1/5 [00:01<00:07,  1.93s/it]  Task 3 (traj):  40%|████      | 2/5 [00:03<00:05,  1.92s/it]  Task 3 (traj):  60%|██████    | 3/5 [00:05<00:03,  1.95s/it]  Task 3 (traj):  80%|████████  | 4/5 [00:07<00:01,  1.94s/it]  Task 3 (traj): 100%|██████████| 5/5 [00:09<00:00,  1.93s/it]                                                                  [TRAJ 000/051] sample 0104 | state 00 | EE=[0.454,-0.008,0.262] | target=akita_black_bowl_1_main dist=0.377m can_close=False
    [TRAJ 012/051] sample 0105 | state 00 | EE=[0.530,0.001,0.173] | target=akita_black_bowl_1_main dist=0.260m can_close=False
    [TRAJ 025/051] sample 0106 | state 00 | EE=[0.629,0.012,0.080] | target=akita_black_bowl_1_main dist=0.124m can_close=False
    [TRAJ 037/051] sample 0107 | state 00 | EE=[0.721,0.020,0.018] | target=akita_black_bowl_1_main dist=0.043m can_close=False
    [TRAJ 050/051] sample 0108 | state 00 | EE=[0.805,0.019,-0.000] | target=akita_black_bowl_1_main dist=0.040m can_close=True
    [TRAJ 000/051] sample 0109 | state 12 | EE=[0.447,-0.006,0.262] | target=akita_black_bowl_1_main dist=0.390m can_close=False
    [TRAJ 012/051] sample 0110 | state 12 | EE=[0.525,0.004,0.175] | target=akita_black_bowl_1_main dist=0.274m can_close=False
    [TRAJ 025/051] sample 0111 | state 12 | EE=[0.627,0.015,0.085] | target=akita_black_bowl_1_main dist=0.138m can_close=False
    [TRAJ 037/051] sample 0112 | state 12 | EE=[0.720,0.024,0.019] | target=akita_black_bowl_1_main dist=0.043m can_close=False
    [TRAJ 050/051] sample 0113 | state 12 | EE=[0.806,0.023,-0.000] | target=akita_black_bowl_1_main dist=0.040m can_close=True
    [TRAJ 000/052] sample 0114 | state 24 | EE=[0.452,-0.008,0.267] | target=akita_black_bowl_1_main dist=0.389m can_close=False
    [TRAJ 012/052] sample 0115 | state 24 | EE=[0.529,-0.001,0.178] | target=akita_black_bowl_1_main dist=0.272m can_close=False
    [TRAJ 025/052] sample 0116 | state 24 | EE=[0.629,0.006,0.086] | target=akita_black_bowl_1_main dist=0.136m can_close=False
    [TRAJ 038/052] sample 0117 | state 24 | EE=[0.729,0.013,0.018] | target=akita_black_bowl_1_main dist=0.043m can_close=False
    [TRAJ 051/052] sample 0118 | state 24 | EE=[0.808,0.012,-0.002] | target=akita_black_bowl_1_main dist=0.039m can_close=True
    [TRAJ 000/051] sample 0119 | state 36 | EE=[0.446,-0.004,0.258] | target=akita_black_bowl_1_main dist=0.371m can_close=False
    [TRAJ 012/051] sample 0120 | state 36 | EE=[0.522,0.005,0.169] | target=akita_black_bowl_1_main dist=0.255m can_close=False
    [TRAJ 025/051] sample 0121 | state 36 | EE=[0.621,0.015,0.076] | target=akita_black_bowl_1_main dist=0.119m can_close=False
    [TRAJ 037/051] sample 0122 | state 36 | EE=[0.714,0.022,0.018] | target=akita_black_bowl_1_main dist=0.043m can_close=False
    [TRAJ 050/051] sample 0123 | state 36 | EE=[0.803,0.022,0.001] | target=akita_black_bowl_1_main dist=0.040m can_close=True
    [TRAJ 000/049] sample 0124 | state 49 | EE=[0.457,-0.004,0.259] | target=akita_black_bowl_1_main dist=0.370m can_close=False
    [TRAJ 012/049] sample 0125 | state 49 | EE=[0.533,0.003,0.169] | target=akita_black_bowl_1_main dist=0.253m can_close=False
    [TRAJ 024/049] sample 0126 | state 49 | EE=[0.624,0.011,0.083] | target=akita_black_bowl_1_main dist=0.128m can_close=False
    [TRAJ 036/049] sample 0127 | state 49 | EE=[0.717,0.018,0.019] | target=akita_black_bowl_1_main dist=0.043m can_close=False
    [TRAJ 048/049] sample 0128 | state 49 | EE=[0.801,0.017,0.002] | target=akita_black_bowl_1_main dist=0.040m can_close=True
  Task 3 (close):   0%|          | 0/3 [00:00<?, ?it/s]  Task 3 (close):  33%|███▎      | 1/3 [00:01<00:03,  1.78s/it]  Task 3 (close):  67%|██████▋   | 2/3 [00:03<00:01,  1.77s/it]  Task 3 (close): 100%|██████████| 3/3 [00:05<00:00,  1.77s/it]                                                                   [CLOSE]         sample 0129 | state 00 | EE=[0.805,0.019,-0.000] | target=akita_black_bowl_1_main dist=0.040m can_close=True
    [CLOSE]         sample 0130 | state 24 | EE=[0.808,0.012,-0.002] | target=akita_black_bowl_1_main dist=0.039m can_close=True
    [CLOSE]         sample 0131 | state 49 | EE=[0.801,0.017,0.002] | target=akita_black_bowl_1_main dist=0.040m can_close=True

[Task 04] pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
  Task 4 (init):   0%|          | 0/5 [00:00<?, ?it/s]  Task 4 (init):  20%|██        | 1/5 [00:00<00:03,  1.21it/s]  Task 4 (init):  40%|████      | 2/5 [00:01<00:02,  1.20it/s]  Task 4 (init):  60%|██████    | 3/5 [00:02<00:01,  1.20it/s]  Task 4 (init):  80%|████████  | 4/5 [00:03<00:00,  1.20it/s]  Task 4 (init): 100%|██████████| 5/5 [00:04<00:00,  1.19it/s]                                                                                  sample 0132 | state 00 | EE=[0.456,-0.000,0.261] | target=wooden_cabinet_1_cabinet_top dist=0.430m can_close=False
                    sample 0133 | state 12 | EE=[0.454,-0.001,0.280] | target=wooden_cabinet_1_cabinet_top dist=0.434m can_close=False
                    sample 0134 | state 24 | EE=[0.453,-0.005,0.259] | target=wooden_cabinet_1_cabinet_top dist=0.431m can_close=False
                    sample 0135 | state 36 | EE=[0.458,-0.012,0.274] | target=wooden_cabinet_1_cabinet_top dist=0.435m can_close=False
                    sample 0136 | state 49 | EE=[0.452,-0.023,0.265] | target=wooden_cabinet_1_cabinet_top dist=0.426m can_close=False
  Task 4 (traj):   0%|          | 0/5 [00:00<?, ?it/s]  Task 4 (traj):  20%|██        | 1/5 [00:06<00:26,  6.55s/it]  Task 4 (traj):  40%|████      | 2/5 [00:13<00:20,  6.70s/it]  Task 4 (traj):  60%|██████    | 3/5 [00:19<00:13,  6.65s/it]  Task 4 (traj):  80%|████████  | 4/5 [00:26<00:06,  6.65s/it]  Task 4 (traj): 100%|██████████| 5/5 [00:33<00:00,  6.61s/it]                                                                  [TRAJ 000/300] sample 0137 | state 00 | EE=[0.456,-0.000,0.261] | target=wooden_cabinet_1_cabinet_top dist=0.424m can_close=False
    [TRAJ 074/300] sample 0138 | state 00 | EE=[0.635,-0.039,0.029] | target=wooden_cabinet_1_cabinet_top dist=0.158m can_close=False
    [TRAJ 149/300] sample 0139 | state 00 | EE=[0.647,-0.040,0.026] | target=wooden_cabinet_1_cabinet_top dist=0.147m can_close=False
    [TRAJ 224/300] sample 0140 | state 00 | EE=[0.647,-0.039,0.025] | target=wooden_cabinet_1_cabinet_top dist=0.147m can_close=False
    [TRAJ 299/300] sample 0141 | state 00 | EE=[0.647,-0.039,0.024] | target=wooden_cabinet_1_cabinet_top dist=0.147m can_close=False
    [TRAJ 000/300] sample 0142 | state 12 | EE=[0.454,-0.001,0.280] | target=wooden_cabinet_1_cabinet_top dist=0.438m can_close=False
    [TRAJ 074/300] sample 0143 | state 12 | EE=[0.629,-0.033,0.064] | target=wooden_cabinet_1_cabinet_top dist=0.178m can_close=False
    [TRAJ 149/300] sample 0144 | state 12 | EE=[0.650,-0.037,0.026] | target=wooden_cabinet_1_cabinet_top dist=0.149m can_close=False
    [TRAJ 224/300] sample 0145 | state 12 | EE=[0.650,-0.036,0.026] | target=wooden_cabinet_1_cabinet_top dist=0.149m can_close=False
    [TRAJ 299/300] sample 0146 | state 12 | EE=[0.650,-0.036,0.025] | target=wooden_cabinet_1_cabinet_top dist=0.148m can_close=False
    [TRAJ 000/300] sample 0147 | state 24 | EE=[0.453,-0.005,0.259] | target=wooden_cabinet_1_cabinet_top dist=0.421m can_close=False
    [TRAJ 074/300] sample 0148 | state 24 | EE=[0.640,-0.044,0.026] | target=wooden_cabinet_1_cabinet_top dist=0.146m can_close=False
    [TRAJ 149/300] sample 0149 | state 24 | EE=[0.642,-0.044,0.024] | target=wooden_cabinet_1_cabinet_top dist=0.145m can_close=False
    [TRAJ 224/300] sample 0150 | state 24 | EE=[0.642,-0.044,0.023] | target=wooden_cabinet_1_cabinet_top dist=0.144m can_close=False
    [TRAJ 299/300] sample 0151 | state 24 | EE=[0.642,-0.044,0.022] | target=wooden_cabinet_1_cabinet_top dist=0.144m can_close=False
    [TRAJ 000/300] sample 0152 | state 36 | EE=[0.458,-0.012,0.274] | target=wooden_cabinet_1_cabinet_top dist=0.421m can_close=False
    [TRAJ 074/300] sample 0153 | state 36 | EE=[0.619,-0.030,0.062] | target=wooden_cabinet_1_cabinet_top dist=0.176m can_close=False
    [TRAJ 149/300] sample 0154 | state 36 | EE=[0.638,-0.033,0.027] | target=wooden_cabinet_1_cabinet_top dist=0.148m can_close=False
    [TRAJ 224/300] sample 0155 | state 36 | EE=[0.638,-0.032,0.026] | target=wooden_cabinet_1_cabinet_top dist=0.148m can_close=False
    [TRAJ 299/300] sample 0156 | state 36 | EE=[0.638,-0.032,0.025] | target=wooden_cabinet_1_cabinet_top dist=0.148m can_close=False
    [TRAJ 000/300] sample 0157 | state 49 | EE=[0.452,-0.023,0.265] | target=wooden_cabinet_1_cabinet_top dist=0.426m can_close=False
    [TRAJ 074/300] sample 0158 | state 49 | EE=[0.633,-0.037,0.027] | target=wooden_cabinet_1_cabinet_top dist=0.163m can_close=False
    [TRAJ 149/300] sample 0159 | state 49 | EE=[0.656,-0.030,0.021] | target=wooden_cabinet_1_cabinet_top dist=0.146m can_close=False
    [TRAJ 224/300] sample 0160 | state 49 | EE=[0.656,-0.030,0.020] | target=wooden_cabinet_1_cabinet_top dist=0.146m can_close=False
    [TRAJ 299/300] sample 0161 | state 49 | EE=[0.656,-0.030,0.019] | target=wooden_cabinet_1_cabinet_top dist=0.146m can_close=False
  Task 4 (close):   0%|          | 0/3 [00:00<?, ?it/s]  Task 4 (close):  33%|███▎      | 1/3 [00:06<00:12,  6.31s/it]  Task 4 (close):  67%|██████▋   | 2/3 [00:12<00:06,  6.38s/it]  Task 4 (close): 100%|██████████| 3/3 [00:18<00:00,  6.32s/it]                                                                   WARNING: Could not reach can_close=True for task 4 within 300 steps — try --close_max_steps larger.

[Task 05] pick up the black bowl on the ramekin and place it on the plate
  Task 5 (init):   0%|          | 0/5 [00:00<?, ?it/s]  Task 5 (init):  20%|██        | 1/5 [00:00<00:03,  1.20it/s]  Task 5 (init):  40%|████      | 2/5 [00:01<00:02,  1.16it/s]  Task 5 (init):  60%|██████    | 3/5 [00:02<00:01,  1.17it/s]  Task 5 (init):  80%|████████  | 4/5 [00:03<00:00,  1.19it/s]  Task 5 (init): 100%|██████████| 5/5 [00:04<00:00,  1.20it/s]                                                                                  sample 0162 | state 00 | EE=[0.448,0.005,0.255] | target=akita_black_bowl_1_main dist=0.277m can_close=False
                    sample 0163 | state 12 | EE=[0.445,0.008,0.258] | target=akita_black_bowl_1_main dist=0.281m can_close=False
                    sample 0164 | state 24 | EE=[0.451,-0.008,0.256] | target=akita_black_bowl_1_main dist=0.278m can_close=False
                    sample 0165 | state 36 | EE=[0.449,-0.002,0.263] | target=akita_black_bowl_1_main dist=0.284m can_close=False
                    sample 0166 | state 49 | EE=[0.453,0.001,0.249] | target=akita_black_bowl_1_main dist=0.282m can_close=False
  Task 5 (traj):   0%|          | 0/5 [00:00<?, ?it/s]  Task 5 (traj):  20%|██        | 1/5 [00:01<00:05,  1.36s/it]  Task 5 (traj):  40%|████      | 2/5 [00:02<00:04,  1.38s/it]  Task 5 (traj):  60%|██████    | 3/5 [00:04<00:02,  1.36s/it]  Task 5 (traj):  80%|████████  | 4/5 [00:05<00:01,  1.37s/it]  Task 5 (traj): 100%|██████████| 5/5 [00:06<00:00,  1.37s/it]                                                                  [TRAJ 000/025] sample 0167 | state 00 | EE=[0.448,0.005,0.255] | target=akita_black_bowl_1_main dist=0.277m can_close=False
    [TRAJ 006/025] sample 0168 | state 00 | EE=[0.448,0.036,0.211] | target=akita_black_bowl_1_main dist=0.224m can_close=False
    [TRAJ 012/025] sample 0169 | state 00 | EE=[0.446,0.073,0.158] | target=akita_black_bowl_1_main dist=0.160m can_close=False
    [TRAJ 018/025] sample 0170 | state 00 | EE=[0.444,0.110,0.106] | target=akita_black_bowl_1_main dist=0.097m can_close=False
    [TRAJ 024/025] sample 0171 | state 00 | EE=[0.444,0.146,0.054] | target=akita_black_bowl_1_main dist=0.034m can_close=True
    [TRAJ 000/025] sample 0172 | state 12 | EE=[0.445,0.008,0.258] | target=akita_black_bowl_1_main dist=0.281m can_close=False
    [TRAJ 006/025] sample 0173 | state 12 | EE=[0.445,0.039,0.214] | target=akita_black_bowl_1_main dist=0.227m can_close=False
    [TRAJ 012/025] sample 0174 | state 12 | EE=[0.442,0.076,0.161] | target=akita_black_bowl_1_main dist=0.164m can_close=False
    [TRAJ 018/025] sample 0175 | state 12 | EE=[0.439,0.112,0.109] | target=akita_black_bowl_1_main dist=0.100m can_close=False
    [TRAJ 024/025] sample 0176 | state 12 | EE=[0.438,0.148,0.057] | target=akita_black_bowl_1_main dist=0.037m can_close=True
    [TRAJ 000/025] sample 0177 | state 24 | EE=[0.451,-0.008,0.256] | target=akita_black_bowl_1_main dist=0.278m can_close=False
    [TRAJ 006/025] sample 0178 | state 24 | EE=[0.453,0.023,0.211] | target=akita_black_bowl_1_main dist=0.225m can_close=False
    [TRAJ 012/025] sample 0179 | state 24 | EE=[0.452,0.060,0.159] | target=akita_black_bowl_1_main dist=0.161m can_close=False
    [TRAJ 018/025] sample 0180 | state 24 | EE=[0.452,0.096,0.106] | target=akita_black_bowl_1_main dist=0.098m can_close=False
    [TRAJ 024/025] sample 0181 | state 24 | EE=[0.453,0.133,0.054] | target=akita_black_bowl_1_main dist=0.034m can_close=True
    [TRAJ 000/026] sample 0182 | state 36 | EE=[0.449,-0.002,0.263] | target=akita_black_bowl_1_main dist=0.284m can_close=False
    [TRAJ 006/026] sample 0183 | state 36 | EE=[0.448,0.028,0.218] | target=akita_black_bowl_1_main dist=0.231m can_close=False
    [TRAJ 012/026] sample 0184 | state 36 | EE=[0.444,0.065,0.166] | target=akita_black_bowl_1_main dist=0.168m can_close=False
    [TRAJ 018/026] sample 0185 | state 36 | EE=[0.440,0.100,0.113] | target=akita_black_bowl_1_main dist=0.104m can_close=False
    [TRAJ 025/026] sample 0186 | state 36 | EE=[0.438,0.142,0.052] | target=akita_black_bowl_1_main dist=0.030m can_close=True
    [TRAJ 000/025] sample 0187 | state 49 | EE=[0.453,0.001,0.249] | target=akita_black_bowl_1_main dist=0.282m can_close=False
    [TRAJ 006/025] sample 0188 | state 49 | EE=[0.450,0.034,0.207] | target=akita_black_bowl_1_main dist=0.230m can_close=False
    [TRAJ 012/025] sample 0189 | state 49 | EE=[0.443,0.074,0.158] | target=akita_black_bowl_1_main dist=0.166m can_close=False
    [TRAJ 018/025] sample 0190 | state 49 | EE=[0.436,0.113,0.107] | target=akita_black_bowl_1_main dist=0.103m can_close=False
    [TRAJ 024/025] sample 0191 | state 49 | EE=[0.431,0.152,0.058] | target=akita_black_bowl_1_main dist=0.040m can_close=True
  Task 5 (close):   0%|          | 0/3 [00:00<?, ?it/s]  Task 5 (close):  33%|███▎      | 1/3 [00:01<00:02,  1.21s/it]  Task 5 (close):  67%|██████▋   | 2/3 [00:02<00:01,  1.20s/it]  Task 5 (close): 100%|██████████| 3/3 [00:03<00:00,  1.20s/it]                                                                   [CLOSE]         sample 0192 | state 00 | EE=[0.444,0.146,0.054] | target=akita_black_bowl_1_main dist=0.034m can_close=True
    [CLOSE]         sample 0193 | state 24 | EE=[0.453,0.133,0.054] | target=akita_black_bowl_1_main dist=0.034m can_close=True
    [CLOSE]         sample 0194 | state 49 | EE=[0.431,0.152,0.058] | target=akita_black_bowl_1_main dist=0.040m can_close=True

[Task 06] pick up the black bowl next to the cookie box and place it on the plate
  Task 6 (init):   0%|          | 0/5 [00:00<?, ?it/s]  Task 6 (init):  20%|██        | 1/5 [00:00<00:03,  1.18it/s]  Task 6 (init):  40%|████      | 2/5 [00:01<00:02,  1.18it/s]  Task 6 (init):  60%|██████    | 3/5 [00:02<00:01,  1.20it/s]  Task 6 (init):  80%|████████  | 4/5 [00:03<00:00,  1.20it/s]  Task 6 (init): 100%|██████████| 5/5 [00:04<00:00,  1.21it/s]                                                                                  sample 0195 | state 00 | EE=[0.454,-0.008,0.268] | target=akita_black_bowl_1_main dist=0.448m can_close=False
                    sample 0196 | state 12 | EE=[0.451,0.002,0.266] | target=akita_black_bowl_1_main dist=0.456m can_close=False
                    sample 0197 | state 24 | EE=[0.452,0.013,0.262] | target=akita_black_bowl_1_main dist=0.451m can_close=False
                    sample 0198 | state 36 | EE=[0.455,0.004,0.266] | target=akita_black_bowl_1_main dist=0.442m can_close=False
                    sample 0199 | state 49 | EE=[0.455,-0.010,0.260] | target=akita_black_bowl_1_main dist=0.428m can_close=False
  Task 6 (traj):   0%|          | 0/5 [00:00<?, ?it/s]  Task 6 (traj):  20%|██        | 1/5 [00:06<00:27,  6.79s/it]  Task 6 (traj):  40%|████      | 2/5 [00:13<00:20,  6.84s/it]  Task 6 (traj):  60%|██████    | 3/5 [00:20<00:13,  6.79s/it]  Task 6 (traj):  80%|████████  | 4/5 [00:27<00:06,  6.86s/it]  Task 6 (traj): 100%|██████████| 5/5 [00:34<00:00,  6.84s/it]                                                                  [TRAJ 000/300] sample 0200 | state 00 | EE=[0.454,-0.008,0.268] | target=akita_black_bowl_1_main dist=0.448m can_close=False
    [TRAJ 074/300] sample 0201 | state 00 | EE=[0.820,-0.074,-0.002] | target=akita_black_bowl_1_main dist=0.045m can_close=False
    [TRAJ 149/300] sample 0202 | state 00 | EE=[0.835,-0.074,-0.002] | target=akita_black_bowl_1_main dist=0.042m can_close=False
    [TRAJ 224/300] sample 0203 | state 00 | EE=[0.841,-0.075,-0.001] | target=akita_black_bowl_1_main dist=0.043m can_close=False
    [TRAJ 299/300] sample 0204 | state 00 | EE=[0.845,-0.075,-0.001] | target=akita_black_bowl_1_main dist=0.044m can_close=False
    [TRAJ 000/300] sample 0205 | state 12 | EE=[0.451,0.002,0.266] | target=akita_black_bowl_1_main dist=0.456m can_close=False
    [TRAJ 074/300] sample 0206 | state 12 | EE=[0.820,-0.071,-0.002] | target=akita_black_bowl_1_main dist=0.044m can_close=False
    [TRAJ 149/300] sample 0207 | state 12 | EE=[0.835,-0.072,-0.002] | target=akita_black_bowl_1_main dist=0.041m can_close=False
    [TRAJ 224/300] sample 0208 | state 12 | EE=[0.842,-0.072,-0.002] | target=akita_black_bowl_1_main dist=0.042m can_close=False
    [TRAJ 299/300] sample 0209 | state 12 | EE=[0.847,-0.072,-0.001] | target=akita_black_bowl_1_main dist=0.044m can_close=False
    [TRAJ 000/300] sample 0210 | state 24 | EE=[0.452,0.013,0.262] | target=akita_black_bowl_1_main dist=0.451m can_close=False
    [TRAJ 074/300] sample 0211 | state 24 | EE=[0.821,-0.060,-0.002] | target=akita_black_bowl_1_main dist=0.047m can_close=False
    [TRAJ 149/300] sample 0212 | state 24 | EE=[0.838,-0.061,-0.002] | target=akita_black_bowl_1_main dist=0.043m can_close=False
    [TRAJ 224/300] sample 0213 | state 24 | EE=[0.844,-0.061,-0.002] | target=akita_black_bowl_1_main dist=0.043m can_close=False
    [TRAJ 299/300] sample 0214 | state 24 | EE=[0.849,-0.061,-0.002] | target=akita_black_bowl_1_main dist=0.044m can_close=False
    [TRAJ 000/300] sample 0215 | state 36 | EE=[0.455,0.004,0.266] | target=akita_black_bowl_1_main dist=0.442m can_close=False
    [TRAJ 074/300] sample 0216 | state 36 | EE=[0.819,-0.072,-0.002] | target=akita_black_bowl_1_main dist=0.044m can_close=False
    [TRAJ 149/300] sample 0217 | state 36 | EE=[0.835,-0.073,-0.002] | target=akita_black_bowl_1_main dist=0.042m can_close=False
    [TRAJ 224/300] sample 0218 | state 36 | EE=[0.841,-0.073,-0.002] | target=akita_black_bowl_1_main dist=0.043m can_close=False
    [TRAJ 299/300] sample 0219 | state 36 | EE=[0.846,-0.073,-0.002] | target=akita_black_bowl_1_main dist=0.044m can_close=False
    [TRAJ 000/300] sample 0220 | state 49 | EE=[0.455,-0.010,0.260] | target=akita_black_bowl_1_main dist=0.428m can_close=False
    [TRAJ 074/300] sample 0221 | state 49 | EE=[0.818,-0.075,-0.002] | target=akita_black_bowl_1_main dist=0.051m can_close=False
    [TRAJ 149/300] sample 0222 | state 49 | EE=[0.834,-0.077,-0.002] | target=akita_black_bowl_1_main dist=0.047m can_close=False
    [TRAJ 224/300] sample 0223 | state 49 | EE=[0.839,-0.077,-0.002] | target=akita_black_bowl_1_main dist=0.047m can_close=False
    [TRAJ 299/300] sample 0224 | state 49 | EE=[0.843,-0.077,-0.002] | target=akita_black_bowl_1_main dist=0.047m can_close=False
  Task 6 (close):   0%|          | 0/3 [00:00<?, ?it/s]  Task 6 (close):  33%|███▎      | 1/3 [00:06<00:12,  6.46s/it]  Task 6 (close):  67%|██████▋   | 2/3 [00:13<00:06,  6.64s/it]  Task 6 (close): 100%|██████████| 3/3 [00:19<00:00,  6.61s/it]                                                                   WARNING: Could not reach can_close=True for task 6 within 300 steps — try --close_max_steps larger.

[Task 07] pick up the black bowl on the stove and place it on the plate
  Task 7 (init):   0%|          | 0/5 [00:00<?, ?it/s]  Task 7 (init):  20%|██        | 1/5 [00:00<00:03,  1.20it/s]  Task 7 (init):  40%|████      | 2/5 [00:01<00:02,  1.18it/s]  Task 7 (init):  60%|██████    | 3/5 [00:02<00:01,  1.18it/s]  Task 7 (init):  80%|████████  | 4/5 [00:03<00:00,  1.17it/s]  Task 7 (init): 100%|██████████| 5/5 [00:04<00:00,  1.18it/s]                                                                                  sample 0225 | state 00 | EE=[0.443,0.015,0.246] | target=akita_black_bowl_1_main dist=0.276m can_close=False
                    sample 0226 | state 12 | EE=[0.448,-0.020,0.262] | target=akita_black_bowl_1_main dist=0.271m can_close=False
                    sample 0227 | state 24 | EE=[0.455,-0.014,0.255] | target=akita_black_bowl_1_main dist=0.275m can_close=False
                    sample 0228 | state 36 | EE=[0.450,0.009,0.252] | target=akita_black_bowl_1_main dist=0.285m can_close=False
                    sample 0229 | state 49 | EE=[0.467,0.019,0.273] | target=akita_black_bowl_1_main dist=0.317m can_close=False
  Task 7 (traj):   0%|          | 0/5 [00:00<?, ?it/s]  Task 7 (traj):  20%|██        | 1/5 [00:01<00:05,  1.40s/it]  Task 7 (traj):  40%|████      | 2/5 [00:02<00:04,  1.36s/it]  Task 7 (traj):  60%|██████    | 3/5 [00:04<00:02,  1.36s/it]  Task 7 (traj):  80%|████████  | 4/5 [00:05<00:01,  1.43s/it]  Task 7 (traj): 100%|██████████| 5/5 [00:07<00:00,  1.44s/it]                                                                  [TRAJ 000/025] sample 0230 | state 00 | EE=[0.443,0.015,0.246] | target=akita_black_bowl_1_main dist=0.276m can_close=False
    [TRAJ 006/025] sample 0231 | state 00 | EE=[0.436,-0.013,0.202] | target=akita_black_bowl_1_main dist=0.223m can_close=False
    [TRAJ 012/025] sample 0232 | state 00 | EE=[0.424,-0.046,0.149] | target=akita_black_bowl_1_main dist=0.160m can_close=False
    [TRAJ 018/025] sample 0233 | state 00 | EE=[0.410,-0.079,0.096] | target=akita_black_bowl_1_main dist=0.095m can_close=False
    [TRAJ 024/025] sample 0234 | state 00 | EE=[0.400,-0.112,0.043] | target=akita_black_bowl_1_main dist=0.032m can_close=True
    [TRAJ 000/024] sample 0235 | state 12 | EE=[0.448,-0.020,0.262] | target=akita_black_bowl_1_main dist=0.271m can_close=False
    [TRAJ 005/024] sample 0236 | state 12 | EE=[0.442,-0.036,0.223] | target=akita_black_bowl_1_main dist=0.229m can_close=False
    [TRAJ 011/024] sample 0237 | state 12 | EE=[0.429,-0.060,0.167] | target=akita_black_bowl_1_main dist=0.166m can_close=False
    [TRAJ 017/024] sample 0238 | state 12 | EE=[0.413,-0.083,0.109] | target=akita_black_bowl_1_main dist=0.102m can_close=False
    [TRAJ 023/024] sample 0239 | state 12 | EE=[0.400,-0.107,0.051] | target=akita_black_bowl_1_main dist=0.038m can_close=True
    [TRAJ 000/025] sample 0240 | state 24 | EE=[0.455,-0.014,0.255] | target=akita_black_bowl_1_main dist=0.275m can_close=False
    [TRAJ 006/025] sample 0241 | state 24 | EE=[0.449,-0.040,0.209] | target=akita_black_bowl_1_main dist=0.222m can_close=False
    [TRAJ 012/025] sample 0242 | state 24 | EE=[0.437,-0.069,0.154] | target=akita_black_bowl_1_main dist=0.159m can_close=False
    [TRAJ 018/025] sample 0243 | state 24 | EE=[0.424,-0.098,0.098] | target=akita_black_bowl_1_main dist=0.094m can_close=False
    [TRAJ 024/025] sample 0244 | state 24 | EE=[0.414,-0.127,0.043] | target=akita_black_bowl_1_main dist=0.031m can_close=True
    [TRAJ 000/026] sample 0245 | state 36 | EE=[0.450,0.009,0.252] | target=akita_black_bowl_1_main dist=0.285m can_close=False
    [TRAJ 006/026] sample 0246 | state 36 | EE=[0.445,-0.021,0.208] | target=akita_black_bowl_1_main dist=0.232m can_close=False
    [TRAJ 012/026] sample 0247 | state 36 | EE=[0.436,-0.055,0.156] | target=akita_black_bowl_1_main dist=0.169m can_close=False
    [TRAJ 018/026] sample 0248 | state 36 | EE=[0.426,-0.090,0.103] | target=akita_black_bowl_1_main dist=0.105m can_close=False
    [TRAJ 025/026] sample 0249 | state 36 | EE=[0.417,-0.130,0.041] | target=akita_black_bowl_1_main dist=0.031m can_close=True
    [TRAJ 000/029] sample 0250 | state 49 | EE=[0.467,0.019,0.273] | target=akita_black_bowl_1_main dist=0.317m can_close=False
    [TRAJ 007/029] sample 0251 | state 49 | EE=[0.460,-0.016,0.225] | target=akita_black_bowl_1_main dist=0.257m can_close=False
    [TRAJ 014/029] sample 0252 | state 49 | EE=[0.443,-0.056,0.165] | target=akita_black_bowl_1_main dist=0.183m can_close=False
    [TRAJ 021/029] sample 0253 | state 49 | EE=[0.425,-0.096,0.104] | target=akita_black_bowl_1_main dist=0.108m can_close=False
    [TRAJ 028/029] sample 0254 | state 49 | EE=[0.411,-0.135,0.044] | target=akita_black_bowl_1_main dist=0.034m can_close=True
  Task 7 (close):   0%|          | 0/3 [00:00<?, ?it/s]  Task 7 (close):  33%|███▎      | 1/3 [00:01<00:02,  1.20s/it]  Task 7 (close):  67%|██████▋   | 2/3 [00:02<00:01,  1.20s/it]  Task 7 (close): 100%|██████████| 3/3 [00:03<00:00,  1.24s/it]                                                                   [CLOSE]         sample 0255 | state 00 | EE=[0.400,-0.112,0.043] | target=akita_black_bowl_1_main dist=0.032m can_close=True
    [CLOSE]         sample 0256 | state 24 | EE=[0.414,-0.127,0.043] | target=akita_black_bowl_1_main dist=0.031m can_close=True
    [CLOSE]         sample 0257 | state 49 | EE=[0.411,-0.135,0.044] | target=akita_black_bowl_1_main dist=0.034m can_close=True

[Task 08] pick up the black bowl next to the plate and place it on the plate
  Task 8 (init):   0%|          | 0/5 [00:00<?, ?it/s]  Task 8 (init):  20%|██        | 1/5 [00:00<00:03,  1.17it/s]  Task 8 (init):  40%|████      | 2/5 [00:01<00:02,  1.20it/s]  Task 8 (init):  60%|██████    | 3/5 [00:02<00:01,  1.20it/s]  Task 8 (init):  80%|████████  | 4/5 [00:03<00:00,  1.20it/s]  Task 8 (init): 100%|██████████| 5/5 [00:04<00:00,  1.21it/s]                                                                                  sample 0258 | state 00 | EE=[0.451,-0.013,0.266] | target=akita_black_bowl_1_main dist=0.471m can_close=False
                    sample 0259 | state 12 | EE=[0.442,-0.006,0.246] | target=akita_black_bowl_1_main dist=0.475m can_close=False
                    sample 0260 | state 24 | EE=[0.452,0.002,0.266] | target=akita_black_bowl_1_main dist=0.464m can_close=False
                    sample 0261 | state 36 | EE=[0.449,0.005,0.263] | target=akita_black_bowl_1_main dist=0.475m can_close=False
                    sample 0262 | state 49 | EE=[0.460,0.001,0.280] | target=akita_black_bowl_1_main dist=0.484m can_close=False
  Task 8 (traj):   0%|          | 0/5 [00:00<?, ?it/s]  Task 8 (traj):  20%|██        | 1/5 [00:05<00:23,  5.84s/it]  Task 8 (traj):  40%|████      | 2/5 [00:11<00:17,  5.94s/it]  Task 8 (traj):  60%|██████    | 3/5 [00:17<00:12,  6.03s/it]  Task 8 (traj):  80%|████████  | 4/5 [00:23<00:05,  5.84s/it]  Task 8 (traj): 100%|██████████| 5/5 [00:25<00:00,  4.58s/it]                                                                  [TRAJ 000/300] sample 0263 | state 00 | EE=[0.451,-0.013,0.266] | target=akita_black_bowl_1_main dist=0.471m can_close=False
    [TRAJ 074/300] sample 0264 | state 00 | EE=[0.736,0.337,0.001] | target=akita_black_bowl_1_main dist=0.078m can_close=False
    [TRAJ 149/300] sample 0265 | state 00 | EE=[0.739,0.337,-0.000] | target=akita_black_bowl_1_main dist=0.075m can_close=False
    [TRAJ 224/300] sample 0266 | state 00 | EE=[0.742,0.338,0.000] | target=akita_black_bowl_1_main dist=0.073m can_close=False
    [TRAJ 299/300] sample 0267 | state 00 | EE=[0.745,0.339,0.001] | target=akita_black_bowl_1_main dist=0.071m can_close=False
    [TRAJ 000/300] sample 0268 | state 12 | EE=[0.442,-0.006,0.246] | target=akita_black_bowl_1_main dist=0.475m can_close=False
    [TRAJ 074/300] sample 0269 | state 12 | EE=[0.735,0.330,-0.001] | target=akita_black_bowl_1_main dist=0.071m can_close=False
    [TRAJ 149/300] sample 0270 | state 12 | EE=[0.739,0.330,-0.000] | target=akita_black_bowl_1_main dist=0.071m can_close=False
    [TRAJ 224/300] sample 0271 | state 12 | EE=[0.741,0.331,-0.000] | target=akita_black_bowl_1_main dist=0.070m can_close=False
    [TRAJ 299/300] sample 0272 | state 12 | EE=[0.744,0.331,0.000] | target=akita_black_bowl_1_main dist=0.069m can_close=False
    [TRAJ 000/300] sample 0273 | state 24 | EE=[0.452,0.002,0.266] | target=akita_black_bowl_1_main dist=0.464m can_close=False
    [TRAJ 074/300] sample 0274 | state 24 | EE=[0.732,0.340,0.001] | target=akita_black_bowl_1_main dist=0.067m can_close=False
    [TRAJ 149/300] sample 0275 | state 24 | EE=[0.740,0.340,0.000] | target=akita_black_bowl_1_main dist=0.065m can_close=False
    [TRAJ 224/300] sample 0276 | state 24 | EE=[0.743,0.341,0.001] | target=akita_black_bowl_1_main dist=0.065m can_close=False
    [TRAJ 299/300] sample 0277 | state 24 | EE=[0.746,0.341,0.001] | target=akita_black_bowl_1_main dist=0.063m can_close=False
    [TRAJ 000/300] sample 0278 | state 36 | EE=[0.449,0.005,0.263] | target=akita_black_bowl_1_main dist=0.475m can_close=False
    [TRAJ 074/300] sample 0279 | state 36 | EE=[0.730,0.364,0.002] | target=akita_black_bowl_1_main dist=0.129m can_close=False
    [TRAJ 149/300] sample 0280 | state 36 | EE=[0.733,0.364,0.001] | target=akita_black_bowl_1_main dist=0.127m can_close=False
    [TRAJ 224/300] sample 0281 | state 36 | EE=[0.735,0.365,0.001] | target=akita_black_bowl_1_main dist=0.124m can_close=False
    [TRAJ 299/300] sample 0282 | state 36 | EE=[0.738,0.365,0.002] | target=akita_black_bowl_1_main dist=0.121m can_close=False
    [TRAJ 000/078] sample 0283 | state 49 | EE=[0.460,0.001,0.280] | target=akita_black_bowl_1_main dist=0.484m can_close=False
    [TRAJ 019/078] sample 0284 | state 49 | EE=[0.539,0.126,0.156] | target=akita_black_bowl_1_main dist=0.292m can_close=False
    [TRAJ 038/078] sample 0285 | state 49 | EE=[0.625,0.255,0.037] | target=akita_black_bowl_1_main dist=0.098m can_close=False
    [TRAJ 057/078] sample 0286 | state 49 | EE=[0.713,0.333,0.037] | target=akita_black_bowl_1_main dist=0.074m can_close=False
    [TRAJ 077/078] sample 0287 | state 49 | EE=[0.733,0.346,0.001] | target=akita_black_bowl_1_main dist=0.039m can_close=True
  Task 8 (close):   0%|          | 0/3 [00:00<?, ?it/s]  Task 8 (close):  33%|███▎      | 1/3 [00:05<00:10,  5.38s/it]  Task 8 (close):  67%|██████▋   | 2/3 [00:10<00:05,  5.50s/it]  Task 8 (close): 100%|██████████| 3/3 [00:13<00:00,  3.99s/it]                                                                   [CLOSE]         sample 0288 | state 49 | EE=[0.733,0.346,0.001] | target=akita_black_bowl_1_main dist=0.039m can_close=True

[Task 09] pick up the black bowl on the wooden cabinet and place it on the plate
  Task 9 (init):   0%|          | 0/5 [00:00<?, ?it/s]  Task 9 (init):  20%|██        | 1/5 [00:00<00:03,  1.20it/s]  Task 9 (init):  40%|████      | 2/5 [00:01<00:02,  1.21it/s]  Task 9 (init):  60%|██████    | 3/5 [00:02<00:01,  1.22it/s]  Task 9 (init):  80%|████████  | 4/5 [00:03<00:00,  1.21it/s]  Task 9 (init): 100%|██████████| 5/5 [00:04<00:00,  1.22it/s]                                                                                  sample 0289 | state 00 | EE=[0.448,-0.005,0.266] | target=wooden_cabinet_1_cabinet_top dist=0.443m can_close=False
                    sample 0290 | state 12 | EE=[0.439,-0.018,0.244] | target=wooden_cabinet_1_cabinet_top dist=0.434m can_close=False
                    sample 0291 | state 24 | EE=[0.450,0.014,0.260] | target=wooden_cabinet_1_cabinet_top dist=0.458m can_close=False
                    sample 0292 | state 36 | EE=[0.459,0.010,0.260] | target=wooden_cabinet_1_cabinet_top dist=0.454m can_close=False
                    sample 0293 | state 49 | EE=[0.453,0.011,0.256] | target=wooden_cabinet_1_cabinet_top dist=0.455m can_close=False
  Task 9 (traj):   0%|          | 0/5 [00:00<?, ?it/s]  Task 9 (traj):  20%|██        | 1/5 [00:06<00:24,  6.17s/it]  Task 9 (traj):  40%|████      | 2/5 [00:12<00:18,  6.14s/it]  Task 9 (traj):  60%|██████    | 3/5 [00:18<00:12,  6.12s/it]  Task 9 (traj):  80%|████████  | 4/5 [00:24<00:06,  6.13s/it]  Task 9 (traj): 100%|██████████| 5/5 [00:30<00:00,  6.23s/it]                                                                  [TRAJ 000/300] sample 0294 | state 00 | EE=[0.448,-0.005,0.266] | target=wooden_cabinet_1_cabinet_top dist=0.459m can_close=False
    [TRAJ 074/300] sample 0295 | state 00 | EE=[0.572,-0.135,0.087] | target=wooden_cabinet_1_cabinet_top dist=0.211m can_close=False
    [TRAJ 149/300] sample 0296 | state 00 | EE=[0.572,-0.135,0.081] | target=wooden_cabinet_1_cabinet_top dist=0.208m can_close=False
    [TRAJ 224/300] sample 0297 | state 00 | EE=[0.572,-0.135,0.075] | target=wooden_cabinet_1_cabinet_top dist=0.206m can_close=False
    [TRAJ 299/300] sample 0298 | state 00 | EE=[0.572,-0.135,0.071] | target=wooden_cabinet_1_cabinet_top dist=0.204m can_close=False
    [TRAJ 000/300] sample 0299 | state 12 | EE=[0.439,-0.018,0.244] | target=wooden_cabinet_1_cabinet_top dist=0.442m can_close=False
    [TRAJ 074/300] sample 0300 | state 12 | EE=[0.567,-0.141,0.074] | target=wooden_cabinet_1_cabinet_top dist=0.203m can_close=False
    [TRAJ 149/300] sample 0301 | state 12 | EE=[0.567,-0.141,0.064] | target=wooden_cabinet_1_cabinet_top dist=0.199m can_close=False
    [TRAJ 224/300] sample 0302 | state 12 | EE=[0.567,-0.141,0.057] | target=wooden_cabinet_1_cabinet_top dist=0.197m can_close=False
    [TRAJ 299/300] sample 0303 | state 12 | EE=[0.567,-0.141,0.054] | target=wooden_cabinet_1_cabinet_top dist=0.196m can_close=False
    [TRAJ 000/300] sample 0304 | state 24 | EE=[0.450,0.014,0.260] | target=wooden_cabinet_1_cabinet_top dist=0.457m can_close=False
    [TRAJ 074/300] sample 0305 | state 24 | EE=[0.565,-0.121,0.071] | target=wooden_cabinet_1_cabinet_top dist=0.209m can_close=False
    [TRAJ 149/300] sample 0306 | state 24 | EE=[0.565,-0.121,0.055] | target=wooden_cabinet_1_cabinet_top dist=0.204m can_close=False
    [TRAJ 224/300] sample 0307 | state 24 | EE=[0.565,-0.121,0.050] | target=wooden_cabinet_1_cabinet_top dist=0.202m can_close=False
    [TRAJ 299/300] sample 0308 | state 24 | EE=[0.565,-0.121,0.048] | target=wooden_cabinet_1_cabinet_top dist=0.202m can_close=False
    [TRAJ 000/300] sample 0309 | state 36 | EE=[0.459,0.010,0.260] | target=wooden_cabinet_1_cabinet_top dist=0.448m can_close=False
    [TRAJ 074/300] sample 0310 | state 36 | EE=[0.565,-0.116,0.075] | target=wooden_cabinet_1_cabinet_top dist=0.212m can_close=False
    [TRAJ 149/300] sample 0311 | state 36 | EE=[0.565,-0.116,0.055] | target=wooden_cabinet_1_cabinet_top dist=0.205m can_close=False
    [TRAJ 224/300] sample 0312 | state 36 | EE=[0.564,-0.116,0.049] | target=wooden_cabinet_1_cabinet_top dist=0.203m can_close=False
    [TRAJ 299/300] sample 0313 | state 36 | EE=[0.564,-0.116,0.046] | target=wooden_cabinet_1_cabinet_top dist=0.202m can_close=False
    [TRAJ 000/300] sample 0314 | state 49 | EE=[0.453,0.011,0.256] | target=wooden_cabinet_1_cabinet_top dist=0.458m can_close=False
    [TRAJ 074/300] sample 0315 | state 49 | EE=[0.577,-0.125,0.090] | target=wooden_cabinet_1_cabinet_top dist=0.215m can_close=False
    [TRAJ 149/300] sample 0316 | state 49 | EE=[0.577,-0.125,0.085] | target=wooden_cabinet_1_cabinet_top dist=0.212m can_close=False
    [TRAJ 224/300] sample 0317 | state 49 | EE=[0.577,-0.125,0.082] | target=wooden_cabinet_1_cabinet_top dist=0.211m can_close=False
    [TRAJ 299/300] sample 0318 | state 49 | EE=[0.577,-0.125,0.080] | target=wooden_cabinet_1_cabinet_top dist=0.210m can_close=False
  Task 9 (close):   0%|          | 0/3 [00:00<?, ?it/s]  Task 9 (close):  33%|███▎      | 1/3 [00:06<00:12,  6.05s/it]  Task 9 (close):  67%|██████▋   | 2/3 [00:12<00:05,  5.99s/it]  Task 9 (close): 100%|██████████| 3/3 [00:17<00:00,  5.93s/it]                                                                   WARNING: Could not reach can_close=True for task 9 within 300 steps — try --close_max_steps larger.

  [libero_spatial] 319 samples saved  |  can_close=True: 50/319 (16%)  [19 from close-approach]  |  manifest: data/gt/manifest.json

============================================================
Grand total: 319 samples across 1 suite(s)
  libero_spatial: 319 samples
[11:08:53] STEP 1 done  →  data/gt/manifest.json
============================================================
[11:08:53] STEP 2/3  VLM evaluation
Loaded 319 samples from data/gt/manifest.json
Run timestamp: 20260416_110853
Loading Qwen/Qwen2.5-VL-7B-Instruct ...
Fetching 5 files:   0%|          | 0/5 [00:00<?, ?it/s]Fetching 5 files: 100%|██████████| 5/5 [00:00<00:00, 11060.93it/s]
Loading weights:   0%|          | 0/729 [00:00<?, ?it/s]Loading weights:   0%|          | 1/729 [00:00<01:29,  8.14it/s]Loading weights:   0%|          | 2/729 [00:00<01:45,  6.92it/s]Loading weights:   2%|▏         | 17/729 [00:00<00:12, 57.30it/s]Loading weights:   5%|▌         | 40/729 [00:00<00:06, 109.01it/s]Loading weights:   7%|▋         | 54/729 [00:00<00:05, 114.57it/s]Loading weights:  10%|█         | 76/729 [00:00<00:04, 139.71it/s]Loading weights:  14%|█▎        | 100/729 [00:00<00:03, 161.41it/s]Loading weights:  16%|█▌        | 117/729 [00:00<00:03, 161.69it/s]Loading weights:  19%|█▊        | 136/729 [00:01<00:03, 169.09it/s]Loading weights:  21%|██        | 154/729 [00:01<00:03, 162.58it/s]Loading weights:  24%|██▎       | 173/729 [00:01<00:03, 165.52it/s]Loading weights:  27%|██▋       | 196/729 [00:01<00:03, 175.65it/s]Loading weights:  29%|██▉       | 214/729 [00:01<00:02, 171.77it/s]Loading weights:  32%|███▏      | 232/729 [00:01<00:02, 173.93it/s]Loading weights:  34%|███▍      | 250/729 [00:01<00:02, 168.13it/s]Loading weights:  37%|███▋      | 269/729 [00:01<00:02, 167.57it/s]Loading weights:  40%|████      | 292/729 [00:01<00:02, 176.36it/s]Loading weights:  43%|████▎     | 310/729 [00:02<00:02, 169.04it/s]Loading weights:  45%|████▍     | 328/729 [00:02<00:02, 171.87it/s]Loading weights:  60%|██████    | 439/729 [00:02<00:00, 432.75it/s]Loading weights:  87%|████████▋ | 637/729 [00:02<00:00, 871.60it/s]Loading weights: 100%|██████████| 729/729 [00:02<00:00, 298.29it/s]
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. 

============================================================
Model: qwen2.5-vl-7b  |  Run: 20260416_110853  |  Output: data/runs/20260416_110326/qwen2.5-vl-7b/20260416_110853
============================================================
qwen2.5-vl-7b:   0%|          | 0/319 [00:00<?, ?it/s]qwen2.5-vl-7b:   4%|▍         | 13/319 [00:00<00:02, 126.06it/s]qwen2.5-vl-7b:  10%|▉         | 31/319 [00:00<00:01, 156.56it/s]qwen2.5-vl-7b:  15%|█▍        | 47/319 [00:00<00:01, 156.63it/s]  sample 0000 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_00/sample_0000.png'
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
  sample 0063 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_01/sample_0063.png'qwen2.5-vl-7b:  20%|██        | 65/319 [00:00<00:01, 165.31it/s]qwen2.5-vl-7b:  26%|██▌       | 82/319 [00:00<00:01, 166.62it/s]qwen2.5-vl-7b:  31%|███▏      | 100/319 [00:00<00:01, 168.80it/s]qwen2.5-vl-7b:  37%|███▋      | 118/319 [00:00<00:01, 171.22it/s]
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
  sample 0127 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_03/sample_0127.png'qwen2.5-vl-7b:  43%|████▎     | 136/319 [00:00<00:01, 172.93it/s]qwen2.5-vl-7b:  48%|████▊     | 154/319 [00:00<00:00, 172.40it/s]qwen2.5-vl-7b:  54%|█████▍    | 172/319 [00:01<00:00, 172.31it/s]qwen2.5-vl-7b:  60%|█████▉    | 190/319 [00:01<00:00, 173.70it/s]
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
  sample 0191 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_05/sample_0191.png'qwen2.5-vl-7b:  65%|██████▌   | 208/319 [00:01<00:00, 173.31it/s]qwen2.5-vl-7b:  71%|███████   | 226/319 [00:01<00:00, 154.41it/s]qwen2.5-vl-7b:  76%|███████▌  | 242/319 [00:01<00:00, 155.50it/s]
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
  sample 0255 FAILED: [Errno 2] No such file or directory: '/umd-datapool/tingting/3DBENCH/data/data/gt/task_07/sample_0255.png'qwen2.5-vl-7b:  82%|████████▏ | 260/319 [00:01<00:00, 161.76it/s]qwen2.5-vl-7b:  87%|████████▋ | 277/319 [00:01<00:00, 160.41it/s]qwen2.5-vl-7b:  93%|█████████▎| 296/319 [00:01<00:00, 166.77it/s]qwen2.5-vl-7b:  98%|█████████▊| 313/319 [00:01<00:00, 162.00it/s]qwen2.5-vl-7b: 100%|██████████| 319/319 [00:01<00:00, 164.64it/s]

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

All done. Responses saved under data/runs/20260416_110326/
Next: python scripts/03_compute_metrics.py
[11:09:11] STEP 2 done  →  data/runs/20260416_110326/
============================================================
[11:09:11] STEP 3/3  Compute metrics
Found 319 GT samples, 1 model(s): ['qwen2.5-vl-7b']

Evaluating qwen2.5-vl-7b (run: 20260416_110853): 217/319 responses found.

==========================================================================================
SPATIAL REASONING BENCHMARK RESULTS
==========================================================================================
------------------------------------------------------------------------------------------
Model                            Parse%   Q1 MAE(m)   Q2 MAE(m)    Q3 Acc     Q3 F1   Q4 CosSim
------------------------------------------------------------------------------------------
qwen2.5-vl-7b                     68.0%       N/A         N/A       N/A       N/A         N/A  
------------------------------------------------------------------------------------------

Per-axis MAE breakdown (meters)
----------------------------------------------------------------------
Model                            Q     MAE_x     MAE_y     MAE_z  note
----------------------------------------------------------------------
qwen2.5-vl-7b                   Q1     N/A       N/A       N/A    n=0
qwen2.5-vl-7b                   Q2     N/A       N/A       N/A    n=0
----------------------------------------------------------------------

Q3 (can_close) detail
----------------------------------------------------------------------
Model                                Acc        F1      GT+%    Pred+%    Parse%
----------------------------------------------------------------------
qwen2.5-vl-7b                      N/A       N/A       N/A       N/A        0.0%
----------------------------------------------------------------------

Summary results saved to data/runs/20260416_110326/results.json
Per-sample comparison saved to data/runs/20260416_110326/comparison_qwen2.5-vl-7b.json

Sample comparison preview (qwen2.5-vl-7b, first 5 samples):
  ID                                      Task   Q1 gt_z   Q1 pr_z   Q2 gt_z   Q2 pr_z    Q4 cos
-----------------------------------------------------------------------------------------------
Traceback (most recent call last):
  File "/umd-datapool/tingting/3DBENCH/scripts/03_compute_metrics.py", line 591, in <module>
    main()
  File "/umd-datapool/tingting/3DBENCH/scripts/03_compute_metrics.py", line 564, in main
    f"{s['q4'].get('cosine_sim', 'N/A'):>8}"
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: unsupported format string passed to NoneType.__format__
