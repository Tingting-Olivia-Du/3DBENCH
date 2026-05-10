# Qwen2.5-VL-3B base vs LoRA on CALVIN

- GT manifest: `data/gt-q6-mv/manifest.json` (463 samples)
- Base responses: `data/runs-mv-base-libero/qwen2.5-vl-3b`
- LoRA responses: `data/runs-mv-lora-libero-new/qwen2.5-vl-3b-mv-lora-<dim>`

## Headline metrics (LoRA only answers its own dim; q1 LoRA also covers q1_dest)

| Dim | Metric | Base | LoRA | Δ (LoRA − Base) | Parse rate (base / LoRA) |
|-----|--------|------|------|------------------|---------------------------|
| **Q1** | mae_overall (↓ better) | 0.1275 | 0.0294 | ↓0.0982 ✓ | 1.0000 / 1.0000 |
| **Q1_DEST** | mae_overall (↓ better) | 0.1586 | 0.0541 | ↓0.1044 ✓ | 1.0000 / 0.8105 |
| **Q2** | mae_overall (↓ better) | 0.0898 | 0.0513 | ↓0.0386 ✓ | 1.0000 / 1.0000 |
| **Q3** | accuracy (↑ better) | 0.9287 | 0.9287 |   0.0000 | 1.0000 / 1.0000 |
| **Q4** | mean_cosine_sim (↑ better) | 0.5203 | 0.2906 | ↓0.2297 ✗ | 1.0000 / 1.0000 |
| **Q5** | mae_overall (↓ better) | 0.1628 | 0.0658 | ↓0.0969 ✓ | 1.0000 / 1.0000 |
| **Q6** | acc_all (↑ better) | 0.0994 | 0.6847 | ↑0.5853 ✓ | 1.0000 / 1.0000 |

### Secondary metrics

| Dim | Metric | Base | LoRA |
|-----|--------|------|------|
| **Q1** | rmse_overall | 0.1946 | 0.0682 |
| **Q1_DEST** | rmse_overall | 0.2314 | 0.1211 |
| **Q2** | rmse_overall | 0.1215 | 0.0836 |
| **Q3** | f1 | 0.0000 | 0.0000 |
| **Q4** | median_cosine_sim | 0.3704 | 0.4089 |
| **Q5** | rmse_overall | 0.1986 | 0.0901 |
| **Q6** | f1_x | 0.2461 | 0.6664 |

### Notes
- ✓ marks an improvement (LoRA better than base on the primary metric).
- ✗ marks a regression.
- Q3 is binary (yes/no can_close); accuracy / F1 are higher-better.
- Q4 is unit-vector cosine similarity (range [-1, 1]); higher is better.
- Q1/Q1_dest/Q2/Q5 are 3D positions/offsets in meters; MAE / RMSE are lower-better.
- Q6 is per-axis categorical; `acc_all` is the strict-all-3-axes accuracy.
- Parse rate = fraction of samples where the model emitted a parseable value for the dim.

## Full per-dim results (incl. per-axis breakdown)

```json
{
  "base": {
    "q1": {
      "n": 463,
      "mae_x": 0.14954385346394264,
      "mae_y": 0.22695062261823754,
      "mae_z": 0.006038415802128134,
      "mae_overall": 0.12751096396143605,
      "rmse_overall": 0.19457418590384787,
      "n_total": 463,
      "n_parsed": 463,
      "n_skipped_null_gt": 0,
      "parse_rate": 1.0
    },
    "q1_dest": {
      "n": 190,
      "mae_x": 0.23955410122281418,
      "mae_y": 0.22829015064296632,
      "mae_z": 0.007898710309704862,
      "mae_overall": 0.1585809873918285,
      "rmse_overall": 0.23143004124653777,
      "n_total": 463,
      "n_parsed": 190,
      "n_skipped_null_gt": 273,
      "parse_rate": 1.0
    },
    "q2": {
      "n": 463,
      "mae_x": 0.0943750985967,
      "mae_y": 0.047266225809983506,
      "mae_z": 0.1278781580794373,
      "mae_overall": 0.08983982749537363,
      "rmse_overall": 0.12152782601441149,
      "n_total": 463,
      "n_parsed": 463,
      "parse_rate": 1.0
    },
    "q3": {
      "n": 463,
      "accuracy": 0.9287257019438445,
      "f1": 0.0,
      "positive_rate_gt": 0.07127429805615551,
      "positive_rate_pred": 0.0,
      "n_total": 463,
      "n_parsed": 463,
      "parse_rate": 1.0
    },
    "q4": {
      "n": 463,
      "mean_cosine_sim": 0.5202852338096096,
      "median_cosine_sim": 0.37038309489865895,
      "n_total": 463,
      "n_parsed": 463,
      "parse_rate": 1.0
    },
    "q5": {
      "n": 463,
      "mae_x": 0.11140376114262901,
      "mae_y": 0.2450600874773901,
      "mae_z": 0.1318592763547523,
      "mae_overall": 0.16277437499159048,
      "rmse_overall": 0.19858359911625303,
      "n_total": 463,
      "n_parsed": 463,
      "parse_rate": 1.0
    },
    "q6": {
      "n": 463,
      "acc_x": 0.5853131749460043,
      "acc_y": 0.19870410367170627,
      "acc_z": 0.9913606911447084,
      "acc_all": 0.09935205183585313,
      "f1_x": 0.24613987284287012,
      "f1_y": 0.12010682560935228,
      "f1_z": 0.49783080260303686,
      "n_total": 463,
      "n_parsed": 463,
      "parse_rate": 1.0
    }
  },
  "lora": {
    "q1": {
      "n": 463,
      "mae_x": 0.05097185846087515,
      "mae_y": 0.034832417993848126,
      "mae_z": 0.002249217170052015,
      "mae_overall": 0.029351164541591757,
      "rmse_overall": 0.06818718899041559,
      "n_total": 463,
      "n_parsed": 463,
      "n_skipped_null_gt": 0,
      "parse_rate": 1.0
    },
    "q1_dest": {
      "n": 154,
      "mae_x": 0.04816473727930174,
      "mae_y": 0.11009676019619383,
      "mae_z": 0.004176571712168489,
      "mae_overall": 0.0541460230625547,
      "rmse_overall": 0.12107259808000453,
      "n_total": 463,
      "n_parsed": 154,
      "n_skipped_null_gt": 273,
      "parse_rate": 0.8105263157894737
    },
    "q2": {
      "n": 463,
      "mae_x": 0.05265169591598158,
      "mae_y": 0.03268262175401152,
      "mae_z": 0.0685024007935736,
      "mae_overall": 0.051278906154522255,
      "rmse_overall": 0.08359151648738745,
      "n_total": 463,
      "n_parsed": 463,
      "parse_rate": 1.0
    },
    "q3": {
      "n": 463,
      "accuracy": 0.9287257019438445,
      "f1": 0.0,
      "positive_rate_gt": 0.07127429805615551,
      "positive_rate_pred": 0.0,
      "n_total": 463,
      "n_parsed": 463,
      "parse_rate": 1.0
    },
    "q4": {
      "n": 463,
      "mean_cosine_sim": 0.2906217777218822,
      "median_cosine_sim": 0.4089424012958489,
      "n_total": 463,
      "n_parsed": 463,
      "parse_rate": 1.0
    },
    "q5": {
      "n": 463,
      "mae_x": 0.07541672806402466,
      "mae_y": 0.05778633827234548,
      "mae_z": 0.06431798833798574,
      "mae_overall": 0.06584035155811861,
      "rmse_overall": 0.09006815810084076,
      "n_total": 463,
      "n_parsed": 463,
      "parse_rate": 1.0
    },
    "q6": {
      "n": 463,
      "acc_x": 0.7537796976241901,
      "acc_y": 0.8639308855291576,
      "acc_z": 0.9913606911447084,
      "acc_all": 0.6846652267818575,
      "f1_x": 0.6663530061226538,
      "f1_y": 0.8693297196867568,
      "f1_z": 0.49783080260303686,
      "n_total": 463,
      "n_parsed": 463,
      "parse_rate": 1.0
    }
  }
}
```