# Qwen2.5-VL-3B base vs LoRA on CALVIN

- GT manifest: `data/gt-q6-mv-calvin/manifest.json` (4320 samples)
- Base responses: `data/runs-mv-base-calvin/qwen2.5-vl-3b`
- LoRA responses: `data/runs-mv-lora-calvin/qwen2.5-vl-3b-mv-lora-<dim>`

## Headline metrics (LoRA only answers its own dim; q1 LoRA also covers q1_dest)

| Dim | Metric | Base | LoRA | Δ (LoRA − Base) | Parse rate (base / LoRA) |
|-----|--------|------|------|------------------|---------------------------|
| **Q1** | mae_overall (↓ better) | 0.2717 | 0.3961 | ↑0.1244 ✗ | 1.0000 / 1.0000 |
| **Q1_DEST** | mae_overall (↓ better) | — | — | — | 0.0000 / 0.0000 |
| **Q2** | mae_overall (↓ better) | 0.2792 | 0.3370 | ↑0.0578 ✗ | 1.0000 / 1.0000 |
| **Q3** | accuracy (↑ better) | 0.6618 | 0.6414 | ↓0.0204 ✗ | 1.0000 / 1.0000 |
| **Q4** | mean_cosine_sim (↑ better) | 0.0184 | 0.0385 | ↑0.0201 ✓ | 0.3391 / 1.0000 |
| **Q5** | mae_overall (↓ better) | 0.0839 | 0.0951 | ↑0.0111 ✗ | 1.0000 / 1.0000 |
| **Q6** | acc_all (↑ better) | 0.0706 | 0.0433 | ↓0.0273 ✗ | 1.0000 / 1.0000 |

### Secondary metrics

| Dim | Metric | Base | LoRA |
|-----|--------|------|------|
| **Q1** | rmse_overall | 0.3513 | 0.4648 |
| **Q1_DEST** | rmse_overall | — | — |
| **Q2** | rmse_overall | 0.3619 | 0.3947 |
| **Q3** | f1 | 0.4642 | 0.0000 |
| **Q4** | median_cosine_sim | 0.0245 | 0.0739 |
| **Q5** | rmse_overall | 0.1496 | 0.1362 |
| **Q6** | f1_x | 0.2470 | 0.1594 |

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
      "n": 4320,
      "mae_x": 0.2902370783120813,
      "mae_y": 0.47578247351555525,
      "mae_z": 0.0489917201921344,
      "mae_overall": 0.2716704240065825,
      "rmse_overall": 0.3513177802394072,
      "n_total": 4320,
      "n_parsed": 4320,
      "n_skipped_null_gt": 0,
      "parse_rate": 1.0
    },
    "q1_dest": {
      "n": 0,
      "mae_x": null,
      "mae_y": null,
      "mae_z": null,
      "mae_overall": null,
      "rmse_overall": null,
      "n_total": 4320,
      "n_parsed": 0,
      "n_skipped_null_gt": 4320,
      "parse_rate": 0.0
    },
    "q2": {
      "n": 4320,
      "mae_x": 0.2949778499531033,
      "mae_y": 0.5026933872917607,
      "mae_z": 0.039833736452515464,
      "mae_overall": 0.2791683245657885,
      "rmse_overall": 0.3619198403005856,
      "n_total": 4320,
      "n_parsed": 4320,
      "parse_rate": 1.0
    },
    "q3": {
      "n": 4320,
      "accuracy": 0.6618055555555555,
      "f1": 0.46424642464246424,
      "positive_rate_gt": 0.35856481481481484,
      "positive_rate_pred": 0.2726851851851852,
      "n_total": 4320,
      "n_parsed": 4320,
      "parse_rate": 1.0
    },
    "q4": {
      "n": 1465,
      "mean_cosine_sim": 0.018395651262498692,
      "median_cosine_sim": 0.024533669819409087,
      "n_total": 4320,
      "n_parsed": 1465,
      "parse_rate": 0.33912037037037035
    },
    "q5": {
      "n": 4320,
      "mae_x": 0.14784146301882745,
      "mae_y": 0.053505260585763444,
      "mae_z": 0.05040582731918034,
      "mae_overall": 0.08391751697459048,
      "rmse_overall": 0.14962601667863795,
      "n_total": 4320,
      "n_parsed": 4320,
      "parse_rate": 1.0
    },
    "q6": {
      "n": 4320,
      "acc_x": 0.32314814814814813,
      "acc_y": 0.44189814814814815,
      "acc_z": 0.25949074074074074,
      "acc_all": 0.07060185185185185,
      "f1_x": 0.2469970957018206,
      "f1_y": 0.23435388105960395,
      "f1_z": 0.22637836887535023,
      "n_total": 4320,
      "n_parsed": 4320,
      "parse_rate": 1.0
    }
  },
  "lora": {
    "q1": {
      "n": 4320,
      "mae_x": 0.5839646127839471,
      "mae_y": 0.11761202744315541,
      "mae_z": 0.48668083754285285,
      "mae_overall": 0.39608582592332237,
      "rmse_overall": 0.4647663245777334,
      "n_total": 4320,
      "n_parsed": 4320,
      "n_skipped_null_gt": 0,
      "parse_rate": 1.0
    },
    "q1_dest": {
      "n": 0,
      "mae_x": null,
      "mae_y": null,
      "mae_z": null,
      "mae_overall": null,
      "rmse_overall": null,
      "n_total": 4320,
      "n_parsed": 0,
      "n_skipped_null_gt": 4320,
      "parse_rate": 0.0
    },
    "q2": {
      "n": 4320,
      "mae_x": 0.5003383966819616,
      "mae_y": 0.10034540645326548,
      "mae_z": 0.41036466089601054,
      "mae_overall": 0.33701615467708734,
      "rmse_overall": 0.3947254542342452,
      "n_total": 4320,
      "n_parsed": 4320,
      "parse_rate": 1.0
    },
    "q3": {
      "n": 4320,
      "accuracy": 0.6414351851851852,
      "f1": 0.0,
      "positive_rate_gt": 0.35856481481481484,
      "positive_rate_pred": 0.0,
      "n_total": 4320,
      "n_parsed": 4320,
      "parse_rate": 1.0
    },
    "q4": {
      "n": 4320,
      "mean_cosine_sim": 0.0385161027963161,
      "median_cosine_sim": 0.07387703700202194,
      "n_total": 4320,
      "n_parsed": 4320,
      "parse_rate": 1.0
    },
    "q5": {
      "n": 4320,
      "mae_x": 0.09535701505521124,
      "mae_y": 0.10890608784176835,
      "mae_z": 0.08093898075476791,
      "mae_overall": 0.09506736121725078,
      "rmse_overall": 0.13616734239041678,
      "n_total": 4320,
      "n_parsed": 4320,
      "parse_rate": 1.0
    },
    "q6": {
      "n": 4320,
      "acc_x": 0.3106481481481482,
      "acc_y": 0.24791666666666667,
      "acc_z": 0.4585648148148148,
      "acc_all": 0.043287037037037034,
      "f1_x": 0.15943029933480748,
      "f1_y": 0.2454147915196462,
      "f1_z": 0.21222795073188672,
      "n_total": 4320,
      "n_parsed": 4320,
      "parse_rate": 1.0
    }
  }
}
```