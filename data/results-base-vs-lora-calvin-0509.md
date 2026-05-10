# Qwen2.5-VL-3B base vs LoRA on CALVIN

- GT manifest: `data/gt-q6-mv-calvin/manifest.json` (4320 samples)
- Base responses: `data/runs-mv-base-calvin-0509-fixed/qwen2.5-vl-3b`
- LoRA responses: `data/runs-mv-lora-calvin-0509-fixed/qwen2.5-vl-3b-mv-lora-<dim>`

## Calvin results before and after finetune

| Dim | Metric | Base | LoRA | Δ (LoRA − Base) | Parse rate (base / LoRA) |
|-----|--------|------|------|------------------|---------------------------|
| **Q1** | mae_overall (↓ better) | 0.1657 | 0.1738 | ↑0.0081 ✗ | 1.0000 / 0.9641 |
| **Q1_DEST** | mae_overall (↓ better) | — | — | — | 0.0000 / 0.0000 |
| **Q2** | mae_overall (↓ better) | 0.1245 | 0.1851 | ↑0.0606 ✗ | 1.0000 / 1.0000 |
| **Q3** | accuracy (↑ better) | 0.6447 | 0.6414 | ↓0.0032 ✗ | 1.0000 / 1.0000 |
| **Q4** | mean_cosine_sim (↑ better) | 0.0480 | 0.2294 | ↑0.1814 ✓ | 0.2961 / 1.0000 |
| **Q5** | mae_overall (↓ better) | 0.0791 | 0.0964 | ↑0.0173 ✗ | 1.0000 / 1.0000 |
| **Q6** | acc_all (↑ better) | 0.0303 | 0.0225 | ↓0.0079 ✗ | 1.0000 / 1.0000 |

Headline metrics (LoRA only answers its own dim; q1 LoRA also covers q1_dest)


### Secondary metrics

| Dim | Metric | Base | LoRA |
|-----|--------|------|------|
| **Q1** | rmse_overall | 0.2416 | 0.2492 |
| **Q1_DEST** | rmse_overall | — | — |
| **Q2** | rmse_overall | 0.1823 | 0.2454 |
| **Q3** | f1 | 0.1891 | 0.0000 |
| **Q4** | median_cosine_sim | 0.0341 | 0.3242 |
| **Q5** | rmse_overall | 0.1468 | 0.1424 |
| **Q6** | f1_x | 0.1374 | 0.1580 |

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
      "mae_x": 0.33364644975033414,
      "mae_y": 0.09525486989743605,
      "mae_z": 0.06829251596524398,
      "mae_overall": 0.1657312785376768,
      "rmse_overall": 0.24159162291904868,
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
      "mae_x": 0.2427499793263799,
      "mae_y": 0.08572186088901436,
      "mae_z": 0.04502607850702807,
      "mae_overall": 0.12449930624080766,
      "rmse_overall": 0.18234854130381017,
      "n_total": 4320,
      "n_parsed": 4320,
      "parse_rate": 1.0
    },
    "q3": {
      "n": 4320,
      "accuracy": 0.6446759259259259,
      "f1": 0.18911780243000528,
      "positive_rate_gt": 0.35856481481481484,
      "positive_rate_pred": 0.07962962962962963,
      "n_total": 4320,
      "n_parsed": 4320,
      "parse_rate": 1.0
    },
    "q4": {
      "n": 1279,
      "mean_cosine_sim": 0.04801281906477129,
      "median_cosine_sim": 0.0340983705106341,
      "n_total": 4320,
      "n_parsed": 1279,
      "parse_rate": 0.29606481481481484
    },
    "q5": {
      "n": 4320,
      "mae_x": 0.1274383269617342,
      "mae_y": 0.05768617377330981,
      "mae_z": 0.05215081269763123,
      "mae_overall": 0.07909177114422494,
      "rmse_overall": 0.1468149802615281,
      "n_total": 4320,
      "n_parsed": 4320,
      "parse_rate": 1.0
    },
    "q6": {
      "n": 4320,
      "acc_x": 0.20625,
      "acc_y": 0.2462962962962963,
      "acc_z": 0.24467592592592594,
      "acc_all": 0.030324074074074073,
      "f1_x": 0.1374091569767442,
      "f1_y": 0.15943017887778796,
      "f1_z": 0.18275665050732656,
      "n_total": 4320,
      "n_parsed": 4320,
      "parse_rate": 1.0
    }
  },
  "lora": {
    "q1": {
      "n": 4165,
      "mae_x": 0.1771954759990412,
      "mae_y": 0.14318108132626522,
      "mae_z": 0.20101501485313478,
      "mae_overall": 0.17379719072614636,
      "rmse_overall": 0.24916173037561679,
      "n_total": 4320,
      "n_parsed": 4165,
      "n_skipped_null_gt": 0,
      "parse_rate": 0.9641203703703703
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
      "mae_x": 0.22341352296235678,
      "mae_y": 0.27106312057644455,
      "mae_z": 0.06072681605821377,
      "mae_overall": 0.18506781986566878,
      "rmse_overall": 0.24541410087247129,
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
      "mean_cosine_sim": 0.22944444952880375,
      "median_cosine_sim": 0.3242336678164591,
      "n_total": 4320,
      "n_parsed": 4320,
      "parse_rate": 1.0
    },
    "q5": {
      "n": 4320,
      "mae_x": 0.13292831950550207,
      "mae_y": 0.08338133540225146,
      "mae_z": 0.07289349676205004,
      "mae_overall": 0.09640105055660173,
      "rmse_overall": 0.14236494702453706,
      "n_total": 4320,
      "n_parsed": 4320,
      "parse_rate": 1.0
    },
    "q6": {
      "n": 4320,
      "acc_x": 0.30972222222222223,
      "acc_y": 0.3013888888888889,
      "acc_z": 0.46087962962962964,
      "acc_all": 0.022453703703703705,
      "f1_x": 0.15796476711720026,
      "f1_y": 0.22554052028917657,
      "f1_z": 0.2354503690732808,
      "n_total": 4320,
      "n_parsed": 4320,
      "parse_rate": 1.0
    }
  }
}
```