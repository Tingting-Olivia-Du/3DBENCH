# Qwen2.5-VL-3B base vs LoRA on CALVIN

- GT manifest: `data/gt-q6-mv-test/manifest.json` (3679 samples)
- Base responses: `data/base-libero-test-0508/qwen2.5-vl-3b`
- LoRA responses: `data/lora-libero-test-0508/qwen2.5-vl-3b-mv-lora-<dim>`

## Headline metrics (LoRA only answers its own dim; q1 LoRA also covers q1_dest)

| Dim | Metric | Base | LoRA | Δ (LoRA − Base) | Parse rate (base / LoRA) |
|-----|--------|------|------|------------------|---------------------------|
| **Q1** | mae_overall (↓ better) | 0.0193 | 0.0464 | ↑0.0271 ✗ | 1.0000 / 1.0000 |
| **Q1_DEST** | mae_overall (↓ better) | 0.0116 | 0.0089 | ↓0.0026 ✓ | 1.0000 / 1.0000 |
| **Q2** | mae_overall (↓ better) | 0.0786 | 0.0799 | ↑0.0013 ✗ | 1.0000 / 1.0000 |
| **Q3** | accuracy (↑ better) | 0.8947 | 0.9121 | ↑0.0174 ✓ | 1.0000 / 1.0000 |
| **Q4** | mean_cosine_sim (↑ better) | 0.9963 | 0.4253 | ↓0.5711 ✗ | 1.0000 / 1.0000 |
| **Q5** | mae_overall (↓ better) | 0.0723 | 0.0636 | ↓0.0086 ✓ | 1.0000 / 1.0000 |
| **Q6** | acc_all (↑ better) | 0.9342 | 0.6451 | ↓0.2892 ✗ | 1.0000 / 1.0000 |

### Secondary metrics

| Dim | Metric | Base | LoRA |
|-----|--------|------|------|
| **Q1** | rmse_overall | 0.0283 | 0.0995 |
| **Q1_DEST** | rmse_overall | 0.0150 | 0.0219 |
| **Q2** | rmse_overall | 0.1058 | 0.1195 |
| **Q3** | f1 | 0.0000 | 0.0000 |
| **Q4** | median_cosine_sim | 0.9991 | 0.4942 |
| **Q5** | rmse_overall | 0.0996 | 0.0905 |
| **Q6** | f1_x | 0.4830 | 0.5116 |

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
      "n": 76,
      "mae_x": 0.04698248211604263,
      "mae_y": 0.0074482224755084,
      "mae_z": 0.003548487341472238,
      "mae_overall": 0.019326397311007756,
      "rmse_overall": 0.028328743031017536,
      "n_total": 76,
      "n_parsed": 76,
      "n_skipped_null_gt": 0,
      "parse_rate": 1.0
    },
    "q1_dest": {
      "n": 76,
      "mae_x": 0.017765015340363614,
      "mae_y": 0.016429383451067935,
      "mae_z": 0.0005063385293356606,
      "mae_overall": 0.011566912440255738,
      "rmse_overall": 0.014966327199259825,
      "n_total": 76,
      "n_parsed": 76,
      "n_skipped_null_gt": 0,
      "parse_rate": 1.0
    },
    "q2": {
      "n": 76,
      "mae_x": 0.05191578542986191,
      "mae_y": 0.0792786807024027,
      "mae_z": 0.10474860998435685,
      "mae_overall": 0.07864769203887381,
      "rmse_overall": 0.10576158629623593,
      "n_total": 76,
      "n_parsed": 76,
      "parse_rate": 1.0
    },
    "q3": {
      "n": 76,
      "accuracy": 0.8947368421052632,
      "f1": 0.0,
      "positive_rate_gt": 0.10526315789473684,
      "positive_rate_pred": 0.0,
      "n_total": 76,
      "n_parsed": 76,
      "parse_rate": 1.0
    },
    "q4": {
      "n": 76,
      "mean_cosine_sim": 0.996337159128539,
      "median_cosine_sim": 0.9991339003899622,
      "n_total": 76,
      "n_parsed": 76,
      "parse_rate": 1.0
    },
    "q5": {
      "n": 76,
      "mae_x": 0.03739314702444199,
      "mae_y": 0.07695060351404301,
      "mae_z": 0.10247201574450249,
      "mae_overall": 0.07227192209432917,
      "rmse_overall": 0.09955996987989922,
      "n_total": 76,
      "n_parsed": 76,
      "parse_rate": 1.0
    },
    "q6": {
      "n": 76,
      "acc_x": 0.9342105263157895,
      "acc_y": 1.0,
      "acc_z": 0.9868421052631579,
      "acc_all": 0.9342105263157895,
      "f1_x": 0.48299319727891155,
      "f1_y": 1.0,
      "f1_z": 0.4966887417218543,
      "n_total": 76,
      "n_parsed": 76,
      "parse_rate": 1.0
    }
  },
  "lora": {
    "q1": {
      "n": 910,
      "mae_x": 0.09276388197173047,
      "mae_y": 0.042301520457770345,
      "mae_z": 0.0041635859317177,
      "mae_overall": 0.046409662787072856,
      "rmse_overall": 0.09954525339925449,
      "n_total": 910,
      "n_parsed": 910,
      "n_skipped_null_gt": 0,
      "parse_rate": 1.0
    },
    "q1_dest": {
      "n": 910,
      "mae_x": 0.013126125203855662,
      "mae_y": 0.012911978467369863,
      "mae_z": 0.0007742900299391244,
      "mae_overall": 0.008937464567054892,
      "rmse_overall": 0.02186485213920802,
      "n_total": 910,
      "n_parsed": 910,
      "n_skipped_null_gt": 0,
      "parse_rate": 1.0
    },
    "q2": {
      "n": 910,
      "mae_x": 0.0572342685970516,
      "mae_y": 0.08017699028087323,
      "mae_z": 0.10237691810355419,
      "mae_overall": 0.07992939232715973,
      "rmse_overall": 0.11951380918556175,
      "n_total": 910,
      "n_parsed": 910,
      "parse_rate": 1.0
    },
    "q3": {
      "n": 910,
      "accuracy": 0.9120879120879121,
      "f1": 0.0,
      "positive_rate_gt": 0.08791208791208792,
      "positive_rate_pred": 0.0,
      "n_total": 910,
      "n_parsed": 910,
      "parse_rate": 1.0
    },
    "q4": {
      "n": 910,
      "mean_cosine_sim": 0.42527794933810303,
      "median_cosine_sim": 0.49418867017945456,
      "n_total": 910,
      "n_parsed": 910,
      "parse_rate": 1.0
    },
    "q5": {
      "n": 910,
      "mae_x": 0.06450818325983777,
      "mae_y": 0.06284797303069181,
      "mae_z": 0.06358110225754404,
      "mae_overall": 0.06364575284935786,
      "rmse_overall": 0.09054104588635203,
      "n_total": 910,
      "n_parsed": 910,
      "parse_rate": 1.0
    },
    "q6": {
      "n": 910,
      "acc_x": 0.8032967032967033,
      "acc_y": 0.8472527472527472,
      "acc_z": 0.9296703296703297,
      "acc_all": 0.6450549450549451,
      "f1_x": 0.511553951582104,
      "f1_y": 0.8461067441330599,
      "f1_z": 0.837092731829574,
      "n_total": 910,
      "n_parsed": 910,
      "parse_rate": 1.0
    }
  }
}
```