# Results Summary - Model Comparison (BDG2 test set)

Best model by RMSE: **LightGBM** (RMSE=39.992 kWh, MAE=14.367, CV-RMSE=19.73%, R2=0.971).

## Overall metrics (sorted by RMSE)

| model          |     MAE |    RMSE |    MAPE |   sMAPE |   CV_RMSE |   NRMSE |      MBE |     R2 |
|:---------------|--------:|--------:|--------:|--------:|----------:|--------:|---------:|-------:|
| LightGBM       | 14.3671 | 39.9918 | 10.8485 |  8.7778 |   19.7337 |  2.6009 |  -1.5774 | 0.971  |
| Transformer    | 13.809  | 40.0913 | 10.4295 |  8.3398 |   19.7828 |  2.6074 |  -0.949  | 0.9709 |
| Attn-LSTM      | 14.1765 | 41.0456 | 10.6125 |  8.5802 |   20.2538 |  2.6695 |  -0.6414 | 0.9695 |
| TFT            | 15.2124 | 42.9312 | 11.0969 |  8.7932 |   21.1842 |  2.7921 |  -0.9472 | 0.9666 |
| GRU            | 15.637  | 43.1533 | 11.368  |  9.2806 |   21.2938 |  2.8066 |  -1.7675 | 0.9663 |
| TCN            | 17.2379 | 43.7108 | 12.7068 | 10.4208 |   21.5689 |  2.8428 |   0.1543 | 0.9654 |
| CNN-LSTM       | 16.1029 | 43.8815 | 11.801  |  9.6361 |   21.6531 |  2.8539 |  -1.1111 | 0.9651 |
| LSTM           | 17.9921 | 48.9367 | 12.3029 | 10.114  |   24.1476 |  3.1827 |  -2.1641 | 0.9566 |
| Seasonal-naive | 19.8538 | 52.1949 | 16.2321 | 12.8092 |   25.7553 |  3.3946 |  -0.0587 | 0.9507 |
| Ridge          | 21.0968 | 52.6826 | 15.5384 | 13.0763 |   25.996  |  3.4263 |  -4.5081 | 0.9497 |
| Persistence    | 35.8758 | 78.6568 | 21.6855 | 24.1367 |   38.8128 |  5.1156 | -20.2333 | 0.888  |

## Tables
- metrics_overall, metrics_per_horizon, metrics_per_type
- skill_scores (vs seasonal-naive), significance (Wilcoxon + DM vs best)

## Figures
- eval_01_comparison, eval_02_per_horizon, eval_03_per_type_heatmap
- eval_04_examples, eval_05_scatter