# March Machine Learning Mania 2026

Predict NCAA tournament outcomes (men's + women's) using an ensemble of **XGBoost**, **LightGBM**, **CatBoost**, and **TabICL** v2.

- **Competition:** [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
- **Notebook:** [March ML Mania 2026 — TabICL](https://www.kaggle.com/code/aelhajj/march-ml-mania-2026-tabicl)
- **Metric:** Brier score (lower is better)
- **Stage 1 result:** **0.13643**

## What’s in this repo

- **`utils.py`** — Data loading, Elo/season stats/Massey features, model cache (train-or-load), LOSO CV, submission generation, and plotting helpers.
- **`theme.py`** — Matplotlib theme (e.g. Catppuccin Mocha).

## Next steps

1. **Test splitting women / men** — Train or evaluate separately by gender (e.g. men-only vs women-only models or LOSO by gender) to see if it improves validation.
2. **Submit Stage 2** — Upload `submission_stage2.csv` as `submission.csv` for the 2026 leaderboard (rescored after the tournament).

## License

See [LICENSE](LICENSE).
