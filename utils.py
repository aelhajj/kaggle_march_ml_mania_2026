"""
Utilities for March Machine Learning Mania 2026:
data loading, feature engineering, model caching, evaluation, and visualization.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
import matplotlib.pyplot as plt


# ── Hyperparameter cache ──────────────────────────────────────

_PARAMS_FILE = Path("best_params.json")


def save_params(name: str, params: dict, path: Path = _PARAMS_FILE) -> None:
    """Save best hyperparameters for a named model to a JSON file."""
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = {k: v.item() if hasattr(v, "item") else v for k, v in params.items()}
    path.write_text(json.dumps(data, indent=2))


def load_params(name: str, path: Path = _PARAMS_FILE) -> dict | None:
    """Return saved params for *name*, or None if not found."""
    if not path.exists():
        return None
    return json.loads(path.read_text()).get(name)


# ── Model cache ───────────────────────────────────────────────

_MODELS_DIR = Path("models")


def save_model(name: str, model, models_dir: Path = _MODELS_DIR) -> None:
    """Persist a fitted model to disk with joblib."""
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / f"{name}.joblib")


def load_model(name: str, models_dir: Path = _MODELS_DIR):
    """Load a fitted model from disk. Returns None if not found."""
    path = models_dir / f"{name}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def _instantiate(model_factory, params: dict, random_state: int, model_kwargs: dict | None = None):
    """
    Build an unfitted model from either a class or a factory callable.
      - class    → cls(**params, **model_kwargs, random_state=random_state)
      - callable → fn(**params)   (caller owns random_state)
    """
    extra = model_kwargs or {}
    if isinstance(model_factory, type):
        return model_factory(**params, **extra, random_state=random_state)
    return model_factory(**params)


def train_or_load(name: str, model_factory, X_train, y_train,
                  param_dist: dict | None = None, random_state: int = 42,
                  model_kwargs: dict | None = None,
                  **search_kwargs):
    """
    Load a fitted model from disk if available, otherwise train one.

    Resolution order:
      1. Saved model  (models/<name>.joblib)  → load, no fitting needed
      2. Saved params (best_params.json)       → refit with those params
      3. RandomizedSearchCV                    → full search, then save params + model

    If param_dist is None, no search is performed.
    """
    model = load_model(name)
    if model is not None:
        print(f"[cache] Loaded {name} from disk.")
        return model

    if param_dist is not None:
        cached = load_params(name)
        if cached:
            print(f"[cache] Loaded {name} params: {cached}")
            model = _instantiate(model_factory, cached, random_state, model_kwargs)
            model.fit(X_train, y_train)
        else:
            search = RandomizedSearchCV(
                _instantiate(model_factory, {}, random_state, model_kwargs),
                param_dist,
                **search_kwargs,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            save_params(name, search.best_params_)
            print(f"Best params: {search.best_params_}")
    else:
        model = _instantiate(model_factory, {}, random_state, model_kwargs)
        model.fit(X_train, y_train)

    save_model(name, model)
    return model


# ── Data loading ──────────────────────────────────────────────

DATA_DIR = Path("data")

_CSV_MAP = {
    "m_teams":          "MTeams.csv",
    "w_teams":          "WTeams.csv",
    "m_regular":        "MRegularSeasonCompactResults.csv",
    "w_regular":        "WRegularSeasonCompactResults.csv",
    "m_regular_detail": "MRegularSeasonDetailedResults.csv",
    "w_regular_detail": "WRegularSeasonDetailedResults.csv",
    "m_tourney":        "MNCAATourneyCompactResults.csv",
    "w_tourney":        "WNCAATourneyCompactResults.csv",
    "m_tourney_detail": "MNCAATourneyDetailedResults.csv",
    "w_tourney_detail": "WNCAATourneyDetailedResults.csv",
    "m_seeds":          "MNCAATourneySeeds.csv",
    "w_seeds":          "WNCAATourneySeeds.csv",
    "m_massey":         "MMasseyOrdinals.csv",
    "sample_sub":       "SampleSubmissionStage1.csv",
    "sample_sub2":      "SampleSubmissionStage2.csv",
}


def load_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load all competition CSVs into a dict keyed by short names."""
    data = {}
    for key, fname in _CSV_MAP.items():
        path = data_dir / fname
        if path.exists():
            data[key] = pd.read_csv(path)
    return data


# ── Seed helpers ──────────────────────────────────────────────

def parse_seed(seed_str: str) -> int:
    """Extract numeric seed from strings like 'W01', 'X16a' → 1, 16."""
    return int(seed_str[1:3])


def build_seed_map(m_seeds: pd.DataFrame, w_seeds: pd.DataFrame) -> dict[tuple[int, int], int]:
    """Return {(Season, TeamID): numeric_seed} for all seeds."""
    seed_map = {}
    for df in [m_seeds, w_seeds]:
        for _, row in df.iterrows():
            seed_map[(row["Season"], row["TeamID"])] = parse_seed(row["Seed"])
    return seed_map


# ── Elo rating system ─────────────────────────────────────────

def compute_elo(
    regular_df: pd.DataFrame,
    tourney_df: pd.DataFrame | None = None,
    k: float = 20.0,
    init: float = 1500.0,
    hca: float = 100.0,
    revert_pct: float = 0.25,
    include_tourney: bool = True,
) -> dict[tuple[int, int], float]:
    """
    Compute end-of-period Elo for every (Season, TeamID).

    If include_tourney=True:  uses regular + tournament games → end-of-season Elo.
    If include_tourney=False: uses regular season only → pre-tournament Elo.

    Between seasons, each team's Elo regresses toward `init` by `revert_pct`.
    """
    if include_tourney and tourney_df is not None:
        all_games = pd.concat([regular_df, tourney_df]).sort_values(["Season", "DayNum"])
    else:
        all_games = regular_df.sort_values(["Season", "DayNum"])

    elo = {}
    season_elos = {}
    prev_season = None

    for _, row in all_games.iterrows():
        season = row["Season"]

        if season != prev_season and prev_season is not None:
            for tid, r in elo.items():
                season_elos[(prev_season, tid)] = r
            elo = {tid: (1 - revert_pct) * r + revert_pct * init for tid, r in elo.items()}
        prev_season = season

        w_id, l_id = row["WTeamID"], row["LTeamID"]
        w_elo = elo.get(w_id, init)
        l_elo = elo.get(l_id, init)

        w_loc = row.get("WLoc", "N")
        w_adj = w_elo + (hca if w_loc == "H" else (-hca if w_loc == "A" else 0))

        exp_w = 1.0 / (1.0 + 10 ** ((l_elo - w_adj) / 400.0))
        elo[w_id] = w_elo + k * (1.0 - exp_w)
        elo[l_id] = l_elo + k * (0.0 - (1.0 - exp_w))

    if prev_season is not None:
        for tid, r in elo.items():
            season_elos[(prev_season, tid)] = r

    return season_elos


# ── Season statistics ─────────────────────────────────────────

def compute_season_stats(detail_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team-per-season aggregated statistics from detailed results.

    Views each game from both the winner's and loser's perspective,
    then groups by (Season, TeamID) to get averages.
    """
    # Winner perspective
    w = detail_df.assign(
        TeamID=detail_df["WTeamID"],
        Score=detail_df["WScore"],
        OppScore=detail_df["LScore"],
        Win=1,
        FGM=detail_df["WFGM"], FGA=detail_df["WFGA"],
        FGM3=detail_df["WFGM3"], FGA3=detail_df["WFGA3"],
        FTM=detail_df["WFTM"], FTA=detail_df["WFTA"],
        OR=detail_df["WOR"], DR=detail_df["WDR"],
        Ast=detail_df["WAst"], TO=detail_df["WTO"],
        Stl=detail_df["WStl"], Blk=detail_df["WBlk"],
        OppFGM=detail_df["LFGM"], OppFGA=detail_df["LFGA"],
        OppFGM3=detail_df["LFGM3"], OppFGA3=detail_df["LFGA3"],
    )[["Season", "TeamID", "Score", "OppScore", "Win",
       "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
       "OR", "DR", "Ast", "TO", "Stl", "Blk",
       "OppFGM", "OppFGA", "OppFGM3", "OppFGA3"]]

    # Loser perspective
    l = detail_df.assign(
        TeamID=detail_df["LTeamID"],
        Score=detail_df["LScore"],
        OppScore=detail_df["WScore"],
        Win=0,
        FGM=detail_df["LFGM"], FGA=detail_df["LFGA"],
        FGM3=detail_df["LFGM3"], FGA3=detail_df["LFGA3"],
        FTM=detail_df["LFTM"], FTA=detail_df["LFTA"],
        OR=detail_df["LOR"], DR=detail_df["LDR"],
        Ast=detail_df["LAst"], TO=detail_df["LTO"],
        Stl=detail_df["LStl"], Blk=detail_df["LBlk"],
        OppFGM=detail_df["WFGM"], OppFGA=detail_df["WFGA"],
        OppFGM3=detail_df["WFGM3"], OppFGA3=detail_df["WFGA3"],
    )[w.columns]

    games = pd.concat([w, l], ignore_index=True)
    grouped = games.groupby(["Season", "TeamID"])

    stats = grouped.agg(
        GamesPlayed=("Win", "count"),
        Wins=("Win", "sum"),
        AvgScore=("Score", "mean"),
        AvgOppScore=("OppScore", "mean"),
        FGM=("FGM", "sum"), FGA=("FGA", "sum"),
        FGM3=("FGM3", "sum"), FGA3=("FGA3", "sum"),
        FTM=("FTM", "sum"), FTA=("FTA", "sum"),
        AvgOR=("OR", "mean"), AvgDR=("DR", "mean"),
        AvgAst=("Ast", "mean"), AvgTO=("TO", "mean"),
        AvgStl=("Stl", "mean"), AvgBlk=("Blk", "mean"),
        OppFGM=("OppFGM", "sum"), OppFGA=("OppFGA", "sum"),
        OppFGM3=("OppFGM3", "sum"), OppFGA3=("OppFGA3", "sum"),
    ).reset_index()

    stats["WinPct"] = stats["Wins"] / stats["GamesPlayed"]
    stats["AvgScoreMargin"] = stats["AvgScore"] - stats["AvgOppScore"]
    stats["FGPct"] = stats["FGM"] / stats["FGA"]
    stats["FG3Pct"] = stats["FGM3"] / stats["FGA3"].replace(0, np.nan)
    stats["FTPct"] = stats["FTM"] / stats["FTA"].replace(0, np.nan)
    stats["OppFGPct"] = stats["OppFGM"] / stats["OppFGA"]
    stats["OppFG3Pct"] = stats["OppFGM3"] / stats["OppFGA3"].replace(0, np.nan)

    keep = ["Season", "TeamID", "GamesPlayed", "Wins", "WinPct",
            "AvgScore", "AvgOppScore", "AvgScoreMargin",
            "FGPct", "FG3Pct", "FTPct",
            "AvgOR", "AvgDR", "AvgAst", "AvgTO", "AvgStl", "AvgBlk",
            "OppFGPct", "OppFG3Pct"]
    return stats[keep]


# ── Massey ordinals ───────────────────────────────────────────

def compute_massey_features(
    massey_df: pd.DataFrame,
    day_threshold: int = 128,
) -> pd.DataFrame:
    """
    Aggregate Massey ordinal rankings into per-team-per-season features.

    Filters to late-season rankings (DayNum >= day_threshold) then computes
    mean and median rank across all systems for each (Season, TeamID).
    Men's only — women's teams will get NaN.
    """
    late = massey_df[massey_df["RankingDayNum"] >= day_threshold]
    agg = late.groupby(["Season", "TeamID"])["OrdinalRank"].agg(
        MasseyMean="mean",
        MasseyMedian="median",
    ).reset_index()
    return agg


# ── Matchup features ─────────────────────────────────────────

_STAT_DIFF_COLS = [
    "WinPct", "AvgScoreMargin", "FGPct", "FG3Pct", "FTPct",
    "AvgOR", "AvgDR", "AvgAst", "AvgTO", "AvgStl", "AvgBlk",
    "OppFGPct", "OppFG3Pct",
]


def _get_team_stats(team_id: int, season: int, stats_df: pd.DataFrame) -> dict:
    """Lookup season stats for a team, returning NaN dict if not found."""
    mask = (stats_df["Season"] == season) & (stats_df["TeamID"] == team_id)
    rows = stats_df.loc[mask]
    if rows.empty:
        return {col: np.nan for col in _STAT_DIFF_COLS}
    row = rows.iloc[0]
    return {col: row[col] for col in _STAT_DIFF_COLS}


def _get_massey(team_id: int, season: int, massey_df: pd.DataFrame | None) -> dict:
    """Lookup Massey features for a team."""
    if massey_df is None:
        return {"MasseyMean": np.nan, "MasseyMedian": np.nan}
    mask = (massey_df["Season"] == season) & (massey_df["TeamID"] == team_id)
    rows = massey_df.loc[mask]
    if rows.empty:
        return {"MasseyMean": np.nan, "MasseyMedian": np.nan}
    row = rows.iloc[0]
    return {"MasseyMean": row["MasseyMean"], "MasseyMedian": row["MasseyMedian"]}


def build_matchup_features(
    t1: int, t2: int, season: int,
    elo_prev: dict, elo_curr: dict,
    seed_map: dict, stats_df: pd.DataFrame,
    massey_df: pd.DataFrame | None,
) -> dict:
    """
    Build feature dict for a Team1-vs-Team2 matchup.

    t1 must be the lower TeamID. All difference features are T1 − T2.
    Uses previous season Elo and current season stats/seeds/Massey.
    """
    e1_prev = elo_prev.get((season - 1, t1), 1500.0)
    e2_prev = elo_prev.get((season - 1, t2), 1500.0)
    e1_curr = elo_curr.get((season, t1), 1500.0)
    e2_curr = elo_curr.get((season, t2), 1500.0)

    s1 = seed_map.get((season, t1), np.nan)
    s2 = seed_map.get((season, t2), np.nan)

    feats = {
        "EloPrevDiff": e1_prev - e2_prev,
        "EloCurrDiff": e1_curr - e2_curr,
        "SeedT1": s1,
        "SeedT2": s2,
    }

    # Seed diff: positive means T1 has better (lower) seed
    if not (np.isnan(s1) or np.isnan(s2)):
        feats["SeedDiff"] = s2 - s1
    else:
        feats["SeedDiff"] = np.nan

    # Season stats diffs
    st1 = _get_team_stats(t1, season, stats_df)
    st2 = _get_team_stats(t2, season, stats_df)
    for col in _STAT_DIFF_COLS:
        if col == "AvgTO":
            feats[f"{col}Diff"] = st2[col] - st1[col]  # fewer TOs is better
        else:
            feats[f"{col}Diff"] = st1[col] - st2[col]

    # Massey diffs
    m1 = _get_massey(t1, season, massey_df)
    m2 = _get_massey(t2, season, massey_df)
    feats["MasseyMeanDiff"] = m2["MasseyMean"] - m1["MasseyMean"]  # lower rank is better
    feats["MasseyMedianDiff"] = m2["MasseyMedian"] - m1["MasseyMedian"]

    return feats


# ── Training data builder ─────────────────────────────────────

def build_training_data(
    m_tourney: pd.DataFrame,
    w_tourney: pd.DataFrame,
    elo_prev: dict,
    elo_curr: dict,
    seed_map: dict,
    stats_df: pd.DataFrame,
    massey_df: pd.DataFrame | None,
    min_season: int = 2003,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build (X, y, seasons) from historical tournament results.

    For each game: Team1 = lower TeamID, y = 1 if Team1 won.
    Features use previous-season Elo + current-season stats/seeds/Massey.
    """
    records = []
    labels = []
    szns = []

    for t_df in [m_tourney, w_tourney]:
        for _, row in t_df.iterrows():
            season = row["Season"]
            if season < min_season:
                continue

            w_id, l_id = row["WTeamID"], row["LTeamID"]
            t1, t2 = min(w_id, l_id), max(w_id, l_id)

            feats = build_matchup_features(
                t1, t2, season,
                elo_prev, elo_curr, seed_map, stats_df, massey_df,
            )
            records.append(feats)
            labels.append(1 if w_id == t1 else 0)
            szns.append(season)

    X = pd.DataFrame(records)
    y = pd.Series(labels, name="y")
    seasons = pd.Series(szns, name="Season")
    return X, y, seasons


# ── Evaluation ────────────────────────────────────────────────

def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier score = mean((y_true - y_pred)^2). Lower is better."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def leave_one_season_out_cv(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    impute: bool = False,
) -> dict[int, float]:
    """
    Leave-one-season-out CV returning {season: brier_score}.

    model_factory: callable returning an unfitted sklearn-compatible model.
    impute: if True, fill NaN with training-set column medians (for models
            that don't handle NaN natively like AdaBoost/TabICL).
    """
    results = {}
    for season in sorted(seasons.unique()):
        train_mask = seasons != season
        test_mask = seasons == season

        X_train, X_test = X[train_mask].copy(), X[test_mask].copy()
        y_train, y_test = y[train_mask], y[test_mask]

        if impute:
            medians = X_train.median()
            X_train = X_train.fillna(medians)
            X_test = X_test.fillna(medians)

        model = model_factory()
        model.fit(X_train.values, y_train.values)
        proba = model.predict_proba(X_test.values)[:, 1]
        results[season] = brier_score(y_test.values, proba)

    mean_b = np.mean(list(results.values()))
    print(f"  LOSO mean Brier: {mean_b:.4f}")
    return results


# ── Submission generation ─────────────────────────────────────

def generate_submission(
    sample_sub: pd.DataFrame,
    models: dict[str, object],
    weights: dict[str, float],
    elo_prev: dict,
    elo_curr: dict,
    seed_map: dict,
    stats_df: pd.DataFrame,
    massey_df: pd.DataFrame | None,
    feature_cols: list[str],
    impute_cols: dict[str, float] | None = None,
    clip_range: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """
    Generate submission by ensembling multiple models.

    Vectorizes feature construction then calls predict_proba per model.
    impute_cols: {col: median_value} for models that need imputation.
    """
    ids = sample_sub["ID"].str.split("_", expand=True)
    ids.columns = ["Season", "T1", "T2"]
    ids = ids.astype(int)

    records = []
    for _, row in ids.iterrows():
        feats = build_matchup_features(
            row["T1"], row["T2"], row["Season"],
            elo_prev, elo_curr, seed_map, stats_df, massey_df,
        )
        records.append(feats)

    X_sub = pd.DataFrame(records)[feature_cols]

    preds = np.zeros(len(X_sub))
    for name, model in models.items():
        w = weights[name]
        X_input = X_sub.copy()
        if impute_cols is not None:
            X_input = X_input.fillna(impute_cols)
        proba = model.predict_proba(X_input.values)[:, 1]
        preds += w * proba

    preds = np.clip(preds, *clip_range)

    sub = sample_sub[["ID"]].copy()
    sub["Pred"] = preds
    return sub


# ── Visualization helpers ─────────────────────────────────────

def plot_brier_by_season(
    cv_results: dict[str, dict[int, float]],
    colors: list[str],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Grouped bar chart of Brier score per season for each model."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    df = pd.DataFrame(cv_results)
    df.index.name = "Season"
    df.plot.bar(ax=ax, color=colors[:len(df.columns)], edgecolor="none", width=0.8)
    ax.set_ylabel("Brier Score")
    ax.set_title("Brier Score by Season (LOSO CV)")
    ax.legend(title="Model")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return ax


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 15,
    color: str = "#89b4fa",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Horizontal bar chart of top feature importances (tree models)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color=color, edgecolor="none",
    )
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    plt.tight_layout()
    return ax


def plot_prediction_distribution(
    preds: pd.Series,
    color: str = "#89b4fa",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Histogram of predicted probabilities."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.hist(preds, bins=50, color=color, edgecolor="none", alpha=0.85)
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.7, label="0.5")
    ax.set_xlabel("P(Team1 wins)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Distribution")
    ax.legend()
    plt.tight_layout()
    return ax


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    color: str = "#89b4fa",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Calibration plot: predicted vs actual win probability."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)

    ax.plot(mean_pred, frac_pos, "o-", color=color, label="Model")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return ax


def plot_model_comparison(
    model_briers: dict[str, float],
    colors: list[str],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar chart comparing mean Brier score across models."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    names = list(model_briers.keys())
    scores = list(model_briers.values())
    bars = ax.bar(names, scores, color=colors[:len(names)], edgecolor="none")
    ax.set_ylabel("Mean Brier Score")
    ax.set_title("Model Comparison (LOSO CV)")

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{score:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return ax
