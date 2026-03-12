"""
March Machine Learning Mania 2026: data loading, features, model cache, evaluation, visualization.

One big util file because it's already complicated to use these in Kaggle.
"""

import gc
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress
from sklearn.calibration import calibration_curve
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt


# ── Hyperparameter & model cache ───────────────────────────────

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
                  sample_weight: np.ndarray | None = None,
                  **search_kwargs):
    """
    Load a fitted model from disk if available, otherwise train one.

    Resolution order:
      1. Saved model  (models/<name>.joblib)  → load, no fitting needed
      2. Saved params (best_params.json)       → refit with those params
      3. RandomizedSearchCV                    → full search, then save params + model

    If param_dist is None, no search is performed.
    sample_weight is passed to fit() when provided (XGBoost/LightGBM support it).
    """
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}

    model = load_model(name)
    if model is not None:
        print(f"[cache] Loaded {name} from disk.")
        return model

    if param_dist is not None:
        cached = load_params(name)
        if cached:
            print(f"[cache] Loaded {name} params: {cached}")
            model = _instantiate(model_factory, cached, random_state, model_kwargs)
            model.fit(X_train, y_train, **fit_kwargs)
        else:
            search = RandomizedSearchCV(
                _instantiate(model_factory, {}, random_state, model_kwargs),
                param_dist,
                random_state=random_state,
                **search_kwargs,
            )
            search.fit(X_train, y_train, **fit_kwargs)
            model = search.best_estimator_
            save_params(name, search.best_params_)
            print(f"Best params: {search.best_params_}")
    else:
        model = _instantiate(model_factory, {}, random_state, model_kwargs)
        model.fit(X_train, y_train, **fit_kwargs)

    save_model(name, model)
    return model


# ── Data loading ───────────────────────────────────────────────

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
    "m_conferences":    "MTeamConferences.csv",
    "w_conferences":    "WTeamConferences.csv",
    "m_coaches":        "MTeamCoaches.csv",
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


# ── Seeds ──────────────────────────────────────────────────────

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


# ── Elo rating ─────────────────────────────────────────────────

def compute_elo(
    regular_df: pd.DataFrame,
    tourney_df: pd.DataFrame | None = None,
    k: float = 20.0,
    init: float = 1500.0,
    hca: float = 100.0,
    revert_pct: float = 0.25,
    include_tourney: bool = True,
    mov: bool = False,
) -> dict[tuple[int, int], float]:
    """
    Compute end-of-period Elo for every (Season, TeamID).

    If include_tourney=True:  uses regular + tournament games → end-of-season Elo.
    If include_tourney=False: uses regular season only → pre-tournament Elo.
    If mov=True: scale K by margin of victory (FiveThirtyEight-style).

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

        if mov and "WScore" in row.index:
            margin = row["WScore"] - row["LScore"]
            elo_diff = w_adj - l_elo
            mov_mult = np.log(abs(margin) + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))
        else:
            mov_mult = 1.0

        elo[w_id] = w_elo + k * mov_mult * (1.0 - exp_w)
        elo[l_id] = l_elo + k * mov_mult * (0.0 - (1.0 - exp_w))

    if prev_season is not None:
        for tid, r in elo.items():
            season_elos[(prev_season, tid)] = r

    return season_elos


def compute_elo_trajectory_stats(
    regular_df: pd.DataFrame,
    k: float = 20.0,
    init: float = 1500.0,
    hca: float = 100.0,
    revert_pct: float = 0.25,
    mov: bool = False,
) -> dict[tuple[int, int], dict]:
    """
    Compute within-season Elo trajectory stats per (Season, TeamID).

    Returns {(season, team_id): {"EloTrend": float, "EloStd": float}}.
    EloTrend = slope of Elo over the regular season (positive = team improving).
    EloStd   = std dev of within-season Elo (lower = more consistent team).
    If mov=True: scale K by margin of victory (FiveThirtyEight-style).
    """
    games = regular_df.sort_values(["Season", "DayNum"])
    elo: dict[int, float] = {}
    trajectory: dict[tuple[int, int], list[float]] = {}
    prev_season = None

    for _, row in games.iterrows():
        season = row["Season"]
        if season != prev_season and prev_season is not None:
            elo = {tid: (1 - revert_pct) * r + revert_pct * init for tid, r in elo.items()}
        prev_season = season

        w_id, l_id = row["WTeamID"], row["LTeamID"]
        w_elo = elo.get(w_id, init)
        l_elo = elo.get(l_id, init)

        w_loc = row.get("WLoc", "N")
        w_adj = w_elo + (hca if w_loc == "H" else (-hca if w_loc == "A" else 0))
        exp_w = 1.0 / (1.0 + 10 ** ((l_elo - w_adj) / 400.0))

        if mov and "WScore" in row.index:
            margin = row["WScore"] - row["LScore"]
            elo_diff = w_adj - l_elo
            mov_mult = np.log(abs(margin) + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))
        else:
            mov_mult = 1.0

        elo[w_id] = w_elo + k * mov_mult * (1.0 - exp_w)
        elo[l_id] = l_elo + k * mov_mult * (0.0 - (1.0 - exp_w))

        for tid, r in [(w_id, elo[w_id]), (l_id, elo[l_id])]:
            key = (season, tid)
            if key not in trajectory:
                trajectory[key] = []
            trajectory[key].append(r)

    result = {}
    for (season, tid), vals in trajectory.items():
        if len(vals) < 3:
            result[(season, tid)] = {"EloTrend": np.nan, "EloStd": np.nan}
        else:
            slope = float(linregress(range(len(vals)), vals).slope)
            result[(season, tid)] = {"EloTrend": slope, "EloStd": float(np.std(vals))}

    return result


# ── Season stats (Kenpom-style) ────────────────────────────────

def compute_season_stats(detail_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team-per-season possession-based statistics from detailed results.

    Uses the Kenpom possession formula: Poss = FGA - OR + TO + 0.475 * FTA.
    Returns efficiency (per 100 possessions) and rate stats instead of raw averages.
    """
    cols = ["Season", "TeamID", "Score", "OppScore", "Win",
            "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
            "OR", "DR", "Ast", "TO",
            "OppFGA", "OppOR", "OppDR", "OppTO", "OppFTA"]

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
        OppFGA=detail_df["LFGA"],
        OppOR=detail_df["LOR"], OppDR=detail_df["LDR"],
        OppTO=detail_df["LTO"], OppFTA=detail_df["LFTA"],
    )[cols]

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
        OppFGA=detail_df["WFGA"],
        OppOR=detail_df["WOR"], OppDR=detail_df["WDR"],
        OppTO=detail_df["WTO"], OppFTA=detail_df["WFTA"],
    )[cols]

    games = pd.concat([w, l], ignore_index=True)
    grouped = games.groupby(["Season", "TeamID"])

    stats = grouped.agg(
        GamesPlayed=("Win", "count"),
        TotalPts=("Score", "sum"),
        TotalOppPts=("OppScore", "sum"),
        FGM=("FGM", "sum"), FGA=("FGA", "sum"),
        FGM3=("FGM3", "sum"), FGA3=("FGA3", "sum"),
        FTA=("FTA", "sum"),
        OR=("OR", "sum"), DR=("DR", "sum"),
        Ast=("Ast", "sum"), TO=("TO", "sum"),
        OppFGA=("OppFGA", "sum"),
        OppOR=("OppOR", "sum"), OppDR=("OppDR", "sum"),
        OppTO=("OppTO", "sum"), OppFTA=("OppFTA", "sum"),
    ).reset_index()

    # Possession estimation (Kenpom formula)
    stats["Poss"] = stats["FGA"] - stats["OR"] + stats["TO"] + 0.475 * stats["FTA"]
    stats["OppPoss"] = stats["OppFGA"] - stats["OppOR"] + stats["OppTO"] + 0.475 * stats["OppFTA"]

    # Efficiency metrics (per 100 possessions)
    stats["OffEff"] = (stats["TotalPts"] / stats["Poss"].replace(0, np.nan)) * 100
    stats["DefEff"] = (stats["TotalOppPts"] / stats["OppPoss"].replace(0, np.nan)) * 100
    stats["Tempo"] = stats["Poss"] / stats["GamesPlayed"]

    # Rate stats (possession/attempt-normalized)
    stats["ORPct"] = stats["OR"] / (stats["OR"] + stats["OppDR"]).replace(0, np.nan)
    stats["TOPct"] = stats["TO"] / stats["Poss"].replace(0, np.nan)
    stats["FTRate"] = stats["FTA"] / stats["FGA"].replace(0, np.nan)
    stats["AstRate"] = stats["Ast"] / stats["FGM"].replace(0, np.nan)
    stats["ThreePtRate"] = stats["FGA3"] / stats["FGA"].replace(0, np.nan)

    keep = ["Season", "TeamID",
            "OffEff", "DefEff", "Tempo",
            "ORPct", "TOPct", "FTRate", "AstRate", "ThreePtRate"]
    return stats[keep]


# ── Massey ordinals (men only) ──────────────────────────────────

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
    ).reset_index()
    return agg


# ── Strength of schedule ───────────────────────────────────────

def compute_sos(
    m_regular: pd.DataFrame,
    w_regular: pd.DataFrame,
    elo_curr: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    """
    Strength of schedule: mean Elo of all regular-season opponents.

    Available for both men's and women's teams. Uses current-season Elo
    so SOS reflects the quality of opponents this year.
    """
    sos = {}
    for df in [m_regular, w_regular]:
        for season in df["Season"].unique():
            sdf = df[df["Season"] == season]
            # Build opponent list per team
            opp_elos: dict[int, list[float]] = {}
            for _, row in sdf.iterrows():
                w_id, l_id = row["WTeamID"], row["LTeamID"]
                w_opp_elo = elo_curr.get((season, l_id), 1500.0)
                l_opp_elo = elo_curr.get((season, w_id), 1500.0)
                opp_elos.setdefault(w_id, []).append(w_opp_elo)
                opp_elos.setdefault(l_id, []).append(l_opp_elo)
            for tid, elos in opp_elos.items():
                sos[(season, tid)] = float(np.mean(elos))
    return sos


# ── Momentum (last-N games) ────────────────────────────────────

def compute_momentum(
    m_regular: pd.DataFrame,
    w_regular: pd.DataFrame,
    last_n: int = 10,
) -> dict[tuple[int, int], float]:
    """
    Late-season momentum: win rate over the last N regular-season games.

    Captures whether a team is hot or cold going into the tournament.
    """
    momentum = {}
    for df in [m_regular, w_regular]:
        games = df.sort_values(["Season", "DayNum"])
        for season in games["Season"].unique():
            sdf = games[games["Season"] == season]
            # Build game results per team
            team_results: dict[int, list[int]] = {}
            for _, row in sdf.iterrows():
                w_id, l_id = row["WTeamID"], row["LTeamID"]
                team_results.setdefault(w_id, []).append(1)
                team_results.setdefault(l_id, []).append(0)
            for tid, results in team_results.items():
                last = results[-last_n:]
                momentum[(season, tid)] = float(np.mean(last))
    return momentum


# ── Conference strength ───────────────────────────────────────

def compute_conference_strength(
    m_conf: pd.DataFrame,
    w_conf: pd.DataFrame,
    elo_curr: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    """
    Conference strength: mean Elo of all teams in a team's conference.

    Available for both men's and women's teams, filling the gap where
    women lack Massey ordinals.
    """
    conf_strength = {}
    for conf_df in [m_conf, w_conf]:
        for season in conf_df["Season"].unique():
            sdf = conf_df[conf_df["Season"] == season]
            # Compute mean Elo per conference
            conf_elo: dict[str, list[float]] = {}
            for _, row in sdf.iterrows():
                tid = row["TeamID"]
                abbrev = row["ConfAbbrev"]
                elo = elo_curr.get((season, tid), 1500.0)
                conf_elo.setdefault(abbrev, []).append(elo)
            conf_means = {abbrev: float(np.mean(elos)) for abbrev, elos in conf_elo.items()}
            # Assign conference mean to each team
            for _, row in sdf.iterrows():
                conf_strength[(season, row["TeamID"])] = conf_means[row["ConfAbbrev"]]
    return conf_strength


# ── Coach experience (men only) ────────────────────────────────

def compute_coach_experience(
    coaches_df: pd.DataFrame,
    m_tourney: pd.DataFrame,
) -> dict[tuple[int, int], int]:
    """
    Cumulative NCAA tournament appearances for each team's head coach.

    Men's only — uses MTeamCoaches.csv. Counts how many times the coach
    has taken a team to the tournament in prior seasons (avoids leakage).
    """
    # Find which teams made the tournament each season
    tourney_teams = set()
    for _, row in m_tourney.iterrows():
        tourney_teams.add((row["Season"], row["WTeamID"]))
        tourney_teams.add((row["Season"], row["LTeamID"]))

    # For each coach, count cumulative tournament appearances (prior seasons only)
    # coaches_df has: Season, TeamID, FirstDayNum, LastDayNum, CoachName
    # A coach active at DayNum >= 132 (tournament time) is the tournament coach
    tourney_coaches = coaches_df[coaches_df["LastDayNum"] >= 132].copy()

    coach_cum_exp: dict[str, int] = {}
    coach_exp = {}

    for season in sorted(tourney_coaches["Season"].unique()):
        sdf = tourney_coaches[tourney_coaches["Season"] == season]
        for _, row in sdf.iterrows():
            coach = row["CoachName"]
            tid = row["TeamID"]
            # Record current cumulative experience (from prior seasons)
            coach_exp[(season, tid)] = coach_cum_exp.get(coach, 0)

        # After recording, update cumulative counts for this season
        for _, row in sdf.iterrows():
            coach = row["CoachName"]
            tid = row["TeamID"]
            if (season, tid) in tourney_teams:
                coach_cum_exp[coach] = coach_cum_exp.get(coach, 0) + 1

    return coach_exp


# ── Matchup features ──────────────────────────────────────────

_STAT_DIFF_COLS = [
    "OffEff", "DefEff", "Tempo",
    "ORPct", "TOPct", "FTRate", "AstRate", "ThreePtRate",
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
        return {"MasseyMean": np.nan}
    mask = (massey_df["Season"] == season) & (massey_df["TeamID"] == team_id)
    rows = massey_df.loc[mask]
    if rows.empty:
        return {"MasseyMean": np.nan}
    row = rows.iloc[0]
    return {"MasseyMean": row["MasseyMean"]}


def build_matchup_features(
    t1: int, t2: int, season: int,
    elo_prev: dict, elo_curr: dict,
    seed_map: dict, stats_df: pd.DataFrame,
    massey_df: pd.DataFrame | None,
    elo_stats: dict | None = None,
    sos: dict | None = None,
    momentum: dict | None = None,
    conf_strength: dict | None = None,
    coach_exp: dict | None = None,
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
        "EloMOVDiff": e1_curr - e2_curr,
        "SeedT1": s1,
        "SeedT2": s2,
    }

    # Seed diff: positive means T1 has better (lower) seed
    if not (np.isnan(s1) or np.isnan(s2)):
        feats["SeedDiff"] = s2 - s1
    else:
        feats["SeedDiff"] = np.nan

    # Season stats diffs (possession-based metrics)
    st1 = _get_team_stats(t1, season, stats_df)
    st2 = _get_team_stats(t2, season, stats_df)
    for col in _STAT_DIFF_COLS:
        if col == "TOPct":
            feats[f"{col}Diff"] = st2[col] - st1[col]  # fewer TOs is better
        elif col == "DefEff":
            feats[f"{col}Diff"] = st2[col] - st1[col]  # lower DefEff is better
        else:
            feats[f"{col}Diff"] = st1[col] - st2[col]

    # Massey diff (men only, NaN for women)
    m1 = _get_massey(t1, season, massey_df)
    m2 = _get_massey(t2, season, massey_df)
    feats["MasseyMeanDiff"] = m2["MasseyMean"] - m1["MasseyMean"]  # lower rank is better

    # Strength of schedule
    if sos is not None:
        feats["SOSDiff"] = sos.get((season, t1), 1500.0) - sos.get((season, t2), 1500.0)

    # Momentum (last-10 game win rate)
    if momentum is not None:
        feats["MomentumDiff"] = momentum.get((season, t1), 0.5) - momentum.get((season, t2), 0.5)

    # Conference strength
    if conf_strength is not None:
        feats["ConfStrengthDiff"] = conf_strength.get((season, t1), 1500.0) - conf_strength.get((season, t2), 1500.0)

    # Coach tournament experience (men only)
    if coach_exp is not None:
        c1 = coach_exp.get((season, t1))
        c2 = coach_exp.get((season, t2))
        if c1 is not None and c2 is not None:
            feats["CoachExpDiff"] = c1 - c2
        else:
            feats["CoachExpDiff"] = np.nan

    # Elo trajectory stats
    if elo_stats is not None:
        _nan = {"EloTrend": np.nan, "EloStd": np.nan}
        e1s = elo_stats.get((season, t1), _nan)
        e2s = elo_stats.get((season, t2), _nan)
        feats["EloTrendDiff"] = e1s["EloTrend"] - e2s["EloTrend"]
        feats["EloStdDiff"] = e1s["EloStd"] - e2s["EloStd"]

    return feats


# ── Training data ─────────────────────────────────────────────

def build_training_data(
    m_tourney: pd.DataFrame,
    w_tourney: pd.DataFrame,
    elo_prev: dict,
    elo_curr: dict,
    seed_map: dict,
    stats_df: pd.DataFrame,
    massey_df: pd.DataFrame | None,
    min_season: int = 2003,
    elo_stats: dict | None = None,
    sos: dict | None = None,
    momentum: dict | None = None,
    conf_strength: dict | None = None,
    coach_exp: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Build (X, y, seasons, genders) from historical tournament results.

    For each game: Team1 = lower TeamID, y = 1 if Team1 won.
    Features use previous-season Elo + current-season stats/seeds/Massey.
    genders: 'M' for men's games, 'W' for women's games.
    """
    records = []
    labels = []
    szns = []
    genders = []

    for t_df, gender in [(m_tourney, "M"), (w_tourney, "W")]:
        for _, row in t_df.iterrows():
            season = row["Season"]
            if season < min_season:
                continue

            w_id, l_id = row["WTeamID"], row["LTeamID"]
            t1, t2 = min(w_id, l_id), max(w_id, l_id)

            feats = build_matchup_features(
                t1, t2, season,
                elo_prev, elo_curr, seed_map, stats_df, massey_df,
                elo_stats=elo_stats,
                sos=sos, momentum=momentum,
                conf_strength=conf_strength, coach_exp=coach_exp,
            )
            records.append(feats)
            labels.append(1 if w_id == t1 else 0)
            szns.append(season)
            genders.append(gender)

    X = pd.DataFrame(records)
    y = pd.Series(labels, name="y")
    seasons = pd.Series(szns, name="Season")
    genders = pd.Series(genders, name="Gender")
    return X, y, seasons, genders


# ── Evaluation ───────────────────────────────────────────────

def compute_sample_weights(seasons: pd.Series, decay: float = 0.60) -> np.ndarray:
    """
    Exponential recency weights: weight = decay^(max_season - season).

    decay=0.60 means 2025→1.0, 2024→0.6, 2023→0.36, 2021→0.13 (COVID bubble).
    Older seasons contribute very little; recent seasons drive the fit.
    """
    max_season = seasons.max()
    return np.array([decay ** (max_season - s) for s in seasons])


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier score = mean((y_true - y_pred)^2). Lower is better."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def leave_one_season_out_cv(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    impute: bool = False,
    sample_weight: np.ndarray | None = None,
    return_preds: bool = False,
) -> dict[int, float]:
    """
    Leave-one-season-out CV returning {season: brier_score}.

    model_factory: callable returning an unfitted sklearn-compatible model.
    impute: if True, fill NaN with training-set column medians.
    sample_weight: per-sample weights (e.g. from compute_sample_weights).
                   Only the training-fold slice is passed to fit().
    return_preds: if True, also return array of OOF predictions aligned to X.
    """
    results = {}
    oof = np.zeros(len(y))
    for season in sorted(seasons.unique()):
        train_mask = (seasons != season).values
        test_mask = (seasons == season).values

        X_train, X_test = X[train_mask].copy(), X[test_mask].copy()
        y_train, y_test = y[train_mask], y[test_mask]

        if impute:
            medians = X_train.median()
            X_train = X_train.fillna(medians)
            X_test = X_test.fillna(medians)

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight[train_mask]

        model = model_factory()
        model.fit(X_train, y_train.values, **fit_kwargs)
        proba = model.predict_proba(X_test)[:, 1]
        results[season] = brier_score(y_test.values, proba)
        oof[test_mask] = proba
        del model, X_train, X_test
        gc.collect()

    mean_b = np.mean(list(results.values()))
    print(f"  LOSO mean Brier: {mean_b:.4f}")
    if return_preds:
        return results, oof
    return results


def leave_one_season_out_cv_gendered(
    model_factory_m,
    model_factory_w,
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    genders: pd.Series,
    feature_cols_m: list[str],
    feature_cols_w: list[str],
    impute: bool = False,
    sample_weight: np.ndarray | None = None,
    return_preds: bool = False,
) -> dict[int, float]:
    """
    Gender-split LOSO CV: trains separate men's and women's models per fold.

    Returns {season: brier_score} computed over both genders combined.
    """
    results = {}
    oof = np.zeros(len(y))
    for season in sorted(seasons.unique()):
        for gender, factory, feat_cols in [("M", model_factory_m, feature_cols_m),
                                            ("W", model_factory_w, feature_cols_w)]:
            mask_g = (genders == gender).values
            train_mask = (seasons != season).values & mask_g
            test_mask = (seasons == season).values & mask_g

            if test_mask.sum() == 0:
                continue

            X_train = X.loc[train_mask, feat_cols].copy()
            X_test = X.loc[test_mask, feat_cols].copy()
            y_train, y_test = y[train_mask], y[test_mask]

            if impute:
                medians = X_train.median()
                X_train = X_train.fillna(medians)
                X_test = X_test.fillna(medians)

            fit_kwargs = {}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight[train_mask]

            model = factory()
            model.fit(X_train, y_train.values, **fit_kwargs)
            proba = model.predict_proba(X_test)[:, 1]
            oof[test_mask] = proba
            del model, X_train, X_test
            gc.collect()

        # Combined Brier for this season
        season_mask = (seasons == season).values
        if season_mask.sum() > 0:
            results[season] = brier_score(y[season_mask].values, oof[season_mask])

    mean_b = np.mean(list(results.values()))
    print(f"  LOSO mean Brier (gendered): {mean_b:.4f}")
    if return_preds:
        return results, oof
    return results


# ── Submission generation ─────────────────────────────────────

def _dicts_to_lookup_dfs(
    elo_prev: dict,
    elo_curr: dict,
    seed_map: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert lookup dicts to DataFrames for fast vectorized joins.

    elo_prev keys are (end_season, team_id); we store as PredSeason = end_season + 1
    so we can join directly on the season being predicted.
    elo_curr keys are (season, team_id); joined directly.
    seed_map keys are (season, team_id); joined directly.
    """
    elo_prev_df = pd.DataFrame(
        [{"Season": s + 1, "TeamID": tid, "EloPrev": elo}
         for (s, tid), elo in elo_prev.items()]
    )
    elo_curr_df = pd.DataFrame(
        [{"Season": s, "TeamID": tid, "EloCurr": elo}
         for (s, tid), elo in elo_curr.items()]
    )
    seed_df = pd.DataFrame(
        [{"Season": s, "TeamID": tid, "Seed": seed}
         for (s, tid), seed in seed_map.items()]
    )
    return elo_prev_df, elo_curr_df, seed_df


def _dict_to_df(d: dict, value_col: str) -> pd.DataFrame:
    """Convert {(Season, TeamID): value} dict to a DataFrame."""
    return pd.DataFrame(
        [{"Season": s, "TeamID": tid, value_col: v} for (s, tid), v in d.items()]
    )


def _join_team_features(
    base: pd.DataFrame,
    team_col: str,
    suffix: str,
    elo_prev_df: pd.DataFrame,
    elo_curr_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    massey_df: pd.DataFrame | None,
    elo_stats_df: pd.DataFrame | None = None,
    sos_df: pd.DataFrame | None = None,
    momentum_df: pd.DataFrame | None = None,
    conf_df: pd.DataFrame | None = None,
    coach_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Join all per-team features onto `base` for a given team column.
    Resulting columns are suffixed (e.g. 'EloPrev_T1', 'OffEff_T1').
    """
    df = base.copy()

    df = df.merge(
        elo_prev_df.rename(columns={"TeamID": team_col, "EloPrev": f"EloPrev_{suffix}"}),
        on=["Season", team_col], how="left",
    )
    df = df.merge(
        elo_curr_df.rename(columns={"TeamID": team_col, "EloCurr": f"EloCurr_{suffix}"}),
        on=["Season", team_col], how="left",
    )
    df = df.merge(
        seed_df.rename(columns={"TeamID": team_col, "Seed": f"Seed_{suffix}"}),
        on=["Season", team_col], how="left",
    )

    stat_cols = ["Season", "TeamID"] + _STAT_DIFF_COLS
    df = df.merge(
        stats_df[stat_cols].rename(
            columns={"TeamID": team_col, **{c: f"{c}_{suffix}" for c in _STAT_DIFF_COLS}}
        ),
        on=["Season", team_col], how="left",
    )

    if massey_df is not None:
        df = df.merge(
            massey_df.rename(
                columns={"TeamID": team_col, "MasseyMean": f"MasseyMean_{suffix}"}
            ),
            on=["Season", team_col], how="left",
        )
    else:
        df[f"MasseyMean_{suffix}"] = np.nan

    for extra_df, col_name in [
        (elo_stats_df, None),  # handled specially below
        (sos_df, "SOS"),
        (momentum_df, "Momentum"),
        (conf_df, "ConfStrength"),
        (coach_df, "CoachExp"),
    ]:
        if col_name is None:
            # Elo stats has two columns
            if extra_df is not None:
                df = df.merge(
                    extra_df.rename(
                        columns={"TeamID": team_col,
                                 "EloTrend": f"EloTrend_{suffix}",
                                 "EloStd": f"EloStd_{suffix}"}
                    ),
                    on=["Season", team_col], how="left",
                )
            else:
                df[f"EloTrend_{suffix}"] = np.nan
                df[f"EloStd_{suffix}"] = np.nan
        elif extra_df is not None:
            df = df.merge(
                extra_df.rename(
                    columns={"TeamID": team_col, col_name: f"{col_name}_{suffix}"}
                ),
                on=["Season", team_col], how="left",
            )
        else:
            df[f"{col_name}_{suffix}"] = np.nan

    return df


def build_features_vectorized(
    sample_sub: pd.DataFrame,
    elo_prev: dict,
    elo_curr: dict,
    seed_map: dict,
    stats_df: pd.DataFrame,
    massey_df: pd.DataFrame | None,
    elo_stats: dict | None = None,
    sos: dict | None = None,
    momentum: dict | None = None,
    conf_strength: dict | None = None,
    coach_exp: dict | None = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix for a submission file vectorized via joins.
    Returns a DataFrame with the same row order as sample_sub, feature columns only.
    """
    ids = sample_sub["ID"].str.split("_", expand=True)
    ids.columns = ["Season", "T1", "T2"]
    ids = ids.astype(int)

    elo_prev_df, elo_curr_df, seed_df = _dicts_to_lookup_dfs(elo_prev, elo_curr, seed_map)
    elo_stats_df = _dict_to_df_multi(elo_stats) if elo_stats else None
    sos_df = _dict_to_df(sos, "SOS") if sos else None
    momentum_df = _dict_to_df(momentum, "Momentum") if momentum else None
    conf_df = _dict_to_df(conf_strength, "ConfStrength") if conf_strength else None
    coach_df = _dict_to_df(coach_exp, "CoachExp") if coach_exp else None

    df = ids.copy()
    join_args = dict(elo_stats_df=elo_stats_df, sos_df=sos_df, momentum_df=momentum_df,
                     conf_df=conf_df, coach_df=coach_df)
    df = _join_team_features(df, "T1", "T1", elo_prev_df, elo_curr_df, seed_df, stats_df, massey_df, **join_args)
    df = _join_team_features(df, "T2", "T2", elo_prev_df, elo_curr_df, seed_df, stats_df, massey_df, **join_args)

    out = pd.DataFrame(index=df.index)
    out["EloPrevDiff"] = df["EloPrev_T1"].fillna(1500) - df["EloPrev_T2"].fillna(1500)
    out["EloMOVDiff"] = df["EloCurr_T1"].fillna(1500) - df["EloCurr_T2"].fillna(1500)
    out["SeedT1"] = df["Seed_T1"]
    out["SeedT2"] = df["Seed_T2"]
    out["SeedDiff"] = df["Seed_T2"] - df["Seed_T1"]

    for col in _STAT_DIFF_COLS:
        if col == "TOPct":
            out[f"{col}Diff"] = df[f"{col}_T2"] - df[f"{col}_T1"]
        elif col == "DefEff":
            out[f"{col}Diff"] = df[f"{col}_T2"] - df[f"{col}_T1"]
        else:
            out[f"{col}Diff"] = df[f"{col}_T1"] - df[f"{col}_T2"]

    out["MasseyMeanDiff"] = df["MasseyMean_T2"] - df["MasseyMean_T1"]

    if sos is not None:
        out["SOSDiff"] = df["SOS_T1"].fillna(1500) - df["SOS_T2"].fillna(1500)
    if momentum is not None:
        out["MomentumDiff"] = df["Momentum_T1"].fillna(0.5) - df["Momentum_T2"].fillna(0.5)
    if conf_strength is not None:
        out["ConfStrengthDiff"] = df["ConfStrength_T1"].fillna(1500) - df["ConfStrength_T2"].fillna(1500)
    if coach_exp is not None:
        out["CoachExpDiff"] = df["CoachExp_T1"] - df["CoachExp_T2"]

    if elo_stats is not None:
        out["EloTrendDiff"] = df["EloTrend_T1"] - df["EloTrend_T2"]
        out["EloStdDiff"] = df["EloStd_T1"] - df["EloStd_T2"]

    return out


def _dict_to_df_multi(elo_stats: dict) -> pd.DataFrame:
    """Convert elo_stats dict to DataFrame with EloTrend and EloStd columns."""
    return pd.DataFrame(
        [{"Season": s, "TeamID": tid, "EloTrend": v["EloTrend"], "EloStd": v["EloStd"]}
         for (s, tid), v in elo_stats.items()]
    )


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
    impute_medians: dict[str, float] | None = None,
    clip_range: tuple[float, float] = (0.01, 0.99),
    elo_stats: dict | None = None,
    sos: dict | None = None,
    momentum: dict | None = None,
    conf_strength: dict | None = None,
    coach_exp: dict | None = None,
) -> pd.DataFrame:
    """
    Generate submission by ensembling multiple models (vectorized).

    impute_medians: {col: value} applied before models that need imputation
    (i.e. TabICL). XGBoost/LightGBM/CatBoost receive NaN as-is.
    """
    print("Building features...")
    X_sub = build_features_vectorized(
        sample_sub, elo_prev, elo_curr, seed_map, stats_df, massey_df,
        elo_stats=elo_stats, sos=sos, momentum=momentum,
        conf_strength=conf_strength, coach_exp=coach_exp,
    )
    X_sub = X_sub[feature_cols]
    print(f"  Feature matrix: {X_sub.shape}")

    preds = np.zeros(len(X_sub))
    X_imputed = X_sub.fillna(impute_medians) if impute_medians else None
    for name, model in models.items():
        w = weights[name]
        if w == 0:
            print(f"  {name}: skipped (weight=0)")
            continue
        X_input = X_imputed if X_imputed is not None else X_sub
        # Batch inference for heavy models (e.g. TabICL) to avoid OOM
        batch_size = 10_000
        proba = np.empty(len(X_input))
        for start in range(0, len(X_input), batch_size):
            end = min(start + batch_size, len(X_input))
            proba[start:end] = model.predict_proba(X_input.iloc[start:end].values)[:, 1]
        preds += w * proba
        print(f"  {name}: mean={proba.mean():.4f}, std={proba.std():.4f}")

    preds = np.clip(preds, *clip_range)

    sub = sample_sub[["ID"]].copy()
    sub["Pred"] = preds
    return sub


def generate_submission_gendered(
    sample_sub: pd.DataFrame,
    models_m: dict[str, object],
    models_w: dict[str, object],
    weights_m: dict[str, float],
    weights_w: dict[str, float],
    elo_prev: dict,
    elo_curr: dict,
    seed_map: dict,
    stats_df: pd.DataFrame,
    massey_df: pd.DataFrame | None,
    feature_cols_m: list[str],
    feature_cols_w: list[str],
    impute_medians_m: dict[str, float] | None = None,
    impute_medians_w: dict[str, float] | None = None,
    clip_range: tuple[float, float] = (0.01, 0.99),
    elo_stats: dict | None = None,
    sos: dict | None = None,
    momentum: dict | None = None,
    conf_strength: dict | None = None,
    coach_exp: dict | None = None,
) -> pd.DataFrame:
    """
    Generate submission using separate men's and women's model ensembles.
    """
    print("Building features...")
    X_all = build_features_vectorized(
        sample_sub, elo_prev, elo_curr, seed_map, stats_df, massey_df,
        elo_stats=elo_stats, sos=sos, momentum=momentum,
        conf_strength=conf_strength, coach_exp=coach_exp,
    )

    # Split by gender using team IDs
    team1 = sample_sub["ID"].str.split("_").str[1].astype(int)
    is_mens = (team1 < 2000).values

    preds = np.zeros(len(X_all))

    for gender_mask, models, weights, feat_cols, impute_medians, label in [
        (is_mens, models_m, weights_m, feature_cols_m, impute_medians_m, "Men"),
        (~is_mens, models_w, weights_w, feature_cols_w, impute_medians_w, "Women"),
    ]:
        X_sub = X_all.loc[gender_mask, feat_cols]
        X_imputed = X_sub.fillna(impute_medians) if impute_medians else None
        print(f"  {label}: {X_sub.shape[0]:,} rows, {len(feat_cols)} features")

        gender_preds = np.zeros(gender_mask.sum())
        for name, model in models.items():
            w = weights[name]
            if w == 0:
                continue
            X_input = X_imputed if X_imputed is not None else X_sub
            batch_size = 10_000
            proba = np.empty(len(X_input))
            for start in range(0, len(X_input), batch_size):
                end = min(start + batch_size, len(X_input))
                proba[start:end] = model.predict_proba(X_input.iloc[start:end].values)[:, 1]
            gender_preds += w * proba
            print(f"    {name}: mean={proba.mean():.4f}, std={proba.std():.4f}")

        preds[gender_mask] = gender_preds

    preds = np.clip(preds, *clip_range)
    sub = sample_sub[["ID"]].copy()
    sub["Pred"] = preds
    return sub


# ── Visualization ─────────────────────────────────────────────

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
    """Calibration plot: predicted vs actual win probability per bin."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
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