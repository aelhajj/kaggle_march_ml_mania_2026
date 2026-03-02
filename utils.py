"""
Utilities for the hospital readmission notebook:
preprocessing, model caching, and evaluation.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats.contingency import association
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Hyperparameter cache

_PARAMS_FILE = Path("best_params.json")


def save_params(name: str, params: dict, path: Path = _PARAMS_FILE) -> None:
    """Save best hyperparameters for a named model to a JSON file."""
    data = json.loads(path.read_text()) if path.exists() else {}
    # Convert numpy scalars → Python native types for JSON serialisation
    data[name] = {k: v.item() if hasattr(v, "item") else v for k, v in params.items()}
    path.write_text(json.dumps(data, indent=2))


def load_params(name: str, path: Path = _PARAMS_FILE) -> dict | None:
    """Return saved params for *name*, or None if not found."""
    if not path.exists():
        return None
    return json.loads(path.read_text()).get(name)


# Model cache

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

    model_factory can be:
      - A class (e.g. RandomForestClassifier) — random_state is injected automatically.
      - A factory callable (e.g. lambda **p: Pipeline([...])) — caller handles random_state.

    If param_dist is None, no search is performed (useful for untuned models like LR).
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


# Preprocessing

def encode_target(df: pd.DataFrame, col: str = "readmitted") -> pd.DataFrame:
    """Map yes → 1, anything else → 0."""
    df = df.copy()
    df[col] = (df[col].str.lower() == "yes").astype(int)
    return df


def encode_binary_cols(df: pd.DataFrame, exclude: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Detect yes/no columns and encode them as 0/1. Returns (df, encoded_cols)."""
    exclude = set(exclude or [])
    yn_cols = [
        c for c in df.select_dtypes(exclude="number").columns
        if c not in exclude and set(df[c].str.lower().unique()) <= {"yes", "no"}
    ]
    df = df.copy()
    for c in yn_cols:
        df[c] = (df[c].str.lower() == "yes").astype(int)
    return df, yn_cols


def encode_age(df: pd.DataFrame, col: str = "age") -> pd.DataFrame:
    """Ordinal-encode age bracket strings like '[50-60)' into integers."""
    if col not in df.columns or df[col].dtype != object:
        return df
    df = df.copy()
    order = sorted(df[col].unique(), key=lambda x: int(x.strip("[]()").split("-")[0]))
    oe = OrdinalEncoder(categories=[order])
    df[col] = oe.fit_transform(df[[col]]).astype(int)
    return df


def onehot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all remaining object columns (preserves 'Missing' as its own category)."""
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    return pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Encode target (yes → 1)
      2. Encode binary yes/no columns
      3. Ordinal-encode age brackets
      4. One-hot encode remaining categoricals
    Returns a fully numeric DataFrame ready for modelling.
    """
    df = encode_target(df)
    df, _ = encode_binary_cols(df, exclude=["readmitted"])
    df = encode_age(df)
    df = onehot_encode(df)
    return df


# Validation

def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramér's V — normalised chi-square association between two categoricals."""
    ct = pd.crosstab(a, b)
    return float(association(ct.to_numpy(), method="cramer"))


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for the difference between two proportions."""
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def check_split(y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """Compare class positive rates between train and test using Cohen's h."""
    p_train, p_test = y_train.mean(), y_test.mean()
    h = cohens_h(p_train, p_test)
    verdict = "negligible" if abs(h) < 0.2 else ("small" if abs(h) < 0.5 else "medium+")
    rows = [
        {"split": "Train", "n": len(y_train), "positive_rate": p_train},
        {"split": "Test",  "n": len(y_test),  "positive_rate": p_test},
    ]
    df = pd.DataFrame(rows).set_index("split")
    df.attrs["cohens_h"] = round(h, 4)
    df.attrs["verdict"] = verdict
    return df


# Evaluation

def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a model on the test set. ROC-AUC is None if predict_proba is unavailable."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="binary", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def format_metrics(metrics: dict, name: str | None = None, decimals: int = 4) -> str:
    """Return a pretty string of metrics from evaluate_model."""
    lines = []
    scalar_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    width = max(len(k) for k in scalar_keys) + 1
    if name:
        lines.append(f"  {name}\n  " + "-" * (width + 12))
    for k in scalar_keys:
        v = metrics.get(k)
        s = "—" if v is None else f"{v:.{decimals}f}" if isinstance(v, float) else str(v)
        lines.append(f"  {k:>{width}} {s}")
    lines.append("")
    lines.append("  Confusion matrix:")
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        lines.append("    " + cm_df.to_string().replace("\n", "\n    "))
    return "\n".join(lines) + "\n"


def plot_confusion_matrix(y_test, y_pred, cm=None) -> None:
    """Plot the confusion matrix. Pass cm to reuse a precomputed matrix."""
    if cm is None:
        cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()
