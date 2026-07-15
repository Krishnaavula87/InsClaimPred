#!/usr/bin/env python3
"""Train and compare models for PRCP-1010 Insurance Claim Prediction.

Intern-friendly end-to-end script mirroring the notebook workflow:
load data -> preprocess -> train multiple models -> compare -> export report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "train.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"


def normalized_gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Competition-style normalized Gini coefficient (2 * AUC - 1)."""
    return 2.0 * roc_auc_score(y_true, y_score) - 1.0


def stratified_sample(df: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    """Take a stratified sample while preserving the target class ratio."""
    parts = []
    for _, group in df.groupby("target"):
        n = max(1, int(round(sample_size * len(group) / len(df))))
        n = min(n, len(group))
        parts.append(group.sample(n=n, random_state=random_state))
    return pd.concat(parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def load_data(path: Path, sample_size: int | None, random_state: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if sample_size is not None and sample_size < len(df):
        df = stratified_sample(df, sample_size, random_state)
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Replace -1 missing sentinel and split features / target."""
    work = df.copy()
    work = work.replace(-1, np.nan)
    y = work["target"].astype(int)
    drop_cols = ["id", "target"]
    # calc features are often weakly predictive; keep them for intern baseline
    feature_cols = [c for c in work.columns if c not in drop_cols]
    X = work[feature_cols]
    return X, y, feature_cols


def get_models(random_state: int) -> dict:
    models: dict = {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=12,
                        min_samples_leaf=20,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=120,
                        learning_rate=0.08,
                        max_depth=3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    if HAS_XGB:
        # scale_pos_weight helps with heavy class imbalance
        models["XGBoost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="auc",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return models


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "model": name,
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "normalized_gini": float(normalized_gini(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }
    return metrics, proba, pred


def plot_roc_curves(results: list[dict], y_test, reports_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    for item in results:
        fpr, tpr, _ = roc_curve(y_test, item["proba"])
        plt.plot(
            fpr,
            tpr,
            label=f"{item['metrics']['model']} (AUC={item['metrics']['roc_auc']:.3f})",
        )
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison — Insurance Claim Prediction")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out = reports_dir / "roc_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_confusion(y_test, pred, model_name: str, reports_dir: Path) -> None:
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    safe = model_name.lower().replace(" ", "_")
    out = reports_dir / f"confusion_{safe}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=80000,
        help="Stratified sample size for faster runs (use 0 for full dataset)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Missing dataset at {args.data}. Run: python scripts/download_data.py"
        )

    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    sample_size = None if args.sample_size == 0 else args.sample_size
    print(f"Loading data from {args.data} (sample_size={sample_size})")
    df = load_data(args.data, sample_size, args.random_state)
    print(f"Working shape: {df.shape}")
    print("Target distribution:\n", df["target"].value_counts(normalize=True))

    X, y, feature_cols = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    # Set XGBoost scale_pos_weight after split for better balance handling
    models = get_models(args.random_state)
    if HAS_XGB and "XGBoost" in models:
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        models["XGBoost"].named_steps["clf"].set_params(
            scale_pos_weight=neg / max(pos, 1)
        )

    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics, proba, pred = evaluate_model(name, model, X_test, y_test)
        print(json.dumps(metrics, indent=2))
        print(classification_report(y_test, pred, zero_division=0))
        results.append(
            {"name": name, "model": model, "metrics": metrics, "proba": proba, "pred": pred}
        )

    metrics_df = pd.DataFrame([r["metrics"] for r in results]).sort_values(
        "roc_auc", ascending=False
    )
    metrics_path = REPORTS / "model_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nModel comparison saved to {metrics_path}")
    print(metrics_df.to_string(index=False))

    best = max(results, key=lambda r: r["metrics"]["roc_auc"])
    best_name = best["name"]
    print(f"\nBest model by ROC-AUC: {best_name}")

    plot_roc_curves(results, y_test, REPORTS)
    plot_confusion(y_test, best["pred"], best_name, REPORTS)

    model_path = MODELS / "best_model.joblib"
    joblib.dump(
        {
            "model": best["model"],
            "feature_cols": feature_cols,
            "model_name": best_name,
            "metrics": best["metrics"],
        },
        model_path,
    )
    print(f"Saved best model to {model_path}")

    summary = {
        "best_model": best_name,
        "metrics": best["metrics"],
        "rows_used": int(len(df)),
        "n_features": int(len(feature_cols)),
        "positive_rate": float(y.mean()),
        "marketing_suggestions": [
            "Prioritize outreach to customers with the highest predicted claim/risk engagement scores from the model.",
            "Because positives are rare (~3.6%), use ranking/probability scores for campaigns instead of a hard 0.5 cutoff.",
            "Bundle safer product tiers or deductibles for predicted high-risk segments to improve conversion and retention.",
            "A/B test offers on top-decile scored customers and measure lift versus random targeting.",
            "Refresh the model regularly as customer mix and claim behavior drift over time.",
        ],
        "challenges": [
            "Severe class imbalance (~3.6% positive) makes accuracy misleading; ROC-AUC/Gini/PR-AUC are preferred.",
            "Missing values encoded as -1 must be handled explicitly before training.",
            "Feature names are anonymized, so business interpretation relies on feature-importance proxies.",
            "Large dataset size requires sampling or efficient model settings for quick iteration.",
        ],
    }
    summary_path = REPORTS / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
