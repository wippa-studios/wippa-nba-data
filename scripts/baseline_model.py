#!/usr/bin/env python3
"""NBA ML baseline: XGBoost with feature importance analysis."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import json

DATA_DIR = Path(__file__).parent / "data"
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load ML dataset."""
    df = pd.read_parquet(DATA_DIR / "nba_ml_dataset.parquet")
    print(f"Loaded {len(df)} games, {len(df.columns)} columns")
    return df


def prepare_features(df: pd.DataFrame):
    """Prepare feature matrix and target."""
    # Drop non-feature columns
    drop_cols = [
        "date", "season", "home_team", "away_team",
        "home_score", "away_score", "home_odds", "away_odds",
        "overtime", "total_points", "point_diff", "home_win",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y = df["home_win"].astype(int)

    # Ensure all features are numeric
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Fill remaining NaN with column median
    X = X.fillna(X.median())

    return X, y


def time_based_split(X, y, df, n_splits=5):
    """Time-based train/test split on unique dates to avoid leakage."""
    unique_dates = df["date"].unique()
    unique_dates = np.sort(unique_dates)
    n_dates = len(unique_dates)
    split_size = n_dates // (n_splits + 1)

    splits = []
    for i in range(n_splits):
        train_end = split_size * (i + 1)
        test_end = train_end + split_size
        if test_end > n_dates:
            test_end = n_dates

        train_dates = unique_dates[:train_end]
        test_dates = unique_dates[train_end:test_end]

        train_idx = df[df["date"].isin(train_dates)].index
        test_idx = df[df["date"].isin(test_dates)].index

        if len(test_idx) == 0:
            continue

        splits.append((train_idx, test_idx))

    return splits


def train_and_evaluate(X, y, df):
    """Train XGBoost with time-series CV and report metrics."""
    splits = time_based_split(X, y, df, n_splits=5)

    results = {
        "accuracy": [], "log_loss": [], "auc": [],
        "feature_importance_gain": {},
        "feature_importance_weight": {},
    }

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            early_stopping_rounds=20,
            verbosity=0,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        results["accuracy"].append(acc)
        results["log_loss"].append(ll)
        results["auc"].append(auc)

        # Accumulate feature importance
        gain = model.get_booster().get_score(importance_type="gain")
        weight = model.get_booster().get_score(importance_type="weight")
        for k, v in gain.items():
            results["feature_importance_gain"][k] = results["feature_importance_gain"].get(k, 0) + v
        for k, v in weight.items():
            results["feature_importance_weight"][k] = results["feature_importance_weight"].get(k, 0) + v

        print(f"Fold {fold+1}: acc={acc:.4f}  logloss={ll:.4f}  auc={auc:.4f}  "
              f"train={len(train_idx)}  test={len(test_idx)}")

    # Average feature importance across folds
    for k in results["feature_importance_gain"]:
        results["feature_importance_gain"][k] /= len(splits)
    for k in results["feature_importance_weight"]:
        results["feature_importance_weight"][k] /= len(splits)

    return results


def print_results(results, X):
    """Print summary and feature importance."""
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS (5-fold time-series)")
    print("=" * 60)
    print(f"Accuracy:  {np.mean(results['accuracy']):.4f} ± {np.std(results['accuracy']):.4f}")
    print(f"Log Loss:  {np.mean(results['log_loss']):.4f} ± {np.std(results['log_loss']):.4f}")
    print(f"AUC:       {np.mean(results['auc']):.4f} ± {np.std(results['auc']):.4f}")

    # Feature importance by gain
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Gain - top 25)")
    print("=" * 60)
    sorted_gain = sorted(results["feature_importance_gain"].items(), key=lambda x: -x[1])
    max_gain = sorted_gain[0][1] if sorted_gain else 1
    for i, (feat, imp) in enumerate(sorted_gain[:25]):
        bar = "█" * int(30 * imp / max_gain)
        print(f"  {i+1:2d}. {feat:45s} {imp:8.1f}  {bar}")

    # Feature importance by weight
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Weight - top 25)")
    print("=" * 60)
    sorted_weight = sorted(results["feature_importance_weight"].items(), key=lambda x: -x[1])
    max_weight = sorted_weight[0][1] if sorted_weight else 1
    for i, (feat, imp) in enumerate(sorted_weight[:25]):
        bar = "█" * int(30 * imp / max_weight)
        print(f"  {i+1:2d}. {feat:45s} {imp:8.0f}  {bar}")

    # Grouped importance
    print("\n" + "=" * 60)
    print("GROUPED FEATURE IMPORTANCE")
    print("=" * 60)
    groups = {
        "Rolling Win %": [f"r{n}_win_pct" for n in [3, 5, 8, 11]],
        "Rolling Points Scored": [f"r{n}_pts_scored" for n in [3, 5, 8, 11]],
        "Rolling Points Allowed": [f"r{n}_pts_allowed" for n in [3, 5, 8, 11]],
        "Rolling Point Diff": [f"r{n}_point_diff" for n in [3, 5, 8, 11]],
        "Rolling Cover Rate": [f"r{n}_cover_rate" for n in [3, 5, 8, 11]],
        "Venue-Specific Form": ["home_rolling_win_pct", "away_rolling_win_pct"],
        "Schedule Density": ["games_last_7d"],
        "Venue Streak": ["venue_streak"],
    }
    for group_name, suffixes in groups.items():
        total = 0
        for feat, imp in results["feature_importance_gain"].items():
            if any(s in feat for s in suffixes):
                total += imp
        print(f"  {group_name:30s} {total:8.1f}")

    # Save results
    out = DATA_DIR / "baseline_results.json"
    serializable = {
        "accuracy": results["accuracy"],
        "log_loss": results["log_loss"],
        "auc": results["auc"],
        "feature_importance_gain": results["feature_importance_gain"],
        "feature_importance_weight": results["feature_importance_weight"],
    }
    out.write_text(json.dumps(serializable, indent=2))
    print(f"\nSaved: {out}")


def main():
    df = load_data()
    X, y = prepare_features(df)
    print(f"Features: {X.shape[1]}, Target balance: {y.mean():.3f}")
    results = train_and_evaluate(X, y, df)
    print_results(results, X)


if __name__ == "__main__":
    main()
