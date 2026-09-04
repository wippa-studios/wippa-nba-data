#!/usr/bin/env python3
"""Enhanced ML model with advanced NBA features."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import json

DATA_DIR = Path(__file__).parent / "data"
RANDOM_STATE = 42


def load_data():
    """Load advanced features dataset."""
    df = pd.read_parquet(DATA_DIR / "nba_advanced_features.parquet")
    print(f"Loaded {len(df)} games, {len(df.columns)} columns")
    return df


def prepare_features(df):
    """Prepare feature matrix and target."""
    # Drop non-feature columns
    drop_cols = [
        "date", "home_abbr", "away_abbr",
        "home_odds", "away_odds",
        "home_score", "away_score", "total_points", "point_diff", "home_win",
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


def time_based_cv(X, y, df, feature_names, n_splits=5):
    """Time-based cross-validation with proper date-based splitting."""
    unique_dates = np.sort(df["date"].unique())
    n_dates = len(unique_dates)
    fold_size = n_dates // (n_splits + 1)

    print(f"Unique dates: {n_dates}, fold_size: {fold_size}")

    results = {
        "accuracy": [], "log_loss": [], "auc": [],
        "feature_importance_gain": {}, "feature_importance_weight": {},
    }

    for fold in range(n_splits):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = min(test_start + fold_size, n_dates)

        train_dates = unique_dates[:train_end]
        test_dates = unique_dates[test_start:test_end]

        if len(test_dates) == 0:
            continue

        train_mask = df["date"].isin(train_dates).values
        test_mask = df["date"].isin(test_dates).values

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            early_stopping_rounds=30,
            verbosity=0,
        )

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

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
            # Map f0, f1, ... to actual feature names
            fname = feature_names[int(k[1:])] if k.startswith('f') and k[1:].isdigit() else k
            results["feature_importance_gain"][fname] = results["feature_importance_gain"].get(fname, 0) + v
        for k, v in weight.items():
            fname = feature_names[int(k[1:])] if k.startswith('f') and k[1:].isdigit() else k
            results["feature_importance_weight"][fname] = results["feature_importance_weight"].get(fname, 0) + v

        print(f"Fold {fold+1}: acc={acc:.4f}  logloss={ll:.4f}  auc={auc:.4f}  "
              f"train={train_mask.sum()}  test={test_mask.sum()}")

    # Average feature importance
    for k in results["feature_importance_gain"]:
        results["feature_importance_gain"][k] /= n_splits
    for k in results["feature_importance_weight"]:
        results["feature_importance_weight"][k] /= n_splits

    return results


def print_results(results):
    """Print summary and feature importance."""
    print("\n" + "=" * 65)
    print("ENHANCED MODEL - CROSS-VALIDATION RESULTS (5-fold time-series)")
    print("=" * 65)
    print(f"Accuracy:  {np.mean(results['accuracy']):.4f} ± {np.std(results['accuracy']):.4f}")
    print(f"Log Loss:  {np.mean(results['log_loss']):.4f} ± {np.std(results['log_loss']):.4f}")
    print(f"AUC:       {np.mean(results['auc']):.4f} ± {np.std(results['auc']):.4f}")

    # Feature importance
    print("\n" + "=" * 65)
    print("FEATURE IMPORTANCE (Gain - top 30)")
    print("=" * 65)
    sorted_gain = sorted(results["feature_importance_gain"].items(), key=lambda x: -x[1])
    max_gain = sorted_gain[0][1] if sorted_gain else 1
    for i, (feat, imp) in enumerate(sorted_gain[:30]):
        bar = "█" * int(30 * imp / max_gain)
        print(f"  {i+1:2d}. {feat:50s} {imp:8.1f}  {bar}")

    # Grouped importance
    print("\n" + "=" * 65)
    print("GROUPED FEATURE IMPORTANCE (by category)")
    print("=" * 65)
    groups = {
        "Rolling eFG% (shooting)": ["roll_efg_pct"],
        "Rolling TS% (scoring eff.)": ["roll_ts_pct"],
        "Rolling TOV% (turnovers)": ["roll_tov_pct", "roll_tov_per_poss"],
        "Rolling FT Rate": ["roll_ft_rate"],
        "Rolling AST Ratio": ["roll_ast_ratio"],
        "Rolling OREB%": ["roll_oreb_pct"],
        "Rolling DREB%": ["roll_dreb_pct"],
        "Rolling Pace": ["roll_pace"],
        "Rolling STL+BLK": ["roll_stl_blk_per_poss"],
        "Rolling PTS": ["roll_PTS"],
        "Rolling +/-": ["roll_PLUS_MINUS"],
        "Rolling Volatility (std)": ["roll_.*_std_"],
        "Schedule (rest/b2b)": ["rest_days", "is_b2b", "games_last"],
        "Diff Features": ["diff_"],
    }
    for group_name, patterns in groups.items():
        total = 0
        for feat, imp in results["feature_importance_gain"].items():
            if any(p in feat for p in patterns):
                total += imp
        print(f"  {group_name:40s} {total:8.1f}")

    # Save results
    out = DATA_DIR / "enhanced_results.json"
    serializable = {
        "accuracy": results["accuracy"],
        "log_loss": results["log_loss"],
        "auc": results["auc"],
        "feature_importance_gain": results["feature_importance_gain"],
    }
    out.write_text(json.dumps(serializable, indent=2))
    print(f"\nSaved: {out}")


def compare_with_baseline():
    """Compare enhanced model with baseline."""
    try:
        baseline = json.loads((DATA_DIR / "baseline_results.json").read_text())
        enhanced = json.loads((DATA_DIR / "enhanced_results.json").read_text())

        print("\n" + "=" * 65)
        print("BASELINE vs ENHANCED COMPARISON")
        print("=" * 65)
        print(f"{'Metric':<20} {'Baseline':>12} {'Enhanced':>12} {'Delta':>10}")
        print("-" * 60)
        for metric in ["accuracy", "log_loss", "auc"]:
            b = np.mean(baseline[metric])
            e = np.mean(enhanced[metric])
            delta = e - b
            sign = "+" if delta > 0 else ""
            print(f"{metric:<20} {b:>12.4f} {e:>12.4f} {sign}{delta:>9.4f}")
    except FileNotFoundError:
        print("\n(baseline results not found for comparison)")


def main():
    df = load_data()
    X, y = prepare_features(df)
    feature_names = list(X.columns)
    print(f"Features: {X.shape[1]}, Target balance: {y.mean():.3f}")

    # Convert to numpy for reliable boolean indexing
    X_np = X.values
    y_np = y.values
    results = time_based_cv(X_np, y_np, df, feature_names)
    print_results(results)
    compare_with_baseline()


if __name__ == "__main__":
    main()
