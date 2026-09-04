#!/usr/bin/env python3
"""Build leak-free ML dataset from scraped NBA odds data."""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
WINDOWS = [3, 5, 8, 11]


def load_and_merge() -> pd.DataFrame:
    """Load all seasons, merge, filter to regular season."""
    dfs = []
    for f in sorted(DATA_DIR.glob("nba_*_results_odds.csv")):
        df = pd.read_csv(f, parse_dates=["date"])
        season = f.stem.split("_")[1]
        df["season"] = season
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)

    # Regular season only (round is NaN/empty for regular season)
    df = df[df["round"].isna() | (df["round"] == "")].copy()
    df = df.drop(columns=["round", "is_allstar"], errors="ignore")

    print(f"Regular season games: {len(df)}")
    print(f"Seasons: {df['season'].value_counts().to_dict()}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def build_team_game_log(df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per team per game, with all game-level facts."""
    home = df[["date", "season", "home_team", "away_team",
               "home_score", "away_score", "home_odds", "away_odds",
               "overtime", "total_points", "point_diff"]].copy()
    home.columns = ["date", "season", "team", "opponent",
                    "pts_scored", "pts_allowed", "team_odds", "opp_odds",
                    "overtime", "total_points", "point_diff"]
    home["is_home"] = 1
    home["won"] = (home["pts_scored"] > home["pts_allowed"]).astype(int)

    away = df[["date", "season", "away_team", "home_team",
               "away_score", "home_score", "away_odds", "home_odds",
               "overtime", "total_points", "point_diff"]].copy()
    away.columns = ["date", "season", "team", "opponent",
                    "pts_scored", "pts_allowed", "team_odds", "opp_odds",
                    "overtime", "total_points", "point_diff"]
    away["point_diff"] = -away["point_diff"]
    away["is_home"] = 0
    away["won"] = (away["pts_scored"] > away["pts_allowed"]).astype(int)

    log = pd.concat([home, away], ignore_index=True)
    log = log.sort_values(["team", "date"]).reset_index(drop=True)
    return log


def add_team_features(log: pd.DataFrame) -> pd.DataFrame:
    """Add rolling features per team (shifted to prevent leakage)."""
    feat_cols = []

    for n in WINDOWS:
        # Rolling win %
        log[f"r{n}_win_pct"] = (
            log.groupby("team")["won"]
            .transform(lambda s: s.shift(1).rolling(n, min_periods=n).mean())
        )
        feat_cols.append(f"r{n}_win_pct")

        # Rolling pts scored
        log[f"r{n}_pts_scored"] = (
            log.groupby("team")["pts_scored"]
            .transform(lambda s: s.shift(1).rolling(n, min_periods=n).mean())
        )
        feat_cols.append(f"r{n}_pts_scored")

        # Rolling pts allowed
        log[f"r{n}_pts_allowed"] = (
            log.groupby("team")["pts_allowed"]
            .transform(lambda s: s.shift(1).rolling(n, min_periods=n).mean())
        )
        feat_cols.append(f"r{n}_pts_allowed")

        # Rolling point diff
        log[f"r{n}_point_diff"] = (
            log.groupby("team")["point_diff"]
            .transform(lambda s: s.shift(1).rolling(n, min_periods=n).mean())
        )
        feat_cols.append(f"r{n}_point_diff")

        # Rolling cover rate: was team favoured (odds < 2.0) and did they win?
        log["_fav_won"] = ((log["team_odds"] < 2.0) & (log["won"] == 1)).astype(int)
        log[f"r{n}_cover_rate"] = (
            log.groupby("team")["_fav_won"]
            .transform(lambda s: s.shift(1).rolling(n, min_periods=n).mean())
        )
        feat_cols.append(f"r{n}_cover_rate")

    # Venue-specific form: rolling home win rate (from home games only)
    mask_home = log["is_home"] == 1
    log.loc[mask_home, "_home_win_pct"] = (
        log.loc[mask_home].groupby("team")["won"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )

    # Venue-specific form: rolling away win rate (from away games only)
    mask_away = log["is_home"] == 0
    log.loc[mask_away, "_away_win_pct"] = (
        log.loc[mask_away].groupby("team")["won"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )

    # Forward-fill venue-specific form so all rows have both stats
    log["home_rolling_win_pct"] = (
        log.groupby("team")["_home_win_pct"].transform(lambda s: s.ffill())
    )
    log["away_rolling_win_pct"] = (
        log.groupby("team")["_away_win_pct"].transform(lambda s: s.ffill())
    )

    # 7-day workload: games played in last 7 days (shifted to avoid leakage)
    def count_games_last_7d(group):
        dates = group["date"].values
        counts = []
        for i, d in enumerate(dates):
            window_start = d - np.timedelta64(7, "D")
            prior = dates[:i]
            counts.append(int(((prior > window_start) & (prior < d)).sum()))
        return counts

    log["games_last_7d"] = (
        log.groupby("team", group_keys=False)
        .apply(lambda g: pd.Series(count_games_last_7d(g), index=g.index))
        .values
    )

    # Home stand / road trip streak
    def compute_streak(series):
        streaks = []
        count = 0
        prev = None
        for val in series:
            if prev is None:
                streaks.append(0)
            elif val == prev:
                count += 1
                streaks.append(count)
            else:
                count = 0
                streaks.append(0)
            prev = val
        return streaks

    log["venue_streak"] = (
        log.groupby("team", group_keys=False)["is_home"]
        .transform(lambda s: pd.Series(compute_streak(s), index=s.index))
    )

    # Clean up temp columns
    log = log.drop(columns=["_fav_won", "_home_win_pct", "_away_win_pct"], errors="ignore")

    return log, feat_cols


def assemble_final_dataset(df: pd.DataFrame, log: pd.DataFrame,
                           feat_cols: list[str]) -> pd.DataFrame:
    """Merge team features back into game-level rows."""
    extra_cols = ["home_rolling_win_pct", "away_rolling_win_pct",
                  "games_last_7d", "venue_streak"]
    team_cols = ["date", "team"] + list(dict.fromkeys(feat_cols + extra_cols))
    team_cols = [c for c in team_cols if c in log.columns]

    # Home team features
    home_log = log[log["is_home"] == 1][team_cols].copy()
    home_rename = {"team": "home_team"}
    for c in team_cols[2:]:
        home_rename[c] = f"home_{c}"
    home_log = home_log.rename(columns=home_rename)

    # Away team features
    away_log = log[log["is_home"] == 0][team_cols].copy()
    away_rename = {"team": "away_team"}
    for c in team_cols[2:]:
        away_rename[c] = f"away_{c}"
    away_log = away_log.rename(columns=away_rename)

    # Merge
    final = df.merge(home_log, on=["date", "home_team"], how="left")
    final = final.merge(away_log, on=["date", "away_team"], how="left")

    # Target
    final["home_win"] = final["home_score"] > final["away_score"]

    # Drop rows with too many NaN features (early season before enough history)
    feature_count = len([c for c in final.columns if c.startswith(("home_r", "away_r"))])
    min_features = feature_count // 2  # require at least half the features
    before = len(final)
    final = final.dropna(subset=[c for c in final.columns if c.startswith(("home_r3_win", "away_r3_win"))])
    print(f"Dropped {before - len(final)} rows with insufficient history")

    return final


def main():
    # 1. Load and merge
    df = load_and_merge()

    # 2. Build team game log
    log = build_team_game_log(df)
    print(f"Team-game log: {len(log)} rows, {log['team'].nunique()} teams")

    # 3. Add features
    log, feat_cols = add_team_features(log)

    # 4. Assemble final dataset
    final = assemble_final_dataset(df, log, feat_cols)

    # Save
    out_parquet = DATA_DIR / "nba_ml_dataset.parquet"
    out_csv = DATA_DIR / "nba_ml_dataset.csv"
    final.to_parquet(out_parquet, index=False)
    final.to_csv(out_csv, index=False)

    print(f"\nFinal dataset: {len(final)} games, {len(final.columns)} columns")
    print(f"Date range: {final['date'].min().date()} to {final['date'].max().date()}")
    print(f"Seasons: {final['season'].value_counts().to_dict()}")
    print(f"Target balance: {final['home_win'].value_counts().to_dict()}")
    print(f"\nSaved: {out_parquet}")
    print(f"       {out_csv}")

    # Feature summary
    print(f"\nFeature columns ({len(final.columns)}):")
    for c in sorted(final.columns):
        dtype = final[c].dtype
        na_pct = final[c].isna().mean() * 100
        print(f"  {c:45s} {str(dtype):10s} {na_pct:5.1f}% NaN")


if __name__ == "__main__":
    main()
