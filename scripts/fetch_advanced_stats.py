#!/usr/bin/env python3
"""Fetch NBA advanced stats and build sophisticated ML features."""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import LeagueGameLog
from bisect import bisect_right

DATA_DIR = Path(__file__).parent / "data"
SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def fetch_all_seasons():
    """Fetch LeagueGameLog for all seasons."""
    all_dfs = []
    for season in SEASONS:
        print(f"Fetching {season}...", end=" ", flush=True)
        lgl = LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="T"
        )
        df = lgl.get_data_frames()[0]
        df["season"] = season
        all_dfs.append(df)
        print(f"{len(df)} rows")
        time.sleep(1.5)

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal: {len(combined)} rows, {combined['TEAM_ABBREVIATION'].nunique()} teams")
    return combined


def compute_advanced_metrics(df):
    """Compute advanced stats per game."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["is_home"] = df["MATCHUP"].str.contains("vs.").astype(int)

    # Four Factors
    df["efg_pct"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"]
    df["tov_pct"] = df["TOV"] / (df["FGA"] + 0.44 * df["FTA"] + df["TOV"])
    df["ft_rate"] = df["FTA"] / df["FGA"]

    # True Shooting
    df["ts_pct"] = df["PTS"] / (2 * (df["FGA"] + 0.44 * df["FTA"]))

    # Assist Ratio
    df["ast_ratio"] = df["AST"] / (df["FGA"] + 0.44 * df["FTA"] + df["AST"] + df["TOV"])

    # Rebound rates
    df["oreb_pct"] = df["OREB"] / (df["OREB"] + df["DREB"])
    df["dreb_pct"] = df["DREB"] / (df["OREB"] + df["DREB"])

    # Pace estimate
    df["poss"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]
    df["pace"] = df["poss"] / (df["MIN"] / 48)

    # Turnovers per possession
    df["tov_per_poss"] = df["TOV"] / df["poss"]

    # Defensive activity
    df["stl_blk_per_poss"] = (df["STL"] + df["BLK"]) / df["poss"]

    return df


def build_rolling_features(df):
    """Build rolling features per team."""
    advanced_cols = [
        "efg_pct", "tov_pct", "ft_rate", "ts_pct", "ast_ratio",
        "oreb_pct", "dreb_pct", "pace", "tov_per_poss", "stl_blk_per_poss",
        "PTS", "PLUS_MINUS"
    ]
    windows = [3, 5, 8, 11]

    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

    for col in advanced_cols:
        for n in windows:
            df[f"roll_{col}_{n}"] = (
                df.groupby("TEAM_ABBREVIATION")[col]
                .transform(lambda s: s.shift(1).rolling(n, min_periods=n).mean())
            )

    # Rolling std for volatility
    for col in ["PTS", "PLUS_MINUS", "efg_pct"]:
        for n in windows:
            df[f"roll_{col}_std_{n}"] = (
                df.groupby("TEAM_ABBREVIATION")[col]
                .transform(lambda s: s.shift(1).rolling(n, min_periods=n).std())
            )

    return df


def add_schedule_features(df):
    """Add rest days, back-to-back, and schedule density features."""
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

    # Rest days
    df["prev_game_date"] = df.groupby("TEAM_ABBREVIATION")["GAME_DATE"].shift(1)
    df["rest_days"] = (df["GAME_DATE"] - df["prev_game_date"]).dt.days
    df["rest_days"] = df["rest_days"].fillna(3).clip(0, 10)

    # Back-to-back indicator
    df["is_b2b"] = (df["rest_days"] == 1).astype(int)

    # Games in last 7 days
    def games_in_window(group, days=7):
        dates = group["GAME_DATE"].values
        counts = []
        for i, d in enumerate(dates):
            prior = dates[:i]
            mask = (prior > (d - np.timedelta64(days, "D"))) & (prior < d)
            counts.append(mask.sum())
        return counts

    df["games_last_7d"] = (
        df.groupby("TEAM_ABBREVIATION", group_keys=False)
        .apply(lambda g: pd.Series(games_in_window(g, 7), index=g.index))
        .values
    )

    # Games in last 14 days
    df["games_last_14d"] = (
        df.groupby("TEAM_ABBREVIATION", group_keys=False)
        .apply(lambda g: pd.Series(games_in_window(g, 14), index=g.index))
        .values
    )

    return df


def merge_with_odds(df):
    """Merge advanced features with odds data using date + team abbreviation."""
    # Load odds
    odds_dfs = []
    for f in sorted(DATA_DIR.glob("nba_*_results_odds.csv")):
        d = pd.read_csv(f, parse_dates=["date"])
        odds_dfs.append(d)

    odds = pd.concat(odds_dfs, ignore_index=True)
    odds = odds[odds["round"].isna() | (odds["round"] == "")].copy()

    # Map team names to abbreviations
    name_to_abbr = {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
    }

    odds["home_abbr"] = odds["home_team"].map(name_to_abbr)
    odds["away_abbr"] = odds["away_team"].map(name_to_abbr)

    # Feature columns to look up
    roll_cols = [c for c in df.columns if c.startswith("roll_")]
    adv_cols = ["efg_pct", "tov_pct", "ft_rate", "ts_pct", "ast_ratio",
                "oreb_pct", "dreb_pct", "pace", "tov_per_poss", "stl_blk_per_poss"]
    schedule_cols = ["rest_days", "is_b2b", "games_last_7d", "games_last_14d"]
    feature_cols = roll_cols + adv_cols + schedule_cols

    # Build lookup: for each team, sorted list of (date, features)
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    team_lookup = {}
    for team, group in df.groupby("TEAM_ABBREVIATION"):
        dates = group["GAME_DATE"].values
        feats = group[feature_cols].values
        team_lookup[team] = (dates, feats)

    # For each game, find most recent features for home and away teams
    def lookup_features(team, game_date):
        if team not in team_lookup:
            return [np.nan] * len(feature_cols)
        dates, feats = team_lookup[team]
        # Find index of most recent game on or before game_date
        idx = bisect_right(dates, game_date) - 1
        if idx < 0:
            return [np.nan] * len(feature_cols)
        return feats[idx]

    # Look up features for all games
    print("Looking up home team features...")
    home_feats = odds.apply(
        lambda row: lookup_features(row["home_abbr"], row["date"].to_datetime64()),
        axis=1, result_type="expand"
    )
    home_feats.columns = [f"home_{c}" for c in feature_cols]

    print("Looking up away team features...")
    away_feats = odds.apply(
        lambda row: lookup_features(row["away_abbr"], row["date"].to_datetime64()),
        axis=1, result_type="expand"
    )
    away_feats.columns = [f"away_{c}" for c in feature_cols]

    # Combine
    result = pd.concat([odds[["date", "home_abbr", "away_abbr", "home_odds", "away_odds",
                              "home_win", "home_score", "away_score", "total_points",
                              "point_diff"]].reset_index(drop=True),
                        home_feats.reset_index(drop=True),
                        away_feats.reset_index(drop=True)], axis=1)

    # Compute differential features
    for col in roll_cols + adv_cols:
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        if home_col in result.columns and away_col in result.columns:
            result[f"diff_{col}"] = result[home_col] - result[away_col]

    # Schedule differentials
    for col in schedule_cols:
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        if home_col in result.columns and away_col in result.columns:
            result[f"diff_{col}"] = result[home_col] - result[away_col]

    return result


def main():
    # 1. Fetch raw data
    df = fetch_all_seasons()

    # 2. Compute advanced metrics
    df = compute_advanced_metrics(df)

    # 3. Build rolling features
    df = build_rolling_features(df)

    # 4. Add schedule features
    df = add_schedule_features(df)

    # 5. Merge with odds
    merged = merge_with_odds(df)

    print(f"\nMerged: {len(merged)} games")
    print(f"Columns: {len(merged.columns)}")

    # Drop rows with too many NaN features
    before = len(merged)
    merged = merged.dropna(subset=["home_roll_PTS_3", "away_roll_PTS_3",
                                    "home_roll_PTS_11", "away_roll_PTS_11"])
    print(f"Dropped {before - len(merged)} rows with insufficient history")
    print(f"Final: {len(merged)} games")

    # 6. Save
    out_path = DATA_DIR / "nba_advanced_features.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Print new feature summary
    new_cols = [c for c in merged.columns if c.startswith(("home_roll", "away_roll",
                                                            "diff_", "home_rest", "away_rest",
                                                            "home_is_b2b", "away_is_b2b",
                                                            "home_games", "away_games"))]
    print(f"\nNew feature columns ({len(new_cols)}):")
    for c in sorted(new_cols)[:30]:
        print(f"  {c}")
    if len(new_cols) > 30:
        print(f"  ... and {len(new_cols) - 30} more")


if __name__ == "__main__":
    main()
