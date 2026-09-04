#!/usr/bin/env python3
"""Fetch NBA player game logs and build leak-free team-level player features."""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import PlayerGameLogs, CommonAllPlayers

DATA_DIR = Path(__file__).parent / "data"
SEASONS = ["2005-06", "2006-07", "2007-08", "2008-09", "2009-10",
           "2010-11", "2011-12", "2012-13", "2013-14", "2014-15",
           "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
           "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

# Stats to track per player
PLAYER_STATS = ["MIN", "PTS", "AST", "REB", "STL", "BLK", "TOV", "PLUS_MINUS", "FG_PCT", "FGA"]
ROLLING_WINDOWS = [3, 5, 8]


def fetch_season_player_logs(season):
    """Fetch all player game logs for a season."""
    print(f"\nFetching {season} player game logs...")

    # Get all players for the season
    cap = CommonAllPlayers(is_only_current_season=0, season=season)
    players_df = cap.get_data_frames()[0]
    player_ids = players_df['PERSON_ID'].tolist()
    print(f"  {len(player_ids)} players in database")

    # Fetch game logs for each player
    all_logs = []
    fetched = 0
    for i, pid in enumerate(player_ids):
        try:
            pgl = PlayerGameLogs(player_id_nullable=pid, season_nullable=season)
            df = pgl.get_data_frames()[0]
            if len(df) > 0:
                all_logs.append(df)
                fetched += 1
        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(player_ids)} players, {fetched} with data")
        time.sleep(0.3)

    if not all_logs:
        print(f"  No data for {season}")
        return pd.DataFrame()

    combined = pd.concat(all_logs, ignore_index=True)
    print(f"  Total: {len(combined)} player-game rows from {fetched} players")
    return combined


def build_leak_free_player_features(df):
    """Build leak-free rolling features per player (shifted to prevent leakage)."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    # Rolling features (shifted by 1 — only past games)
    for stat in PLAYER_STATS:
        for n in ROLLING_WINDOWS:
            df[f"roll_{stat}_{n}"] = (
                df.groupby("PLAYER_ID")[stat]
                .transform(lambda s: s.shift(1).rolling(n, min_periods=n).mean())
            )

    # Rolling standard deviation for volatility
    for stat in ["PTS", "MIN", "PLUS_MINUS"]:
        for n in ROLLING_WINDOWS:
            df[f"roll_{stat}_std_{n}"] = (
                df.groupby("PLAYER_ID")[stat]
                .transform(lambda s: s.shift(1).rolling(n, min_periods=n).std())
            )

    # Games played in last 14 days (workload)
    def games_in_window(group, days=14):
        dates = group["GAME_DATE"].values
        counts = []
        for i, d in enumerate(dates):
            prior = dates[:i]
            mask = (prior > (d - np.timedelta64(days, "D"))) & (prior < d)
            counts.append(mask.sum())
        return counts

    df["games_last_14d"] = (
        df.groupby("PLAYER_ID", group_keys=False)
        .apply(lambda g: pd.Series(games_in_window(g, 14), index=g.index))
        .values
    )

    return df


def aggregate_to_team_game(player_df):
    """Aggregate player features to team-game level."""
    df = player_df.copy()

    # Rolling feature columns
    roll_cols = [c for c in df.columns if c.startswith("roll_")] + ["games_last_14d"]

    # For each team-game, aggregate player stats
    # Key: sum of top players' rolling stats, minutes concentration, etc.

    # Aggregate by team-game
    team_game = df.groupby(["GAME_DATE", "TEAM_ABBREVIATION", "GAME_ID"]).agg(
        # Star player stats (top player by minutes in that game)
        top_player_pts=("PTS", lambda x: x.iloc[0] if len(x) > 0 else np.nan),
        top_player_min=("MIN", lambda x: x.iloc[0] if len(x) > 0 else np.nan),

        # Team totals
        total_min=("MIN", "sum"),
        total_pts=("PTS", "sum"),
        total_ast=("AST", "sum"),
        total_reb=("REB", "sum"),
        total_stl=("STL", "sum"),
        total_blk=("BLK", "sum"),
        total_tov=("TOV", "sum"),

        # Rolling stats (mean across players)
        **{f"team_{col}": (col, "mean") for col in roll_cols},

        # Minutes concentration (top 5 players share of total minutes)
        n_players=("MIN", "count"),

        # Star player rolling stats (top player by minutes)
        star_roll_pts_5=("roll_PTS_5", lambda x: x.iloc[0] if len(x) > 0 else np.nan),
        star_roll_min_5=("roll_MIN_5", lambda x: x.iloc[0] if len(x) > 0 else np.nan),
        star_roll_pm_5=("roll_PLUS_MINUS_5", lambda x: x.iloc[0] if len(x) > 0 else np.nan),

        # Workload
        avg_games_last_14d=("games_last_14d", "mean"),
        max_games_last_14d=("games_last_14d", "max"),
    ).reset_index()

    # Minutes concentration: top 5 players' share of total minutes
    def top5_min_share(group):
        mins = group["MIN"].sort_values(ascending=False)
        return mins.head(5).sum() / mins.sum() if mins.sum() > 0 else np.nan

    min_conc = df.groupby(["GAME_DATE", "TEAM_ABBREVIATION", "GAME_ID"]).apply(
        top5_min_share
    ).reset_index()
    min_conc.columns = ["GAME_DATE", "TEAM_ABBREVIATION", "GAME_ID", "top5_min_share"]
    team_game = team_game.merge(min_conc, on=["GAME_DATE", "TEAM_ABBREVIATION", "GAME_ID"], how="left")

    return team_game


def main():
    all_seasons = []
    for season in SEASONS:
        raw = fetch_season_player_logs(season)
        if len(raw) == 0:
            continue

        # Build leak-free features
        featured = build_leak_free_player_features(raw)

        # Aggregate to team-game
        team_game = aggregate_to_team_game(featured)
        team_game["season"] = season
        all_seasons.append(team_game)

        # Save intermediate
        team_game.to_parquet(DATA_DIR / f"player_features_{season}.parquet", index=False)
        print(f"  Saved {len(team_game)} team-game rows")

    if all_seasons:
        full = pd.concat(all_seasons, ignore_index=True)
        full.to_parquet(DATA_DIR / "player_features_all.parquet", index=False)
        print(f"\nTotal: {len(full)} team-game rows across {len(all_seasons)} seasons")
        print(f"Saved: {DATA_DIR / 'player_features_all.parquet'}")


if __name__ == "__main__":
    main()
