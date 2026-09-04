# wippa-nba-data

NBA betting data: game results, odds, player stats, and ML datasets spanning 2016-2026.

## Structure

```
seasons/          Season-by-season game results, odds, and scrape progress
datasets/         Aggregated ML-ready datasets
model-results/    Model evaluation outputs
scripts/          Data collection and model scripts
```

## seasons/

Each season folder contains:
- `nba_<season>_results_odds.csv` / `.parquet` — Game results with betting odds
- `scrape_progress_<season>.json` — Scrape state tracking (where available)
- `player_gamelogs_<season>.csv` — Per-player game logs (2025-26 only)

Seasons covered: 2016-17 through 2025-26

## datasets/

| File | Description |
|------|-------------|
| `nba_advanced_features.parquet` | Advanced stats feature set (11MB) |
| `nba_ml_dataset.csv` | ML training dataset |
| `nba_ml_dataset.parquet` | ML training dataset (parquet) |

## model-results/

| File | Description |
|------|-------------|
| `baseline_results.json` | Baseline model performance |
| `enhanced_results.json` | Enhanced model performance |
| `walkforward_results.json` | Walk-forward validation results |
| `holdout_results.json` | Holdout set evaluation |

## scripts/

| Script | Purpose |
|--------|---------|
| `scrape_oddsportal.py` | Scrape game results and odds from OddsPortal |
| `fetch_player_data.py` | Fetch player game log data |
| `fetch_advanced_stats.py` | Fetch advanced team/player stats |
| `build_ml_dataset.py` | Combine sources into ML-ready datasets |
| `baseline_model.py` | Baseline prediction model |
| `enhanced_model.py` | Enhanced prediction model |

## Usage

```bash
pip install pandas pyarrow scikit-learn

# Load a season
import pandas as pd
df = pd.read_csv("seasons/2024-2025/nba_2024-2025_results_odds.csv")

# Load ML dataset
ml = pd.read_parquet("datasets/nba_ml_dataset.parquet")
```

## Data Sources

- **OddsPortal** — Historical game results and betting odds
- **NBA API / Basketball Reference** — Player stats and box scores
