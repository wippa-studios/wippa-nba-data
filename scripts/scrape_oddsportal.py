#!/usr/bin/env python3
"""Scrape NBA 2025-2026 results and odds from OddsPortal."""

import json
import random
import re
import time
import sys
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

DEFAULT_SEASON = "2025-2026"
TOTAL_PAGES = 28
OUTPUT_DIR = Path(__file__).parent / "data"

USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
]


def parse_page(text: str) -> list[dict]:
    """Parse game results from visible page text."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    games = []
    i = 0
    current_date = None
    current_round = None

    while i < len(lines):
        date_match = re.match(r"^(\d{2} \w+ \d{4})(?:\s*-\s*(.+))?$", lines[i])
        if date_match:
            current_date = date_match.group(1)
            current_round = (date_match.group(2) or "").replace("Play Offs", "Playoffs").strip()
            i += 1
            continue

        if current_date is None:
            i += 1
            continue

        try:
            if (i + 1 < len(lines) and lines[i] == "1" and lines[i + 1] == "2"):
                if i + 10 > len(lines):
                    break
                block = lines[i:i + 10]
                i += 10
                status = block[2]
                home_score = int(block[3])
                home_team = block[4]
                away_team = block[6]
                away_score = int(block[7])
                home_odds = float(block[8])
                away_odds = float(block[9])
            elif (lines[i] in ("Finished", "After OT") and
                  i + 7 < len(lines) and
                  lines[i + 3] == "-"):
                block = lines[i:i + 8]
                i += 8
                status = block[0]
                home_score = int(block[1])
                home_team = block[2]
                away_team = block[4]
                away_score = int(block[5])
                home_odds = float(block[6])
                away_odds = float(block[7])
            else:
                i += 1
                continue
        except (ValueError, IndexError):
            i += 1
            continue

        games.append({
            "date": current_date,
            "round": current_round,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "home_odds": home_odds,
            "away_odds": away_odds,
            "overtime": status == "After OT",
        })

    return games


def load_progress(season: str) -> dict:
    """Load scrape progress from disk."""
    path = OUTPUT_DIR / f"scrape_progress_{season}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"completed_pages": [], "games": []}


def save_progress(season: str, progress: dict):
    """Save scrape progress to disk."""
    path = OUTPUT_DIR / f"scrape_progress_{season}.json"
    path.write_text(json.dumps(progress))


def scrape_page_with_retry(browser, url: str, max_retries: int = 3) -> str | None:
    """Scrape a single page with retries."""
    for attempt in range(max_retries):
        page = browser.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=45000)
            page.wait_for_timeout(4000)

            # Dismiss cookie banners if present
            try:
                cookie_btn = page.locator("button:has-text('Accept'), button:has-text('OK'), button:has-text('Got it')")
                if cookie_btn.count() > 0:
                    cookie_btn.first.click(timeout=2000)
                    page.wait_for_timeout(1000)
            except Exception:
                pass

            # Find the inner main that contains results
            text = ""
            mains = page.locator("main")
            for idx in range(mains.count()):
                t = mains.nth(idx).inner_text()
                if "Finished" in t or "After OT" in t:
                    text = t
                    break

            if not text:
                # Try waiting longer - page might still be loading
                page.wait_for_timeout(5000)
                mains = page.locator("main")
                for idx in range(mains.count()):
                    t = mains.nth(idx).inner_text()
                    if "Finished" in t or "After OT" in t:
                        text = t
                        break

            if text:
                return text

            print(f"    Attempt {attempt + 1}: no results text, retrying...", file=sys.stderr)

        except Exception as e:
            print(f"    Attempt {attempt + 1} error: {e}", file=sys.stderr)
        finally:
            page.close()

        # Backoff before retry
        wait = (attempt + 1) * 5 + random.uniform(0, 3)
        print(f"    Waiting {wait:.1f}s before retry...", file=sys.stderr)
        time.sleep(wait)

    return None


def scrape_all(season: str = DEFAULT_SEASON):
    """Scrape all pages and save to Parquet + CSV."""
    base_url = f"https://www.oddsportal.com/basketball/usa/nba-{season}/results/"
    progress = load_progress(season)
    all_games = [dict(g) for g in progress["games"]]
    completed = set(progress["completed_pages"])
    remaining = [p for p in range(1, TOTAL_PAGES + 1) if p not in completed]

    if completed:
        print(f"Resuming: {len(completed)} pages done, {len(remaining)} remaining", flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for i, page_num in enumerate(remaining):
            url = f"{base_url}#/page/{page_num}/" if page_num > 1 else base_url
            print(f"Page {page_num}/{TOTAL_PAGES} ({i + 1}/{len(remaining)} remaining): {url}", flush=True)

            text = scrape_page_with_retry(browser, url)

            if text:
                games = parse_page(text)
                print(f"  Found {len(games)} games", flush=True)
                all_games.extend(games)
                completed.add(page_num)
            else:
                print(f"  FAILED after retries, skipping page {page_num}", file=sys.stderr, flush=True)

            # Save progress after each page
            progress = {"completed_pages": sorted(completed), "games": all_games}
            save_progress(season, progress)

            # Rate limit with jitter
            if i < len(remaining) - 1:
                delay = random.uniform(3.0, 6.0)
                print(f"  Waiting {delay:.1f}s...", flush=True)
                time.sleep(delay)

        browser.close()

    if not all_games:
        print("No games found!", file=sys.stderr)
        return

    # Build DataFrame
    df = pd.DataFrame(all_games)
    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y")
    df = df.sort_values("date").drop_duplicates(subset=["date", "home_team", "away_team"]).reset_index(drop=True)

    # Add derived columns useful for ML
    df["home_win"] = df["home_score"] > df["away_score"]
    df["total_points"] = df["home_score"] + df["away_score"]
    df["point_diff"] = df["home_score"] - df["away_score"]

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUTPUT_DIR / f"nba_{season}_results_odds.parquet"
    csv_path = OUTPUT_DIR / f"nba_{season}_results_odds.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f"\nDone! {len(df)} games saved.")
    print(f"  Parquet: {parquet_path}")
    print(f"  CSV:     {csv_path}")
    print(f"\nDate range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Teams: {sorted(df['home_team'].unique())}")
    print(f"\nSample:\n{df.head(10).to_string()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape NBA results from OddsPortal")
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Season to scrape (e.g. 2024-2025)")
    args = parser.parse_args()
    scrape_all(args.season)
