"""
=============================================================================
Task A: The "Official" Baseline — FRED & BLS Data Extraction
=============================================================================
Project : The Reality Gap: Contrasting Official US Labor Statistics
          with Public Sentiment (2020-2026)
Course  : STAT 5243 — Spring 2026 — Columbia University
Purpose : Fetches monthly time-series data from the Federal Reserve Economic
          Data (FRED) API to construct the "Official Baseline" of U.S. labor
          market indicators. The series are chosen to contrast the *general*
          economy against the *youth/entry-level* economy.

Output  : data/df_official.csv
=============================================================================
"""

import os
import sys
import time
import json
import pandas as pd
from fredapi import Fred

# ─────────────────────────────────────────────────────────────────────────────
# 1. CREDENTIALS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Load API keys from secrets.json (git-ignored for public repo safety)
SECRETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets.json")
with open(SECRETS_PATH, "r") as f:
    _secrets = json.load(f)

FRED_API_KEY = _secrets["FRED_API_KEY"]

# Date range for the study
START_DATE = "2020-01-01"
END_DATE   = "2026-01-31"

# FRED Series IDs and their descriptions
# We deliberately pair "general" indicators with "youth-specific" indicators
# to quantify the divergence — the "Reality Gap."
SERIES = {
    # ── General Economy ──────────────────────────────────────────────────
    "UNRATE":       "Unemployment Rate (U-3) — The headline 'official' rate",
    "U6RATE":       "Unemployment Rate (U-6) — Includes discouraged workers "
                    "and those marginally attached or part-time for economic "
                    "reasons. This is the 'real' unemployment rate.",
    "CIVPART":      "Civilian Labor Force Participation Rate — Captures people "
                    "who have completely dropped out of the labor force",
    # ── Youth / Entry-Level Economy ──────────────────────────────────────
    "LNS14000036":  "Unemployment Rate, 20-24 years — Proxy for entry-level "
                    "job market health",
    "CGBD2024":     "Unemployment Rate, Bachelor's Degree holders 20-24 years "
                    "— The 'Degree Mismatch' proxy: even with a degree, "
                    "young workers face elevated unemployment",
}

# Output configuration
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "df_official.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 2. HELPER: Fetch a single FRED series with retry logic
# ─────────────────────────────────────────────────────────────────────────────
def fetch_series(fred_client, series_id, description, start, end,
                 max_retries=3, backoff_factor=2):
    """
    Fetch a single FRED time-series. Implements exponential backoff to
    handle API rate limits (HTTP 429) or transient network errors.

    Parameters
    ----------
    fred_client   : fredapi.Fred instance
    series_id     : str, e.g. "UNRATE"
    description   : str, human-readable description for logging
    start / end   : str, date boundaries
    max_retries   : int, number of retry attempts
    backoff_factor: int, multiplier for exponential wait

    Returns
    -------
    pd.Series with DatetimeIndex
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [{attempt}/{max_retries}] Fetching {series_id}: "
                  f"{description[:60]}...")
            data = fred_client.get_series(
                series_id,
                observation_start=start,
                observation_end=end,
            )
            print(f"  ✓ {series_id}: {len(data)} observations retrieved.")
            return data

        except Exception as e:
            wait = backoff_factor ** attempt
            print(f"  ✗ {series_id} attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ✗✗ {series_id}: All {max_retries} attempts exhausted. "
                      f"Returning empty series.")
                return pd.Series(dtype="float64")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("TASK A — Official Baseline Extraction (FRED API)")
    print("=" * 72)
    print(f"Date Range : {START_DATE} → {END_DATE}")
    print(f"Series     : {', '.join(SERIES.keys())}")
    print(f"Output     : {OUTPUT_FILE}")
    print("-" * 72)

    # Initialize FRED client
    fred = Fred(api_key=FRED_API_KEY)

    # ── Fetch all series ─────────────────────────────────────────────────
    results = {}
    for series_id, description in SERIES.items():
        data = fetch_series(fred, series_id, description, START_DATE, END_DATE)
        results[series_id] = data
        time.sleep(0.5)  # Polite delay between requests

    # ── Merge into a single DataFrame ────────────────────────────────────
    print("\n" + "-" * 72)
    print("Merging series into a single DataFrame...")

    df = pd.DataFrame(results)
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index)

    # Sort by date
    df = df.sort_index()

    # ── Data Quality Report ──────────────────────────────────────────────
    print("\n📊 Data Quality Report:")
    print(f"  Shape          : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Date Range     : {df.index.min()} → {df.index.max()}")
    print(f"  Missing Values :")
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct = n_missing / len(df) * 100
        status = "✓" if n_missing == 0 else "⚠"
        print(f"    {status} {col:<16s}: {n_missing:>3d} missing ({pct:.1f}%)")

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE)
    file_size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"\n✅ Saved to {OUTPUT_FILE} ({file_size_kb:.1f} KB)")

    # ── Preview ──────────────────────────────────────────────────────────
    print("\n📋 Preview (first 5 rows):")
    print(df.head().to_string())
    print("\n📋 Preview (last 5 rows):")
    print(df.tail().to_string())

    print("\n" + "=" * 72)
    print("TASK A COMPLETE")
    print("=" * 72)

    return df


if __name__ == "__main__":
    main()
