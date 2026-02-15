"""
=============================================================================
Task B: The "Demographic Context" — Census ACS Data Extraction
=============================================================================
Project : The Reality Gap: Contrasting Official US Labor Statistics
          with Public Sentiment (2020-2026)
Course  : STAT 5243 — Spring 2026 — Columbia University
Purpose : Fetches ACS 1-Year Estimates from the U.S. Census Bureau API to
          construct a "Degree Mismatch" metric — comparing WHAT people studied
          (B15011: Field of Bachelor's Degree) vs. WHERE they actually work
          (C24030: Industry of Employment). This static snapshot reveals
          structural underemployment that the U-3 rate completely ignores.

Output  : data/df_census_degree_mismatch.csv
=============================================================================
"""

import os
import sys
import time
import json
import requests
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 1. CREDENTIALS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Load API keys from secrets.json (git-ignored for public repo safety)
SECRETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets.json")
with open(SECRETS_PATH, "r") as f:
    _secrets = json.load(f)

CENSUS_API_KEY = _secrets["CENSUS_API_KEY"]

# ACS 1-Year Detailed Tables
# B15011: Sex by Age by Field of Bachelor's Degree for First Major (25+)
# C24030: Sex by Industry for the Civilian Employed Population (16+)
TABLES = {
    "B15011": {
        "title": "Sex by Age by Field of Bachelor's Degree",
        "universe": "Population 25 years and over with a bachelor's degree",
        "purpose": "Shows WHAT people studied — the supply side of skills",
    },
    "C24030": {
        "title": "Sex by Industry for Civilian Employed Population",
        "universe": "Civilian employed population 16 years and over",
        "purpose": "Shows WHERE people actually work — the demand side",
    },
}

# Years to try (most recent first); ACS 1-Year 2023 is the most likely
# available as of Feb 2026. We fall back to 2022 if 2023 is not yet released.
YEARS_TO_TRY = [2023, 2022]

# Output configuration
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "df_census_degree_mismatch.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 2. HELPER: Fetch a full ACS table group from the Census API
# ─────────────────────────────────────────────────────────────────────────────
def fetch_acs_table(table_id, year, max_retries=3, backoff_factor=2):
    """
    Fetch ALL variables from an ACS 1-Year Detailed Table using the Census
    API 'group' endpoint. This avoids the 50-variable limit per call.

    Parameters
    ----------
    table_id     : str, e.g. "B15011"
    year         : int, e.g. 2023
    max_retries  : int
    backoff_factor : int

    Returns
    -------
    pd.DataFrame or None if all retries fail
    """
    url = (
        f"https://api.census.gov/data/{year}/acs/acs1"
        f"?get=group({table_id})"
        f"&for=us:*"
        f"&key={CENSUS_API_KEY}"
    )

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [{attempt}/{max_retries}] Fetching {table_id} "
                  f"(ACS 1-Year {year})...")
            resp = requests.get(url, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                # First row is headers, subsequent rows are data
                df = pd.DataFrame(data[1:], columns=data[0])
                print(f"  ✓ {table_id}: {df.shape[1]} variables retrieved.")
                return df

            elif resp.status_code == 204:
                print(f"  ⚠ {table_id}: Year {year} not available (HTTP 204).")
                return None

            elif resp.status_code == 429:
                wait = backoff_factor ** attempt
                print(f"  ⚠ Rate limited (HTTP 429). Waiting {wait}s...")
                time.sleep(wait)

            else:
                print(f"  ✗ {table_id}: HTTP {resp.status_code} — {resp.text[:200]}")
                if attempt < max_retries:
                    time.sleep(backoff_factor ** attempt)

        except requests.exceptions.RequestException as e:
            wait = backoff_factor ** attempt
            print(f"  ✗ {table_id} attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)

    print(f"  ✗✗ {table_id}: All attempts exhausted.")
    return None


def fetch_variable_labels(table_id, year):
    """
    Fetch the variable metadata (labels) for a Census table group.
    Returns a dict mapping variable codes → human-readable labels.
    """
    url = (
        f"https://api.census.gov/data/{year}/acs/acs1/groups/{table_id}.json"
    )
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            group_data = resp.json()
            labels = {}
            for var_id, var_info in group_data.get("variables", {}).items():
                labels[var_id] = var_info.get("label", var_id)
            return labels
    except Exception as e:
        print(f"  ⚠ Could not fetch labels for {table_id}: {e}")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROCESSING: Clean and structure Census data
# ─────────────────────────────────────────────────────────────────────────────
def process_b15011(df_raw, labels):
    """
    Process B15011 (Field of Bachelor's Degree).
    Extract Estimate columns only, map to human-readable labels,
    and create a clean summary of degree fields.
    """
    # Keep only Estimate columns (ending in 'E'), drop annotations/margins
    estimate_cols = [c for c in df_raw.columns
                     if c.startswith("B15011_") and c.endswith("E")]

    df = df_raw[estimate_cols].copy()

    # Rename columns to human-readable labels
    rename_map = {}
    for col in estimate_cols:
        if col in labels:
            # Clean the label: remove "Estimate!!" prefix
            clean = labels[col].replace("Estimate!!", "").strip()
            rename_map[col] = clean
        else:
            rename_map[col] = col
    df = df.rename(columns=rename_map)

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Transpose so fields are rows for easier reading
    df_t = df.T
    df_t.columns = ["Count"]
    df_t.index.name = "Field_of_Degree"
    df_t = df_t.reset_index()

    return df_t


def process_c24030(df_raw, labels):
    """
    Process C24030 (Sex by Industry).
    Extract Estimate columns only, map to human-readable labels,
    and create a clean summary of industry employment.
    """
    # Keep only Estimate columns
    estimate_cols = [c for c in df_raw.columns
                     if c.startswith("C24030_") and c.endswith("E")]

    df = df_raw[estimate_cols].copy()

    # Rename columns to human-readable labels
    rename_map = {}
    for col in estimate_cols:
        if col in labels:
            clean = labels[col].replace("Estimate!!", "").strip()
            rename_map[col] = clean
        else:
            rename_map[col] = col
    df = df.rename(columns=rename_map)

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Transpose
    df_t = df.T
    df_t.columns = ["Count"]
    df_t.index.name = "Industry_Category"
    df_t = df_t.reset_index()

    return df_t


def compute_degree_mismatch(df_degree, df_industry):
    """
    Create a combined "Degree Mismatch" DataFrame that puts degree fields
    and industry employment side by side for analysis.

    This is a structural comparison:
    - STEM degrees vs. employment in Professional/Scientific/Tech sectors
    - Business degrees vs. employment in Finance/Management sectors
    - Arts/Humanities degrees vs. employment in Education/Arts sectors

    The mismatch becomes apparent when degree supply exceeds sector demand.
    """
    # Tag each DataFrame with its source
    df_degree = df_degree.copy()
    df_degree["Source"] = "B15011_Degree_Field"
    df_degree = df_degree.rename(columns={"Field_of_Degree": "Category"})

    df_industry = df_industry.copy()
    df_industry["Source"] = "C24030_Industry"
    df_industry = df_industry.rename(columns={"Industry_Category": "Category"})

    # Combine vertically
    df_combined = pd.concat([df_degree, df_industry], ignore_index=True)

    return df_combined


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("TASK B — Demographic Context Extraction (Census ACS API)")
    print("=" * 72)
    print(f"Tables     : {', '.join(TABLES.keys())}")
    print(f"Output     : {OUTPUT_FILE}")
    print("-" * 72)

    data_year = None
    raw_tables = {}

    # ── Try each year until we find available data ───────────────────────
    for year in YEARS_TO_TRY:
        print(f"\n▸ Trying ACS 1-Year {year}...")
        success = True

        for table_id, info in TABLES.items():
            print(f"\n  Table: {table_id} — {info['title']}")
            print(f"  Purpose: {info['purpose']}")
            df = fetch_acs_table(table_id, year)

            if df is None:
                success = False
                print(f"  ⚠ {table_id} not available for {year}. "
                      f"Trying next year...")
                break

            raw_tables[table_id] = df
            time.sleep(1)  # Polite delay

        if success:
            data_year = year
            print(f"\n✓ Successfully fetched all tables for ACS {year}.")
            break

    if data_year is None:
        print("\n✗ ERROR: Could not fetch Census data for any year.")
        print("  Tried years:", YEARS_TO_TRY)
        sys.exit(1)

    # ── Fetch variable labels for human-readable column names ────────────
    print("\n" + "-" * 72)
    print("Fetching variable labels...")
    labels = {}
    for table_id in TABLES:
        tbl_labels = fetch_variable_labels(table_id, data_year)
        labels.update(tbl_labels)
        print(f"  ✓ {table_id}: {len(tbl_labels)} variable labels loaded.")
        time.sleep(0.5)

    # ── Process each table ───────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("Processing tables...")

    df_degree = process_b15011(raw_tables["B15011"], labels)
    print(f"  ✓ B15011 processed: {len(df_degree)} degree field categories")

    df_industry = process_c24030(raw_tables["C24030"], labels)
    print(f"  ✓ C24030 processed: {len(df_industry)} industry categories")

    # ── Compute Degree Mismatch ──────────────────────────────────────────
    print("\nComputing Degree Mismatch metric...")
    df_mismatch = compute_degree_mismatch(df_degree, df_industry)

    # Add metadata columns
    df_mismatch["ACS_Year"] = data_year
    df_mismatch["Geography"] = "United States"

    # ── Data Quality Report ──────────────────────────────────────────────
    print("\n📊 Data Quality Report:")
    print(f"  Shape          : {df_mismatch.shape[0]} rows × "
          f"{df_mismatch.shape[1]} columns")
    print(f"  ACS Year       : {data_year}")
    print(f"  Degree Fields  : {len(df_degree)} categories")
    print(f"  Industry Cats  : {len(df_industry)} categories")
    n_null = df_mismatch["Count"].isna().sum()
    print(f"  Null Counts    : {n_null} ({n_null/len(df_mismatch)*100:.1f}%)")

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_mismatch.to_csv(OUTPUT_FILE, index=False)
    file_size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"\n✅ Saved to {OUTPUT_FILE} ({file_size_kb:.1f} KB)")

    # ── Preview ──────────────────────────────────────────────────────────
    print("\n📋 Preview — Degree Fields (top 10):")
    print(df_mismatch[df_mismatch["Source"] == "B15011_Degree_Field"]
          .head(10).to_string(index=False))

    print("\n📋 Preview — Industry Employment (top 10):")
    print(df_mismatch[df_mismatch["Source"] == "C24030_Industry"]
          .head(10).to_string(index=False))

    print("\n" + "=" * 72)
    print("TASK B COMPLETE")
    print("=" * 72)

    return df_mismatch


if __name__ == "__main__":
    main()
