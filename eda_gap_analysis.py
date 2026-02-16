"""
=============================================================================
EDA & Gap Analysis — The Reality Gap
=============================================================================
Project : The Reality Gap: Contrasting Official US Labor Statistics
          with Public Sentiment (2020-2026)
Course  : STAT 5243 — Spring 2026 — Columbia University

This script performs:
  1. Data Quality Audit (Reddit scraped data)
  2. Preprocessing & Feature Engineering (all 3 datasets)
  3. EDA Visualizations (8 key plots)
  4. Gap Analysis Metrics & Correlation

Outputs:
  data/df_merged_features.csv  — Aligned monthly time-series with features
  data/plots/*.png             — All visualizations
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
PLOT_DIR    = os.path.join(DATA_DIR, "plots")

OFFICIAL_CSV = os.path.join(DATA_DIR, "df_official.csv")
CENSUS_CSV   = os.path.join(DATA_DIR, "df_census_degree_mismatch.csv")
REDDIT_CSV   = os.path.join(DATA_DIR, "df_reddit_sentiment.csv")

# Plot styling — dark theme, premium feel
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "text.color":        "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "#0d1117",
})

# Color palette — modern, accessible
COLORS = {
    "u3":       "#58a6ff",  # Blue — official headline rate
    "u6":       "#f78166",  # Orange — the "real" rate
    "youth":    "#d2a8ff",  # Purple — 20-24 age group
    "degree":   "#7ee787",  # Green — bachelor's holders
    "civpart":  "#ffa657",  # Amber — participation rate
    "reddit":   "#ff7b72",  # Red — sentiment/distress
    "spread":   "#79c0ff",  # Light blue — spreads
    "accent":   "#f0883e",  # Accent
    "bg_fill":  "#1c2128",  # Fill areas
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA QUALITY AUDIT (Reddit)
# ═════════════════════════════════════════════════════════════════════════════
def audit_reddit_data(df_reddit):
    """Comprehensive data quality audit for the scraped Reddit dataset."""

    print("=" * 72)
    print("SECTION 1 — Data Quality Audit: Reddit Sentiment Data")
    print("=" * 72)

    # ── 1.1 Basic Shape & Types ──────────────────────────────────────────
    print("\n📐 Shape & Dtypes:")
    print(f"  Rows: {len(df_reddit):,}  |  Columns: {df_reddit.shape[1]}")
    print(f"  Columns: {list(df_reddit.columns)}")
    print(f"\n  Dtypes:")
    for col in df_reddit.columns:
        print(f"    {col:<15s}: {df_reddit[col].dtype}")

    # ── 1.2 Missing Values ───────────────────────────────────────────────
    print("\n🔍 Missing Values:")
    nulls = df_reddit.isnull().sum()
    for col in df_reddit.columns:
        n = nulls[col]
        pct = n / len(df_reddit) * 100
        status = "✓" if n == 0 else "⚠"
        print(f"  {status} {col:<15s}: {n:>5,} nulls ({pct:.1f}%)")

    # ── 1.3 Duplicates ───────────────────────────────────────────────────
    n_dup = df_reddit.duplicated(subset=["post_id"]).sum()
    print(f"\n🔄 Duplicates (by post_id): {n_dup:,}")
    if n_dup > 0:
        print(f"  ⚠ Removing {n_dup} duplicates...")
        df_reddit = df_reddit.drop_duplicates(subset=["post_id"]).reset_index(drop=True)

    # ── 1.4 Date Range Coverage ──────────────────────────────────────────
    print("\n📅 Date Coverage:")
    df_reddit["created_utc"] = pd.to_datetime(df_reddit["created_utc"])
    print(f"  Earliest: {df_reddit['created_utc'].min()}")
    print(f"  Latest:   {df_reddit['created_utc'].max()}")

    df_reddit["year_month"] = df_reddit["created_utc"].dt.to_period("M")
    monthly_counts = df_reddit.groupby("year_month").size()
    print(f"  Months covered: {len(monthly_counts)}")
    print(f"  Posts per month: min={monthly_counts.min()}, "
          f"max={monthly_counts.max()}, "
          f"median={monthly_counts.median():.0f}")

    # Check for gaps
    all_months = pd.period_range(
        start=df_reddit["created_utc"].min().to_period("M"),
        end=df_reddit["created_utc"].max().to_period("M"),
        freq="M"
    )
    missing_months = set(all_months) - set(monthly_counts.index)
    if missing_months:
        print(f"  ⚠ Missing months: {sorted(missing_months)}")
    else:
        print(f"  ✓ No gaps — all months covered.")

    # ── 1.5 Text Quality ─────────────────────────────────────────────────
    print("\n📝 Text Quality:")
    empty_body = (df_reddit["selftext"] == "").sum()
    print(f"  Empty selftext: {empty_body:,} ({empty_body/len(df_reddit)*100:.1f}%)")

    df_reddit["text_length"] = df_reddit["selftext"].str.len()
    non_empty = df_reddit[df_reddit["selftext"] != ""]["text_length"]
    if len(non_empty) > 0:
        print(f"  Body length (non-empty): "
              f"min={non_empty.min()}, median={non_empty.median():.0f}, "
              f"max={non_empty.max()}, mean={non_empty.mean():.0f}")

    title_lengths = df_reddit["title"].str.len()
    print(f"  Title length: min={title_lengths.min()}, "
          f"median={title_lengths.median():.0f}, max={title_lengths.max()}")

    # ── 1.6 Score Distribution ───────────────────────────────────────────
    print("\n⭐ Score Distribution:")
    scores = df_reddit["score"]
    print(f"  Min: {scores.min()}, Q1: {scores.quantile(0.25):.0f}, "
          f"Median: {scores.median():.0f}, Q3: {scores.quantile(0.75):.0f}, "
          f"Max: {scores.max()}")
    print(f"  Mean: {scores.mean():.1f}, Std: {scores.std():.1f}")

    outlier_threshold = scores.quantile(0.75) + 1.5 * (
        scores.quantile(0.75) - scores.quantile(0.25)
    )
    n_outliers = (scores > outlier_threshold).sum()
    print(f"  Score outliers (>{outlier_threshold:.0f}): {n_outliers:,}")

    # ── 1.7 Subreddit & Term Distribution ────────────────────────────────
    print("\n📊 Subreddit Distribution:")
    for sub, count in df_reddit["subreddit"].value_counts().items():
        pct = count / len(df_reddit) * 100
        print(f"  r/{sub:<20s}: {count:>5,} ({pct:.1f}%)")

    print("\n🔎 Search Term Distribution:")
    for term, count in df_reddit["search_term"].value_counts().items():
        pct = count / len(df_reddit) * 100
        print(f"  '{term}'{'':.<28s}: {count:>5,} ({pct:.1f}%)")

    print("\n" + "=" * 72)
    print("DATA QUALITY AUDIT COMPLETE")
    print("=" * 72)

    return df_reddit


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: PREPROCESSING & FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
def engineer_official_features(df_official):
    """Engineer features from FRED official data."""
    print("\n" + "=" * 72)
    print("SECTION 2A — Feature Engineering: Official Data")
    print("=" * 72)

    df = df_official.copy()

    # Interpolate the 1 missing month (Oct 2025)
    n_before = df.isnull().sum().sum()
    df = df.interpolate(method="linear")
    n_after = df.isnull().sum().sum()
    print(f"\n  Interpolated {n_before - n_after} missing values.")

    # ── Computed Spreads ─────────────────────────────────────────────────
    df["U6_U3_SPREAD"]    = df["U6RATE"] - df["UNRATE"]
    df["YOUTH_PREMIUM"]   = df["LNS14000036"] - df["UNRATE"]
    df["DEGREE_PREMIUM"]  = df["CGBD2024"] - df["UNRATE"]

    print(f"  ✓ U6_U3_SPREAD: mean={df['U6_U3_SPREAD'].mean():.2f}, "
          f"latest={df['U6_U3_SPREAD'].iloc[-1]:.2f}")
    print(f"  ✓ YOUTH_PREMIUM: mean={df['YOUTH_PREMIUM'].mean():.2f}, "
          f"latest={df['YOUTH_PREMIUM'].iloc[-1]:.2f}")
    print(f"  ✓ DEGREE_PREMIUM: mean={df['DEGREE_PREMIUM'].mean():.2f}, "
          f"latest={df['DEGREE_PREMIUM'].iloc[-1]:.2f}")

    # ── Month-over-month changes ─────────────────────────────────────────
    df["UNRATE_MOM"]   = df["UNRATE"].diff()
    df["CIVPART_MOM"]  = df["CIVPART"].diff()

    # ── 3-month rolling averages ─────────────────────────────────────────
    for col in ["UNRATE", "U6RATE", "LNS14000036", "CGBD2024", "CIVPART"]:
        df[f"{col}_3MA"] = df[col].rolling(window=3, min_periods=1).mean()

    # ── Year-over-year change ────────────────────────────────────────────
    df["UNRATE_YOY"] = df["UNRATE"].diff(12)

    print(f"  ✓ Added MoM changes, 3-month rolling averages, YoY change")
    print(f"  ✓ Final shape: {df.shape}")

    return df


def engineer_reddit_features(df_reddit):
    """
    Engineer features from Reddit sentiment data.
    
    Updates for audit compliance:
    - Normalizes post volume by subreddit subscribers
    - Filters out sparse months (N<10 posts)
    - Revises distress index to avoid circular correlation
    - Separates positive vs negative search terms
    """
    print("\n" + "=" * 72)
    print("SECTION 2B — Feature Engineering: Reddit Sentiment")
    print("=" * 72)

    # ── Load Subreddit Subscriber Data ───────────────────────────────────
    subscriber_path = os.path.join(DATA_DIR, "subreddit_subscribers.csv")
    if os.path.exists(subscriber_path):
        print(f"\n  Loading subscriber data from {subscriber_path}...")
        subscribers_df = pd.read_csv(subscriber_path)
        # Create a dictionary for easy lookup
        subscriber_map = dict(zip(
            subscribers_df["subreddit"],
            subscribers_df["subscribers_current"]
        ))
        print(f"  ✓ Loaded subscriber counts for {len(subscriber_map)} subreddits")
    else:
        print(f"  ⚠ Subscriber data not found. Post normalization will be skipped.")
        subscriber_map = {}

    # ── VADER Sentiment Analysis ─────────────────────────────────────────
    print("\n  Running VADER sentiment analysis on titles + selftext...")
    analyzer = SentimentIntensityAnalyzer()

    # Combine title and selftext for richer sentiment signal
    texts = (df_reddit["title"].fillna("") + " " +
             df_reddit["selftext"].fillna(""))

    sentiments = []
    for i, text in enumerate(texts):
        if i % 500 == 0:
            print(f"    Progress: {i:,}/{len(texts):,} posts scored...")
        scores = analyzer.polarity_scores(str(text)[:5000])  # Cap length
        sentiments.append(scores)

    sent_df = pd.DataFrame(sentiments)
    df_reddit["vader_neg"]      = sent_df["neg"].values
    df_reddit["vader_neu"]      = sent_df["neu"].values
    df_reddit["vader_pos"]      = sent_df["pos"].values
    df_reddit["vader_compound"] = sent_df["compound"].values
    print(f"  ✓ VADER scores computed for {len(texts):,} posts.")
    print(f"    Mean compound: {df_reddit['vader_compound'].mean():.3f}")
    print(f"    Median compound: {df_reddit['vader_compound'].median():.3f}")

    # ── Monthly Aggregation ──────────────────────────────────────────────
    print("\n  Aggregating to monthly time-series...")
    df_reddit["year_month"] = df_reddit["created_utc"].dt.to_period("M")

    # Aggregate overall metrics
    monthly = df_reddit.groupby("year_month").agg(
        post_count=("post_id", "count"),
        avg_score=("score", "mean"),
        median_score=("score", "median"),
        total_score=("score", "sum"),
        avg_sentiment=("vader_compound", "mean"),
        median_sentiment=("vader_compound", "median"),
        pct_negative=("vader_compound", lambda x: (x < -0.05).mean()),
        pct_positive=("vader_compound", lambda x: (x > 0.05).mean()),
        unique_subreddits=("subreddit", "nunique"),
        avg_text_length=("text_length", "mean"),
    ).reset_index()

    # Convert period to datetime for merging
    monthly["Date"] = monthly["year_month"].dt.to_timestamp()

    # ── Separate Positive vs Negative Term Aggregation ──────────────────
    # Track posts by term category if available
    if "term_category" in df_reddit.columns:
        print("\n  Separating positive vs negative search terms...")
        term_monthly = df_reddit.groupby(["year_month", "term_category"]).agg(
            post_count_by_category=("post_id", "count"),
            avg_sentiment_by_category=("vader_compound", "mean"),
        ).reset_index()
        
        # Pivot to get separate columns for positive and negative
        term_pivot = term_monthly.pivot(
            index="year_month",
            columns="term_category",
            values=["post_count_by_category", "avg_sentiment_by_category"]
        ).reset_index()
        
        # Flatten column names
        term_pivot.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0]
            for col in term_pivot.columns
        ]
        
        # Merge with monthly data
        monthly = monthly.merge(term_pivot, on="year_month", how="left")
        monthly.fillna(0, inplace=True)  # Fill missing categories with 0
        
        print(f"  ✓ Separated positive and negative term metrics")
    
    # ── Normalize by Subreddit Subscribers ───────────────────────────────
    if subscriber_map:
        print("\n  Normalizing post volume by subreddit subscribers...")
        
        # For each month, calculate average subscribers across active subreddits
        # This is approximate since we only have current subscriber counts
        subreddit_monthly = df_reddit.groupby(["year_month", "subreddit"]).agg(
            posts_in_subreddit=("post_id", "count")
        ).reset_index()
        
        # Map subscriber counts
        subreddit_monthly["subscribers"] = subreddit_monthly["subreddit"].map(
            subscriber_map
        )
        
        # Calculate normalized post rate per 10k subscribers per subreddit
        subreddit_monthly["posts_per_10k_subs"] = (
            subreddit_monthly["posts_in_subreddit"] / 
            subreddit_monthly["subscribers"].replace(0, np.nan) * 10000
        )
        
        # Aggregate to monthly level
        normalized_monthly = subreddit_monthly.groupby("year_month").agg(
            total_subscribers=("subscribers", "sum"),
            avg_posts_per_10k_subs=("posts_per_10k_subs", "mean"),
        ).reset_index()
        
        # Merge with main monthly data
        monthly = monthly.merge(normalized_monthly, on="year_month", how="left")
        
        # Also calculate overall normalized volume
        monthly["post_volume_normalized"] = (
            monthly["post_count"] / 
            monthly["total_subscribers"].replace(0, np.nan) * 10000
        )
        
        print(f"  ✓ Post volume normalized by subscriber counts")
    else:
        monthly["post_volume_normalized"] = np.nan
        monthly["avg_posts_per_10k_subs"] = np.nan

    # ── Filter Sparse Data (N<10 posts per month) ────────────────────────
    print(f"\n  Filtering sparse months (N<10 posts)...")
    print(f"    Before filtering: {len(monthly)} months")
    
    # Mark sparse months but keep them in dataset with a flag
    monthly["is_sparse"] = monthly["post_count"] < 10
    sparse_count = monthly["is_sparse"].sum()
    print(f"    Sparse months (N<10): {sparse_count}")
    print(f"    Reliable months (N≥10): {len(monthly) - sparse_count}")
    
    # For plotting and correlation analysis, we'll filter these out
    # but keep them in the data with the flag

    # ── Revised Distress Index ───────────────────────────────────────────
    # New formula: Uses standardized normalized volume, standardized sentiment,
    # and diversity factor (unique subreddits as proxy for breadth)
    # This avoids direct correlation with raw post_count
    
    print("\n  Computing revised distress index...")
    
    # Standardize components (z-scores)
    # Only use non-sparse data for standardization
    reliable = monthly[~monthly["is_sparse"]].copy()
    
    if len(reliable) > 0:
        # Use normalized volume if available, otherwise raw count
        volume_col = ("post_volume_normalized" if "post_volume_normalized" in monthly.columns 
                     else "post_count")
        
        # Calculate z-scores on reliable data
        volume_mean = reliable[volume_col].mean()
        volume_std = reliable[volume_col].std()
        sentiment_mean = reliable["avg_sentiment"].mean()
        sentiment_std = reliable["avg_sentiment"].std()
        diversity_mean = reliable["unique_subreddits"].mean()
        diversity_std = reliable["unique_subreddits"].std()
        
        # Apply standardization to all data
        monthly["volume_zscore"] = (
            (monthly[volume_col] - volume_mean) / 
            (volume_std if volume_std > 0 else 1)
        )
        monthly["sentiment_zscore"] = (
            (monthly["avg_sentiment"] - sentiment_mean) / 
            (sentiment_std if sentiment_std > 0 else 1)
        )
        monthly["diversity_zscore"] = (
            (monthly["unique_subreddits"] - diversity_mean) / 
            (diversity_std if diversity_std > 0 else 1)
        )
        
        # Distress index formula:
        # Higher volume (positive z) + Lower sentiment (negative z) + Higher diversity (positive z)
        # We invert sentiment so negative sentiment contributes positively to distress
        monthly["distress_index_composite"] = (
            monthly["volume_zscore"] + 
            (-1 * monthly["sentiment_zscore"]) +  # Invert: negative sentiment = more distress
            (0.5 * monthly["diversity_zscore"])   # Weight diversity less
        )
        
        # Normalize to 0-100 scale for interpretability
        if monthly["distress_index_composite"].max() > monthly["distress_index_composite"].min():
            monthly["distress_index_norm"] = (
                (monthly["distress_index_composite"] - monthly["distress_index_composite"].min()) /
                (monthly["distress_index_composite"].max() - monthly["distress_index_composite"].min()) * 100
            )
        else:
            monthly["distress_index_norm"] = 0
            
        print(f"  ✓ Revised distress index computed using:")
        print(f"    - Standardized {'normalized ' if volume_col == 'post_volume_normalized' else ''}volume")
        print(f"    - Standardized (inverted) sentiment")
        print(f"    - Standardized diversity (unique subreddits)")
    else:
        monthly["distress_index_composite"] = 0
        monthly["distress_index_norm"] = 0
        print(f"  ⚠ Not enough reliable data to compute distress index")

    print(f"\n  ✓ Monthly aggregation complete: {len(monthly)} months")
    print(f"    - {(~monthly['is_sparse']).sum()} reliable months (N≥10)")
    print(f"    - {monthly['is_sparse'].sum()} sparse months (N<10, flagged)")

    return df_reddit, monthly


def process_census_data(df_census):
    """Process Census data for degree-mismatch visualization."""
    print("\n" + "=" * 72)
    print("SECTION 2C — Feature Engineering: Census Degree Mismatch")
    print("=" * 72)

    # Split into degree fields and industry
    df_degree = df_census[df_census["Source"] == "B15011_Degree_Field"].copy()
    df_industry = df_census[df_census["Source"] == "C24030_Industry"].copy()

    print(f"  Degree categories: {len(df_degree)}")
    print(f"  Industry categories: {len(df_industry)}")

    return df_degree, df_industry


def merge_datasets(df_official, monthly_reddit):
    """Merge official and Reddit datasets on monthly timestamps."""
    print("\n" + "=" * 72)
    print("SECTION 2D — Merging Datasets")
    print("=" * 72)

    # Reddit monthly has Date column, official has Date as index
    df_merged = pd.merge(
        df_official.reset_index(),
        monthly_reddit,
        on="Date",
        how="outer",
    )
    df_merged = df_merged.sort_values("Date").reset_index(drop=True)

    print(f"  ✓ Merged shape: {df_merged.shape}")
    print(f"  ✓ Date range: {df_merged['Date'].min()} → {df_merged['Date'].max()}")

    # Fill NaN post counts with 0 for months before Reddit data
    for col in ["post_count", "avg_sentiment", "distress_index_norm"]:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)

    return df_merged


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: EDA VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════
def plot_1_unemployment_rates(df_official, save_dir):
    """Plot 1: Multi-line comparison of all unemployment rates over time."""
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df_official.index, df_official["UNRATE"],
            color=COLORS["u3"], linewidth=2.5, label="U-3 (Official Rate)",
            zorder=5)
    ax.plot(df_official.index, df_official["U6RATE"],
            color=COLORS["u6"], linewidth=2, label="U-6 (Real Rate)",
            linestyle="-", alpha=0.9)
    ax.plot(df_official.index, df_official["LNS14000036"],
            color=COLORS["youth"], linewidth=2, label="Ages 20-24",
            linestyle="-", alpha=0.9)
    ax.plot(df_official.index, df_official["CGBD2024"],
            color=COLORS["degree"], linewidth=2,
            label="Bachelor's 20-24", linestyle="-", alpha=0.9)

    # Shade COVID period
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-01"),
               alpha=0.15, color="#ff7b72", label="COVID Shock")
    # Shade tech layoff period
    ax.axvspan(pd.Timestamp("2022-10-01"), pd.Timestamp("2023-06-01"),
               alpha=0.1, color="#ffa657", label="Tech Layoff Wave")

    ax.set_title("The Hidden Divergence: Official vs. Youth Unemployment",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.savefig(os.path.join(save_dir, "01_unemployment_rates.png"))
    plt.close()
    print("  ✓ Plot 1: Unemployment rates comparison saved.")


def plot_2_u6_u3_spread(df_official, save_dir):
    """Plot 2: Area chart showing U6-U3 spread over time."""
    fig, ax = plt.subplots(figsize=(14, 6))

    spread = df_official["U6_U3_SPREAD"]
    ax.fill_between(df_official.index, 0, spread,
                    alpha=0.4, color=COLORS["spread"],
                    label="U-6 minus U-3 Spread")
    ax.plot(df_official.index, spread,
            color=COLORS["spread"], linewidth=2)

    # Reference line at mean
    mean_spread = spread.mean()
    ax.axhline(y=mean_spread, color=COLORS["accent"],
               linestyle="--", alpha=0.7,
               label=f"Mean Spread ({mean_spread:.1f}pp)")

    ax.set_title("The 'Hidden' Unemployed: U-6 minus U-3 Spread",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Spread (percentage points)")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.savefig(os.path.join(save_dir, "02_u6_u3_spread.png"))
    plt.close()
    print("  ✓ Plot 2: U6-U3 spread saved.")


def plot_3_reality_gap(df_merged, save_dir):
    """
    Plot 3: Dual-axis — Official UNRATE vs Reddit distress volume.
    
    Updated to show both raw and normalized post volume, and filter sparse months.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    dates = pd.to_datetime(df_merged["Date"])
    
    # Filter out sparse months for reliable visualization
    # But show them with lighter transparency
    reliable_mask = ~df_merged.get("is_sparse", pd.Series([False]*len(df_merged)))
    sparse_mask = df_merged.get("is_sparse", pd.Series([False]*len(df_merged)))
    
    # ── Top Panel: Raw Post Volume ──────────────────────────────────────
    # Reliable months (N>=10)
    reliable_dates = dates[reliable_mask]
    if len(reliable_dates) > 0:
        ax1.bar(reliable_dates, df_merged.loc[reliable_mask, "post_count"],
                width=25, alpha=0.7, color=COLORS["reddit"],
                label="Reddit Posts (N≥10)", zorder=3)
    
    # Sparse months (N<10) - shown with lighter color
    sparse_dates = dates[sparse_mask]
    if len(sparse_dates) > 0:
        ax1.bar(sparse_dates, df_merged.loc[sparse_mask, "post_count"],
                width=25, alpha=0.3, color=COLORS["reddit"],
                label="Reddit Posts (N<10, excluded from analysis)", zorder=2)
    
    ax1.set_ylabel("Reddit Posts per Month (Raw)", color=COLORS["reddit"])
    ax1.tick_params(axis="y", labelcolor=COLORS["reddit"])
    
    # Unemployment rates on second y-axis
    ax1_right = ax1.twinx()
    ax1_right.plot(dates, df_merged["UNRATE"],
                   color=COLORS["u3"], linewidth=2.5, label="U-3 Rate (%)",
                   zorder=5)
    ax1_right.plot(dates, df_merged["U6RATE"],
                   color=COLORS["u6"], linewidth=2, label="U-6 Rate (%)",
                   alpha=0.8, zorder=4)
    ax1_right.set_ylabel("Unemployment Rate (%)", color=COLORS["u3"])
    ax1_right.tick_params(axis="y", labelcolor=COLORS["u3"])
    
    ax1.set_title("THE REALITY GAP: Raw Volume vs Official Rates",
                  fontsize=14, fontweight="bold", pad=10)
    
    # Combined legend for top panel
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_right.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ── Bottom Panel: Normalized Post Volume ────────────────────────────
    if "post_volume_normalized" in df_merged.columns and df_merged["post_volume_normalized"].notna().any():
        # Reliable months
        if len(reliable_dates) > 0:
            ax2.bar(reliable_dates, 
                    df_merged.loc[reliable_mask, "post_volume_normalized"],
                    width=25, alpha=0.7, color=COLORS["reddit"],
                    label="Normalized Posts/10k Subs (N≥10)", zorder=3)
        
        # Sparse months
        if len(sparse_dates) > 0:
            ax2.bar(sparse_dates,
                    df_merged.loc[sparse_mask, "post_volume_normalized"],
                    width=25, alpha=0.3, color=COLORS["reddit"],
                    label="Normalized Posts/10k Subs (N<10, excluded)", zorder=2)
        
        ax2.set_ylabel("Posts per 10k Subscribers", color=COLORS["reddit"])
        ax2.tick_params(axis="y", labelcolor=COLORS["reddit"])
        
        # Unemployment rates on second y-axis
        ax2_right = ax2.twinx()
        ax2_right.plot(dates, df_merged["UNRATE"],
                       color=COLORS["u3"], linewidth=2.5, label="U-3 Rate (%)",
                       zorder=5)
        ax2_right.plot(dates, df_merged["U6RATE"],
                       color=COLORS["u6"], linewidth=2, label="U-6 Rate (%)",
                       alpha=0.8, zorder=4)
        ax2_right.set_ylabel("Unemployment Rate (%)", color=COLORS["u3"])
        ax2_right.tick_params(axis="y", labelcolor=COLORS["u3"])
        
        ax2.set_title("THE REALITY GAP: Normalized Volume (Controlling for Subreddit Growth)",
                      fontsize=14, fontweight="bold", pad=10)
        
        # Combined legend for bottom panel
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_right.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "Normalized volume not available\n(requires subreddit subscriber data)",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=12, color="#8b949e")
        ax2.set_ylabel("Posts per 10k Subscribers", color=COLORS["reddit"])
    
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "03_reality_gap.png"))
    plt.close()
    print("  ✓ Plot 3: Reality Gap (raw + normalized) saved.")



def plot_4_heatmap(df_reddit, save_dir):
    """Plot 4: Heatmap of posts per subreddit × month."""
    fig, ax = plt.subplots(figsize=(16, 6))

    # Create pivot table
    df_reddit["ym_str"] = df_reddit["created_utc"].dt.strftime("%Y-%m")
    pivot = df_reddit.groupby(["subreddit", "ym_str"]).size().unstack(fill_value=0)

    # Sort chronologically
    pivot = pivot.reindex(columns=sorted(pivot.columns))

    # Show every 3rd month label for readability
    cols = pivot.columns.tolist()
    x_labels = [c if i % 3 == 0 else "" for i, c in enumerate(cols)]

    sns.heatmap(pivot, ax=ax, cmap="YlOrRd",
                linewidths=0.3, linecolor="#30363d",
                xticklabels=x_labels,
                cbar_kws={"label": "Post Count"})

    ax.set_title("Reddit Distress Activity: Subreddit × Month",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Month")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right", fontsize=8)

    plt.savefig(os.path.join(save_dir, "04_heatmap.png"))
    plt.close()
    print("  ✓ Plot 4: Heatmap saved.")


def plot_5_search_terms(df_reddit, save_dir):
    """Plot 5: Horizontal bar chart of search term frequency."""
    fig, ax = plt.subplots(figsize=(10, 7))

    term_counts = df_reddit["search_term"].value_counts().sort_values()

    bars = ax.barh(term_counts.index, term_counts.values,
                   color=[COLORS["reddit"]] * len(term_counts), alpha=0.8,
                   edgecolor="#30363d")

    # Add value labels
    for bar, val in zip(bars, term_counts.values):
        ax.text(val + 5, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", fontsize=10, color="#c9d1d9")

    ax.set_title("Distress Signal Frequency: Which Keywords Dominate?",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Posts")
    ax.grid(axis="x", alpha=0.3)

    plt.savefig(os.path.join(save_dir, "05_search_terms.png"))
    plt.close()
    print("  ✓ Plot 5: Search term frequency saved.")


def plot_6_sentiment_timeseries(monthly, save_dir):
    """
    Plot 6: Monthly average VADER sentiment over time.
    
    Updated to filter sparse months and show confidence intervals.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                     gridspec_kw={"height_ratios": [2, 1]})

    dates = monthly["Date"]
    
    # Identify sparse vs reliable months
    is_sparse = monthly.get("is_sparse", pd.Series([False]*len(monthly)))
    reliable_mask = ~is_sparse
    
    # Top: Sentiment compound score with confidence estimation
    # For reliable months, plot with confidence band
    if reliable_mask.any():
        reliable_dates = dates[reliable_mask]
        reliable_sentiment = monthly.loc[reliable_mask, "avg_sentiment"]
        
        ax1.plot(reliable_dates, reliable_sentiment,
                 color=COLORS["u3"], linewidth=2, label="Avg Compound (N≥10)",
                 marker='o', markersize=4)
        
        # Add shaded region for negative/positive territory
        ax1.fill_between(reliable_dates, 0, reliable_sentiment,
                         where=reliable_sentiment < 0,
                         alpha=0.3, color=COLORS["reddit"],
                         label="Negative Territory")
        ax1.fill_between(reliable_dates, 0, reliable_sentiment,
                         where=reliable_sentiment >= 0,
                         alpha=0.3, color=COLORS["degree"],
                         label="Positive Territory")
    
    # For sparse months, plot with lighter transparency
    if is_sparse.any():
        sparse_dates = dates[is_sparse]
        sparse_sentiment = monthly.loc[is_sparse, "avg_sentiment"]
        ax1.plot(sparse_dates, sparse_sentiment,
                 color=COLORS["u3"], linewidth=1, alpha=0.3,
                 label="Avg Compound (N<10, unreliable)", 
                 linestyle='--', marker='x', markersize=4)
    
    ax1.axhline(y=0, color="#8b949e", linestyle="-", alpha=0.5)
    ax1.set_title("Monthly Sentiment Trajectory (VADER Compound Score)\n" +
                  "Sparse months (N<10) shown with reduced opacity",
                  fontsize=14, fontweight="bold", pad=15)
    ax1.set_ylabel("Avg VADER Compound")
    ax1.legend(loc="lower left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom: % negative posts (bar chart with sparse indication)
    # Reliable months
    if reliable_mask.any():
        ax2.bar(dates[reliable_mask], 
                monthly.loc[reliable_mask, "pct_negative"] * 100,
                width=25, color=COLORS["reddit"], alpha=0.7,
                label="% Negative (N≥10)")
    
    # Sparse months
    if is_sparse.any():
        ax2.bar(dates[is_sparse],
                monthly.loc[is_sparse, "pct_negative"] * 100,
                width=25, color=COLORS["reddit"], alpha=0.3,
                label="% Negative (N<10, unreliable)")
    
    ax2.set_ylabel("% Negative Posts")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "06_sentiment_timeseries.png"))
    plt.close()
    print("  ✓ Plot 6: Sentiment time-series (with sparse month filtering) saved.")


def plot_7_correlation_scatter(df_merged, save_dir):
    """
    Plot 7: Scatter plots — time-colored U-3 vs volume, and sentiment vs U-3.
    
    Updated to show both raw and normalized volume, and filter sparse months.
    """
    # Determine if we have normalized volume
    has_normalized = ("post_volume_normalized" in df_merged.columns and 
                     df_merged["post_volume_normalized"].notna().any())
    
    # Create figure with appropriate number of subplots
    n_cols = 3 if has_normalized else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(7*n_cols, 7))
    if n_cols == 2:
        axes = np.array([axes[0], axes[1], None])  # Pad for consistent indexing

    # Filter to reliable months (N>=10) for correlation analysis
    reliable_mask = ~df_merged.get("is_sparse", pd.Series([False]*len(df_merged)))
    mask = (df_merged["post_count"] > 0) & reliable_mask
    data = df_merged[mask].copy()

    if len(data) < 5:
        print("  ⚠ Not enough reliable data for correlation scatter.")
        plt.close()
        return

    # ── Left Panel: Time-colored U-3 vs Raw Post Volume ─────────────────
    ax = axes[0]
    dates_numeric = mdates.date2num(pd.to_datetime(data["Date"]))
    scatter = ax.scatter(
        data["UNRATE"], data["post_count"],
        c=dates_numeric, cmap="cool", alpha=0.75, s=60,
        edgecolors="#30363d", linewidths=0.5, zorder=5,
    )
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.ax.set_ylabel("Date", fontsize=9)
    # Format colorbar ticks as years
    tick_locs = cbar.get_ticks()
    cbar.set_ticklabels([
        mdates.num2date(t).strftime("%Y") if t > 0 else ""
        for t in tick_locs
    ])

    # Annotate regimes
    if data["UNRATE"].max() > 8:  # COVID spike exists
        ax.annotate("COVID spike\n(high U-3, low posts)",
                    xy=(10, 5), fontsize=8, color="#8b949e",
                    fontstyle="italic", ha="center")
    if data["post_count"].max() > 50:  # High distress period exists
        ax.annotate("Post-2023\n(low U-3, high distress)",
                    xy=(4.0, data["post_count"].max() * 0.8), fontsize=8, 
                    color="#ffa657", fontstyle="italic", ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#1c2128",
                              edgecolor="#ffa657", alpha=0.8))

    corr = data["UNRATE"].corr(data["post_count"])
    ax.set_title(f"U-3 vs Raw Distress Volume\n(r = {corr:.3f}, N≥10 only)\nColor = Time →",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("U-3 Rate (%)")
    ax.set_ylabel("Reddit Posts / Month (Raw)")
    ax.grid(True)

    # ── Middle Panel: U-3 vs Normalized Volume (if available) ────────────
    if has_normalized:
        ax = axes[1]
        scatter_norm = ax.scatter(
            data["UNRATE"], data["post_volume_normalized"],
            c=dates_numeric, cmap="cool", alpha=0.75, s=60,
            edgecolors="#30363d", linewidths=0.5, zorder=5,
        )
        cbar_norm = plt.colorbar(scatter_norm, ax=ax, pad=0.02)
        cbar_norm.ax.set_ylabel("Date", fontsize=9)
        tick_locs_norm = cbar_norm.get_ticks()
        cbar_norm.set_ticklabels([
            mdates.num2date(t).strftime("%Y") if t > 0 else ""
            for t in tick_locs_norm
        ])

        corr_norm = data["UNRATE"].corr(data["post_volume_normalized"])
        ax.set_title(f"U-3 vs Normalized Volume\n(r = {corr_norm:.3f}, N≥10 only)\n" + 
                    "Controls for Subreddit Growth",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("U-3 Rate (%)")
        ax.set_ylabel("Posts per 10k Subscribers")
        ax.grid(True)

    # ── Right Panel: UNRATE vs Avg Sentiment ─────────────────────────────
    ax = axes[2] if has_normalized else axes[1]
    sent_data = data.dropna(subset=["avg_sentiment", "UNRATE"])

    scatter2 = ax.scatter(
        sent_data["UNRATE"], sent_data["avg_sentiment"],
        c=sent_data["pct_negative"], cmap="RdYlGn_r", alpha=0.75,
        s=60, edgecolors="#30363d", linewidths=0.5, zorder=5,
    )
    cbar2 = plt.colorbar(scatter2, ax=ax, pad=0.02)
    cbar2.ax.set_ylabel("% Negative Posts", fontsize=9)

    # Add trend line
    z = np.polyfit(sent_data["UNRATE"], sent_data["avg_sentiment"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sent_data["UNRATE"].min(),
                         sent_data["UNRATE"].max(), 100)
    ax.plot(x_line, p(x_line), color=COLORS["accent"],
            linewidth=2, linestyle="--", alpha=0.7)

    # Neutral sentiment line
    ax.axhline(y=0, color="#8b949e", linestyle=":", alpha=0.4)

    corr2 = sent_data["UNRATE"].corr(sent_data["avg_sentiment"])
    ax.set_title(f"U-3 vs Avg Sentiment\n(r = {corr2:.3f}, N≥10 only)\nColor = % Negative →",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("U-3 Rate (%)")
    ax.set_ylabel("Avg VADER Compound Score")
    ax.grid(True)

    plt.suptitle("Statistical Evidence: Do Official Rates Predict Distress?\n" +
                 "(Excluding sparse months with N<10 posts)",
                 fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "07_correlation_scatter.png"))
    plt.close()
    print("  ✓ Plot 7: Correlation scatter (raw + normalized + sentiment) saved.")


def plot_8_census_mismatch(df_degree, df_industry, save_dir):
    """Plot 8: Degree fields vs Industry employment — structural mismatch."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # ── Degree Fields (top categories, 25-39 age group) ──────────────────
    # Filter for 25-39 age group summary rows
    degree_data = df_degree[
        df_degree["Category"].str.contains("25 to 39 years", na=False) &
        ~df_degree["Category"].str.contains("Total:!!Male:!!25|Total:!!Female:!!25", na=False, regex=True)
    ].copy()

    if len(degree_data) == 0:
        # Fallback: use all rows with field names
        degree_data = df_degree[
            df_degree["Category"].str.contains(
                "Science and Engineering|Business|Education|Arts|Humanities",
                na=False, case=False
            )
        ].head(10).copy()

    if len(degree_data) > 0:
        degree_data["short_label"] = degree_data["Category"].apply(
            lambda x: x.split("!!")[-1][:35] if "!!" in str(x) else str(x)[:35]
        )
        degree_data = degree_data.nlargest(8, "Count")

        ax1.barh(degree_data["short_label"], degree_data["Count"],
                 color=COLORS["degree"], alpha=0.8, edgecolor="#30363d")
        ax1.set_title("What People Studied (B15011)\nAges 25-39",
                      fontsize=13, fontweight="bold")
        ax1.set_xlabel("Number of People")
        ax1.grid(axis="x", alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No matching degree data", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=14, color="#8b949e")

    # ── Industry Employment (top sectors) ────────────────────────────────
    industry_data = df_industry[
        ~df_industry["Category"].str.contains("Total:", na=False) |
        df_industry["Category"].str.match(r"^Total:!!(Male|Female):!![A-Z]", na=False)
    ].copy()

    # Get top-level industry categories (Total:!!Male:!!IndustryName pattern)
    industry_top = df_industry[
        df_industry["Category"].str.match(
            r"^Total:!!(Male|Female):!![A-Z]", na=False
        )
    ].copy()

    if len(industry_top) > 0:
        industry_top["short_label"] = industry_top["Category"].apply(
            lambda x: x.split("!!")[-1][:35] if "!!" in str(x) else str(x)[:35]
        )
        # Aggregate Male + Female for same industry
        industry_agg = industry_top.groupby("short_label")["Count"].sum()
        industry_agg = industry_agg.nlargest(10).sort_values()

        ax2.barh(industry_agg.index, industry_agg.values,
                 color=COLORS["u6"], alpha=0.8, edgecolor="#30363d")
        ax2.set_title("Where People Work (C24030)\nCivilian Employed Pop.",
                      fontsize=13, fontweight="bold")
        ax2.set_xlabel("Number of People")
        ax2.grid(axis="x", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No matching industry data", ha="center",
                 va="center", transform=ax2.transAxes, fontsize=14,
                 color="#8b949e")

    plt.suptitle("Structural Mismatch: Degrees vs. Jobs",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "08_census_mismatch.png"))
    plt.close()
    print("  ✓ Plot 8: Census mismatch saved.")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: CORRELATION ANALYSIS & SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
def correlation_analysis(df_merged):
    """
    Print correlation matrix between key features.
    
    Updated to filter sparse months and include normalized volume.
    """
    print("\n" + "=" * 72)
    print("SECTION 4 — Correlation Analysis")
    print("=" * 72)
    
    # Filter to reliable months (N>=10) for correlation analysis
    if "is_sparse" in df_merged.columns:
        reliable = df_merged[~df_merged["is_sparse"]].copy()
        print(f"\n  Using {len(reliable)} reliable months (N≥10 posts)")
        print(f"  Excluding {df_merged['is_sparse'].sum()} sparse months (N<10)")
    else:
        reliable = df_merged.copy()
        print(f"\n  Using all {len(reliable)} months (sparse filtering not available)")

    # Include both raw and normalized volume if available
    cols = ["UNRATE", "U6RATE", "U6_U3_SPREAD", "YOUTH_PREMIUM",
            "DEGREE_PREMIUM", "CIVPART", "post_count",
            "avg_sentiment", "distress_index_norm"]
    
    # Add normalized volume if available
    if "post_volume_normalized" in reliable.columns:
        cols.append("post_volume_normalized")

    # Filter to only columns that exist
    cols = [c for c in cols if c in reliable.columns]
    data = reliable[cols].dropna()

    if len(data) < 5:
        print("  ⚠ Not enough overlapping data for correlation.")
        return

    corr_matrix = data.corr()
    
    # Print correlations with raw post volume
    print("\n  Key Correlations with Raw Reddit Post Volume:")
    if "post_count" in corr_matrix:
        pc_corr = corr_matrix["post_count"].drop("post_count").sort_values()
        for var, val in pc_corr.items():
            direction = "↗" if val > 0 else "↘"
            strength = "strong" if abs(val) > 0.5 else (
                "moderate" if abs(val) > 0.3 else "weak")
            print(f"    {direction} {var:<28s}: r = {val:+.3f} ({strength})")
    
    # Print correlations with normalized post volume
    if "post_volume_normalized" in corr_matrix:
        print("\n  Key Correlations with Normalized Post Volume (per 10k subs):")
        pvn_corr = corr_matrix["post_volume_normalized"].drop(
            "post_volume_normalized").sort_values()
        for var, val in pvn_corr.items():
            direction = "↗" if val > 0 else "↘"
            strength = "strong" if abs(val) > 0.5 else (
                "moderate" if abs(val) > 0.3 else "weak")
            print(f"    {direction} {var:<28s}: r = {val:+.3f} ({strength})")

    # Print correlations with sentiment
    print("\n  Key Correlations with Avg Sentiment:")
    if "avg_sentiment" in corr_matrix:
        sent_corr = corr_matrix["avg_sentiment"].drop(
            "avg_sentiment").sort_values()
        for var, val in sent_corr.items():
            direction = "↗" if val > 0 else "↘"
            strength = "strong" if abs(val) > 0.5 else (
                "moderate" if abs(val) > 0.3 else "weak")
            print(f"    {direction} {var:<28s}: r = {val:+.3f} ({strength})")
    
    # Print note about distress index
    print("\n  NOTE: Revised distress index now uses standardized normalized volume,")
    print("        inverted sentiment, and diversity factor to avoid circular correlation")
    print("        with raw post_count.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "🔬" * 36)
    print("  THE REALITY GAP — EDA & Gap Analysis")
    print("🔬" * 36 + "\n")

    # ── Load Data ────────────────────────────────────────────────────────
    print("Loading datasets...")
    df_official = pd.read_csv(OFFICIAL_CSV, index_col="Date", parse_dates=True)
    df_census   = pd.read_csv(CENSUS_CSV)
    df_reddit   = pd.read_csv(REDDIT_CSV)
    df_reddit["created_utc"] = pd.to_datetime(df_reddit["created_utc"])

    print(f"  ✓ Official: {df_official.shape}")
    print(f"  ✓ Census:   {df_census.shape}")
    print(f"  ✓ Reddit:   {df_reddit.shape}")

    # ── Section 1: Data Quality Audit ────────────────────────────────────
    df_reddit = audit_reddit_data(df_reddit)

    # ── Section 2: Feature Engineering ───────────────────────────────────
    df_official = engineer_official_features(df_official)
    df_reddit, monthly_reddit = engineer_reddit_features(df_reddit)
    df_degree, df_industry = process_census_data(df_census)
    df_merged = merge_datasets(df_official, monthly_reddit)

    # ── Save engineered data ─────────────────────────────────────────────
    merged_path = os.path.join(DATA_DIR, "df_merged_features.csv")
    df_merged.to_csv(merged_path, index=False)
    print(f"\n  ✅ Merged features saved to {merged_path}")
    print(f"     Shape: {df_merged.shape}")
    print(f"     Columns: {list(df_merged.columns)}")

    # Also save the sentiment-scored Reddit data
    reddit_scored_path = os.path.join(DATA_DIR, "df_reddit_scored.csv")
    df_reddit.to_csv(reddit_scored_path, index=False)
    print(f"  ✅ Scored Reddit data saved to {reddit_scored_path}")

    # ── Section 3: Visualizations ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SECTION 3 — EDA Visualizations")
    print("=" * 72)
    os.makedirs(PLOT_DIR, exist_ok=True)

    plot_1_unemployment_rates(df_official, PLOT_DIR)
    plot_2_u6_u3_spread(df_official, PLOT_DIR)
    plot_3_reality_gap(df_merged, PLOT_DIR)
    plot_4_heatmap(df_reddit, PLOT_DIR)
    plot_5_search_terms(df_reddit, PLOT_DIR)
    plot_6_sentiment_timeseries(monthly_reddit, PLOT_DIR)
    plot_7_correlation_scatter(df_merged, PLOT_DIR)
    plot_8_census_mismatch(df_degree, df_industry, PLOT_DIR)

    # ── Section 4: Correlation Analysis ──────────────────────────────────
    correlation_analysis(df_merged)

    # ── Final Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("EDA & GAP ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"\n  Output files:")
    print(f"    {merged_path}")
    print(f"    {reddit_scored_path}")
    print(f"    {PLOT_DIR}/ (8 PNG visualizations)")
    print(f"\n  Total engineered features: {df_merged.shape[1]} columns")
    print(f"  Months with both official + Reddit data: "
          f"{(df_merged['post_count'] > 0).sum()}")

    return df_merged


if __name__ == "__main__":
    main()
