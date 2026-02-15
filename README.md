# The Reality Gap: Contrasting Official US Labor Statistics with Public Sentiment (2020–2026)

> **STAT 5243 — Spring 2026 — Columbia University**
> A graduate-level comparative study demonstrating that the official U-3 Unemployment Rate is an overly optimistic indicator that fails to capture the "silent recession" plaguing white-collar and entry-level labor markets.

---

## 🎯 Hypothesis

Official unemployment metrics (U-3) understate the true severity of labor market distress for **Entry-Level Workers (Ages 20–24)** and **Recent College Graduates**. We call this divergence the **"Reality Gap"** and measure it through three lenses:

1. **The Official Baseline** — Government statistics from the Federal Reserve (FRED)
2. **The Demographic Context** — Census Bureau data revealing structural "Degree Mismatch"
3. **The Sentiment Index** — Reddit discussions as a proxy for real-time public distress

---

## 📈 Key Findings

### Finding 1: The Core Gap Is Real

From **2022 onward**, U-3 and U-6 flatten near historic lows (~3.5% and ~7%), yet Reddit distress posts **surge exponentially** — from ~10/month in 2021 to **263/month by late 2025**. The correlation between U-3 and distress volume is **negative** (r = −0.264): as official rates go *down*, distress goes *up*.

### Finding 2: Youth Unemployment Diverges from the Headline

After the COVID recovery, U-3 settles around 3.5–4%, but youth unemployment (Ages 20–24) remains **2–4 percentage points higher** with increasing volatility from 2024 onward. Bachelor's holders aged 20–24 show even more erratic swings (4%–10%), suggesting the entry-level market is far more unstable than the headline rate implies.

### Finding 3: The "Hidden Unemployed" Are Persistent

The U6–U3 spread (measuring discouraged workers and involuntary part-timers excluded from the headline rate) hovers around **3.3–3.7 percentage points** throughout 2022–2026, representing ~3.5% of the labor force that is effectively unemployed but not counted. This spread has been *rising slightly* since 2024 even as U-3 stays flat.

### Finding 4: Sentiment Is Deteriorating Over Time

VADER compound scores show a **downward trend** from 2020 to 2026. Average monthly sentiment dropped from +0.3 to +0.7 (early 2020) to **0.0 to +0.1** (2024–2026). The percentage of negative posts stabilizes at a higher baseline (~35–45%) compared to earlier periods.

### Finding 5: Distress Is Broad-Based, Not Isolated

All four subreddits light up simultaneously from late 2023 onward. Search terms are remarkably evenly distributed (255–385 posts each), meaning distress isn't driven by a single type of complaint — layoffs, ghosting, overqualification, hiring freezes, and the "hundreds of applications" phenomenon are all co-occurring.

### Finding 6: Structural Degree–Job Mismatch Exists

~7.5 million people aged 25–39 hold Science & Engineering degrees, but the matching employment sectors don't absorb them all. The largest hiring industries are healthcare and education services — not the STEM-adjacent industries these degrees target.

### Finding 7: Two Regimes in the Scatter Plot

When U-3 was high (COVID, 8–14%), distress volume was paradoxically low. When U-3 sits at its lowest (3.5–4.5%), distress volume is at its **highest** (100–263 posts/month). This L-shaped pattern is the gap in its starkest statistical form.

### Correlation Summary

| Variable | vs Post Volume | vs Avg Sentiment |
|----------|---------------|-----------------|
| UNRATE (U-3) | r = −0.264 | r = +0.261 |
| U6RATE | r = −0.249 | r = +0.265 |
| CIVPART | r = +0.395 | r = −0.319 |
| Distress Index | r = +0.991 | r = −0.167 |

### Conclusion

> The U-3 Unemployment Rate is failing as a measure of labor market health for entry-level and white-collar workers. From 2022–2026, while headline unemployment sits near historic lows, public distress has surged exponentially. The negative correlation between official rates and distress volume, the persistent U6–U3 spread, the deteriorating sentiment trajectory, and the structural degree–job mismatch all point to a **"silent recession"** that official statistics are not designed to capture.

---

## 📊 Data Sources & Series IDs

### Task A: FRED API — Official Labor Market Indicators

| Series ID | Description | Why It Matters |
|-----------|-------------|----------------|
| `UNRATE` | General Unemployment Rate (U-3) | The "headline" number — our null hypothesis |
| `U6RATE` | U-6 Rate (includes discouraged + part-time) | The "real" rate — includes people the U-3 excludes |
| `CIVPART` | Civilian Labor Force Participation Rate | Captures people who gave up looking entirely |
| `LNS14000036` | Unemployment Rate, Ages 20–24 | Entry-level proxy — isolates the most impacted demographic |
| `CGBD2024` | Unemployment Rate, Bachelor's 20–24 | "Degree Mismatch" proxy — even a degree doesn't guarantee employment |

### Task B: Census ACS API — Structural Underemployment

| Table | Description | Purpose |
|-------|-------------|---------|
| `B15011` | Sex by Age by Field of Bachelor's Degree | What people *studied* — the supply side |
| `C24030` | Sex by Industry for Civilian Employed Population | Where people *actually work* — the demand side |

### Task C: Reddit API — Sentiment Time-Series

| Subreddit | Signal |
|-----------|--------|
| `r/layoffs` | Direct layoff announcements and firsthand experiences |
| `r/jobs` | General job market sentiment — ghosting, rejections |
| `r/recruitinghell` | Systemic failures in hiring — "100+ applications, 0 responses" |
| `r/csMajors` | Tech-specific recession signal — CS degree holders struggling |

**Search Terms** (12): `layoff`, `unemployed`, `severance`, `ghosted`, `hundred applications`, `hiring freeze`, `overqualified`, `entry level experience`, `no response`, `job market`, `recession`, `cost of living`

**Time-Balanced Scraping**: Reddit's API is biased toward recent, high-engagement content. Without intervention, a search for "layoff" with `time_filter="all"` returns mostly 2024–2025 posts. To fix this, we iterate **year-by-year** using CloudSearch timestamp syntax (e.g., `"layoff timestamp:1577836800..1609459200"`), which forces Reddit to return the top posts *per year*. This produces 7 years × 4 subreddits × 12 terms = **336 queries** and a substantially more balanced temporal distribution.

---

## 📁 Project Structure

```
STAT 5243/
├── data/
│   ├── df_official.csv                  # Task A: FRED time-series (73 months × 5 series)
│   ├── df_census_degree_mismatch.csv    # Task B: Census ACS snapshot (94 rows)
│   ├── df_reddit_sentiment.csv          # Task C: 3,919 Reddit posts (primary)
│   ├── df_reddit_sentiment.parquet      # Task C: compressed backup
│   ├── df_merged_features.csv           # EDA: merged monthly time-series (73 × 30 cols)
│   ├── df_reddit_scored.csv             # EDA: Reddit posts with VADER sentiment scores
│   └── plots/                           # EDA: 8 PNG visualizations
│       ├── 01_unemployment_rates.png    # U-3, U-6, Youth, and Degree rates
│       ├── 02_u6_u3_spread.png          # The "hidden unemployed" spread
│       ├── 03_reality_gap.png           # ★ Core finding: dual-axis gap chart
│       ├── 04_heatmap.png              # Subreddit × month activity heatmap
│       ├── 05_search_terms.png          # Distress keyword frequency
│       ├── 06_sentiment_timeseries.png  # VADER compound + % negative over time
│       ├── 07_correlation_scatter.png   # U-3 and U6-U3 spread vs distress volume
│       └── 08_census_mismatch.png       # Degrees earned vs industry employment
├── task_a_official_baseline.py          # FRED API extraction
├── task_b_census_demographics.py        # Census ACS extraction
├── task_c_reddit_sentiment.py           # Reddit scraping via OAuth2
├── eda_gap_analysis.py                  # EDA, feature engineering & visualization
├── requirements.txt                     # Python dependencies
├── secrets.json                         # API keys (git-ignored)
├── secrets.example.json                 # Template for API keys
└── README.md                            # This file
```

---

## 🚀 How to Run

### Prerequisites

- Python 3.9+
- API keys for FRED, Census Bureau, and Reddit (see **API Keys Setup** below)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp secrets.example.json secrets.json
# Edit secrets.json with your actual API keys
```

### 3. Run the Pipeline

```bash
# Phase 1: Data Ingestion
python task_a_official_baseline.py      # ~10 sec → data/df_official.csv
python task_b_census_demographics.py    # ~15 sec → data/df_census_degree_mismatch.csv
python task_c_reddit_sentiment.py       # ~7 min  → data/df_reddit_sentiment.csv

# Phase 2: EDA & Gap Analysis
python eda_gap_analysis.py              # ~15 sec → data/df_merged_features.csv + 8 plots
```

---

## 🔑 API Keys Setup

All credentials are loaded from `secrets.json` (git-ignored):

```json
{
    "FRED_API_KEY": "YOUR_FRED_KEY",
    "BLS_API_KEY": "YOUR_BLS_KEY",
    "CENSUS_API_KEY": "YOUR_CENSUS_KEY",
    "REDDIT_CLIENT_ID": "YOUR_REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET": "YOUR_REDDIT_CLIENT_SECRET",
    "REDDIT_USER_AGENT": "linux:employment_gap_analysis:v1.0 (by /u/YOUR_USERNAME)"
}
```

| Service | Registration |
|---------|-------------|
| FRED API | [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) |
| Census API | [api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html) |
| Reddit API | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) |

---

## 🛠️ Methodology

### Feature Engineering (30 columns)

| Category | Features |
|----------|----------|
| **Official Spreads** | U6–U3 Spread, Youth Premium, Degree Premium |
| **Momentum** | Month-over-month changes, 3-month rolling averages, year-over-year change |
| **Reddit Aggregates** | Monthly post count, avg/median score, total score |
| **Sentiment** | VADER compound (avg, median), % negative, % positive |
| **Composite** | Distress Index = post_count × pct_negative (normalized 0–100) |

### Sentiment Analysis

- **Tool**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Input**: Title + selftext concatenated per post (capped at 5,000 chars)
- **Output**: Compound score (−1 to +1), classified as negative (< −0.05), neutral, or positive (> +0.05)

---

## ⚠️ Limitations & Caveats

### Census Degree–Industry Proxy
Comparing "Field of Degree" (B15011) to "Industry of Employment" (C24030) is an imperfect proxy for underemployment. A Biology major working in "Educational Services" might be a Biology Teacher (a match) or a Janitor (a mismatch). We assume that **aggregate trends** across millions of workers still reveal structural misalignment, even if individual-level classification is noisy.

### VADER Sentiment Limitations
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon-based tool optimized for social media. However, it struggles with:
- **Sarcasm** (e.g., "Love getting ghosted after 5 rounds of interviews" scores as positive)
- **Domain-specific jargon** (e.g., "I got a severance package" may score neutral despite distress context)
- **Long-form posts** where overall tone is mixed

A transformer-based model (e.g., RoBERTa fine-tuned on employment forums) would improve accuracy but was outside the scope of this initial analysis.

### Reddit Selection Bias
- Reddit skews younger, more tech-literate, and more male than the general population
- Users who post about job struggles are self-selecting — people with good jobs rarely post
- High-scoring posts are over-represented in API results even within timestamp-filtered queries
- Subreddit growth over time (r/layoffs grew significantly from 2020→2026) naturally inflates post volume independent of actual distress levels

---

## 👥 Team

| Name | UNI | Contribution |
|------|-----|-------------|
| **Megan Wang** | mw3856 | Data Engineering & Pipeline |
| **Ayaz Khan** | aak2259 | Data Science & Gap Analysis |
| **Sherry Wang** | yw4542 | EDA & Visualization |
| **Pingyu Zhou** | pz2341 | Feature Engineering & Report |

---

## 📄 License

This project is for academic use only (Columbia University STAT 5243).
