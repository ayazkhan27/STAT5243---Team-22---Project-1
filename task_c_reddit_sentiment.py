"""
=============================================================================
Task C: The "Sentiment" Gap — Reddit Scraping via OAuth2 + requests
=============================================================================
Project : The Reality Gap: Contrasting Official US Labor Statistics
          with Public Sentiment (2020-2026)
Course  : STAT 5243 — Spring 2026 — Columbia University
Purpose : Constructs a "Sentiment Index" by scraping employment-distress
          posts from targeted subreddits. This unstructured text data
          represents the *lived experience* of job seekers — the signal
          that official statistics suppress.

          We search for specific distress keywords across 4 subreddits,
          using Reddit's OAuth2 JSON API. To avoid Reddit's recency bias
          (which returns mostly 2024-2025 results when querying "all"),
          we iterate YEAR-BY-YEAR using CloudSearch timestamp syntax:
              query = "layoff timestamp:1577836800..1609459200"
          This forces Reddit to return the top posts for each specific
          year, producing a time-balanced dataset across 2020-2026.

          For each year × subreddit × search term, we pull up to 100
          posts, then deduplicate globally. The progress bar shows LIVE
          post counts and a checkpoint CSV is saved every 50 queries.

Output  : data/df_reddit_sentiment.csv (primary)
          data/df_reddit_sentiment.parquet (gzip backup)
          data/temp_reddit_checkpoint.csv (incremental, deleted on success)
=============================================================================
"""

import os
import re
import sys
import time
import json
import datetime
import requests
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 1. CREDENTIALS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Load API keys from secrets.json (git-ignored for public repo safety)
SECRETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets.json")
with open(SECRETS_PATH, "r") as f:
    _secrets = json.load(f)

REDDIT_CLIENT_ID     = _secrets["REDDIT_CLIENT_ID"]
REDDIT_CLIENT_SECRET = _secrets["REDDIT_CLIENT_SECRET"]
REDDIT_USER_AGENT    = _secrets["REDDIT_USER_AGENT"]

# Target subreddits — chosen to capture different facets of the "silent recession"
SUBREDDITS = [
    "layoffs",          # Direct layoff announcements and experiences
    "jobs",             # General job market sentiment
    "recruitinghell",   # Frustration with hiring processes — "ghosting," etc.
    "csMajors",         # Tech-specific recession; CS degree holders struggling
]

# Search terms — these are "distress signals" that indicate the reality
# behind optimistic official numbers
SEARCH_TERMS = [
    # ── Original distress signals ────────────────────────────────────────
    "layoff",
    "unemployed",
    "severance",
    "ghosted",
    "hundred applications",
    # ── Expanded terms for deeper gap analysis ───────────────────────────
    "hiring freeze",            # companies quietly stopping hiring
    "overqualified",            # degree mismatch signal
    "entry level experience",   # the catch-22 requiring 3+ yrs for entry-level
    "no response",              # broader ghosting / application black hole
    "job market",               # general sentiment about market conditions
    "recession",                # macro-economic anxiety
    "cost of living",           # economic pressure even for employed workers
]

# Date range: Jan 2020 – Jan 2026 (as Unix timestamps for filtering)
START_DATE = datetime.datetime(2020, 1, 1)
END_DATE   = datetime.datetime(2026, 2, 1)  # exclusive upper bound
START_TS   = int(START_DATE.timestamp())
END_TS     = int(END_DATE.timestamp())

# Year-by-year iteration to force time-balanced scraping
# Without this, Reddit's API returns mostly 2024-2025 content
YEARS = list(range(2020, 2027))  # 2020, 2021, ..., 2026

# Output configuration
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "df_reddit_sentiment.csv")
OUTPUT_PQ   = os.path.join(OUTPUT_DIR, "df_reddit_sentiment.parquet")
CHECKPOINT  = os.path.join(OUTPUT_DIR, "temp_reddit_checkpoint.csv")

# Rate limiting — Reddit allows 60 requests/min for OAuth
REQUEST_DELAY = 1.0
MAX_RETRIES   = 3
BACKOFF_FACTOR = 2
CHECKPOINT_EVERY = 50  # Save checkpoint every N queries

# ─────────────────────────────────────────────────────────────────────────────
# 2. OAuth2: Get access token
# ─────────────────────────────────────────────────────────────────────────────
def get_oauth_token():
    """
    Obtain a Reddit OAuth2 bearer token using client credentials flow.
    This is the standard approach for script-type Reddit apps.
    """
    auth = requests.auth.HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
    data = {"grant_type": "client_credentials"}
    headers = {"User-Agent": REDDIT_USER_AGENT}

    resp = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=auth, data=data, headers=headers, timeout=15,
    )
    resp.raise_for_status()
    token_data = resp.json()
    return token_data["access_token"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. HELPER: Clean text (remove URLs, excess whitespace)
# ─────────────────────────────────────────────────────────────────────────────
URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|"
    r"(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)

def clean_text(text):
    """Remove URLs, newlines, and extra whitespace."""
    if not text or text in ("[deleted]", "[removed]"):
        return ""
    text = URL_PATTERN.sub("", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4. HELPER: Search with retry and pagination
# ─────────────────────────────────────────────────────────────────────────────
def search_subreddit(subreddit, query, headers, sort="relevance",
                     limit=100, time_filter="all"):
    """
    Search a subreddit using Reddit's OAuth2 JSON API.
    Returns a list of post dicts. Handles pagination via 'after'.
    """
    url = f"https://oauth.reddit.com/r/{subreddit}/search"
    params = {
        "q": query,
        "sort": sort,
        "limit": min(limit, 100),  # Reddit max is 100 per page
        "restrict_sr": True,
        "t": time_filter,
    }

    all_posts = []
    after = None

    for page in range(1, 11):  # Max 10 pages = 1000 posts per query
        if after:
            params["after"] = after

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    url, headers=headers, params=params, timeout=15,
                )
                if resp.status_code == 429:
                    wait = BACKOFF_FACTOR ** attempt
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException:
                if attempt == MAX_RETRIES:
                    return all_posts
                time.sleep(BACKOFF_FACTOR ** attempt)
        else:
            return all_posts

        data = resp.json().get("data", {})
        children = data.get("children", [])
        if not children:
            break

        for child in children:
            all_posts.append(child["data"])

        after = data.get("after")
        if not after or len(all_posts) >= limit:
            break

        time.sleep(REQUEST_DELAY)

    return all_posts[:limit]


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("TASK C — Sentiment Gap Extraction (Reddit OAuth2 API)")
    print("=" * 72)
    print(f"Subreddits   : {', '.join(SUBREDDITS)}")
    print(f"Search Terms : {', '.join(SEARCH_TERMS)}")
    print(f"Date Filter  : {START_DATE.strftime('%Y-%m-%d')} → "
          f"{END_DATE.strftime('%Y-%m-%d')}")
    print(f"Output       : {OUTPUT_CSV}")
    print("-" * 72)

    # ── Authenticate ─────────────────────────────────────────────────────
    print("\nAuthenticating with Reddit OAuth2...")
    try:
        access_token = get_oauth_token()
        print("  ✓ OAuth2 token obtained.")
    except Exception as e:
        print(f"  ✗ Authentication failed: {e}")
        sys.exit(1)

    oauth_headers = {
        "Authorization": f"bearer {access_token}",
        "User-Agent": REDDIT_USER_AGENT,
    }

    # Verify token with a simple call
    test_resp = requests.get(
        "https://oauth.reddit.com/api/v1/me",
        headers=oauth_headers, timeout=10,
    )
    print(f"  ✓ Token verified (status: {test_resp.status_code})")

    # ── Calculate total iterations ───────────────────────────────────────
    # Now we loop years × subreddits × terms for time-balanced results
    total_queries = len(YEARS) * len(SUBREDDITS) * len(SEARCH_TERMS)
    print(f"\n  Total queries: {total_queries} "
          f"({len(YEARS)} years × {len(SUBREDDITS)} subs × "
          f"{len(SEARCH_TERMS)} terms)")
    print(f"  ⏱ Estimated time: ~{total_queries * 1.2 / 60:.0f} minutes\n")

    # ── Main scraping loop ───────────────────────────────────────────────
    all_posts = []
    seen_ids = set()  # Deduplication across queries
    query_count = 0

    pbar = tqdm(
        total=total_queries,
        desc="Scraping",
        unit="query",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}] {postfix}",
    )

    for year in YEARS:
        # Build CloudSearch timestamp range for this year
        year_start_ts = int(datetime.datetime(year, 1, 1).timestamp())
        if year < 2026:
            year_end_ts = int(datetime.datetime(year + 1, 1, 1).timestamp())
        else:
            year_end_ts = END_TS  # 2026 only goes to Feb 1
        timestamp_clause = f"timestamp:{year_start_ts}..{year_end_ts}"

        for sub_name in SUBREDDITS:
            for term in SEARCH_TERMS:
                pbar.set_postfix_str(
                    f"{year} r/{sub_name} '{term}' | "
                    f"{len(all_posts)} posts",
                    refresh=True,
                )

                # Append timestamp clause to force year-specific results
                full_query = f"{term} {timestamp_clause}"

                raw_posts = search_subreddit(
                    sub_name, full_query, oauth_headers,
                    sort="relevance", limit=100, time_filter="all",
                )

                # Extract and filter
                new_this_query = 0
                for p in raw_posts:
                    pid = p.get("id", "")
                    if pid in seen_ids:
                        continue

                    created_utc = p.get("created_utc", 0)
                    if created_utc < START_TS or created_utc >= END_TS:
                        continue

                    seen_ids.add(pid)
                    new_this_query += 1

                    all_posts.append({
                        "post_id":     pid,
                        "title":       clean_text(p.get("title", "")),
                        "selftext":    clean_text(p.get("selftext", "")),
                        "subreddit":   p.get("subreddit", sub_name),
                        "created_utc": datetime.datetime.utcfromtimestamp(
                            created_utc
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "score":       p.get("score", 0),
                        "search_term": term,
                    })

                query_count += 1
                pbar.update(1)

                pbar.set_postfix_str(
                    f"{year} r/{sub_name} '{term}' +{new_this_query} | "
                    f"TOTAL: {len(all_posts)} posts",
                    refresh=True,
                )

                # ── Checkpoint every N queries ──────────────────────────
                if query_count % CHECKPOINT_EVERY == 0 and all_posts:
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    pd.DataFrame(all_posts).to_csv(CHECKPOINT, index=False)

                # Rate limiting
                time.sleep(REQUEST_DELAY)

    pbar.close()

    # ── Build DataFrame ──────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("Building DataFrame...")

    if not all_posts:
        print("  ⚠ WARNING: No posts were retrieved.")
        df = pd.DataFrame(columns=[
            "post_id", "title", "selftext", "subreddit",
            "created_utc", "score", "search_term",
        ])
    else:
        df = pd.DataFrame(all_posts)

    # Convert created_utc to datetime
    if len(df) > 0:
        df["created_utc"] = pd.to_datetime(df["created_utc"])
        df = df.sort_values("created_utc").reset_index(drop=True)

    # ── Data Quality Report ──────────────────────────────────────────────
    print("\n📊 Data Quality Report:")
    print(f"  Total Posts    : {len(df):,}")
    print(f"  Unique Posts   : {df['post_id'].nunique():,}" if len(df) > 0
          else "  Unique Posts   : 0")

    if len(df) > 0:
        print(f"  Date Range     : {df['created_utc'].min()} → "
              f"{df['created_utc'].max()}")

        print(f"\n  Posts by Subreddit:")
        for sub, count in df["subreddit"].value_counts().items():
            print(f"    r/{sub:<20s}: {count:>6,}")

        print(f"\n  Posts by Search Term:")
        for term, count in df["search_term"].value_counts().items():
            print(f"    '{term}'{'':.<24s}: {count:>6,}")

        empty_body = (df["selftext"] == "").sum()
        print(f"\n  Posts with empty body: {empty_body:,} "
              f"({empty_body/len(df)*100:.1f}%)")

    # ── Save as CSV (primary) and Parquet (secondary) ────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False)
    csv_kb = os.path.getsize(OUTPUT_CSV) / 1024
    print(f"\n✅ CSV  → {OUTPUT_CSV} ({csv_kb:.1f} KB)")

    df.to_parquet(OUTPUT_PQ, compression="gzip", index=False)
    pq_kb = os.path.getsize(OUTPUT_PQ) / 1024
    print(f"✅ PQ   → {OUTPUT_PQ} ({pq_kb:.1f} KB)")

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)
        print("✅ Checkpoint cleaned up")

    # ── Preview ──────────────────────────────────────────────────────────
    if len(df) > 0:
        print("\n📋 Preview (first 5 rows):")
        preview_cols = ["post_id", "subreddit", "created_utc", "score", "title"]
        print(df[preview_cols].head().to_string(index=False))

    print("\n" + "=" * 72)
    print("TASK C COMPLETE")
    print("=" * 72)

    return df


if __name__ == "__main__":
    main()
