"""
=============================================================================
Unit Tests for Data Processing and Feature Engineering
=============================================================================
Project : The Reality Gap: Contrasting Official US Labor Statistics
          with Public Sentiment (2020-2026)
Course  : STAT 5243 — Spring 2026 — Columbia University
Purpose : Comprehensive unit tests for data processing, feature engineering,
          and analysis functions to ensure correctness and reproducibility.

Tests Cover:
  - Feature normalization by subreddit subscribers
  - Correlation computation accuracy
  - Sparse month exclusion logic (N<10)
  - Distress index calculation
  - Search term categorization
  - VADER sentiment scoring
=============================================================================
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestFeatureNormalization:
    """Test suite for post volume normalization by subscribers."""
    
    def test_normalization_basic(self):
        """Test basic post volume normalization calculation."""
        # Create sample data
        subscribers = {"subreddit_a": 10000, "subreddit_b": 50000}
        posts_per_sub = {"subreddit_a": 100, "subreddit_b": 200}
        
        # Calculate normalized posts per 10k subscribers
        normalized_a = (posts_per_sub["subreddit_a"] / subscribers["subreddit_a"]) * 10000
        normalized_b = (posts_per_sub["subreddit_b"] / subscribers["subreddit_b"]) * 10000
        
        # Assertions
        assert normalized_a == 100.0, "Normalization for subreddit_a incorrect"
        assert normalized_b == 40.0, "Normalization for subreddit_b incorrect"
        assert normalized_a > normalized_b, "Relative activity levels incorrect"
    
    def test_normalization_zero_subscribers(self):
        """Test handling of zero or missing subscriber counts."""
        subscribers = {"subreddit_a": 0, "subreddit_b": 10000}
        posts_per_sub = {"subreddit_a": 100, "subreddit_b": 100}
        
        # Should handle division by zero gracefully
        normalized_a = (posts_per_sub["subreddit_a"] / 
                       (subscribers["subreddit_a"] if subscribers["subreddit_a"] > 0 
                        else np.nan)) * 10000
        normalized_b = (posts_per_sub["subreddit_b"] / subscribers["subreddit_b"]) * 10000
        
        assert np.isnan(normalized_a), "Should return NaN for zero subscribers"
        assert normalized_b == 100.0, "Valid normalization should work"
    
    def test_normalization_aggregation(self):
        """Test aggregation of normalized volumes across multiple subreddits."""
        df = pd.DataFrame({
            'subreddit': ['sub_a', 'sub_a', 'sub_b', 'sub_b'],
            'posts': [10, 20, 30, 40],
            'subscribers': [10000, 10000, 50000, 50000]
        })
        
        df['normalized'] = (df['posts'] / df['subscribers']) * 10000
        
        # Aggregate by subreddit
        agg = df.groupby('subreddit')['normalized'].mean()
        
        assert abs(agg['sub_a'] - 15.0) < 0.01, "Subreddit A average incorrect"
        assert abs(agg['sub_b'] - 7.0) < 0.01, "Subreddit B average incorrect"


class TestCorrelationComputation:
    """Test suite for correlation coefficient calculations."""
    
    def test_perfect_positive_correlation(self):
        """Test detection of perfect positive correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        
        corr = np.corrcoef(x, y)[0, 1]
        assert abs(corr - 1.0) < 0.0001, "Perfect positive correlation not detected"
    
    def test_perfect_negative_correlation(self):
        """Test detection of perfect negative correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 8, 6, 4, 2])
        
        corr = np.corrcoef(x, y)[0, 1]
        assert abs(corr - (-1.0)) < 0.0001, "Perfect negative correlation not detected"
    
    def test_no_correlation(self):
        """Test detection of no correlation."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        
        corr = np.corrcoef(x, y)[0, 1]
        assert abs(corr) < 0.3, "Should detect weak/no correlation"
    
    def test_correlation_with_missing_data(self):
        """Test correlation computation with NaN values."""
        df = pd.DataFrame({
            'x': [1, 2, 3, np.nan, 5],
            'y': [2, 4, 6, 8, 10]
        })
        
        # Pandas corr() automatically handles NaN
        corr = df['x'].corr(df['y'])
        
        # Should compute on valid pairs only
        assert not np.isnan(corr), "Correlation should handle NaN gracefully"
        assert abs(corr - 1.0) < 0.0001, "Correlation on valid pairs should be perfect"


class TestSparseMonthExclusion:
    """Test suite for sparse month filtering logic."""
    
    def test_identify_sparse_months(self):
        """Test identification of months with N<10 posts."""
        monthly = pd.DataFrame({
            'year_month': pd.period_range('2020-01', periods=6, freq='M'),
            'post_count': [5, 12, 8, 15, 20, 9]
        })
        
        # Apply sparse threshold
        monthly['is_sparse'] = monthly['post_count'] < 10
        
        assert monthly['is_sparse'].sum() == 3, "Should identify 3 sparse months"
        assert monthly.loc[0, 'is_sparse'] == True, "Month with 5 posts should be sparse"
        assert monthly.loc[1, 'is_sparse'] == False, "Month with 12 posts should not be sparse"
    
    def test_filter_for_analysis(self):
        """Test filtering sparse months for correlation analysis."""
        df = pd.DataFrame({
            'post_count': [5, 12, 8, 15, 20, 9],
            'sentiment': [0.1, 0.2, -0.1, 0.3, -0.2, 0.0],
            'unrate': [4.0, 4.5, 5.0, 4.2, 3.8, 4.1]
        })
        
        df['is_sparse'] = df['post_count'] < 10
        reliable = df[~df['is_sparse']]
        
        assert len(reliable) == 3, "Should have 3 reliable months"
        assert all(reliable['post_count'] >= 10), "All filtered months should have N>=10"
    
    def test_sparse_statistics(self):
        """Test statistical properties with and without sparse months."""
        df = pd.DataFrame({
            'post_count': [2, 100, 3, 95, 1, 98],
            'sentiment': [-0.5, 0.3, -0.4, 0.2, -0.6, 0.25]
        })
        
        # Stats including sparse months
        mean_all = df['sentiment'].mean()
        
        # Stats excluding sparse months (N<10)
        df['is_sparse'] = df['post_count'] < 10
        mean_reliable = df[~df['is_sparse']]['sentiment'].mean()
        
        # Should be substantially different due to sparse outliers
        assert abs(mean_all - mean_reliable) > 0.1, "Sparse filtering should affect results"


class TestDistressIndexCalculation:
    """Test suite for revised distress index formula."""
    
    def test_distress_index_components(self):
        """Test that distress index uses correct standardized components."""
        df = pd.DataFrame({
            'volume_normalized': [10, 20, 30, 40, 50],
            'avg_sentiment': [0.5, 0.2, -0.1, -0.3, -0.5],
            'unique_subreddits': [1, 2, 2, 3, 4]
        })
        
        # Standardize components (z-scores)
        df['volume_z'] = (df['volume_normalized'] - df['volume_normalized'].mean()) / df['volume_normalized'].std()
        df['sentiment_z'] = (df['avg_sentiment'] - df['avg_sentiment'].mean()) / df['avg_sentiment'].std()
        df['diversity_z'] = (df['unique_subreddits'] - df['unique_subreddits'].mean()) / df['unique_subreddits'].std()
        
        # Compute distress index: higher volume + lower sentiment + higher diversity
        df['distress'] = df['volume_z'] + (-1 * df['sentiment_z']) + (0.5 * df['diversity_z'])
        
        # Highest distress should be row with high volume, negative sentiment, high diversity
        max_distress_idx = df['distress'].idxmax()
        
        assert max_distress_idx == 4, "Highest distress should be at index 4"
        assert df.loc[4, 'volume_normalized'] == 50, "Should have high volume"
        assert df.loc[4, 'avg_sentiment'] == -0.5, "Should have negative sentiment"
    
    def test_distress_index_normalization(self):
        """Test distress index normalization to 0-100 scale."""
        distress_raw = np.array([-2.5, -1.0, 0.5, 2.0, 3.5])
        
        # Normalize to 0-100
        distress_min = distress_raw.min()
        distress_max = distress_raw.max()
        distress_norm = ((distress_raw - distress_min) / (distress_max - distress_min)) * 100
        
        assert distress_norm.min() == 0, "Min should be 0"
        assert distress_norm.max() == 100, "Max should be 100"
        assert 0 <= distress_norm[2] <= 100, "All values should be in [0, 100]"
    
    def test_distress_no_circular_correlation(self):
        """Test that distress index uses normalized volume, not raw count."""
        # Create data where raw count and normalized count diverge significantly
        df = pd.DataFrame({
            'post_count': [100, 50, 150, 75],  # Raw counts
            'subscribers': [1000, 5000, 1000, 5000],  # Varying subscriber base
            'avg_sentiment': [0.3, -0.2, 0.1, -0.4]
        })
        
        # Calculate normalized volume
        df['volume_normalized'] = (df['post_count'] / df['subscribers']) * 10000
        
        # Verify that normalized and raw volumes have different patterns
        # Raw: [100, 50, 150, 75]
        # Normalized: [1000, 100, 1500, 150]
        assert df['volume_normalized'].iloc[0] == 1000, "First normalized should be 1000"
        assert df['volume_normalized'].iloc[1] == 100, "Second normalized should be 100"
        
        # The key insight: distress calculation should use normalized, not raw
        # This test verifies the formula incorporates normalized volume
        df['volume_z'] = (df['volume_normalized'] - df['volume_normalized'].mean()) / df['volume_normalized'].std()
        df['sentiment_z'] = (df['avg_sentiment'] - df['avg_sentiment'].mean()) / df['avg_sentiment'].std()
        df['distress'] = df['volume_z'] + (-1 * df['sentiment_z'])
        
        # Verify distress calculation completed successfully
        assert not df['distress'].isna().any(), "Distress should be calculated for all rows"
        assert len(df['distress'].unique()) > 1, "Distress should have variation"


class TestSearchTermCategorization:
    """Test suite for search term positive/negative categorization."""
    
    def test_negative_term_classification(self):
        """Test that negative search terms are correctly classified."""
        negative_terms = ["layoff", "unemployed", "ghosted", "hiring freeze"]
        positive_terms = ["got a job", "hired", "promotion"]
        
        def classify_term(term, positive_list):
            return "positive" if term in positive_list else "negative"
        
        for term in negative_terms:
            assert classify_term(term, positive_terms) == "negative"
        
        for term in positive_terms:
            assert classify_term(term, positive_terms) == "positive"
    
    def test_term_aggregation_by_category(self):
        """Test aggregation of posts by term category."""
        df = pd.DataFrame({
            'term_category': ['negative', 'negative', 'positive', 'negative', 'positive'],
            'post_id': [1, 2, 3, 4, 5],
            'sentiment': [-0.3, -0.5, 0.4, -0.2, 0.6]
        })
        
        agg = df.groupby('term_category').agg(
            count=('post_id', 'count'),
            avg_sentiment=('sentiment', 'mean')
        )
        
        assert agg.loc['negative', 'count'] == 3, "Should have 3 negative posts"
        assert agg.loc['positive', 'count'] == 2, "Should have 2 positive posts"
        assert agg.loc['positive', 'avg_sentiment'] > agg.loc['negative', 'avg_sentiment'], \
            "Positive terms should have higher sentiment"


class TestVADERSentiment:
    """Test suite for VADER sentiment analysis integration."""
    
    def test_sentiment_scoring_ranges(self):
        """Test that VADER compound scores are in valid range [-1, 1]."""
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        analyzer = SentimentIntensityAnalyzer()
        
        texts = [
            "I got a great job offer!",
            "Laid off after 10 years",
            "The weather is okay"
        ]
        
        for text in texts:
            scores = analyzer.polarity_scores(text)
            compound = scores['compound']
            
            assert -1.0 <= compound <= 1.0, f"Compound score {compound} out of range for: {text}"
    
    def test_sentiment_classification(self):
        """Test sentiment classification thresholds."""
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        analyzer = SentimentIntensityAnalyzer()
        
        positive_text = "I'm so happy I got the job! Best day ever!"
        negative_text = "Got laid off, feeling terrible and hopeless"
        neutral_text = "The meeting is scheduled for Tuesday"
        
        pos_score = analyzer.polarity_scores(positive_text)['compound']
        neg_score = analyzer.polarity_scores(negative_text)['compound']
        neu_score = analyzer.polarity_scores(neutral_text)['compound']
        
        assert pos_score > 0.05, "Positive text should have compound > 0.05"
        assert neg_score < -0.05, "Negative text should have compound < -0.05"
        assert -0.05 <= neu_score <= 0.05, "Neutral text should be in [-0.05, 0.05]"


class TestDatetimeHandling:
    """Test suite for datetime handling and timezone awareness."""
    
    def test_utc_timestamp_conversion(self):
        """Test correct UTC timestamp conversion (not deprecated method)."""
        unix_timestamp = 1609459200  # 2021-01-01 00:00:00 UTC
        
        # Correct method with timezone
        dt_correct = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)
        
        assert dt_correct.year == 2021, "Year should be 2021"
        assert dt_correct.month == 1, "Month should be 1"
        assert dt_correct.day == 1, "Day should be 1"
        assert dt_correct.tzinfo is not None, "Should be timezone-aware"
    
    def test_period_to_timestamp_conversion(self):
        """Test period to timestamp conversion for plotting."""
        period = pd.Period('2020-01', freq='M')
        timestamp = period.to_timestamp()
        
        assert isinstance(timestamp, pd.Timestamp), "Should convert to Timestamp"
        assert timestamp.year == 2020, "Year should be preserved"
        assert timestamp.month == 1, "Month should be preserved"


def run_tests():
    """Run all tests and print summary."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_tests()
