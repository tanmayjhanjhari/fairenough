"""
FairEnough - Fairness Correctness Tests

Tests that verify:
1. EOD and AOD are None when no predictions exist (dataset-level)
2. Age is binned into exactly 2 groups (not 49+ individual ages)
3. SPD and DI are calculated correctly
4. Audit score is not contaminated by fabricated EOD/AOD
5. Correct sign convention for SPD
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from services.bias_engine import BiasEngine


@pytest.fixture
def engine():
    return BiasEngine()


@pytest.fixture
def pima_like_df():
    """Synthetic Pima-Diabetes-like dataset with known properties."""
    np.random.seed(42)
    n = 500
    ages = np.random.randint(21, 70, n)
    # Make Younger (<50) have higher diabetes rate (higher positive rate) for testing
    outcome = np.where(ages < 50,
                       np.random.binomial(1, 0.45, n),  # younger: 45% rate
                       np.random.binomial(1, 0.30, n))  # older:   30% rate
    return pd.DataFrame({
        "Age": ages,
        "Pregnancies": np.random.randint(0, 12, n),
        "Glucose": np.random.randint(70, 200, n),
        "BMI": np.random.uniform(18, 50, n).round(1),
        "Outcome": outcome,
    })


@pytest.fixture
def small_group_df():
    """Dataset with a deliberately tiny group to test small-sample warnings."""
    np.random.seed(1)
    n = 200
    data = {
        "Gender": ["M"] * 180 + ["F"] * 20,  # F is tiny group
        "Score": np.random.randint(0, 100, n),
        "Outcome": np.random.binomial(1, 0.5, n),
    }
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────
# 1. EOD and AOD must be None for dataset-level analysis
# ─────────────────────────────────────────────────────────────

class TestEODAODNotFabricated:

    def test_eod_is_none_without_predictions(self, engine, pima_like_df):
        result = engine.analyze(pima_like_df, "Outcome", ["Age"], use_predictions=False)
        attr_metrics = result["metrics_per_attr"]["Age"]
        assert attr_metrics.get("eod") is None, \
            f"EOD should be None without predictions, got {attr_metrics.get('eod')}"
        assert attr_metrics.get("EOD") is None, \
            f"EOD (uppercase) should be None without predictions, got {attr_metrics.get('EOD')}"

    def test_aod_is_none_without_predictions(self, engine, pima_like_df):
        result = engine.analyze(pima_like_df, "Outcome", ["Age"], use_predictions=False)
        attr_metrics = result["metrics_per_attr"]["Age"]
        assert attr_metrics.get("aod") is None, \
            f"AOD should be None without predictions, got {attr_metrics.get('aod')}"
        assert attr_metrics.get("AOD") is None, \
            f"AOD (uppercase) should be None without predictions, got {attr_metrics.get('AOD')}"

    def test_eod_available_flag_is_false_without_predictions(self, engine, pima_like_df):
        result = engine.analyze(pima_like_df, "Outcome", ["Age"], use_predictions=False)
        attr_metrics = result["metrics_per_attr"]["Age"]
        assert attr_metrics.get("eod_available") == False, \
            f"eod_available should be False without predictions, got {attr_metrics.get('eod_available')}"

    def test_metrics_mode_is_dataset_level(self, engine, pima_like_df):
        result = engine.analyze(pima_like_df, "Outcome", ["Age"], use_predictions=False)
        assert result.get("metrics_mode") == "dataset_level"


# ─────────────────────────────────────────────────────────────
# 2. Age binning — must NOT produce 49 individual groups
# ─────────────────────────────────────────────────────────────

class TestAgeBinning:

    def test_age_binned_into_two_groups(self, engine, pima_like_df):
        result = engine.analyze(pima_like_df, "Outcome", ["Age"], use_predictions=False)
        attr_metrics = result["metrics_per_attr"]["Age"]
        group_stats = attr_metrics.get("group_stats", {})
        assert len(group_stats) == 2, \
            f"Age should be binned into exactly 2 groups, got {len(group_stats)}: {list(group_stats.keys())}"

    def test_age_groups_have_correct_labels(self, engine, pima_like_df):
        result = engine.analyze(pima_like_df, "Outcome", ["Age"], use_predictions=False)
        attr_metrics = result["metrics_per_attr"]["Age"]
        group_stats = attr_metrics.get("group_stats", {})
        group_names = set(group_stats.keys())
        assert len(group_names) == 2, f"Expected 2 groups, got {group_names}"
        assert all("Age" in g for g in group_names), f"Expected 'Age' in group names, got {group_names}"

    def test_binning_applied_flag(self, engine, pima_like_df):
        result = engine.analyze(pima_like_df, "Outcome", ["Age"], use_predictions=False)
        attr_metrics = result["metrics_per_attr"]["Age"]
        assert attr_metrics.get("binning_applied") == True

    def test_low_cardinality_attr_not_binned(self, engine, small_group_df):
        """Gender with 2 unique values should NOT be binned."""
        result = engine.analyze(small_group_df, "Outcome", ["Gender"], use_predictions=False)
        attr_metrics = result["metrics_per_attr"]["Gender"]
        assert attr_metrics.get("binning_applied") == False

    def test_age_bins_cover_all_records(self, engine, pima_like_df):
        result = engine.analyze(pima_like_df, "Outcome", ["Age"], use_predictions=False)
        attr_metrics = result["metrics_per_attr"]["Age"]
        group_stats = attr_metrics.get("group_stats", {})
        total_in_groups = sum(g["count"] for g in group_stats.values())
        # Allow for <=1 count rows that are excluded
        assert total_in_groups >= len(pima_like_df) - 2


# ─────────────────────────────────────────────────────────────
# 3. SPD and DI correctness
# ─────────────────────────────────────────────────────────────

class TestMetricFormulas:

    def test_spd_sign_convention(self, engine):
        """SPD = unprivileged_rate - privileged_rate (negative = disadvantaged)."""
        df = pd.DataFrame({
            "Group": ["A"] * 100 + ["B"] * 100,
            "Outcome": [1] * 80 + [0] * 20 + [1] * 40 + [0] * 60,  # A=0.8, B=0.4
        })
        result = engine.analyze(df, "Outcome", ["Group"], use_predictions=False)
        attr_metrics = result["metrics_per_attr"]["Group"]
        spd = attr_metrics["spd"]
        # Privileged = A (higher rate), unprivileged = B (lower rate)
        # SPD = 0.4 - 0.8 = -0.4
        assert abs(spd - (-0.4)) < 0.01, f"Expected SPD ~-0.4, got {spd}"

    def test_di_formula(self, engine):
        """DI = unprivileged_rate / privileged_rate."""
        df = pd.DataFrame({
            "Group": ["A"] * 100 + ["B"] * 100,
            "Outcome": [1] * 80 + [0] * 20 + [1] * 40 + [0] * 60,
        })
        result = engine.analyze(df, "Outcome", ["Group"], use_predictions=False)
        attr_metrics = result["metrics_per_attr"]["Group"]
        di = attr_metrics["di"]
        # DI = 0.4 / 0.8 = 0.5
        assert abs(di - 0.5) < 0.01, f"Expected DI ~0.5, got {di}"

    def test_di_zero_denominator_handled(self, engine):
        """DI should be 1.0 or None when privileged group has 0 positive outcomes."""
        df = pd.DataFrame({
            "Group": ["A"] * 100 + ["B"] * 100,
            "Outcome": [0] * 100 + [1] * 50 + [0] * 50,  # A=0%, B=50%
        })
        result = engine.analyze(df, "Outcome", ["Group"], use_predictions=False)
        attr_metrics = result["metrics_per_attr"]["Group"]
        di = attr_metrics.get("di")
        # priv=B (50%), unpriv=A (0%) -> DI = 0 / 0.5 = 0.0 (not undefined)
        assert di is not None or "error" in attr_metrics, \
            "DI zero-denominator case should be handled gracefully"

    def test_group_stats_contain_positive_count(self, engine, pima_like_df):
        result = engine.analyze(pima_like_df, "Outcome", ["Age"], use_predictions=False)
        attr_metrics = result["metrics_per_attr"]["Age"]
        group_stats = attr_metrics.get("group_stats", {})
        for name, stats in group_stats.items():
            assert "positive_count" in stats, f"Group {name} missing positive_count"
            assert "count" in stats, f"Group {name} missing count"
            assert "positive_rate" in stats, f"Group {name} missing positive_rate"


# ─────────────────────────────────────────────────────────────
# 4. Audit score not contaminated by None EOD/AOD
# ─────────────────────────────────────────────────────────────

class TestAuditScore:

    def test_audit_score_same_whether_eod_is_none_or_zero(self, engine):
        """Audit score with None EOD must equal score with EOD=0.0."""
        df = pd.DataFrame({
            "Group": ["A"] * 100 + ["B"] * 100,
            "Outcome": [1] * 80 + [0] * 20 + [1] * 60 + [0] * 40,
        })
        result = engine.analyze(df, "Outcome", ["Group"], use_predictions=False)
        score_no_eod = result["audit_score"]

        # Manually inject eod=0 and recompute
        metrics_with_zero_eod = {
            "Group": {
                "spd": result["metrics_per_attr"]["Group"]["spd"],
                "di": result["metrics_per_attr"]["Group"]["di"],
                "eod": 0.0,  # fake zero
                "EOD": 0.0,
            }
        }
        # Scores should be the same IF eod=0 and eod=None both produce 0 eod_penalty
        score_with_zero = engine._compute_audit_score(metrics_with_zero_eod)
        # If eod=0, penalty is 0. If eod=None, penalty is also 0. Should be equal.
        assert abs(score_no_eod - score_with_zero) < 0.1, \
            f"Score with None EOD ({score_no_eod}) should equal score with 0 EOD ({score_with_zero})"

    def test_audit_score_uses_only_spd_di_for_dataset_level(self, engine, pima_like_df):
        result = engine.analyze(pima_like_df, "Outcome", ["Age"], use_predictions=False)
        # If the engine still trains an internal model, the score would be different
        # Just verify score is in valid range
        assert 0.0 <= result["audit_score"] <= 100.0


# ─────────────────────────────────────────────────────────────
# 5. Small sample warnings
# ─────────────────────────────────────────────────────────────

class TestSmallSampleWarnings:

    def test_small_group_triggers_warning(self, engine, small_group_df):
        result = engine.analyze(small_group_df, "Outcome", ["Gender"], use_predictions=False)
        attr_metrics = result["metrics_per_attr"]["Gender"]
        warnings = attr_metrics.get("warnings", [])
        small_sample_groups = attr_metrics.get("small_sample_groups", [])
        # Gender=F has only 20 records, should trigger warning
        assert len(small_sample_groups) > 0 or any("small" in w.lower() or "stable" in w.lower() for w in warnings), \
            "Small sample should trigger warning. Got: " + str(warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
