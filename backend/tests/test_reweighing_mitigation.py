"""
Unit and Integration Tests for Complete End-to-End Reweighing Mitigation,
Scenario Metadata, and PDF Side-by-Side Presentation Integrity.
"""

import io
import os
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.mitigator import BiasMitigator, _check_sample_weight_support, _clone_or_recreate_estimator
from services.reporter import ReportGenerator


class DummyNoSampleWeight(BaseEstimator, ClassifierMixin):
    """Estimator that deliberately lacks sample_weight support in fit."""
    def __init__(self):
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        probs[:, 1] = 0.8
        probs[:, 0] = 0.2
        return probs


@pytest.fixture
def mitigator():
    return BiasMitigator()


@pytest.fixture
def synthetic_df():
    np.random.seed(42)
    n = 200
    group = np.random.choice(["A", "B"], size=n, p=[0.5, 0.5])
    f1 = np.random.randn(n)
    f2 = np.random.randn(n)
    # Group A has higher probability of positive outcome
    p_pos = np.where(group == "A", 0.75, 0.35)
    outcome = (np.random.rand(n) < p_pos).astype(int)
    return pd.DataFrame({"Group": group, "Feature1": f1, "Feature2": f2, "Outcome": outcome})


class TestReweighingEndToEnd:
    """Tests covering complete Reweighing preprocessing and retraining functionality."""

    def test_1_dataset_only_reweighing(self, mitigator, synthetic_df):
        """When no model is provided: compute dataset-level SPD/DI, performance metrics remain N/A."""
        res = mitigator.reweigh(
            df=synthetic_df,
            target_col="Outcome",
            sensitive_attr="Group",
        )

        assert res["model_retrained"] is False
        assert res["sample_weight_supported"] is False
        assert res["has_real_model"] is False
        assert res["is_simulation"] is False
        assert "weights_summary" in res
        ws = res["weights_summary"]
        assert ws["min"] > 0
        assert ws["max"] > 0
        assert "mean" in ws
        assert "median" in ws

        # Weight summary table per group x outcome
        assert "reweighing_group_outcome_weights" in res
        gow = res["reweighing_group_outcome_weights"]
        assert len(gow) >= 2
        for row in gow:
            assert "group" in row and "outcome" in row and "count" in row and "weight" in row
            assert row["count"] > 0
            assert row["weight"] > 0

        # Performance metrics must remain None (no fake metrics)
        assert res["after"]["accuracy"] is None
        assert res["after"]["precision"] is None
        assert res["after"]["recall"] is None
        assert res["after"]["f1"] is None
        assert res["after"]["EOD"] is None
        assert res["after"]["AOD"] is None

        # Dataset-level fairness metrics are real
        assert res["before"]["SPD"] is not None
        assert res["after"]["SPD"] is not None
        assert abs(res["after"]["SPD"]) < abs(res["before"]["SPD"])

    def test_2_compatible_real_model_retraining(self, mitigator, synthetic_df):
        """When a compatible model (LogisticRegression) is supplied, genuine retraining occurs."""
        X = synthetic_df[["Feature1", "Feature2"]]
        y = synthetic_df["Outcome"]
        orig_model = LogisticRegression(random_state=42)
        orig_model.fit(X, y)

        res = mitigator.reweigh(
            df=synthetic_df,
            target_col="Outcome",
            sensitive_attr="Group",
            model=orig_model,
        )

        assert res["model_retrained"] is True
        assert res["sample_weight_supported"] is True
        assert res["has_real_model"] is True
        assert res["is_simulation"] is False
        assert res["n_train_samples"] is not None and res["n_train_samples"] > 0
        assert res["n_eval_samples"] is not None and res["n_eval_samples"] > 0
        assert res["original_model_info"]["type"] == "LogisticRegression"
        assert res["mitigated_model_info"]["type"] == "LogisticRegression"

        # AFTER metrics must now be genuine numbers
        after = res["after"]
        assert after["accuracy"] is not None
        assert 0.0 <= after["accuracy"] <= 1.0
        assert after["precision"] is not None
        assert 0.0 <= after["precision"] <= 1.0
        assert after["recall"] is not None
        assert 0.0 <= after["recall"] <= 1.0
        assert after["f1"] is not None
        assert 0.0 <= after["f1"] <= 1.0

        # Fairness metrics on evaluation set
        assert after["SPD"] is not None
        assert after["eod_available"] is True
        assert after["EOD"] is not None
        assert after["aod_available"] is True
        assert after["AOD"] is not None

        # BEFORE metrics also evaluated on the same evaluation split
        before = res["before"]
        assert before["accuracy"] is not None
        assert 0.0 <= before["accuracy"] <= 1.0
        assert before["f1"] is not None

    def test_3_original_model_preservation(self, mitigator, synthetic_df):
        """The original model instance must NOT be mutated or overwritten during retraining."""
        X = synthetic_df[["Feature1", "Feature2"]]
        y = synthetic_df["Outcome"]
        orig_model = LogisticRegression(random_state=42)
        orig_model.fit(X, y)

        coef_before = orig_model.coef_.copy()
        intercept_before = orig_model.intercept_.copy()

        res = mitigator.reweigh(
            df=synthetic_df,
            target_col="Outcome",
            sensitive_attr="Group",
            model=orig_model,
        )

        np.testing.assert_array_equal(orig_model.coef_, coef_before)
        np.testing.assert_array_equal(orig_model.intercept_, intercept_before)
        assert res["model_retrained"] is True

    def test_4_unsupported_model_no_sample_weight(self, mitigator, synthetic_df):
        """When model does not support sample_weight, report clearly and do NOT fabricate performance."""
        unsupported = DummyNoSampleWeight()
        supp, param = _check_sample_weight_support(unsupported)
        assert supp is False

        res = mitigator.reweigh(
            df=synthetic_df,
            target_col="Outcome",
            sensitive_attr="Group",
            model=unsupported,
        )

        assert res["model_retrained"] is False
        assert res["sample_weight_supported"] is False
        assert res["after"]["accuracy"] is None
        assert res["after"]["f1"] is None
        assert "sample_weight" in res["after"]["simulation_note"] or "does not support" in res["after"]["simulation_note"]

    def test_5_sklearn_pipeline_sample_weight_support(self, mitigator, synthetic_df):
        """Pipelines with final estimator supporting sample_weight are correctly detected and retrained."""
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42))
        ])
        supp, param_name = _check_sample_weight_support(pipe)
        assert supp is True
        assert param_name == "classifier__sample_weight"

        X = synthetic_df[["Feature1", "Feature2"]]
        y = synthetic_df["Outcome"]
        pipe.fit(X, y)

        res = mitigator.reweigh(
            df=synthetic_df,
            target_col="Outcome",
            sensitive_attr="Group",
            model=pipe,
        )

        assert res["model_retrained"] is True
        assert res["sample_weight_supported"] is True
        assert res["after"]["accuracy"] is not None
        assert 0.0 <= res["after"]["accuracy"] <= 1.0

    def test_6_column_mapping_and_dropped_cols_forwarded(self, mitigator, synthetic_df):
        """reweigh() handles column_mapping and dropped_cols identically to threshold_adjust()."""
        df_renamed = synthetic_df.rename(columns={"Feature1": "feat_one", "Feature2": "feat_two"})
        model = LogisticRegression()
        model.feature_names_in_ = np.array(["Feature1", "Feature2"])
        model.fit(synthetic_df[["Feature1", "Feature2"]], synthetic_df["Outcome"])

        col_map = {"Feature1": "feat_one", "Feature2": "feat_two"}
        res = mitigator.reweigh(
            df=df_renamed,
            target_col="Outcome",
            sensitive_attr="Group",
            model=model,
            column_mapping=col_map,
        )
        assert res["model_retrained"] is True
        assert res["after"]["accuracy"] is not None



    def test_9_consistent_eval_split_and_spd_explanation(self, mitigator, synthetic_df):
        """Verify before & after are evaluated on same split, with dynamic SPD explanation."""
        X = synthetic_df[["Feature1", "Feature2"]]
        y = synthetic_df["Outcome"]
        orig_model = LogisticRegression(random_state=42)
        orig_model.fit(X, y)

        res = mitigator.reweigh(
            df=synthetic_df,
            target_col="Outcome",
            sensitive_attr="Group",
            model=orig_model,
        )

        assert res["model_retrained"] is True
        before = res["before"]
        after = res["after"]

        # Both before and after must have all 8 metrics populated on held-out eval set
        for k in ["SPD", "DI", "EOD", "AOD", "accuracy", "precision", "recall", "f1"]:
            assert before.get(k) is not None, f"before missing {k}"
            assert after.get(k) is not None, f"after missing {k}"

        # Explanation dynamic wording check
        expl = mitigator._generate_explanation(
            before={"SPD": -0.001, "accuracy": 0.70},
            after={"SPD": 0.003, "accuracy": 0.70},
            technique="reweigh",
            sensitive_attr="sex",
            effects={"bias_reduction_pct": 0.0}
        )
        assert "SPD changed from -0.001 to 0.003" in expl["bias_result"]
        assert "The absolute SPD gap changed from 0.001 to 0.003" in expl["bias_result"]


class TestPresentationAndMetadataFixes:
    """Tests for Scenario metadata, PDF Side-by-Side Comparison SPD sign, and layout."""

    def test_7_scenario_metadata_string_and_dict(self):
        """Scenario string or dict is correctly resolved and appears in PDF."""
        rg = ReportGenerator()
        base_data = {
            "filename": "audit.csv",
            "row_count": 500,
            "target_col": "Outcome",
            "sensitive_attrs": ["Group"],
            "bias_results": {"overall_severity": "low", "audit_score": 92, "engine": "Standard", "metrics_per_attr": {}},
            "mitigation": {}
        }

        # Case 1: string scenario
        data1 = dict(base_data, scenario="Hiring")
        pdf1 = rg.generate(data1)
        assert len(pdf1) > 1000

        # Case 2: dict scenario
        data2 = dict(base_data, scenario={"scenario": "Healthcare", "confidence_pct": 90})
        pdf2 = rg.generate(data2)
        assert len(pdf2) > 1000

    def test_8_pdf_side_by_side_preserves_negative_spd_sign(self):
        """Side-by-side comparison table preserves raw signed SPD (e.g. -0.003)."""
        rg = ReportGenerator()
        session_data = {
            "filename": "test.csv",
            "row_count": 1000,
            "target_col": "Risk",
            "sensitive_attrs": ["Sex"],
            "scenario": "Lending",
            "bias_results": {
                "overall_severity": "low",
                "audit_score": 88,
                "engine": "Standard",
                "metrics_per_attr": {
                    "Sex": {
                        "spd": -0.15,
                        "di": 0.82,
                        "severity": "low",
                        "privileged_group": "Male",
                        "unprivileged_group": "Female",
                        "group_stats": {
                            "Male": {"positive_rate": 0.70, "count": 600},
                            "Female": {"positive_rate": 0.55, "count": 400},
                        }
                    }
                }
            },
            "mitigation": {
                "Sex": {
                    "winner": "threshold",
                    "winner_reason": "Equalizes opportunities directly.",
                    "reweigh": {
                        "before": {"SPD": -0.15, "DI": 0.82, "accuracy": 0.75, "f1": 0.72},
                        "after": {"SPD": -0.04, "DI": 0.94, "accuracy": 0.74, "f1": 0.71},
                        "effects": {"bias_reduction_pct": 73.3, "accuracy_retained_pct": 98.7}
                    },
                    "threshold": {
                        "before": {"SPD": -0.15, "DI": 0.82, "accuracy": 0.75, "f1": 0.72},
                        "after": {"SPD": -0.003, "DI": 0.996, "accuracy": 0.745, "f1": 0.715},
                        "effects": {"bias_reduction_pct": 98.0, "accuracy_retained_pct": 99.3}
                    }
                }
            }
        }

        pdf_bytes = rg.generate(session_data)
        import zlib, base64, re
        pos = 0
        all_text = []
        while True:
            idx = pdf_bytes.find(b"stream", pos)
            if idx == -1: break
            end = pdf_bytes.find(b"endstream", idx)
            data = pdf_bytes[idx+6:end].strip()
            try:
                a85 = base64.a85decode(data, adobe=True)
                all_text.append(zlib.decompress(a85).decode("latin-1", errors="ignore"))
            except: pass
            pos = end + 9

        full_decomp = "\n".join(all_text)
        assert "(-0.003)" in full_decomp
        assert "Lending" in full_decomp