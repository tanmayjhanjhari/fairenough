"""
FairEnough - Feature Name Compatibility Regression Tests

Verifies that models trained on PascalCase, CamelCase, or case-sensitive
column names (such as the German Credit dataset's 'CheckingStatus', 'LoanDuration', 'Sex')
seamlessly work with FairEnough's normalized lowercase dataset columns across:
1. Preprocessing normalization and column_mapping preservation
2. Feature resolver service (resolve_model_features, build_model_feature_matrix)
3. German Credit scenario: PascalCase model feature_names_in_ evaluated against lowercase dataset
4. Model prediction flow producing predictions for BiasEngine (EOD, AOD computation)
5. Mitigation flow executing real model threshold adjustment (is_simulation=False, has_real_model=True)
6. Second distinct domain-agnostic scenario (HR Hiring / Candidate evaluation)
7. Missing feature detection (clear descriptive ValueError)
8. Ambiguous feature collision detection
9. Target column and internal column exclusion
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.feature_resolver import (
    resolve_model_features,
    build_model_feature_matrix,
    normalize_column_name,
    canonical_alphanumeric,
)
from services.preprocessor import DataPreprocessor
from services.bias_engine import BiasEngine
from services.mitigator import BiasMitigator


class TestFeatureResolverUnit(unittest.TestCase):
    """Unit tests for feature_resolver service."""

    def test_normalize_and_canonical_helpers(self):
        self.assertEqual(normalize_column_name("CheckingStatus"), "checkingstatus")
        self.assertEqual(normalize_column_name("Loan Duration (Months)"), "loan_duration_months")
        self.assertEqual(canonical_alphanumeric("Loan-Duration_Months"), "loandurationmonths")
        self.assertEqual(canonical_alphanumeric("Sex"), "sex")

    def test_exact_match(self):
        df = pd.DataFrame({"age": [25, 40], "income": [50000, 80000], "target": [0, 1]})
        model = LogisticRegression()
        X_train = df[["age", "income"]]
        model.fit(X_train, df["target"])

        mapping, missing = resolve_model_features(model, df, target_col="target")
        self.assertEqual(missing, [])
        self.assertEqual(mapping, {"age": "age", "income": "income"})

        X = build_model_feature_matrix(model, df, target_col="target")
        self.assertListEqual(list(X.columns), ["age", "income"])
        self.assertEqual(X.shape, (2, 2))

    def test_pascalcase_to_lowercase_german_credit(self):
        """Simulates German Credit where model expects PascalCase features."""
        pascal_cols = [
            "CheckingStatus", "LoanDuration", "CreditHistory", "LoanPurpose", "LoanAmount",
            "ExistingSavings", "EmploymentDuration", "InstallmentPercent", "Sex", "OthersOnLoan",
            "CurrentResidenceDuration", "OwnsProperty", "Age", "InstallmentPlans", "Housing",
            "ExistingCreditsCount", "Job", "Dependents", "Telephone", "ForeignWorker"
        ]
        np.random.seed(42)
        n = 60
        data = {col: np.random.randint(0, 5, size=n) for col in pascal_cols}
        data["Risk"] = np.random.choice([0, 1], size=n)
        df_raw = pd.DataFrame(data)

        # Train model with PascalCase column names
        model = LogisticRegression(max_iter=200)
        model.fit(df_raw[pascal_cols], df_raw["Risk"])

        # Preprocess CSV (simulating FairEnough upload normalization)
        preprocessor = DataPreprocessor()
        csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
        df_norm, report = preprocessor.process(csv_bytes, "german_credit.csv")

        # Check mapping and resolution
        column_mapping = report.get("column_mapping")
        self.assertIsNotNone(column_mapping)
        self.assertIn("CheckingStatus", column_mapping)
        self.assertEqual(column_mapping["CheckingStatus"], "checkingstatus")
        self.assertEqual(column_mapping["Sex"], "sex")

        resolved_map, missing = resolve_model_features(
            model=model,
            df=df_norm,
            target_col="risk",
            column_mapping=column_mapping,
        )
        self.assertEqual(missing, [])
        self.assertEqual(len(resolved_map), len(pascal_cols))
        self.assertEqual(resolved_map["CheckingStatus"], "checkingstatus")
        self.assertEqual(resolved_map["Sex"], "sex")
        self.assertEqual(resolved_map["Age"], "age")

        # Build feature matrix
        X = build_model_feature_matrix(
            model=model,
            df=df_norm,
            target_col="risk",
            sensitive_attr="sex",
            column_mapping=column_mapping,
        )
        self.assertListEqual(list(X.columns), pascal_cols)
        preds = model.predict(X)
        self.assertEqual(len(preds), n)

    def test_second_domain_hiring_dataset(self):
        """Domain-agnostic test with completely different schema and naming conventions."""
        raw_cols = ["Applicant_ID", "Years_Experience", "Interview_Score_Pct", "Degree_Level", "Gender", "Hired"]
        df_raw = pd.DataFrame({
            "Applicant_ID": [101, 102, 103, 104],
            "Years_Experience": [2, 5, 8, 1],
            "Interview_Score_Pct": [82.5, 91.0, 74.5, 60.0],
            "Degree_Level": ["BS", "MS", "PhD", "BS"],
            "Gender": ["Female", "Male", "Female", "Male"],
            "Hired": [1, 1, 0, 0],
        })

        train_features = ["Years_Experience", "Interview_Score_Pct", "Degree_Level", "Gender"]
        # Train model with encoded categoricals
        X_train = df_raw[train_features].copy()
        X_train["Degree_Level"] = [0, 1, 2, 0]
        X_train["Gender"] = [0, 1, 0, 1]
        model = LogisticRegression()
        model.fit(X_train, df_raw["Hired"])

        # Preprocess DataFrame
        preprocessor = DataPreprocessor()
        csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
        df_norm, report = preprocessor.process(csv_bytes, "hiring.csv")

        X = build_model_feature_matrix(
            model=model,
            df=df_norm,
            target_col="hired",
            sensitive_attr="gender",
            column_mapping=report["column_mapping"],
        )
        self.assertListEqual(list(X.columns), train_features)
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (4, 2))

    def test_missing_features_detected(self):
        """Model expecting features that are truly not in the dataset raises ValueError."""
        df = pd.DataFrame({"feat_a": [1, 2], "feat_b": [3, 4], "label": [0, 1]})
        model = LogisticRegression()
        # Mock feature_names_in_
        model.feature_names_in_ = np.array(["feat_a", "feat_b", "feat_completely_missing"], dtype=object)

        with self.assertRaises(ValueError) as ctx:
            build_model_feature_matrix(model, df, target_col="label")
        self.assertIn("feat_completely_missing", str(ctx.exception))

    def test_ambiguous_collision_handling(self):
        """When multiple model features map to the same dataset column, raises ValueError."""
        df = pd.DataFrame({"account_status": [1, 2], "label": [0, 1]})
        model = LogisticRegression()
        # Two model features that both reduce to account_status
        model.feature_names_in_ = np.array(["Account_Status", "accountstatus"], dtype=object)

        with self.assertRaises(ValueError) as ctx:
            build_model_feature_matrix(model, df, target_col="label")
        self.assertIn("Ambiguous feature collision", str(ctx.exception))

    def test_target_and_internal_columns_excluded(self):
        """Ensure target column and internal columns (__predictions__, etc.) are excluded."""
        df = pd.DataFrame({
            "feat_1": [1, 2],
            "risk": [0, 1],
            "__predictions__": [0, 1],
            "__sens_binned__": ["a", "b"],
        })
        model = LogisticRegression()
        # If model expects 'risk', it should not map it from target_col
        model.feature_names_in_ = np.array(["feat_1", "Risk"], dtype=object)

        mapping, missing = resolve_model_features(
            model=model,
            df=df,
            target_col="risk",
            column_mapping={"Risk": "risk"},
        )
        self.assertIn("Risk", missing)
        self.assertEqual(mapping, {"feat_1": "feat_1"})


class TestEndToEndFeatureCompatibilityFlow(unittest.TestCase):
    """Verifies the complete pipeline: upload -> preprocess -> model -> analyze -> mitigate."""

    def test_german_credit_analyze_and_mitigate_pipeline(self):
        """
        Full German Credit simulation:
        1. Model fitted on 20 PascalCase features.
        2. Raw CSV with PascalCase headers processed by DataPreprocessor (normalized to lowercase).
        3. Feature matrix reconstructed with exact model names and order.
        4. Model predictions added to DataFrame.
        5. BiasEngine computes SPD, DI, EOD, AOD.
        6. BiasMitigator runs threshold adjustment using real model (is_simulation=False).
        """
        pascal_cols = [
            "CheckingStatus", "LoanDuration", "CreditHistory", "LoanPurpose", "LoanAmount",
            "ExistingSavings", "EmploymentDuration", "InstallmentPercent", "Sex", "OthersOnLoan",
            "CurrentResidenceDuration", "OwnsProperty", "Age", "InstallmentPlans", "Housing",
            "ExistingCreditsCount", "Job", "Dependents", "Telephone", "ForeignWorker"
        ]
        np.random.seed(42)
        n = 200
        data = {col: np.random.randint(1, 10, size=n) for col in pascal_cols}
        # Biased label based on Sex
        data["Risk"] = np.where(np.array(data["Sex"]) > 4, 1, 0)
        df_raw = pd.DataFrame(data)

        # 1. Train model on PascalCase features
        model = LogisticRegression(max_iter=300)
        model.fit(df_raw[pascal_cols], df_raw["Risk"])

        # 2. Preprocess CSV
        preprocessor = DataPreprocessor()
        csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
        df_norm, report = preprocessor.process(csv_bytes, "german_credit.csv")

        # Verify normalization took place
        self.assertIn("checkingstatus", df_norm.columns)
        self.assertIn("sex", df_norm.columns)
        self.assertNotIn("CheckingStatus", df_norm.columns)

        # 3. Build model feature matrix
        column_mapping = report["column_mapping"]
        X = build_model_feature_matrix(
            model=model,
            df=df_norm,
            target_col="risk",
            sensitive_attr="sex",
            column_mapping=column_mapping,
        )
        self.assertListEqual(list(X.columns), pascal_cols)

        # 4. Generate predictions
        predictions = model.predict(X)
        df_norm["__predictions__"] = predictions
        self.assertEqual(len(predictions), n)

        # 5. Run BiasEngine
        engine = BiasEngine()
        bias_res = engine.analyze(
            df=df_norm,
            target_col="risk",
            sensitive_attrs=["sex"],
            use_predictions=True,
        )
        metrics = bias_res["metrics_per_attr"]["sex"]
        self.assertNotIn("error", metrics)
        self.assertIsNotNone(metrics.get("SPD"))
        self.assertIsNotNone(metrics.get("DI"))
        self.assertIsNotNone(metrics.get("EOD"))
        self.assertIsNotNone(metrics.get("AOD"))
        self.assertTrue(metrics.get("eod_available"))
        self.assertEqual(bias_res["metrics_mode"], "model_level")

        # 6. Run BiasMitigator
        mitigator = BiasMitigator()
        mit_res = mitigator.run_both(
            df=df_norm,
            target_col="risk",
            sensitive_attr="sex",
            model=model,
            df_with_pred=df_norm,
            allow_simulation=False,
            column_mapping=column_mapping,
        )

        thr = mit_res.get("threshold", {})
        self.assertFalse(thr.get("is_simulation", True), "Threshold adjustment should not be simulated when real model is provided")
        self.assertTrue(thr.get("has_real_model", False), "Threshold adjustment must acknowledge real model")
        self.assertIsNotNone(thr.get("after"), "Mitigation after metrics must be computed")
        # Real model performance metrics computed
        after_perf = thr["after"]
        self.assertIn("accuracy", after_perf)
        self.assertIn("f1", after_perf)

    def test_incompatible_model_stops_pipeline(self):
        """Incompatible model raises ValueError before prediction or mitigation."""
        df_raw = pd.DataFrame({
            "ColA": [1, 2, 3],
            "ColB": [4, 5, 6],
            "Label": [0, 1, 0],
        })
        preprocessor = DataPreprocessor()
        csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
        df_norm, report = preprocessor.process(csv_bytes, "incompat.csv")

        incompat_model = LogisticRegression()
        incompat_model.feature_names_in_ = np.array(["NonExistent1", "NonExistent2"], dtype=object)

        with self.assertRaises(ValueError) as ctx:
            build_model_feature_matrix(
                model=incompat_model,
                df=df_norm,
                target_col="label",
                column_mapping=report["column_mapping"],
            )
        self.assertIn("NonExistent1", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
