# -*- coding: utf-8 -*-
import io
import os
import pickle
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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

PASCAL_COLS_20 = [
    'CheckingStatus', 'LoanDuration', 'CreditHistory', 'LoanPurpose',
    'LoanAmount', 'ExistingSavings', 'EmploymentDuration',
    'InstallmentPercent', 'Sex', 'OthersOnLoan',
    'CurrentResidenceDuration', 'OwnsProperty', 'Age',
    'InstallmentPlans', 'Housing', 'ExistingCreditsCount', 'Job',
    'Dependents', 'Telephone', 'ForeignWorker'
]

def generate_synthetic_german_credit_df(n=150, seed=42):
    rng = np.random.RandomState(seed)
    data = {col: rng.randint(0, 5, size=n) for col in PASCAL_COLS_20}
    data['Telephone'] = rng.choice(['none', 'yes'], size=n)
    data['Risk'] = np.where((data['Sex'] > 2) | (data['LoanAmount'] > 2), 1, 0)
    return pd.DataFrame(data)

def train_synthetic_model(df_raw):
    X_train = df_raw[PASCAL_COLS_20].copy()
    X_train['Telephone'] = (X_train['Telephone'] == 'yes').astype(int)
    model = LogisticRegression(max_iter=300, random_state=42)
    model.fit(X_train, df_raw['Risk'])
    return model

class TestFeatureResolverUnit(unittest.TestCase):
    def test_1_normalize_and_canonical_helpers(self):
        self.assertEqual(normalize_column_name('CheckingStatus'), 'checkingstatus')
        self.assertEqual(normalize_column_name('Loan Duration (Months)'), 'loan_duration_months')
        self.assertEqual(canonical_alphanumeric('Loan-Duration_Months'), 'loandurationmonths')
        self.assertEqual(canonical_alphanumeric('Telephone'), 'telephone')
        self.assertEqual(canonical_alphanumeric(chr(65279) + 'CheckingStatus'), 'checkingstatus')

    def test_2_model_feature_ordering_preserved(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6], 'risk': [0, 1]})
        model = LogisticRegression()
        model.feature_names_in_ = np.array(['C', 'A', 'B'], dtype=object)

        X = build_model_feature_matrix(model, df, target_col='risk')
        self.assertListEqual(list(X.columns), ['C', 'A', 'B'])
        self.assertListEqual(list(X['C'].values), [5, 6])
        self.assertListEqual(list(X['A'].values), [1, 2])
        self.assertListEqual(list(X['B'].values), [3, 4])

    def test_3_already_lowercase_model_succeeds(self):
        cols = ['feat_alpha', 'feat_beta', 'feat_gamma']
        df = pd.DataFrame({
            'feat_alpha': [1.0, 2.0],
            'feat_beta': [3.0, 4.0],
            'feat_gamma': [5.0, 6.0],
            'label': [0, 1]
        })
        model = LogisticRegression()
        model.fit(df[cols], df['label'])

        resolved, missing = resolve_model_features(model, df, target_col='label')
        self.assertEqual(missing, [])
        self.assertEqual(resolved, {'feat_alpha': 'feat_alpha', 'feat_beta': 'feat_beta', 'feat_gamma': 'feat_gamma'})

        X = build_model_feature_matrix(model, df, target_col='label')
        self.assertListEqual(list(X.columns), cols)

    def test_4_all_21_features_including_telephone_with_none_and_yes(self):
        df_raw = generate_synthetic_german_credit_df(n=120)
        model = train_synthetic_model(df_raw)

        preprocessor = DataPreprocessor()
        csv_bytes = df_raw.to_csv(index=False).encode('utf-8')
        df_norm, report = preprocessor.process(csv_bytes, 'german_credit.csv')

        self.assertIn('telephone', df_norm.columns)
        self.assertIn('checkingstatus', df_norm.columns)
        self.assertIn('CheckingStatus', report['column_mapping'])
        self.assertIn('Telephone', report['column_mapping'])

        resolved_map, missing = resolve_model_features(
            model=model,
            df=df_norm,
            target_col='risk',
            column_mapping=report['column_mapping'],
        )
        self.assertEqual(missing, [], f'Expected zero missing features, got: {missing}')
        self.assertEqual(resolved_map['Telephone'], 'telephone')

        X = build_model_feature_matrix(
            model=model,
            df=df_norm,
            target_col='risk',
            sensitive_attr='sex',
            column_mapping=report['column_mapping'],
            dropped_cols=report.get('dropped_column_values'),
        )
        self.assertListEqual(list(X.columns), PASCAL_COLS_20)
        self.assertEqual(len(X), len(df_norm))

    def test_5_truly_missing_features_raise_value_error(self):
        df = pd.DataFrame({'feat_a': [1, 2], 'feat_b': [3, 4], 'label': [0, 1]})
        model = LogisticRegression()
        model.feature_names_in_ = np.array(['feat_a', 'feat_b', 'feat_completely_missing'], dtype=object)

        with self.assertRaises(ValueError) as ctx:
            build_model_feature_matrix(model, df, target_col='label')
        self.assertIn('feat_completely_missing', str(ctx.exception))

    def test_6_ambiguous_collision_handling(self):
        df = pd.DataFrame({'account_status': [1, 2], 'label': [0, 1]})
        model = LogisticRegression()
        model.feature_names_in_ = np.array(['Account_Status', 'accountstatus'], dtype=object)

        with self.assertRaises(ValueError) as ctx:
            build_model_feature_matrix(model, df, target_col='label')
        self.assertIn('Ambiguous feature collision', str(ctx.exception))

    def test_7_target_and_internal_columns_excluded(self):
        df = pd.DataFrame({
            'feat_1': [1, 2],
            'risk': [0, 1],
            '__predictions__': [0, 1],
            '__sens_binned__': ['a', 'b'],
        })
        model = LogisticRegression()
        model.feature_names_in_ = np.array(['feat_1', 'Risk'], dtype=object)

        mapping, missing = resolve_model_features(
            model=model,
            df=df,
            target_col='risk',
            column_mapping={'Risk': 'risk'},
        )
        self.assertIn('Risk', missing)
        self.assertEqual(mapping, {'feat_1': 'feat_1'})

    def test_8_fuzzy_or_spelling_variation_resolved(self):
        df = pd.DataFrame({'telephon': [1, 0], 'age': [25, 30], 'risk': [0, 1]})
        model = LogisticRegression()
        model.feature_names_in_ = np.array(['Telephone', 'Age'], dtype=object)

        resolved, missing = resolve_model_features(model, df, target_col='risk')
        self.assertEqual(missing, [])
        self.assertEqual(resolved['Telephone'], 'telephon')
        self.assertEqual(resolved['Age'], 'age')

    def test_9_zero_variance_dropped_column_safe_recovery(self):
        df = pd.DataFrame({'feat_a': [1, 2], 'risk': [0, 1]})
        model = LogisticRegression()
        model.feature_names_in_ = np.array(['feat_a', 'ConstantCol'], dtype=object)

        X = build_model_feature_matrix(
            model=model,
            df=df,
            target_col='risk',
            dropped_cols={'constantcol': 1},
        )
        self.assertListEqual(list(X.columns), ['feat_a', 'ConstantCol'])
        self.assertListEqual(list(X['ConstantCol'].values), [1, 1])

class TestEndToEndFeatureCompatibilityFlow(unittest.TestCase):
    def test_full_pipeline_with_all_21_synthetic_features(self):
        df_raw = generate_synthetic_german_credit_df(n=150)
        model = train_synthetic_model(df_raw)

        preprocessor = DataPreprocessor()
        csv_bytes = df_raw.to_csv(index=False).encode('utf-8')
        df_norm, report = preprocessor.process(csv_bytes, 'german_credit.csv')

        column_mapping = report['column_mapping']
        dropped_cols = report.get('dropped_column_values')
        X = build_model_feature_matrix(
            model=model,
            df=df_norm,
            target_col='risk',
            sensitive_attr='sex',
            column_mapping=column_mapping,
            dropped_cols=dropped_cols,
        )
        self.assertListEqual(list(X.columns), PASCAL_COLS_20)

        predictions = model.predict(X)
        df_norm['__predictions__'] = predictions

        engine = BiasEngine()
        bias_res = engine.analyze(
            df=df_norm,
            target_col='risk',
            sensitive_attrs=['sex'],
            use_predictions=True,
        )
        metrics = bias_res['metrics_per_attr']['sex']
        self.assertNotIn('error', metrics)
        self.assertEqual(bias_res['metrics_mode'], 'model_level')
        self.assertTrue(metrics.get('eod_available'))
        self.assertTrue(metrics.get('aod_available'))

        mitigator = BiasMitigator()
        mit_res = mitigator.run_both(
            df=df_norm,
            target_col='risk',
            sensitive_attr='sex',
            model=model,
            df_with_pred=df_norm,
            allow_simulation=False,
            column_mapping=column_mapping,
            dropped_cols=dropped_cols,
        )
        thr = mit_res.get('threshold', {})
        self.assertFalse(thr.get('is_simulation', True))
        self.assertTrue(thr.get('has_real_model', False))
        self.assertIsNotNone(thr.get('after'))
        self.assertIn('accuracy', thr['after'])

class TestApiFeatureCompatibilityEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from main import app
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def test_api_analyze_and_mitigate_pascalcase_model(self):
        df_raw = generate_synthetic_german_credit_df(n=120)
        model = train_synthetic_model(df_raw)

        csv_buf = io.BytesIO()
        df_raw.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        res_up = self.client.post('/api/upload', files={'file': ('credit.csv', csv_buf, 'text/csv')})
        self.assertIn(res_up.status_code, (200, 201), f'Upload failed: {res_up.text}')
        session_id = res_up.json()['session_id']

        pkl_buf = io.BytesIO()
        pickle.dump(model, pkl_buf)
        pkl_buf.seek(0)
        res_mod = self.client.post(f'/api/upload-model?session_id={session_id}', files={'file': ('model.pkl', pkl_buf, 'application/octet-stream')})
        self.assertIn(res_mod.status_code, (200, 201), f'Upload model failed: {res_mod.text}')

        res_an = self.client.post('/api/analyze', json={
            'session_id': session_id,
            'target_col': 'Risk',
            'sensitive_attrs': ['Sex']
        })
        self.assertEqual(res_an.status_code, 200, f'Analyze failed: {res_an.text}')
        an_json = res_an.json()
        metrics = an_json['metrics_per_attr']['sex']
        self.assertEqual(metrics['metrics_mode'], 'model_level')
        self.assertTrue(metrics['eod_available'])
        self.assertTrue(metrics['aod_available'])

        res_mit = self.client.post('/api/mitigate', json={
            'session_id': session_id,
            'target_col': 'Risk',
            'sensitive_attr': 'Sex',
            'simulate_threshold': True
        })
        self.assertEqual(res_mit.status_code, 200, f'Mitigate failed: {res_mit.text}')
        mit_json = res_mit.json()
        thr = mit_json['threshold']
        self.assertTrue(thr['has_real_model'])
        self.assertFalse(thr['is_simulation'])
        self.assertIn('accuracy', thr['after'])

    def test_api_truly_missing_feature_returns_422(self):
        df = pd.DataFrame({
            'feat_1': np.random.randn(120),
            'feat_2': np.random.randn(120),
            'label': np.random.choice([0, 1], 120),
            'sens': np.random.choice([0, 1], 120),
        })
        model = LogisticRegression()
        model.feature_names_in_ = np.array(['feat_1', 'feat_2', 'truly_missing_feature'])

        csv_buf = io.BytesIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        res_up = self.client.post('/api/upload', files={'file': ('data.csv', csv_buf, 'text/csv')})
        session_id = res_up.json()['session_id']

        pkl_buf = io.BytesIO()
        pickle.dump(model, pkl_buf)
        pkl_buf.seek(0)
        self.client.post(f'/api/upload-model?session_id={session_id}', files={'file': ('model.pkl', pkl_buf, 'application/octet-stream')})

        res_an = self.client.post('/api/analyze', json={
            'session_id': session_id,
            'target_col': 'label',
            'sensitive_attrs': ['sens']
        })
        self.assertEqual(res_an.status_code, 422)
        self.assertIn('truly_missing_feature', str(res_an.json()['detail']))

    def test_api_ambiguous_collision_returns_422(self):
        df = pd.DataFrame({
            'feature_col': np.random.randn(120),
            'label': np.random.choice([0, 1], 120),
            'sens': np.random.choice([0, 1], 120),
        })
        model = LogisticRegression()
        model.feature_names_in_ = np.array(['Feature_Col', 'featurecol'])

        csv_buf = io.BytesIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        res_up = self.client.post('/api/upload', files={'file': ('data.csv', csv_buf, 'text/csv')})
        session_id = res_up.json()['session_id']

        pkl_buf = io.BytesIO()
        pickle.dump(model, pkl_buf)
        pkl_buf.seek(0)
        self.client.post(f'/api/upload-model?session_id={session_id}', files={'file': ('model.pkl', pkl_buf, 'application/octet-stream')})

        res_an = self.client.post('/api/analyze', json={
            'session_id': session_id,
            'target_col': 'label',
            'sensitive_attrs': ['sens']
        })
        self.assertEqual(res_an.status_code, 422)
        self.assertIn('Ambiguous feature collision', str(res_an.json()['detail']))

    def test_api_already_lowercase_model_succeeds(self):
        cols = ['feature_a', 'feature_b']
        df = pd.DataFrame({
            'feature_a': np.random.randn(120),
            'feature_b': np.random.randn(120),
            'label': np.random.choice([0, 1], 120),
            'sens': np.random.choice([0, 1], 120),
        })
        model = LogisticRegression()
        model.fit(df[cols], df['label'])

        csv_buf = io.BytesIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        res_up = self.client.post('/api/upload', files={'file': ('data.csv', csv_buf, 'text/csv')})
        session_id = res_up.json()['session_id']

        pkl_buf = io.BytesIO()
        pickle.dump(model, pkl_buf)
        pkl_buf.seek(0)
        self.client.post(f'/api/upload-model?session_id={session_id}', files={'file': ('model.pkl', pkl_buf, 'application/octet-stream')})

        res_an = self.client.post('/api/analyze', json={
            'session_id': session_id,
            'target_col': 'label',
            'sensitive_attrs': ['sens']
        })
        self.assertEqual(res_an.status_code, 200)
        self.assertEqual(res_an.json()['metrics_per_attr']['sens']['metrics_mode'], 'model_level')



class TestProductionIssuesRegression(unittest.TestCase):
    """
    Focused regression suite verifying the 3 production issues:
    1. BiasExplainer and /api/explain handle object/categorical columns without agg/mean errors.
    2. Threshold adjustment with sklearn Pipelines / ColumnTransformers receiving DataFrames
       and surviving cross-version unpickling without 'Specifying columns using strings' error.
    3. Mitigation metrics before/after integrity for chart consumption.
    """

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from main import app
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def test_issue1_explainer_categorical_object_columns_unit(self):
        """BiasExplainer must safely handle object/string target and categorical columns without mean() error."""
        from services.explainer import BiasExplainer

        df = pd.DataFrame({
            "Sex": ["male", "female", "male", "female", "male", "female"],
            "Risk": ["good", "bad", "good", "good", "bad", "good"],
            "Telephone": ["none", "yes", "none", "none", "yes", "yes"],
            "Age": [25, 40, 32, 29, 55, 41]
        })
        explainer = BiasExplainer()

        # Case A: with group_stats
        metrics = {
            "SPD": -0.15,
            "DI": 0.75,
            "group_stats": {
                "male": {"positive_rate": 0.6667, "count": 3},
                "female": {"positive_rate": 0.5, "count": 3}
            }
        }
        res_a = explainer.explain(df, target_col="Risk", sensitive_attr="Sex", metrics=metrics)
        self.assertIn("plain_reason", res_a)
        self.assertEqual(res_a["sensitive_attr"], "Sex")
        self.assertEqual(res_a["target_col"], "Risk")

        # Case B: without metrics (raw fallback)
        res_b = explainer.explain(df, target_col="Risk", sensitive_attr="Sex", metrics={})
        self.assertIn("plain_reason", res_b)
        self.assertIsInstance(res_b["historical_skew"], float)

    def test_issue1_api_explain_endpoint_categorical(self):
        """/api/explain endpoint must return HTTP 200 with explanation for object-dtype dataset."""
        df = pd.DataFrame({
            "sex": ["male", "female"] * 50,
            "risk": ["good", "bad", "good", "good", "bad"] * 20,
            "loan_amount": np.random.randint(100, 1000, 100),
            "telephone": ["none", "yes"] * 50
        })
        csv_buf = io.BytesIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        res_up = self.client.post("/api/upload", files={"file": ("explain_test.csv", csv_buf, "text/csv")})
        self.assertIn(res_up.status_code, (200, 201))
        session_id = res_up.json()["session_id"]

        # Run analyze first
        res_an = self.client.post("/api/analyze", json={
            "session_id": session_id,
            "target_col": "risk",
            "sensitive_attrs": ["sex"]
        })
        self.assertEqual(res_an.status_code, 200)

        # Call /api/explain
        res_exp = self.client.post("/api/explain", json={
            "session_id": session_id,
            "target_col": "risk",
            "sensitive_attr": "sex"
        })
        self.assertEqual(res_exp.status_code, 200, f"Explain failed: {res_exp.text}")
        exp_json = res_exp.json()
        self.assertIn("plain_reason", exp_json)
        self.assertIn("data_imbalance", exp_json)

    def test_issue2_threshold_adjustment_with_pipeline_real_model(self):
        """
        Threshold adjustment with a ColumnTransformer Pipeline must:
        1. Receive a valid pandas DataFrame with exact string columns.
        2. Successfully predict_proba even when cross-version unpickling stripped multi_class.
        3. Never crash with 'Specifying the columns using strings is only supported for dataframes'.
        4. Set is_simulation=False and compute genuine performance metrics.
        """
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        n = 120
        df_raw = pd.DataFrame({
            "CheckingStatus": np.random.randint(0, 5, n),
            "LoanDuration": np.random.randint(10, 50, n),
            "Age": np.random.randint(20, 65, n),
            "Telephone": np.random.choice(["none", "yes"], n),
            "Sex": np.random.choice([0, 1], n),
            "Risk": np.random.choice([0, 1], n),
        })

        prep = ColumnTransformer([
            ("num", StandardScaler(), ["LoanDuration", "Age"]),
        ], remainder="drop")
        clf = LogisticRegression(max_iter=300)
        pipe = Pipeline([("prep", prep), ("clf", clf)])

        # Train pipeline
        pipe.fit(df_raw[["LoanDuration", "Age"]], df_raw["Risk"])

        # Simulate unpickling from older sklearn where multi_class is missing
        if hasattr(clf, "multi_class"):
            del clf.multi_class

        # Upload and analyze via API
        csv_buf = io.BytesIO()
        df_raw.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        res_up = self.client.post("/api/upload", files={"file": ("pipe_credit.csv", csv_buf, "text/csv")})
        self.assertIn(res_up.status_code, (200, 201))
        session_id = res_up.json()["session_id"]

        pkl_buf = io.BytesIO()
        pickle.dump(pipe, pkl_buf)
        pkl_buf.seek(0)
        res_mod = self.client.post(f"/api/upload-model?session_id={session_id}", files={"file": ("pipe.pkl", pkl_buf, "application/octet-stream")})
        self.assertIn(res_mod.status_code, (200, 201))

        res_an = self.client.post("/api/analyze", json={
            "session_id": session_id,
            "target_col": "risk",
            "sensitive_attrs": ["sex"]
        })
        self.assertEqual(res_an.status_code, 200)

        # Call /api/mitigate
        res_mit = self.client.post("/api/mitigate", json={
            "session_id": session_id,
            "target_col": "risk",
            "sensitive_attr": "sex",
            "simulate_threshold": True
        })
        self.assertEqual(res_mit.status_code, 200, f"Mitigate failed: {res_mit.text}")
        mit_json = res_mit.json()
        thr = mit_json["threshold"]
        self.assertTrue(thr.get("has_real_model"))
        self.assertFalse(thr.get("is_simulation"))
        self.assertIsNotNone(thr.get("after"))
        self.assertIn("accuracy", thr["after"])
        self.assertNotIn("Specifying the columns using strings", str(mit_json))

    def test_issue3_graph_metric_integrity_between_analyze_and_mitigate(self):
        """
        Ensures the backend mitigation response provides honest, unadulterated before metrics
        matching the analyze baseline exactly so the frontend chart can display them without fabrication.
        """
        df = pd.DataFrame({
            "sex": [0, 1] * 60,
            "risk": [1, 0, 1, 1, 0, 1] * 20,
            "feature": np.random.randn(120),
        })
        csv_buf = io.BytesIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        res_up = self.client.post("/api/upload", files={"file": ("metric_check.csv", csv_buf, "text/csv")})
        session_id = res_up.json()["session_id"]

        res_an = self.client.post("/api/analyze", json={
            "session_id": session_id,
            "target_col": "risk",
            "sensitive_attrs": ["sex"]
        })
        an_metrics = res_an.json()["metrics_per_attr"]["sex"]
        expected_spd = an_metrics["SPD"]
        expected_di = an_metrics["DI"]

        res_mit = self.client.post("/api/mitigate", json={
            "session_id": session_id,
            "target_col": "risk",
            "sensitive_attr": "sex",
            "simulate_threshold": True
        })
        mit_json = res_mit.json()
        rew_before = mit_json["reweigh"]["before"]
        thr_before = mit_json["threshold"]["before"]

        self.assertAlmostEqual(rew_before["SPD"], expected_spd, places=3)
        self.assertAlmostEqual(rew_before["DI"], expected_di, places=3)
        self.assertAlmostEqual(thr_before["SPD"], expected_spd, places=3)
        self.assertAlmostEqual(thr_before["DI"], expected_di, places=3)


if __name__ == '__main__':
    unittest.main()
