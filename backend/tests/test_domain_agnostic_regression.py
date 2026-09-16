# -*- coding: utf-8 -*-
import unittest
import numpy as np
import pandas as pd
import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.bias_engine import BiasEngine
from services.mitigator import BiasMitigator

class TestDomainAgnosticRegression(unittest.TestCase):
    def setUp(self):
        self.engine = BiasEngine()
        self.mitigator = BiasMitigator()

    def test_sbic_case_high_cardinality_fractional_attribute(self):
        # SBIC with > 10 unique fractions (e.g. 21 fractional annotation scores)
        np.random.seed(42)
        n = 35424
        fractions = np.linspace(0.0, 1.0, 21)
        sex_vals = np.random.choice(fractions, size=n)
        p_bias = np.where(sex_vals > 0.5, 0.72, 0.58)
        has_bias = np.random.binomial(1, p_bias, n)
        df = pd.DataFrame({'sexYN': sex_vals, 'hasBiasedImplication': has_bias})

        res = self.engine.analyze(df, 'hasBiasedImplication', ['sexYN'], use_predictions=False)
        metrics = res['metrics_per_attr']['sexYN']

        self.assertNotIn('error', metrics)
        group_stats = metrics['group_stats']
        self.assertEqual(len(group_stats), 2)
        for g in group_stats:
            self.assertNotIn('Younger', g)
            self.assertNotIn('Older', g)
            self.assertNotIn('<50', g)
            self.assertIn('sexYN', g)
        self.assertIsNotNone(metrics['SPD'])
        self.assertIsNotNone(metrics['DI'])
        self.assertIsNone(metrics['EOD'])
        self.assertIsNone(metrics['AOD'])

    def test_sbic_case_discrete_fractional_attribute(self):
        # SBIC with <= 10 unique values
        np.random.seed(42)
        n = 5000
        sex_vals = np.random.choice([0.0, 0.333333, 0.5, 0.666667, 1.0], size=n)
        has_bias = np.random.binomial(1, 0.65, n)
        df = pd.DataFrame({'sexYN': sex_vals, 'hasBiasedImplication': has_bias})

        res = self.engine.analyze(df, 'hasBiasedImplication', ['sexYN'], use_predictions=False)
        metrics = res['metrics_per_attr']['sexYN']

        self.assertNotIn('error', metrics)
        group_stats = metrics['group_stats']
        self.assertEqual(len(group_stats), 5)
        for g in group_stats:
            self.assertNotIn('Younger', g)
            self.assertNotIn('Older', g)
        self.assertIsNotNone(metrics['SPD'])
        self.assertIsNotNone(metrics['DI'])

    def test_unrelated_continuous_income_dataset(self):
        np.random.seed(123)
        n = 1000
        incomes = np.random.uniform(20000, 120000, n).round(2)
        approved = np.where(incomes > 65000, np.random.binomial(1, 0.7, n), np.random.binomial(1, 0.4, n))
        df = pd.DataFrame({'income': incomes, 'approved': approved})

        res = self.engine.analyze(df, 'approved', ['income'], use_predictions=False)
        metrics = res['metrics_per_attr']['income']

        self.assertNotIn('error', metrics)
        self.assertEqual(len(metrics['group_stats']), 2)
        for g in metrics['group_stats']:
            self.assertIn('income', g)
            self.assertNotIn('Younger', g)
            self.assertNotIn('Older', g)
        self.assertIsNotNone(metrics['SPD'])
        self.assertIsNotNone(metrics['DI'])

    def test_unrelated_credit_score_dataset(self):
        np.random.seed(999)
        n = 1500
        scores = np.random.randint(300, 851, n).astype(float)
        default = np.where(scores < 600, np.random.binomial(1, 0.6, n), np.random.binomial(1, 0.2, n))
        df = pd.DataFrame({'credit_score': scores, 'default': default})

        res = self.engine.analyze(df, 'default', ['credit_score'], use_predictions=False)
        metrics = res['metrics_per_attr']['credit_score']

        self.assertNotIn('error', metrics)
        self.assertEqual(len(metrics['group_stats']), 2)
        for g in metrics['group_stats']:
            self.assertIn('credit_score', g)
        self.assertIsNotNone(metrics['SPD'])
        self.assertIsNotNone(metrics['DI'])

    def test_mitigator_uses_consistent_binning_on_sbic(self):
        np.random.seed(42)
        n = 500
        sex_vals = np.random.choice(np.linspace(0.0, 1.0, 15), size=n)
        has_bias = np.random.binomial(1, 0.65, n)
        df = pd.DataFrame({'sexYN': sex_vals, 'hasBiasedImplication': has_bias, 'f1': np.random.randn(n)})

        res = self.mitigator.run_both(df, 'hasBiasedImplication', 'sexYN', allow_simulation=True)
        reweigh = res['reweigh']

        self.assertNotIn('error', reweigh)
        self.assertIsNotNone(reweigh['before']['SPD'])
        self.assertIsNotNone(reweigh['after']['SPD'])
        for g in reweigh['after']['group_stats']:
            self.assertIn('sexYN', g)
            self.assertNotIn('Younger', g)
            self.assertNotIn('Older', g)

    def test_zero_variance_handled_gracefully(self):
        df = pd.DataFrame({'constant_col': [1.0]*100, 'target': [0, 1]*50})
        res = self.engine.analyze(df, 'target', ['constant_col'], use_predictions=False)
        self.assertIn('error', res['metrics_per_attr']['constant_col'])

if __name__ == '__main__':
    unittest.main()
