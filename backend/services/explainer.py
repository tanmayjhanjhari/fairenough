"""
FairEnough â€” Bias Explainer Service

Analyses *why* bias exists in a dataset:
  - Correlation between the sensitive attribute and target
  - Proxy feature detection (non-sensitive columns correlated with the attribute)
  - Data imbalance across demographic groups
  - Historical skew (variance in positive rates)
  - A plain-English reason string combining all findings
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class BiasExplainer:
    """Explain the root causes of bias for a single sensitive attribute."""

    TOP_PROXY_N: int = 3
    IMBALANCE_THRESHOLD: float = 0.5

    def explain(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_attr: str,
        metrics: dict | None = None,
    ) -> dict[str, Any]:
        """
        Run all explanation analyses and return a unified dict.

        Parameters
        ----------
        df : pd.DataFrame
        target_col : str
        sensitive_attr : str

        Returns
        -------
        dict with keys:
            correlation, proxy_features, data_imbalance,
            historical_skew, positive_rate_gap, plain_reason
        """
        work = df.copy().dropna(subset=[target_col, sensitive_attr])

        # â”€â”€ 1. Correlation: sensitive_attr â†” target â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        correlation = self._correlation(work, sensitive_attr, target_col)

        # â”€â”€ 2. Proxy features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        proxy_features = self._proxy_features(work, target_col, sensitive_attr)

        # â”€â”€ 3. Data imbalance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        group_counts = work[sensitive_attr].value_counts()
        smallest = int(group_counts.min())
        largest = int(group_counts.max())
        imbalance_ratio = round(smallest / largest, 4) if largest > 0 else 1.0
        imbalance_flagged = imbalance_ratio < self.IMBALANCE_THRESHOLD

        data_imbalance = {
            "ratio": imbalance_ratio,
            "smallest_group_count": smallest,
            "largest_group_count": largest,
            "flagged": imbalance_flagged,
            "interpretation": (
                f"The smallest group has only {imbalance_ratio:.0%} as many samples "
                "as the largest group. This may cause the model to underfit minority groups."
                if imbalance_flagged
                else "Group sizes are reasonably balanced."
            ),
        }

        # â”€â”€ 4. Historical skew (std of positive rates across groups) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        positive_rates = (
            work.groupby(sensitive_attr)[target_col]
            .mean()
            .dropna()
        )
        historical_skew = round(float(positive_rates.std()), 4) if len(positive_rates) > 1 else 0.0

        # â”€â”€ 5. Positive rate gap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        positive_rate_gap = round(
            float(positive_rates.max() - positive_rates.min()), 4
        ) if len(positive_rates) > 1 else 0.0

        # â”€â”€ 6. Plain-English reason â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        plain_reason = self._build_plain_reason(
            sensitive_attr=sensitive_attr,
            target_col=target_col,
            correlation=correlation,
            proxy_features=proxy_features,
            positive_rate_gap=positive_rate_gap,
            imbalance_flagged=imbalance_flagged,
            historical_skew=historical_skew,
        )

        # --- WHY IS SPD THIS VALUE ---
        spd_val = abs(metrics.get("SPD", 0) or 0) if metrics else abs(positive_rate_gap)
        group_stats = metrics.get("group_stats", {}) if metrics else {}

        if group_stats:
            sorted_groups = sorted(group_stats.items(),
                                   key=lambda x: x[1].get("positive_rate", 0))
            worst_group_name = sorted_groups[0][0]
            worst_rate = sorted_groups[0][1].get("positive_rate", 0)
            best_group_name = sorted_groups[-1][0]
            best_rate = sorted_groups[-1][1].get("positive_rate", 0)
            worst_count = sorted_groups[0][1].get("count", 0)
            best_count = sorted_groups[-1][1].get("count", 0)
            total = sum(g[1].get("count",0) for g in sorted_groups)
        else:
            worst_group_name = "disadvantaged group"
            best_group_name = "advantaged group"
            worst_rate = 0
            best_rate = 1
            worst_count = 0
            best_count = 0
            total = 1

        gap_pct = round(abs(best_rate - worst_rate) * 100, 1)
        size_ratio = round(worst_count / max(best_count, 1), 2)
        is_imbalanced = size_ratio < 0.5

        # Build spd_explanation â€” plain English reason WHY spd is this number
        if spd_val < 0.05:
            spd_explanation = (
                f"SPD of {spd_val:.3f} is very close to 0, which means nearly ideal "
                f"fairness for '{sensitive_attr}'. The outcome rates across groups are "
                f"almost identical â€” this could mean the model genuinely treats groups "
                f"equally, OR that everyone is getting the same outcome regardless of "
                f"group (e.g. nearly everyone approved), which can mask real-world "
                f"discrimination hidden in other ways."
            )
        elif spd_val < 0.1:
            spd_explanation = (
                f"SPD of {spd_val:.3f} shows a small but real gap. "
                f"'{best_group_name}' gets positive outcomes {best_rate:.0%} of the time "
                f"vs {worst_rate:.0%} for '{worst_group_name}' â€” a {gap_pct}% difference. "
                f"While below the high-severity threshold, this gap affects real people "
                f"and should be monitored."
            )
        elif spd_val < 0.2:
            spd_explanation = (
                f"SPD of {spd_val:.3f} shows medium bias. "
                f"'{best_group_name}' receives positive outcomes {best_rate:.0%} of the time "
                f"compared to only {worst_rate:.0%} for '{worst_group_name}' â€” "
                f"a {gap_pct} percentage point gap. In practical terms, for every 100 "
                f"people from '{worst_group_name}', approximately {round(gap_pct)} fewer "
                f"receive a positive outcome compared to '{best_group_name}'."
            )
        else:
            spd_explanation = (
                f"SPD of {spd_val:.3f} indicates HIGH bias. "
                f"'{best_group_name}' gets positive outcomes {best_rate:.0%} of the time "
                f"while '{worst_group_name}' gets them only {worst_rate:.0%} of the time â€” "
                f"a {gap_pct} percentage point difference. This means for every 100 "
                f"people from '{worst_group_name}', roughly {round(gap_pct)} of them miss "
                f"out on positive outcomes purely based on their group membership."
            )

        # --- WHY IS DI THIS VALUE ---
        di_val = metrics.get("DI", 1.0) or 1.0 if metrics else 1.0
        if di_val >= 0.8:
            di_explanation = (
                f"Disparate Impact of {di_val:.3f} is above the legal 0.8 threshold. "
                f"The ratio of positive outcome rates ({worst_rate:.0%} Ã· {best_rate:.0%}) "
                f"meets the legal '80% rule' used in employment and lending regulation. "
                f"However, passing this threshold does not mean no bias exists."
            )
        elif di_val >= 0.5:
            di_explanation = (
                f"Disparate Impact of {di_val:.3f} FAILS the legal 0.8 threshold. "
                f"'{worst_group_name}' receives positive outcomes at only "
                f"{di_val:.0%} the rate of '{best_group_name}' "
                f"({worst_rate:.0%} vs {best_rate:.0%}). Under US employment law "
                f"(EEOC guidelines) and EU anti-discrimination directives, this level "
                f"of disparity is considered evidence of indirect discrimination."
            )
        else:
            di_explanation = (
                f"Disparate Impact of {di_val:.3f} is severely below the legal 0.8 threshold. "
                f"'{worst_group_name}' receives positive outcomes at less than half the rate "
                f"of '{best_group_name}' ({worst_rate:.0%} vs {best_rate:.0%}). "
                f"This level of disparity would likely not survive legal scrutiny in "
                f"most jurisdictions that have fairness regulations."
            )

        # --- WHY NEAR-ZERO SPD CAN STILL MEAN BIAS ---
        ceiling_effect = False
        ceiling_explanation = None
        if best_rate > 0.90:
            ceiling_effect = True
            ceiling_explanation = (
                f"Important caveat: '{best_group_name}' already has a {best_rate:.0%} "
                f"approval rate â€” nearly everyone gets approved. When approval rates are "
                f"this high, SPD cannot detect meaningful gaps because there is no 'room' "
                f"for disparity to show. This is called a ceiling effect. "
                f"The real bias may be hidden â€” for example in which cases get flagged "
                f"for manual review, or in continuous scores rather than binary outcomes."
            )

        # --- DATA IMBALANCE EXPLANATION ---
        if is_imbalanced:
            imbalance_explanation = (
                f"'{worst_group_name}' has {worst_count} records vs {best_count} for "
                f"'{best_group_name}' (ratio {size_ratio:.2f}). This imbalance means "
                f"the model has seen far fewer examples of '{worst_group_name}' during "
                f"training, reducing its ability to make accurate and fair decisions "
                f"for this group."
            )
        else:
            imbalance_explanation = (
                f"Group sizes are reasonably balanced: '{worst_group_name}' has "
                f"{worst_count} records and '{best_group_name}' has {best_count}. "
                f"Imbalance is not a major driver of bias here."
            )

        # --- PROXY EXPLANATION (if proxy found) ---
        proxy_explanation = None
        if proxy_features and proxy_features[0].get("correlation", 0) > 0.15:
            top = proxy_features[0]
            proxy_explanation = (
                f"'{top['feature']}' is correlated with '{sensitive_attr}' "
                f"(r={top['correlation']:.2f}). This means even if '{sensitive_attr}' "
                f"is removed from the model, '{top['feature']}' can act as a hidden "
                f"stand-in and produce the same discriminatory outcome. This is called "
                f"proxy discrimination and is often harder to detect than direct bias."
            )

        return {
            "sensitive_attr": sensitive_attr,
            "target_col": target_col,
            "correlation": round(correlation, 4),
            "proxy_features": proxy_features,
            "data_imbalance": data_imbalance,
            "historical_skew": historical_skew,
            "positive_rate_gap": positive_rate_gap,
            "plain_reason": plain_reason,
            "spd_explanation": spd_explanation,
            "di_explanation": di_explanation,
            "ceiling_effect": ceiling_effect,
            "ceiling_explanation": ceiling_explanation,
            "imbalance_explanation": imbalance_explanation,
            "proxy_explanation": proxy_explanation,
            "worst_group": {"name": worst_group_name, "rate": worst_rate, "count": worst_count},
            "best_group": {"name": best_group_name, "rate": best_rate, "count": best_count},
            "gap_pct": gap_pct,
        }

    # â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _correlation(
        self,
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
    ) -> float:
        """
        Return Pearson correlation between two columns.
        Categorical columns are label-encoded first.
        """
        a = self._to_numeric(df[col_a])
        b = self._to_numeric(df[col_b])
        corr = a.corr(b)
        return float(abs(corr)) if not np.isnan(corr) else 0.0

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)
        le = LabelEncoder()
        encoded = le.fit_transform(series.astype(str).fillna("__missing__"))
        return pd.Series(encoded, index=series.index, dtype=float)

    def _proxy_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_attr: str,
    ) -> list[dict[str, Any]]:
        """
        Find the top-N non-target, non-sensitive columns most correlated
        with the sensitive attribute.
        """
        sensitive_numeric = self._to_numeric(df[sensitive_attr])
        candidates = [
            c for c in df.columns
            if c not in (target_col, sensitive_attr)
        ]

        scores: list[tuple[str, float]] = []
        for col in candidates:
            try:
                col_numeric = self._to_numeric(df[col].dropna())
                # Align indices
                aligned_sensitive = sensitive_numeric.loc[col_numeric.index]
                corr = float(abs(aligned_sensitive.corr(col_numeric)))
                if not np.isnan(corr):
                    scores.append((col, corr))
            except Exception:
                continue

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[: self.TOP_PROXY_N]

        result = []
        for feature, corr in top:
            if corr >= 0.5:
                strength = "strong"
                interp = (
                    f"'{feature}' is strongly correlated (r={corr:.2f}) with "
                    f"'{sensitive_attr}' and may act as a proxy variable, "
                    "may produce discriminatory outcomes indirectly, even without explicitly using the sensitive attribute."
                )
            elif corr >= 0.3:
                strength = "moderate"
                interp = (
                    f"'{feature}' has a moderate correlation (r={corr:.2f}) with "
                    f"'{sensitive_attr}'. Monitor for indirect discrimination."
                )
            else:
                strength = "weak"
                interp = (
                    f"'{feature}' has a weak correlation (r={corr:.2f}) with "
                    f"'{sensitive_attr}'. Unlikely to be a significant proxy."
                )

            result.append(
                {
                    "feature": feature,
                    "correlation": round(corr, 4),
                    "strength": strength,
                    "interpretation": interp,
                }
            )

        return result

    def _build_plain_reason(
        self,
        sensitive_attr: str,
        target_col: str,
        correlation: float,
        proxy_features: list[dict],
        positive_rate_gap: float,
        imbalance_flagged: bool,
        historical_skew: float,
    ) -> str:
        """Synthesise a plain-English explanation from all computed signals."""
        if proxy_features and proxy_features[0]["correlation"] >= 0.3:
            top = proxy_features[0]
            return f"'{top['feature']}' correlates with '{sensitive_attr}' (r={top['correlation']:.2f}), {positive_rate_gap:.0%} outcome gap between groups."
        
        return f"{positive_rate_gap:.0%} outcome gap detected between groups of '{sensitive_attr}'. No strong proxy feature found."

