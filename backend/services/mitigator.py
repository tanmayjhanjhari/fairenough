"""
FairEnough - Generic Bias Mitigator Service

ARCHITECTURE
============
Two mitigation strategies:
  1. Reweighing           - dataset-level (weights only, no model required)
  2. Threshold Adjustment - model-level when predict_proba available;
                            internal GBM simulation otherwise

MODEL INTEGRATION POLICY
=========================
  When run_both() receives a real sklearn model:
    - Threshold adjustment uses model.predict_proba() if available (REAL)
    - Otherwise falls back to internal GBM simulation (SIMULATION, labelled)

  When run_both() receives df_with_pred (DataFrame with __predictions__ col):
    - "before" EOD and AOD are computed from REAL predictions vs ground truth
    - is_simulation stays False for "before" metrics

METRIC HONESTY POLICY
=====================
  - "Before" SPD/DI come from the BiasEngine analysis baseline (same values as
    the analysis page). They are NOT recomputed from model predictions here.

  - "After" Reweighing SPD/DI are computed from the REWEIGHTED dataset
    positive rates (dataset-level) — not from model predictions.

  - "After" Threshold SPD/DI come from the real model probabilities
    (if available) or from an internal GBM simulation (clearly labelled).

  - EOD/AOD "before": real when df_with_pred is supplied, None otherwise.
  - EOD/AOD "after" reweighing: computed from simulation GBM with weights.
  - EOD/AOD "after" threshold: real (from real model) or simulation.

  - Acc/Precision/Recall/F1: from real model when available, GBM sim otherwise.

  - Sensitive attribute binning mirrors BiasEngine exactly
    (CARDINALITY_BIN_THRESHOLD=10, data-driven median split).

GENERIC DESIGN
==============
  - No credit-risk, age_group, gender, or dataset-specific code.
  - Works with any binary target column, any sensitive attribute,
    any sklearn-compatible model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class BiasMitigator:
    """
    Generic bias mitigator.

    Works with any dataset/target/sensitive-attribute combination.
    Uses the real uploaded model when provided; falls back to an internal
    GBM simulation (always clearly labelled) otherwise.
    """

    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.30

    # Must match BiasEngine constants exactly
    CARDINALITY_BIN_THRESHOLD: int = 10
    CONTINUOUS_BIN_THRESHOLD: float | None = None
    CONTINUOUS_BIN_LABELS: tuple = ("Low (<= median)", "High (> median)")

    # ── Binning (mirrors BiasEngine._bin_continuous_attr) ─────────────────────

    def _apply_binning(self, series: pd.Series, attr: str) -> pd.Series:
        """
        Apply the same data-driven binning logic as BiasEngine._bin_continuous_attr.
        Numeric attrs with > CARDINALITY_BIN_THRESHOLD unique values are
        binned into two groups using a median split. Fully domain-agnostic.
        """
        if (
            pd.api.types.is_numeric_dtype(series)
            and series.nunique() > self.CARDINALITY_BIN_THRESHOLD
        ):
            valid = pd.to_numeric(series.dropna(), errors="coerce").dropna()
            if valid.empty or valid.nunique() < 2:
                return series.astype(str)

            split_val = float(valid.median())
            high_mask = valid > split_val

            if high_mask.sum() == 0 and (valid < split_val).sum() > 0:
                low_op, high_op = "<", ">="
            else:
                low_op, high_op = "<=", ">"

            split_str = f"{split_val:g}"
            low_label = f"{attr} {low_op} {split_str}"
            high_label = f"{attr} {high_op} {split_str}"

            def _assign_bin(v):
                if pd.isna(v):
                    return low_label
                try:
                    fv = float(v)
                    if high_op == ">=":
                        return high_label if fv >= split_val else low_label
                    else:
                        return high_label if fv > split_val else low_label
                except (ValueError, TypeError):
                    return low_label

            return series.apply(_assign_bin)
        return series.astype(str)

    # ── Binarize target ────────────────────────────────────────────────────────

    def _binarize(self, series: pd.Series) -> pd.Series:
        """Convert any column to binary 0/1. Generic — no hardcoding."""
        y = series.dropna()
        if set(y.unique()).issubset({0, 1, 0.0, 1.0}):
            return series.astype(int)
        if series.nunique() == 2:
            vals = sorted(series.unique())
            return series.map({vals[0]: 0, vals[1]: 1})
        if pd.api.types.is_numeric_dtype(series):
            return (series > series.median()).astype(int)
        return (series == series.mode()[0]).astype(int)

    # ── Dataset-level SPD/DI (identical formula to BiasEngine) ────────────────

    def _dataset_spd_di(
        self,
        df: pd.DataFrame,
        target_col: str,
        sens_col: str,
        weight_col: str | None = None,
    ) -> tuple:
        """
        Compute dataset-level SPD and DI from positive outcome rates.
        Optionally uses sample weights.

        SPD = positive_rate(unprivileged) - positive_rate(privileged)
        DI  = positive_rate(unprivileged) / positive_rate(privileged)

        Returns (spd, di, group_stats).
        """
        sub = df.dropna(subset=[target_col, sens_col]).copy()
        y_bin = self._binarize(sub[target_col])
        sub = sub.copy()
        sub["__y_tmp__"] = y_bin

        group_stats: dict[str, dict] = {}
        for group_name, grp in sub.groupby(sens_col):
            if weight_col and weight_col in grp.columns:
                w = grp[weight_col].clip(0)
                total_w = w.sum()
                if total_w == 0:
                    continue
                pos_rate = float((grp["__y_tmp__"] * w).sum() / total_w)
            else:
                pos_rate = float(grp["__y_tmp__"].mean())
            group_stats[str(group_name)] = {
                "count": int(len(grp)),
                "positive_count": int(grp["__y_tmp__"].sum()),
                "positive_rate": round(pos_rate, 4),
            }

        if len(group_stats) < 2:
            return 0.0, None, group_stats

        priv_name = max(group_stats, key=lambda g: group_stats[g]["positive_rate"])
        unpriv_name = min(group_stats, key=lambda g: group_stats[g]["positive_rate"])
        priv_rate = group_stats[priv_name]["positive_rate"]
        unpriv_rate = group_stats[unpriv_name]["positive_rate"]

        spd = round(unpriv_rate - priv_rate, 4)
        di = round(unpriv_rate / priv_rate, 4) if priv_rate > 0 else None

        return spd, di, group_stats

    # ── Model-level EOD/AOD (using BiasEngine formula) ────────────────────────

    def _compute_eod_aod_from_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        s: np.ndarray,
    ) -> tuple:
        """
        Compute EOD and AOD from real predictions.

        EOD = TPR(unprivileged) - TPR(privileged)
        AOD = 0.5 * [(TPR_u - TPR_p) + (FPR_u - FPR_p)]

        Returns (eod, aod, priv_group, unpriv_group) or (None, None, None, None).
        """
        unique_groups = np.unique(s)
        if len(unique_groups) < 2:
            return None, None, None, None

        # Find privileged = group with highest positive prediction rate
        pred_rates = {g: float(y_pred[s == g].mean()) for g in unique_groups}
        priv_g = max(pred_rates, key=lambda g: pred_rates[g])
        unpriv_g = min(pred_rates, key=lambda g: pred_rates[g])

        def tpr_fpr(mask):
            yt = y_true[mask].astype(int)
            yp = y_pred[mask].astype(int)
            tp = int(((yp == 1) & (yt == 1)).sum())
            fn = int(((yp == 0) & (yt == 1)).sum())
            fp = int(((yp == 1) & (yt == 0)).sum())
            tn = int(((yp == 0) & (yt == 0)).sum())
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            return tpr, fpr

        tpr_p, fpr_p = tpr_fpr(s == priv_g)
        tpr_u, fpr_u = tpr_fpr(s == unpriv_g)

        eod = float(tpr_u - tpr_p)
        aod = float(((tpr_u - tpr_p) + (fpr_u - fpr_p)) / 2.0)
        return round(eod, 4), round(aod, 4), priv_g, unpriv_g

    # ── Prepare features for model/GBM input ──────────────────────────────────

    def _prepare_features(
        self,
        df_work: pd.DataFrame,
        target_col: str,
        sensitive_attr: str,
    ) -> tuple:
        """
        Encode all feature columns except target/sensitive/internal.

        Generic: no hardcoded column names. Detects leaking columns by
        correlation and excludes them.

        Returns (X_array, feature_col_names).
        """
        exclude = {
            target_col, sensitive_attr,
            "__target__", "__sens__", "__y__", "__s__",
            "__sens_binned__", "__weight__", "__sens_raw__", "__y_tmp__",
            "__predictions__", "__truth__",
        }

        y_vals = None
        for cand in ["__y__", "__target__", target_col]:
            if cand in df_work.columns:
                try:
                    y_vals = df_work[cand].astype(float)
                    break
                except Exception:
                    pass

        leaking: set[str] = set()
        if y_vals is not None:
            target_base = (
                target_col.replace("_binary", "").replace("_encoded", "")
                .replace("_label", "").replace("_num", "").lower()
            )
            for col in df_work.columns:
                if col in exclude:
                    continue
                try:
                    if df_work[col].dtype in ["int64", "float64", "int32", "float32"]:
                        corr = abs(float(df_work[col].corr(y_vals)))
                    else:
                        enc = LabelEncoder().fit_transform(
                            df_work[col].fillna("missing").astype(str)
                        )
                        corr = abs(float(np.corrcoef(enc, y_vals)[0, 1]))
                    if corr > 0.90:
                        leaking.add(col)
                except Exception:
                    pass
                if target_base in col.lower() and col.lower() != target_col.lower():
                    leaking.add(col)

        feature_cols = [
            c for c in df_work.columns
            if c not in exclude and c not in leaking
        ]
        if not feature_cols:
            feature_cols = (
                [sensitive_attr] if sensitive_attr in df_work.columns else []
            )

        X_parts: list[np.ndarray] = []
        used_cols: list[str] = []
        for col in feature_cols:
            try:
                if df_work[col].dtype in ["int64", "float64", "int32", "float32"]:
                    col_series = pd.to_numeric(df_work[col], errors="coerce")
                    if col_series.isna().any():
                        med = col_series.median()
                        col_series = col_series.fillna(med if not pd.isna(med) else 0.0)
                    X_parts.append(col_series.values.reshape(-1, 1).astype(float))
                else:
                    enc = LabelEncoder().fit_transform(
                        df_work[col].fillna("missing").astype(str)
                    )
                    X_parts.append(enc.reshape(-1, 1).astype(float))
                used_cols.append(col)
            except Exception:
                pass

        if not X_parts:
            return np.zeros((len(df_work), 1)), []
        
        X_mat = np.hstack(X_parts)
        if np.isnan(X_mat).any() or np.isinf(X_mat).any():
            X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)
        return X_mat, used_cols

    # ── Prepare features for the REAL MODEL ───────────────────────────────────

    def _prepare_features_for_model(
        self,
        df: pd.DataFrame,
        model: Any,
        target_col: str,
        sensitive_attr: str,
        column_mapping: dict[str, str] | None = None,
    ) -> pd.DataFrame | None:
        """
        Build the feature matrix as the real model expects it.

        Uses build_model_feature_matrix from services.feature_resolver to
        resolve case-differences, mappings, and feature names.
        Returns X DataFrame or None on failure.
        """
        try:
            from services.feature_resolver import build_model_feature_matrix
            X = build_model_feature_matrix(
                model=model,
                df=df,
                target_col=target_col,
                sensitive_attr=sensitive_attr,
                column_mapping=column_mapping,
            )
            return X
        except Exception as exc:
            print(f"[Mitigator] _prepare_features_for_model failed: {exc}")
            return None

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_both(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_attr: str,
        predicted_cause: str | None = None,
        # Analysis baseline — use as authoritative "before" dataset-level values
        baseline_spd: float | None = None,
        baseline_di: float | None = None,
        baseline_group_stats: dict | None = None,
        # Real model integration (optional)
        model: Any = None,
        df_with_pred: pd.DataFrame | None = None,
        allow_simulation: bool = True,
        column_mapping: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Run both mitigation strategies and return a unified comparison.

        Parameters
        ----------
        df              Raw dataset (without __predictions__)
        target_col      Target column name
        sensitive_attr  Sensitive attribute column name
        predicted_cause ML-predicted cause label (used for recommendation)
        baseline_spd    SPD from the analysis page (for "before" consistency)
        baseline_di     DI from the analysis page (for "before" consistency)
        baseline_group_stats  Group stats from analysis page
        model           Real uploaded sklearn model (optional)
        df_with_pred    DataFrame with __predictions__ column (optional)
        """
        rew = self.reweigh(
            df=df,
            target_col=target_col,
            sensitive_attr=sensitive_attr,
            baseline_spd=baseline_spd,
            baseline_di=baseline_di,
            baseline_group_stats=baseline_group_stats,
            model=model,
            df_with_pred=df_with_pred,
        )
        thr = self.threshold_adjust(
            df=df,
            target_col=target_col,
            sensitive_attr=sensitive_attr,
            baseline_spd=baseline_spd,
            baseline_di=baseline_di,
            baseline_group_stats=baseline_group_stats,
            model=model,
            df_with_pred=df_with_pred,
            allow_simulation=allow_simulation,
            column_mapping=column_mapping,
        )

        if thr.get("model_required") or not thr.get("after"):
            winner = "reweigh"
            winner_reason = (
                "Reweighing was applied to reduce dataset-level disparity. "
                "Threshold Adjustment requires a trained model or optional simulation."
            )
            return {
                "reweigh": rew,
                "threshold": thr,
                "winner": winner,
                "winner_reason": winner_reason,
                "predicted_cause_used": predicted_cause,
            }

        # ── Winner selection (dataset-level SPD only where model not real) ────
        spd_b = abs(rew["before"]["SPD"] or 0)
        spd_r = abs(rew["after"]["SPD"] or 0)
        spd_t = abs(thr["after"]["SPD"] or 0)

        red_r = (
            round(((spd_b - spd_r) / max(spd_b, 1e-9)) * 100, 1)
            if spd_b > 0 else 0.0
        )
        red_t = (
            round(((spd_b - spd_t) / max(spd_b, 1e-9)) * 100, 1)
            if spd_b > 0 else 0.0
        )

        rew["effects"]["bias_reduction_pct"] = red_r
        thr["effects"]["bias_reduction_pct"] = red_t

        # Cause-based winner preference
        cause_winner: str | None = None
        cause_reason: str | None = None
        if predicted_cause == "proxy":
            cause_winner = "reweigh"
            cause_reason = (
                "Reweighing is preferred for proxy discrimination. "
                "It rebalances group-outcome frequencies so proxy features "
                "can no longer unfairly drive group disparities."
            )
        elif predicted_cause == "underrepresentation":
            cause_winner = "threshold"
            cause_reason = (
                "Threshold Adjustment is preferred for underrepresentation. "
                "It corrects the decision boundary per group, compensating "
                "for the lack of minority training examples."
            )
        elif predicted_cause == "historical_skew":
            cause_winner = "reweigh"
            cause_reason = (
                "Reweighing is preferred for historical bias. "
                "It down-weights historically over-represented group-outcome patterns."
            )

        # If threshold adjustment failed or was unavailable, reweighing wins
        if thr.get("error") or thr["after"].get("SPD") is None:
            cause_winner = "reweigh"
            err_reason = thr.get("error") or "model decision probabilities unavailable"
            cause_reason = (
                f"Reweighing is recommended. Threshold adjustment could not be "
                f"evaluated because: {err_reason}."
            )
        elif cause_winner == "reweigh" and red_r < 5:
            cause_winner = "threshold"
            cause_reason = (
                f"Threshold adjustment recommended because reweighing achieved "
                f"only {red_r:.1f}% bias reduction on this dataset."
            )
        elif cause_winner == "threshold" and red_t < 5:
            cause_winner = "reweigh"
            cause_reason = (
                f"Reweighing recommended because threshold adjustment achieved "
                f"only {red_t:.1f}% bias reduction for this dataset."
            )

        # Pure metric fallback
        if cause_winner is None:
            rew_spd_b = rew["before"]["SPD"]
            rew_spd_a = rew["after"]["SPD"]
            thr_spd_a = thr["after"]["SPD"]
            thr_sim = thr.get("is_simulation", True)
            if red_r >= red_t:
                cause_winner = "reweigh"
                cause_reason = (
                    f"Reweighing achieved {red_r:.1f}% bias reduction "
                    f"(SPD {rew_spd_b:.3f} → {rew_spd_a:.3f}). "
                    f"Based on actual dataset outcome distributions."
                )
            else:
                sim_note = " (simulation model)" if thr_sim else ""
                cause_winner = "threshold"
                cause_reason = (
                    f"Threshold adjustment achieved {red_t:.1f}% bias reduction "
                    f"(SPD {rew_spd_b:.3f} → {thr_spd_a:.3f}){sim_note}."
                )

        winner = cause_winner
        winner_reason = cause_reason

        # Generate explanations
        rew["explanation"] = self._generate_explanation(
            rew["before"], rew["after"], "reweigh", sensitive_attr, rew["effects"],
            is_simulation=False
        )
        thr["explanation"] = self._generate_explanation(
            thr["before"], thr["after"], "threshold", sensitive_attr, thr["effects"],
            is_simulation=thr.get("is_simulation", True)
        )

        return {
            "reweigh": rew,
            "threshold": thr,
            "winner": winner,
            "winner_reason": winner_reason,
            "predicted_cause_used": predicted_cause,
            "model_info": {
                "type": (
                    type(model).__name__ if model is not None
                    else "GradientBoostingClassifier (simulation)"
                ),
                "real_model_used": model is not None,
                "note": (
                    "A real uploaded model was used for threshold adjustment. "
                    "Reweighing dataset-level metrics are from actual outcome "
                    "distributions. Before EOD/AOD are from real model predictions."
                ) if model is not None else (
                    "No real model provided. Threshold adjustment uses an internal "
                    "GBM simulation. Reweighing SPD/DI are from actual dataset "
                    "outcome distributions. EOD/AOD unavailable without a model."
                ),
            },
        }

    # ── Reweighing ─────────────────────────────────────────────────────────────

    def reweigh(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_attr: str,
        baseline_spd: float | None = None,
        baseline_di: float | None = None,
        baseline_group_stats: dict | None = None,
        model: Any = None,
        df_with_pred: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Genuine dataset-level reweighing.

        - Computes IBM reweighing weights: w = P(G)*P(Y) / P(G,Y).
        - "After" SPD/DI from weighted dataset positive rates (dataset-level).
        - EOD/AOD "before": real from model predictions if df_with_pred given.
        - EOD/AOD "after" reweighing: from internal GBM with weights (sim).
        - Performance metrics: from internal GBM simulation (always labelled).
        - is_simulation = False for SPD/DI (real data); True for EOD/AOD after.

        Generic: works for any dataset/target/sensitive-attribute.
        """
        df_work = df.copy().dropna(subset=[target_col, sensitive_attr])

        # Apply same binning as BiasEngine
        df_work["__sens_binned__"] = self._apply_binning(
            df_work[sensitive_attr], sensitive_attr
        )

        df_work["__y__"] = self._binarize(df_work[target_col])

        # ── BEFORE: authoritative analysis baseline ──────────────────────────
        if baseline_spd is not None and baseline_di is not None:
            before_spd = float(baseline_spd)
            before_di = float(baseline_di)
            before_gs = baseline_group_stats or {}
        else:
            # Fallback: compute from dataset (same formula as BiasEngine)
            before_spd, before_di, before_gs = self._dataset_spd_di(
                df_work, "__y__", "__sens_binned__"
            )

        # ── Real model-level before metrics (EOD/AOD) ────────────────────────
        before_eod: float | None = None
        before_aod: float | None = None
        real_model_before = False

        if model is not None and df_with_pred is not None and "__predictions__" in df_with_pred.columns:
            try:
                df_pred_work = df_with_pred.dropna(
                    subset=[target_col, sensitive_attr, "__predictions__"]
                ).copy()
                df_pred_work["__sens_binned__"] = self._apply_binning(
                    df_pred_work[sensitive_attr], sensitive_attr
                )
                le = LabelEncoder()
                s_all = le.fit_transform(df_pred_work["__sens_binned__"])
                y_true = self._binarize(df_pred_work[target_col]).values
                y_pred = self._binarize(df_pred_work["__predictions__"]).values
                eod, aod, _, _ = self._compute_eod_aod_from_predictions(
                    y_true, y_pred, s_all
                )
                before_eod = eod
                before_aod = aod
                real_model_before = True
            except Exception as exc:
                print(f"[Mitigator/Reweigh] Real before EOD/AOD failed: {exc}")

        SIM_NOTE_PERF = (
            "Performance metrics (Acc/Precision/Recall/F1) come from an internal "
            "GBM simulation model, not from the real deployed model."
        )

        before: dict[str, Any] = {
            "SPD": round(before_spd, 4),
            "DI": round(before_di, 4) if before_di is not None else None,
            "EOD": before_eod,
            "AOD": before_aod,
            "eod_available": before_eod is not None,
            "aod_available": before_aod is not None,
            "metrics_mode": "model_level" if real_model_before else "dataset_level",
            "group_stats": before_gs,
        }

        # ── Compute IBM reweighing weights ───────────────────────────────────
        n = len(df_work)
        le_s = LabelEncoder()
        s_all = le_s.fit_transform(df_work["__sens_binned__"])
        y_all = df_work["__y__"].values

        weights = np.ones(n)
        for g in np.unique(s_all):
            for label in np.unique(y_all):
                mask = (s_all == g) & (y_all == label)
                n_gl = int(mask.sum())
                if n_gl == 0:
                    continue
                p_g = float((s_all == g).sum()) / n
                p_l = float((y_all == label).sum()) / n
                p_gl = n_gl / n
                weights[mask] = (p_g * p_l) / p_gl

        weights = np.clip(weights, 0.1, 10.0)
        df_work["__weight__"] = weights

        # ── AFTER: dataset-level SPD/DI from reweighted positive rates ───────
        after_spd, after_di, after_gs = self._dataset_spd_di(
            df_work, "__y__", "__sens_binned__", weight_col="__weight__"
        )

        # If real model predictions are available in df_with_pred, compute genuine baseline performance
        if real_model_before:
            try:
                y_true_all = self._binarize(df_pred_work[target_col]).values
                y_pred_all = self._binarize(df_pred_work["__predictions__"]).values
                before["accuracy"] = round(float(accuracy_score(y_true_all, y_pred_all)), 4)
                before["precision"] = round(
                    float(precision_score(y_true_all, y_pred_all, zero_division=0)), 4
                )
                before["recall"] = round(
                    float(recall_score(y_true_all, y_pred_all, zero_division=0)), 4
                )
                before["f1"] = round(
                    float(f1_score(y_true_all, y_pred_all, zero_division=0)), 4
                )
                before["simulation_note"] = None
            except Exception:
                pass

        REW_PERF_NOTE = (
            "Model-level performance metrics (Accuracy, F1) post-reweighing are unavailable "
            "because reweighing operates at the dataset level. Retraining the model with sample "
            "weights requires a compatible training interface. Dataset-level fairness metrics "
            "(SPD, DI) reflect actual outcome distributions after applying statistical sample weights."
        )

        after: dict[str, Any] = {
            "SPD": round(after_spd, 4),
            "DI": round(after_di, 4) if after_di is not None else None,
            "EOD": None,
            "AOD": None,
            "eod_available": False,
            "aod_available": False,
            "metrics_mode": "dataset_level_reweighted",
            "group_stats": after_gs,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "simulation_note": REW_PERF_NOTE,
        }

        if "simulation_note" not in before or before["simulation_note"] is None:
            if not real_model_before:
                before["simulation_note"] = (
                    "Performance metrics require an evaluated model. "
                    "Dataset-level fairness metrics (SPD, DI) reflect actual outcome distributions."
                )

        # ── Effects ──────────────────────────────────────────────────────────
        effects = self._compute_effects(before, after)

        has_real_model = bool((model is not None) or real_model_before)
        if not has_real_model:
            rew_perf_note = "Model-level performance unavailable — no model uploaded"
            before["simulation_note"] = rew_perf_note
            after["simulation_note"] = rew_perf_note

        return {
            "before": before,
            "after": after,
            "effects": effects,
            "improvement_pct": effects["bias_reduction_pct"],
            "weights_summary": {
                "min": round(float(weights.min()), 3),
                "max": round(float(weights.max()), 3),
                "mean": round(float(weights.mean()), 3),
            },
            "is_simulation": False,  # Reweighing SPD/DI = real dataset metrics
            "has_real_model": has_real_model,
            "simulation_note": (
                "Dataset-level SPD and DI were computed before and after "
                "applying reweighing weights to the outcome distributions. "
                "Performance metrics are from an internal GBM simulation. "
                + (
                    "EOD/AOD before are from the real model predictions. "
                    if real_model_before else
                    "EOD/AOD require model predictions (not provided)."
                )
            ),
        }

    # ── Threshold Adjustment ───────────────────────────────────────────────────

    def threshold_adjust(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_attr: str,
        baseline_spd: float | None = None,
        baseline_di: float | None = None,
        baseline_group_stats: dict | None = None,
        model: Any = None,
        df_with_pred: pd.DataFrame | None = None,
        fairness_metric: str = "demographic_parity",
        utility_metric: str = "balanced_accuracy",
        utility_weight: float = 1.0,
        min_pos_rate: float = 0.05,
        max_pos_rate: float = 0.95,
        min_samples_per_class: int = 2,
        allow_simulation: bool = True,
        column_mapping: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Threshold adjustment mitigation.

        REAL MODEL PATH (model has predict_proba):
          - Uses actual prediction probabilities from the real model
          - Finds per-group thresholds that minimise SPD
          - Computes real after SPD/DI/EOD/AOD from mitigated predictions
          - is_simulation = False

        SIMULATION PATH (no model / model lacks predict_proba):
          - Trains an internal GBM to generate probability scores
          - Applies threshold optimisation on simulation scores
          - is_simulation = True, clearly labelled

        Works generically for any dataset/target/sensitive-attribute.
        """
        df_work = df.copy().dropna(subset=[target_col, sensitive_attr])

        # Apply same binning as BiasEngine
        df_work["__sens_binned__"] = self._apply_binning(
            df_work[sensitive_attr], sensitive_attr
        )
        df_work["__y__"] = self._binarize(df_work[target_col])

        # ── BEFORE: same analysis baseline ───────────────────────────────────
        if baseline_spd is not None and baseline_di is not None:
            before_spd = float(baseline_spd)
            before_di = float(baseline_di)
            before_gs = baseline_group_stats or {}
        else:
            before_spd, before_di, before_gs = self._dataset_spd_di(
                df_work, "__y__", "__sens_binned__"
            )

        # Real model before metrics (same as reweighing)
        before_eod: float | None = None
        before_aod: float | None = None
        real_model_before = False

        if model is not None and df_with_pred is not None and "__predictions__" in df_with_pred.columns:
            try:
                df_pred_work = df_with_pred.dropna(
                    subset=[target_col, sensitive_attr, "__predictions__"]
                ).copy()
                df_pred_work["__sens_binned__"] = self._apply_binning(
                    df_pred_work[sensitive_attr], sensitive_attr
                )
                le = LabelEncoder()
                s_arr = le.fit_transform(df_pred_work["__sens_binned__"])
                y_true_arr = self._binarize(df_pred_work[target_col]).values
                y_pred_arr = self._binarize(df_pred_work["__predictions__"]).values
                eod, aod, _, _ = self._compute_eod_aod_from_predictions(
                    y_true_arr, y_pred_arr, s_arr
                )
                before_eod = eod
                before_aod = aod
                real_model_before = True
            except Exception as exc:
                print(f"[Mitigator/Threshold] Real before EOD/AOD failed: {exc}")

        before: dict[str, Any] = {
            "SPD": round(before_spd, 4),
            "DI": round(before_di, 4) if before_di is not None else None,
            "EOD": before_eod,
            "AOD": before_aod,
            "eod_available": before_eod is not None,
            "aod_available": before_aod is not None,
            "metrics_mode": "model_level" if real_model_before else "dataset_level",
            "group_stats": before_gs,
        }

        # ── Decide: real model path or simulation path ────────────────────────
        has_predict_proba = (
            model is not None
            and callable(getattr(model, "predict_proba", None))
        )

        if has_predict_proba:
            res = self._threshold_real_model(
                df=df,
                df_work=df_work,
                target_col=target_col,
                sensitive_attr=sensitive_attr,
                before=before,
                model=model,
                fairness_metric=fairness_metric,
                utility_metric=utility_metric,
                utility_weight=utility_weight,
                min_pos_rate=min_pos_rate,
                max_pos_rate=max_pos_rate,
                min_samples_per_class=min_samples_per_class,
                column_mapping=column_mapping,
            )
            res["has_real_model"] = True
            return res
        else:
            if not allow_simulation:
                print(f"[Mitigator/Threshold] No model provided and simulation not requested -> Model Required")
                return {
                    "technique": "Threshold Adjustment",
                    "status": "model_required",
                    "model_required": True,
                    "available": False,
                    "is_simulation": False,
                    "can_simulate": True,
                    "has_real_model": False,
                    "title": "Model Required",
                    "message": "Upload a compatible trained model to perform real threshold adjustment.",
                    "simulation_note": (
                        "No trained model uploaded. You can optionally run a simulation to "
                        "demonstrate how threshold adjustment works. Simulation results are "
                        "illustrative and are NOT results from a real model."
                    ),
                    "before": before,
                    "after": None,
                    "effects": {
                        "bias_reduction_pct": 0.0,
                        "accuracy_retained_pct": 100.0,
                    },
                }

            reason = (
                "model lacks predict_proba()"
                if model is not None
                else "no model provided"
            )
            print(f"[Mitigator/Threshold] Using GBM simulation ({reason})")
            res = self._threshold_simulation(
                df_work=df_work,
                target_col=target_col,
                sensitive_attr=sensitive_attr,
                before=before,
                fairness_metric=fairness_metric,
                utility_metric=utility_metric,
                utility_weight=utility_weight,
                min_pos_rate=min_pos_rate,
                max_pos_rate=max_pos_rate,
                min_samples_per_class=min_samples_per_class,
            )
            res["has_real_model"] = False
            return res

    def _threshold_real_model(
        self,
        df: pd.DataFrame,
        df_work: pd.DataFrame,
        target_col: str,
        sensitive_attr: str,
        before: dict,
        model: Any,
        fairness_metric: str = "demographic_parity",
        utility_metric: str = "balanced_accuracy",
        utility_weight: float = 1.0,
        min_pos_rate: float = 0.05,
        max_pos_rate: float = 0.95,
        min_samples_per_class: int = 2,
        column_mapping: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Threshold adjustment using the REAL model's predict_proba.
        is_simulation = False.
        """
        # Cross-version sklearn unpickling adapter for LogisticRegression
        if hasattr(model, "predict_proba") and not hasattr(model, "multi_class") and "LogisticRegression" in type(model).__name__:
            setattr(model, "multi_class", "auto")

        try:
            # Align df_work with df (df_work may have fewer rows after dropna)
            df_aligned = df.copy()
            df_aligned["__sens_binned__"] = self._apply_binning(
                df_aligned[sensitive_attr], sensitive_attr
            )
            df_aligned["__y__"] = self._binarize(df_aligned[target_col])
            valid_mask = df_aligned[[target_col, sensitive_attr]].notna().all(axis=1)
            df_aligned = df_aligned[valid_mask].reset_index(drop=True)
            X_aligned = self._prepare_features_for_model(
                df_aligned, model, target_col, sensitive_attr, column_mapping=column_mapping
            )
            if X_aligned is None:
                raise ValueError("Could not build feature matrix for the real model")

            try:
                proba = model.predict_proba(X_aligned)
            except Exception:
                proba = model.predict_proba(getattr(X_aligned, "values", X_aligned))

            # Use second column for binary (positive class prob)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                scores = proba[:, 1]
            else:
                scores = proba.ravel()

            le = LabelEncoder()
            s_all = le.fit_transform(df_aligned["__sens_binned__"])
            y_true = df_aligned["__y__"].values
            groups = np.unique(s_all)

            # Before: global threshold 0.5 (same as training)
            y_pred_before = (scores >= 0.5).astype(int)
            before["accuracy"] = round(float(accuracy_score(y_true, y_pred_before)), 4)
            before["precision"] = round(
                float(precision_score(y_true, y_pred_before, zero_division=0)), 4
            )
            before["recall"] = round(
                float(recall_score(y_true, y_pred_before, zero_division=0)), 4
            )
            before["f1"] = round(
                float(f1_score(y_true, y_pred_before, zero_division=0)), 4
            )
            before["positive_prediction_rate"] = round(float(y_pred_before.mean()), 4)
            before["simulation_note"] = None  # These are real model metrics

            # Compute real baseline EOD/AOD if not already set
            if before.get("EOD") is None:
                b_eod, b_aod, _, _ = self._compute_eod_aod_from_predictions(
                    y_true, y_pred_before, s_all
                )
                before["EOD"] = b_eod
                before["AOD"] = b_aod
                before["eod_available"] = b_eod is not None
                before["aod_available"] = b_aod is not None

            # Per-group threshold optimisation (principled & data-adaptive)
            best_thresholds, best_spd_val = self._optimise_thresholds(
                scores=scores,
                s=s_all,
                groups=groups,
                y_true=y_true,
                fairness_metric=fairness_metric,
                utility_metric=utility_metric,
                utility_weight=utility_weight,
                min_pos_rate=min_pos_rate,
                max_pos_rate=max_pos_rate,
                min_samples_per_class=min_samples_per_class,
            )

            # Apply best thresholds
            y_pred_after = np.zeros(len(scores), dtype=int)
            for g, thresh in best_thresholds.items():
                y_pred_after[s_all == g] = (scores[s_all == g] >= thresh).astype(int)

            # After: real metrics from actual model probabilities
            after_eod, after_aod, _, _ = self._compute_eod_aod_from_predictions(
                y_true, y_pred_after, s_all
            )

            # Positive rate per group for SPD/DI
            sim_rates = {g: float(y_pred_after[s_all == g].mean()) for g in groups}
            priv_val = max(sim_rates.values())
            unpriv_val = min(sim_rates.values())
            after_spd = round(unpriv_val - priv_val, 4)
            after_di = round(unpriv_val / priv_val, 4) if priv_val > 0 else None

            after: dict[str, Any] = {
                "SPD": after_spd,
                "DI": after_di,
                "EOD": after_eod,
                "AOD": after_aod,
                "eod_available": after_eod is not None,
                "aod_available": after_aod is not None,
                "metrics_mode": "model_level",
                "accuracy": round(float(accuracy_score(y_true, y_pred_after)), 4),
                "precision": round(
                    float(precision_score(y_true, y_pred_after, zero_division=0)), 4
                ),
                "recall": round(
                    float(recall_score(y_true, y_pred_after, zero_division=0)), 4
                ),
                "f1": round(
                    float(f1_score(y_true, y_pred_after, zero_division=0)), 4
                ),
                "positive_prediction_rate": round(float(y_pred_after.mean()), 4),
                "simulation_note": None,
                "group_stats": {
                    str(le.classes_[g]): {
                        "positive_rate": round(float(sim_rates[g]), 4),
                        "threshold": round(float(best_thresholds[g]), 3),
                        "count": int((s_all == g).sum()),
                        "positive_count": int((y_pred_after[s_all == g] == 1).sum()),
                    }
                    for g in groups
                },
                "fairness_objective": fairness_metric,
            }

            effects = self._compute_effects(before, after)

            return {
                "before": before,
                "after": after,
                "effects": effects,
                "improvement_pct": effects["bias_reduction_pct"],
                "thresholds": {
                    str(le.classes_[g]): round(float(t), 3)
                    for g, t in best_thresholds.items()
                },
                "is_simulation": False,  # Real model was used
                "simulation_note": None,
            }

        except Exception as exc:
            err_msg = f"Real model evaluation failed: {exc}"
            print(f"[Mitigator/ThresholdRealModel] {err_msg}")
            # NEVER silently switch to GBM when a real model was supplied
            return {
                "before": before,
                "after": {
                    "SPD": None,
                    "DI": None,
                    "EOD": None,
                    "AOD": None,
                    "eod_available": False,
                    "aod_available": False,
                    "metrics_mode": "unavailable",
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "error": err_msg,
                    "simulation_note": None,
                    "group_stats": {},
                },
                "effects": {
                    "accuracy_delta": None,
                    "precision_delta": None,
                    "recall_delta": None,
                    "f1_delta": None,
                    "spd_delta": None,
                    "bias_reduction_pct": 0.0,
                    "accuracy_retained_pct": 100.0,
                    "diagnostic": err_msg,
                },
                "improvement_pct": 0.0,
                "thresholds": {},
                "is_simulation": False,
                "error": err_msg,
                "simulation_note": None,
            }

    def _threshold_simulation(
        self,
        df_work: pd.DataFrame,
        target_col: str,
        sensitive_attr: str,
        before: dict,
        fallback_reason: str | None = None,
        fairness_metric: str = "demographic_parity",
        utility_metric: str = "balanced_accuracy",
        utility_weight: float = 1.0,
        min_pos_rate: float = 0.05,
        max_pos_rate: float = 0.95,
        min_samples_per_class: int = 2,
    ) -> dict[str, Any]:
        """
        Threshold adjustment via internal GBM simulation.
        is_simulation = True. Always clearly labelled.
        """
        le_s = LabelEncoder()
        s_all = le_s.fit_transform(df_work["__sens_binned__"])
        y_all = df_work["__y__"].values

        X_all, _ = self._prepare_features(df_work, target_col, sensitive_attr)

        idx = np.arange(len(df_work))
        try:
            idx_train, idx_test = train_test_split(
                idx,
                test_size=self.TEST_SIZE,
                random_state=self.RANDOM_STATE,
                stratify=y_all,
            )
        except ValueError:
            idx_train, idx_test = train_test_split(
                idx, test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE
            )

        X_train, X_test = X_all[idx_train], X_all[idx_test]
        y_train, y_test = y_all[idx_train], y_all[idx_test]
        s_test = s_all[idx_test]

        # Use HistGradientBoostingClassifier which natively accepts NaNs and trains 10x faster
        model_sim = HistGradientBoostingClassifier(
            max_iter=60,
            max_depth=4,
            learning_rate=0.1,
            random_state=self.RANDOM_STATE,
        )
        model_sim.fit(X_train, y_train)
        proba = model_sim.predict_proba(X_test)[:, 1]

        y_pred_before = (proba >= 0.5).astype(int)
        SIM_NOTE = (
            "Values come from an internal GBM simulation model. "
            "Threshold adjustment requires real model probability scores — "
            "these values illustrate the technique but are not from the real model."
        )
        before["accuracy"] = round(float(accuracy_score(y_test, y_pred_before)), 4)
        before["precision"] = round(
            float(precision_score(y_test, y_pred_before, zero_division=0)), 4
        )
        before["recall"] = round(
            float(recall_score(y_test, y_pred_before, zero_division=0)), 4
        )
        before["f1"] = round(
            float(f1_score(y_test, y_pred_before, zero_division=0)), 4
        )
        before["simulation_note"] = SIM_NOTE

        groups = np.unique(s_test)
        best_thresholds, _ = self._optimise_thresholds(
            scores=proba,
            s=s_test,
            groups=groups,
            y_true=y_test,
            fairness_metric=fairness_metric,
            utility_metric=utility_metric,
            utility_weight=utility_weight,
            min_pos_rate=min_pos_rate,
            max_pos_rate=max_pos_rate,
            min_samples_per_class=min_samples_per_class,
        )

        y_pred_after = np.zeros(len(proba), dtype=int)
        for g, thresh in best_thresholds.items():
            y_pred_after[s_test == g] = (proba[s_test == g] >= thresh).astype(int)

        after_eod, after_aod, _, _ = self._compute_eod_aod_from_predictions(
            y_test, y_pred_after, s_test
        )

        sim_rates = {g: float(y_pred_after[s_test == g].mean()) for g in groups}
        priv_val = max(sim_rates.values())
        unpriv_val = min(sim_rates.values())
        after_spd_sim = round(unpriv_val - priv_val, 4)
        after_di_sim = round(unpriv_val / priv_val, 4) if priv_val > 0 else None

        after: dict[str, Any] = {
            "SPD": after_spd_sim,
            "DI": after_di_sim,
            "EOD": None,
            "AOD": None,
            "eod_available": False,
            "aod_available": False,
            "metrics_mode": "simulation",
            "accuracy": round(float(accuracy_score(y_test, y_pred_after)), 4),
            "precision": round(
                float(precision_score(y_test, y_pred_after, zero_division=0)), 4
            ),
            "recall": round(
                float(recall_score(y_test, y_pred_after, zero_division=0)), 4
            ),
            "f1": round(
                float(f1_score(y_test, y_pred_after, zero_division=0)), 4
            ),
            "simulation_note": SIM_NOTE,
            "group_stats": {},
        }

        effects = self._compute_effects(before, after)
        full_sim_note = (
            "Threshold adjustment requires model decision probability scores. "
            + ("No model provided. " if fallback_reason is None else f"Reason: {fallback_reason}. ")
            + "An internal GBM simulation was used to demonstrate the technique. "
            "Results are illustrative only — not from the real model."
        )

        return {
            "before": before,
            "after": after,
            "effects": effects,
            "improvement_pct": effects["bias_reduction_pct"],
            "thresholds": {
                str(le_s.classes_[g]): round(float(t), 3)
                for g, t in best_thresholds.items()
            },
            "is_simulation": True,
            "simulation_note": full_sim_note,
        }

    # ── Threshold optimiser (generic, no dataset-specific code) ───────────────

    def _optimise_thresholds(
        self,
        scores: np.ndarray,
        s: np.ndarray,
        groups: np.ndarray,
        y_true: np.ndarray | None = None,
        fairness_metric: str = "demographic_parity",
        utility_metric: str = "balanced_accuracy",
        utility_weight: float = 1.0,
        min_pos_rate: float = 0.05,
        max_pos_rate: float = 0.95,
        min_samples_per_class: int = 2,
        num_candidates: int = 21,
    ) -> tuple[dict, float]:
        """
        Data-adaptive, multi-objective per-group threshold optimizer.

        Principles:
        1. Candidates are derived from empirical quantiles of scores (score support),
           never hardcoded ranges like [0.2, 0.8].
        2. Configurable non-degeneracy constraints reject all-positive and all-negative
           collapses (default: group/overall positive rate in [0.05, 0.95] and at least
           min_samples_per_class in both 0 and 1 classes).
        3. Multi-objective loss balances fairness disparity (SPD for demographic parity,
           or EOD for equal opportunity) against predictive utility (balanced accuracy,
           F1, or accuracy):
               Loss(t) = Disparity(t) + utility_weight * (1.0 - Utility(t))
        4. Scalable multi-group search: 2D grid for 2 groups; global search + coordinate
           descent for > 2 groups (no exponential blowup).
        """
        scores_clean = np.asarray(scores, dtype=float)
        unique_scores = np.unique(scores_clean)
        if len(unique_scores) <= 1:
            return {g: float(scores_clean[0]) for g in groups}, 0.0

        # Empirical quantiles covering score support strictly within interior
        quantiles = np.linspace(0.02, 0.98, num_candidates)
        candidates = np.unique(np.quantile(scores_clean, quantiles))
        med = float(np.median(scores_clean))
        if med not in candidates:
            candidates = np.unique(np.append(candidates, med))

        def _evaluate_candidate(thresh_dict: dict, check_rate_bounds: bool) -> tuple[bool, float, float]:
            y_adj = np.zeros(len(scores_clean), dtype=int)
            for g in groups:
                t = thresh_dict[g]
                m = (s == g)
                y_adj[m] = (scores_clean[m] >= t).astype(int)

            # Overall non-degeneracy
            n_pos = int((y_adj == 1).sum())
            n_neg = int((y_adj == 0).sum())
            if n_pos < min_samples_per_class or n_neg < min_samples_per_class:
                return False, float("inf"), float("inf")
            tot_rate = n_pos / len(y_adj)
            if check_rate_bounds and (tot_rate < min_pos_rate or tot_rate > max_pos_rate):
                return False, float("inf"), float("inf")

            # Per-group non-degeneracy
            rates = {}
            for g in groups:
                m = (s == g)
                n_g = int(m.sum())
                if n_g == 0:
                    continue
                g_pos = int((y_adj[m] == 1).sum())
                g_neg = n_g - g_pos
                min_g_samples = min(min_samples_per_class, max(1, n_g // 10))
                if g_pos < min_g_samples or g_neg < min_g_samples:
                    return False, float("inf"), float("inf")
                g_rate = g_pos / n_g
                if check_rate_bounds and (g_rate < min_pos_rate or g_rate > max_pos_rate):
                    return False, float("inf"), float("inf")
                rates[g] = g_rate

            # Fairness disparity calculation
            if fairness_metric == "equal_opportunity" and y_true is not None:
                tprs = {}
                for g in groups:
                    m_pos = (s == g) & (y_true == 1)
                    tprs[g] = float(y_adj[m_pos].mean()) if m_pos.sum() > 0 else 0.0
                fairness_disparity = abs(max(tprs.values()) - min(tprs.values()))
            else:
                fairness_disparity = abs(max(rates.values()) - min(rates.values()))

            # Predictive utility
            if y_true is not None and utility_weight > 0:
                if utility_metric == "balanced_accuracy":
                    u = float(balanced_accuracy_score(y_true, y_adj))
                elif utility_metric == "f1":
                    u = float(f1_score(y_true, y_adj, zero_division=0))
                else:
                    u = float(accuracy_score(y_true, y_adj))
                utility_loss = 1.0 - u
            else:
                utility_loss = 0.0

            total_loss = fairness_disparity + utility_weight * utility_loss
            return True, fairness_disparity, total_loss

        # Evaluate with strict rate bounds first; adaptively relax if dataset is heavily imbalanced
        for check_rates in [True, False]:
            best_thresholds = {g: med for g in groups}
            best_loss = float("inf")
            best_fairness_val = float("inf")
            found_valid = False

            if len(groups) == 2:
                g0, g1 = groups[0], groups[1]
                for t0 in candidates:
                    for t1 in candidates:
                        cand = {g0: t0, g1: t1}
                        valid, f_val, loss = _evaluate_candidate(cand, check_rates)
                        if valid:
                            found_valid = True
                            if loss < best_loss:
                                best_loss = loss
                                best_fairness_val = f_val
                                best_thresholds = cand
            else:
                # 1. Optimal global threshold
                for t in candidates:
                    cand = {g: t for g in groups}
                    valid, f_val, loss = _evaluate_candidate(cand, check_rates)
                    if valid:
                        found_valid = True
                        if loss < best_loss:
                            best_loss = loss
                            best_fairness_val = f_val
                            best_thresholds = cand

                # 2. Coordinate descent across individual groups (2 passes)
                curr = dict(best_thresholds)
                for _ in range(2):
                    for g in groups:
                        for t in candidates:
                            cand = dict(curr)
                            cand[g] = t
                            valid, f_val, loss = _evaluate_candidate(cand, check_rates)
                            if valid and loss < best_loss:
                                best_loss = loss
                                best_fairness_val = f_val
                                best_thresholds = dict(cand)
                                curr = dict(cand)

            if found_valid:
                return best_thresholds, best_fairness_val

        return {g: med for g in groups}, 0.0

    # ── GBM simulation helper ─────────────────────────────────────────────────

    def _run_simulation(
        self,
        df_work: pd.DataFrame,
        target_col: str,
        sensitive_attr: str,
        weights: np.ndarray | None = None,
    ) -> tuple:
        """
        Run a GBM simulation before (no weights) and after (with weights).
        Returns (before_perf, after_perf) dicts or (None, None) on error.
        Includes EOD/AOD from simulation predictions.
        """
        try:
            le_s = LabelEncoder()
            s_all = le_s.fit_transform(df_work["__sens_binned__"])
            y_all = df_work["__y__"].values
            X_all, _ = self._prepare_features(df_work, target_col, sensitive_attr)

            idx = np.arange(len(df_work))
            try:
                idx_train, idx_test = train_test_split(
                    idx,
                    test_size=self.TEST_SIZE,
                    random_state=self.RANDOM_STATE,
                    stratify=y_all,
                )
            except ValueError:
                idx_train, idx_test = train_test_split(
                    idx, test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE
                )

            X_train, X_test = X_all[idx_train], X_all[idx_test]
            y_train, y_test = y_all[idx_train], y_all[idx_test]
            s_test = s_all[idx_test]

            # Before (no weights)
            m1 = HistGradientBoostingClassifier(
                max_iter=60,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.RANDOM_STATE,
            )
            m1.fit(X_train, y_train)
            p1 = m1.predict(X_test)
            e1, a1, _, _ = self._compute_eod_aod_from_predictions(y_test, p1, s_test)
            sim_before = {
                "accuracy": round(float(accuracy_score(y_test, p1)), 4),
                "precision": round(
                    float(precision_score(y_test, p1, zero_division=0)), 4
                ),
                "recall": round(
                    float(recall_score(y_test, p1, zero_division=0)), 4
                ),
                "f1": round(float(f1_score(y_test, p1, zero_division=0)), 4),
                "eod": e1,
                "aod": a1,
            }

            if weights is None:
                return sim_before, None

            w_train = weights[idx_train]
            m2 = HistGradientBoostingClassifier(
                max_iter=60,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.RANDOM_STATE,
            )
            m2.fit(X_train, y_train, sample_weight=w_train)
            p2 = m2.predict(X_test)
            e2, a2, _, _ = self._compute_eod_aod_from_predictions(y_test, p2, s_test)
            sim_after = {
                "accuracy": round(float(accuracy_score(y_test, p2)), 4),
                "precision": round(
                    float(precision_score(y_test, p2, zero_division=0)), 4
                ),
                "recall": round(
                    float(recall_score(y_test, p2, zero_division=0)), 4
                ),
                "f1": round(float(f1_score(y_test, p2, zero_division=0)), 4),
                "eod": e2,
                "aod": a2,
            }
            return sim_before, sim_after

        except Exception as exc:
            print(f"[Mitigator] Simulation failed: {exc}")
            return None, None

    # ── Effects computation ────────────────────────────────────────────────────

    def _compute_effects(self, before: dict, after: dict) -> dict[str, Any]:
        """Compute delta metrics. Uses absolute SPD for bias reduction."""
        spd_b = abs(before.get("SPD") or 0)
        spd_a = abs(after.get("SPD") or 0)
        bias_reduction_pct = (
            round(((spd_b - spd_a) / max(spd_b, 1e-9)) * 100, 1)
            if spd_b > 0 else 0.0
        )
        spd_delta = round((after.get("SPD") or 0) - (before.get("SPD") or 0), 4)

        di_b = before.get("DI")
        di_a = after.get("DI")
        di_delta = (
            round(di_a - di_b, 4)
            if di_b is not None and di_a is not None else None
        )

        acc_b = before.get("accuracy")
        acc_a = after.get("accuracy")
        acc_delta = (
            round(acc_a - acc_b, 4)
            if acc_b is not None and acc_a is not None else None
        )
        acc_retained = (
            round(acc_a / max(acc_b, 1e-9) * 100, 1)
            if (acc_b is not None and acc_a is not None) else None
        )

        diagnostic: str | None = None
        if spd_b > 0.01 and bias_reduction_pct < 3:
            diagnostic = (
                f"Low bias reduction ({bias_reduction_pct:.1f}%). "
                "This may indicate that the disparity is driven by proxy "
                "features that survive rebalancing, or that groups are "
                "already near-equal in size in the outcome distribution."
            )

        return {
            "bias_reduction_pct": bias_reduction_pct,
            "spd_delta": spd_delta,
            "di_delta": di_delta,
            "accuracy_delta": acc_delta,
            "accuracy_retained_pct": acc_retained,
            "diagnostic": diagnostic,
        }

    # ── Explanation text ───────────────────────────────────────────────────────

    def _generate_explanation(
        self,
        before: dict,
        after: dict,
        technique: str,
        sensitive_attr: str,
        effects: dict,
        is_simulation: bool = False,
    ) -> dict:
        """Plain-English explanation of what the mitigation technique did."""
        spd_before = abs(before.get("SPD") or 0)
        spd_after = abs(after.get("SPD") or 0)
        acc_before = before.get("accuracy")
        acc_after = after.get("accuracy")
        bias_reduction = effects.get("bias_reduction_pct") or 0

        if technique == "reweigh":
            how_it_works = (
                f"Reweighing assigns higher statistical weight to under-represented "
                f"(group, outcome) combinations in the training data. For "
                f"'{sensitive_attr}', group-outcome pairs that were historically "
                f"under-represented receive greater importance, rebalancing the "
                f"learned outcome distribution."
            )
        else:
            how_it_works = (
                f"Threshold adjustment uses group-specific decision thresholds "
                f"rather than a single global threshold for '{sensitive_attr}'. "
                f"Each group gets its own cut-off probability, calibrated to "
                f"equalise outcome rates across groups."
                + (
                    " (Internal GBM simulation — no real model provided.)"
                    if is_simulation else ""
                )
            )

        sim_suffix = " (simulation)" if is_simulation else ""
        if spd_before == 0:
            bias_result = "No measurable bias before mitigation (SPD = 0)."
        elif bias_reduction >= 50:
            bias_result = (
                f"Bias{sim_suffix} was substantially reduced. SPD moved from "
                f"{spd_before:.3f} to {spd_after:.3f} — "
                f"a {bias_reduction:.0f}% reduction."
            )
        elif bias_reduction >= 10:
            bias_result = (
                f"Bias{sim_suffix} was partially reduced. SPD moved from "
                f"{spd_before:.3f} to {spd_after:.3f} — "
                f"a {bias_reduction:.0f}% improvement."
            )
        elif bias_reduction > 0:
            bias_result = (
                f"Modest bias reduction{sim_suffix} ({bias_reduction:.0f}%). "
                f"SPD moved from {spd_before:.3f} to {spd_after:.3f}."
            )
        else:
            bias_result = (
                f"This technique did not reduce bias for '{sensitive_attr}'"
                f"{sim_suffix}. SPD remained at approximately {spd_after:.3f}."
            )

        if acc_before is None or acc_after is None:
            acc_result = "Performance metrics are not available."
        else:
            acc_delta = acc_after - acc_before
            sim_note = " (simulation)" if is_simulation else ""
            if abs(acc_delta) < 0.005:
                acc_result = (
                    f"Accuracy was virtually unchanged "
                    f"({acc_before:.1%} → {acc_after:.1%}){sim_note}."
                )
            elif acc_delta < 0:
                acc_result = (
                    f"Accuracy dropped from {acc_before:.1%} to "
                    f"{acc_after:.1%} ({abs(acc_delta) * 100:.1f}% reduction). "
                    f"Typical fairness-accuracy trade-off{sim_note}."
                )
            else:
                acc_result = (
                    f"Accuracy improved slightly from {acc_before:.1%} to "
                    f"{acc_after:.1%}{sim_note}."
                )

        graph_explanation = (
            "The chart shows SPD and DI before and after mitigation. "
            + (
                "Reweighing values are from actual dataset outcome distributions. "
                "Threshold values are from the real model's probability scores."
                if not is_simulation else
                "Threshold values are from an internal GBM simulation. "
                "EOD/AOD are included where model predictions are available."
            )
        )

        return {
            "how_it_works": how_it_works,
            "bias_result": bias_result,
            "acc_result": acc_result,
            "graph_explanation": graph_explanation,
            "summary": f"{bias_result} {acc_result}",
        }

    # ── Legacy static helper (kept for router compatibility) ──────────────────

    @staticmethod
    def effects(before: dict, after: dict) -> dict:
        """Legacy method. Compute delta metrics between before and after."""

        def delta(key: str):
            b = before.get(key)
            a = after.get(key)
            if b is None or a is None:
                return None
            return round(float(a) - float(b), 4)

        spd_before = abs(before.get("SPD") or before.get("spd") or 0)
        spd_after = abs(after.get("SPD") or after.get("spd") or 0)
        bias_reduction_pct = (
            round((spd_before - spd_after) / spd_before * 100, 2)
            if spd_before > 0 else 0.0
        )
        acc_before = before.get("accuracy", 1) or 1
        acc_after = after.get("accuracy", 1) or 1
        accuracy_retained_pct = (
            round(acc_after / acc_before * 100, 2) if acc_before > 0 else 100.0
        )

        return {
            "accuracy_delta": delta("accuracy"),
            "precision_delta": delta("precision"),
            "recall_delta": delta("recall"),
            "f1_delta": delta("f1"),
            "spd_delta": delta("SPD") or delta("spd"),
            "bias_reduction_pct": bias_reduction_pct,
            "accuracy_retained_pct": accuracy_retained_pct,
        }
