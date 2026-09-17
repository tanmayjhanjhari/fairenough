"""
FairEnough — Analyze Router

POST /api/analyze
  1. Validate the dataset with DataValidator
  2. Optionally run model.predict() for prediction-based bias
  3. Run BiasEngine.analyze()
  4. Persist and return results
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from services.validator import DataValidator
from services.bias_engine import BiasEngine
from services.feature_resolver import build_model_feature_matrix

# Import TRAINING_FILE path for learning-stats endpoint
try:
    from services.bias_pattern_model import TRAINING_FILE
except ImportError:
    TRAINING_FILE = ""

router = APIRouter(tags=["Analysis"])

validator = DataValidator()
engine = BiasEngine()


# ── Request body ──────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    session_id: str
    target_col: str
    sensitive_attrs: list[str] = Field(min_length=1)
    model_id: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/analyze
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/analyze", status_code=status.HTTP_200_OK)
async def analyze(
    body: AnalyzeRequest,
    request: Request,
) -> dict[str, Any]:
    """
    Run the full FairEnough bias analysis pipeline.

    Steps:
    1. Retrieve the dataset from the session store.
    2. Validate target column and sensitive attributes with DataValidator.
    3. If model_id is supplied, run model.predict() and attach predictions
       as the ``__predictions__`` column.
    4. Run BiasEngine.analyze() and store results back in the session.
    """
    sessions: dict = request.app.state.sessions

    # ── 1. Retrieve session data ───────────────────────────────────────────────
    if body.session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{body.session_id}' not found. Please upload a CSV first.",
        )

    session = sessions[body.session_id]
    df: pd.DataFrame = session["df"].copy()

    # If detected_scenario is present in preprocessing_report, set session["scenario"]
    preprocessing_report = session.get("preprocessing_report", {})
    if preprocessing_report.get("detected_scenario"):
        session["scenario"] = preprocessing_report["detected_scenario"]

    # ── Basic column existence checks ─────────────────────────────────────────
    if body.target_col not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Target column '{body.target_col}' not found in dataset.",
        )

    missing_attrs = [a for a in body.sensitive_attrs if a not in df.columns]
    if missing_attrs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Sensitive attribute(s) not found in dataset: {missing_attrs}",
        )

    # ── 2. Validate ───────────────────────────────────────────────────────────
    validation = validator.validate(df, body.target_col, body.sensitive_attrs)

    # Hard-stop if the dataset has fewer than the absolute minimum rows
    if validation["row_count"] < DataValidator.MIN_ROWS_ABSOLUTE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Dataset has only {validation['row_count']} rows. "
                f"A minimum of {DataValidator.MIN_ROWS_ABSOLUTE} rows is required."
            ),
        )

    # ── 3. Optional model predictions ─────────────────────────────────────────
    model_used = False
    use_predictions = False
    effective_model_id = body.model_id or session.get("model_id")

    model_session = _find_model(sessions, effective_model_id, body.session_id)
    if body.model_id and model_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{body.model_id}' not found. Please upload the model first.",
        )

    if model_session is not None and "model" in model_session:
        model = model_session["model"]
        column_mapping = session.get("column_mapping")

        try:
            X = build_model_feature_matrix(
                model=model,
                df=df,
                target_col=body.target_col,
                sensitive_attr=body.sensitive_attrs[0] if body.sensitive_attrs else None,
                column_mapping=column_mapping,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )

        try:
            predictions = model.predict(X)
            df["__predictions__"] = predictions
            use_predictions = True
            model_used = True
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model prediction failed: {exc}",
            )
    else:
        # Without a model, there are no predictions
        use_predictions = False
        model_used = False

    # ── 4. Run bias engine ────────────────────────────────────────────────────
    try:
        bias_results = engine.analyze(
            df=df,
            target_col=body.target_col,
            sensitive_attrs=body.sensitive_attrs,
            use_predictions=use_predictions,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bias analysis failed: {exc}",
        )

    # ── 5. Persist results ────────────────────────────────────────────────────
    session["validation"] = validation
    session["bias_results"] = bias_results
    session["target_col"] = body.target_col
    session["sensitive_attrs"] = body.sensitive_attrs
    session["df_with_predictions"] = df if model_used else None  # needed for mitigation only when model used
    if model_used:
        session["model"] = model
        session["model_id"] = effective_model_id or model_session.get("model_id", body.model_id)
        print(f"[Analyze] Persisted real model in session '{body.session_id}' (model_id={session.get('model_id')})")
    # Store top-level keys so report generator can access directly
    session["audit_score"]      = bias_results["audit_score"]
    session["overall_severity"] = bias_results["overall_severity"]
    session["grade"]            = bias_results["grade"]
    session["grade_label"]      = bias_results.get("grade_label", "")
    session["metrics_per_attr"] = bias_results["metrics_per_attr"]

    # ── 6. Build response ─────────────────────────────────────────────────────
    raw_scenario = session.get("scenario", "other")
    scenario_str = raw_scenario if isinstance(raw_scenario, str) else raw_scenario.get("scenario", "other")

    response: dict[str, Any] = {
        "session_id": body.session_id,
        "validation": validation,
        "metrics_per_attr": _serialise(bias_results["metrics_per_attr"]),
        "audit_score": bias_results["audit_score"],
        "grade": bias_results["grade"],
        "overall_severity": bias_results["overall_severity"],
        "grade_label": bias_results.get("grade_label", ""),
        "model_used": model_used,
        "scenario": scenario_str,
    }

    # ── 7. Auto-learning + pattern predictions ───────────────────────────────
    import threading
    from services.bias_pattern_model import get_bias_pattern_classifier

    results = response  # alias for clarity in the block below

    try:
        classifier = get_bias_pattern_classifier()
        dataset_name = session.get("filename", "unknown")
        scenario     = session.get("scenario", "other")
        metrics_per_attr = bias_results.get("metrics_per_attr", {})

        pattern_predictions = {}
        # Severity ordering for comparison
        _SEV_RANK = {"low": 0, "medium": 1, "high": 2}

        for attr, m in metrics_per_attr.items():
            if "error" in m:
                continue

            spd         = abs(m.get("SPD", 0) or 0)
            di          = m.get("DI", 1.0) or 1.0

            # Compute proxy correlations from the dataframe if not in engine output
            proxies     = m.get("proxy_features", [])
            if not proxies and attr in df.columns:
                feature_cols = [
                    c for c in df.select_dtypes(include="number").columns
                    if c not in [attr, body.target_col, "__predictions__"]
                ]
                if body.target_col in df.columns:
                    try:
                        attr_encoded = pd.Categorical(df[attr].astype(str)).codes
                        for fc in feature_cols[:10]:  # top 10 numeric features
                            r = abs(float(df[fc].corr(pd.Series(attr_encoded, index=df.index))))
                            if not np.isnan(r):
                                proxies.append({"feature": fc, "correlation": round(r, 4)})
                        proxies.sort(key=lambda x: -x["correlation"])
                    except Exception:
                        pass

            top_proxy_r = proxies[0].get("correlation", 0.0) if proxies else 0.0
            proxy_count = sum(1 for p in proxies if p.get("correlation", 0) > 0.2)
            group_stats = m.get("group_stats", {})
            rates       = [g.get("positive_rate", 0) for g in group_stats.values()]
            rate_var    = float(np.std(rates)) if len(rates) > 1 else 0.0
            counts      = [g.get("count", 1) for g in group_stats.values()]
            group_ratio = (min(counts) / max(counts)) if counts and max(counts) > 0 else 1.0

            # Get ML prediction for this attribute
            pred = classifier.predict(
                spd, di, top_proxy_r, group_ratio,
                proxy_count, rate_var, scenario
            )
            pattern_predictions[attr] = pred

            # ── Real impact: merge classifier output into the metric ──────────
            # Override per-attribute severity if ML has high confidence
            classifier_sev  = pred.get("predicted_severity", "")
            classifier_conf = pred.get("severity_confidence", 0)
            engine_sev      = m.get("severity", "low")

            if classifier_conf >= 70 and classifier_sev in _SEV_RANK:
                # Take the *higher* of engine severity and ML severity
                if _SEV_RANK.get(classifier_sev, 0) > _SEV_RANK.get(engine_sev, 0):
                    m["severity"] = classifier_sev
                    m["severity_source"] = "ml_classifier"
                else:
                    m["severity_source"] = "bias_engine"
            else:
                m["severity_source"] = "bias_engine"

            # Attach ML cause label to the metric (visible in API response)
            m["predicted_cause"]       = pred.get("predicted_cause", "")
            m["cause_label"]           = pred.get("cause_label", "")
            m["classifier_confidence"] = pred.get("confidence_pct", 0)
            m["proxy_features"]        = proxies  # computed proxies available downstream

            # Auto-learn in background — never blocks the API response
            t = threading.Thread(
                target=classifier.add_training_example,
                args=(spd, di, top_proxy_r, group_ratio,
                      proxy_count, rate_var, scenario,
                      dataset_name, attr),
                daemon=True
            )
            t.start()

        # Recompute overall_severity from updated per-attr severities
        updated_severities = [
            m.get("severity", "low")
            for m in metrics_per_attr.values()
            if isinstance(m, dict) and "severity" in m and "error" not in m
        ]
        if updated_severities:
            if "high" in updated_severities:
                updated_overall = "high"
            elif "medium" in updated_severities:
                updated_overall = "medium"
            else:
                updated_overall = "low"
            results["overall_severity"] = updated_overall

        # Store predictions in session so mitigator can use them
        session["pattern_predictions"] = pattern_predictions

        # Re-serialise metrics_per_attr since we mutated it in-place
        results["metrics_per_attr"] = _serialise(metrics_per_attr)
        results["pattern_predictions"] = pattern_predictions

    except Exception as e:
        # Auto-learning failure must NEVER break the analysis response
        print(f"[AutoLearn] Silent failure: {e}")
        results["pattern_predictions"] = {}

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# GET /api/learning-stats
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/learning-stats", status_code=status.HTTP_200_OK)
async def learning_stats() -> dict:
    """
    Return statistics about the BiasPatternClassifier's training data.
    Reads from the JSON file first so the count is never 0 after a restart.
    """
    from services.bias_pattern_model import get_bias_pattern_classifier, get_stats_from_file

    # Read from file first -- never returns 0 after restart
    stats = get_stats_from_file()

    # Try live classifier -- use whichever has more examples
    try:
        clf = get_bias_pattern_classifier()
        live = clf.get_stats()
        if live.get("total_examples", 0) >= stats.get("total_examples", 0):
            stats = live
            stats["file_path"]  = TRAINING_FILE
            stats["file_exists"] = os.path.exists(TRAINING_FILE)
    except Exception as e:
        stats["classifier_error"] = str(e)

    return stats


def _find_model(
    sessions: dict,
    model_id: str | None,
    session_id: str,
) -> dict | None:
    """
    Look for the model in the given session first, then in the top-level store.
    """
    session = sessions.get(session_id, {})
    # Check inside session directly
    if "model" in session and session["model"] is not None:
        return session

    if model_id:
        if session.get("model_id") == model_id and "model" in session:
            return session
        standalone = sessions.get(model_id)
        if standalone and "model" in standalone:
            return standalone

    # Check session's stored model_id
    stored_mid = session.get("model_id")
    if stored_mid and stored_mid in sessions and "model" in sessions[stored_mid]:
        return sessions[stored_mid]

    # Global search for any model linked to this session
    for k, v in sessions.items():
        if isinstance(v, dict) and "model" in v and v["model"] is not None:
            if v.get("session_id") == session_id:
                return v

    return None


def _serialise(obj: Any) -> Any:
    """Recursively convert numpy scalars to native Python for JSON safety."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return _serialise(obj.tolist())
    return obj
