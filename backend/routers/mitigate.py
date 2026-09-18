"""
FairEnough — Mitigate Router

POST /api/mitigate
  Runs BiasMitigator (reweighing + threshold adjustment).
  If validation flagged fallback_needed, also runs FairlearnFallback
  to enrich the response with extended fairness metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from services.mitigator import BiasMitigator
from services.fairlearn_fallback import FairlearnFallback

router = APIRouter(tags=["Mitigation"])

mitigator = BiasMitigator()
fallback = FairlearnFallback()


class MitigateRequest(BaseModel):
    session_id: str
    target_col: str
    sensitive_attr: str
    model_id: str | None = None
    simulate_threshold: bool = False


@router.post("/mitigate", status_code=status.HTTP_200_OK)
async def mitigate(
    body: MitigateRequest,
    request: Request,
) -> dict[str, Any]:
    """
    Run reweighing and threshold-adjustment mitigation strategies.

    If the dataset requires a Fairlearn fallback (continuous or multiclass
    target), fairlearn metrics are also computed and included in the response.
    """
    sessions: dict = request.app.state.sessions

    # ── Session lookup ────────────────────────────────────────────────────────
    if body.session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{body.session_id}' not found. Upload a CSV first.",
        )

    session = sessions[body.session_id]
    df: pd.DataFrame = session["df"].copy()

    # ── Column validation ─────────────────────────────────────────────────────
    print(f"[Mitigate] Running mitigation for attr='{body.sensitive_attr}' "
          f"target='{body.target_col}' session='{body.session_id}'")

    if body.target_col not in df.columns:
        for c in df.columns:
            if c.lower() == body.target_col.lower():
                body.target_col = c
                break
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"The target column '{body.target_col}' was not found in the dataset.",
            )

    if body.sensitive_attr not in df.columns:
        for c in df.columns:
            if c.lower() == body.sensitive_attr.lower():
                body.sensitive_attr = c
                break
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"The sensitive attribute column '{body.sensitive_attr}' was not found in the dataset.",
            )

    # ── Check if fairlearn fallback is needed ─────────────────────────────────
    validation: dict = session.get("validation", {})
    fallback_needed: bool = validation.get("fallback_needed", False)

    fairlearn_result: dict | None = None
    if fallback_needed:
        try:
            fairlearn_result = fallback.analyze(
                df=df,
                target_col=body.target_col,
                sensitive_attr=body.sensitive_attr,
            )
        except Exception as exc:
            fairlearn_result = {"error": str(exc), "fallback_used": True}

    # ── Run BiasMitigator ─────────────────────────────────────────────────────
    # Extract predicted_cause from prior analyze() auto-learning
    pattern_preds   = session.get("pattern_predictions", {})
    predicted_cause = pattern_preds.get(body.sensitive_attr, {}).get("predicted_cause")

    # BiasMitigator works on any tabular data regardless of fallback flag
    # Extract analysis baseline for this sensitive attribute
    # This ensures mitigation "before" values match the analysis page exactly.
    bias_results = session.get("bias_results", {})
    attr_metrics = bias_results.get("metrics_per_attr", {}).get(body.sensitive_attr, {})
    baseline_spd = attr_metrics.get("spd") or attr_metrics.get("SPD")
    baseline_di  = attr_metrics.get("di")  or attr_metrics.get("DI")
    baseline_gs  = attr_metrics.get("group_stats")
    print(f"[Mitigate] Analysis baseline for {body.sensitive_attr!r}: "
          f"SPD={baseline_spd}, DI={baseline_di}")

    # ── Real model + predictions integration ─────────────────────────────────
    df_with_pred: pd.DataFrame | None = session.get("df_with_predictions")

    # Multi-stage model resolution with disk fallback
    from services.model_storage import resolve_model
    real_model, resolved_mid = resolve_model(
        sessions=sessions,
        session_id=body.session_id,
        model_id=body.model_id or session.get("model_id"),
    )

    if real_model is not None:
        session["model"] = real_model
        if resolved_mid:
            session["model_id"] = resolved_mid
        print(f"[Mitigate] Resolved model '{resolved_mid}' (type: {type(real_model).__name__}) for session '{body.session_id}'")

    # If real_model is present but df_with_pred is missing, generate predictions automatically
    if real_model is not None and (df_with_pred is None or "__predictions__" not in df_with_pred.columns):
        try:
            column_mapping = session.get("column_mapping")
            prep_rep = session.get("preprocessing_report", {})
            dropped_cols = prep_rep.get("dropped_column_values") or prep_rep.get("zero_variance_cols_dropped")
            from services.feature_resolver import build_model_feature_matrix
            X_base = build_model_feature_matrix(
                model=real_model,
                df=df,
                target_col=body.target_col,
                sensitive_attr=body.sensitive_attr,
                column_mapping=column_mapping,
                dropped_cols=dropped_cols,
            )
            if X_base is not None:
                try:
                    base_preds = real_model.predict(X_base)
                except Exception:
                    base_preds = real_model.predict(getattr(X_base, "values", X_base))
                df_with_pred = df.copy()
                df_with_pred["__predictions__"] = base_preds
                session["df_with_predictions"] = df_with_pred
                print(f"[Mitigate] Auto-generated df_with_predictions for session '{body.session_id}'")
        except Exception as exc:
            print(f"[Mitigate] Note: baseline prediction generation deferred: {exc}")

    print(f"[Mitigate] Real model available: {real_model is not None} "
          f"(type: {getattr(type(real_model), '__name__', 'None')}, "
          f"has predict_proba: {callable(getattr(real_model, 'predict_proba', None))})")
    print(f"[Mitigate] df_with_predictions available: {df_with_pred is not None}")

    allow_simulation = bool(body.simulate_threshold)
    column_mapping = session.get("column_mapping")
    prep_rep = session.get("preprocessing_report", {})
    dropped_cols = prep_rep.get("dropped_column_values") or prep_rep.get("zero_variance_cols_dropped")

    try:
        mitigation_results = mitigator.run_both(
            df=df,
            target_col=body.target_col,
            sensitive_attr=body.sensitive_attr,
            predicted_cause=predicted_cause,
            baseline_spd=baseline_spd,
            baseline_di=baseline_di,
            baseline_group_stats=baseline_gs,
            model=real_model,
            df_with_pred=df_with_pred,
            allow_simulation=allow_simulation,
            column_mapping=column_mapping,
            dropped_cols=dropped_cols,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mitigation failed: {exc}",
        )

    # ── Persist in session ────────────────────────────────────────────────────
    session["mitigation_results"] = mitigation_results
    session["fairlearn_results"] = fairlearn_result

    # Store per-attribute so report can access each one
    if "mitigation" not in session or not isinstance(session["mitigation"], dict):
        session["mitigation"] = {}
    session["mitigation"][body.sensitive_attr] = mitigation_results

    # Also store the last-run attribute for quick access
    session["mitigation"]["__last_attr__"] = body.sensitive_attr
    session["mitigation"]["winner"] = mitigation_results.get("winner")
    session["mitigation"]["winner_reason"] = mitigation_results.get("winner_reason")

    session["winner"]         = mitigation_results.get("winner")
    session["winner_reason"]  = mitigation_results.get("winner_reason", "")

    # ── Build response ────────────────────────────────────────────────────────
    eff_mid = session.get("model_id") or resolved_mid or body.model_id if real_model is not None else None
    response: dict[str, Any] = {
        "session_id": body.session_id,
        "has_real_model": (real_model is not None),
        "model_id": eff_mid,
        "reweigh": _serialise(mitigation_results["reweigh"]),
        "threshold": _serialise(mitigation_results["threshold"]),
        "winner": mitigation_results["winner"],
        "winner_reason": mitigation_results.get("winner_reason"),
        "predicted_cause_used": mitigation_results.get("predicted_cause_used"),
        "fairlearn_used": fallback_needed,
    }

    if fairlearn_result is not None:
        response["fairlearn_metrics"] = _serialise(fairlearn_result)

    return response


# ── JSON serialisation helper ─────────────────────────────────────────────────

def _serialise(obj: Any) -> Any:
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
