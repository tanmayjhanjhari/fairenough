"""
FairEnough — Explain Router

POST /api/explain
  Runs BiasExplainer for a single sensitive attribute and stores the
  explanation in the session for use by the PDF report and Copilot.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from services.explainer import BiasExplainer
from services.gemini_service import GeminiService

router = APIRouter(tags=["Explanation"])

explainer = BiasExplainer()
gemini = GeminiService()


class ExplainRequest(BaseModel):
    session_id: str
    target_col: str
    sensitive_attr: str


@router.post("/explain", status_code=status.HTTP_200_OK)
async def explain(
    body: ExplainRequest,
    request: Request,
) -> dict[str, Any]:
    """
    Run bias explanation analysis for a single sensitive attribute.

    Returns correlation, proxy features, data imbalance, historical skew,
    positive rate gap, and a plain-English reason string.
    """
    sessions: dict = request.app.state.sessions

    # ── Session lookup ────────────────────────────────────────────────────────
    if body.session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{body.session_id}' not found. Upload a CSV first.",
        )

    session = sessions[body.session_id]

    import pandas as pd
    df: pd.DataFrame = session["df"]

    # ── Column validation (with case-insensitive fallback) ────────
    if body.target_col not in df.columns:
        for c in df.columns:
            if c.lower() == body.target_col.lower():
                body.target_col = c
                break
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Target column '{body.target_col}' not found in dataset.",
            )

    if body.sensitive_attr not in df.columns:
        for c in df.columns:
            if c.lower() == body.sensitive_attr.lower():
                body.sensitive_attr = c
                break
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Sensitive attribute '{body.sensitive_attr}' not found in dataset.",
            )

    # ── Fetch Metrics ─────────────────────────────────────────────────────────
    bias_results = session.get("bias_results", {})
    metrics_per_attr = bias_results.get("metrics_per_attr", {})
    attr_metrics = metrics_per_attr.get(body.sensitive_attr, {})
    if not attr_metrics:
        for k, v in metrics_per_attr.items():
            if k.lower() == body.sensitive_attr.lower():
                attr_metrics = v
                break

    # ── Run explainer ─────────────────────────────────────────────────────────
    try:
        explanation = explainer.explain(
            df=df,
            target_col=body.target_col,
            sensitive_attr=body.sensitive_attr,
            metrics=attr_metrics,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {exc}",
        )

    # ── Call Gemini for narrative explanation ─────────────────────────────────
    scenario_data = session.get("scenario", {})
    scenario = scenario_data if isinstance(scenario_data, str) else scenario_data.get("scenario", "unknown")
    
    plain_reason = explanation.get("plain_reason", "")
    
    gemini_explanation = gemini.explain_bias(
        metrics=attr_metrics,
        sensitive_attr=body.sensitive_attr,
        scenario=scenario,
        plain_reason=plain_reason
    )
    
    explanation["gemini_explanation"] = gemini_explanation
    pattern_preds = session.get("pattern_predictions", {})
    if body.sensitive_attr in pattern_preds:
        explanation["pattern_prediction"] = pattern_preds[body.sensitive_attr]

    # ── Persist in session (keyed by attribute for multi-attr support) ────────
    if "explanations" not in session:
        session["explanations"] = {}
    session["explanations"][body.sensitive_attr] = explanation

    return {"session_id": body.session_id, **explanation}
