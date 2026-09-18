"""
FairEnough — Gemini Chat Router

POST /api/detect-scenario  — classify dataset domain via Google Gemini
POST /api/gemini-explain   — generate manager-friendly bias explanation
POST /api/gemini-chat      — multi-turn Bias Copilot conversation
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from services.gemini_service import GeminiService

router = APIRouter(tags=["Gemini AI"])

gemini = GeminiService()


# ── Request models ────────────────────────────────────────────────────────────

class DetectScenarioRequest(BaseModel):
    session_id: str
    columns: list[str]


class GeminiExplainRequest(BaseModel):
    session_id: str
    sensitive_attr: str


class ChatMessage(BaseModel):
    role: str    # "user" or "model"
    content: str


class GeminiChatRequest(BaseModel):
    session_id: str
    message: str
    history: list[ChatMessage] = []
    context: dict[str, Any] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/gemini-test
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/gemini-test", status_code=status.HTTP_200_OK)
async def gemini_test() -> dict[str, Any]:
    """
    Diagnostic endpoint to test if Gemini API is working properly.
    """
    try:
        from services.gemini_service import _model
        response = _model.generate_content("Say hello in one word")
        return {"status": "ok", "response": response.text.strip()}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# POST /api/detect-scenario
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/detect-scenario", status_code=status.HTTP_200_OK)
async def detect_scenario(
    body: DetectScenarioRequest,
    request: Request,
) -> dict[str, Any]:
    """
    Use Google Gemini to classify the dataset into a domain scenario.
    Result is stored in the session for downstream use in explanations.
    """
    sessions: dict = request.app.state.sessions

    if body.session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{body.session_id}' not found. Upload a CSV first.",
        )

    try:
        result = gemini.detect_scenario(columns=body.columns)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Gemini API error during scenario detection: {exc}",
        )

    sessions[body.session_id]["scenario"] = result
    return {"session_id": body.session_id, **result}


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/gemini-explain
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/gemini-explain", status_code=status.HTTP_200_OK)
async def gemini_explain(
    body: GeminiExplainRequest,
    request: Request,
) -> dict[str, Any]:
    """
    Generate a 3-paragraph plain-English bias explanation for a given
    sensitive attribute using the session's analysis results.
    """
    sessions: dict = request.app.state.sessions

    if body.session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{body.session_id}' not found.",
        )

    session = sessions[body.session_id]

    # ── Gather metrics for this attribute ─────────────────────────────────────
    bias_results: dict = session.get("bias_results", {})
    metrics_per_attr: dict = bias_results.get("metrics_per_attr", {})

    if body.sensitive_attr not in metrics_per_attr:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"No analysis results found for '{body.sensitive_attr}'. "
                "Run /api/analyze first."
            ),
        )

    attr_metrics = metrics_per_attr[body.sensitive_attr]
    if "error" in attr_metrics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Analysis error for '{body.sensitive_attr}': {attr_metrics['error']}",
        )

    # ── Gather explanation and scenario ───────────────────────────────────────
    explanations: dict = session.get("explanations", {})
    plain_reason: str = explanations.get(body.sensitive_attr, {}).get(
        "plain_reason",
        "No prior explanation available. Analysis shows measurable bias in this attribute.",
    )

    scenario_data: dict = session.get("scenario", {})
    scenario: str = scenario_data if isinstance(scenario_data, str) else scenario_data.get("scenario", "other")

    # ── Call Gemini ───────────────────────────────────────────────────────────
    try:
        explanation = gemini.explain_bias(
            metrics=attr_metrics,
            sensitive_attr=body.sensitive_attr,
            scenario=scenario,
            plain_reason=plain_reason,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Gemini API error: {exc}",
        )

    # ── Persist in session ────────────────────────────────────────────────────
    if "gemini_explanations" not in session:
        session["gemini_explanations"] = {}
    session["gemini_explanations"][body.sensitive_attr] = explanation

    return {
        "session_id": body.session_id,
        "sensitive_attr": body.sensitive_attr,
        "explanation": explanation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/gemini-chat
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/gemini-chat", status_code=status.HTTP_200_OK)
async def gemini_chat(
    body: GeminiChatRequest,
    request: Request,
) -> dict[str, Any]:
    """
    Multi-turn Bias Copilot conversation.

    The Copilot receives a condensed session context (audit score, scenario,
    top metrics) injected as a system preamble so it can answer questions
    like "Why is gender biased?" or "Which mitigation should I use?"
    """
    sessions: dict = request.app.state.sessions
    session = sessions.get(body.session_id)

    # ── Build condensed context (with graceful client-context fallback across server reloads) ───
    if session is not None:
        bias_results: dict = session.get("bias_results", {})
        scenario_data = session.get("scenario", {})
        scenario_str: str = scenario_data if isinstance(scenario_data, str) else scenario_data.get("scenario", "other")
        mitigation: dict = session.get("mitigation_results", {})

        metrics_summary: dict = {}
        for attr, m in bias_results.get("metrics_per_attr", {}).items():
            if isinstance(m, dict) and "spd" in m:
                metrics_summary[attr] = {
                    "spd": m.get("spd"),
                    "di": m.get("di"),
                    "severity": m.get("severity"),
                    "legal_flag": m.get("legal_flag"),
                }

        session_context = {
            "filename": session.get("filename", "unknown"),
            "row_count": session.get("row_count"),
            "target_col": session.get("target_col"),
            "sensitive_attrs": session.get("sensitive_attrs", []),
            "audit_score": bias_results.get("audit_score"),
            "grade": bias_results.get("grade"),
            "overall_severity": bias_results.get("overall_severity"),
            "scenario": scenario_str,
            "metrics_summary": metrics_summary,
            "metrics_per_attr": bias_results.get("metrics_per_attr", {}),
            "mitigation_winner": mitigation.get("winner"),
        }
    elif body.context:
        # Recover context directly from client store state when in-memory session was recycled
        session_context = body.context
    else:
        session_context = {
            "scenario": "general tabular fairness",
            "audit_score": "N/A",
            "overall_severity": "moderate",
            "metrics_per_attr": {},
        }

    # ── Format history ────────────────────────────────────────────────────────
    history = [
        {"role": msg.role, "content": msg.content}
        for msg in body.history
        if msg.role in ("user", "model")
    ]

    # ── Call Gemini chat ──────────────────────────────────────────────────────
    try:
        reply = gemini.chat(
            user_message=body.message,
            history=history,
            session_context=session_context,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Gemini API error: {exc}",
        )

    return {"session_id": body.session_id, "reply": reply}
