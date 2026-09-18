"""
FairEnough Model Artifact Storage and Resolution Service
Provides disk persistence fallback and multi-stage session resolution for uploaded models.
"""

import os
import io
import joblib
from typing import Any

# Resolve model storage directory
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_CURRENT_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)

# Priority: data/uploaded_models in project root, fallback to backend/data/uploaded_models
MODELS_DIR = os.path.join(_PROJECT_ROOT, "data", "uploaded_models")
os.makedirs(MODELS_DIR, exist_ok=True)


def _ensure_compatibility(model: Any) -> None:
    """Apply sklearn unpickling bridges for cross-version compatibility."""
    try:
        from services.mitigator import _ensure_estimator_compatibility
        _ensure_estimator_compatibility(model)
    except Exception:
        # Basic fallback compatibility
        if "LogisticRegression" in type(model).__name__:
            if not hasattr(model, "multi_class"):
                setattr(model, "multi_class", "auto")
            if getattr(model, "penalty", None) == "deprecated":
                model.penalty = "l2"
            if getattr(model, "l1_ratio", None) is not None and getattr(model, "penalty", "l2") != "elasticnet":
                model.l1_ratio = None


def save_uploaded_model(
    raw_bytes: bytes,
    model: Any,
    model_id: str,
    session_id: str | None = None,
    filename: str = "",
) -> str:
    """
    Persist uploaded model file to disk and return the file path.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    file_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    try:
        with open(file_path, "wb") as f:
            f.write(raw_bytes)
    except Exception as exc:
        print(f"[ModelStorage] Warning: Failed to write model {model_id} to disk: {exc}")

    # Also save session alias if session_id is provided
    if session_id:
        alias_path = os.path.join(MODELS_DIR, f"session_{session_id}.pkl")
        try:
            with open(alias_path, "wb") as f:
                f.write(raw_bytes)
        except Exception:
            pass

    return file_path


def load_model_from_disk(model_id_or_path: str) -> Any | None:
    """
    Load a model from disk by model_id or file path with estimator compatibility bridges.
    """
    target_path = model_id_or_path
    if not os.path.isabs(target_path):
        candidate1 = os.path.join(MODELS_DIR, f"{model_id_or_path}.pkl")
        candidate2 = os.path.join(MODELS_DIR, f"session_{model_id_or_path}.pkl")
        candidate3 = os.path.join(MODELS_DIR, model_id_or_path)
        if os.path.exists(candidate1):
            target_path = candidate1
        elif os.path.exists(candidate2):
            target_path = candidate2
        elif os.path.exists(candidate3):
            target_path = candidate3
        else:
            return None

    if not os.path.exists(target_path):
        return None

    try:
        model = joblib.load(target_path)
        _ensure_compatibility(model)
        return model
    except Exception as exc:
        print(f"[ModelStorage] Failed to load model from {target_path}: {exc}")
        return None


def resolve_model(
    sessions: dict,
    session_id: str | None = None,
    model_id: str | None = None,
) -> tuple[Any | None, str | None]:
    """
    Multi-stage model resolution:
    1. Direct session model lookup
    2. Model lookup by model_id from request or session
    3. Global session scan
    4. Disk storage fallback
    Returns (model_object, resolved_model_id).
    """
    session = sessions.get(session_id, {}) if session_id else {}

    # Stage 1: Already inside session
    if session and "model" in session and session["model"] is not None:
        return session["model"], session.get("model_id") or model_id

    # Stage 2: Look up by model_id in sessions store
    eff_mid = model_id or session.get("model_id")
    if eff_mid and eff_mid in sessions and "model" in sessions[eff_mid] and sessions[eff_mid]["model"] is not None:
        model = sessions[eff_mid]["model"]
        if session:
            session["model"] = model
            session["model_id"] = eff_mid
        return model, eff_mid

    # Stage 3: Scan all sessions for matching session_id or model_id
    if session_id:
        for k, v in sessions.items():
            if isinstance(v, dict) and v.get("model") is not None:
                if v.get("session_id") == session_id or k == eff_mid:
                    model = v["model"]
                    mid = v.get("model_id", k)
                    if session:
                        session["model"] = model
                        session["model_id"] = mid
                    return model, mid

    # Stage 4: Disk storage fallback
    if eff_mid:
        loaded = load_model_from_disk(eff_mid)
        if loaded is not None:
            if session:
                session["model"] = loaded
                session["model_id"] = eff_mid
            sessions[eff_mid] = {
                "model": loaded,
                "model_id": eff_mid,
                "session_id": session_id,
            }
            print(f"[ModelStorage] Restored model {eff_mid} from disk storage fallback")
            return loaded, eff_mid

    if session_id:
        loaded = load_model_from_disk(f"session_{session_id}")
        if loaded is not None:
            mid = session.get("model_id") or f"disk_{session_id[:8]}"
            if session:
                session["model"] = loaded
                session["model_id"] = mid
            sessions[mid] = {
                "model": loaded,
                "model_id": mid,
                "session_id": session_id,
            }
            print(f"[ModelStorage] Restored model for session {session_id} from disk alias")
            return loaded, mid

    return None, None
