"""
FairEnough Backend — FastAPI Application Entry Point
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from database import connect_db, disconnect_db
from routers import upload, analyze, mitigate, report, gemini_chat, explain, auth, reports as reports_router

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise in-memory session store on startup
    app.state.sessions = {}
    await connect_db()

    # Ensure data directory always exists (Railway, Docker, local dev)
    import os as _os
    _os.makedirs(_os.path.join(_os.getcwd(), 'data'), exist_ok=True)
    _os.makedirs('data', exist_ok=True)

    # Verify classifier loads correctly
    try:
        from services.bias_pattern_model import get_bias_pattern_classifier, get_stats_from_file
        clf   = get_bias_pattern_classifier()
        stats = get_stats_from_file()
        print(f"[Startup] BiasPatternClassifier: {stats['total_examples']} examples")
        print(f"[Startup] File: {stats['file_path']}")
        print(f"[Startup] File exists: {stats['file_exists']}")
        if stats['total_examples'] < 20:
            print("[Startup] WARNING: Less than 20 training examples. "
                  "Run: python backend/scripts/train_classifier.py")
    except Exception as e:
        print(f"[Startup] BiasPatternClassifier error: {e}")

    yield
    await disconnect_db()


async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": True, "detail": str(exc), "code": 500}
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    content = {"error": True, "detail": str(exc.detail), "code": exc.status_code}
    if exc.status_code == 503:
        content["fallback"] = True
    return JSONResponse(
        status_code=exc.status_code,
        content=content
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="FairEnough API",
        description=(
            "Detect, explain, and mitigate bias in datasets and ML models "
            "powered by Google Gemini."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── Exception Handlers ──────────────────────────────────────────────────
    app.add_exception_handler(Exception, global_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    # Allow frontend domains (or standard origins) with credentials
    env_origins = os.getenv("CORS_ORIGINS", "") or os.getenv("ALLOWED_ORIGINS", "")
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "https://fairenough-rosy.vercel.app",
    ]
    if env_origins and env_origins != "*":
        for o in env_origins.split(","):
            cleaned = o.strip()
            if cleaned and cleaned not in origins:
                origins.append(cleaned)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=r"https://.*\.vercel\.app|http://localhost:\d+|http://127\.0\.0\.1:\d+",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ─────────────────────────────────────────────────────────────
    app.include_router(upload.router, prefix="/api")
    app.include_router(analyze.router, prefix="/api")
    app.include_router(mitigate.router, prefix="/api")
    app.include_router(report.router, prefix="/api")
    app.include_router(gemini_chat.router, prefix="/api")
    app.include_router(explain.router, prefix="/api")
    app.include_router(auth.router)
    app.include_router(reports_router.router)

    # ── Health check ─────────────────────────────────────────────────────────
    @app.get("/api/health", tags=["Health"])
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
