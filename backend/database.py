import os
import re
import asyncio
import logging
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from fastapi import HTTPException, status

load_dotenv()
_backend_env = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(_backend_env):
    load_dotenv(_backend_env)

logger = logging.getLogger(__name__)

# Sanitize URL: strip any trailing whitespace, newlines, or tabs from dashboard copy-paste
_raw_mongo_url = os.getenv("MONGODB_URL", "")
MONGODB_URL = re.sub(r"\s+", "", _raw_mongo_url) if _raw_mongo_url else ""

JWT_SECRET = os.getenv("JWT_SECRET", "byus_jwt_secret_key_prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", 168))

client = None
db = None

async def connect_db():
    global client, db
    if not MONGODB_URL:
        print("[Database] WARNING: MONGODB_URL environment variable is not set. Running in in-memory session mode.")
        client = None
        db = None
        return

    print("[Database] Connecting to MongoDB Atlas...")
    try:
        client = AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        # Verify server connectivity with strict 5s timeout to prevent deployment hang
        await asyncio.wait_for(client.admin.command('ping'), timeout=5.0)
        
        # Resolve database
        try:
            default_db = client.get_default_database()
            db = default_db if default_db is not None else client.byus
        except Exception:
            db = client.byus
        
        # Create required indexes
        await db.users.create_index("email", unique=True)
        await db.reports.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        await db.reports.create_index("session_id", unique=True)
        print("[Database] Connected successfully to MongoDB Atlas")
    except Exception as exc:
        print(f"[Database] WARNING: Could not connect to MongoDB Atlas ({exc}). Running in in-memory session mode.")
        client = None
        db = None

async def disconnect_db():
    global client
    if client:
        try:
            client.close()
            print("[Database] Disconnected from MongoDB")
        except Exception:
            pass

def get_db():
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is not available. Please verify MongoDB Atlas connection."
        )
    return db
