import os
import re
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
        raise RuntimeError(
            "FATAL: MONGODB_URL environment variable is required. "
            "Please set MONGODB_URL in your hosting environment variables."
        )

    print(f"[Database] Connecting to MongoDB Atlas...")
    client = AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=10000)
    
    # Verify server connectivity
    await client.admin.command('ping')
    
    # Use 'byus' database directly
    db = client.byus
    
    # Create required indexes
    await db.users.create_index("email", unique=True)
    await db.reports.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    await db.reports.create_index("session_id", unique=True)
    print("[Database] Connected successfully to MongoDB Atlas (database: byus)")

async def disconnect_db():
    global client
    if client:
        client.close()
        print("[Database] Disconnected from MongoDB")

def get_db():
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is not available."
        )
    return db
