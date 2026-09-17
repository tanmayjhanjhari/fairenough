"""
FairEnough Root Entry Point for Render / Cloud Hosts
Forwards to backend/main.py regardless of execution working directory.
"""

import os
import sys

# Ensure backend directory is in Python module search path
backend_dir = os.path.join(os.path.dirname(__file__), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import backend.main as backend_main

app = backend_main.app
create_app = backend_main.create_app

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
