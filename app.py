"""
Production ASGI Entry Point for Render / Cloud Hosting
Enables 1-click execution for default 'python app.py' or 'uvicorn app:app'.
"""
import os
import uvicorn
from backend.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[AgroAI Production] Starting on 0.0.0.0:{port}")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, log_level="info")
