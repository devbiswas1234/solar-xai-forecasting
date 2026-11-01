"""
run_app.py — Launch the Solar XAI Forecasting Web Application.

Usage:
    python run_app.py
"""

import os
import sys
import uvicorn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(CURRENT_DIR, "app")
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, APP_DIR)

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    print("=" * 70)
    print("  Starting Solar XAI Forecasting & Grid Intelligence System")
    print(f"  Web Dashboard:  http://{host}:{port}")
    print(f"  API Docs:       http://{host}:{port}/docs")
    print("=" * 70)

    uvicorn.run("backend.main:app", host=host, port=port, reload=False, app_dir=APP_DIR)
