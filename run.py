"""
Universal Application Launcher for AI Crop Predictor
Manages virtual environment verification, dependencies, server initialization, and browser launch.
"""

import sys
import os
import subprocess
import webbrowser
import time
import socket

# Ensure standard streams handle UTF-8 cleanly on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def is_port_in_use(port: int) -> bool:
    """Check if a local port is already occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find first available TCP port starting from start_port."""
    for p in range(start_port, start_port + max_attempts):
        if not is_port_in_use(p):
            return p
    return start_port

def ensure_venv():
    """Ensure script is executed within project virtual environment."""
    venv_dir = os.path.join(os.path.dirname(__file__), ".venv")
    if sys.platform == "win32":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")
        
    # If running with global python and .venv python exists, re-execute with .venv python
    if os.path.exists(venv_python) and sys.executable.lower() != venv_python.lower():
        print(f"[Launcher] Switching to virtual environment python: {venv_python}")
        os.execv(venv_python, [venv_python] + sys.argv)

def check_dependencies():
    """Verify that essential packages are importable."""
    required = ["fastapi", "uvicorn", "torch", "torchvision", "cv2", "PIL", "reportlab", "httpx"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
            
    if missing:
        print(f"[Launcher] Notice: Missing packages detected: {missing}")
        print("[Launcher] Installing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def main():
    print("=" * 65)
    print(" [AgroAI] Crop Disease Diagnosis & Prediction Platform")
    print("=" * 65)
    
    ensure_venv()
    check_dependencies()
    
    import uvicorn
    
    # In cloud environments (e.g. Render, Railway, Docker), listen on 0.0.0.0 and PORT env
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    url = f"http://localhost:{port}"
    
    print(f"\n[Launcher] Starting AgroAI Server at: http://{host}:{port}")
    print(f"[Launcher] Press CTRL+C to stop the server.\n")
    
    # Launch browser only on local interactive desktops
    if "PORT" not in os.environ and sys.platform == "win32":
        def open_browser():
            time.sleep(1.2)
            try:
                webbrowser.open(url)
                print(f"[Launcher] Opened browser at {url}")
            except Exception as e:
                print(f"[Launcher] Could not automatically open browser: {e}")
                
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
    
    # Run Uvicorn server
    uvicorn.run("backend.main:app", host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
