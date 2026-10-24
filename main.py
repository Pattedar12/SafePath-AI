"""
SafePath AI — Intelligent Mobility & Pedestrian Safety Platform
================================================================
Final-Year CSE Major Project Entry Point

Run this file directly in PyCharm (Shift + F10 or click the green Play button).
It launches the SafePath AI platform server and automatically opens the
dashboard in your browser.
"""

import os
import sys
import time
import webbrowser
import threading
from pathlib import Path

# Add project root and backend to python path for clean PyCharm imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def open_browser(url: str, delay_seconds: float = 1.8):
    """Open web browser after the server has initialized."""
    time.sleep(delay_seconds)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def run_platform(host: str = "127.0.0.1", port: int = 8000):
    """Run the SafePath AI FastAPI platform serving both backend and React frontend."""
    banner = f"""
===============================================================================
  [+] SAFEPATH AI -- Intelligent Pedestrian Safety & Mobility Platform
  Final-Year CSE Major Project Platform
===============================================================================
  [+] Project Root : {PROJECT_ROOT}
  [+] Web Dashboard: http://{host}:{port}/
  [+] API Docs     : http://{host}:{port}/docs
  [+] PyCharm Run  : Press the Red Stop button or Ctrl+C to terminate
===============================================================================
    """
    print(banner)


    # Launch browser automatically in PyCharm
    threading.Thread(
        target=open_browser,
        args=(f"http://{host}:{port}/", 1.8),
        daemon=True
    ).start()

    import uvicorn
    # Use backend.server:app so hot-reload and module discovery work smoothly in PyCharm
    uvicorn.run(
        "backend.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


def run_legacy_streamlit():
    """Fallback runner if user explicitly wishes to view the original Streamlit monolith."""
    import subprocess
    print("\n[+] Launching legacy Streamlit interface (main_backup.py)...")
    backup_path = PROJECT_ROOT / "main_backup.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(backup_path)])


if __name__ == "__main__":
    # Check if user passed --streamlit flag in PyCharm Run Configuration
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("--streamlit", "-s", "streamlit"):
        run_legacy_streamlit()
    else:
        run_platform(host="127.0.0.1", port=8000)
