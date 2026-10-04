"""
LogVerse AI Platform Desktop Launcher
=====================================
Run this script to launch the LogVerse AI Desktop Application:
    python launch_desktop.py
"""

import sys
import os
import socket
import subprocess

# Ensure UTF-8 stdout encoding on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def find_available_port(start_port=8501, max_attempts=10):
    """Finds an available TCP port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port

def is_port_in_use(port):
    """Checks if a port is currently listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def main():
    print("==========================================================")
    print("   LAUNCHING LOGVERSE AI PLATFORM DESKTOP PRODUCT   ")
    print("==========================================================")

    # Check if 8501 is already serving
    if is_port_in_use(8501):
        print("ℹ️  LogVerse AI Desktop App is ALREADY RUNNING on http://localhost:8501")
        print("👉 Simply open http://localhost:8501 in your web browser!")
        print("   (Or starting a new instance on an alternative port below...)")
        print("----------------------------------------------------------")

    target_port = find_available_port(8501, max_attempts=20)
    print(f"Starting Streamlit Desktop Interface on http://localhost:{target_port} ...")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", "logverse_desktop_app.py",
        f"--server.port={target_port}",
        "--server.headless=true"
    ]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nLogVerse AI Platform Desktop Application stopped.")

if __name__ == "__main__":
    main()
