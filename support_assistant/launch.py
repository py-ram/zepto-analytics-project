import webbrowser
import threading
import subprocess
import sys
import time

def open_browser():
    """Open the FastAPI Swagger UI"""
    time.sleep(3)  # Wait for server to start
    webbrowser.open("http://localhost:7860/docs")
    print("\n✓ Browser opened with Swagger UI at http://localhost:7860/docs")

if __name__ == "__main__":
    print("Starting Zepto Support Assistant...")
    print("Swagger UI will open automatically in your browser")
    print("If it doesn't, manually go to: http://localhost:7860/docs")
    
    # Start browser in background
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start the FastAPI server
    subprocess.run([sys.executable, "main.py"])