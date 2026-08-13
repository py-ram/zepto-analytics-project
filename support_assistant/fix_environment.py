# support_assistant/fix_environment.py
import subprocess
import sys

def fix_environment():
    """Fix dependency version conflicts"""
    
    print("Fixing dependency versions...")
    print("="*50)
    
    packages = [
        "huggingface-hub==0.16.4",
        "sentence-transformers==2.2.2",
        "transformers==4.34.0",
        "tokenizers==0.14.1",
        "chromadb==0.4.15",
    ]
    
    for package in packages:
        print(f"\nInstalling {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", package],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ Successfully installed {package}")
        else:
            print(f"✗ Failed to install {package}")
            print(f"Error: {result.stderr}")
    
    print("\n" + "="*50)
    print("Environment fix complete!")
    print("\nNow try running the assistant again:")
    print("  python test_assistant.py")
    print("  # or")
    print("  python main.py")

if __name__ == "__main__":
    fix_environment()