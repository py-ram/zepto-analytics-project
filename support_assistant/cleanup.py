# support_assistant/cleanup.py
import os
import shutil
import sys

def cleanup():
    """Clean up corrupted ChromaDB data and other temporary files"""
    
    print("Cleaning up support assistant directory...")
    print("="*50)
    
    # Remove ChromaDB directories
    chroma_dirs = ["./chroma_db", "./chroma_db_new", "./chroma"]
    for dir_path in chroma_dirs:
        if os.path.exists(dir_path):
            print(f"Removing {dir_path}...")
            try:
                shutil.rmtree(dir_path)
                print(f"✓ Removed {dir_path}")
            except Exception as e:
                print(f"✗ Failed to remove {dir_path}: {e}")
    
    # Remove Python cache files
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            print(f"Removing {pycache_path}...")
            try:
                shutil.rmtree(pycache_path)
                print(f"✓ Removed {pycache_path}")
            except Exception as e:
                print(f"✗ Failed to remove {pycache_path}: {e}")
    
    print("\n" + "="*50)
    print("Cleanup complete!")
    print("\nNow run the assistant again:")
    print("  python test_assistant.py")
    print("  # or")
    print("  python main.py")

if __name__ == "__main__":
    cleanup()