import os
import sys

# Add the parent directory (querygym root) to Python path so `import app` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from app import app

def main():
    """Entry point for openenv HF Space validator"""
    uvicorn.run("app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
