#!/usr/bin/env python3
"""Load environment variables from .env file."""

import os
from pathlib import Path

def load_env():
    """Load environment variables from .env file."""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("Error: .env file not found!")
        return False
    
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
                    print(f"Loaded: {key} = {'*' * len(value) if 'KEY' in key else value}")
        return True
    except Exception as e:
        print(f"Error loading .env file: {e}")
        return False

if __name__ == "__main__":
    print("Loading environment variables from .env file...")
    if load_env():
        print("\nEnvironment variables loaded successfully!")
        print(f"OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
        print(f"GOOGLE_API_KEY: {'SET' if os.getenv('GOOGLE_API_KEY') else 'NOT SET'}")
    else:
        print("Failed to load environment variables!")
