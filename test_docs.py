#!/usr/bin/env python3
"""
Test script to verify documentation builds correctly.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and check if it succeeds."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Test documentation build process."""
    print("🚀 Testing LLMBuilder Documentation Build")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("mkdocs.yml").exists():
        print("❌ mkdocs.yml not found. Please run from project root.")
        sys.exit(1)
    
    # Test commands
    tests = [
        ("python -m pip install -r docs/requirements.txt", "Installing documentation dependencies"),
        ("python -m mkdocs build --clean --strict", "Building documentation"),
        ("python -c \"import os; print('Site directory exists:', os.path.exists('site'))\"", "Verifying build output"),
    ]
    
    success_count = 0
    for cmd, description in tests:
        if run_command(cmd, description):
            success_count += 1
        print()
    
    # Summary
    print("=" * 50)
    print(f"📊 Test Results: {success_count}/{len(tests)} tests passed")
    
    if success_count == len(tests):
        print("🎉 All tests passed! Documentation is ready for deployment.")
        print("🚀 You can now push to GitHub to trigger automatic deployment.")
        print("🌐 Or run 'mkdocs serve' to test locally.")
    else:
        print("⚠️  Some tests failed. Please fix the issues before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main()