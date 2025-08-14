#!/usr/bin/env python3
"""
Development Environment Setup Script for LLMBuilder

This script helps contributors set up their development environment
with all necessary dependencies and tools.

Usage:
    python scripts/setup_dev.py [--full] [--check-only]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def run_command(
    cmd: List[str], check: bool = True, capture_output: bool = False
) -> Tuple[int, str, str]:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        return (
            result.returncode,
            result.stdout if capture_output else "",
            result.stderr if capture_output else "",
        )
    except subprocess.CalledProcessError as e:
        return (
            e.returncode,
            e.stdout if capture_output else "",
            e.stderr if capture_output else "",
        )
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        return 1, "", f"Command not found: {cmd[0]}"


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    print("Checking Python version...")

    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(
            f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher."
        )
        return False

    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_git() -> bool:
    """Check if git is available."""
    print("Checking git...")

    returncode, _, _ = run_command(
        ["git", "--version"], check=False, capture_output=True
    )
    if returncode != 0:
        print("❌ Git is not installed or not in PATH")
        return False

    print("✅ Git is available")
    return True


def setup_virtual_environment() -> bool:
    """Set up virtual environment if not already in one."""
    print("Checking virtual environment...")

    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        print("✅ Already in a virtual environment")
        return True

    print("⚠️  Not in a virtual environment. It's recommended to use one.")
    response = input("Create a virtual environment? (y/N): ").lower().strip()

    if response == "y":
        print("Creating virtual environment...")
        returncode, _, _ = run_command([sys.executable, "-m", "venv", "venv"])
        if returncode != 0:
            print("❌ Failed to create virtual environment")
            return False

        print("✅ Virtual environment created at ./venv")
        print("Please activate it and run this script again:")
        if os.name == "nt":
            print("  venv\\Scripts\\activate")
        else:
            print("  source venv/bin/activate")
        return False

    return True


def install_package_dev() -> bool:
    """Install the package in development mode."""
    print("Installing LLMBuilder in development mode...")

    returncode, _, _ = run_command([sys.executable, "-m", "pip", "install", "-e", "."])
    if returncode != 0:
        print("❌ Failed to install LLMBuilder in development mode")
        return False

    print("✅ LLMBuilder installed in development mode")
    return True


def install_dev_dependencies() -> bool:
    """Install development dependencies."""
    print("Installing development dependencies...")

    returncode, _, _ = run_command(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev]"]
    )
    if returncode != 0:
        print("❌ Failed to install development dependencies")
        return False

    print("✅ Development dependencies installed")
    return True


def install_optional_dependencies(full: bool = False) -> bool:
    """Install optional dependencies."""
    if full:
        print("Installing all optional dependencies...")
        returncode, _, _ = run_command(
            [sys.executable, "-m", "pip", "install", "-e", ".[all]"]
        )
        if returncode != 0:
            print("❌ Failed to install optional dependencies")
            return False
        print("✅ All optional dependencies installed")
    else:
        print("Skipping optional dependencies (use --full to install all)")

    return True


def setup_pre_commit() -> bool:
    """Set up pre-commit hooks."""
    print("Setting up pre-commit hooks...")

    # Check if pre-commit is installed
    returncode, _, _ = run_command(
        ["pre-commit", "--version"], check=False, capture_output=True
    )
    if returncode != 0:
        print("Installing pre-commit...")
        returncode, _, _ = run_command(
            [sys.executable, "-m", "pip", "install", "pre-commit"]
        )
        if returncode != 0:
            print("❌ Failed to install pre-commit")
            return False

    # Install pre-commit hooks
    returncode, _, _ = run_command(["pre-commit", "install"])
    if returncode != 0:
        print("❌ Failed to install pre-commit hooks")
        return False

    print("✅ Pre-commit hooks installed")
    return True


def run_initial_checks() -> bool:
    """Run initial code quality checks."""
    print("Running initial code quality checks...")

    # Run pre-commit on all files
    print("Running pre-commit checks...")
    returncode, _, _ = run_command(["pre-commit", "run", "--all-files"], check=False)
    if returncode != 0:
        print("⚠️  Some pre-commit checks failed. This is normal for initial setup.")
        print("   Run 'pre-commit run --all-files' again after fixing any issues.")
    else:
        print("✅ All pre-commit checks passed")

    return True


def run_tests() -> bool:
    """Run basic tests to verify setup."""
    print("Running basic tests...")

    returncode, _, _ = run_command(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"], check=False
    )
    if returncode != 0:
        print(
            "⚠️  Some tests failed. This might be due to missing optional dependencies."
        )
        print("   Install optional dependencies with: pip install -e .[all]")
    else:
        print("✅ All tests passed")

    return True


def check_system_dependencies() -> List[str]:
    """Check for system dependencies."""
    print("Checking system dependencies...")

    missing = []

    # Check Tesseract (for PDF OCR)
    returncode, _, _ = run_command(
        ["tesseract", "--version"], check=False, capture_output=True
    )
    if returncode != 0:
        missing.append("tesseract (for PDF OCR processing)")
    else:
        print("✅ Tesseract OCR is available")

    # Check for llama.cpp (for GGUF conversion)
    returncode, _, _ = run_command(
        ["python", "-c", "import llama_cpp"], check=False, capture_output=True
    )
    if returncode != 0:
        # Try to find llama.cpp binary
        returncode, _, _ = run_command(
            ["which", "llama-cpp-python" if os.name != "nt" else "where"],
            check=False,
            capture_output=True,
        )
        if returncode != 0:
            missing.append("llama.cpp or llama-cpp-python (for GGUF model conversion)")
        else:
            print("✅ llama.cpp is available")
    else:
        print("✅ llama-cpp-python is available")

    if missing:
        print("⚠️  Missing optional system dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\nThese are optional but required for some features.")
        print("See CONTRIBUTING.md for installation instructions.")

    return missing


def print_next_steps():
    """Print next steps for the developer."""
    print("\n" + "=" * 60)
    print("🎉 Development environment setup complete!")
    print("=" * 60)

    print("\n📋 Next steps:")
    print("1. Make your changes to the code")
    print("2. Run tests: python -m pytest tests/ -v")
    print("3. Check code style: pre-commit run --all-files")
    print("4. Commit your changes (pre-commit hooks will run automatically)")
    print("5. Push and create a pull request")

    print("\n🔧 Useful commands:")
    print("- Run all tests: python -m pytest tests/ -v")
    print("- Run specific tests: python -m pytest tests/test_specific.py -v")
    print("- Format code: black llmbuilder/ tests/")
    print("- Sort imports: isort llmbuilder/ tests/")
    print("- Type checking: mypy llmbuilder/")
    print("- Build docs: cd docs && mkdocs serve")

    print("\n📚 Resources:")
    print("- Contributing guide: CONTRIBUTING.md")
    print("- Documentation: docs/")
    print("- Examples: examples/")
    print("- Issue templates: .github/ISSUE_TEMPLATE/")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Set up LLMBuilder development environment"
    )
    parser.add_argument(
        "--full", action="store_true", help="Install all optional dependencies"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check environment, don't install anything",
    )
    args = parser.parse_args()

    print("🚀 Setting up LLMBuilder development environment")
    print("=" * 60)

    # Basic checks
    if not check_python_version():
        sys.exit(1)

    if not check_git():
        sys.exit(1)

    if args.check_only:
        print("\n🔍 Environment check complete")
        check_system_dependencies()
        return

    # Setup steps
    if not setup_virtual_environment():
        sys.exit(1)

    if not install_package_dev():
        sys.exit(1)

    if not install_dev_dependencies():
        sys.exit(1)

    if not install_optional_dependencies(args.full):
        sys.exit(1)

    if not setup_pre_commit():
        sys.exit(1)

    # Verification steps
    run_initial_checks()
    run_tests()
    check_system_dependencies()

    print_next_steps()


if __name__ == "__main__":
    main()
