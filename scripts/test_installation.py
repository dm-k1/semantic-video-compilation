#!/usr/bin/env python3
"""
Internal environment verification script for the Semantic Video Compilation pipeline.
Provided for reference within the admissions review materials.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Verify Python version is 3.11 or higher."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("ERROR: Python 3.11+ required")
        return False
    print("OK: Python version")
    return True


def check_ffmpeg():
    """Verify FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split("\n")[0]
            print(f"OK: FFmpeg installed: {version_line}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    print("ERROR: FFmpeg not found. Install it:")
    print("   macOS:   brew install ffmpeg")
    print("   Ubuntu:  sudo apt-get install ffmpeg")
    print("   Windows: https://ffmpeg.org/download.html")
    return False


def check_python_packages():
    """Verify all required Python packages are installed."""
    packages = {
        "numpy": "np",
        "pandas": "pd",
        "PIL": "Pillow",
        "torch": "PyTorch",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
    }

    all_ok = True
    for module_name, display_name in packages.items():
        try:
            if module_name == "PIL":
                import PIL

                version = PIL.__version__
            else:
                module = __import__(module_name)
                version = getattr(module, "__version__", "unknown")

            print(f"OK: {display_name} {version}")
        except ImportError:
            print(f"ERROR: {display_name} not installed")
            all_ok = False
        except Exception as e:
            print(f"WARN: {display_name} (error loading: {type(e).__name__})")
            all_ok = False

    # Check for CLIP separately with better error handling (non-fatal)
    try:
        import clip
        print("OK: CLIP installed")
    except ImportError:
        print("WARN: CLIP not found (will be auto-installed from requirements)")
    except Exception as e:
        print(f"WARN: CLIP (error loading: {type(e).__name__})")

    return all_ok


def check_clip_model():
    """Verify CLIP is importable (skips model download)."""
    try:
        import clip
        print("OK: CLIP module available")
        return True
    except ImportError:
        print("WARN: CLIP will be installed from requirements")
        return True  # Non-fatal


def check_directory_structure():
    """Verify required directories exist."""
    required_dirs = [
        "inputs",
        "inputs/videos",
        "src",
        "src/steps",
        "notebooks",
        "scripts",
    ]

    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"OK: Directory exists: {dir_path}")
        else:
            print(f"ERROR: Missing directory: {dir_path}")
            all_ok = False

    return all_ok


def check_required_files():
    """Verify required files exist."""
    required_files = [
        "notebooks/Demo.ipynb",
        "src/core.py",
        "src/data.py",
        "main.py",
        "requirements.txt",
        "inputs/custom_scenes.csv",
    ]

    all_ok = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"OK: File exists: {file_path}")
        else:
            print(f"ERROR: Missing file: {file_path}")
            all_ok = False

    return all_ok


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("INSTALLATION VERIFICATION")
    print("=" * 70)
    print()

    checks = [
        ("Python Version", check_python_version),
        ("FFmpeg", check_ffmpeg),
        ("Python Packages", check_python_packages),
        ("CLIP Model", check_clip_model),
        ("Directory Structure", check_directory_structure),
        ("Required Files", check_required_files),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n--- {name} ---")
        try:
            success = check_func()
            results.append((name, success))
        except Exception as e:
            print(f"ERROR: Check failed with error: {e}")
            results.append((name, False))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = all(success for _, success in results)

    for name, success in results:
        status = "OK" if success else "FAIL"
        print(f"{status:8} {name}")

    print()
    if all_passed:
        print("All checks passed.")
        return 0
    else:
        print("WARN: Some checks failed. Resolve the issues listed above.")
        print("\nReferences:")
        print("  - REPRODUCTION.md")
        print("  - README.md")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
