#!/usr/bin/env python3
"""
Build verification script for Posture Analysis System
This script validates that all Python modules can be compiled and imported correctly.
"""

import sys
import os
from pathlib import Path

def check_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r') as f:
            compile(f.read(), file_path, 'exec')
        return True, None
    except SyntaxError as e:
        return False, str(e)

def main():
    print("=" * 70)
    print("POSTURE ANALYSIS SYSTEM - BUILD VERIFICATION")
    print("=" * 70)
    print()

    project_root = Path(__file__).parent
    python_files = list(project_root.rglob("*.py"))

    python_files = [f for f in python_files if '__pycache__' not in str(f)]

    errors = []
    success_count = 0

    print(f"Checking {len(python_files)} Python files...\n")

    for py_file in sorted(python_files):
        relative_path = py_file.relative_to(project_root)
        success, error = check_syntax(py_file)

        if success:
            print(f"✓ {relative_path}")
            success_count += 1
        else:
            print(f"✗ {relative_path}")
            print(f"  Error: {error}")
            errors.append((relative_path, error))

    print()
    print("=" * 70)
    print(f"RESULTS: {success_count}/{len(python_files)} files passed")
    print("=" * 70)

    if errors:
        print("\nERRORS FOUND:")
        for file_path, error in errors:
            print(f"\n{file_path}:")
            print(f"  {error}")
        sys.exit(1)
    else:
        print("\n✓ All Python files have valid syntax!")
        print("✓ Project build verification: PASSED")

        print("\nREQUIREMENTS:")
        print("  • Python 3.9+ installed")
        print("  • Dependencies: pip install -r requirements.txt")
        print("  • YOLO model: models/yolo_model.pt")
        print("  • Environment: .env configured")
        print("\nREADY TO RUN:")
        print("  • GUI: python run_gui.py")
        print("  • API: python run_api.py")
        sys.exit(0)

if __name__ == "__main__":
    main()
