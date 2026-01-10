#!/usr/bin/env python3
"""
Check PTO-ISA CPU simulation environment.
"""

import subprocess
import sys
import os
from pathlib import Path

PTO_ISA_ROOT = Path(__file__).parent.parent.parent.parent / "code-repos" / "pto-isa"

def check_compiler():
    """Check C++ compiler version."""
    print("=== Compiler Check ===")

    # Try clang++ first (preferred for C++23)
    for cxx in ["clang++", "g++-14", "g++"]:
        try:
            result = subprocess.run([cxx, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                version_line = result.stdout.split("\n")[0]
                print(f"  {cxx}: {version_line}")
                return cxx
        except FileNotFoundError:
            continue

    print("  ERROR: No suitable C++ compiler found")
    return None

def check_cmake():
    """Check CMake version."""
    print("\n=== CMake Check ===")
    try:
        result = subprocess.run(["cmake", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split("\n")[0]
            print(f"  {version_line}")
            return True
    except FileNotFoundError:
        pass

    print("  ERROR: CMake not found")
    return False

def check_pto_isa():
    """Check PTO-ISA repository."""
    print("\n=== PTO-ISA Repository ===")

    if not PTO_ISA_ROOT.exists():
        print(f"  ERROR: PTO-ISA not found at {PTO_ISA_ROOT}")
        return False

    print(f"  Location: {PTO_ISA_ROOT}")

    # Check key directories
    dirs_to_check = [
        "include/pto/common",
        "include/pto/cpu",
        "tests/cpu/st/testcase",
        "docs/isa",
    ]

    for d in dirs_to_check:
        path = PTO_ISA_ROOT / d
        if path.exists():
            count = len(list(path.iterdir()))
            print(f"  {d}/: {count} items")
        else:
            print(f"  {d}/: MISSING")

    return True

def run_quick_test(cxx):
    """Run a quick CPU test to verify everything works."""
    print("\n=== Quick Test (tadd) ===")

    os.chdir(PTO_ISA_ROOT)

    cmd = [
        sys.executable, "tests/run_cpu.py",
        "--cxx", cxx,
        "--cc", cxx.replace("++", ""),
        "--testcase", "tadd"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if "PASS" in result.stdout:
        print("  tadd test: PASSED")
        return True
    else:
        print("  tadd test: FAILED")
        print(result.stderr[-500:] if result.stderr else "No error output")
        return False

def main():
    print("PTO-ISA Environment Check")
    print("=" * 50)

    cxx = check_compiler()
    cmake_ok = check_cmake()
    pto_ok = check_pto_isa()

    if cxx and cmake_ok and pto_ok:
        test_ok = run_quick_test(cxx)

        print("\n" + "=" * 50)
        if test_ok:
            print("Environment ready for PTO-ISA development!")
            print(f"\nRecommended compiler: {cxx}")
        else:
            print("Environment check failed. Please fix issues above.")
            return 1
    else:
        print("\n" + "=" * 50)
        print("Missing dependencies. Please install required tools.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
