#!/usr/bin/env python3
"""
Comprehensive Test Runner

Runs all tests for the Orbital Inspector satellite system.
"""

from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent


def run_test(name: str, script: str) -> bool:
    """Run a test script and return success status."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print("=" * 60)

    script_path = ROOT / "scripts" / script
    if not script_path.exists():
        print(f"  ❌ Script not found: {script}")
        return False

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start

    # Print last 10 lines of output
    lines = result.stdout.strip().split("\n")
    for line in lines[-10:]:
        print(f"  {line}")

    if result.returncode == 0:
        print(f"\n  ✅ PASS ({elapsed:.1f}s)")
        return True
    else:
        print(f"\n  ❌ FAIL ({elapsed:.1f}s)")
        if result.stderr:
            print(f"  Error: {result.stderr[-200:]}")
        return False


def main():
    print("\n" + "=" * 60)
    print("  ORBITAL INSPECTOR - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print(f"\nRunning from: {ROOT}")

    tests = [
        ("Reaction Wheel Control (Phase 1)", "test_rw_control.py"),
        ("Orbital Dynamics (Phase 2)", "test_orbital.py"),
        ("Fleet Coordination (Phase 3)", "test_fleet.py"),
        ("Mission System (Phase 5)", "test_mission.py"),
    ]

    results = {}

    for name, script in tests:
        results[name] = run_test(name, script)

    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}  {name}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\n  Total: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
