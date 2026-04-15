#!/usr/bin/env python3
"""Test LIBERO import and basic functionality."""

import sys
import os

# LIBERO should be installed via pip install -e .
# No need to manually add to path

try:
    from libero.libero import benchmark, get_libero_path
    print("✓ LIBERO imported successfully")

    # Test benchmark loading
    benchmark_dict = benchmark.get_benchmark_dict()
    print(f"✓ Available benchmarks: {list(benchmark_dict.keys())}")

    # Test libero_spatial
    task_suite = benchmark_dict["libero_spatial"]()
    print(f"✓ libero_spatial loaded: {task_suite.n_tasks} tasks")

    # Get first task
    task = task_suite.get_task(0)
    print(f"✓ Task 0: {task.name}")
    print(f"  Language: {task.language}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()