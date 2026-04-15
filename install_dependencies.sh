#!/bin/bash
# Installation script for VLM Spatial Reasoning Benchmark
# Run this in your vlmben conda environment

set -e  # Exit on error

echo "=== Installing VLM Benchmark Dependencies ==="
echo ""

# Check if in conda environment
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "Warning: No conda environment detected"
    echo "Please run: conda activate vlmben"
    echo ""
fi

# Install core dependencies
echo "Installing core dependencies..."
pip install numpy Pillow pyyaml scikit-learn

# Install LIBERO
echo ""
echo "Installing LIBERO..."
cd /Users/tdu/Documents/GitHub/3DBench/LIBERO
pip install -e .

# Return to project root
cd /Users/tdu/Documents/GitHub/3DBench

echo ""
echo "=== Installation complete! ==="
echo ""
echo "To test the installation, run:"
echo "  python test_libero_import.py"
echo ""
echo "To run the benchmark with mock VLM:"
echo "  python run_benchmark.py --mock --num-tasks 2 --samples-per-task 2"