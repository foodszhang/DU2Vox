#!/usr/bin/env python3
"""
Visualization script for GT vs Prediction comparison.

Usage:
    python scripts/visualize.py --config configs/stage1/gcain_full.yaml --checkpoint checkpoints/best.pth --n_samples 4
"""

import sys
from pathlib import Path

# Add parent to path for du2vox imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.visualization.visualize_3d import main

if __name__ == "__main__":
    main()
