#!/usr/bin/env python3
"""Launch the VCD Fusion Dashboard.

Usage:
    pixi run python scripts/run_dashboard.py
    JAAD_ROOT=/custom/path pixi run python scripts/run_dashboard.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VCD.dashboard.app import main

if __name__ == "__main__":
    main(jaad_root=os.environ.get("JAAD_ROOT", "/mnt/sda/edward/data_vcd/JAAD"))
