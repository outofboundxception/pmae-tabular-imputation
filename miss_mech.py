"""Backward-compatible access to the original paper/experiment utilities.

The new generic missingness generation implementation lives in
pmae_tabular.data.amputation.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pmae_tabular.legacy.miss_mech import *  # noqa: F401,F403
