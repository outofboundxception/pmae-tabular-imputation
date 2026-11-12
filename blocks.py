"""Backward-compatible root import for model blocks."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pmae_tabular.blocks import *  # noqa: F401,F403
