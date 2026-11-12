"""Backward-compatible root import for evaluation functions."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pmae_tabular.evaluate import *  # noqa: F401,F403
