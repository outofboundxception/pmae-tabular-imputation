"""Backward-compatible root import for packaged utilities."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pmae_tabular.utils import *  # noqa: F401,F403
