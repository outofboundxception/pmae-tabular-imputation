"""Backward-compatible root import for the packaged configuration module."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pmae_tabular.configs import *  # noqa: F401,F403
