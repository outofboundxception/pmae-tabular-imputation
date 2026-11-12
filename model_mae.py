"""Backward-compatible import; new code should use pmae_tabular.model_mae."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pmae_tabular.model_mae import *  # noqa: F401,F403
