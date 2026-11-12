"""Backward-compatible import; new code should use pmae_tabular.fit_remasker."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pmae_tabular.fit_remasker import *  # noqa: F401,F403
