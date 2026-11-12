from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Support both the installed package and the documented legacy root shims.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
