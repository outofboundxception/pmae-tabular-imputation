"""Console-script adapter for running the repository CLI."""
from pathlib import Path
import os
import sys


def main():
    # The CLI operates on the checked-out project (raw datasets, config, and
    # output folders are repository assets). Require invocation from its root.
    root = Path(os.environ.get("PMAE_PROJECT_ROOT", Path.cwd())).expanduser().resolve()
    runner = root / "run_experiment.py"
    if not runner.is_file():
        raise RuntimeError(
            "Run `pmae-experiment` from the PMAE repository root, or set "
            "PMAE_PROJECT_ROOT to the repository directory."
        )
    sys.path.insert(0, str(root))
    from run_experiment import main as run_main
    return run_main()
