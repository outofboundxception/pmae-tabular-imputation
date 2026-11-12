"""Public PMAE trainer API; retains the original ProportionalMasker class name."""
from .training.base import TabularMAETrainer, set_all_seeds


class ProportionalMasker(TabularMAETrainer):
    model_kind = "pmae"

    def __init__(self, args):
        super().__init__(args)
        # Original experiment configuration: MLP-Mixer blocks and the newer
        # proportional-loss/imputation path.
        self.new_imp = True
        self.block_mlp = 0
        self.old_loss = False
        self.check_epochs = 600


__all__ = ["ProportionalMasker", "set_all_seeds"]
