"""ReMasker baseline trainer with a vectorized inference path."""
from .training.base import TabularMAETrainer, set_all_seeds


class ReMasker(TabularMAETrainer):
    model_kind = "remasker"

    def __init__(self, args):
        super().__init__(args)
        self.new_imp = False
        self.block_mlp = None
        self.old_loss = True
        self.check_epochs = 600


__all__ = ["ReMasker", "set_all_seeds"]
