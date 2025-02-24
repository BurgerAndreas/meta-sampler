from lightning.pytorch.callbacks import Callback
import torch


class TensorCoresCallback(Callback):
    """Callback to set Tensor Cores precision at the start of training."""

    def __init__(self, precision: str = "medium"):
        """
        Args:
            precision: Either "medium" or "high"
        """
        super().__init__()
        self.precision = precision

    def setup(self, trainer, pl_module, stage=None):
        torch.set_float32_matmul_precision(self.precision)
