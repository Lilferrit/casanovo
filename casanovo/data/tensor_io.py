import os
from pathlib import Path
from typing import Union
import numpy as np
import torch


class BatchTensorWriter:
    """
    Simple utility to write batch tensors to .npy files in output_dir/batches.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save batch files into.
    """

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir) / "batches"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def write(
        self, batch: Union[np.ndarray, torch.Tensor], prefix: str = "batch"
    ):
        """
        Save a batch tensor to a .npy file.

        Parameters
        ----------
        batch : np.ndarray or torch.Tensor
            The batch tensor to save.
        prefix : str
            Optional prefix for the filename.
        """
        if isinstance(batch, torch.Tensor):
            batch = batch.detach().cpu().numpy()
        filename = self.output_dir / f"{prefix}_{self.counter:07d}.npy"
        np.save(filename, batch)
        self.counter += 1
