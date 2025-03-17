import time
from typing import Callable, Any

import lightning as pl


class ThroughputMonitor(pl.Callback):
    def __init__(self, batch_size_fun: Callable[[Any], int]):
        self.batch_size_fun = batch_size_fun
        self.batch_counts = {"train": 0, "predict": 0}
        self.start_times = {"train": None, "predict": None}

    def _on_batch_start(self, mode: str):
        """Start timing if it's the first batch of a mode (train or predict)."""
        if self.start_times[mode] is None:
            self.start_times[mode] = time.perf_counter()

    def _on_batch_end(
        self, mode: str, pl_module: pl.LightningModule, batch: Any
    ) -> None:
        """Log elapsed time and batch count for the given mode."""
        self.batch_counts[mode] += 1
        batch_size = self.batch_size_fun(batch)
        elapsed_time = time.perf_counter() - self.start_times[mode]

        pl_module.log(
            f"{mode}_time_elapsed",
            elapsed_time,
            on_step=True,
            batch_size=batch_size,
        )
        pl_module.log(
            f"{mode}_num_batches",
            self.batch_counts[mode],
            on_step=True,
            batch_size=batch_size,
        )

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._on_batch_start("train")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._on_batch_end("train", pl_module, batch)

    def on_predict_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_start("predict")

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: int,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_end("predict", pl_module, batch)
