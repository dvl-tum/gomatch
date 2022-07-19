from argparse import Namespace
from typing import Any, Dict, Optional, Union

from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard.summary import hparams


class TensorBoardLoggerMetrics(TensorBoardLogger):
    def log_hyperparams(
        self,
        params: Union[Dict[str, Any], Namespace],
        metrics: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> None:
        """
        Record hyperparameters. TensorBoard logs with and without saved hyperparameters
        are incompatible, the hyperparameters are then not displayed in the TensorBoard.
        Please delete or move the previously saved logs to display the new ones with hyperparameters.

        Args:
            params: a dictionary-like container with the hyperparameters
            metrics: Dictionary with metric names as keys and measured quantities as values
        """

        params = self._convert_params(params)

        # store params to output
        self.hparams.update(params)

        # format params into the suitable for tensorboard
        params = self._flatten_dict(self.hparams)
        params = self._sanitize_params(params)

        if metrics is None:
            if self._default_hp_metric:
                if isinstance(self._default_hp_metric, dict):
                    metrics = self._default_hp_metric
                else:
                    metrics = {"hp_metric": float("inf")}
        elif not isinstance(metrics, dict):
            metrics = {"hp_metric": metrics}

        if metrics:
            metrics = {f"metrics/{k}": v for k, v in metrics.items()}
            self.log_metrics(metrics, step)
            exp, ssi, sei = hparams(params, metrics)
            writer = self.experiment._get_file_writer()
            writer.add_summary(exp)
            writer.add_summary(ssi)
            writer.add_summary(sei)
