import numpy as np
from optuna.trial import Trial

from typing import Any, Callable, Dict, Tuple

from . import SemiSLModel, SLModel


class SelfTrainingModel(SemiSLModel):
    def __init__(self, base_model_fn: Callable[[], SLModel]):
        """
        :param base_model_fn: constructor for base supervised learning (SL) model, used
        for pseudolabelling
        """
        super().__init__()
        self.pl_model = base_model_fn()
        self.new_model = base_model_fn()

    def train_ssl(
        self,
        trial: Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_train_ul: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        is_sweep: bool = True,
    ) -> Tuple[float, Dict[str, Any]]:
        if X_train_ul is None:
            return self.new_model.train(
                trial, X_train, y_train, X_val, y_val, is_sweep=is_sweep
            )

        _, pl_metrics_dict = self.pl_model.train(
            trial, X_train, y_train, X_val, y_val, is_sweep=is_sweep
        )
        y_train_pl = self.pl_model.predict(X_train_ul)
        X_train_new, y_train_new = np.concatenate(
            (X_train, X_train_ul)
        ), np.concatenate((y_train, y_train_pl))
        score, new_metrics_dict = self.new_model.train(
            trial, X_train_new, y_train_new, X_val, y_val, is_sweep=is_sweep
        )
        return (score, {"pl": pl_metrics_dict, **new_metrics_dict})

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        return self.new_model.predict_log_proba(X)
