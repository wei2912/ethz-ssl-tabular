import numpy as np
from optuna.trial import Trial

import math
from typing import Any, Callable, Dict, Tuple

from . import SemiSLModel, SLModel


class SelfTrainingModel_SingleIterate(SemiSLModel):
    def __init__(self, base_model_fn: Callable[[], SLModel]):
        """
        :param base_model_fn: constructor for base supervised learning (SL) model, used
        for pseudolabelling
        """
        super().__init__()
        self._pl_model = base_model_fn()
        self._new_model = base_model_fn()

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
        if not is_sweep:
            pl_threshold = trial.suggest_categorical("pl_threshold", [0.9])
        else:
            pl_threshold = trial.suggest_float("pl_threshold", 0.5, 1.0, log=True)

        _, pl_metrics_dict = self._pl_model.train(
            trial, X_train, y_train, X_val, y_val, is_sweep=is_sweep
        )
        y_train_pl_probs = self._pl_model.predict_proba(X_train_ul)

        X_train_pl = X_train_ul[y_train_pl_probs[:, 0] >= pl_threshold]
        y_train_pl = y_train_pl_probs.argmax(axis=1)[
            y_train_pl_probs[:, 0] >= pl_threshold
        ]
        X_train_new, y_train_new = np.concatenate(
            (X_train, X_train_pl)
        ), np.concatenate((y_train, y_train_pl))
        score, new_metrics_dict = self._new_model.train(
            trial, X_train_new, y_train_new, X_val, y_val, is_sweep=is_sweep
        )
        return (
            score,
            {"pl": pl_metrics_dict, "n_pl": len(X_train_pl), **new_metrics_dict},
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._new_model.predict_proba(X)


class SelfTrainingModel_CurriculumLearning(SemiSLModel):
    """
    Adapted from https://arxiv.org/abs/2001.06001.
    """

    def __init__(self, base_model_fn: Callable[[], SLModel]):
        """
        :param base_model_fn: constructor for base supervised learning (SL) model, used
        for pseudolabelling
        """
        super().__init__()
        self.base_model_fn = base_model_fn

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
        STEP_THRESHOLD: float = 0.2

        metrics = {}
        threshold = STEP_THRESHOLD
        i = 0
        self._model = self.base_model_fn()
        while True:
            X, y = X_train, y_train
            _, pl_metrics = self._model.train(
                trial, X_train, y_train, X_val, y_val, is_sweep=is_sweep
            )

            y_pl_probs = self._model.predict_proba(X_train_ul)
            y_pl = y_pl_probs.argmax(axis=1)
            y_pl_max_prob = y_pl_probs.max(axis=1)

            pivot_id = math.ceil(threshold * len(X_train_ul))
            if pivot_id < len(X_train_ul):
                ids = np.argpartition(y_pl_max_prob, -pivot_id)[-pivot_id:]
                X, y = np.concatenate(
                    (X_train, np.take(X_train_ul, ids, axis=0))
                ), np.concatenate((y_train, np.take(y_pl, ids, axis=0)))
            else:
                X, y = np.concatenate((X_train, X_train_ul)), np.concatenate(
                    (y_train, y_pl)
                )

            self._model = self.base_model_fn()
            score, new_metrics = self._model.train(
                trial, X, y, X_val, y_val, is_sweep=is_sweep
            )

            metrics[f"pl_iter{i}"] = {
                "pl": pl_metrics,
                "n_pl": len(ids),
                "threshold": threshold,
                **new_metrics,
            }
            i += 1

            if threshold == 1.0:
                break
            threshold += STEP_THRESHOLD

        return (score, metrics)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
