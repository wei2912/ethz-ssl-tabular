import numpy as np
import numpy.typing as npt
from optuna.trial import Trial
import pandas as pd

import math
import random
from typing import Any, Callable, Dict, Optional, Union

from . import SemiSLModel, SLModel
from utils.typing import Dataset


class SelfTrainingModel_ThresholdSingleIterate(SemiSLModel):
    def __init__(self, base_model_fn: Callable[[], SLModel]):
        """
        :param base_model_fn: constructor for base supervised learning (SL) model, used
        for pseudolabelling
        """
        super().__init__()
        self.base_model_fn = base_model_fn

    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Dataset:
        return self.base_model_fn().preprocess_data(X, y)

    def train_ssl(
        self,
        train_l: Dataset,
        train_ul: Union[Dataset, npt.NDArray[np.float32]],
        val: Dataset,
        trial: Optional[Trial] = None,
    ) -> Dict[str, Any]:
        X_train_l, y_train_l = train_l
        X_train_ul, y_train_ul = (
            train_ul if type(train_ul) is tuple else (train_ul, None)
        )
        X_val, y_val = val

        PROB_THRESHOLD: float = 0.9
        metrics: Dict[str, Any] = {}

        self._model = self.base_model_fn()
        metrics["initial"] = self._model.train(
            (X_train_l, y_train_l),
            (X_val, y_val),
            trial=trial,
        )

        y_pl_probs = self._model.predict_proba(X_train_ul)
        y_pl = y_pl_probs.argmax(axis=1)
        y_pl_max_prob = y_pl_probs.max(axis=1)

        is_selects = y_pl_max_prob >= PROB_THRESHOLD
        X_train_pl, y_train_pl = X_train_ul[is_selects].copy(), y_pl[is_selects].copy()

        pl_acc = (
            (y_train_pl == y_train_ul[is_selects]).sum() / y_train_pl.shape[0]
            if y_train_ul is not None
            else None
        )
        X = np.concatenate((X_train_l, X_train_pl))
        y = np.concatenate((y_train_l, y_train_pl))

        new_metrics = self._model.train((X, y), (X_val, y_val), trial=trial)

        metrics["pl_iter0"] = {
            "size_ul": len(X_train_ul),
            "pl_acc": pl_acc,
            "n_pl": len(X_train_pl),
            "prob_threshold": PROB_THRESHOLD,
            **new_metrics,
        }

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)


class SelfTrainingModel_Curriculum(SemiSLModel):
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

    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Dataset:
        return self.base_model_fn().preprocess_data(X, y)

    def train_ssl(
        self,
        train_l: Dataset,
        train_ul: Union[Dataset, npt.NDArray[np.float32]],
        val: Dataset,
        trial: Optional[Trial] = None,
    ) -> Dict[str, Any]:
        X_train_l, y_train_l = train_l
        X_train_ul, y_train_ul = (
            train_ul if type(train_ul) is tuple else (train_ul, None)
        )
        X_val, y_val = val

        STEP_THRESHOLD: float = 0.2

        metrics = {}

        self._model = self.base_model_fn()
        metrics["initial"] = self._model.train(
            (X_train_l, y_train_l),
            (X_val, y_val),
            trial=trial,
        )

        threshold = STEP_THRESHOLD
        i = 0
        while threshold <= 1.0:
            y_pl_probs = self._model.predict_proba(X_train_ul)
            y_pl = y_pl_probs.argmax(axis=1)
            y_pl_max_prob = y_pl_probs.max(axis=1)

            pivot_id = min(math.ceil(threshold * len(X_train_ul)), len(X_train_ul))
            if pivot_id < len(X_train_ul):
                ids = np.argpartition(y_pl_max_prob, -pivot_id)[-pivot_id:]
                X_train_pl, y_train_pl = np.take(X_train_ul, ids, axis=0), np.take(
                    y_pl, ids, axis=0
                )
            else:
                ids = list(range(len(X_train_ul)))
                X_train_pl, y_train_pl = X_train_ul.copy(), y_pl.copy()

            pl_acc = (
                (y_train_pl == np.take(y_train_ul, ids, axis=0)).sum()
                / y_train_pl.shape[0]
                if y_train_ul is not None
                else None
            )
            X = np.concatenate((X_train_l, X_train_pl))
            y = np.concatenate((y_train_l, y_train_pl))

            self._model = self.base_model_fn()
            new_metrics = self._model.train(
                (X, y),
                (X_val, y_val),
                trial=trial,
            )

            metrics[f"pl_iter{i}"] = {
                "size_ul": len(X_train_ul),
                "size_l_pl": len(X_train_l) + pivot_id,
                "pl_acc": pl_acc,
                "l_pl_acc": 1 - (1 - pl_acc) * pivot_id / (len(X_train_l) + pivot_id),
                "n_pl": pivot_id,
                "threshold": threshold,
                **new_metrics,
            }

            i += 1
            threshold += STEP_THRESHOLD

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)


class SelfTrainingModel_Fake(SemiSLModel):
    """
    Simulates random label noise on the unlabelled dataset, to observe if there is any
    significant difference in test accuracy due to pseudolabelled datasets of similar
    label error rates.
    """

    def __init__(self, base_model_fn: Callable[[], SLModel]):
        """
        :param base_model_fn: constructor for base supervised learning (SL) model
        """
        super().__init__()
        self.base_model_fn = base_model_fn

    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Dataset:
        return self.base_model_fn().preprocess_data(X, y)

    def train_ssl(
        self,
        train_l: Dataset,
        train_ul: Union[Dataset, npt.NDArray[np.float32]],
        val: Dataset,
        trial: Optional[Trial] = None,
    ) -> Dict[str, Any]:
        X_train_l, y_train_l = train_l
        if type(train_ul) is not tuple:
            raise ValueError("train_ul must be a tuple")
        X_train_ul, y_train_ul = train_ul
        X_val, y_val = val

        FAKE_L_PL_ACC = 0.64
        FAKE_PL_N_WRONG = min(
            math.ceil((1 - FAKE_L_PL_ACC) * (len(X_train_l) + len(X_train_ul))),
            len(X_train_ul),
        )

        metrics = {}

        X_train_pl, y_train_pl = X_train_ul.copy(), y_train_ul.copy()
        rng = np.random.default_rng()
        ids = rng.choice(
            range(len(X_train_pl)),
            size=(FAKE_PL_N_WRONG,),
            replace=False,
            shuffle=False,
        )
        n_class = len(np.unique(y_val))  # FIXME: contains all classes with high prob.

        def corrupt_label(label: int) -> int:
            choices = list(range(n_class))
            choices.remove(label)
            return random.choice(choices)

        y_train_pl[ids] = np.array(list(map(corrupt_label, y_train_pl[ids])))

        pl_acc = (y_train_pl == y_train_ul).sum() / y_train_pl.shape[0]
        X = np.concatenate((X_train_l, X_train_pl))
        y = np.concatenate((y_train_l, y_train_pl))

        self._model = self.base_model_fn()
        new_metrics = self._model.train(
            (X, y),
            (X_val, y_val),
            trial=trial,
        )

        metrics["pl_iter4"] = {
            "size_ul": len(X_train_ul),
            "pl_acc": pl_acc,
            "l_pl_acc": 1
            - (1 - pl_acc) * len(X_train_pl) / (len(X_train_l) + len(X_train_pl)),
            "n_pl": len(X_train_ul),
            **new_metrics,
        }

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
