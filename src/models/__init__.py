import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd

import abc
from typing import Any, Dict, Tuple


class Model(abc.ABC):
    def preprocess_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int8]]:
        """
        :param X: input dataframe
        :param y: labels of input data
        :return: processed input data and labels
        """
        return (X.to_numpy(dtype=np.float32), y.cat.codes.to_numpy(dtype=np.int8))

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: test data
        :return: class probabilities
        """
        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def top_1_acc(self, X: np.ndarray, y: np.ndarray) -> float:
        assert X.shape[0] == y.shape[0]
        return (y == self.predict(X)).sum() / X.shape[0]


class SLModel(Model):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def train(
        self,
        trial: optuna.trial.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        :param run: WandB run object
        :param trial: Optuna trial object
        :param X_train: training data
        :param y_train: labels of training data
        :param X_val: validation data
        :param y_val: labels of validation data
        :return: log metrics
        """
        raise NotImplementedError()


class SemiSLModel(Model):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def train_ssl(
        self,
        trial: optuna.trial.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_train_ul: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        :param run: WandB run object
        :param trial: Optuna trial object
        :param X_train: training data
        :param y_train: labels of training data
        :param X_train_ul: training data (unlabelled)
        :param X_val: validation data
        :param y_val: labels of validation data
        :return: log metrics
        """
        raise NotImplementedError()
