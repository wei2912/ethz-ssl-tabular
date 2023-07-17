import numpy as np
from optuna.trial import Trial
from sklearn.metrics import top_k_accuracy_score

import abc
from typing import Any, Dict, Tuple


class Model(abc.ABC):
    @abc.abstractmethod
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: test data
        :return: class log-probabilities
        """
        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_log_proba(X).argmax(axis=1)

    def top_1_acc(self, X: np.ndarray, y: np.ndarray) -> float:
        return top_k_accuracy_score(y, self.predict(X), k=1)


class SLModel(Model):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def train(
        self,
        trial: Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        :param run: WandB run object
        :param trial: Optuna trial object
        :param X_train: training data
        :param y_train: labels of training data
        :param X_val: validation data
        :param y_val: labels of validation data
        :return: objective score
        """
        raise NotImplementedError()


class SemiSLModel(Model):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def train_ssl(
        self,
        trial: Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_train_ul: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        :param run: WandB run object
        :param trial: Optuna trial object
        :param X_train: training data
        :param y_train: labels of training data
        :param X_train_ul: training data (unlabelled)
        :param X_val: validation data
        :param y_val: labels of validation data
        :return: objective score
        """
        raise NotImplementedError()
