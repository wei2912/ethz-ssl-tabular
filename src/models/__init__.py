import numpy as np
from sklearn.metrics import top_k_accuracy_score

import abc
from typing import Dict, Optional


class Model(abc.ABC):
    SWEEP_CONFIG: Dict  # WandB sweep config, assumed to be constant

    @abc.abstractmethod
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: test data
        :return: class log-probabilities
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def val(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        :param X_val: validation data
        :param y_val: labels of validation data
        :return: validation metrics
        """
        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: test data
        :return: class predictions
        """
        return self.predict_log_proba(X).argmax(axis=1)

    def top_1_acc(self, X: np.ndarray, y: np.ndarray) -> float:
        return top_k_accuracy_score(y, self.predict(X), k=1)


class SLModel(Model):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Dict) -> Dict:
        """
        :param X_train: training data
        :param y_train: labels of training data
        :param kwargs: additional arguments
        :return: training metrics
        """
        raise NotImplementedError()


class SemiSLModel(Model):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def train_ssl(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_train_ul: Optional[np.ndarray],
        **kwargs: Dict
    ) -> Dict:
        """
        :param X_train: training data
        :param y_train: labels of training data
        :param X_train_ul: training data (unlabelled)
        :return: metrics
        """
        raise NotImplementedError()
