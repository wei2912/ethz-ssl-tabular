import numpy as np
from sklearn.metrics import top_k_accuracy_score

from typing import Dict


class BaseModel:
    SWEEP_CONFIG = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> Dict:
        """
        :param X: training data
        :param y: labels of training data
        :param X_val: validation data
        :param y_val: labels of validation data
        :return: metrics
        """
        raise NotImplementedError()

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: test data
        :return: class log-probabilities
        """
        raise NotImplementedError()

    def top_1_acc(self, X: np.ndarray, y: np.ndarray) -> float:
        return top_k_accuracy_score(y, self.predict_log_proba(X), k=1)
