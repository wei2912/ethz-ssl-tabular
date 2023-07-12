import numpy as np

from typing import Callable, Dict

from . import SemiSLModel, SLModel


class PseudolabelModel(SemiSLModel):
    def __init__(self, base_model_fn: Callable[[], SLModel]):
        """
        :param base_model_fn: constructor for base supervised learning (SL) model, used
        for pseudolabelling
        """
        super().__init__()
        self.model = base_model_fn()
        self.SWEEP_CONFIG = self.model.SWEEP_CONFIG

    def train_ssl(
        self, X_train: np.ndarray, y_train: np.ndarray, *_, **kwargs: Dict
    ) -> Dict:
        return self.model.train(X_train, y_train, **kwargs)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_log_proba(X)

    def val(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        return self.model.val(X_val, y_val)


class SelfTrainingModel(SemiSLModel):
    def __init__(self, base_model_fn: Callable[[], SLModel]):
        """
        :param base_model_fn: constructor for base supervised learning (SL) model, used
        for pseudolabelling
        """
        super().__init__()
        self.pl_model = base_model_fn()
        self.new_model = base_model_fn()
        self.SWEEP_CONFIG = self.new_model.SWEEP_CONFIG

    def train_ssl(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_train_ul: np.ndarray,
        **kwargs: Dict
    ) -> Dict:
        self.pl_model.train(X_train, y_train)
        y_train_pl: np.ndarray = self.pl_model.predict(X_train_ul)
        X_train_new, y_train_new = np.concatenate(
            (X_train, X_train_ul)
        ), np.concatenate((y_train, y_train_pl))
        return self.new_model.train(X_train_new, y_train_new, **kwargs)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        return self.new_model.predict_log_proba(X)

    def val(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        return self.new_model.val(X_val, y_val)
