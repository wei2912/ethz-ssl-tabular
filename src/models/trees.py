import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from typing import Dict

from . import SLModel


class RandomForestModel(SLModel):
    def __init__(self):
        super().__init__()
        # hyperparams space adapted from https://arxiv.org/pdf/2207.08815.pdf pg. 20
        self.SWEEP_CONFIG = {
            "method": "random",
            "metric": {"goal": "maximize", "name": "val.acc"},
            "parameters": {
                "max_depth": {
                    "values": [None, 2, 3, 4],
                    "probabilities": [0.7, 0.1, 0.1, 0.1],
                },
                "n_estimators": {
                    "min": 9,
                    "max": 3000,
                    "distribution": "q_log_uniform_values",
                },
            },
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        max_depth: int | None = None,
        n_estimators: int = 100,
        **_,
    ) -> Dict:
        model = RandomForestClassifier(
            max_depth=max_depth, n_estimators=n_estimators, n_jobs=-1
        )
        self.model = model.fit(X_train, y_train)
        return {"acc": self.top_1_acc(X_train, y_train)}

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return self.model.predict_log_proba(X)

    def val(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        return {"acc": self.top_1_acc(X_val, y_val)}


class GBTModel(SLModel):
    def __init__(self):
        super().__init__()
        # hyperparams space adapted from https://arxiv.org/pdf/2207.08815.pdf pg. 20
        self.SWEEP_CONFIG = {
            "method": "random",
            "metric": {"goal": "maximize", "name": "val.acc"},
            "parameters": {
                "learning_rate": {
                    "min": 0.01,
                    "max": 10,
                    "distribution": "log_uniform_values",
                },
                "subsample": {
                    "min": 0.5,
                    "max": 1.0,
                    "distribution": "uniform",
                },
                "n_estimators": {
                    "min": 10,
                    "max": 1000,
                    "distribution": "q_log_uniform_values",
                },
                "max_depth": {
                    "values": [None, 2, 3, 4, 5],
                    "probabilities": [0.1, 0.1, 0.6, 0.1, 0.1],
                },
            },
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        learning_rate: float = 0.1,
        subsample: float = 1,
        n_estimators: int = 100,
        max_depth: int | None = 3,
        **_,
    ) -> Dict:
        model = GradientBoostingClassifier(
            learning_rate=learning_rate,
            subsample=subsample,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        self.model = model.fit(X_train, y_train)
        return {"acc": self.top_1_acc(X_train, y_train)}

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return self.model.predict_log_proba(X)

    def val(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        return {"acc": self.top_1_acc(X_val, y_val)}
