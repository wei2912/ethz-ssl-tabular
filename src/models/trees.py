import numpy as np
from sklearn.ensemble import RandomForestClassifier

from typing import Dict

from . import SLModel


class RandomForestModel(SLModel):
    def __init__(self):
        super().__init__()
        self.SWEEP_CONFIG = {
            "method": "random",
            "metric": {"goal": "maximize", "name": "val.acc"},
            "parameters": {
                "n_estimators": {
                    "min": 5,
                    "max": 100,
                    "distribution": "q_log_uniform_values",
                },
                "max_depth": {
                    "min": 2,
                    "max": 12,
                    "distribution": "q_log_uniform_values",
                },
            },
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = None,
        **_,
    ) -> Dict:
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1
        )
        self.model = model.fit(X_train, y_train)
        return {"acc": self.top_1_acc(X_train, y_train)}

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return self.model.predict_log_proba(X)

    def val(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        return {"acc": self.top_1_acc(X_val, y_val)}
