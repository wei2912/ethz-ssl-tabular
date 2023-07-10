import numpy as np
from sklearn.ensemble import RandomForestClassifier

from typing import Dict

from . import BaseModel


class RandomForestModel(BaseModel):
    SWEEP_CONFIG = {
        "method": "random",
        "metric": {"goal": "minimize", "name": "train.acc"},
        "parameters": {
            "n_estimators": {
                "min": 5,
                "max": 100,
                "distribution": "q_log_uniform_values",
            },
            "max_depth": {"min": 2, "max": 12, "distribution": "q_log_uniform_values"},
        },
    }

    def __init__(self):
        super().__init__()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **config: Dict
    ) -> Dict:
        model = RandomForestClassifier(
            n_estimators=config.n_estimators, max_depth=config.max_depth, n_jobs=-1
        )
        self.model = model.fit(X, y)
        return {"val.acc": self.top_1_acc(X_val, y_val)}

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_log_proba(X)
