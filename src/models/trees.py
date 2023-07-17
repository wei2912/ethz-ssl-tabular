import numpy as np
from optuna.trial import Trial
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from typing import Any, Dict, Tuple

from . import SLModel


class RandomForestModel(SLModel):
    def __init__(self):
        super().__init__()

    def train(
        self,
        trial: Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        # hyperparams space adapted from https://arxiv.org/pdf/2207.08815.pdf pg. 20
        max_depth = trial.suggest_categorical("max_depth", [None, 2, 3, 4, 5])
        n_estimators = trial.suggest_int("n_estimators", 9, 3000, log=True)

        model = RandomForestClassifier(
            max_depth=max_depth, n_estimators=n_estimators, n_jobs=-1
        )
        self.model = model.fit(X_train, y_train)

        train_acc = self.top_1_acc(X_train, y_train)
        val_acc = self.top_1_acc(X_val, y_val)
        return (val_acc, {"train": {"acc": train_acc}, "val": {"acc": val_acc}})

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return self.model.predict_log_proba(X)


class GBTModel(SLModel):
    def __init__(self):
        super().__init__()

    def train(
        self,
        trial: Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        # hyperparams space adapted from https://arxiv.org/pdf/2207.08815.pdf pg. 20
        learning_rate = trial.suggest_float("learning_rate", 0.01, 10, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        n_estimators = trial.suggest_int("n_estimators", 9, 3000, log=True)
        max_depth = trial.suggest_categorical("max_depth", [None, 2, 3, 4, 5])

        model = GradientBoostingClassifier(
            learning_rate=learning_rate,
            subsample=subsample,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        self.model = model.fit(X_train, y_train)

        train_acc = self.top_1_acc(X_train, y_train)
        val_acc = self.top_1_acc(X_val, y_val)
        return (val_acc, {"train": {"acc": train_acc}, "val": {"acc": val_acc}})

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return self.model.predict_log_proba(X)
