import numpy as np
from optuna.trial import Trial
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

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
        is_sweep: bool = True,
    ) -> Tuple[float, Dict[str, Any]]:
        # hyperparams space adapted from https://arxiv.org/pdf/2207.08815.pdf pg. 20
        if not is_sweep:
            max_depth = trial.suggest_categorical("max_depth", [5])
            n_estimators = trial.suggest_categorical("n_estimators", [100])
        else:
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


class HGBTModel(SLModel):
    def __init__(self):
        super().__init__()

    def train(
        self,
        trial: Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        is_sweep: bool = True,
    ) -> Tuple[float, Dict[str, Any]]:
        # hyperparams space adapted from https://arxiv.org/pdf/2207.08815.pdf pg. 20
        if not is_sweep:
            learning_rate = trial.suggest_categorical("learning_rate", [0.03])
            max_depth = trial.suggest_categorical("max_depth", [None])
            max_iter = trial.suggest_categorical("max_iter", [300])
        else:
            learning_rate = trial.suggest_float("learning_rate", 0.01, 10, log=True)
            max_depth = trial.suggest_categorical("max_depth", [None, 2, 3, 4, 5])
            max_iter = trial.suggest_int("max_iter", 10, 1000, log=True)

        model = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_iter=max_iter,
        )
        self.model = model.fit(X_train, y_train)

        train_acc = self.top_1_acc(X_train, y_train)
        val_acc = self.top_1_acc(X_val, y_val)
        return (val_acc, {"train": {"acc": train_acc}, "val": {"acc": val_acc}})

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.log(self.model.predict_proba(X))
