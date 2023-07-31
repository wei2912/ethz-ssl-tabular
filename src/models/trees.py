import numpy as np
from optuna.trial import Trial
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from typing import Any, Dict

from . import SLModel
from utils.typing import Dataset


class RandomForestModel(SLModel):
    def __init__(self):
        super().__init__()

    def train(
        self, trial: Trial, train: Dataset, val: Dataset, is_sweep: bool, **_
    ) -> Dict[str, Any]:
        X_train, y_train = train
        X_val, y_val = val

        # hyperparams space adapted from https://arxiv.org/pdf/2207.08815.pdf pg. 20
        if not is_sweep:
            max_depth = trial.suggest_categorical("max_depth", [None])
            n_estimators = trial.suggest_categorical("n_estimators", [300])
            min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [1])
        else:
            max_depth = trial.suggest_categorical("max_depth", [None, 2, 3, 4, 5])
            n_estimators = trial.suggest_int("n_estimators", 10, 3000, log=True)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

        model = RandomForestClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
        )
        self._model = model.fit(X_train, y_train)

        train_acc = self.top_1_acc((X_train, y_train))
        val_acc = self.top_1_acc((X_val, y_val))
        return {"train": {"acc": train_acc}, "val": {"acc": val_acc}}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)


class HGBTModel(SLModel):
    def __init__(self):
        super().__init__()

    def train(
        self,
        trial: Trial,
        train: Dataset,
        val: Dataset,
        is_sweep: bool,
    ) -> Dict[str, Any]:
        X_train, y_train = train
        X_val, y_val = val

        # hyperparams space adapted from https://arxiv.org/pdf/2207.08815.pdf pg. 20
        if not is_sweep:
            max_depth = trial.suggest_categorical("max_depth", [None])
            learning_rate = trial.suggest_categorical("learning_rate", [0.03])
            max_iter = trial.suggest_categorical("max_iter", [300])
        else:
            max_depth = trial.suggest_categorical("max_depth", [None, 2, 3, 4, 5])
            learning_rate = trial.suggest_float("learning_rate", 0.01, 1.0, log=True)
            max_iter = trial.suggest_int("max_iter", 50, 500, log=True)
        min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [5])

        model = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
        )
        self._model = model.fit(X_train, y_train)

        train_acc = self.top_1_acc((X_train, y_train))
        val_acc = self.top_1_acc((X_val, y_val))
        return {"train": {"acc": train_acc}, "val": {"acc": val_acc}}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return self._model.predict_proba(X)
