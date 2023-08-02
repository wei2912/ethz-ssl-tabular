import numpy as np
from optuna.trial import Trial
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
import wandb

from typing import Any, Dict, Optional

from . import SLModel
from utils.typing import Dataset


class RandomForestModel(SLModel):
    def __init__(self):
        super().__init__()

    def train(
        self,
        train: Dataset,
        val: Dataset,
        trial: Optional[Trial] = None,
    ) -> Dict[str, Any]:
        X_train, y_train = train
        X_val, y_val = val

        # hyperparams space adapted from https://arxiv.org/pdf/2207.08815.pdf pg. 20
        if trial is None:
            params = wandb.config["params"]
            max_depth = params["max_depth"]
            min_samples_leaf = params["min_samples_leaf"]
        else:
            max_depth = trial.suggest_categorical("max_depth", [None, 2, 3, 4])
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5, log=True)
        n_estimators = 100

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
        train: Dataset,
        val: Dataset,
        trial: Optional[Trial] = None,
    ) -> Dict[str, Any]:
        X_train, y_train = train
        X_val, y_val = val

        # hyperparams space adapted from https://arxiv.org/pdf/2207.08815.pdf pg. 20
        if trial is None:
            params = wandb.config["params"]
            max_depth = params["max_depth"]
            lr = params["lr"]
            min_samples_leaf = params["min_samples_leaf"]
        else:
            max_depth = trial.suggest_categorical("max_depth", [None, 2, 3, 4])
            lr = trial.suggest_float("lr", 0.05, 0.5, step=0.05)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5, log=True)

        model = HistGradientBoostingClassifier(
            learning_rate=lr,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        self._model = model.fit(X_train, y_train)

        train_acc = self.top_1_acc((X_train, y_train))
        val_acc = self.top_1_acc((X_val, y_val))
        return {"train": {"acc": train_acc}, "val": {"acc": val_acc}}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return self._model.predict_proba(X)
