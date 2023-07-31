import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd

import abc
from typing import Any, Dict, Union

from utils.typing import Dataset


class Model(abc.ABC):
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Dataset:
        """
        :param X: input data
        :param y: labels of input data
        :return: processed input dataset
        """
        return (X.to_numpy(dtype=np.float32), y.cat.codes.to_numpy(dtype=np.int8))

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: test data
        :return: class probabilities
        """
        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def top_1_acc(self, dataset: Dataset) -> float:
        X, y = dataset
        assert X.shape[0] == y.shape[0]
        return (y == self.predict(X)).sum() / X.shape[0]


class SLModel(Model):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def train(
        self,
        trial: optuna.trial.Trial,
        train: Dataset,
        val: Dataset,
        is_sweep: bool,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        :param run: WandB run object
        :param trial: Optuna trial object
        :param train: training dataset
        :param val: validation dataset
        :param is_sweep: flag indicating if performing hyperparameter sweeps
        :return: log metrics
        """
        raise NotImplementedError()


class SemiSLModel(Model):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def train_ssl(
        self,
        trial: optuna.trial.Trial,
        train_l: Dataset,
        train_ul: Union[Dataset, npt.NDArray[np.float32]],
        val: Dataset,
        is_sweep: bool,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        :param run: WandB run object
        :param trial: Optuna trial object
        :param train_l: training data (labelled)
        :param train_ul: training data (unlabelled), either in the form of Dataset with
        the corresponding labels (not used for training), or as np.ndarray without the
        corresponding labels
        :param val: validation data
        :param is_sweep: flag indicating if performing hyperparameter sweeps
        :return: log metrics
        """
        raise NotImplementedError()
