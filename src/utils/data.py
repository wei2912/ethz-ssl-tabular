from imblearn.datasets import make_imbalance
import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split

from collections.abc import Iterator
import itertools
import logging
import random
from typing import Callable, List, Optional, Tuple, Union

from models import SemiSLModel, SLModel
from utils.typing import Dataset

N_TEST: int = 1000
N_VAL: int = 1000
N_TRAIN: int = 2000

SMALL_SPLIT_VALS: List[float] = [0.025 * x for x in range(4, 0, -1)]
LARGE_SPLIT_VALS: List[float] = [0.25 * x for x in range(4, 0, -1)]
L_SPLITS: List[float] = LARGE_SPLIT_VALS + SMALL_SPLIT_VALS
L_UL_SPLITS: List[Tuple[float, float]] = list(
    filter(
        lambda t: t[0] + t[1] <= 1,
        itertools.chain(
            itertools.product(LARGE_SPLIT_VALS, LARGE_SPLIT_VALS),
            itertools.product(SMALL_SPLIT_VALS, LARGE_SPLIT_VALS),
            itertools.product(SMALL_SPLIT_VALS, SMALL_SPLIT_VALS),
        ),
    )
)


def __balance_data(
    X: pd.DataFrame, y: pd.Series, random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    n_samples = y.value_counts().min()
    return make_imbalance(
        X,
        y,
        sampling_strategy={label: n_samples for label in y.unique()},
        random_state=random_state,
    )


def prepare_train_test_val(
    dataset_id: int,
    seed: int,
    preprocess_func: Optional[Callable[[pd.DataFrame, pd.Series], Dataset]] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    n_class = len(y.unique())
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    X_bd, y_bd = __balance_data(X, y, random_state=seed)
    if preprocess_func is not None:
        X_pp, y_pp = preprocess_func(X_bd, y_bd)
    else:
        X_pp, y_pp = X_bd, y_bd

    logging.info(f"Original dataset size: {len(X)}")
    logging.info(f"Class-balanced dataset size: {len(X_pp)}")
    logging.info(f"No. of classes: {n_class}")
    logging.info(f"No. of samples per class: {len(X_pp) // n_class}")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_pp,
        y_pp,
        train_size=N_TRAIN + N_VAL,
        test_size=N_TEST,
        stratify=y_pp,
        random_state=seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        train_size=N_TRAIN,
        test_size=N_VAL,
        stratify=y_train_val,
        random_state=seed,
    )

    return ((X_train, y_train), (X_test, y_test), (X_val, y_val))


def get_splits(model: Union[SLModel, SemiSLModel]) -> Iterator[Tuple[float, float]]:
    if isinstance(model, SLModel):
        return zip(L_SPLITS, itertools.cycle([0.0]))
    elif isinstance(model, SemiSLModel):
        return iter(L_UL_SPLITS)
    else:
        raise NotImplementedError("model type not supported")


def prepare_l_ul(
    train: Dataset, l_split: float, ul_split: float, seed: int
) -> Tuple[Dataset, Dataset]:
    assert l_split > 0.0
    assert ul_split >= 0.0
    assert l_split + ul_split <= 1.0

    X_train, y_train = train
    if l_split < 1.0 and ul_split > 0.0:
        X_train_l, X_train_ul, y_train_l, y_train_ul = train_test_split(
            X_train,
            y_train,
            train_size=l_split,
            test_size=ul_split,
            stratify=y_train,
            random_state=seed,
        )
    elif l_split < 1.0 and ul_split == 0.0:
        X_train_l, _, y_train_l, _ = train_test_split(
            X_train,
            y_train,
            train_size=l_split,
            stratify=y_train,
            random_state=seed,
        )
        X_train_ul, y_train_ul = np.array([]), np.array([])
    elif l_split == 1.0:
        l_ids = random.Random(seed).sample(range(len(X_train)), k=len(X_train))
        X_train_l, y_train_l = X_train[l_ids], y_train[l_ids]
        X_train_ul, y_train_ul = np.array([]), np.array([])

    return ((X_train_l, y_train_l), (X_train_ul, y_train_ul))
