import openml
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import QuantileTransformer
from tqdm.auto import tqdm
import wandb

import argparse
import itertools
import os
import random
from typing import Any, Callable, Dict, List, Tuple, TypeAlias, Union

from log_utils import Stepwise, flatten_metrics
from models import SemiSLModel, SLModel
from models.nn import MLPModel
from models.self_training import (
    SelfTrainingModel_Curriculum,
    SelfTrainingModel_CurriculumSingleIterate,
    SelfTrainingModel_ThresholdSingleIterate,
)
from models.trees import HGBTModel, RandomForestModel

TrainFnType: TypeAlias = Callable[[optuna.trial.Trial], Tuple[float, Dict[str, Any]]]
TestFnType: TypeAlias = Callable[[], Dict[str, Any]]

MODELS: Dict[str, Callable[[], Union[SLModel, SemiSLModel]]] = {
    "random-forest": lambda: RandomForestModel(),
    "random-forest-st-th-si": lambda: SelfTrainingModel_ThresholdSingleIterate(
        RandomForestModel
    ),
    "random-forest-st-curr": lambda: SelfTrainingModel_Curriculum(RandomForestModel),
    "hgbt": lambda: HGBTModel(),
    "hgbt-st-th-si": lambda: SelfTrainingModel_ThresholdSingleIterate(HGBTModel),
    "hgbt-st-curr": lambda: SelfTrainingModel_Curriculum(HGBTModel),
    "mlp": lambda: MLPModel(),
    "mlp-st-th-si": lambda: SelfTrainingModel_ThresholdSingleIterate(MLPModel),
    "mlp-st-curr": lambda: SelfTrainingModel_Curriculum(MLPModel),
    "mlp-st-curr-si": lambda: SelfTrainingModel_CurriculumSingleIterate(MLPModel),
}

N_TEST: int = 1000
N_TRAIN: int = 1000
VAL_SPLIT: float = 0.3

SMALL_SPLIT_VALS: List[float] = [0.05 * x for x in range(4, 0, -1)]
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

# datasets are taken from https://arxiv.org/pdf/2207.08815.pdf pg. 13
# and https://arxiv.org/pdf/2106.03253.pdf pg. 12
DATASETS: Dict[str, int] = {
    # 57.5k samples, 55 features, 2 classes
    "jannis": 45021,
    # 13.9k samples, 130 features, 6 classes
    "gas-drift-different-concentrations": 1477,
}


def preload_data(_) -> None:
    for dataset_id in DATASETS.values():
        openml.datasets.get_dataset(dataset_id)


def convert_metrics(
    metrics: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    max_steps = max(
        (len(val) for val in metrics.values() if isinstance(val, Stepwise)),
        default=0,
    )
    non_step_metric_dict = {}
    step_metric_dicts = [{} for _ in range(max_steps)]
    for key, val in metrics.items():
        if isinstance(val, Stepwise):
            for i, val_item in enumerate(val):
                step_metric_dicts[i][key] = val_item
        else:
            non_step_metric_dict[key] = val

    return (non_step_metric_dict, step_metric_dicts)


def main(args: argparse.Namespace) -> None:
    dataset_name: str
    model_name: str
    entity: str
    prefix: str
    seed: int
    n_sweep: int
    dataset_name, model_name, entity, prefix, seed, n_sweep = (
        args.dataset,
        args.model,
        args.entity,
        args.prefix,
        args.seed,
        args.n_sweep,
    )
    assert n_sweep > 0

    os.environ["WANDB_SILENT"] = "true"

    dataset_id = DATASETS[dataset_name]
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    X, y = X.to_numpy(), y.cat.codes.to_numpy()

    # transformation adapted from https://arxiv.org/pdf/2207.08815.pdf pg. 4
    qt = QuantileTransformer(output_distribution="normal", random_state=seed)
    X_t = qt.fit_transform(X, y)

    project_name = f"{prefix}{dataset_name}"

    print("===")
    print(f"Project: {project_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Model: {model_name}")
    print("---")
    print(f"Seed: {seed}")
    print(f"No. of hyperparameter sweeps: {n_sweep}")
    print("===")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_t, y, train_size=N_TRAIN, test_size=N_TEST, random_state=seed
    )
    print(f"> Train/Test Split: {len(X_train_val)}/{len(X_test)}")

    is_sl_model = isinstance(MODELS[model_name](), SLModel)
    is_semisl_model = isinstance(MODELS[model_name](), SemiSLModel)

    splits: List[Tuple[float, float]]
    if is_sl_model:
        splits = zip(L_SPLITS, itertools.cycle([0.0]))
    elif is_semisl_model:
        splits = L_UL_SPLITS
    else:
        raise NotImplementedError("model type not supported")

    for l_split, ul_split in tqdm(splits):
        wandbc = WeightsAndBiasesCallback(
            wandb_kwargs={
                "config": {
                    "model": model_name,
                    "l_split": l_split,
                    "ul_split": ul_split,
                    "seed": seed,
                    "n_sweep": n_sweep,
                },
                "entity": entity,
                "project": project_name,
                "group": f"{model_name}_{l_split:.3}_{ul_split:.3}",
            },
        )

        run_metricss = {}
        test_metricss = {}

        assert l_split > 0.0
        assert ul_split >= 0.0
        assert l_split + ul_split <= 1.0

        if l_split < 1.0:
            sss = StratifiedShuffleSplit(
                n_splits=1,
                train_size=l_split,
                test_size=ul_split if ul_split > 0.0 else None,
                random_state=seed,
            )
            l_ids, ul_ids = next(sss.split(X_train_val, y_train_val))
        elif l_split == 1.0:
            l_ids, ul_ids = (
                random.Random(seed).sample(range(len(X_train_val)), k=len(X_train_val)),
                [],
            )

        X_train_l, y_train_l = X_train_val[l_ids], y_train_val[l_ids]
        X_train_ul, y_train_ul = X_train_val[ul_ids], y_train_val[ul_ids]

        print("---")
        print(
            f">> L/UL Split: {len(X_train_l)}/{len(X_train_ul)} "
            f"({l_split:.3}/{ul_split:.3})"
        )

        sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=seed)
        train_ids, val_ids = next(sss.split(X_train_l, y_train_l))
        X_train, y_train = X_train_l[train_ids], y_train_l[train_ids]
        X_val, y_val = X_train_l[val_ids], y_train_l[val_ids]

        print(
            f">> Train/Val Split: {len(X_train)}/{len(X_val)} "
            f"({1 - VAL_SPLIT:.3}/{VAL_SPLIT:.3})"
        )

        study = optuna.create_study(direction="maximize")

        @wandbc.track_in_wandb()
        def objective_fn(trial: optuna.trial.Trial) -> Tuple[float, Dict[str, Any]]:
            is_sweep = n_sweep > 1
            model = MODELS[model_name]()

            if is_sl_model:
                run_metrics = model.train(
                    trial, X_train, y_train, X_val, y_val, is_sweep=is_sweep
                )
            elif is_semisl_model:
                run_metrics = model.train_ssl(
                    trial,
                    X_train,
                    y_train,
                    X_train_ul,
                    X_val,
                    y_val,
                    is_sweep=is_sweep,
                    y_train_ul=y_train_ul,
                )

            test_acc = model.top_1_acc(X_test, y_test)
            test_metrics = {"acc": test_acc}

            run_metricss[trial.number] = run_metrics
            test_metricss[trial.number] = test_metrics

            wandb.log(
                {
                    f"trial{trial.number}": {
                        "run": run_metrics,
                        "test": test_metrics,
                        "params": trial.params,
                    }
                }
            )
            return test_acc

        study.optimize(
            objective_fn,
            n_trials=n_sweep,
            callbacks=[wandbc],
        )

        non_step_metric_dict, step_metric_dicts = convert_metrics(
            flatten_metrics(
                {
                    "best": {
                        "trial": study.best_trial.number,
                        "params": study.best_trial.params,
                        "value": study.best_trial.value,
                        "run": run_metricss[study.best_trial.number],
                        "test": test_metricss[study.best_trial.number],
                    },
                }
            )
        )
        wandb.log(non_step_metric_dict)
        for step_metric_dict in step_metric_dicts:
            wandb.log(step_metric_dict)

        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_preload_data = subparsers.add_parser("preload_data")
    parser_preload_data.set_defaults(func=preload_data)

    parser_run = subparsers.add_parser("run")
    parser_run.add_argument("--entity", type=str, required=True)
    parser_run.add_argument(
        "--dataset", type=str, choices=DATASETS.keys(), required=True
    )
    parser_run.add_argument("--model", type=str, choices=MODELS.keys(), required=True)
    parser_run.add_argument("--prefix", type=str, default="ethz-tabular-ssl_")
    parser_run.add_argument("--seed", type=int, default=0)
    parser_run.add_argument("--n-sweep", type=int, default=5)
    parser_run.set_defaults(func=main)

    args = parser.parse_args()
    args.func(args)
