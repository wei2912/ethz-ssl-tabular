import openml
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import wandb

import argparse
import os
import itertools
from typing import Any, Callable, Dict, List, Tuple, Union

from models import SemiSLModel, SLModel
from models.trees import HGBTModel, RandomForestModel
from models.self_training import SelfTrainingModel

SEED: int = 123456
MODELS: Dict[str, Callable[[], Union[SLModel, SemiSLModel]]] = {
    "random-forest": lambda: RandomForestModel(),
    "random-forest-st": lambda: SelfTrainingModel(RandomForestModel),
    "hgbt": lambda: HGBTModel(),
    "hgbt-st": lambda: SelfTrainingModel(HGBTModel),
}

VAL_SPLIT: float = 0.1
SMALL_SPLIT_VALS: List[float] = [0.002 * x for x in range(5, 0, -1)]
LARGE_SPLIT_VALS: List[float] = [0.2 * x - 0.1 for x in range(5, 1, -1)]
L_SPLITS: List[float] = [1.0] + LARGE_SPLIT_VALS + SMALL_SPLIT_VALS
L_UL_SPLITS: List[Tuple[float, float]] = list(
    filter(
        lambda t: t[0] + t[1] <= 1,
        itertools.chain(
            itertools.product(LARGE_SPLIT_VALS, LARGE_SPLIT_VALS),
            itertools.product([0.01], LARGE_SPLIT_VALS),
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


def main(args: argparse.Namespace) -> None:
    datasets: List[str]
    models: List[str]
    entity: str
    prefix: str
    num_sweep: int
    datasets, models, entity, prefix, num_sweep = (
        args.datasets,
        args.models,
        args.entity,
        args.prefix,
        args.num_sweep,
    )

    assert num_sweep > 0

    if not models:
        print("No models were specified; exiting.")
        return
    if not datasets:
        print("No datasets were specified; exiting.")
        return

    os.environ["WANDB_SILENT"] = "true"

    for dataset_str in datasets:
        dataset_id = DATASETS[dataset_str]
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        X, y = X.to_numpy(), y.cat.codes
        X_train_full, X_val, y_train_full, y_val = train_test_split(
            X, y, test_size=VAL_SPLIT, random_state=SEED
        )

        print("===")
        print(f"Dataset: {dataset_str}")
        print(f"Dataset ID: {dataset_id}")
        print("---")
        print(f"Total: {len(X)}")
        print(f"Training: {len(X_train_full)}")
        print(f"Validation: {len(X_val)}")
        print("===")

        for model_name in models:

            def sweep_sl(l_split: float) -> Callable[[], None]:
                assert l_split > 0.0
                assert l_split <= 1.0

                model = MODELS[model_name]()
                assert isinstance(model, SLModel)

                if l_split == 1.0:
                    X_train, y_train = X_train_full, y_train_full
                else:
                    X_train, _, y_train, _ = train_test_split(
                        X_train_full,
                        y_train_full,
                        train_size=l_split,
                        random_state=SEED + 1,
                    )
                print(f">> L Split: {len(X_train)} ({l_split})")

                def train_fn(trial: optuna.trial.Trial):
                    return model.train(
                        trial, X_train, y_train, X_val, y_val, is_sweep=(num_sweep > 1)
                    )

                return train_fn

            def sweep_semisl(l_split: float, ul_split: float) -> Callable[[], None]:
                assert l_split > 0.0
                assert ul_split >= 0.0
                assert l_split + ul_split <= 1.0

                model = MODELS[model_name]()
                assert isinstance(model, SemiSLModel)

                if l_split == 1.0:
                    X_train, X_train_ul, y_train = X_train_full, None, y_train_full
                else:
                    X_train, X_train_ul, y_train, _ = train_test_split(
                        X_train_full,
                        y_train_full,
                        train_size=l_split,
                        test_size=ul_split,
                        random_state=SEED + 1,
                    )
                print(
                    f">> L/UL Split: {len(X_train)}/{len(X_train_ul)} "
                    f"({l_split:.3}/{ul_split:.3})"
                )

                def train_fn(trial: optuna.trial.Trial):
                    return model.train_ssl(
                        trial,
                        X_train,
                        y_train,
                        X_train_ul,
                        X_val,
                        y_val,
                        is_sweep=(num_sweep > 1),
                    )

                return train_fn

            print(f"> Model: {model_name}")
            project_name = f"{prefix}{dataset_str}"

            IS_SL_MODEL = isinstance(MODELS[model_name](), SLModel)
            IS_SEMISL_MODEL = isinstance(MODELS[model_name](), SemiSLModel)

            if IS_SL_MODEL:
                SPLITS = L_SPLITS
            elif IS_SEMISL_MODEL:
                SPLITS = L_UL_SPLITS
            else:
                raise NotImplementedError("model type not supported")

            for split in tqdm(SPLITS):
                if IS_SL_MODEL:
                    l_split, ul_split = split, 0
                elif IS_SEMISL_MODEL:
                    l_split, ul_split = split

                wandbc = WeightsAndBiasesCallback(
                    wandb_kwargs={
                        "config": {
                            "model": model_name,
                            "l_split": l_split,
                            "ul_split": ul_split,
                        },
                        "entity": entity,
                        "project": project_name,
                    },
                )

                study = optuna.create_study(direction="maximize")
                if IS_SL_MODEL:
                    train_fn = sweep_sl(l_split)
                elif IS_SEMISL_MODEL:
                    train_fn = sweep_semisl(l_split, ul_split)

                @wandbc.track_in_wandb()
                def get_score_and_log_metrics(
                    train_fn: Callable[[optuna.Trial], Tuple[float, Dict[str, Any]]]
                ):
                    def objective_fn(trial: optuna.Trial):
                        score, metrics = train_fn(trial)
                        wandb.log(
                            {
                                "run": metrics,
                                "number": trial.number,
                                "params": trial.params,
                                "value": score,
                            }
                        )
                        return score

                    return objective_fn

                study.optimize(
                    get_score_and_log_metrics(train_fn),
                    n_trials=num_sweep,
                    show_progress_bar=True,
                    callbacks=[wandbc],
                )

                wandb.run.summary = {
                    "number": study.best_trial.number,
                    "params": study.best_trial.params,
                    "value": study.best_trial.value,
                }
                wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_preload_data = subparsers.add_parser("preload_data")
    parser_preload_data.set_defaults(func=preload_data)

    parser_run = subparsers.add_parser("run")
    parser_run.add_argument("--entity", type=str, required=True)
    parser_run.add_argument(
        "--datasets", type=str, nargs="*", choices=DATASETS.keys(), required=True
    )
    parser_run.add_argument(
        "--models", type=str, nargs="*", choices=MODELS.keys(), required=True
    )
    parser_run.add_argument("--prefix", type=str, default="ethz-tabular-ssl_")
    parser_run.add_argument("--num-sweep", type=int, default=20)
    parser_run.set_defaults(func=main)

    args = parser.parse_args()
    args.func(args)
