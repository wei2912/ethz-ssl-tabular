import openml
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from tqdm.auto import tqdm
import wandb

import argparse
import os
import itertools
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

SEED: int = 123456
MODELS: Dict[str, Callable[[], Union[SLModel, SemiSLModel]]] = {
    "random-forest": lambda: RandomForestModel(),
    "random-forest-st-th-si": lambda: SelfTrainingModel_ThresholdSingleIterate(
        RandomForestModel
    ),
    "random-forest-st-curr": lambda: SelfTrainingModel_Curriculum(RandomForestModel),
    "random-forest-st-curr-si": lambda: SelfTrainingModel_CurriculumSingleIterate(
        RandomForestModel
    ),
    "hgbt": lambda: HGBTModel(),
    "hgbt-st-th-si": lambda: SelfTrainingModel_ThresholdSingleIterate(HGBTModel),
    "hgbt-st-curr": lambda: SelfTrainingModel_Curriculum(HGBTModel),
    "hgbt-st-curr-si": lambda: SelfTrainingModel_CurriculumSingleIterate(HGBTModel),
    "mlp": lambda: MLPModel(),
    "mlp-st-th-si": lambda: SelfTrainingModel_ThresholdSingleIterate(MLPModel),
    "mlp-st-curr": lambda: SelfTrainingModel_Curriculum(MLPModel),
    "mlp-st-curr-si": lambda: SelfTrainingModel_CurriculumSingleIterate(MLPModel),
}

N_TEST_VAL: int = 1000
VAL_SPLIT: float = 0.3

SMALL_SPLIT_VALS: List[float] = [0.0025 * x for x in range(4, 0, -1)]
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
        X, y = X.to_numpy(), y.cat.codes.to_numpy()

        # transformation adapted from https://arxiv.org/pdf/2207.08815.pdf pg. 4
        qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
        X_t = qt.fit_transform(X, y)

        X_train_full, X_test_val, y_train_full, y_test_val = train_test_split(
            X_t,
            y,
            test_size=N_TEST_VAL,
            random_state=SEED,
        )
        X_test, X_val, y_test, y_val = train_test_split(
            X_test_val, y_test_val, test_size=VAL_SPLIT, random_state=SEED + 1
        )

        print("===")
        print(f"Dataset: {dataset_str}")
        print(f"Dataset ID: {dataset_id}")
        print("---")
        print(f"Total: {len(X)}")
        print(f"Training: {len(X_train_full)}")
        print(f"Test: {len(X_test)}")
        print(f"Validation: {len(X_val)}")
        print("===")

        for model_name in models:

            def sweep_sl(l_split: float) -> Tuple[TrainFnType, TestFnType]:
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
                        random_state=SEED + 2,
                    )
                print(f">> L Split: {len(X_train)} ({l_split})")

                def train_fn(trial: optuna.trial.Trial) -> Tuple[float, Dict[str, Any]]:
                    return model.train(
                        trial, X_train, y_train, X_val, y_val, is_sweep=(num_sweep > 1)
                    )

                def test_fn() -> Dict[str, Any]:
                    test_acc = model.top_1_acc(X_test, y_test)
                    return {"acc": test_acc}

                return (train_fn, test_fn)

            def sweep_semisl(
                l_split: float, ul_split: float
            ) -> Tuple[TrainFnType, TestFnType]:
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
                        random_state=SEED + 2,
                    )
                print(
                    f">> L/UL Split: {len(X_train)}/{len(X_train_ul)} "
                    f"({l_split:.3}/{ul_split:.3})"
                )

                def train_fn(trial: optuna.trial.Trial) -> Tuple[float, Dict[str, Any]]:
                    return model.train_ssl(
                        trial,
                        X_train,
                        y_train,
                        X_train_ul,
                        X_val,
                        y_val,
                        is_sweep=(num_sweep > 1),
                    )

                def test_fn() -> Dict[str, Any]:
                    test_acc = model.top_1_acc(X_test, y_test)
                    return {"acc": test_acc}

                return (train_fn, test_fn)

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
                    train_fn, test_fn = sweep_sl(l_split)
                elif IS_SEMISL_MODEL:
                    train_fn, test_fn = sweep_semisl(l_split, ul_split)

                run_metricss = {}
                test_metricss = {}

                def get_score_and_log_metrics(
                    train_fn: TrainFnType, test_fn: TestFnType
                ):
                    @wandbc.track_in_wandb()
                    def objective_fn(trial: optuna.Trial):
                        score, run_metrics = train_fn(trial)
                        test_metrics = test_fn()
                        run_metricss[trial.number] = run_metrics
                        test_metricss[trial.number] = test_metrics

                        wandb.log(
                            {
                                "run": run_metrics,
                                "test": test_metrics,
                                "number": trial.number,
                                "params": trial.params,
                                "value": score,
                            }
                        )
                        return score

                    return objective_fn

                study.optimize(
                    get_score_and_log_metrics(train_fn, test_fn),
                    n_trials=num_sweep,
                    show_progress_bar=True,
                    callbacks=[wandbc],
                )

                metrics = flatten_metrics(
                    {
                        "best": {
                            "number": study.best_trial.number,
                            "params": study.best_trial.params,
                            "value": study.best_trial.value,
                            "run": run_metricss[study.best_trial.number],
                            "test": test_metricss[study.best_trial.number],
                        }
                    }
                )

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
