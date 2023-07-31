import openml
import optuna
from optuna.integration import WeightsAndBiasesCallback
from tqdm.auto import tqdm
import wandb

import argparse
import os
from typing import Any, Callable, Dict, Tuple, Union
import warnings

from utils.data import get_splits, prepare_train_test_val, prepare_l_ul
from utils.logging import convert_metrics
from models import SemiSLModel, SLModel
from models.nn import MLPModel
from models.self_training import (
    SelfTrainingModel_Curriculum,
    SelfTrainingModel_CurriculumSingleIterate,
    SelfTrainingModel_ThresholdSingleIterate,
)
from models.trees import HGBTModel, RandomForestModel

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

# datasets are taken from https://arxiv.org/pdf/2207.08815.pdf pg. 13
# and https://arxiv.org/pdf/2106.03253.pdf pg. 12
DATASETS: Dict[str, int] = {
    # 57.5k samples, 55 features, 2 classes
    "jannis": 45021,
    # 13.9k samples, 130 features, 6 classes
    "gas-drift-different-concentrations": 1477,
    # 98k samples, 29 features, 2 classes
    "higgs": 23512,
    # 581k samples, 55 features, 7 classes
    "covertype": 1596,
}


def preload_data(_) -> None:
    for dataset_id in DATASETS.values():
        openml.datasets.get_dataset(dataset_id)


def run_eval(args: argparse.Namespace) -> None:
    dataset_name: str
    model_name: str
    entity: str
    prefix: str
    seed: int
    n_trial: int
    dataset_name, model_name, entity, prefix, seed, n_trial = (
        args.dataset,
        args.model,
        args.entity,
        args.prefix,
        args.seed,
        args.n_trial,
    )
    assert n_trial > 0

    dataset_id = DATASETS[dataset_name]
    project_name = f"{prefix}{dataset_name}"

    print("===")
    print("*** run_eval ***")
    print(f"Project: {project_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Model: {model_name}")
    print("---")
    print(f"Seed: {seed}")
    print(f"No. of trials: {n_trial}")
    print("---")

    (X_train, y_train), (X_test, y_test), (X_val, y_val) = prepare_train_test_val(
        dataset_id, seed, preprocess_func=MODELS[model_name]().preprocess_data
    )

    print(f"> Train/Test/Val Split: {len(X_train)}/{len(X_test)}/{len(X_val)}")

    for l_split, ul_split in tqdm(get_splits(MODELS[model_name]())):
        wandbc = WeightsAndBiasesCallback(
            wandb_kwargs={
                "job_type": "run_eval",
                "config": {
                    "model": model_name,
                    "l_split": l_split,
                    "ul_split": ul_split,
                    "seed": seed,
                    "n_trial": n_trial,
                },
                "entity": entity,
                "project": project_name,
                "group": f"{model_name}_{l_split:.3}_{ul_split:.3}",
            },
        )

        run_metricss = {}
        test_metricss = {}

        (X_train_l, y_train_l), (X_train_ul, y_train_ul) = prepare_l_ul(
            (X_train, y_train), l_split, ul_split, seed
        )

        print(
            f">> L/UL Split: {len(X_train_l)}/{len(X_train_ul)} "
            f"({l_split:.3}/{ul_split:.3})"
        )

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.RandomSampler()
        )

        @wandbc.track_in_wandb()
        def objective_fn(trial: optuna.trial.Trial) -> Tuple[float, Dict[str, Any]]:
            model = MODELS[model_name]()
            if isinstance(model, SLModel):
                run_metrics = model.train(
                    trial, (X_train_l, y_train_l), (X_val, y_val), is_sweep=False
                )
            elif isinstance(model, SemiSLModel):
                run_metrics = model.train_ssl(
                    trial,
                    (X_train_l, y_train_l),
                    (X_train_ul, y_train_ul),
                    (X_val, y_val),
                    is_sweep=False,
                )

            val_acc = model.top_1_acc((X_val, y_val))
            test_acc = model.top_1_acc((X_test, y_test))
            test_metrics = {"acc": test_acc}

            run_metricss[trial.number] = run_metrics
            test_metricss[trial.number] = test_metrics

            non_step_metric_dict, step_metric_dicts = convert_metrics(
                {
                    f"trial{trial.number}": {
                        "run": run_metrics,
                        "test": test_metrics,
                        "params": trial.params,
                    },
                }
            )
            for metric_dict in [non_step_metric_dict] + step_metric_dicts:
                wandb.log(metric_dict)
            return val_acc

        study.optimize(
            objective_fn,
            n_trials=n_trial,
            callbacks=[wandbc],
        )

        non_step_metric_dict, step_metric_dicts = convert_metrics(
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
        for metric_dict in [non_step_metric_dict] + step_metric_dicts:
            wandb.log(metric_dict)

        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_preload_data = subparsers.add_parser("preload_data")
    parser_preload_data.set_defaults(func=preload_data)

    parser_run = subparsers.add_parser("run_eval")
    parser_run.add_argument("--entity", type=str, required=True)
    parser_run.add_argument(
        "--dataset", type=str, choices=DATASETS.keys(), required=True
    )
    parser_run.add_argument("--model", type=str, choices=MODELS.keys(), required=True)
    parser_run.add_argument("--prefix", type=str, default="ethz-tabular-ssl_")
    parser_run.add_argument("--seed", type=int, default=0)
    parser_run.add_argument("--n-trial", type=int, default=5)
    parser_run.set_defaults(func=run_eval)

    args = parser.parse_args()

    os.environ["WANDB_SILENT"] = "true"
    warnings.filterwarnings("ignore", category=FutureWarning, module="openml.datasets")
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    args.func(args)
