import openml
import optuna
from optuna.integration import WeightsAndBiasesCallback
from scipy.stats import hmean
from tqdm.auto import tqdm
import wandb

import argparse
import os
import random
from typing import Callable, Dict, Tuple, Union
import warnings

from utils.data import get_splits, prepare_train_test_val, prepare_l_ul
from utils.logging import convert_metrics
from models import SemiSLModel, SLModel
from models.nn import MLPModel
from models.self_training import (
    SelfTrainingModel_Curriculum,
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
    dataset_name, model_name, entity, prefix, seed = (
        args.dataset,
        args.model,
        args.entity,
        args.prefix,
        args.seed,
    )

    dataset_id = DATASETS[dataset_name]
    project_name = f"{prefix}{dataset_name}"

    print("===")
    print("*** eval ***")
    print(f"Project: {project_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Model: {model_name}")
    print("---")
    print(f"Seed: {seed}")
    print("---")

    (X_train, y_train), (X_test, y_test), (X_val, y_val) = next(
        prepare_train_test_val(
            dataset_id, 1, seed, preprocess_func=MODELS[model_name]().preprocess_data
        )
    )

    print(f"> Train/Test/Val Split: {len(X_train)}/{len(X_test)}/{len(X_val)}")

    for l_split, ul_split in tqdm(get_splits(MODELS[model_name]())):
        wandb.init(
            job_type="eval",
            config={
                **vars(args),
                "l_split": l_split,
                "ul_split": ul_split,
            },
            entity=entity,
            project=project_name,
            group=f"{model_name}_{l_split:.3}_{ul_split:.3}",
        )

        (X_train_l, y_train_l), (X_train_ul, y_train_ul) = prepare_l_ul(
            (X_train, y_train), l_split, ul_split, seed
        )

        print(
            f">> L/UL Split: {len(X_train_l)}/{len(X_train_ul)} "
            f"({l_split:.3}/{ul_split:.3})"
        )

        model = MODELS[model_name]()
        if isinstance(model, SLModel):
            run_metrics = model.train((X_train_l, y_train_l), (X_val, y_val))
        elif isinstance(model, SemiSLModel):
            run_metrics = model.train_ssl(
                (X_train_l, y_train_l),
                (X_train_ul, y_train_ul),
                (X_val, y_val),
            )

        test_acc = model.top_1_acc((X_test, y_test))
        test_metrics = {"acc": test_acc}

        non_step_metric_dict, step_metric_dicts = convert_metrics(
            {
                "run": run_metrics,
                "test": test_metrics,
            }
        )
        for metric_dict in [non_step_metric_dict] + step_metric_dicts:
            wandb.log(metric_dict)

        wandb.finish()


def run_sweep(args: argparse.Namespace) -> None:
    dataset_name: str
    model_name: str
    entity: str
    prefix: str
    seed: int
    n_sweep: int
    n_split: int
    dataset_name, model_name, entity, prefix, seed, n_sweep, n_split = (
        args.dataset,
        args.model,
        args.entity,
        args.prefix,
        args.seed,
        args.n_sweep,
        args.n_split,
    )
    assert n_sweep > 0
    assert n_split > 0

    dataset_id = DATASETS[dataset_name]
    project_name = f"{prefix}{dataset_name}"

    print("===")
    print("*** sweep ***")
    print(f"Project: {project_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Model: {model_name}")
    print("---")
    print(f"Seed: {seed}")
    print(f"No. of hyperparameter sweeps: {n_sweep}")
    print(f"No. of splits: {n_split}")
    print("---")

    (X_train, y_train), (X_test, y_test), (X_val, y_val) = next(
        prepare_train_test_val(
            dataset_id, 1, seed, preprocess_func=MODELS[model_name]().preprocess_data
        )
    )

    print(f"> Train/Val Split: {len(X_train)}/{len(X_val)}")

    model = MODELS[model_name]()
    if isinstance(model, SLModel):
        l_split_small, ul_split_small = (0.025, 0.0)
        l_split_large, ul_split_large = (0.25, 0.0)
    elif isinstance(model, SemiSLModel):
        l_split_small, ul_split_small = (0.025, 0.05)
        l_split_large, ul_split_large = (0.25, 0.5)
    else:
        raise NotImplementedError("model type not supported")

    wandbc = WeightsAndBiasesCallback(
        wandb_kwargs={
            "job_type": "sweep",
            "config": {
                "model": model_name,
                "l_split_small": l_split_small,
                "ul_split_small": ul_split_small,
                "l_split_large": l_split_large,
                "ul_split_large": ul_split_large,
                "seed": seed,
                "n_sweep": n_sweep,
            },
            "entity": entity,
            "project": project_name,
            "group": f"{model_name}",
        },
        as_multirun=True,
    )

    (X_train_l_small, _), (X_train_ul_small, _) = prepare_l_ul(
        (X_train, y_train), l_split_small, ul_split_small, seed
    )
    (X_train_l_large, _), (
        X_train_ul_large,
        _,
    ) = prepare_l_ul((X_train, y_train), l_split_large, ul_split_large, seed)

    print(
        f">> L/UL Split (Small): {len(X_train_l_small)}/{len(X_train_ul_small)} "
        f"({l_split_small:.3}/{ul_split_small:.3})"
    )
    print(
        f">> L/UL Split (Large): {len(X_train_l_large)}/{len(X_train_ul_large)} "
        f"({l_split_large:.3}/{ul_split_large:.3})"
    )

    study_name = f"{model_name}.{seed}.sweep_{random.randrange(0, 16**6):06x}"
    storage = f"sqlite:///{project_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(),
    )
    print(f">>> Study Name: {study_name}")
    print(f">>> Storage: {storage}")

    @wandbc.track_in_wandb()
    def objective_fn(trial: optuna.trial.Trial) -> Tuple[float, float]:
        splits = prepare_train_test_val(
            dataset_id,
            n_split,
            seed,
            preprocess_func=MODELS[model_name]().preprocess_data,
        )

        test_acc_smalls = []
        test_acc_larges = []
        metricss = {}
        for i, ((X_train, y_train), (X_test, y_test), (X_val, y_val)) in enumerate(
            splits
        ):
            (X_train_l_small, y_train_l_small), (
                X_train_ul_small,
                y_train_ul_small,
            ) = prepare_l_ul((X_train, y_train), l_split_small, ul_split_small, seed)
            (X_train_l_large, y_train_l_large), (
                X_train_ul_large,
                y_train_ul_large,
            ) = prepare_l_ul((X_train, y_train), l_split_large, ul_split_large, seed)

            model = MODELS[model_name]()
            if isinstance(model, SLModel):
                model = MODELS[model_name]()
                run_metrics_small = model.train(
                    (X_train_l_small, y_train_l_small), (X_val, y_val), trial=trial
                )
                test_acc_smalls.append(model.top_1_acc((X_test, y_test)))

                model = MODELS[model_name]()
                run_metrics_large = model.train(
                    (X_train_l_large, y_train_l_large), (X_val, y_val), trial=trial
                )
                test_acc_larges.append(model.top_1_acc((X_test, y_test)))
            elif isinstance(model, SemiSLModel):
                model = MODELS[model_name]()
                run_metrics_small = model.train_ssl(
                    (X_train_l_small, y_train_l_small),
                    (X_train_ul_small, y_train_ul_small),
                    (X_val, y_val),
                    trial=trial,
                )
                test_acc_smalls.append(model.top_1_acc((X_test, y_test)))

                model = MODELS[model_name]()
                run_metrics_large = model.train_ssl(
                    (X_train_l_large, y_train_l_large),
                    (X_train_ul_large, y_train_ul_large),
                    (X_val, y_val),
                    trial=trial,
                )
                test_acc_larges.append(model.top_1_acc((X_test, y_test)))

            metricss[f"split{i}"] = {
                "run_small": run_metrics_small,
                "run_large": run_metrics_large,
                "test_small": {"acc": test_acc_smalls[i]},
                "test_large": {"acc": test_acc_larges[i]},
            }

        test_hmean_acc_small = hmean(test_acc_smalls)
        test_hmean_acc_large = hmean(test_acc_larges)
        non_step_metric_dict, step_metric_dicts = convert_metrics(
            {
                **metricss,
                "params": trial.params,
                "test_small": {
                    "hmean_acc": test_hmean_acc_small,
                    "accs": test_acc_smalls,
                },
                "test_large": {
                    "hmean_acc": test_hmean_acc_large,
                    "accs": test_acc_larges,
                },
                "value": (test_hmean_acc_small, test_hmean_acc_large),
            }
        )
        for metric_dict in [non_step_metric_dict] + step_metric_dicts:
            wandb.log(metric_dict)

        return (test_hmean_acc_small, test_hmean_acc_large)

    study.optimize(
        objective_fn, n_trials=n_sweep, callbacks=[wandbc], show_progress_bar=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_preload_data = subparsers.add_parser("preload_data")
    parser_preload_data.set_defaults(func=preload_data)

    parser_eval = subparsers.add_parser("eval")
    parser_eval.add_argument("entity", type=str)
    parser_eval.add_argument("dataset", type=str, choices=DATASETS.keys())
    parser_eval.add_argument("model", type=str, choices=MODELS.keys())
    parser_eval.add_argument("seed", type=int)
    parser_eval.add_argument("--prefix", type=str, default="ethz-tabular-ssl_")
    hyperparams = parser_eval.add_argument_group("hyperparams")
    hyperparams.add_argument("--layer-size", type=int)
    hyperparams.add_argument("--lr", type=float)
    hyperparams.add_argument("--max-depth", type=int, default=None)
    hyperparams.add_argument("--min-samples-leaf", type=int)
    hyperparams.add_argument("--prob-threshold", type=float)
    parser_eval.set_defaults(func=run_eval)

    parser_sweep = subparsers.add_parser("sweep")
    parser_sweep.add_argument("entity", type=str)
    parser_sweep.add_argument("dataset", type=str, choices=DATASETS.keys())
    parser_sweep.add_argument("model", type=str, choices=MODELS.keys())
    parser_sweep.add_argument("seed", type=int)
    parser_sweep.add_argument("--prefix", type=str, default="ethz-tabular-ssl_")
    parser_sweep.add_argument("--n-sweep", type=int, default=40)
    parser_sweep.add_argument("--n-split", type=int, default=5)
    parser_sweep.set_defaults(func=run_sweep)

    args = parser.parse_args()

    os.environ["WANDB_SILENT"] = "true"
    warnings.filterwarnings("ignore", category=FutureWarning, module="openml.datasets")
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    args.func(args)
