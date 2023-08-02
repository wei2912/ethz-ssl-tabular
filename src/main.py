import openml
import optuna
from optuna.integration import WeightsAndBiasesCallback
from scipy.stats import hmean
from tqdm.auto import tqdm
import wandb

import argparse
from pathlib import Path
import json
import logging
import os
from typing import Callable, Dict, Optional, Tuple, Union
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
    "random-forest": RandomForestModel,
    "hgbt": HGBTModel,
    "mlp": MLPModel,
}
ST_TYPES: Dict[Optional[str], Callable[[Union[SLModel, SemiSLModel]], SemiSLModel]] = {
    None: lambda model: model(),
    "th-si": lambda model: SelfTrainingModel_ThresholdSingleIterate(model),
    "curr": lambda model: SelfTrainingModel_Curriculum(model),
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
    dataset_name: str = args.dataset
    model_name: str = args.model
    st_type: str = args.st_type
    entity: str = args.entity
    prefix: str = args.prefix
    seed: int = args.seed

    dataset_id = DATASETS[dataset_name]
    project_name = f"{prefix}{dataset_name}"

    run = wandb.init(
        job_type="eval_load",
        config={"args": vars(args)},
        entity=entity,
        project=project_name,
    )

    sweep_config_art_name = f"sweep-config-{model_name}:latest"
    sweep_config_art_fp = f"{project_name}_{model_name}.json"
    run.use_artifact(sweep_config_art_name).get_path(sweep_config_art_fp).download(
        "config/"
    )
    with Path(f"config/{sweep_config_art_fp}").open() as f:
        params = json.load(f)["params"]

    run.finish()

    print("===")
    print("*** eval ***")
    print(f"Project: {project_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Model: {model_name}")
    print(f"ST Type: {st_type}")
    print("---")
    print(f"Seed: {seed}")
    print(f"Params: {params}")
    print("===")

    def model_fn() -> Union[SLModel, SemiSLModel]:
        return ST_TYPES[st_type](MODELS[model_name])

    (X_train, y_train), (X_test, y_test), (X_val, y_val) = next(
        prepare_train_test_val(
            dataset_id, 1, seed, preprocess_func=model_fn().preprocess_data
        )
    )

    print(f"> Train/Test/Val Split: {len(X_train)}/{len(X_test)}/{len(X_val)}")

    for l_split, ul_split in tqdm(get_splits(model_fn())):
        run = wandb.init(
            job_type="eval",
            config={
                "args": vars(args),
                "params": params,
                "split": {
                    "l_split": l_split,
                    "ul_split": ul_split,
                },
            },
            entity=entity,
            project=project_name,
            group=f"{model_name}_{st_type}_{l_split:.3}_{ul_split:.3}",
        )

        (X_train_l, y_train_l), (X_train_ul, y_train_ul) = prepare_l_ul(
            (X_train, y_train), l_split, ul_split, seed
        )

        print(
            f">> L/UL Split: {len(X_train_l)}/{len(X_train_ul)} "
            f"({l_split:.3}/{ul_split:.3})"
        )

        model = model_fn()
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
            {"run": run_metrics, "test": test_metrics}
        )
        for metric_dict in [non_step_metric_dict] + step_metric_dicts:
            run.log(metric_dict)

        run.finish()


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
    PARETO_TOLERANCE = 0.01

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
    print("===")

    (X_train, y_train), _, (X_val, _) = next(
        prepare_train_test_val(
            dataset_id, 1, seed, preprocess_func=MODELS[model_name]().preprocess_data
        )
    )

    print(f"> Train/Val Split: {len(X_train)}/{len(X_val)}")

    model = MODELS[model_name]()
    if isinstance(model, SLModel):
        l_split_0, ul_split_0 = (0.025, 0.0)
        l_split_1, ul_split_1 = (0.25, 0.0)
    elif isinstance(model, SemiSLModel):
        l_split_0, ul_split_0 = (0.025, 0.05)
        l_split_1, ul_split_1 = (0.25, 0.5)
    else:
        raise NotImplementedError("model type not supported")

    wandb_kwargs = {
        "job_type": "sweep",
        "config": {
            "args": vars(args),
            "split": {
                "l_split_0": l_split_0,
                "ul_split_0": ul_split_0,
                "l_split_1": l_split_1,
                "ul_split_1": ul_split_1,
            },
        },
        "entity": entity,
        "project": project_name,
        "group": f"{model_name}",
    }
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

    (X_train_l_0, _), (X_train_ul_0, _) = prepare_l_ul(
        (X_train, y_train), l_split_0, ul_split_0, seed
    )
    (X_train_l_1, _), (X_train_ul_1, _) = prepare_l_ul(
        (X_train, y_train), l_split_1, ul_split_1, seed
    )

    print(
        f">> L/UL Split (Small): {len(X_train_l_0)}/{len(X_train_ul_0)} "
        f"({l_split_0:.3}/{ul_split_0:.3})"
    )
    print(
        f">> L/UL Split (Large): {len(X_train_l_1)}/{len(X_train_ul_1)} "
        f"({l_split_1:.3}/{ul_split_1:.3})"
    )

    study_name = f"{model_name}.{seed}.sweep"
    storage_fp = Path(f"optuna/{project_name}.db")
    storage_fp.parent.mkdir(exist_ok=True, parents=True)

    storage_url = f"sqlite:///{storage_fp}"
    if any(
        study_name == summary.study_name
        for summary in optuna.study.get_all_study_summaries(storage=storage_url)
    ):
        logging.info(f"Deleted {study_name} from database.")
        optuna.delete_study(study_name=study_name, storage=storage_url)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(),
    )
    print(f">>> Study Name: {study_name}")
    print(f">>> Storage: {storage_fp}")

    @wandbc.track_in_wandb()
    def objective_fn(trial: optuna.trial.Trial) -> Tuple[float, float]:
        splits = prepare_train_test_val(
            dataset_id,
            n_split,
            seed,
            preprocess_func=MODELS[model_name]().preprocess_data,
        )

        test_acc_0s = []
        test_acc_1s = []
        metricss = {}
        for i, ((X_train, y_train), (X_test, y_test), (X_val, y_val)) in enumerate(
            splits
        ):
            (X_train_l_0, y_train_l_0), (X_train_ul_0, y_train_ul_0) = prepare_l_ul(
                (X_train, y_train), l_split_0, ul_split_0, seed
            )
            (X_train_l_1, y_train_l_1), (X_train_ul_1, y_train_ul_1) = prepare_l_ul(
                (X_train, y_train), l_split_1, ul_split_1, seed
            )

            model = MODELS[model_name]()
            if isinstance(model, SLModel):
                model = MODELS[model_name]()
                run_metrics_0 = model.train(
                    (X_train_l_0, y_train_l_0), (X_val, y_val), trial=trial
                )
                test_acc_0s.append(model.top_1_acc((X_test, y_test)))

                model = MODELS[model_name]()
                run_metrics_1 = model.train(
                    (X_train_l_1, y_train_l_1), (X_val, y_val), trial=trial
                )
                test_acc_1s.append(model.top_1_acc((X_test, y_test)))
            elif isinstance(model, SemiSLModel):
                model = MODELS[model_name]()
                run_metrics_0 = model.train_ssl(
                    (X_train_l_0, y_train_l_0),
                    (X_train_ul_0, y_train_ul_0),
                    (X_val, y_val),
                    trial=trial,
                )
                test_acc_0s.append(model.top_1_acc((X_test, y_test)))

                model = MODELS[model_name]()
                run_metrics_1 = model.train_ssl(
                    (X_train_l_1, y_train_l_1),
                    (X_train_ul_1, y_train_ul_1),
                    (X_val, y_val),
                    trial=trial,
                )
                test_acc_1s.append(model.top_1_acc((X_test, y_test)))

            metricss[f"split{i}"] = {
                "run_0": run_metrics_0,
                "run_1": run_metrics_1,
                "test_0": {"acc": test_acc_0s[i]},
                "test_1": {"acc": test_acc_1s[i]},
            }

        test_hmean_acc_0 = hmean(test_acc_0s)
        test_hmean_acc_1 = hmean(test_acc_1s)
        # round to closest multiple of PARETO_TOLERANCE (inexact)
        values = (
            round(test_hmean_acc_0 / PARETO_TOLERANCE) * PARETO_TOLERANCE,
            round(test_hmean_acc_1 / PARETO_TOLERANCE) * PARETO_TOLERANCE,
        )
        non_step_metric_dict, step_metric_dicts = convert_metrics(
            {
                **metricss,
                "test_0": {"hmean_acc": test_hmean_acc_0, "accs": test_acc_0s},
                "test_1": {"hmean_acc": test_hmean_acc_1, "accs": test_acc_1s},
            }
        )
        for metric_dict in [non_step_metric_dict] + step_metric_dicts:
            wandb.log(metric_dict)

        return values

    study.optimize(
        objective_fn, n_trials=n_sweep, callbacks=[wandbc], show_progress_bar=True
    )

    run = wandb.init(**{**wandb_kwargs, "job_type": "sweep_save"})

    """
    Choose hyperparameters that perform best in the high data regime, while still
    achieving good accuracies in the low data regime. This assumes that the sweep will
    eventually reach a set of "optimal" hyperparameters which achieve within
    PARETO_TOLERANCE/2 of the best accuracy in the high data regime, while also
    performing well in the low data regime.
    """
    best_trial = max(study.best_trials, key=lambda trial: trial.values[1])
    config_fp = Path(f"config/{project_name}_{model_name}.json")
    config_fp.parent.mkdir(exist_ok=True, parents=True)
    with config_fp.open(mode="w") as f:
        config = {
            "storage": str(storage_fp),
            "study": study_name,
            "number": best_trial.number,
            "values": best_trial.values,
            "params": best_trial.params,
        }
        print(config)
        json.dump(config, f)

    sweep_config_art = wandb.Artifact(name=f"sweep-config-{model_name}", type="json")
    sweep_config_art.add_file(config_fp)
    run.log_artifact(sweep_config_art)

    sweep_db_art = wandb.Artifact(name="sweep-db", type="database")
    sweep_db_art.add_file(storage_fp)
    run.log_artifact(sweep_db_art)

    run.finish()


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
    parser_eval.add_argument(
        "--st-type", type=str, choices=ST_TYPES.keys(), default=None
    )
    parser_eval.add_argument("--prefix", type=str, default="ethz-ssl-tabular_")
    parser_eval.set_defaults(func=run_eval)

    parser_sweep = subparsers.add_parser("sweep")
    parser_sweep.add_argument("entity", type=str)
    parser_sweep.add_argument("dataset", type=str, choices=DATASETS.keys())
    parser_sweep.add_argument("model", type=str, choices=MODELS.keys())
    parser_sweep.add_argument("seed", type=int)
    parser_sweep.add_argument("--prefix", type=str, default="ethz-ssl-tabular_")
    parser_sweep.add_argument("--n-sweep", type=int, default=40)
    parser_sweep.add_argument("--n-split", type=int, default=5)
    parser_sweep.set_defaults(func=run_sweep)

    args = parser.parse_args()

    os.environ["WANDB_SILENT"] = "true"
    warnings.filterwarnings("ignore", category=FutureWarning, module="openml.datasets")
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    args.func(args)
