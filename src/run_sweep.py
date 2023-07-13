import openml
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import wandb

import argparse
import os
import itertools
from typing import Any, Callable, Dict, List, Tuple, Union

from models import SemiSLModel, SLModel
from models.trees import GBTModel, RandomForestModel
from models.self_training import SelfTrainingModel

SEED: int = 123456
MODELS: Dict[str, Callable[[], Union[SLModel, SemiSLModel]]] = {
    "random-forest": lambda: RandomForestModel(),
    "random-forest-st": lambda: SelfTrainingModel(RandomForestModel),
    "gbt": lambda: GBTModel(),
    "gbt-st": lambda: SelfTrainingModel(GBTModel),
}
NUM_SWEEP: int = 20

VAL_SPLIT: float = 0.1
L_SPLITS: List[float] = [0.1] + [0.2 * (x + 1) for x in range(4)] + [0.9] + [1.0]
L_UL_SPLITS: List[Tuple[float, float]] = list(
    filter(lambda t: t[0] + t[1] <= 1, itertools.product(L_SPLITS, L_SPLITS))
)

# see https://github.com/LeoGrin/tabular-benchmark
# tasks are taken from the suite for "classification on numerical features"
# SUITE = openml.study.get_suite(337)
TASKS: List[int] = [
    361055,
    361060,
    361061,
    361062,
    361063,
    361065,
    361066,
    361068,
    361069,
    361070,
    361273,
    361274,
    361275,
    361276,
    361277,
    361278,
]


def preload_data(args: argparse.Namespace) -> None:
    tasks: List[int] = args.tasks

    if not tasks:
        print("No tasks were specified; exiting.")
        return

    for task_id in tasks:
        task = openml.tasks.get_task(task_id, download_splits=True)
        task.get_dataset()


def main(args: argparse.Namespace) -> None:
    tasks: List[int]
    models: List[str]
    entity: str
    prefix: str
    tasks, models, entity, prefix = args.tasks, args.models, args.entity, args.prefix

    if not models:
        print("No models were specified; exiting.")
        return
    if not tasks:
        print("No tasks were specified; exiting.")
        return

    os.environ["WANDB_SILENT"] = "true"

    for task_id in tasks:
        task = openml.tasks.get_task(task_id, download_splits=True)
        dataset = task.get_dataset()
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
        print(f"Task ID: {task_id}")
        print("---")
        print(f"Total: {len(X)}")
        print(f"Training: {len(X_train_full)}")
        print(f"Validation: {len(X_val)}")
        print("===")

        for model_name in models:

            def sweep_semisl(
                l_split: float, ul_split: float
            ) -> Tuple[Dict[str, Any], Callable[[], None]]:
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
                    f"({l_split}/{ul_split})"
                )

                def run_fn():
                    wandb.init(
                        config={
                            "model": model_name,
                            "l_split": l_split,
                            "ul_split": ul_split,
                        }
                    )
                    train_metrics = model.train_ssl(
                        X_train, y_train, X_train_ul, **wandb.config
                    )
                    val_metrics = model.val(X_val, y_val)
                    wandb.log({"train": train_metrics, "val": val_metrics})
                    wandb.finish()

                return (model.SWEEP_CONFIG, run_fn)

            def sweep_sl(l_split: float) -> Tuple[Dict[str, Any], Callable[[], None]]:
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

                def run_fn():
                    wandb.init(
                        config={
                            "model": model_name,
                            "l_split": l_split,
                            "ul_split": 0,
                        }
                    )
                    train_metrics = model.train(X_train, y_train, **wandb.config)
                    val_metrics = model.val(X_val, y_val)
                    wandb.log({"train": train_metrics, "val": val_metrics})
                    wandb.finish()

                return (model.SWEEP_CONFIG, run_fn)

            print(f"> Model: {model_name}")
            project_name = f"{prefix}{task_id}"

            if isinstance(MODELS[model_name](), SLModel):
                for l_split in tqdm(L_SPLITS):
                    sweep_config, run_fn = sweep_sl(l_split)
                    sweep_id = wandb.sweep(
                        sweep=sweep_config, entity=entity, project=project_name
                    )
                    wandb.agent(sweep_id, function=run_fn, count=NUM_SWEEP)
            elif isinstance(MODELS[model_name](), SemiSLModel):
                for l_split, ul_split in tqdm(L_UL_SPLITS):
                    sweep_config, run_fn = sweep_semisl(l_split, ul_split)
                    sweep_id = wandb.sweep(
                        sweep=sweep_config, entity=entity, project=project_name
                    )
                    wandb.agent(sweep_id, function=run_fn, count=NUM_SWEEP)
            else:
                raise NotImplementedError("model type not supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_preload_data = subparsers.add_parser("preload_data")
    parser_preload_data.add_argument(
        "--tasks", type=int, nargs="*", choices=TASKS, required=True
    )
    parser_preload_data.set_defaults(func=preload_data)

    parser_run = subparsers.add_parser("run")
    parser_run.add_argument("--entity", type=str, required=True)
    parser_run.add_argument(
        "--tasks", type=int, nargs="*", choices=TASKS, required=True
    )
    parser_run.add_argument(
        "--models", type=str, nargs="*", choices=MODELS.keys(), required=True
    )
    parser_run.add_argument("--prefix", type=str, default="ethz-tabular-ssl_")
    parser_run.set_defaults(func=main)

    args = parser.parse_args()
    args.func(args)
