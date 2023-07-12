import openml
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import wandb

import argparse
import os
from typing import Callable, Dict, List, Tuple

from models import SemiSLModel
from models.trees import RandomForestModel
from models.self_training import WastefulSemiSLModel, SelfTrainingModel

MODELS: Dict[str, Callable[[], SemiSLModel]] = {
    "random-forest": lambda: WastefulSemiSLModel(RandomForestModel),
    "random-forest-st": lambda: SelfTrainingModel(RandomForestModel),
}
NUM_SWEEP: int = 10
OPENML_SUITE_ID: int = 337  # classification on numerical features
# see https://github.com/LeoGrin/tabular-benchmark
SEED: int = 123456
L_UL_SPLITS: List[Tuple[float, float]] = list(
    filter(
        lambda t: t[0] + t[1] <= 1,
        ((0.1 * (x + 1), 0.1 * (y + 1)) for x in range(10) for y in range(10)),
    )
)
VAL_SPLIT: float = 0.1

SUITE = openml.study.get_suite(OPENML_SUITE_ID)


def main(*, tasks: List[int], models: List[str], entity: str, prefix: str):
    os.environ["WANDB_SILENT"] = "true"

    if len(models) == 0:
        print("No models were specified; exiting.")

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
            print(f"> Model: {model_name}")

            for l_split, ul_split in tqdm(L_UL_SPLITS):
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
                model = MODELS[model_name]()

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

                project_name = f"{prefix}{task_id}"
                sweep_id = wandb.sweep(
                    sweep=model.SWEEP_CONFIG, entity=entity, project=project_name
                )
                wandb.agent(sweep_id, function=run_fn, count=NUM_SWEEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument(
        "--tasks", type=int, nargs="*", choices=SUITE.tasks, required=True
    )
    parser.add_argument(
        "--models", type=str, nargs="*", choices=MODELS.keys(), required=True
    )
    parser.add_argument("--prefix", type=str, default="ethz-tabular-ssl_")
    args = parser.parse_args()
    main(**vars(args))
