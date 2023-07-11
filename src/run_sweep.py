import openml
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import wandb

import argparse
import os
from typing import Dict, List

from models import SemiSLModel
from models.trees import RandomForestModel
from models.self_training import PseudolabelModel, SelfTrainingModel

MODELS: Dict[str, SemiSLModel] = {
    "random-forest-pl": lambda: PseudolabelModel(RandomForestModel),
    "random-forest-st": lambda: SelfTrainingModel(RandomForestModel),
}
NUM_SWEEP: int = 10
OPENML_SUITE_ID: int = 337  # classification on numerical features
# see https://github.com/LeoGrin/tabular-benchmark
SEED: int = 123456
UL_SPLITS: List[float] = [0.999, 0.995, 0.99, 0.9, 0.7, 0.5, 0.3, 0.1]
VAL_SPLIT: float = 0.1


def main(*, models: List[str], entity: str, project_name: str):
    os.environ["WANDB_SILENT"] = "true"

    if len(models) == 0:
        print("No models were specified; exiting.")

    suite = openml.study.get_suite(OPENML_SUITE_ID)

    for task_id in suite.tasks[:3]:  # FIXME
        task = openml.tasks.get_task(task_id, download_splits=True)
        dataset = task.get_dataset()
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
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

            for ul_split in tqdm(UL_SPLITS):
                X_train, X_train_ul, y_train, _ = train_test_split(
                    X_train_full,
                    y_train_full,
                    test_size=ul_split,
                    random_state=SEED + 1,
                )
                model = MODELS[model_name]()

                def run_fn():
                    wandb.init(
                        config={
                            "model": model_name,
                            "task": task_id,
                            "ul_split": ul_split,
                        }
                    )
                    train_metrics = model.train_ssl(
                        X_train, y_train, X_train_ul, **wandb.config
                    )
                    val_metrics = model.val(X_val, y_val)
                    wandb.log({"train": train_metrics, "val": val_metrics})
                    wandb.finish()

                sweep_id = wandb.sweep(
                    sweep=model.SWEEP_CONFIG, entity=entity, project=project_name
                )
                wandb.agent(sweep_id, function=run_fn, count=NUM_SWEEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--models", nargs="*", choices=MODELS.keys(), required=True)
    parser.add_argument("--project-name", type=str, default="ethz-tabular-ssl")
    args = parser.parse_args()
    main(**vars(args))
