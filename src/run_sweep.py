import openml
import wandb

import argparse

from models import BaseModel
from models.trees import RandomForestModel

NUM_SWEEP: int = 1
OPENML_SUITE_ID: int = 337  # classification on numerical features
# see https://github.com/LeoGrin/tabular-benchmark

MODELS: dict[str, BaseModel] = {
    "random-forest": RandomForestModel(),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--models", nargs="*", choices=MODELS.keys(), required=True)
    parser.add_argument("--prefix", type=str, default="ethz-tabular-ssl_")
    parser.add_argument("--suffix", type=str, default="")

    args = parser.parse_args()

    if len(args.models) == 0:
        print("No models were specified; exiting.")

    for model_name in args.models:
        project_name = f"{args.prefix}{model_name}{args.suffix}"
        wandb.init(entity=args.entity, project=project_name)

        model = MODELS[model_name]
        suite = openml.study.get_suite(OPENML_SUITE_ID)
        for task_id in suite.tasks[:1]:  # FIXME
            task = openml.tasks.get_task(task_id, download_splits=False)
            dataset = task.get_dataset()
            data = dataset.get_data(target=dataset.default_target_attribute)

            def train_func():
                wandb.init()
                print(wandb.config)
                X, y, _, _ = data
                metrics = model.fit(X, y, X, y, **wandb.config)
                wandb.log(metrics)

            sweep_id = wandb.sweep(sweep=model.SWEEP_CONFIG, project=project_name)
            wandb.agent(sweep_id, function=train_func, count=NUM_SWEEP)
