from collections.abc import MutableMapping
from typing import Any, Dict, List, Tuple


class Stepwise(list):
    """
    Wrapper class to indicate list metrics that should be logged stepwise.
    """

    def __init__(self, seq: list):
        list.__init__(self, seq)


def __flatten_metrics(
    metrics_dict: MutableMapping, parent_key: str = "", sep: str = "."
):
    """
    Adapted from https://stackoverflow.com/a/6027615.
    """
    items = []
    for key, val in metrics_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(val, MutableMapping):
            items.extend(__flatten_metrics(val, new_key, sep=sep).items())
        else:
            items.append((new_key, val))
    return dict(items)


def convert_metrics(
    metrics: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metrics = __flatten_metrics(metrics)

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
