from collections.abc import MutableMapping


class Stepwise(list):
    """
    Wrapper class to indicate list metrics that should be logged stepwise.
    """

    def __init__(self, seq: list):
        list.__init__(self, seq)


def flatten_metrics(metrics_dict: MutableMapping, parent_key: str = "", sep: str = "."):
    """
    Adapted from https://stackoverflow.com/a/6027615.
    """
    items = []
    for key, val in metrics_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(val, MutableMapping):
            items.extend(flatten_metrics(val, new_key, sep=sep).items())
        else:
            items.append((new_key, val))
    return dict(items)
