import numpy as np
import math

CORRUPTION_RATE = 0.6


def scarf_corrupt(X: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng()
    N_FEATURE = X.shape[1]
    N_CORR_FEATURE = math.ceil(CORRUPTION_RATE * N_FEATURE)

    X_row_ids = np.concatenate([np.full((N_CORR_FEATURE,), i) for i in range(len(X))])
    X_train_row_ids = np.concatenate(
        [rng.choice(len(X_train), size=N_CORR_FEATURE) for _ in range(len(X))]
    )
    X_corr_col_ids = np.concatenate(
        [
            rng.choice(N_FEATURE, size=N_CORR_FEATURE, replace=False)
            for _ in range(len(X))
        ]
    )

    X_corr = X.copy()
    X_corr[X_row_ids, X_corr_col_ids] = X_train[X_train_row_ids, X_corr_col_ids]
    return X_corr
