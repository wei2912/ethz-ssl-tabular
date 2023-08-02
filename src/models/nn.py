import numpy as np
from optuna.trial import Trial
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import wandb

from typing import Any, Dict, Optional

from . import SLModel
from utils.logging import Stepwise
from utils.typing import Dataset


class MLPBlock(nn.Module):
    """
    Implementation of MLP for tabular data, adapted from
    https://arxiv.org/pdf/2106.11959.pdf.
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.block = nn.ModuleList(
            [
                nn.Linear(input_size, output_size),
                nn.ReLU(),
            ]
        )

    def forward(self, x: torch.FloatTensor):
        for layer in self.block:
            x = layer(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_class: int,
        n_blocks: int,
        layer_size: int,
    ):
        super().__init__()

        assert n_blocks > 0
        self.blocks = nn.ModuleList(
            [MLPBlock(input_size, layer_size)]
            + [MLPBlock(layer_size, layer_size) for _ in range(n_blocks - 1)]
        )
        self.classifier = nn.Linear(layer_size, n_class)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)


MAX_BATCH_SIZE = 128
N_ITER = 1000
SEED = 123456


class MLPModel(SLModel):
    def __init__(self):
        super().__init__()

        self._is_device_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if self._is_device_cuda else "cpu")

    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Dataset:
        # transformation adapted from https://arxiv.org/pdf/2207.08815.pdf pg. 4
        qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
        cols = X.select_dtypes("number").columns.tolist()
        X[cols] = qt.fit_transform(X[cols].to_numpy())

        X = pd.get_dummies(X, dtype=np.float32)

        return (X.to_numpy(dtype=np.float32), y.cat.codes.to_numpy(dtype=np.int8))

    def train(
        self, train: Dataset, val: Dataset, trial: Optional[Trial], **_
    ) -> Dict[str, Any]:
        X_train, y_train = train
        X_val, y_val = val

        input_size = X_train.shape[1]
        n_class = len(np.unique(y_val))  # FIXME: contains all classes with high prob.

        # hyperparams space adapted from https://arxiv.org/pdf/2207.08815.pdf pg. 20
        if trial is None:
            layer_size = wandb.config["layer_size"]
            lr = wandb.config["lr"]
            batch_size = wandb.config["batch_size"]
        else:
            layer_size = trial.suggest_int("layer_size", 64, 256, step=64)
            lr = trial.suggest_float("lr", 0.001, 0.01, step=0.001)
            batch_size = trial.suggest_int("batch_size", 64, 128, step=32)
        n_blocks = 4

        # to ensure drop_last does not discard the entire dataset
        batch_size = min(batch_size, len(X_train))

        self._mlp = MLP(input_size, n_class, n_blocks, layer_size).to(
            self._device, non_blocking=True
        )

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self._is_device_cuda,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()
            ),
            batch_size=MAX_BATCH_SIZE,
            shuffle=False,
            pin_memory=self._is_device_cuda,
        )

        n_iter_per_epoch = len(train_loader)
        patience = N_ITER // (4 * n_iter_per_epoch)
        factor = 0.2

        optimizer = torch.optim.SGD(self._mlp.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, factor=factor
        )
        # account for different minibatch sizes
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        i, epoch = 0, 0
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        lrs = []
        while i < N_ITER:
            train_loss = 0.0
            for X_train_b, y_train_b in train_loader:
                if i >= N_ITER:
                    break

                self._mlp.train()

                X_train_b = X_train_b.to(self._device, non_blocking=True)
                y_train_b = y_train_b.to(self._device, non_blocking=True)

                optimizer.zero_grad()

                loss = criterion(self._mlp(X_train_b), y_train_b) / len(X_train_b)
                train_loss += loss.item()

                loss.backward()
                optimizer.step()

                i += 1
            else:
                self._mlp.eval()

                train_loss /= len(train_loader)
                train_acc = self.top_1_acc((X_train, y_train))
                train_losses.append(train_loss)
                train_accs.append(train_acc)

                val_loss = 0.0
                for X_val_b, y_val_b in val_loader:
                    X_val_b = X_val_b.to(self._device, non_blocking=True)
                    y_val_b = y_val_b.to(self._device, non_blocking=True)

                    z_val_b = self._mlp(X_val_b)
                    val_loss += criterion(z_val_b, y_val_b).item() / len(X_val_b)
                val_loss /= len(val_loader)
                val_acc = self.top_1_acc((X_val, y_val))

                scheduler.step(val_loss)

                lrs.append(optimizer.param_groups[0]["lr"])
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                epoch += 1

        train_acc = self.top_1_acc((X_train, y_train))
        val_acc = self.top_1_acc((X_val, y_val))
        return {
            "train": {
                "acc": train_acc,
                "batch_size": batch_size,
                "max_epochs": N_ITER // n_iter_per_epoch,
                "policy": {
                    "patience": patience,
                    "factor": factor,
                },
                "per_epoch": {
                    "lrs": Stepwise(lrs),
                    "train_losses": Stepwise(train_losses),
                    "train_accs": Stepwise(train_accs),
                    "val_losses": Stepwise(val_losses),
                    "val_accs": Stepwise(val_accs),
                },
                "size": len(X_train),
            },
            "val": {"acc": val_acc},
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._mlp.eval()

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X).float()),
            batch_size=MAX_BATCH_SIZE,
            shuffle=False,
            pin_memory=self._is_device_cuda,
        )

        probss = []
        for (X_b,) in loader:
            X_b = X_b.to(self._device, non_blocking=True)
            probss.append(self._mlp(X_b).detach().cpu().numpy())

        return np.concatenate(probss)
