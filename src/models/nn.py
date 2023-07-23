import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import math
from typing import Any, Dict, Tuple

from . import SLModel
from log_utils import Stepwise


class MLPBlock(nn.Module):
    """
    Implementation of MLP for tabular data, adapted from
    https://arxiv.org/pdf/2106.11959.pdf.
    """

    def __init__(self, input_size: int, output_size: int, dropout_p: float = 0.5):
        super().__init__()

        self.block = nn.ModuleList(
            [
                nn.Linear(input_size, output_size),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
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
        n_classes: int,
        n_blocks: int,
        layer_size: int,
        dropout_p: float,
    ):
        super().__init__()

        assert n_blocks > 0
        self.blocks = nn.ModuleList(
            [MLPBlock(input_size, layer_size, dropout_p=dropout_p)]
            + [
                MLPBlock(layer_size, layer_size, dropout_p=dropout_p)
                for _ in range(n_blocks - 1)
            ]
        )
        self.classifier = nn.Linear(layer_size, n_classes)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)


BATCH_SIZE = 128
N_ITER = 4000
N_ITER_PER_RESTART = 800
TOLERANCE = 1e-2
SEED = 654321


class MLPModel(SLModel):
    def __init__(self):
        super().__init__()

        self._is_device_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if self._is_device_cuda else "cpu")

    def train(
        self,
        trial: optuna.trial.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        is_sweep: bool = True,
    ) -> Tuple[float, Dict[str, Any]]:
        input_size = X_train.shape[1]
        # FIXME: should contain all classes with high prob.
        n_classes = len(np.unique(y_val))

        # hyperparams space adapted from https://arxiv.org/pdf/2207.08815.pdf pg. 20
        if not is_sweep:
            dropout_p = trial.suggest_categorical("dropout_p", [0])
        else:
            dropout_p = trial.suggest_float("dropout_p", 0, 0.1)
        n_blocks = trial.suggest_categorical("n_blocks", [8])
        layer_size = trial.suggest_categorical("layer_size", [512])
        lr = trial.suggest_categorical("lr", [0.1])

        self._mlp = MLP(input_size, n_classes, n_blocks, layer_size, dropout_p).to(
            self._device
        )

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=self._is_device_cuda,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=self._is_device_cuda,
        )

        N_ITER_PER_EPOCH = len(train_loader)
        N_EPOCH_PER_RESTART = max(math.ceil(N_ITER_PER_RESTART / N_ITER_PER_EPOCH), 40)
        PATIENCE = N_EPOCH_PER_RESTART // 4

        optimizer = torch.optim.SGD(self._mlp.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, N_EPOCH_PER_RESTART
        )
        criterion = torch.nn.CrossEntropyLoss()

        i, epoch = 0, 0
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        best_val_loss = float("inf")
        is_val_loss_betters = []
        lrs = []
        while i < N_ITER:
            train_loss = 0.0
            for X_train_b, y_train_b in train_loader:
                if i >= N_ITER:
                    break

                self._mlp.train()

                X_train_b, y_train_b = X_train_b.to(self._device), y_train_b.to(
                    self._device
                )

                optimizer.zero_grad()

                loss = criterion(self._mlp(X_train_b), y_train_b)
                train_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step(i / N_ITER_PER_EPOCH)

                i += 1
            else:
                lrs.append(scheduler.get_last_lr()[0])

                self._mlp.eval()

                train_acc = self.top_1_acc(X_train, y_train)
                train_losses.append(train_loss)
                train_accs.append(train_acc)

                val_loss = 0.0
                for X_val_b, y_val_b in val_loader:
                    X_val_b, y_val_b = X_val_b.to(self._device), y_val_b.to(
                        self._device
                    )
                    val_loss += criterion(self._mlp(X_val_b), y_val_b).item()
                val_acc = self.top_1_acc(X_val, y_val)
                is_val_loss_betters.append(val_loss < best_val_loss - TOLERANCE)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                epoch += 1

                # only stop early some time after a warm restart
                # and only perform early stopping after 2 restarts
                if all(
                    (
                        len(is_val_loss_betters) >= PATIENCE,
                        epoch >= 2 * N_EPOCH_PER_RESTART,
                        epoch % N_EPOCH_PER_RESTART >= 3 * PATIENCE,
                        not any(is_val_loss_betters[-PATIENCE:]),
                    )
                ):
                    break

        train_acc = self.top_1_acc(X_train, y_train)
        val_acc = self.top_1_acc(X_val, y_val)
        return (
            val_acc,
            {
                "train": {
                    "acc": train_acc,
                    "max_epochs": N_ITER // N_ITER_PER_EPOCH,
                    "policy": {
                        "n_epoch_per_restart": N_EPOCH_PER_RESTART,
                        "patience": PATIENCE,
                        "tolerance": TOLERANCE,
                    },
                    "per_epoch": {
                        "lrs": Stepwise(lrs),
                        "train_losses": Stepwise(train_losses),
                        "train_accs": Stepwise(train_accs),
                        "val_losses": Stepwise(val_losses),
                        "val_accs": Stepwise(val_accs),
                    },
                },
                "val": {"acc": val_acc},
            },
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._mlp.eval()

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X).float()),
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=self._is_device_cuda,
        )

        probss = []
        for (X_b,) in loader:
            X_b = X_b.to(self._device)
            probss.append(self._mlp(X_b).detach().cpu().numpy())

        return np.concatenate(probss)
