import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from typing import Any, Dict, Tuple

from . import SLModel


class MLPBlock(nn.Module):
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
N_ITER = 2000


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
        n_classes = len(np.unique(y_train))

        # hyperparams space adapted from https://arxiv.org/pdf/2207.08815.pdf pg. 20
        if not is_sweep:
            n_blocks = trial.suggest_categorical("n_blocks", [8])
            layer_size = trial.suggest_categorical("layer_size", [512])
            dropout_p = trial.suggest_categorical("dropout_p", [0])
            lr = trial.suggest_categorical("lr", [1e-3])
        else:
            n_blocks = trial.suggest_int("n_blocks", 4, 16, log=True)
            layer_size = trial.suggest_int("layer_size", 64, 1024, log=True)
            dropout_p = trial.suggest_float("dropout_p", 0, 0.5)
            lr = trial.suggest_float("lr", 1e-4, 1e-2)

        self._mlp = MLP(input_size, n_classes, n_blocks, layer_size, dropout_p).to(
            self._device
        )

        optimizer = torch.optim.SGD(self._mlp.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        criterion = torch.nn.CrossEntropyLoss()

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
        )

        i = 0
        train_losses = []
        val_losses = []
        lrs = []
        while i < N_ITER:
            train_loss = 0.0
            for X_train_b, y_train_b in train_loader:
                if i > N_ITER:
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

                self._mlp.eval()

                i += 1
            else:
                train_losses.append(train_loss)

                val_loss = 0.0
                for X_val_b, y_val_b in val_loader:
                    X_val_b, y_val_b = X_val_b.to(self._device), y_val_b.to(
                        self._device
                    )
                    val_loss += criterion(self._mlp(X_val_b), y_val_b).item()
                val_losses.append(val_loss)

                scheduler.step(val_loss)
                lrs.append(optimizer.param_groups[0]["lr"])

        train_acc = self.top_1_acc(X_train, y_train)
        val_acc = self.top_1_acc(X_val, y_val)
        return (
            val_acc,
            {
                "train": {
                    "acc": train_acc,
                    "per_epoch": {
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "lrs": lrs,
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
        )

        probss = []
        for (X_batch,) in loader:
            X_batch = X_batch.to(self._device)
            probss.append(self._mlp(X_batch).detach().cpu().numpy())

        return np.concatenate(probss)
