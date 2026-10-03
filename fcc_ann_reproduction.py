"""Reproduce Dasila et al.'s eight-input, three-network FCC ANN model.

The paper's Model 2 trains one network per output and normalizes the three
predictions afterward. MATLAB's Neural Network Toolbox also scales targets by
default; omitting that step makes LM training poorly conditioned.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_levenberg_marquardt as tlm
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


FEATURES = ["Sp. Gr", "5 %", "10 %", "30 %", "50 %", "70 %", "90 %", "95 %"]
TARGETS = ["P", "N", "A"]
ARCHITECTURES = {"P": (12, 13), "N": (12, 12), "A": (9, 9)}
DTYPE = torch.float64


class ANN(nn.Module):
    def __init__(self, hidden_1: int, hidden_2: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(len(FEATURES), hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.out = nn.Linear(hidden_2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))       # MATLAB tansig
        x = torch.sigmoid(self.fc2(x))    # MATLAB logsig
        return self.out(x)                # MATLAB purelin


@dataclass
class FittedTarget:
    model: ANN
    scaler: MinMaxScaler
    seed: int
    epoch: int
    test_rmse: float


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_excel(path, sheet_name="Feed Data")
    data = data[pd.to_numeric(data["Sample No."], errors="coerce").notna()].copy()
    data["Sample No."] = data["Sample No."].astype(int)
    data.loc[:, data.columns != "Sample No."] = data.loc[
        :, data.columns != "Sample No."
    ].apply(pd.to_numeric)
    return data


def train_once(
    x_train: torch.Tensor,
    y_train: np.ndarray,
    x_test: torch.Tensor,
    y_test: np.ndarray,
    target_scaler: MinMaxScaler,
    hidden_1: int,
    hidden_2: int,
    seed: int,
    max_epochs: int,
    max_failures: int = 6,
) -> FittedTarget:
    torch.manual_seed(seed)
    model = ANN(hidden_1, hidden_2).to(dtype=DTYPE)
    train_target = torch.tensor(y_train, dtype=DTYPE).reshape(-1, 1)
    loader = DataLoader(
        TensorDataset(x_train, train_target),
        batch_size=len(x_train),
        shuffle=False,
    )
    trainer = tlm.training.LevenbergMarquardtModule(
        model=model,
        loss_fn=tlm.loss.MSELoss(),
        learning_rate=1.0,
        attempts_per_step=10,
        solve_method="qr",
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    failures = 0

    for epoch in range(1, max_epochs + 1):
        # The package prints a progress bar for every one-epoch LM step.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            tlm.utils.fit(trainer, loader, epochs=1)

        with torch.no_grad():
            prediction = model(x_test).cpu().numpy()
        loss = float(np.mean((prediction - y_test) ** 2))

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            failures = 0
        else:
            failures += 1
            if failures >= max_failures:
                break

    model.load_state_dict(best_state)
    scaled_rmse = np.sqrt(best_loss)
    target_range = float(target_scaler.data_max_[0] - target_scaler.data_min_[0])
    return FittedTarget(
        model=model,
        scaler=target_scaler,
        seed=seed,
        epoch=best_epoch,
        test_rmse=scaled_rmse * target_range / 2.0,
    )


def predict_target(fitted: FittedTarget, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        scaled = fitted.model(x).cpu().numpy()
    return fitted.scaler.inverse_transform(scaled).ravel()


def metrics(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    observed = observed.ravel()
    predicted = predicted.ravel()
    rmse = float(np.sqrt(np.mean((observed - predicted) ** 2)))
    sse = float(np.sum((observed - predicted) ** 2))
    sst = float(np.sum((observed - observed.mean()) ** 2))
    return rmse, 1.0 - sse / sst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", nargs="?", type=Path, default=Path("FCC_feed_data.xlsx"))
    parser.add_argument("--starts", type=int, default=10, help="seeded restarts per target")
    parser.add_argument("--max-epochs", type=int, default=1000)
    args = parser.parse_args()

    data = load_data(args.workbook)

    # These row groups reproduce the workbook/paper organization. Rows 17-22
    # are the six validation observations printed in Tables 5 and 6.
    train = data[data["Sample No."].between(1, 16)]
    validation = data[data["Sample No."].between(17, 22)]
    test = data[data["Sample No."].between(23, 28)]

    input_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train_np = input_scaler.fit_transform(train[FEATURES])
    x_test_np = input_scaler.transform(test[FEATURES])
    x_validation_np = input_scaler.transform(validation[FEATURES])
    x_train = torch.tensor(x_train_np, dtype=DTYPE)
    x_test = torch.tensor(x_test_np, dtype=DTYPE)
    x_validation = torch.tensor(x_validation_np, dtype=DTYPE)

    fitted: dict[str, FittedTarget] = {}
    for target_index, target in enumerate(TARGETS):
        # MATLAB applies mapminmax to each output as well as each input.
        target_scaler = MinMaxScaler(feature_range=(-1, 1))
        y_train = target_scaler.fit_transform(train[[target]])
        y_test = target_scaler.transform(test[[target]])
        hidden_1, hidden_2 = ARCHITECTURES[target]

        candidates = [
            train_once(
                x_train,
                y_train,
                x_test,
                y_test,
                target_scaler,
                hidden_1,
                hidden_2,
                seed=1000 * target_index + seed,
                max_epochs=args.max_epochs,
            )
            for seed in range(args.starts)
        ]
        fitted[target] = min(candidates, key=lambda candidate: candidate.test_rmse)
        chosen = fitted[target]
        print(
            f"{target}: seed={chosen.seed}, best_epoch={chosen.epoch}, "
            f"test_RMSE={chosen.test_rmse:.3f}"
        )

    raw_prediction = np.column_stack(
        [predict_target(fitted[target], x_validation) for target in TARGETS]
    )
    prediction = 100.0 * raw_prediction / raw_prediction.sum(axis=1, keepdims=True)
    observed = validation[TARGETS].to_numpy(dtype=float)
    rmse, r2 = metrics(observed, prediction)

    result = pd.DataFrame({"Sample": validation["Sample No."].to_numpy()})
    for index, target in enumerate(TARGETS):
        result[f"{target}_exp"] = observed[:, index]
        result[f"{target}_pred"] = prediction[:, index]

    print("\nHeld-out validation (samples 17-22):")
    print(result.round(2).to_string(index=False))
    print(f"\nRMSE = {rmse:.3f}")
    print(f"R2   = {r2:.4f}")
    print("Paper Table 6 Model 2: RMSE = 1.71, R2 = 0.994")


if __name__ == "__main__":
    main()
