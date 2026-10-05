"""Reproduce Dasila et al.'s eight-input, three-network FCC ANN model.

The paper's Model 2 trains one network per output and normalizes the three
predictions afterward. MATLAB's Neural Network Toolbox also scales targets by
default; omitting that step makes LM training poorly conditioned.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.func import functional_call, jacrev
from torch.nn.utils import parameters_to_vector, vector_to_parameters

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
        x = torch.tanh(self.fc1(x))  # MATLAB tansig
        x = torch.sigmoid(self.fc2(x))  # MATLAB logsig
        return self.out(x)  # MATLAB purelin

    def initialize_nguyen_widrow(self, seed: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for layer in (self.fc1, self.fc2, self.out):
            neurons, inputs = layer.weight.shape
            weights = (
                torch.rand((neurons, inputs), generator=generator, dtype=DTYPE) - 0.5
            )
            weights /= torch.linalg.vector_norm(weights, dim=1, keepdim=True)
            beta = 0.7 * neurons ** (1.0 / inputs)
            layer.weight.data.copy_(beta * weights)

            biases = beta * (
                2.0 * torch.rand(neurons, generator=generator, dtype=DTYPE) - 1.0
            )
            layer.bias.data.copy_(biases)


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
    regularization_ratio: float = 1.0,
) -> FittedTarget:
    torch.manual_seed(seed)
    model = ANN(hidden_1, hidden_2).to(dtype=DTYPE)
    model.initialize_nguyen_widrow(seed)
    train_target = torch.tensor(y_train, dtype=DTYPE).reshape(-1, 1)
    named_parameters = list(model.named_parameters())
    parameter_names = [name for name, _ in named_parameters]
    parameter_sizes = [parameter.numel() for _, parameter in named_parameters]
    parameter_shapes = [parameter.shape for _, parameter in named_parameters]

    def residuals(flat_parameters: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(flat_parameters, parameter_sizes)
        parameters = {
            name: chunk.view(shape)
            for name, chunk, shape in zip(parameter_names, chunks, parameter_shapes)
        }
        prediction = functional_call(model, parameters, (x_train,))
        errors = prediction.ravel() - train_target.ravel()
        data_part = np.sqrt(regularization_ratio / errors.numel()) * errors
        if regularization_ratio == 1.0:
            return data_part

        # MATLAB msereg: ratio*MSE + (1-ratio)*mean(square(weights and biases)).
        weight_part = (
            np.sqrt((1.0 - regularization_ratio) / flat_parameters.numel())
            * flat_parameters
        )
        return torch.cat((data_part, weight_part))

    mu = 1e-3
    mu_decrease = 0.1
    mu_increase = 10.0
    mu_max = 1e10
    minimum_gradient = 1e-7
    with torch.no_grad():
        initial_prediction = model(x_test).cpu().numpy()
    best_loss = float(np.mean((initial_prediction - y_test) ** 2))
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    failures = 0

    for epoch in range(1, max_epochs + 1):
        flat_parameters = parameters_to_vector(model.parameters()).detach()
        residual_vector = residuals(flat_parameters)
        jacobian = jacrev(residuals)(flat_parameters)
        gradient = jacobian.T @ residual_vector
        gradient_norm = torch.linalg.vector_norm(gradient, ord=float("inf"))
        if gradient_norm <= minimum_gradient:
            break

        current_performance = torch.dot(residual_vector, residual_vector)
        accepted = False
        while mu <= mu_max:
            try:
                if jacobian.shape[0] < jacobian.shape[1]:
                    system = jacobian @ jacobian.T
                    system += mu * torch.eye(system.shape[0], dtype=DTYPE)
                    update = jacobian.T @ torch.linalg.solve(system, residual_vector)
                else:
                    system = jacobian.T @ jacobian
                    system += mu * torch.eye(system.shape[0], dtype=DTYPE)
                    update = torch.linalg.solve(system, gradient)
            except RuntimeError:
                mu *= mu_increase
                continue

            candidate = flat_parameters - update
            candidate_residuals = residuals(candidate)
            candidate_performance = torch.dot(candidate_residuals, candidate_residuals)
            if (
                torch.isfinite(candidate_performance)
                and candidate_performance < current_performance
            ):
                vector_to_parameters(candidate, model.parameters())
                mu = max(mu * mu_decrease, torch.finfo(DTYPE).eps)
                accepted = True
                break
            mu *= mu_increase

        if not accepted:
            break

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
    parser.add_argument(
        "workbook", nargs="?", type=Path, default=Path("FCC_feed_data.xlsx")
    )
    parser.add_argument(
        "--starts", type=int, default=10, help="seeded restarts per target"
    )
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument(
        "--msereg-ratio",
        type=float,
        default=0.45,
        help="A-model fraction assigned to MSE; MATLAB msereg default is 0.5",
    )
    args = parser.parse_args()
    if not 0.0 < args.msereg_ratio <= 1.0:
        parser.error("--msereg-ratio must be in (0, 1]")

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
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
                regularization_ratio=(args.msereg_ratio if target == "A" else 1.0),
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
