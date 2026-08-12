from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MODEL_NAMES = ["1d_cnn", "lstm"]
SEEDS = [42, 43, 44]
BATCH_SIZE = 256
MAX_EPOCHS = 40
MIN_EPOCHS = 5
PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DECISION_THRESHOLD = 0.5


class OneDimensionalCnn(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.transpose(1, 2)).squeeze(1)


class LstmClassifier(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=32, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.output(self.dropout(hidden[-1])).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--targets", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-folds", type=int, default=0)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_model(name: str, input_size: int) -> nn.Module:
    if name == "1d_cnn":
        return OneDimensionalCnn(input_size)
    if name == "lstm":
        return LstmClassifier(input_size)
    raise ValueError(f"Unknown model: {name}")


def standardize(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return tuple(((array - mean) / std).astype(np.float32) for array in (train_x, *others))


def make_loader(x: np.ndarray, y: np.ndarray, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator, num_workers=0)


def class_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0:
        raise ValueError("Training data has no positive windows.")
    return torch.tensor([negatives / positives], dtype=torch.float32, device=device)


def train_epoch(model, loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(y_batch)
        total_rows += len(y_batch)
    return total_loss / total_rows


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        loss = criterion(model(x_batch), y_batch)
        total_loss += float(loss.detach().cpu()) * len(y_batch)
        total_rows += len(y_batch)
    return total_loss / total_rows


def select_epoch(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[int, float]:
    set_seed(seed)
    model = make_model(model_name, train_x.shape[-1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight(train_y, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = make_loader(train_x, train_y, True, seed)
    val_loader = make_loader(val_x, val_y, False, seed)

    best_epoch = 1
    best_val_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch >= MIN_EPOCHS and stale_epochs >= PATIENCE:
            break
    return best_epoch, best_val_loss


def fit_final_model(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
) -> nn.Module:
    set_seed(seed)
    model = make_model(model_name, train_x.shape[-1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight(train_y, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = make_loader(train_x, train_y, True, seed)
    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)
    return model


@torch.no_grad()
def predict_scores(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    loader = DataLoader(torch.from_numpy(x), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    scores = []
    model.eval()
    for x_batch in loader:
        logits = model(x_batch.to(device))
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


def selected_values(raw: str, defaults: list) -> list:
    if not raw:
        return defaults
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if defaults and isinstance(defaults[0], int):
        return [int(value) for value in values]
    return values


def main() -> None:
    args = parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = choose_device(args.device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cache_dir = args.manifest.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_filter = set(selected_values(args.targets, [entry["target"] for entry in manifest["targets"]]))
    model_names = selected_values(args.models, MODEL_NAMES)
    seeds = selected_values(args.seeds, SEEDS)
    blocks = [int(value) for value in manifest["blocks"]]
    if args.max_folds > 0:
        blocks = blocks[: args.max_folds]

    prediction_path = args.output_dir / "10_sequence_window_predictions.csv"
    run_path = args.output_dir / "10_sequence_training_runs.csv"
    prediction_tmp = prediction_path.with_suffix(".csv.tmp")
    run_tmp = run_path.with_suffix(".csv.tmp")
    prediction_fields = [
        "target", "model", "seed", "test_block", "validation_block", "cycle", "cycle_run",
        "cycle_occurrence", "scenario_block_25", "window_start_step", "window_end_step", "label",
        "score", "prediction", "decision_threshold", "best_epoch",
    ]
    run_fields = [
        "target", "model", "seed", "test_block", "validation_block", "core_train_blocks",
        "final_train_blocks", "core_train_windows", "validation_windows", "final_train_windows",
        "test_windows", "train_positive_count", "best_epoch", "best_validation_loss",
        "model_parameters", "device", "torch_version", "cuda_version", "elapsed_seconds",
    ]

    with prediction_tmp.open("w", newline="", encoding="utf-8") as prediction_handle, run_tmp.open(
        "w", newline="", encoding="utf-8"
    ) as run_handle:
        prediction_writer = csv.DictWriter(prediction_handle, fieldnames=prediction_fields)
        run_writer = csv.DictWriter(run_handle, fieldnames=run_fields)
        prediction_writer.writeheader()
        run_writer.writeheader()

        for entry in manifest["targets"]:
            target = entry["target"]
            if target not in target_filter:
                continue
            arrays = np.load(cache_dir / entry["file"])
            x = arrays["X"].astype(np.float32)
            y = arrays["y"].astype(np.int64)
            window_blocks = arrays["scenario_block_25"].astype(np.int64)

            for model_name in model_names:
                for test_block in blocks:
                    test_index_in_all = manifest["blocks"].index(test_block)
                    validation_block = int(manifest["blocks"][(test_index_in_all + 1) % len(manifest["blocks"])])
                    test_mask = window_blocks == test_block
                    validation_mask = window_blocks == validation_block
                    core_train_mask = ~(test_mask | validation_mask)
                    final_train_mask = ~test_mask

                    for seed in seeds:
                        started = time.perf_counter()
                        core_train_x, val_x = standardize(x[core_train_mask], x[validation_mask])
                        best_epoch, best_val_loss = select_epoch(
                            model_name,
                            core_train_x,
                            y[core_train_mask],
                            val_x,
                            y[validation_mask],
                            seed,
                            device,
                        )
                        final_train_x, test_x = standardize(x[final_train_mask], x[test_mask])
                        model = fit_final_model(
                            model_name,
                            final_train_x,
                            y[final_train_mask],
                            best_epoch,
                            seed,
                            device,
                        )
                        scores = predict_scores(model, test_x, device)
                        predictions = (scores > DECISION_THRESHOLD).astype(np.int64)
                        elapsed = time.perf_counter() - started
                        parameter_count = sum(parameter.numel() for parameter in model.parameters())

                        test_indices = np.flatnonzero(test_mask)
                        for local_index, source_index in enumerate(test_indices):
                            prediction_writer.writerow(
                                {
                                    "target": target,
                                    "model": model_name,
                                    "seed": seed,
                                    "test_block": test_block,
                                    "validation_block": validation_block,
                                    "cycle": int(arrays["cycle"][source_index]),
                                    "cycle_run": int(arrays["cycle_run"][source_index]),
                                    "cycle_occurrence": int(arrays["cycle_occurrence"][source_index]),
                                    "scenario_block_25": int(arrays["scenario_block_25"][source_index]),
                                    "window_start_step": int(arrays["window_start_step"][source_index]),
                                    "window_end_step": int(arrays["window_end_step"][source_index]),
                                    "label": int(y[source_index]),
                                    "score": float(scores[local_index]),
                                    "prediction": int(predictions[local_index]),
                                    "decision_threshold": DECISION_THRESHOLD,
                                    "best_epoch": best_epoch,
                                }
                            )
                        run_writer.writerow(
                            {
                                "target": target,
                                "model": model_name,
                                "seed": seed,
                                "test_block": test_block,
                                "validation_block": validation_block,
                                "core_train_blocks": int(np.unique(window_blocks[core_train_mask]).size),
                                "final_train_blocks": int(np.unique(window_blocks[final_train_mask]).size),
                                "core_train_windows": int(core_train_mask.sum()),
                                "validation_windows": int(validation_mask.sum()),
                                "final_train_windows": int(final_train_mask.sum()),
                                "test_windows": int(test_mask.sum()),
                                "train_positive_count": int(y[final_train_mask].sum()),
                                "best_epoch": best_epoch,
                                "best_validation_loss": best_val_loss,
                                "model_parameters": parameter_count,
                                "device": str(device),
                                "torch_version": torch.__version__,
                                "cuda_version": torch.version.cuda,
                                "elapsed_seconds": elapsed,
                            }
                        )
                        prediction_handle.flush()
                        run_handle.flush()
                        print(
                            f"{target} {model_name} seed={seed} test={test_block} "
                            f"val={validation_block} epoch={best_epoch} seconds={elapsed:.2f}",
                            flush=True,
                        )
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

    prediction_tmp.replace(prediction_path)
    run_tmp.replace(run_path)
    print(prediction_path)
    print(run_path)


if __name__ == "__main__":
    main()
