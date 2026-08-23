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


MODEL_NAMES = ["1d_cnn", "lstm_autoencoder"]
SEEDS = [42, 43, 44]
THRESHOLD_QUANTILES = [0.90, 0.95, 0.975]
RAW_FEATURE_COUNT = 19
BATCH_SIZE = 256
MAX_EPOCHS = 40
MIN_EPOCHS = 5
PATIENCE = 5
LEARNING_RATE = 1e-3
CNN_WEIGHT_DECAY = 1e-4
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


class LstmAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        context = hidden[-1].unsqueeze(1).expand(-1, x.shape[1], -1)
        decoded, _ = self.decoder(context)
        return self.output(decoded)


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


def selected_values(raw: str, defaults: list) -> list:
    if not raw:
        return defaults
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if defaults and isinstance(defaults[0], int):
        return [int(value) for value in values]
    return values


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


def standardize(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    flattened = train_x.reshape(-1, train_x.shape[-1])
    mean = flattened.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = flattened.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return tuple(((array - mean) / std).astype(np.float32) for array in (train_x, *others))


def classification_loader(x: np.ndarray, y: np.ndarray, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator, num_workers=0)


def autoencoder_loader(x: np.ndarray, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator, num_workers=0)


def class_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0:
        raise ValueError("Training data has no positive windows.")
    return torch.tensor([negatives / positives], dtype=torch.float32, device=device)


def train_classifier_epoch(model, loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(y_batch)
        total_rows += len(y_batch)
    return total_loss / total_rows


@torch.no_grad()
def classifier_loss(model, loader, criterion, device: torch.device) -> float:
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


def select_classifier_epoch(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[int, float]:
    set_seed(seed)
    model = OneDimensionalCnn(train_x.shape[-1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight(train_y, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=CNN_WEIGHT_DECAY)
    train_loader = classification_loader(train_x, train_y, True, seed)
    val_loader = classification_loader(val_x, val_y, False, seed)
    best_epoch, best_loss, stale = 1, float("inf"), 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_classifier_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = classifier_loss(model, val_loader, criterion, device)
        if val_loss < best_loss - 1e-5:
            best_epoch, best_loss, stale = epoch, val_loss, 0
        else:
            stale += 1
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    return best_epoch, best_loss


def fit_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
) -> nn.Module:
    set_seed(seed)
    model = OneDimensionalCnn(train_x.shape[-1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight(train_y, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=CNN_WEIGHT_DECAY)
    loader = classification_loader(train_x, train_y, True, seed)
    for _ in range(epochs):
        train_classifier_epoch(model, loader, optimizer, criterion, device)
    return model


@torch.no_grad()
def classifier_scores(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    loader = DataLoader(torch.from_numpy(x), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    scores = []
    model.eval()
    for x_batch in loader:
        scores.append(torch.sigmoid(model(x_batch.to(device))).cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


def train_autoencoder_epoch(model, loader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x_batch) - x_batch) ** 2)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(x_batch)
        total_rows += len(x_batch)
    return total_loss / total_rows


@torch.no_grad()
def autoencoder_loss(model, loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        loss = torch.mean((model(x_batch) - x_batch) ** 2)
        total_loss += float(loss.detach().cpu()) * len(x_batch)
        total_rows += len(x_batch)
    return total_loss / total_rows


def select_autoencoder_epoch(
    train_x: np.ndarray,
    val_x: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[int, float]:
    set_seed(seed)
    model = LstmAutoencoder(train_x.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = autoencoder_loader(train_x, True, seed)
    val_loader = autoencoder_loader(val_x, False, seed)
    best_epoch, best_loss, stale = 1, float("inf"), 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_autoencoder_epoch(model, train_loader, optimizer, device)
        val_loss = autoencoder_loss(model, val_loader, device)
        if val_loss < best_loss - 1e-5:
            best_epoch, best_loss, stale = epoch, val_loss, 0
        else:
            stale += 1
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    return best_epoch, best_loss


def fit_autoencoder(train_x: np.ndarray, epochs: int, seed: int, device: torch.device) -> nn.Module:
    set_seed(seed)
    model = LstmAutoencoder(train_x.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = autoencoder_loader(train_x, True, seed)
    for _ in range(epochs):
        train_autoencoder_epoch(model, loader, optimizer, device)
    return model


@torch.no_grad()
def reconstruction_scores(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    loader = DataLoader(torch.from_numpy(x), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    scores = []
    model.eval()
    for x_batch in loader:
        x_batch = x_batch.to(device)
        reconstruction = model(x_batch)
        scores.append(torch.mean((reconstruction - x_batch) ** 2, dim=(1, 2)).cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


def cycle_max_scores(scores: np.ndarray, cycle_runs: np.ndarray) -> np.ndarray:
    return np.asarray([scores[cycle_runs == cycle_run].max() for cycle_run in np.unique(cycle_runs)])


def validate_cached_arrays(manifest: dict, arrays_by_target: dict[str, dict[str, np.ndarray]]) -> None:
    if manifest["window_size"] != 10 or manifest["feature_count"] < RAW_FEATURE_COUNT:
        raise AssertionError("Unexpected sequence cache structure.")
    master = arrays_by_target["System_Failure"]
    metadata_fields = [
        "cycle", "cycle_run", "cycle_occurrence", "scenario_block_25", "window_start_step", "window_end_step"
    ]
    for target, arrays in arrays_by_target.items():
        if not np.array_equal(master["X"], arrays["X"]):
            raise AssertionError(f"Feature arrays differ for {target}.")
        for field in metadata_fields:
            if not np.array_equal(master[field], arrays[field]):
                raise AssertionError(f"Metadata {field} differs for {target}.")


def complete_normal_cycle_mask(cycle_runs: np.ndarray, system_labels: np.ndarray) -> np.ndarray:
    event_cycles = {
        int(cycle_run)
        for cycle_run in np.unique(cycle_runs)
        if system_labels[cycle_runs == cycle_run].max() == 1
    }
    return np.asarray([int(cycle_run) not in event_cycles for cycle_run in cycle_runs], dtype=bool)


def write_prediction_rows(
    writer,
    arrays: dict[str, np.ndarray],
    target: str,
    model_name: str,
    seed: int,
    test_block: int,
    validation_block: int,
    calibration_blocks: list[int],
    test_indices: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    threshold_quantile: str,
    best_epoch: int,
) -> None:
    for local_index, source_index in enumerate(test_indices):
        writer.writerow(
            {
                "target": target,
                "model": model_name,
                "seed": seed,
                "test_block": test_block,
                "validation_block": validation_block,
                "calibration_blocks": ";".join(str(value) for value in calibration_blocks),
                "cycle": int(arrays["cycle"][source_index]),
                "cycle_run": int(arrays["cycle_run"][source_index]),
                "cycle_occurrence": int(arrays["cycle_occurrence"][source_index]),
                "scenario_block_25": int(arrays["scenario_block_25"][source_index]),
                "window_start_step": int(arrays["window_start_step"][source_index]),
                "window_end_step": int(arrays["window_end_step"][source_index]),
                "label": int(labels[source_index]),
                "score": float(scores[local_index]),
                "prediction": int(predictions[local_index]),
                "decision_threshold": float(threshold),
                "threshold_quantile": threshold_quantile,
                "best_epoch": best_epoch,
            }
        )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = choose_device(args.device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cache_dir = args.manifest.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_names = [entry["target"] for entry in manifest["targets"]]
    target_filter = set(selected_values(args.targets, target_names))
    model_filter = set(selected_values(args.models, MODEL_NAMES))
    seeds = selected_values(args.seeds, SEEDS)
    blocks = [int(value) for value in manifest["blocks"]]
    if args.max_folds > 0:
        test_blocks = blocks[: args.max_folds]
    else:
        test_blocks = blocks

    arrays_by_target = {}
    for entry in manifest["targets"]:
        with np.load(cache_dir / entry["file"]) as loaded:
            arrays_by_target[entry["target"]] = {name: loaded[name].copy() for name in loaded.files}
    validate_cached_arrays(manifest, arrays_by_target)
    master = arrays_by_target["System_Failure"]
    x = master["X"][:, :, :RAW_FEATURE_COUNT].astype(np.float32)
    window_blocks = master["scenario_block_25"].astype(np.int64)
    cycle_runs = master["cycle_run"].astype(np.int64)
    normal_cycle_mask = complete_normal_cycle_mask(cycle_runs, master["y"].astype(np.int64))

    prediction_path = args.output_dir / "12_matched_torch_window_predictions.csv"
    run_path = args.output_dir / "12_matched_torch_runs.csv"
    prediction_tmp = prediction_path.with_suffix(".csv.tmp")
    run_tmp = run_path.with_suffix(".csv.tmp")
    prediction_fields = [
        "target", "model", "seed", "test_block", "validation_block", "calibration_blocks",
        "cycle", "cycle_run", "cycle_occurrence", "scenario_block_25", "window_start_step",
        "window_end_step", "label", "score", "prediction", "decision_threshold",
        "threshold_quantile", "best_epoch",
    ]
    run_fields = [
        "training_scope", "model", "seed", "test_block", "validation_block", "calibration_blocks",
        "core_train_blocks", "final_train_blocks", "core_train_windows", "validation_windows",
        "final_train_windows", "calibration_windows", "calibration_normal_cycles", "test_windows",
        "train_positive_count", "best_epoch", "best_validation_loss", "threshold_q90",
        "threshold_q95", "threshold_q975", "model_parameters", "device", "torch_version",
        "cuda_version", "elapsed_seconds",
    ]

    with prediction_tmp.open("w", newline="", encoding="utf-8") as prediction_handle, run_tmp.open(
        "w", newline="", encoding="utf-8"
    ) as run_handle:
        prediction_writer = csv.DictWriter(prediction_handle, fieldnames=prediction_fields)
        run_writer = csv.DictWriter(run_handle, fieldnames=run_fields)
        prediction_writer.writeheader()
        run_writer.writeheader()

        if "1d_cnn" in model_filter:
            for target in target_names:
                if target not in target_filter:
                    continue
                labels = arrays_by_target[target]["y"].astype(np.int64)
                for test_block in test_blocks:
                    block_index = blocks.index(test_block)
                    validation_block = blocks[(block_index + 1) % len(blocks)]
                    test_mask = window_blocks == test_block
                    validation_mask = window_blocks == validation_block
                    core_train_mask = ~(test_mask | validation_mask)
                    final_train_mask = ~test_mask
                    for seed in seeds:
                        started = time.perf_counter()
                        core_x, val_x = standardize(x[core_train_mask], x[validation_mask])
                        best_epoch, best_val_loss = select_classifier_epoch(
                            core_x, labels[core_train_mask], val_x, labels[validation_mask], seed, device
                        )
                        final_x, test_x = standardize(x[final_train_mask], x[test_mask])
                        model = fit_classifier(final_x, labels[final_train_mask], best_epoch, seed, device)
                        scores = classifier_scores(model, test_x, device)
                        predictions = (scores > DECISION_THRESHOLD).astype(np.int64)
                        elapsed = time.perf_counter() - started
                        test_indices = np.flatnonzero(test_mask)
                        write_prediction_rows(
                            prediction_writer, master, target, "1d_cnn_19_raw", seed, test_block,
                            validation_block, [], test_indices, labels, scores, predictions,
                            DECISION_THRESHOLD, "", best_epoch,
                        )
                        run_writer.writerow(
                            {
                                "training_scope": target,
                                "model": "1d_cnn_19_raw",
                                "seed": seed,
                                "test_block": test_block,
                                "validation_block": validation_block,
                                "calibration_blocks": "",
                                "core_train_blocks": 7,
                                "final_train_blocks": 8,
                                "core_train_windows": int(core_train_mask.sum()),
                                "validation_windows": int(validation_mask.sum()),
                                "final_train_windows": int(final_train_mask.sum()),
                                "calibration_windows": 0,
                                "calibration_normal_cycles": 0,
                                "test_windows": int(test_mask.sum()),
                                "train_positive_count": int(labels[final_train_mask].sum()),
                                "best_epoch": best_epoch,
                                "best_validation_loss": best_val_loss,
                                "threshold_q90": "",
                                "threshold_q95": "",
                                "threshold_q975": "",
                                "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
                                "device": str(device),
                                "torch_version": torch.__version__,
                                "cuda_version": torch.version.cuda,
                                "elapsed_seconds": elapsed,
                            }
                        )
                        prediction_handle.flush()
                        run_handle.flush()
                        print(
                            f"{target} 1d_cnn_19_raw seed={seed} test={test_block} "
                            f"epoch={best_epoch} seconds={elapsed:.2f}",
                            flush=True,
                        )
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

        if "lstm_autoencoder" in model_filter:
            for test_block in test_blocks:
                block_index = blocks.index(test_block)
                validation_block = blocks[(block_index + 1) % len(blocks)]
                calibration_blocks = [
                    blocks[(block_index + 2) % len(blocks)],
                    blocks[(block_index + 3) % len(blocks)],
                ]
                test_mask = window_blocks == test_block
                validation_mask = (window_blocks == validation_block) & normal_cycle_mask
                calibration_mask = np.isin(window_blocks, calibration_blocks) & normal_cycle_mask
                core_train_mask = (
                    ~np.isin(window_blocks, [test_block, validation_block, *calibration_blocks])
                    & normal_cycle_mask
                )
                final_train_mask = core_train_mask | validation_mask
                for seed in seeds:
                    started = time.perf_counter()
                    core_x, val_x = standardize(x[core_train_mask], x[validation_mask])
                    best_epoch, best_val_loss = select_autoencoder_epoch(core_x, val_x, seed, device)
                    final_x, calibration_x, test_x = standardize(
                        x[final_train_mask], x[calibration_mask], x[test_mask]
                    )
                    model = fit_autoencoder(final_x, best_epoch, seed, device)
                    calibration_scores = reconstruction_scores(model, calibration_x, device)
                    calibration_cycle_scores = cycle_max_scores(
                        calibration_scores, cycle_runs[calibration_mask]
                    )
                    thresholds = {
                        quantile: float(np.quantile(calibration_cycle_scores, quantile, method="higher"))
                        for quantile in THRESHOLD_QUANTILES
                    }
                    test_scores = reconstruction_scores(model, test_x, device)
                    elapsed = time.perf_counter() - started
                    test_indices = np.flatnonzero(test_mask)
                    for target in target_names:
                        if target not in target_filter:
                            continue
                        labels = arrays_by_target[target]["y"].astype(np.int64)
                        for quantile in THRESHOLD_QUANTILES:
                            threshold = thresholds[quantile]
                            predictions = (test_scores > threshold).astype(np.int64)
                            write_prediction_rows(
                                prediction_writer, master, target, "lstm_autoencoder_19_raw", seed,
                                test_block, validation_block, calibration_blocks, test_indices, labels,
                                test_scores, predictions, threshold, str(quantile), best_epoch,
                            )
                    run_writer.writerow(
                        {
                            "training_scope": "System_Failure_normal_cycles",
                            "model": "lstm_autoencoder_19_raw",
                            "seed": seed,
                            "test_block": test_block,
                            "validation_block": validation_block,
                            "calibration_blocks": ";".join(str(value) for value in calibration_blocks),
                            "core_train_blocks": 5,
                            "final_train_blocks": 6,
                            "core_train_windows": int(core_train_mask.sum()),
                            "validation_windows": int(validation_mask.sum()),
                            "final_train_windows": int(final_train_mask.sum()),
                            "calibration_windows": int(calibration_mask.sum()),
                            "calibration_normal_cycles": int(np.unique(cycle_runs[calibration_mask]).size),
                            "test_windows": int(test_mask.sum()),
                            "train_positive_count": 0,
                            "best_epoch": best_epoch,
                            "best_validation_loss": best_val_loss,
                            "threshold_q90": thresholds[0.90],
                            "threshold_q95": thresholds[0.95],
                            "threshold_q975": thresholds[0.975],
                            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
                            "device": str(device),
                            "torch_version": torch.__version__,
                            "cuda_version": torch.version.cuda,
                            "elapsed_seconds": elapsed,
                        }
                    )
                    prediction_handle.flush()
                    run_handle.flush()
                    print(
                        f"lstm_autoencoder_19_raw seed={seed} test={test_block} "
                        f"epoch={best_epoch} cal_cycles={len(calibration_cycle_scores)} "
                        f"q95={thresholds[0.95]:.5f} seconds={elapsed:.2f}",
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
