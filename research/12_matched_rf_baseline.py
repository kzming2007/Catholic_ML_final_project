from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import BASE_SENSOR_COLS, OUTPUT_DIR, ensure_output_dir


CACHE_DIR = Path(__file__).with_name(".sequence_cache")
MANIFEST_PATH = CACHE_DIR / "10_sequence_manifest.json"
MODEL_NAME = "rf_19_raw_window"
DECISION_THRESHOLD = 0.5


def load_window_module():
    module_path = Path(__file__).with_name("02_window_feature_baseline.py")
    spec = importlib.util.spec_from_file_location("window_feature_baseline_12", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize_sequences(x: np.ndarray) -> np.ndarray:
    means = x.mean(axis=1)
    stds = x.std(axis=1, ddof=0)
    mins = x.min(axis=1)
    maxs = x.max(axis=1)
    ranges = maxs - mins
    deltas = x[:, -1, :] - x[:, 0, :]
    slopes = deltas / max(x.shape[1] - 1, 1)
    return np.stack([means, stds, mins, maxs, ranges, deltas, slopes], axis=2).reshape(len(x), -1)


def validate_manifest(manifest: dict) -> None:
    if manifest["window_size"] != 10:
        raise AssertionError("Experiment 12 requires the fixed 10-step cache.")
    if manifest["feature_columns"][: len(BASE_SENSOR_COLS)] != BASE_SENSOR_COLS:
        raise AssertionError("The first 19 cached features do not match BASE_SENSOR_COLS.")


def main() -> None:
    ensure_output_dir()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    window_module = load_window_module()
    blocks = [int(value) for value in manifest["blocks"]]

    prediction_path = OUTPUT_DIR / "12_matched_rf_window_predictions.csv"
    run_path = OUTPUT_DIR / "12_matched_rf_runs.csv"
    prediction_tmp = prediction_path.with_suffix(".csv.tmp")
    run_tmp = run_path.with_suffix(".csv.tmp")
    prediction_fields = [
        "target", "model", "seed", "test_block", "cycle", "cycle_run", "cycle_occurrence",
        "scenario_block_25", "window_start_step", "window_end_step", "label", "score",
        "prediction", "decision_threshold", "threshold_quantile",
    ]
    run_fields = [
        "target", "model", "seed", "test_block", "train_blocks", "train_windows", "test_windows",
        "train_positive_count", "feature_count", "summary_feature_count", "decision_threshold",
        "elapsed_seconds",
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
            with np.load(CACHE_DIR / entry["file"]) as arrays:
                x = arrays["X"][:, :, : len(BASE_SENSOR_COLS)].astype(np.float64)
                y = arrays["y"].astype(np.int64)
                window_blocks = arrays["scenario_block_25"].astype(np.int64)
                metadata = {name: arrays[name].astype(np.int64) for name in [
                    "cycle", "cycle_run", "cycle_occurrence", "scenario_block_25",
                    "window_start_step", "window_end_step",
                ]}
            features = summarize_sequences(x)

            for test_block in blocks:
                started = time.perf_counter()
                train_mask = window_blocks != test_block
                test_mask = window_blocks == test_block
                y_train = pd.Series(y[train_mask])
                model = window_module.build_model("rf_smote", y_train)
                model.fit(features[train_mask], y_train)
                scores = model.predict_proba(features[test_mask])[:, 1]
                predictions = (scores > DECISION_THRESHOLD).astype(np.int64)
                elapsed = time.perf_counter() - started

                test_indices = np.flatnonzero(test_mask)
                for local_index, source_index in enumerate(test_indices):
                    prediction_writer.writerow(
                        {
                            "target": target,
                            "model": MODEL_NAME,
                            "seed": 42,
                            "test_block": test_block,
                            "cycle": int(metadata["cycle"][source_index]),
                            "cycle_run": int(metadata["cycle_run"][source_index]),
                            "cycle_occurrence": int(metadata["cycle_occurrence"][source_index]),
                            "scenario_block_25": int(metadata["scenario_block_25"][source_index]),
                            "window_start_step": int(metadata["window_start_step"][source_index]),
                            "window_end_step": int(metadata["window_end_step"][source_index]),
                            "label": int(y[source_index]),
                            "score": float(scores[local_index]),
                            "prediction": int(predictions[local_index]),
                            "decision_threshold": DECISION_THRESHOLD,
                            "threshold_quantile": "",
                        }
                    )
                run_writer.writerow(
                    {
                        "target": target,
                        "model": MODEL_NAME,
                        "seed": 42,
                        "test_block": test_block,
                        "train_blocks": int(np.unique(window_blocks[train_mask]).size),
                        "train_windows": int(train_mask.sum()),
                        "test_windows": int(test_mask.sum()),
                        "train_positive_count": int(y[train_mask].sum()),
                        "feature_count": len(BASE_SENSOR_COLS),
                        "summary_feature_count": int(features.shape[1]),
                        "decision_threshold": DECISION_THRESHOLD,
                        "elapsed_seconds": elapsed,
                    }
                )
                prediction_handle.flush()
                run_handle.flush()
                print(f"{target} {MODEL_NAME} test={test_block} seconds={elapsed:.2f}", flush=True)

    prediction_tmp.replace(prediction_path)
    run_tmp.replace(run_path)
    print(prediction_path)
    print(run_path)


if __name__ == "__main__":
    main()
