from __future__ import annotations

import argparse
import csv
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import BASE_SENSOR_COLS, OUTPUT_DIR, ensure_output_dir


CACHE_DIR = Path(__file__).with_name(".sequence_cache")
MANIFEST_PATH = CACHE_DIR / "10_sequence_manifest.json"
REFERENCE_PATH = OUTPUT_DIR / "12_matched_rf_window_predictions.csv"
DECISION_THRESHOLD = 0.5
RANDOM_STATE = 42
MODEL_NAMES = ["logistic_regression", "rbf_svm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def selected_values(raw: str, defaults: list[str]) -> list[str]:
    if not raw:
        return defaults
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(values) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown selections: {unknown}")
    return values


def summarize_sequences(x: np.ndarray) -> np.ndarray:
    means = x.mean(axis=1)
    stds = x.std(axis=1, ddof=0)
    mins = x.min(axis=1)
    maxs = x.max(axis=1)
    ranges = maxs - mins
    deltas = x[:, -1, :] - x[:, 0, :]
    slopes = deltas / max(x.shape[1] - 1, 1)
    return np.stack([means, stds, mins, maxs, ranges, deltas, slopes], axis=2).reshape(
        len(x), -1
    )


def validate_manifest(manifest: dict) -> None:
    if manifest["window_size"] != 10:
        raise AssertionError("Experiment 15 requires the fixed 10-step cache.")
    if manifest["feature_columns"][: len(BASE_SENSOR_COLS)] != BASE_SENSOR_COLS:
        raise AssertionError("The first 19 cached features do not match BASE_SENSOR_COLS.")


def build_model(model_name: str, y_train: np.ndarray) -> ImbPipeline:
    minority_count = int(pd.Series(y_train).value_counts().min())
    if minority_count < 2:
        raise ValueError("SMOTE requires at least two minority samples.")
    k_neighbors = min(5, minority_count - 1)
    common_steps = [
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)),
    ]
    estimators = OrderedDict(
        [
            (
                "logistic_regression",
                LogisticRegression(
                    C=1.0,
                    l1_ratio=0.0,
                    solver="saga",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "rbf_svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    if model_name not in estimators:
        raise ValueError(f"Unknown model: {model_name}")
    return ImbPipeline([*common_steps, ("classifier", estimators[model_name])])


def reference_lookup() -> dict[tuple[str, int], pd.DataFrame]:
    reference = pd.read_csv(REFERENCE_PATH, encoding="utf-8-sig")
    keys = ["cycle_run", "window_start_step"]
    lookup = {}
    for (target, test_block), group in reference.groupby(["target", "test_block"], sort=False):
        lookup[(str(target), int(test_block))] = group.sort_values(keys).reset_index(drop=True)
    return lookup


def validate_fold(
    target: str,
    test_block: int,
    rows: pd.DataFrame,
    reference: dict[tuple[str, int], pd.DataFrame],
) -> None:
    keys = ["cycle_run", "window_start_step"]
    current = rows.sort_values(keys).reset_index(drop=True)
    expected = reference[(target, test_block)]
    for column in [
        *keys,
        "cycle",
        "cycle_occurrence",
        "scenario_block_25",
        "window_end_step",
        "label",
    ]:
        if not np.array_equal(current[column].to_numpy(), expected[column].to_numpy()):
            raise AssertionError(f"Reference mismatch for {target}, block {test_block}: {column}")


def estimator_diagnostics(model: ImbPipeline) -> tuple[int, int]:
    estimator = model.named_steps["classifier"]
    if isinstance(estimator, LogisticRegression):
        return 0, int(np.max(estimator.n_iter_))
    return int(estimator.fit_status_), int(np.max(estimator.n_iter_))


def main() -> None:
    args = parse_args()
    ensure_output_dir()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    blocks = [int(value) for value in manifest["blocks"]]
    target_names = [entry["target"] for entry in manifest["targets"]]
    selected_targets = set(selected_values(args.targets, target_names))
    selected_models = selected_values(args.models, MODEL_NAMES)
    test_blocks = blocks[: args.max_folds] if args.max_folds > 0 else blocks
    reference = reference_lookup()

    prediction_path = args.output_dir / "15_classical_model_predictions.csv"
    run_path = args.output_dir / "15_classical_model_runs.csv"
    prediction_tmp = prediction_path.with_suffix(".csv.tmp")
    run_tmp = run_path.with_suffix(".csv.tmp")
    prediction_fields = [
        "target",
        "model",
        "seed",
        "test_block",
        "cycle",
        "cycle_run",
        "cycle_occurrence",
        "scenario_block_25",
        "window_start_step",
        "window_end_step",
        "label",
        "score",
        "prediction",
        "decision_threshold",
    ]
    run_fields = [
        "target",
        "model",
        "seed",
        "test_block",
        "train_blocks",
        "train_windows",
        "test_windows",
        "train_positive_count",
        "feature_count",
        "summary_feature_count",
        "decision_threshold",
        "reference_match",
        "fit_status",
        "n_iter",
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
            if target not in selected_targets:
                continue
            with np.load(CACHE_DIR / entry["file"]) as arrays:
                raw_x = arrays["X"][:, :, : len(BASE_SENSOR_COLS)].astype(np.float64)
                y = arrays["y"].astype(np.int64)
                window_blocks = arrays["scenario_block_25"].astype(np.int64)
                metadata = {
                    name: arrays[name].astype(np.int64)
                    for name in [
                        "cycle",
                        "cycle_run",
                        "cycle_occurrence",
                        "scenario_block_25",
                        "window_start_step",
                        "window_end_step",
                    ]
                }
            features = summarize_sequences(raw_x)

            for model_name in selected_models:
                for test_block in test_blocks:
                    started = time.perf_counter()
                    train_mask = window_blocks != test_block
                    test_mask = window_blocks == test_block
                    model = build_model(model_name, y[train_mask])
                    model.fit(features[train_mask], y[train_mask])
                    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
                    scores = model.predict_proba(features[test_mask])[:, positive_index]
                    if not np.isfinite(scores).all():
                        raise AssertionError(f"Non-finite score: {target}, {model_name}, {test_block}")
                    predictions = (scores > DECISION_THRESHOLD).astype(np.int64)
                    fit_status, n_iter = estimator_diagnostics(model)
                    if fit_status != 0:
                        raise RuntimeError(
                            f"Model fit failed: {target}, {model_name}, block {test_block}, "
                            f"status {fit_status}"
                        )
                    elapsed = time.perf_counter() - started
                    test_indices = np.flatnonzero(test_mask)

                    fold_rows = []
                    for local_index, source_index in enumerate(test_indices):
                        row = {
                            "target": target,
                            "model": model_name,
                            "seed": RANDOM_STATE,
                            "test_block": test_block,
                            "cycle": int(metadata["cycle"][source_index]),
                            "cycle_run": int(metadata["cycle_run"][source_index]),
                            "cycle_occurrence": int(metadata["cycle_occurrence"][source_index]),
                            "scenario_block_25": int(
                                metadata["scenario_block_25"][source_index]
                            ),
                            "window_start_step": int(metadata["window_start_step"][source_index]),
                            "window_end_step": int(metadata["window_end_step"][source_index]),
                            "label": int(y[source_index]),
                            "score": float(scores[local_index]),
                            "prediction": int(predictions[local_index]),
                            "decision_threshold": DECISION_THRESHOLD,
                        }
                        prediction_writer.writerow(row)
                        fold_rows.append(row)
                    validate_fold(target, test_block, pd.DataFrame(fold_rows), reference)

                    run_writer.writerow(
                        {
                            "target": target,
                            "model": model_name,
                            "seed": RANDOM_STATE,
                            "test_block": test_block,
                            "train_blocks": int(np.unique(window_blocks[train_mask]).size),
                            "train_windows": int(train_mask.sum()),
                            "test_windows": int(test_mask.sum()),
                            "train_positive_count": int(y[train_mask].sum()),
                            "feature_count": len(BASE_SENSOR_COLS),
                            "summary_feature_count": int(features.shape[1]),
                            "decision_threshold": DECISION_THRESHOLD,
                            "reference_match": True,
                            "fit_status": fit_status,
                            "n_iter": n_iter,
                            "elapsed_seconds": elapsed,
                        }
                    )
                    prediction_handle.flush()
                    run_handle.flush()
                    print(
                        f"{target} {model_name} test={test_block} n_iter={n_iter} "
                        f"seconds={elapsed:.2f}",
                        flush=True,
                    )

    prediction_tmp.replace(prediction_path)
    run_tmp.replace(run_path)
    print(prediction_path)
    print(run_path)


if __name__ == "__main__":
    main()
