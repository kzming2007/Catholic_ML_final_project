from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from common import BASE_SENSOR_COLS, OUTPUT_DIR, ensure_output_dir


CACHE_DIR = Path(__file__).with_name(".sequence_cache")
MANIFEST_PATH = CACHE_DIR / "10_sequence_manifest.json"
REFERENCE_PATH = OUTPUT_DIR / "12_matched_rf_window_predictions.csv"
DECISION_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="")
    parser.add_argument("--variants", default="")
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


def load_window_module():
    module_path = Path(__file__).with_name("02_window_feature_baseline.py")
    spec = importlib.util.spec_from_file_location("window_feature_baseline_14", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def feature_variants() -> OrderedDict[str, list[int]]:
    joint_current = list(range(0, 6))
    temperature = list(range(6, 12))
    speed = list(range(12, 18))
    tool_current = [18]
    current_family = joint_current + tool_current
    return OrderedDict(
        [
            ("all_19", list(range(19))),
            ("current_family_only", current_family),
            ("joint_current_only", joint_current),
            ("tool_current_only", tool_current),
            ("speed_only", speed),
            ("temperature_only", temperature),
            ("drop_current_family", temperature + speed),
            ("drop_speed", joint_current + temperature + tool_current),
            ("drop_temperature", joint_current + speed + tool_current),
            ("drop_tool_current", joint_current + temperature + speed),
        ]
    )


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
        raise AssertionError("Experiment 14 requires the fixed 10-step cache.")
    if manifest["feature_columns"][: len(BASE_SENSOR_COLS)] != BASE_SENSOR_COLS:
        raise AssertionError("The first 19 cached features do not match BASE_SENSOR_COLS.")
    variants = feature_variants()
    for name, indices in variants.items():
        if not indices or len(indices) != len(set(indices)):
            raise AssertionError(f"Invalid feature indices for {name}.")
        if min(indices) < 0 or max(indices) >= len(BASE_SENSOR_COLS):
            raise AssertionError(f"Out-of-range feature index for {name}.")


def reference_lookup() -> dict[tuple[str, int], pd.DataFrame]:
    reference = pd.read_csv(REFERENCE_PATH, encoding="utf-8-sig")
    keys = ["cycle_run", "window_start_step"]
    lookup = {}
    for (target, test_block), group in reference.groupby(["target", "test_block"], sort=False):
        ordered = group.sort_values(keys).reset_index(drop=True)
        lookup[(str(target), int(test_block))] = ordered
    return lookup


def validate_all_19_fold(
    target: str,
    test_block: int,
    rows: pd.DataFrame,
    reference: dict[tuple[str, int], pd.DataFrame],
) -> None:
    keys = ["cycle_run", "window_start_step"]
    current = rows.sort_values(keys).reset_index(drop=True)
    expected = reference[(target, test_block)]
    for column in [*keys, "label", "prediction"]:
        if not np.array_equal(current[column].to_numpy(), expected[column].to_numpy()):
            raise AssertionError(f"all_19 mismatch for {target}, block {test_block}: {column}")
    if not np.allclose(current["score"], expected["score"], rtol=0.0, atol=1e-12):
        max_difference = float(np.max(np.abs(current["score"] - expected["score"])))
        raise AssertionError(
            f"all_19 score mismatch for {target}, block {test_block}: {max_difference}"
        )


def main() -> None:
    args = parse_args()
    ensure_output_dir()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    window_module = load_window_module()
    blocks = [int(value) for value in manifest["blocks"]]
    variants = feature_variants()
    target_names = [entry["target"] for entry in manifest["targets"]]
    selected_targets = set(selected_values(args.targets, target_names))
    selected_variants = selected_values(args.variants, list(variants))
    test_blocks = blocks[: args.max_folds] if args.max_folds > 0 else blocks
    reference = reference_lookup()

    prediction_path = args.output_dir / "14_sensor_group_ablation_predictions.csv"
    run_path = args.output_dir / "14_sensor_group_ablation_runs.csv"
    prediction_tmp = prediction_path.with_suffix(".csv.tmp")
    run_tmp = run_path.with_suffix(".csv.tmp")
    prediction_fields = [
        "target", "variant", "feature_count", "summary_feature_count", "test_block",
        "cycle", "cycle_run", "cycle_occurrence", "scenario_block_25", "window_start_step",
        "window_end_step", "label", "score", "prediction", "decision_threshold",
    ]
    run_fields = [
        "target", "variant", "feature_columns", "feature_count", "summary_feature_count",
        "test_block", "train_blocks", "train_windows", "test_windows", "train_positive_count",
        "decision_threshold", "baseline_12_match", "elapsed_seconds",
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
                        "cycle", "cycle_run", "cycle_occurrence", "scenario_block_25",
                        "window_start_step", "window_end_step",
                    ]
                }

            for variant in selected_variants:
                feature_indices = variants[variant]
                features = summarize_sequences(raw_x[:, :, feature_indices])
                selected_columns = [BASE_SENSOR_COLS[index] for index in feature_indices]
                for test_block in test_blocks:
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

                    fold_rows = []
                    for local_index, source_index in enumerate(test_indices):
                        row = {
                            "target": target,
                            "variant": variant,
                            "feature_count": len(feature_indices),
                            "summary_feature_count": int(features.shape[1]),
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
                        }
                        prediction_writer.writerow(row)
                        if variant == "all_19":
                            fold_rows.append(row)

                    baseline_match = ""
                    if variant == "all_19":
                        validate_all_19_fold(target, test_block, pd.DataFrame(fold_rows), reference)
                        baseline_match = True

                    run_writer.writerow(
                        {
                            "target": target,
                            "variant": variant,
                            "feature_columns": ";".join(selected_columns),
                            "feature_count": len(feature_indices),
                            "summary_feature_count": int(features.shape[1]),
                            "test_block": test_block,
                            "train_blocks": int(np.unique(window_blocks[train_mask]).size),
                            "train_windows": int(train_mask.sum()),
                            "test_windows": int(test_mask.sum()),
                            "train_positive_count": int(y[train_mask].sum()),
                            "decision_threshold": DECISION_THRESHOLD,
                            "baseline_12_match": baseline_match,
                            "elapsed_seconds": elapsed,
                        }
                    )
                    prediction_handle.flush()
                    run_handle.flush()
                    print(
                        f"{target} {variant} test={test_block} features={len(feature_indices)} "
                        f"seconds={elapsed:.2f}",
                        flush=True,
                    )

    prediction_tmp.replace(prediction_path)
    run_tmp.replace(run_path)
    print(prediction_path)
    print(run_path)


if __name__ == "__main__":
    main()
