from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATASET_PATH, MODEL_FEATURE_COLS, TARGET_COLS, clean_for_model, load_model_data


WINDOW_SIZE = 10
MIN_BLOCK_CYCLE_RUNS = 20
CACHE_DIR = Path(__file__).with_name(".sequence_cache")
MANIFEST_PATH = CACHE_DIR / "10_sequence_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def complete_blocks(df: pd.DataFrame) -> list[int]:
    cycle_runs = df[["cycle_run", "cycle"]].drop_duplicates().copy()
    cycle_runs["scenario_block_25"] = ((cycle_runs["cycle"].astype(int) - 1) // 25 + 1).astype(int)
    counts = cycle_runs.groupby("scenario_block_25")["cycle_run"].nunique()
    return counts[counts >= MIN_BLOCK_CYCLE_RUNS].index.astype(int).tolist()


def build_sequence_dataset(df: pd.DataFrame, target_col: str, blocks: list[int]) -> dict[str, np.ndarray]:
    clean = clean_for_model(df, MODEL_FEATURE_COLS, target_col).copy()
    clean["Timestamp"] = pd.to_datetime(clean["Timestamp"], errors="coerce")
    clean = clean.dropna(subset=["Timestamp"]).sort_values(["cycle_run", "Timestamp"]).reset_index(drop=True)

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    metadata: dict[str, list[int]] = {
        "cycle": [],
        "cycle_run": [],
        "cycle_occurrence": [],
        "scenario_block_25": [],
        "window_start_step": [],
        "window_end_step": [],
    }
    allowed_blocks = set(blocks)

    for cycle_run, group in clean.groupby("cycle_run", sort=True):
        group = group.reset_index(drop=True)
        if len(group) < WINDOW_SIZE:
            continue

        cycle = int(group["cycle"].iloc[0])
        block = int((cycle - 1) // 25 + 1)
        if block not in allowed_blocks:
            continue

        values = group[MODEL_FEATURE_COLS].to_numpy(dtype=np.float32)
        target_values = group[target_col].astype(int).to_numpy()
        for start in range(0, len(group) - WINDOW_SIZE + 1):
            end = start + WINDOW_SIZE
            sequences.append(values[start:end])
            labels.append(int(target_values[start:end].max()))
            metadata["cycle"].append(cycle)
            metadata["cycle_run"].append(int(cycle_run))
            metadata["cycle_occurrence"].append(int(group["cycle_occurrence"].iloc[0]))
            metadata["scenario_block_25"].append(block)
            metadata["window_start_step"].append(start)
            metadata["window_end_step"].append(end - 1)

    if not sequences:
        raise RuntimeError(f"No sequence windows were produced for {target_col}.")

    arrays: dict[str, np.ndarray] = {
        "X": np.stack(sequences).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
    }
    arrays.update({key: np.asarray(values, dtype=np.int64) for key, values in metadata.items()})
    return arrays


def validate_against_event_baseline(target: str, arrays: dict[str, np.ndarray]) -> bool | None:
    prediction_path = Path(__file__).with_name("outputs") / "09_event_level_window_predictions.csv"
    if not prediction_path.exists():
        return None

    baseline = pd.read_csv(prediction_path, encoding="utf-8-sig")
    baseline = baseline[
        baseline["target"].eq(target) & baseline["feature_set"].eq("all_sensors")
    ][["cycle_run", "scenario_block_25", "window_start_step", "label"]].copy()
    prepared = pd.DataFrame(
        {
            "cycle_run": arrays["cycle_run"],
            "scenario_block_25": arrays["scenario_block_25"],
            "window_start_step": arrays["window_start_step"],
            "label": arrays["y"],
        }
    )
    keys = ["cycle_run", "scenario_block_25", "window_start_step", "label"]
    baseline = baseline.sort_values(keys).reset_index(drop=True)
    prepared = prepared.sort_values(keys).reset_index(drop=True)
    return baseline.equals(prepared)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_model_data()
    blocks = complete_blocks(df)
    target_entries = []

    for target in TARGET_COLS:
        arrays = build_sequence_dataset(df, target, blocks)
        baseline_match = validate_against_event_baseline(target, arrays)
        if baseline_match is False:
            raise AssertionError(f"Prepared sequence windows do not match experiment 09 for {target}.")

        filename = f"10_{target}.npz"
        np.savez_compressed(CACHE_DIR / filename, **arrays)
        target_entries.append(
            {
                "target": target,
                "file": filename,
                "windows": int(len(arrays["y"])),
                "positive_windows": int(arrays["y"].sum()),
                "cycle_runs": int(np.unique(arrays["cycle_run"]).size),
                "baseline_09_match": baseline_match,
            }
        )

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(DATASET_PATH),
        "dataset_sha256": sha256(DATASET_PATH),
        "window_size": WINDOW_SIZE,
        "feature_columns": MODEL_FEATURE_COLS,
        "feature_count": len(MODEL_FEATURE_COLS),
        "blocks": blocks,
        "targets": target_entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
