from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "dataset" / "ur3_cobotops.csv"
OUTPUT_DIR = REPO_ROOT / "research" / "outputs"


CURRENT_COLS = [f"Current_J{i}" for i in range(6)]
TEMP_COLS = [f"Temperature_J{i}" for i in range(6)]
SPEED_COLS = [f"Speed_J{i}" for i in range(6)]
BASE_SENSOR_COLS = CURRENT_COLS + TEMP_COLS + SPEED_COLS + ["Tool_current"]
POWER_COLS = [f"Power_J{i}" for i in range(6)]
ENGINEERED_COLS = POWER_COLS + ["Abs_Current_Sum"]
MODEL_FEATURE_COLS = BASE_SENSOR_COLS + ENGINEERED_COLS
TARGET_COLS = ["System_Failure", "ProtectiveStop", "GripLost"]


def parse_bool(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    mapped = values.map({"true": True, "false": False})
    return mapped.astype("boolean")


def load_raw_data(path: Path = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Temperature_T0": "Temperature_J0", "cycle ": "cycle"})
    return df


def add_targets_and_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ProtectiveStop"] = parse_bool(out["Robot_ProtectiveStop"])
    out["GripLost"] = parse_bool(out["grip_lost"])
    out["System_Failure"] = (out["ProtectiveStop"].fillna(False) | out["GripLost"].fillna(False)).astype(int)

    # Rows with unknown ProtectiveStop are not used in model training, but keeping
    # System_Failure available helps audit target construction explicitly.
    out.loc[out["ProtectiveStop"].isna(), "System_Failure"] = np.nan

    for i in range(6):
        out[f"Power_J{i}"] = out[f"Current_J{i}"] * out[f"Speed_J{i}"]
    out["Abs_Current_Sum"] = out[CURRENT_COLS].abs().sum(axis=1)

    return out


def add_cycle_run_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cycle_run"] = out["cycle"].ne(out["cycle"].shift()).cumsum().astype(int)
    out["cycle_occurrence"] = out.groupby("cycle")["cycle_run"].transform(
        lambda values: pd.factorize(values, sort=False)[0] + 1
    )
    return out


def load_model_data() -> pd.DataFrame:
    return add_cycle_run_id(add_targets_and_features(load_raw_data()))


def clean_for_model(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    needed = feature_cols + [target_col, "cycle", "cycle_run", "cycle_occurrence", "Timestamp"]
    return df.dropna(subset=needed).copy()


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def write_markdown_table(df: pd.DataFrame, path: Path, title: str) -> None:
    lines = [f"# {title}", "", markdown_table(df), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    display = display.astype(object).where(pd.notna(display), "")
    headers = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell.replace("\n", " ") for cell in row) + " |")
    return "\n".join(lines)
