from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUTPUT_DIR, TARGET_COLS, ensure_output_dir, load_model_data, markdown_table


WINDOW_SIZE = 10
MODEL_NAME = "rf_smote"
DECISION_THRESHOLD = 0.5
MIN_BLOCK_CYCLE_RUNS = 20


def load_window_module():
    module_path = Path(__file__).with_name("02_window_feature_baseline.py")
    spec = importlib.util.spec_from_file_location("window_feature_baseline", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def complete_blocks(df: pd.DataFrame) -> list[int]:
    cycle_runs = df[["cycle_run", "cycle"]].drop_duplicates().copy()
    cycle_runs["scenario_block_25"] = ((cycle_runs["cycle"].astype(int) - 1) // 25 + 1).astype(int)
    counts = cycle_runs.groupby("scenario_block_25")["cycle_run"].nunique()
    return counts[counts >= MIN_BLOCK_CYCLE_RUNS].index.astype(int).tolist()


def feature_sets(feature_cols: list[str]) -> dict[str, list[str]]:
    return {
        "all_sensors": feature_cols,
        "no_temperature": [col for col in feature_cols if not col.startswith("Temperature_J")],
    }


def make_window_predictions(
    window_module,
    window_df: pd.DataFrame,
    feature_cols: list[str],
    feature_set: str,
    test_block: int,
) -> pd.DataFrame | None:
    train_df = window_df[window_df["scenario_block_25"].ne(test_block)].copy()
    test_df = window_df[window_df["scenario_block_25"].eq(test_block)].copy()
    y_train = train_df["label"].astype(int)
    y_test = test_df["label"].astype(int)
    if train_df.empty or test_df.empty or y_train.nunique() < 2 or y_test.nunique() < 2:
        return None
    if int(y_train.value_counts().min()) < 2:
        return None

    model = window_module.build_model(MODEL_NAME, y_train)
    model.fit(train_df[feature_cols], y_train)
    y_score = model.predict_proba(test_df[feature_cols])[:, 1]

    columns = [
        "target",
        "window_size",
        "cycle",
        "cycle_run",
        "cycle_occurrence",
        "scenario_block_25",
        "window_start_step",
        "window_end_step",
        "window_start_timestamp",
        "window_end_timestamp",
        "positive_points_in_window",
        "label",
    ]
    predictions = test_df[columns].copy()
    predictions["feature_set"] = feature_set
    predictions["model"] = MODEL_NAME
    predictions["test_block"] = int(test_block)
    predictions["decision_threshold"] = DECISION_THRESHOLD
    predictions["score"] = y_score
    # Match sklearn's binary predict tie behavior: a score of exactly 0.5 stays negative.
    predictions["prediction"] = (y_score > DECISION_THRESHOLD).astype(int)
    predictions["train_blocks"] = int(train_df["scenario_block_25"].nunique())
    predictions["train_cycle_runs"] = int(train_df["cycle_run"].nunique())
    predictions["train_windows"] = int(len(train_df))
    return predictions


def aggregate_cycle_results(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    keys = ["target", "feature_set", "model", "test_block", "cycle_run"]
    for group_keys, group in predictions.groupby(keys, sort=False):
        target, feature_set, model, test_block, cycle_run = group_keys
        positive_mask = group["label"].eq(1)
        detected_positive_mask = positive_mask & group["prediction"].eq(1)
        negative_alert_mask = group["label"].eq(0) & group["prediction"].eq(1)
        true_event = bool(positive_mask.any())

        first_positive_end = group.loc[positive_mask, "window_end_step"].min() if true_event else np.nan
        first_detected_end = group.loc[detected_positive_mask, "window_end_step"].min() if detected_positive_mask.any() else np.nan
        detection_delay = first_detected_end - first_positive_end if detected_positive_mask.any() else np.nan

        rows.append(
            {
                "target": target,
                "window_size": WINDOW_SIZE,
                "feature_set": feature_set,
                "model": model,
                "test_block": int(test_block),
                "cycle": int(group["cycle"].iloc[0]),
                "cycle_run": int(cycle_run),
                "cycle_occurrence": int(group["cycle_occurrence"].iloc[0]),
                "window_count": int(len(group)),
                "positive_window_count": int(positive_mask.sum()),
                "predicted_positive_window_count": int(group["prediction"].sum()),
                "negative_alert_window_count": int(negative_alert_mask.sum()),
                "true_event_cycle": int(true_event),
                "event_detected_on_positive_window": int(detected_positive_mask.any()) if true_event else np.nan,
                "normal_cycle_false_alarm": int(group["prediction"].any()) if not true_event else np.nan,
                "first_positive_window_end_step": first_positive_end,
                "first_detected_positive_window_end_step": first_detected_end,
                "anomaly_window_detection_delay_steps": detection_delay,
            }
        )
    return pd.DataFrame(rows)


def aggregate_block_results(window_module, predictions: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    keys = ["target", "feature_set", "model", "test_block"]
    for group_keys, window_group in predictions.groupby(keys, sort=False):
        target, feature_set, model, test_block = group_keys
        cycle_group = cycles[
            cycles["target"].eq(target)
            & cycles["feature_set"].eq(feature_set)
            & cycles["model"].eq(model)
            & cycles["test_block"].eq(test_block)
        ]
        y_true = window_group["label"].astype(int)
        y_pred = window_group["prediction"].astype(int)
        y_score = window_group["score"].astype(float)
        window_metrics = window_module.evaluate_binary(y_true, y_pred, y_score)
        negative_windows = int((y_true == 0).sum())

        event_cycles = cycle_group[cycle_group["true_event_cycle"].eq(1)]
        normal_cycles = cycle_group[cycle_group["true_event_cycle"].eq(0)]
        detected_events = int(event_cycles["event_detected_on_positive_window"].sum())
        false_alarm_cycles = int(normal_cycles["normal_cycle_false_alarm"].sum())
        normal_false_positive_windows = int(normal_cycles["predicted_positive_window_count"].sum())
        detected_delays = event_cycles["anomaly_window_detection_delay_steps"].dropna()

        row: dict[str, float | int | str] = {
            "target": target,
            "window_size": WINDOW_SIZE,
            "feature_set": feature_set,
            "model": model,
            "decision_threshold": DECISION_THRESHOLD,
            "test_block": int(test_block),
            "train_blocks": int(window_group["train_blocks"].iloc[0]),
            "train_cycle_runs": int(window_group["train_cycle_runs"].iloc[0]),
            "train_windows": int(window_group["train_windows"].iloc[0]),
            "eligible_test_cycle_runs": int(len(cycle_group)),
            "event_cycle_count": int(len(event_cycles)),
            "detected_event_cycle_count": detected_events,
            "event_cycle_recall": float(detected_events / len(event_cycles)) if len(event_cycles) else np.nan,
            "normal_cycle_count": int(len(normal_cycles)),
            "false_alarm_cycle_count": false_alarm_cycles,
            "normal_cycle_false_alarm_rate": float(false_alarm_cycles / len(normal_cycles)) if len(normal_cycles) else np.nan,
            "false_positive_windows_on_normal_cycles": normal_false_positive_windows,
            "false_positive_windows_per_normal_cycle": (
                float(normal_false_positive_windows / len(normal_cycles)) if len(normal_cycles) else np.nan
            ),
            "normal_window_false_positive_rate": (
                float(normal_false_positive_windows / int(normal_cycles["window_count"].sum()))
                if int(normal_cycles["window_count"].sum())
                else np.nan
            ),
            "detected_event_delay_steps_mean": float(detected_delays.mean()) if not detected_delays.empty else np.nan,
            "detected_event_delay_steps_median": float(detected_delays.median()) if not detected_delays.empty else np.nan,
            "detected_event_delay_steps_max": float(detected_delays.max()) if not detected_delays.empty else np.nan,
            "window_false_positive_rate": float(window_metrics["fp"] / negative_windows) if negative_windows else np.nan,
        }
        row.update({f"window_{key}": value for key, value in window_metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_summary(blocks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    metrics = [
        "event_cycle_recall",
        "normal_cycle_false_alarm_rate",
        "false_positive_windows_per_normal_cycle",
        "normal_window_false_positive_rate",
        "detected_event_delay_steps_mean",
        "window_macro_f1",
        "window_positive_recall",
        "window_positive_f1",
        "window_pr_auc",
    ]
    for keys, group in blocks.groupby(["target", "window_size", "feature_set", "model"], sort=False):
        target, window_size, feature_set, model = keys
        event_total = int(group["event_cycle_count"].sum())
        detected_total = int(group["detected_event_cycle_count"].sum())
        normal_total = int(group["normal_cycle_count"].sum())
        false_alarm_total = int(group["false_alarm_cycle_count"].sum())
        normal_window_total = int(group["normal_cycle_count"].sum())
        false_positive_windows_total = int(group["false_positive_windows_on_normal_cycles"].sum())

        row: dict[str, float | int | str] = {
            "target": target,
            "window_size": int(window_size),
            "feature_set": feature_set,
            "model": model,
            "decision_threshold": DECISION_THRESHOLD,
            "valid_test_blocks": int(len(group)),
            "event_cycle_count_total": event_total,
            "detected_event_cycle_count_total": detected_total,
            "event_cycle_recall_micro": float(detected_total / event_total) if event_total else np.nan,
            "normal_cycle_count_total": normal_total,
            "false_alarm_cycle_count_total": false_alarm_total,
            "normal_cycle_false_alarm_rate_micro": float(false_alarm_total / normal_total) if normal_total else np.nan,
            "false_positive_windows_on_normal_cycles_total": false_positive_windows_total,
            "false_positive_windows_per_normal_cycle_micro": (
                float(false_positive_windows_total / normal_window_total) if normal_window_total else np.nan
            ),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def round_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    return out


def format_report(block_ids: list[int], summary: pd.DataFrame) -> str:
    selected_cols = [
        "target",
        "feature_set",
        "valid_test_blocks",
        "event_cycle_count_total",
        "event_cycle_recall_micro",
        "event_cycle_recall_min",
        "normal_cycle_count_total",
        "normal_cycle_false_alarm_rate_micro",
        "normal_cycle_false_alarm_rate_max",
        "false_positive_windows_per_normal_cycle_micro",
        "window_positive_f1_mean",
        "window_pr_auc_mean",
    ]
    return "\n".join(
        [
            "# 09 Event-Level Block-Held-Out Validation",
            "",
            "## 목적",
            "",
            "딥러닝 비교 전에 공통 10-step과 동일한 후보 block held-out 분할을 고정하고, window 성능 외에 고장 event cycle 탐지율과 정상 cycle 오경보를 평가한다.",
            "",
            "## 고정 설계",
            "",
            f"- Window: {WINDOW_SIZE} step.",
            f"- 평가 block: {block_ids}.",
            f"- 모델: `SMOTE + Random Forest`, positive decision rule `score > {DECISION_THRESHOLD:.2f}`.",
            "- Feature set: 전체 sensor를 주 설정으로 두고 temperature 제거를 ablation으로 병기한다.",
            "- Event detection: 실제 positive가 포함된 window 중 하나 이상을 positive로 예측한 event cycle의 비율.",
            "- Normal-cycle false alarm: 실제 positive window가 없는 정상 cycle에서 하나 이상의 positive 예측이 발생한 비율.",
            "- 제한: 25-cycle block은 실제 공정조건 정답표가 아니라 cycle 번호 기반 proxy다.",
            "- 제한: 이 평가는 이상이 포함된 window 탐지이며 고장 발생 전 예측이 아니다.",
            "",
            "## 요약",
            "",
            markdown_table(round_metrics(summary[selected_cols])),
            "",
            "## 해석 기준",
            "",
            "- `event_cycle_recall_micro`는 전체 event cycle을 합쳐 계산한 탐지율이다.",
            "- `event_cycle_recall_min`은 가장 어려운 held-out block의 탐지율이다.",
            "- `normal_cycle_false_alarm_rate_micro`는 정상 cycle 가운데 경보가 한 번 이상 발생한 cycle 비율이다.",
            "- 겹치는 window 때문에 window false positive가 cycle 오경보로 누적될 수 있으므로 두 지표를 함께 본다.",
            "- 이 고정 결과를 Random Forest 기준선으로 사용하고, 후속 sequence model도 같은 10-step과 block을 사용한다.",
            "",
        ]
    )


def write_outputs(window_module, window_predictions: pd.DataFrame, block_ids: list[int]) -> None:
    cycle_results = aggregate_cycle_results(window_predictions)
    block_results = aggregate_block_results(window_module, window_predictions, cycle_results)
    summary = aggregate_summary(block_results)

    window_predictions.to_csv(OUTPUT_DIR / "09_event_level_window_predictions.csv", index=False, encoding="utf-8-sig")
    cycle_results.to_csv(OUTPUT_DIR / "09_event_level_cycle_results.csv", index=False, encoding="utf-8-sig")
    block_results.to_csv(OUTPUT_DIR / "09_event_level_block_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "09_event_level_block_summary.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "09_event_level_block_validation.md").write_text(
        format_report(block_ids, summary),
        encoding="utf-8",
    )


def main(from_predictions: bool = False) -> None:
    ensure_output_dir()
    window_module = load_window_module()
    if from_predictions:
        window_predictions = pd.read_csv(
            OUTPUT_DIR / "09_event_level_window_predictions.csv",
            encoding="utf-8-sig",
        )
        window_predictions["decision_threshold"] = DECISION_THRESHOLD
        window_predictions["prediction"] = (window_predictions["score"] > DECISION_THRESHOLD).astype(int)
        block_ids = sorted(window_predictions["test_block"].astype(int).unique().tolist())
        write_outputs(window_module, window_predictions, block_ids)
        print("Event-level 결과 재집계 완료")
        print(OUTPUT_DIR / "09_event_level_block_validation.md")
        return

    df = load_model_data()
    block_ids = complete_blocks(df)
    prediction_frames = []

    for target in TARGET_COLS:
        window_df, feature_cols = window_module.build_window_dataset(df, target, WINDOW_SIZE)
        window_df = window_df[window_df["scenario_block_25"].isin(block_ids)].copy()
        for feature_set, cols in feature_sets(feature_cols).items():
            for test_block in block_ids:
                predictions = make_window_predictions(window_module, window_df, cols, feature_set, test_block)
                if predictions is not None:
                    prediction_frames.append(predictions)

    if not prediction_frames:
        raise RuntimeError("No valid block-held-out predictions were produced.")

    window_predictions = pd.concat(prediction_frames, ignore_index=True)
    write_outputs(window_module, window_predictions, block_ids)

    print("Event-level block-held-out validation 완료")
    print(OUTPUT_DIR / "09_event_level_block_validation.md")


if __name__ == "__main__":
    main(from_predictions="--from-predictions" in sys.argv)
