from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common import OUTPUT_DIR, ensure_output_dir, markdown_table


RF_PATH = OUTPUT_DIR / "12_matched_rf_window_predictions.csv"
TORCH_PATH = OUTPUT_DIR / "12_matched_torch_window_predictions.csv"
CYCLE_PATH = OUTPUT_DIR / "12_matched_consensus_cycle_results.csv"
MODEL_SUMMARY_PATH = OUTPUT_DIR / "12_matched_model_seed_summary.csv"
CONSENSUS_SUMMARY_PATH = OUTPUT_DIR / "12_matched_consensus_summary.csv"
PRIMARY_VARIANTS = [
    "rf_19_raw_window",
    "1d_cnn_19_raw",
    "lstm_autoencoder_19_raw_q95",
]
TARGETS = ["System_Failure", "ProtectiveStop", "GripLost"]


def variant_name(row: pd.Series) -> str:
    if row["model"] != "lstm_autoencoder_19_raw":
        return str(row["model"])
    quantile = float(row["threshold_quantile"])
    label = "97.5" if math.isclose(quantile, 0.975) else str(int(round(quantile * 100)))
    return f"lstm_autoencoder_19_raw_q{label}"


def load_primary_predictions() -> pd.DataFrame:
    rf = pd.read_csv(RF_PATH, encoding="utf-8-sig")
    torch_predictions = pd.read_csv(TORCH_PATH, encoding="utf-8-sig", low_memory=False)
    predictions = pd.concat([rf, torch_predictions], ignore_index=True, sort=False)
    predictions["threshold_quantile"] = pd.to_numeric(
        predictions["threshold_quantile"], errors="coerce"
    )
    predictions["model_variant"] = predictions.apply(variant_name, axis=1)
    predictions = predictions[predictions["model_variant"].isin(PRIMARY_VARIANTS)].copy()
    predictions["seed"] = predictions["seed"].astype(int)
    predictions["label"] = predictions["label"].astype(int)
    predictions["prediction"] = predictions["prediction"].astype(int)
    return predictions


def binary_metric_row(y_true: pd.Series, prediction: pd.Series, score: pd.Series) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "positive_precision": float(precision_score(y_true, prediction, zero_division=0)),
        "positive_recall": float(recall_score(y_true, prediction, zero_division=0)),
        "positive_f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
    }


def window_seed_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, variant, seed), group in predictions.groupby(
        ["target", "model_variant", "seed"], sort=False
    ):
        if len(group) != 4035 or group["test_block"].nunique() != 9:
            raise AssertionError(f"Incomplete held-out predictions: {target}, {variant}, {seed}")
        row: dict[str, float | int | str] = {
            "target": target,
            "model_variant": variant,
            "seed": int(seed),
            "windows": int(len(group)),
            "negative_windows": int(group["label"].eq(0).sum()),
            "positive_windows": int(group["label"].eq(1).sum()),
        }
        row.update(binary_metric_row(group["label"], group["prediction"], group["score"]))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_window_models(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "tn", "fp", "fn", "tp", "accuracy", "macro_f1", "positive_precision",
        "positive_recall", "positive_f1", "roc_auc", "pr_auc",
    ]
    rows = []
    for (target, variant), group in seed_metrics.groupby(["target", "model_variant"], sort=False):
        row: dict[str, float | int | str] = {
            "target": target,
            "model_variant": variant,
            "seed_count": int(group["seed"].nunique()),
            "windows_per_seed": int(group["windows"].min()),
            "negative_windows_per_seed": int(group["negative_windows"].min()),
            "positive_windows_per_seed": int(group["positive_windows"].min()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def cycle_consensus_metrics() -> pd.DataFrame:
    cycles = pd.read_csv(CYCLE_PATH, encoding="utf-8-sig")
    cycles = cycles[cycles["model_variant"].isin(PRIMARY_VARIANTS)].copy()
    rows = []
    for (target, variant), group in cycles.groupby(["target", "model_variant"], sort=False):
        if len(group) != 202:
            raise AssertionError(f"Expected 202 cycles for {target}, {variant}.")
        y_true = group["true_event_cycle"].astype(int)
        event_aware_prediction = np.where(
            y_true.eq(1),
            group["consensus_event_detected"].astype(int),
            group["consensus_cycle_alert"].astype(int),
        )
        tn, fp, fn, tp = confusion_matrix(y_true, event_aware_prediction, labels=[0, 1]).ravel()
        rows.append(
            {
                "target": target,
                "model_variant": variant,
                "seed_count": int(group["seed_count"].max()),
                "cycles": int(len(group)),
                "normal_cycles": int(y_true.eq(0).sum()),
                "event_cycles": int(y_true.eq(1).sum()),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "accuracy": float(accuracy_score(y_true, event_aware_prediction)),
                "macro_f1": float(
                    f1_score(y_true, event_aware_prediction, average="macro", zero_division=0)
                ),
                "event_precision": float(
                    precision_score(y_true, event_aware_prediction, zero_division=0)
                ),
                "event_recall": float(recall_score(y_true, event_aware_prediction, zero_division=0)),
                "event_f1": float(f1_score(y_true, event_aware_prediction, zero_division=0)),
                "normal_cycle_false_alarm_rate": float(fp / (tn + fp)),
            }
        )
    return pd.DataFrame(rows)


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["target"] = pd.Categorical(result["target"], TARGETS, ordered=True)
    result["model_variant"] = pd.Categorical(
        result["model_variant"], PRIMARY_VARIANTS, ordered=True
    )
    return result.sort_values(["target", "model_variant"]).reset_index(drop=True)


def round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_float_dtype(result[column]):
            result[column] = result[column].round(4)
    return result


def report(window_summary: pd.DataFrame, cycles: pd.DataFrame) -> str:
    window_columns = [
        "target", "model_variant", "seed_count", "tn_mean", "fp_mean", "fn_mean", "tp_mean",
        "accuracy_mean", "macro_f1_mean", "positive_precision_mean", "positive_recall_mean",
        "positive_f1_mean", "roc_auc_mean", "pr_auc_mean",
    ]
    cycle_columns = [
        "target", "model_variant", "seed_count", "tn", "fp", "fn", "tp", "accuracy",
        "macro_f1", "event_precision", "event_recall", "event_f1",
        "normal_cycle_false_alarm_rate",
    ]
    return "\n".join(
        [
            "# 13 최종 평가 지표표",
            "",
            "## 목적",
            "",
            "연구계획서에 명시한 confusion matrix, Accuracy, Macro F1, Precision, Recall, "
            "ROC-AUC, PR-AUC를 기존 `12` prediction에서 재집계한다. 새 모델 학습이나 "
            "threshold 선택은 수행하지 않는다.",
            "",
            "## Window-level seed 평균",
            "",
            markdown_table(round_frame(window_summary[window_columns])),
            "",
            "- Random Forest는 고정 seed 1회이고 딥러닝 모델은 3개 seed 평균이다.",
            "- `tn/fp/fn/tp_mean`은 딥러닝에서 seed별 confusion count의 평균이므로 정수형 "
            "단일 confusion matrix가 아니다. Seed별 원값은 `13_window_seed_metrics.csv`에 있다.",
            "- ROC-AUC와 PR-AUC는 저장된 window score로 계산했다.",
            "",
            "## Event-aware cycle consensus confusion matrix",
            "",
            markdown_table(round_frame(cycles[cycle_columns])),
            "",
            "- Event cycle은 실제 positive window를 하나 이상 탐지해야 TP로 계산한다.",
            "- Normal cycle은 어느 window에서든 경보가 발생하면 FP로 계산한다.",
            "- 딥러닝은 3개 seed 중 2개 이상인 consensus, Random Forest는 고정 1회다.",
            "- Cycle score aggregation을 사전 고정하지 않았으므로 cycle ROC-AUC와 PR-AUC는 "
            "사후 생성하지 않는다.",
            "",
            "## 해석 범위",
            "",
            "- Window-level과 cycle-level confusion matrix는 평가 단위가 다르므로 직접 합치지 않는다.",
            "- 이 표는 기존 결과의 보고 지표를 보완하며 새로운 독립 실험 결과가 아니다.",
            "- Primary 결론은 `12_matched_lstm_autoencoder_comparison.md`의 event recall과 "
            "정상 cycle 오경보율을 유지한다.",
            "",
        ]
    )


def validate(window_summary: pd.DataFrame, cycles: pd.DataFrame) -> None:
    if len(window_summary) != 9 or len(cycles) != 9:
        raise AssertionError("Expected three models for each of three targets.")
    for frame in [window_summary, cycles]:
        if not frame.groupby("target")["model_variant"].nunique().eq(3).all():
            raise AssertionError("Each target must contain all primary model variants.")
    if not np.allclose(
        cycles["normal_cycle_false_alarm_rate"], cycles["fp"] / cycles["normal_cycles"]
    ):
        raise AssertionError("Cycle false-alarm reconstruction failed.")

    expected_windows = pd.read_csv(MODEL_SUMMARY_PATH, encoding="utf-8-sig")
    expected_windows = expected_windows[
        expected_windows["model_variant"].isin(PRIMARY_VARIANTS)
    ]
    window_check = window_summary.merge(
        expected_windows,
        on=["target", "model_variant"],
        suffixes=("_13", "_12"),
        validate="one_to_one",
    )
    for metric in ["macro_f1", "positive_precision", "positive_recall", "positive_f1", "pr_auc"]:
        if not np.allclose(
            window_check[f"{metric}_mean"], window_check[f"window_{metric}_mean"]
        ):
            raise AssertionError(f"Window metric does not match experiment 12: {metric}")

    expected_cycles = pd.read_csv(CONSENSUS_SUMMARY_PATH, encoding="utf-8-sig")
    expected_cycles = expected_cycles[expected_cycles["model_variant"].isin(PRIMARY_VARIANTS)]
    cycle_check = cycles.merge(
        expected_cycles,
        on=["target", "model_variant"],
        suffixes=("_13", "_12"),
        validate="one_to_one",
    )
    if not np.allclose(cycle_check["event_recall"], cycle_check["event_cycle_recall"]):
        raise AssertionError("Cycle event recall does not match experiment 12.")
    if not np.allclose(
        cycle_check["normal_cycle_false_alarm_rate_13"],
        cycle_check["normal_cycle_false_alarm_rate_12"],
    ):
        raise AssertionError("Cycle false-alarm rate does not match experiment 12.")


def main() -> None:
    ensure_output_dir()
    predictions = load_primary_predictions()
    seed_metrics = ordered(window_seed_metrics(predictions))
    window_summary = ordered(summarize_window_models(seed_metrics))
    cycles = ordered(cycle_consensus_metrics())
    validate(window_summary, cycles)

    seed_metrics.to_csv(
        OUTPUT_DIR / "13_window_seed_metrics.csv", index=False, encoding="utf-8-sig"
    )
    window_summary.to_csv(
        OUTPUT_DIR / "13_window_model_summary.csv", index=False, encoding="utf-8-sig"
    )
    cycles.to_csv(
        OUTPUT_DIR / "13_cycle_consensus_confusion_metrics.csv", index=False, encoding="utf-8-sig"
    )
    report_path = OUTPUT_DIR / "13_final_evaluation_tables.md"
    report_path.write_text(report(window_summary, cycles), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
