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


PREDICTION_PATH = OUTPUT_DIR / "15_classical_model_predictions.csv"
RUN_PATH = OUTPUT_DIR / "15_classical_model_runs.csv"
RF_PATH = OUTPUT_DIR / "12_matched_rf_window_predictions.csv"
MODEL_ORDER = ["random_forest", "logistic_regression", "rbf_svm"]
TARGET_ORDER = ["System_Failure", "ProtectiveStop", "GripLost"]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return center - margin, center + margin


def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(PREDICTION_PATH, encoding="utf-8-sig")
    rf = pd.read_csv(RF_PATH, encoding="utf-8-sig")
    rf = rf[
        [
            "target",
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
    ].copy()
    rf["model"] = "random_forest"
    rf["seed"] = 42
    predictions = pd.concat([rf, candidates], ignore_index=True, sort=False)
    for column in ["seed", "test_block", "cycle", "cycle_run", "cycle_occurrence"]:
        predictions[column] = predictions[column].astype(int)
    predictions["label"] = predictions["label"].astype(int)
    predictions["prediction"] = predictions["prediction"].astype(int)
    return predictions, candidates


def window_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), group in predictions.groupby(["target", "model"], sort=False):
        tn, fp, fn, tp = confusion_matrix(
            group["label"], group["prediction"], labels=[0, 1]
        ).ravel()
        rows.append(
            {
                "target": target,
                "model": model,
                "windows": int(len(group)),
                "window_tn": int(tn),
                "window_fp": int(fp),
                "window_fn": int(fn),
                "window_tp": int(tp),
                "window_accuracy": float(accuracy_score(group["label"], group["prediction"])),
                "window_macro_f1": float(
                    f1_score(group["label"], group["prediction"], average="macro", zero_division=0)
                ),
                "window_positive_precision": float(
                    precision_score(group["label"], group["prediction"], zero_division=0)
                ),
                "window_positive_recall": float(
                    recall_score(group["label"], group["prediction"], zero_division=0)
                ),
                "window_positive_f1": float(
                    f1_score(group["label"], group["prediction"], zero_division=0)
                ),
                "window_roc_auc": float(roc_auc_score(group["label"], group["score"])),
                "window_pr_auc": float(
                    average_precision_score(group["label"], group["score"])
                ),
            }
        )
    return pd.DataFrame(rows)


def cycle_results(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["target", "model", "test_block", "cycle", "cycle_run", "cycle_occurrence"]
    for group_keys, group in predictions.groupby(keys, sort=False):
        target, model, test_block, cycle, cycle_run, occurrence = group_keys
        positive = group["label"].eq(1)
        rows.append(
            {
                "target": target,
                "model": model,
                "test_block": int(test_block),
                "cycle": int(cycle),
                "cycle_run": int(cycle_run),
                "cycle_occurrence": int(occurrence),
                "true_event_cycle": int(positive.any()),
                "cycle_alert": int(group["prediction"].any()),
                "event_detected": int((positive & group["prediction"].eq(1)).any()),
            }
        )
    return pd.DataFrame(rows)


def block_metrics(cycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model, block), group in cycles.groupby(
        ["target", "model", "test_block"], sort=False
    ):
        events = group[group["true_event_cycle"].eq(1)]
        normal = group[group["true_event_cycle"].eq(0)]
        detected = int(events["event_detected"].sum())
        false_alarms = int(normal["cycle_alert"].sum())
        rows.append(
            {
                "target": target,
                "model": model,
                "test_block": int(block),
                "event_cycles": int(len(events)),
                "detected_event_cycles": detected,
                "event_cycle_recall": float(detected / len(events)) if len(events) else np.nan,
                "normal_cycles": int(len(normal)),
                "false_alarm_cycles": false_alarms,
                "normal_cycle_false_alarm_rate": float(false_alarms / len(normal))
                if len(normal)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def cycle_summary(cycles: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), group in cycles.groupby(["target", "model"], sort=False):
        events = group[group["true_event_cycle"].eq(1)]
        normal = group[group["true_event_cycle"].eq(0)]
        detected = int(events["event_detected"].sum())
        false_alarms = int(normal["cycle_alert"].sum())
        recall_low, recall_high = wilson_interval(detected, len(events))
        far_low, far_high = wilson_interval(false_alarms, len(normal))
        block_group = blocks[blocks["target"].eq(target) & blocks["model"].eq(model)]
        rows.append(
            {
                "target": target,
                "model": model,
                "event_cycles": int(len(events)),
                "detected_event_cycles": detected,
                "event_cycle_recall": float(detected / len(events)),
                "event_cycle_recall_ci95_low": recall_low,
                "event_cycle_recall_ci95_high": recall_high,
                "event_cycle_recall_min_block": float(block_group["event_cycle_recall"].min()),
                "normal_cycles": int(len(normal)),
                "false_alarm_cycles": false_alarms,
                "normal_cycle_false_alarm_rate": float(false_alarms / len(normal)),
                "normal_cycle_false_alarm_rate_ci95_low": far_low,
                "normal_cycle_false_alarm_rate_ci95_high": far_high,
                "normal_cycle_false_alarm_rate_max_block": float(
                    block_group["normal_cycle_false_alarm_rate"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_errors(cycles: pd.DataFrame) -> pd.DataFrame:
    index = ["target", "test_block", "cycle", "cycle_run", "cycle_occurrence", "true_event_cycle"]
    alert = cycles.pivot(index=index, columns="model", values="cycle_alert").reset_index()
    detected = cycles.pivot(index=index, columns="model", values="event_detected").reset_index()
    rows = []
    for target in TARGET_ORDER:
        for model in MODEL_ORDER[1:]:
            event_group = detected[
                detected["target"].eq(target) & detected["true_event_cycle"].eq(1)
            ]
            reference_event_error = event_group["random_forest"].eq(0)
            model_event_error = event_group[model].eq(0)
            normal_group = alert[
                alert["target"].eq(target) & alert["true_event_cycle"].eq(0)
            ]
            reference_normal_error = normal_group["random_forest"].eq(1)
            model_normal_error = normal_group[model].eq(1)
            for error_type, reference_error, candidate_error, eligible in [
                ("event_miss", reference_event_error, model_event_error, event_group),
                ("false_alarm", reference_normal_error, model_normal_error, normal_group),
            ]:
                rows.append(
                    {
                        "target": target,
                        "model": model,
                        "error_type": error_type,
                        "eligible_cycles": int(len(eligible)),
                        "random_forest_error_count": int(reference_error.sum()),
                        "model_error_count": int(candidate_error.sum()),
                        "shared_error_count": int((reference_error & candidate_error).sum()),
                        "new_model_error_count": int((~reference_error & candidate_error).sum()),
                        "corrected_rf_error_count": int((reference_error & ~candidate_error).sum()),
                    }
                )
    return pd.DataFrame(rows)


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["target"] = pd.Categorical(result["target"], TARGET_ORDER, ordered=True)
    result["model"] = pd.Categorical(result["model"], MODEL_ORDER, ordered=True)
    sort_columns = [column for column in ["target", "model", "test_block"] if column in result]
    return result.sort_values(sort_columns).reset_index(drop=True)


def round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_float_dtype(result[column]):
            result[column] = result[column].round(4)
    return result


def comparison_statement(summary: pd.DataFrame, target: str, model: str) -> str:
    indexed = summary[summary["target"].eq(target)].set_index("model")
    reference = indexed.loc["random_forest"]
    candidate = indexed.loc[model]
    recall_delta = candidate["event_cycle_recall"] - reference["event_cycle_recall"]
    far_delta = (
        candidate["normal_cycle_false_alarm_rate"]
        - reference["normal_cycle_false_alarm_rate"]
    )
    reference_better = recall_delta <= 0 and far_delta >= 0 and (recall_delta < 0 or far_delta > 0)
    candidate_better = recall_delta >= 0 and far_delta <= 0 and (recall_delta > 0 or far_delta < 0)
    if reference_better:
        verdict = "Random Forest가 두 cycle 지표에서 기술적으로 우세했다"
    elif candidate_better:
        verdict = f"{model}이 두 cycle 지표에서 기술적으로 우세했다"
    elif recall_delta == 0 and far_delta == 0:
        verdict = "두 cycle 지표가 같았다"
    else:
        verdict = "recall과 오경보 방향이 엇갈리는 trade-off였다"
    return (
        f"`{model}`: event recall {candidate['event_cycle_recall']:.4f} "
        f"({recall_delta:+.4f}), 정상 cycle 오경보율 "
        f"{candidate['normal_cycle_false_alarm_rate']:.4f} ({far_delta:+.4f}); {verdict}."
    )


def validate(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    runs: pd.DataFrame,
    windows: pd.DataFrame,
    cycles: pd.DataFrame,
    blocks: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    if len(candidates) != 24210 or len(runs) != 54 or len(predictions) != 36315:
        raise AssertionError("Unexpected prediction or run count.")
    keys = ["target", "model", "test_block", "cycle_run", "window_start_step"]
    if predictions.duplicated(keys).any():
        raise AssertionError("Duplicate prediction keys.")
    if not predictions.groupby(["target", "model"]).size().eq(4035).all():
        raise AssertionError("Each target/model must have 4,035 predictions.")
    if not predictions.groupby(["target", "model"])["test_block"].nunique().eq(9).all():
        raise AssertionError("Each target/model must have nine test blocks.")
    if not np.isfinite(predictions["score"]).all():
        raise AssertionError("Non-finite score found.")
    expected = predictions["score"].to_numpy() > predictions["decision_threshold"].to_numpy()
    if not np.array_equal(predictions["prediction"].to_numpy(), expected.astype(int)):
        raise AssertionError("Decision threshold reconstruction failed.")
    reference_keys = [
        "target",
        "test_block",
        "cycle_run",
        "window_start_step",
        "label",
    ]
    reference = (
        predictions[predictions["model"].eq("random_forest")][reference_keys]
        .sort_values(reference_keys[:-1])
        .reset_index(drop=True)
    )
    for model in MODEL_ORDER[1:]:
        current = (
            predictions[predictions["model"].eq(model)][reference_keys]
            .sort_values(reference_keys[:-1])
            .reset_index(drop=True)
        )
        if not current.equals(reference):
            raise AssertionError(f"Metadata or label mismatch for {model}.")
    if not runs["reference_match"].astype(str).str.lower().eq("true").all():
        raise AssertionError("At least one candidate fold did not match the RF reference.")
    if not runs["fit_status"].eq(0).all():
        raise AssertionError("At least one estimator reported a failed fit.")
    lr_iterations = runs[runs["model"].eq("logistic_regression")]["n_iter"]
    if lr_iterations.ge(5000).any():
        raise AssertionError("Logistic Regression reached max_iter.")
    if len(windows) != 9 or len(summary) != 9 or len(blocks) != 81:
        raise AssertionError("Unexpected summary dimensions.")
    if not cycles.groupby(["target", "model"]).size().eq(202).all():
        raise AssertionError("Each target/model must contain 202 cycles.")
    if len(paired) != 12:
        raise AssertionError("Expected two models, three targets and two paired error types.")
    confusion_total = windows[["window_tn", "window_fp", "window_fn", "window_tp"]].sum(axis=1)
    if not confusion_total.eq(4035).all():
        raise AssertionError("Window confusion counts do not sum to 4,035.")


def format_report(summary: pd.DataFrame, paired: pd.DataFrame, runs: pd.DataFrame) -> str:
    display_columns = [
        "target",
        "model",
        "event_cycles",
        "detected_event_cycles",
        "event_cycle_recall",
        "event_cycle_recall_ci95_low",
        "event_cycle_recall_ci95_high",
        "event_cycle_recall_min_block",
        "normal_cycles",
        "false_alarm_cycles",
        "normal_cycle_false_alarm_rate",
        "normal_cycle_false_alarm_rate_ci95_low",
        "normal_cycle_false_alarm_rate_ci95_high",
        "normal_cycle_false_alarm_rate_max_block",
        "window_macro_f1",
        "window_positive_f1",
        "window_roc_auc",
        "window_pr_auc",
    ]
    findings = []
    for target in TARGET_ORDER:
        findings.append(f"- `{target}` {comparison_statement(summary, target, 'logistic_regression')}")
        findings.append(f"- `{target}` {comparison_statement(summary, target, 'rbf_svm')}")
    paired_columns = [
        "target",
        "model",
        "error_type",
        "eligible_cycles",
        "random_forest_error_count",
        "model_error_count",
        "shared_error_count",
        "new_model_error_count",
        "corrected_rf_error_count",
    ]
    return "\n".join(
        [
            "# 15 동일 시계열 특징 기반 기본 분류 모델 비교",
            "",
            "## 고정 설계",
            "",
            "- 사전 고정: `research/2026-08-23_classical_model_comparison_preregistration.md`.",
            "- 입력: cycle_run 경계를 넘지 않는 10-step×19개 원본 센서의 133개 통계 특징.",
            "- 모델: Logistic Regression L2 C=1, RBF SVM C=1·gamma=scale, 기존 Random Forest 300 trees.",
            "- 전처리: 기본 모델은 학습 fold 내부 StandardScaler와 SMOTE, Random Forest는 기존 SMOTE 결과 재사용.",
            "- 평가: 9개 acquisition block을 한 번씩 test로 사용하고 score > 0.50 적용.",
            f"- 새 기본 모델 학습: {len(runs)}회, 기록된 실행 시간 합 {runs['elapsed_seconds'].sum():.1f}초.",
            "",
            "## Primary cycle 결과와 secondary window 결과",
            "",
            markdown_table(round_frame(summary[display_columns])),
            "",
            "## Random Forest 대비 해석",
            "",
            *findings,
            "",
            "## Paired cycle 오류",
            "",
            markdown_table(round_frame(paired[paired_columns])),
            "",
            "- Event miss는 실제 positive window를 하나도 잡지 못한 event cycle이다.",
            "- False alarm은 정상 cycle의 window 중 하나 이상에서 경보한 경우다.",
            "- `new_model_error_count`는 Random Forest가 맞혔지만 비교 모델이 틀린 cycle이다.",
            "- `corrected_rf_error_count`는 Random Forest 오류를 비교 모델이 바로잡은 cycle이다.",
            "",
            "## 해석 제한",
            "",
            "- 같은 입력과 split을 사용했지만 Logistic Regression과 SVM에는 모델에 필요한 표준화를 적용했다.",
            "- Hyperparameter와 threshold를 탐색하지 않았으므로 각 모델 계열의 최고 성능 비교가 아니다.",
            "- SVM probability와 Random Forest probability는 산출 방식이 다르다.",
            "- Cycle 지표를 우선하며 window 지표만 좋아진 경우 운영상 우위로 해석하지 않는다.",
            "- 구간 탐지 결과이며 조기 고장 예측 또는 외부 일반화의 근거가 아니다.",
            "- 이 결과를 본 뒤 모델, C, kernel, gamma 또는 threshold를 추가로 조정하지 않는다.",
            "",
        ]
    )


def main() -> None:
    ensure_output_dir()
    predictions, candidates = load_predictions()
    runs = pd.read_csv(RUN_PATH, encoding="utf-8-sig")
    windows = ordered(window_metrics(predictions))
    cycles = ordered(cycle_results(predictions))
    blocks = ordered(block_metrics(cycles))
    summary = ordered(cycle_summary(cycles, blocks).merge(
        windows, on=["target", "model"], validate="one_to_one"
    ))
    paired = ordered(paired_errors(cycles))
    validate(predictions, candidates, runs, windows, cycles, blocks, summary, paired)

    windows.to_csv(
        OUTPUT_DIR / "15_classical_model_window_metrics.csv", index=False, encoding="utf-8-sig"
    )
    cycles.to_csv(
        OUTPUT_DIR / "15_classical_model_cycle_results.csv", index=False, encoding="utf-8-sig"
    )
    blocks.to_csv(
        OUTPUT_DIR / "15_classical_model_block_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        OUTPUT_DIR / "15_classical_model_summary.csv", index=False, encoding="utf-8-sig"
    )
    paired.to_csv(
        OUTPUT_DIR / "15_classical_model_paired_errors.csv", index=False, encoding="utf-8-sig"
    )
    report_path = OUTPUT_DIR / "15_classical_model_comparison.md"
    report_path.write_text(format_report(summary, paired, runs), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
