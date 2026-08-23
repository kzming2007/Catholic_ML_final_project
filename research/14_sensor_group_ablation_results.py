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


PREDICTION_PATH = OUTPUT_DIR / "14_sensor_group_ablation_predictions.csv"
RUN_PATH = OUTPUT_DIR / "14_sensor_group_ablation_runs.csv"
VARIANT_ORDER = [
    "all_19",
    "current_family_only",
    "joint_current_only",
    "tool_current_only",
    "speed_only",
    "temperature_only",
    "drop_current_family",
    "drop_speed",
    "drop_temperature",
    "drop_tool_current",
]
TARGET_ORDER = ["System_Failure", "ProtectiveStop", "GripLost"]
DROP_VARIANTS = [
    "drop_current_family", "drop_speed", "drop_temperature", "drop_tool_current"
]
ONLY_VARIANTS = [
    "current_family_only", "joint_current_only", "tool_current_only", "speed_only",
    "temperature_only",
]


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


def window_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, variant), group in predictions.groupby(["target", "variant"], sort=False):
        tn, fp, fn, tp = confusion_matrix(
            group["label"], group["prediction"], labels=[0, 1]
        ).ravel()
        rows.append(
            {
                "target": target,
                "variant": variant,
                "windows": int(len(group)),
                "feature_count": int(group["feature_count"].iloc[0]),
                "summary_feature_count": int(group["summary_feature_count"].iloc[0]),
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
                "window_pr_auc": float(average_precision_score(group["label"], group["score"])),
            }
        )
    return pd.DataFrame(rows)


def cycle_results(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["target", "variant", "test_block", "cycle", "cycle_run", "cycle_occurrence"]
    for group_keys, group in predictions.groupby(keys, sort=False):
        target, variant, test_block, cycle, cycle_run, occurrence = group_keys
        positive = group["label"].eq(1)
        rows.append(
            {
                "target": target,
                "variant": variant,
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
    for (target, variant, block), group in cycles.groupby(
        ["target", "variant", "test_block"], sort=False
    ):
        events = group[group["true_event_cycle"].eq(1)]
        normal = group[group["true_event_cycle"].eq(0)]
        detected = int(events["event_detected"].sum())
        false_alarms = int(normal["cycle_alert"].sum())
        rows.append(
            {
                "target": target,
                "variant": variant,
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
    for (target, variant), group in cycles.groupby(["target", "variant"], sort=False):
        events = group[group["true_event_cycle"].eq(1)]
        normal = group[group["true_event_cycle"].eq(0)]
        detected = int(events["event_detected"].sum())
        false_alarms = int(normal["cycle_alert"].sum())
        recall_low, recall_high = wilson_interval(detected, len(events))
        far_low, far_high = wilson_interval(false_alarms, len(normal))
        block_group = blocks[blocks["target"].eq(target) & blocks["variant"].eq(variant)]
        rows.append(
            {
                "target": target,
                "variant": variant,
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
    alert = cycles.pivot(index=index, columns="variant", values="cycle_alert").reset_index()
    detected = cycles.pivot(index=index, columns="variant", values="event_detected").reset_index()
    rows = []
    for target in TARGET_ORDER:
        for variant in VARIANT_ORDER[1:]:
            event_group = detected[
                detected["target"].eq(target) & detected["true_event_cycle"].eq(1)
            ]
            baseline_event_error = event_group["all_19"].eq(0)
            variant_event_error = event_group[variant].eq(0)
            normal_group = alert[
                alert["target"].eq(target) & alert["true_event_cycle"].eq(0)
            ]
            baseline_normal_error = normal_group["all_19"].eq(1)
            variant_normal_error = normal_group[variant].eq(1)
            for error_type, base_error, candidate_error, eligible in [
                ("event_miss", baseline_event_error, variant_event_error, event_group),
                ("false_alarm", baseline_normal_error, variant_normal_error, normal_group),
            ]:
                rows.append(
                    {
                        "target": target,
                        "variant": variant,
                        "error_type": error_type,
                        "eligible_cycles": int(len(eligible)),
                        "all_19_error_count": int(base_error.sum()),
                        "variant_error_count": int(candidate_error.sum()),
                        "shared_error_count": int((base_error & candidate_error).sum()),
                        "new_variant_error_count": int((~base_error & candidate_error).sum()),
                        "corrected_all_19_error_count": int((base_error & ~candidate_error).sum()),
                    }
                )
    return pd.DataFrame(rows)


def comparison_statement(summary: pd.DataFrame, target: str, variant: str) -> str:
    indexed = summary[summary["target"].eq(target)].set_index("variant")
    baseline = indexed.loc["all_19"]
    candidate = indexed.loc[variant]
    baseline_better = (
        baseline["event_cycle_recall"] >= candidate["event_cycle_recall"]
        and baseline["normal_cycle_false_alarm_rate"]
        <= candidate["normal_cycle_false_alarm_rate"]
    )
    candidate_better = (
        candidate["event_cycle_recall"] >= baseline["event_cycle_recall"]
        and candidate["normal_cycle_false_alarm_rate"]
        <= baseline["normal_cycle_false_alarm_rate"]
    )
    exact = (
        baseline["event_cycle_recall"] == candidate["event_cycle_recall"]
        and baseline["normal_cycle_false_alarm_rate"]
        == candidate["normal_cycle_false_alarm_rate"]
    )
    if exact:
        return "두 cycle 지표가 같다."
    if baseline_better and not candidate_better:
        return "all_19가 recall과 오경보 기준에서 기술적으로 우세하다."
    if candidate_better and not baseline_better:
        return f"{variant}가 recall과 오경보 기준에서 기술적으로 우세하다."
    return "Recall과 오경보 방향이 엇갈리는 trade-off다."


def standalone_statement(summary: pd.DataFrame, target: str) -> str:
    target_summary = summary[summary["target"].eq(target)].set_index("variant")
    current = target_summary.loc["current_family_only"]
    comparisons = []
    for variant in ["speed_only", "temperature_only"]:
        other = target_summary.loc[variant]
        current_better = (
            current["event_cycle_recall"] >= other["event_cycle_recall"]
            and current["normal_cycle_false_alarm_rate"]
            <= other["normal_cycle_false_alarm_rate"]
            and (
                current["event_cycle_recall"] > other["event_cycle_recall"]
                or current["normal_cycle_false_alarm_rate"]
                < other["normal_cycle_false_alarm_rate"]
            )
        )
        comparisons.append(current_better)
    if all(comparisons):
        return "current_family_only가 speed_only와 temperature_only를 두 cycle 지표에서 각각 Pareto 우세했다."
    return "current_family_only가 speed_only와 temperature_only를 모두 Pareto 우세하지는 않아 전류 단독 우위를 일반화할 수 없다."


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["target"] = pd.Categorical(result["target"], TARGET_ORDER, ordered=True)
    result["variant"] = pd.Categorical(result["variant"], VARIANT_ORDER, ordered=True)
    sort_columns = [column for column in ["target", "variant", "test_block"] if column in result]
    return result.sort_values(sort_columns).reset_index(drop=True)


def round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_float_dtype(result[column]):
            result[column] = result[column].round(4)
    return result


def delta_frame(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["variant"].eq("all_19")][
        [
            "target", "event_cycle_recall", "normal_cycle_false_alarm_rate",
            "window_macro_f1", "window_pr_auc",
        ]
    ].rename(
        columns={
            "event_cycle_recall": "baseline_event_cycle_recall",
            "normal_cycle_false_alarm_rate": "baseline_normal_cycle_false_alarm_rate",
            "window_macro_f1": "baseline_window_macro_f1",
            "window_pr_auc": "baseline_window_pr_auc",
        }
    )
    result = summary[~summary["variant"].eq("all_19")].merge(
        baseline, on="target", validate="many_to_one"
    )
    result["event_cycle_recall_delta"] = (
        result["event_cycle_recall"] - result["baseline_event_cycle_recall"]
    )
    result["normal_cycle_false_alarm_rate_delta"] = (
        result["normal_cycle_false_alarm_rate"]
        - result["baseline_normal_cycle_false_alarm_rate"]
    )
    result["window_macro_f1_delta"] = (
        result["window_macro_f1"] - result["baseline_window_macro_f1"]
    )
    result["window_pr_auc_delta"] = result["window_pr_auc"] - result["baseline_window_pr_auc"]
    return ordered(result)


def transition(summary: pd.DataFrame, target: str, variant: str) -> str:
    indexed = summary[summary["target"].eq(target)].set_index("variant")
    baseline = indexed.loc["all_19"]
    candidate = indexed.loc[variant]
    return (
        f"recall {baseline['event_cycle_recall']:.4f}->{candidate['event_cycle_recall']:.4f} "
        f"({candidate['event_cycle_recall'] - baseline['event_cycle_recall']:+.4f}), "
        f"오경보율 {baseline['normal_cycle_false_alarm_rate']:.4f}->"
        f"{candidate['normal_cycle_false_alarm_rate']:.4f} "
        f"({candidate['normal_cycle_false_alarm_rate'] - baseline['normal_cycle_false_alarm_rate']:+.4f})"
    )


def key_findings(summary: pd.DataFrame) -> list[str]:
    lines = []
    for target in TARGET_ORDER:
        lines.append(
            f"- `{target}` 전류 계열 제거: {transition(summary, target, 'drop_current_family')}."
        )
    for target in TARGET_ORDER:
        lines.append(
            f"- `{target}` Tool current 제거: {transition(summary, target, 'drop_tool_current')}."
        )
    speed = summary[summary["variant"].eq("speed_only")]
    temperature = summary[summary["variant"].eq("temperature_only")]
    lines.extend(
        [
            f"- Speed 단독은 event recall {speed['event_cycle_recall'].min():.4f}-"
            f"{speed['event_cycle_recall'].max():.4f}를 보였지만 정상 cycle 오경보율도 "
            f"{speed['normal_cycle_false_alarm_rate'].min():.4f}-"
            f"{speed['normal_cycle_false_alarm_rate'].max():.4f}였다.",
            f"- Temperature 단독 event recall은 {temperature['event_cycle_recall'].min():.4f}-"
            f"{temperature['event_cycle_recall'].max():.4f}로 낮았다. 낮은 오경보와 함께 "
            "대부분의 event를 경보하지 않은 결과이므로 고장 탐지력이 높다고 해석하지 않는다.",
            "- 종합하면 joint current는 System_Failure와 GripLost 구간 탐지에 중요한 정보를 "
            "제공하지만, 전류 계열이 모든 타깃에서 유일하거나 항상 최적인 것은 아니다. "
            "Tool current의 기여도 joint current와 분리해 해석해야 한다.",
        ]
    )
    return lines


def validate(
    predictions: pd.DataFrame,
    runs: pd.DataFrame,
    windows: pd.DataFrame,
    cycles: pd.DataFrame,
    blocks: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    if len(predictions) != 121050 or len(runs) != 270:
        raise AssertionError("Unexpected prediction or run count.")
    if predictions.duplicated(
        ["target", "variant", "test_block", "cycle_run", "window_start_step"]
    ).any():
        raise AssertionError("Duplicate prediction keys.")
    if not predictions.groupby(["target", "variant"]).size().eq(4035).all():
        raise AssertionError("Each target/variant must have 4,035 predictions.")
    if not predictions.groupby(["target", "variant"])["test_block"].nunique().eq(9).all():
        raise AssertionError("Each target/variant must have nine test blocks.")
    if not np.isfinite(predictions["score"]).all():
        raise AssertionError("Non-finite score found.")
    expected_prediction = predictions["score"].to_numpy() > predictions["decision_threshold"].to_numpy()
    if not np.array_equal(predictions["prediction"].to_numpy(), expected_prediction.astype(int)):
        raise AssertionError("Decision threshold reconstruction failed.")
    all_runs = runs[runs["variant"].eq("all_19")]
    if len(all_runs) != 27 or not all_runs["baseline_12_match"].astype(str).str.lower().eq("true").all():
        raise AssertionError("all_19 did not match experiment 12 in every fold.")
    if len(windows) != 30 or len(summary) != 30 or len(blocks) != 270:
        raise AssertionError("Unexpected summary dimensions.")
    if not cycles.groupby(["target", "variant"]).size().eq(202).all():
        raise AssertionError("Each target/variant must contain 202 cycles.")
    if len(paired) != 54:
        raise AssertionError("Expected paired errors for nine variants, three targets, two error types.")
    if not (windows["window_tn"] + windows["window_fp"] + windows["window_fn"] + windows["window_tp"]).eq(4035).all():
        raise AssertionError("Window confusion counts do not sum to 4,035.")


def format_report(summary: pd.DataFrame, paired: pd.DataFrame, runs: pd.DataFrame) -> str:
    display_columns = [
        "target", "variant", "feature_count", "event_cycles", "detected_event_cycles",
        "event_cycle_recall", "event_cycle_recall_ci95_low", "event_cycle_recall_ci95_high",
        "event_cycle_recall_min_block", "normal_cycles", "false_alarm_cycles",
        "normal_cycle_false_alarm_rate", "normal_cycle_false_alarm_rate_ci95_low",
        "normal_cycle_false_alarm_rate_ci95_high", "normal_cycle_false_alarm_rate_max_block",
        "window_macro_f1", "window_positive_f1", "window_roc_auc", "window_pr_auc",
    ]
    baseline = summary[summary["variant"].eq("all_19")]
    standalone = summary[summary["variant"].isin(ONLY_VARIANTS)]
    removal = summary[summary["variant"].isin(DROP_VARIANTS)]
    deltas = delta_frame(summary)
    delta_columns = [
        "target", "variant", "event_cycle_recall_delta",
        "normal_cycle_false_alarm_rate_delta", "window_macro_f1_delta", "window_pr_auc_delta",
    ]
    conclusions = []
    for target in TARGET_ORDER:
        conclusions.append(f"- `{target}` 단독 그룹: {standalone_statement(summary, target)}")
        for variant in DROP_VARIANTS:
            conclusions.append(
                f"- `{target}` `{variant}`: {comparison_statement(summary, target, variant)}"
            )
    paired_display = paired[
        [
            "target", "variant", "error_type", "eligible_cycles", "all_19_error_count",
            "variant_error_count", "shared_error_count", "new_variant_error_count",
            "corrected_all_19_error_count",
        ]
    ]
    return "\n".join(
        [
            "# 14 센서 그룹 ablation 결과",
            "",
            "## 고정 설계",
            "",
            "- 사전 고정: `research/2026-08-23_sensor_group_ablation_preregistration.md`.",
            "- 입력: cycle_run 경계를 넘지 않는 10-step×19개 원본 센서.",
            "- 특징: 선택 센서마다 mean, std, min, max, range, delta, slope.",
            "- 모델: SMOTE + Random Forest 300 trees, random_state 42, score > 0.50.",
            "- 평가: 9개 acquisition block을 한 번씩 test로 사용.",
            "- all_19 27개 fold는 기존 실험 12의 score와 prediction에 모두 일치했다.",
            "",
            "## 핵심 결과",
            "",
            *key_findings(summary),
            "",
            "## All-sensor 기준선",
            "",
            markdown_table(round_frame(baseline[display_columns])),
            "",
            "## 센서 그룹 단독 사용",
            "",
            markdown_table(round_frame(standalone[display_columns])),
            "",
            "## 센서 그룹 제거",
            "",
            markdown_table(round_frame(removal[display_columns])),
            "",
            "## All-sensor 대비 변화량",
            "",
            markdown_table(round_frame(deltas[delta_columns])),
            "",
            "- Recall delta는 클수록 유리하고 정상 cycle 오경보율 delta는 작을수록 유리하다.",
            "- Window 지표 delta와 cycle 지표 delta의 방향이 다르면 평가 단위에 따른 trade-off로 본다.",
            "",
            "## All-sensor 대비 paired cycle 오류",
            "",
            markdown_table(paired_display),
            "",
            "## 사전 고정 해석 규칙 적용",
            "",
            *conclusions,
            "- Pareto 우세는 recall과 오경보의 방향을 함께 본 기술적 비교이며 통계적 유의성이나 인과관계를 뜻하지 않는다.",
            "",
            "## 실행 범위",
            "",
            f"- Random Forest run: {len(runs)}개.",
            f"- Window prediction: {len(pd.read_csv(PREDICTION_PATH, encoding='utf-8-sig')):,}개.",
            f"- 기록된 학습 시간 합: {runs['elapsed_seconds'].sum():.1f}초.",
            "",
            "## 해석 제한",
            "",
            "- Feature 수와 SMOTE 공간이 variant마다 달라 제거 효과를 순수한 인과 기여로 해석할 수 없다.",
            "- Tool current의 구간 탐지 성능은 고장 발생 전 유용성을 뜻하지 않는다.",
            "- Temperature 성능에는 thermal/session drift가 포함될 수 있다.",
            "- 4,035개 window는 겹치며 독립 평가 단위는 202개 cycle이다.",
            "- 9개 block은 실제 공정조건 정답이 아닌 수집 구간 proxy다.",
            "- 단일 공개 데이터와 Random Forest 고정 seed 1회의 내부 비교다.",
            "",
        ]
    )


def main() -> None:
    ensure_output_dir()
    predictions = pd.read_csv(PREDICTION_PATH, encoding="utf-8-sig")
    runs = pd.read_csv(RUN_PATH, encoding="utf-8-sig")
    windows = ordered(window_metrics(predictions))
    cycles = ordered(cycle_results(predictions))
    blocks = ordered(block_metrics(cycles))
    cycle_models = ordered(cycle_summary(cycles, blocks))
    summary = ordered(cycle_models.merge(windows, on=["target", "variant"], validate="one_to_one"))
    paired = ordered(paired_errors(cycles))
    validate(predictions, runs, windows, cycles, blocks, summary, paired)

    windows.to_csv(
        OUTPUT_DIR / "14_sensor_group_ablation_window_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cycles.to_csv(
        OUTPUT_DIR / "14_sensor_group_ablation_cycle_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    blocks.to_csv(
        OUTPUT_DIR / "14_sensor_group_ablation_block_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        OUTPUT_DIR / "14_sensor_group_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        OUTPUT_DIR / "14_sensor_group_ablation_paired_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report_path = OUTPUT_DIR / "14_sensor_group_ablation.md"
    report_path.write_text(format_report(summary, paired, runs), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
