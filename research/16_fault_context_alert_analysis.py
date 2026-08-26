from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common import OUTPUT_DIR, ensure_output_dir, markdown_table


TARGET_ORDER = ["System_Failure", "ProtectiveStop", "GripLost"]
CYCLE_KEYS = ["test_block", "cycle", "cycle_run", "cycle_occurrence"]
SOURCE_ORDER = ["12_matched_models", "14_sensor_ablation", "15_classical_models"]


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return center - margin, center + margin


def canonicalize(
    path: str,
    source: str,
    candidate_column: str,
    alert_column: str,
    detected_column: str,
) -> pd.DataFrame:
    frame = pd.read_csv(OUTPUT_DIR / path, encoding="utf-8-sig")
    frame = frame.rename(
        columns={
            candidate_column: "candidate",
            alert_column: "cycle_alert",
            detected_column: "event_detected",
        }
    )
    frame["source"] = source
    columns = [
        "source",
        "candidate",
        "target",
        *CYCLE_KEYS,
        "true_event_cycle",
        "cycle_alert",
        "event_detected",
    ]
    return frame[columns].copy()


def load_cycle_results() -> pd.DataFrame:
    frames = [
        canonicalize(
            "12_matched_consensus_cycle_results.csv",
            "12_matched_models",
            "model_variant",
            "consensus_cycle_alert",
            "consensus_event_detected",
        ),
        canonicalize(
            "14_sensor_group_ablation_cycle_results.csv",
            "14_sensor_ablation",
            "variant",
            "cycle_alert",
            "event_detected",
        ),
        canonicalize(
            "15_classical_model_cycle_results.csv",
            "15_classical_models",
            "model",
            "cycle_alert",
            "event_detected",
        ),
    ]
    cycles = pd.concat(frames, ignore_index=True)
    integer_columns = CYCLE_KEYS + ["true_event_cycle", "cycle_alert", "event_detected"]
    cycles[integer_columns] = cycles[integer_columns].astype(int)
    return cycles


def attach_fault_context(cycles: pd.DataFrame) -> pd.DataFrame:
    context_keys = ["source", "candidate", *CYCLE_KEYS]
    system_truth = (
        cycles[cycles["target"].eq("System_Failure")][context_keys + ["true_event_cycle"]]
        .rename(columns={"true_event_cycle": "any_failure_cycle"})
        .drop_duplicates(context_keys)
    )
    result = cycles.merge(system_truth, on=context_keys, how="left", validate="many_to_one")
    if result["any_failure_cycle"].isna().any():
        raise ValueError("System_Failure cycle truth is missing for at least one row.")
    result["any_failure_cycle"] = result["any_failure_cycle"].astype(int)
    invalid = result["true_event_cycle"].gt(result["any_failure_cycle"])
    if invalid.any():
        raise ValueError("A target event is marked positive while System_Failure is negative.")
    result["true_normal_cycle"] = result["any_failure_cycle"].eq(0).astype(int)
    result["other_fault_only_cycle"] = (
        result["true_event_cycle"].eq(0) & result["any_failure_cycle"].eq(1)
    ).astype(int)
    return result


def summarize(cycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, candidate, target), group in cycles.groupby(
        ["source", "candidate", "target"], sort=False
    ):
        event = group[group["true_event_cycle"].eq(1)]
        target_negative = group[group["true_event_cycle"].eq(0)]
        true_normal = group[group["true_normal_cycle"].eq(1)]
        cross_fault = group[group["other_fault_only_cycle"].eq(1)]

        detected = int(event["event_detected"].sum())
        target_negative_alerts = int(target_negative["cycle_alert"].sum())
        true_normal_alerts = int(true_normal["cycle_alert"].sum())
        cross_fault_alerts = int(cross_fault["cycle_alert"].sum())
        true_normal_low, true_normal_high = wilson_interval(
            true_normal_alerts, len(true_normal)
        )
        cross_low, cross_high = wilson_interval(cross_fault_alerts, len(cross_fault))

        rows.append(
            {
                "source": source,
                "candidate": candidate,
                "target": target,
                "event_cycles": int(len(event)),
                "detected_event_cycles": detected,
                "event_cycle_recall": detected / len(event) if len(event) else np.nan,
                "target_negative_cycles": int(len(target_negative)),
                "target_negative_alert_cycles": target_negative_alerts,
                "target_negative_alert_rate": target_negative_alerts / len(target_negative)
                if len(target_negative)
                else np.nan,
                "true_normal_cycles": int(len(true_normal)),
                "true_normal_false_alarm_cycles": true_normal_alerts,
                "true_normal_false_alarm_rate": true_normal_alerts / len(true_normal)
                if len(true_normal)
                else np.nan,
                "true_normal_false_alarm_rate_ci95_low": true_normal_low,
                "true_normal_false_alarm_rate_ci95_high": true_normal_high,
                "other_fault_only_cycles": int(len(cross_fault)),
                "cross_fault_alert_cycles": cross_fault_alerts,
                "cross_fault_alert_rate": cross_fault_alerts / len(cross_fault)
                if len(cross_fault)
                else np.nan,
                "cross_fault_alert_rate_ci95_low": cross_low,
                "cross_fault_alert_rate_ci95_high": cross_high,
            }
        )
    return pd.DataFrame(rows)


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["source"] = pd.Categorical(result["source"], SOURCE_ORDER, ordered=True)
    result["target"] = pd.Categorical(result["target"], TARGET_ORDER, ordered=True)
    return result.sort_values(["source", "candidate", "target"]).reset_index(drop=True)


def report_table(summary: pd.DataFrame, source: str, candidates: list[str]) -> pd.DataFrame:
    columns = [
        "candidate",
        "target",
        "event_cycle_recall",
        "target_negative_alert_rate",
        "true_normal_false_alarm_rate",
        "cross_fault_alert_rate",
        "other_fault_only_cycles",
    ]
    selected = summary[
        summary["source"].eq(source) & summary["candidate"].isin(candidates)
    ][columns].copy()
    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].round(4)
    return selected


def write_report(summary: pd.DataFrame) -> None:
    model_table = pd.concat(
        [
            report_table(
                summary,
                "12_matched_models",
                [
                    "rf_19_raw_window",
                    "1d_cnn_19_raw",
                    "lstm_autoencoder_19_raw_q95",
                ],
            ),
            report_table(
                summary,
                "15_classical_models",
                ["logistic_regression", "rbf_svm"],
            ),
        ],
        ignore_index=True,
    )
    ablation_table = report_table(
        summary,
        "14_sensor_ablation",
        summary[summary["source"].eq("14_sensor_ablation")]["candidate"].unique().tolist(),
    )
    lines = [
        "# Fault-context alert metric correction",
        "",
        "## 목적",
        "",
        "기존 결과의 `normal_cycle_false_alarm_rate`는 실제로 각 target이 없는 모든 cycle을 분모로 사용했다. "
        "따라서 `ProtectiveStop` 평가에는 `GripLost`만 발생한 cycle이, `GripLost` 평가에는 "
        "`ProtectiveStop`만 발생한 cycle이 포함됐다. 이 분석은 재학습이나 threshold 변경 없이 기존 cycle 결과를 다음 세 범주로 분리한다.",
        "",
        "- `target_negative_alert_rate`: 해당 target이 없는 모든 cycle에서의 경보율",
        "- `true_normal_false_alarm_rate`: 어떤 고장도 없는 cycle에서의 오경보율",
        "- `cross_fault_alert_rate`: 해당 target은 없지만 다른 고장만 있는 cycle에서의 경보율",
        "",
        "`System_Failure`는 두 개별 고장의 합집합이므로 교차 고장 범주가 없다.",
        "",
        "## 모델 비교",
        "",
        markdown_table(model_table),
        "",
        "## 센서 그룹 ablation",
        "",
        markdown_table(ablation_table),
        "",
        "## 해석 제한",
        "",
        "- 교차 고장 경보는 곧바로 오류라고 단정할 수 없다. 두 고장 유형이 일부 센서 패턴을 공유할 수 있기 때문이다.",
        "- 반대로 target별 분류 성능으로 보고할 때는 교차 고장 경보를 해당 target의 정답으로 계산할 수 없다.",
        "- 세 지표는 동일한 cycle 예측을 서로 다른 운영 질문에 맞춰 재집계한 값이며 독립적인 새 실험이 아니다.",
        "- 현재 결과는 동일 데이터셋의 block-held-out 내부 검증이며 외부 로봇·외부 수집 세션 일반화 증거가 아니다.",
        "",
    ]
    (OUTPUT_DIR / "16_fault_context_alert_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    ensure_output_dir()
    cycles = ordered(attach_fault_context(load_cycle_results()))
    summary = ordered(summarize(cycles))
    cycles.to_csv(
        OUTPUT_DIR / "16_fault_context_cycle_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        OUTPUT_DIR / "16_fault_context_alert_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_report(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
