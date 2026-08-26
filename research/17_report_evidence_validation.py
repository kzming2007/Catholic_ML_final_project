from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATASET_PATH, OUTPUT_DIR, REPO_ROOT, ensure_output_dir, markdown_table


REPORT_PATH = REPO_ROOT / "research" / "2026-08-27_research_report_draft.md"
EXPECTED_DATASET_SHA256 = "C789CDA10ACB354A7C1689F617D94A5F39A93FD8CB6C004AD16D36CEA55A74A3"
TOLERANCE = 5e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def add_check(
    rows: list[dict],
    check_id: str,
    source: str,
    expected,
    observed,
    passed: bool,
    tolerance: str = "exact",
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "source": source,
            "expected": expected,
            "observed": observed,
            "tolerance": tolerance,
            "passed": bool(passed),
        }
    )


def check_equal(rows: list[dict], check_id: str, source: str, expected, observed) -> None:
    add_check(rows, check_id, source, expected, observed, expected == observed)


def check_close(
    rows: list[dict], check_id: str, source: str, expected: float, observed: float
) -> None:
    add_check(
        rows,
        check_id,
        source,
        expected,
        observed,
        bool(np.isclose(expected, observed, atol=TOLERANCE, rtol=0)),
        f"abs<={TOLERANCE}",
    )


def selected_row(frame: pd.DataFrame, **filters) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in filters.items():
        mask &= frame[column].eq(value)
    selected = frame[mask]
    if len(selected) != 1:
        raise AssertionError(f"Expected one row for {filters}, found {len(selected)}.")
    return selected.iloc[0]


def validate_dataset(rows: list[dict]) -> None:
    data = pd.read_csv(DATASET_PATH, encoding="utf-8")
    check_equal(rows, "dataset_sha256", "dataset/ur3_cobotops.csv", EXPECTED_DATASET_SHA256, sha256(DATASET_PATH))
    check_equal(rows, "dataset_rows", "dataset/ur3_cobotops.csv", 7409, len(data))
    check_equal(rows, "dataset_columns", "dataset/ur3_cobotops.csv", 24, len(data.columns))


def validate_row_window_results(rows: list[dict]) -> None:
    frame = pd.read_csv(OUTPUT_DIR / "02_row_vs_window_comparison.csv", encoding="utf-8-sig")
    expected = {
        ("System_Failure", "row_baseline", "macro_f1"): 0.7901,
        ("System_Failure", "best_window_feature", "macro_f1"): 0.8166,
        ("System_Failure", "row_baseline", "pr_auc"): 0.6430,
        ("System_Failure", "best_window_feature", "pr_auc"): 0.8613,
        ("ProtectiveStop", "row_baseline", "macro_f1"): 0.8428,
        ("ProtectiveStop", "best_window_feature", "macro_f1"): 0.8642,
        ("ProtectiveStop", "row_baseline", "pr_auc"): 0.7409,
        ("ProtectiveStop", "best_window_feature", "pr_auc"): 0.9039,
        ("GripLost", "row_baseline", "macro_f1"): 0.7623,
        ("GripLost", "best_window_feature", "macro_f1"): 0.8412,
        ("GripLost", "row_baseline", "pr_auc"): 0.5604,
        ("GripLost", "best_window_feature", "pr_auc"): 0.8493,
    }
    for (target, level, metric), value in expected.items():
        observed = float(selected_row(frame, target=target, feature_level=level)[metric])
        check_close(rows, f"02_{target}_{level}_{metric}", "02_row_vs_window_comparison.csv", value, observed)


def validate_fault_context(rows: list[dict]) -> None:
    summary = pd.read_csv(OUTPUT_DIR / "16_fault_context_alert_summary.csv", encoding="utf-8-sig")
    cycles = pd.read_csv(OUTPUT_DIR / "16_fault_context_cycle_results.csv", encoding="utf-8-sig")
    check_equal(rows, "16_summary_rows", "16_fault_context_alert_summary.csv", 54, len(summary))
    check_equal(rows, "16_cycle_rows", "16_fault_context_cycle_results.csv", 10908, len(cycles))
    check_equal(
        rows,
        "16_unique_cycle_keys",
        "16_fault_context_cycle_results.csv",
        10908,
        len(cycles.drop_duplicates(["source", "candidate", "target", "test_block", "cycle", "cycle_run", "cycle_occurrence"])),
    )

    for target, expected_cross in {"System_Failure": 0, "ProtectiveStop": 25, "GripLost": 48}.items():
        target_rows = summary[summary["target"].eq(target)]
        check_equal(
            rows,
            f"16_{target}_true_normal_cycles",
            "16_fault_context_alert_summary.csv",
            [115],
            sorted(target_rows["true_normal_cycles"].unique().tolist()),
        )
        check_equal(
            rows,
            f"16_{target}_other_fault_cycles",
            "16_fault_context_alert_summary.csv",
            [expected_cross],
            sorted(target_rows["other_fault_only_cycles"].unique().tolist()),
        )

    expected_metrics = {
        ("12_matched_models", "rf_19_raw_window", "System_Failure"): (0.9770, 0.0435, np.nan),
        ("12_matched_models", "rf_19_raw_window", "ProtectiveStop"): (0.9516, 0.0348, 0.0000),
        ("12_matched_models", "rf_19_raw_window", "GripLost"): (0.9231, 0.0174, 0.0833),
        ("12_matched_models", "1d_cnn_19_raw", "System_Failure"): (0.9770, 0.1913, np.nan),
        ("12_matched_models", "1d_cnn_19_raw", "ProtectiveStop"): (0.9355, 0.2087, 0.1200),
        ("12_matched_models", "1d_cnn_19_raw", "GripLost"): (0.9231, 0.1652, 0.4375),
        ("12_matched_models", "lstm_autoencoder_19_raw_q95", "System_Failure"): (0.0460, 0.1304, np.nan),
        ("12_matched_models", "lstm_autoencoder_19_raw_q95", "ProtectiveStop"): (0.0645, 0.1304, 0.0000),
        ("12_matched_models", "lstm_autoencoder_19_raw_q95", "GripLost"): (0.0000, 0.1304, 0.1042),
        ("15_classical_models", "logistic_regression", "System_Failure"): (0.9885, 0.5304, np.nan),
        ("15_classical_models", "logistic_regression", "ProtectiveStop"): (0.9677, 0.2696, 0.2800),
        ("15_classical_models", "logistic_regression", "GripLost"): (1.0000, 0.4522, 0.5833),
        ("15_classical_models", "rbf_svm", "System_Failure"): (0.9540, 0.0957, np.nan),
        ("15_classical_models", "rbf_svm", "ProtectiveStop"): (0.9032, 0.0174, 0.0400),
        ("15_classical_models", "rbf_svm", "GripLost"): (0.9487, 0.0957, 0.3750),
    }
    metric_names = ["event_cycle_recall", "true_normal_false_alarm_rate", "cross_fault_alert_rate"]
    for (source, candidate, target), expected_values in expected_metrics.items():
        row = selected_row(summary, source=source, candidate=candidate, target=target)
        for metric, expected in zip(metric_names, expected_values):
            observed = float(row[metric])
            check_id = f"16_{candidate}_{target}_{metric}"
            if np.isnan(expected):
                add_check(rows, check_id, "16_fault_context_alert_summary.csv", "NaN", observed, np.isnan(observed))
            else:
                check_close(rows, check_id, "16_fault_context_alert_summary.csv", expected, observed)


def validate_window_pr_auc(rows: list[dict]) -> None:
    frame = pd.read_csv(OUTPUT_DIR / "13_window_model_summary.csv", encoding="utf-8-sig")
    expected = {
        ("rf_19_raw_window", "System_Failure"): 0.8137,
        ("rf_19_raw_window", "ProtectiveStop"): 0.8557,
        ("rf_19_raw_window", "GripLost"): 0.7267,
        ("1d_cnn_19_raw", "System_Failure"): 0.6879,
        ("1d_cnn_19_raw", "ProtectiveStop"): 0.7659,
        ("1d_cnn_19_raw", "GripLost"): 0.5888,
        ("lstm_autoencoder_19_raw_q95", "System_Failure"): 0.2228,
        ("lstm_autoencoder_19_raw_q95", "ProtectiveStop"): 0.1629,
        ("lstm_autoencoder_19_raw_q95", "GripLost"): 0.0834,
    }
    for (model, target), value in expected.items():
        row = selected_row(frame, model_variant=model, target=target)
        check_close(rows, f"13_{model}_{target}_pr_auc", "13_window_model_summary.csv", value, float(row["pr_auc_mean"]))


def validate_pre_failure(rows: list[dict]) -> None:
    frame = pd.read_csv(OUTPUT_DIR / "04_pre_failure_repeated_split_summary.csv", encoding="utf-8-sig")
    expected = {
        "System_Failure": (0.0920, 0.0644, 0.8333),
        "ProtectiveStop": (0.0867, 0.0667, 0.1333),
        "GripLost": (0.3058, 0.2248, 0.9667),
    }
    for target, (f1, recall, runs_with_tp) in expected.items():
        row = selected_row(frame, target=target)
        check_close(rows, f"04_{target}_positive_f1", "04_pre_failure_repeated_split_summary.csv", f1, float(row["positive_f1_mean"]))
        check_close(rows, f"04_{target}_positive_recall", "04_pre_failure_repeated_split_summary.csv", recall, float(row["positive_recall_mean"]))
        check_close(rows, f"04_{target}_runs_with_tp", "04_pre_failure_repeated_split_summary.csv", runs_with_tp, float(row["runs_with_tp_rate"]))


def validate_report_structure(rows: list[dict]) -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    headings = [
        "## 초록",
        "## 1. 서론",
        "## 2. 관련 연구",
        "## 3. 연구 방법",
        "## 4. 연구 결과",
        "## 5. 논의",
        "## 6. 연구의 한계",
        "## 7. 결론",
        "## 8. 재현성 및 검증 계획",
        "## 참고문헌",
    ]
    for heading in headings:
        check_equal(rows, f"report_heading_{heading}", REPORT_PATH.name, True, heading in report)


def validate_report_claims(rows: list[dict]) -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    required_snippets = {
        "dataset_shape": "공개 CSV는 7,409행×24열",
        "sensor_count": "원본 센서 19개를 사용하였다",
        "cycle_run_count": "202개 `cycle_run`",
        "window_count": "서로 겹치는 window는 총 4,035개",
        "event_cycle_counts": "`System_Failure` 87개, `ProtectiveStop` 62개, `GripLost` 39개",
        "rf_feature_count": "133개 특징을 만들었다",
        "rf_tree_count": "300-tree Random Forest",
        "row_window_system": "| `System_Failure` | 0.7901 | 5-step | 0.8166 | 0.6430 | 0.8613 |",
        "row_window_protective": "| `ProtectiveStop` | 0.8428 | 5-step | 0.8642 | 0.7409 | 0.9039 |",
        "row_window_grip": "| `GripLost` | 0.7623 | 10-step | 0.8412 | 0.5604 | 0.8493 |",
        "rf_system": "| `System_Failure` | Random Forest | 0.9770 | 0.0435 | 해당 없음 |",
        "rf_protective": "| `ProtectiveStop` | Random Forest | 0.9516 | 0.0348 | 0.0000 |",
        "rf_grip": "| `GripLost` | Random Forest | 0.9231 | 0.0174 | 0.0833 |",
        "cnn_system": "|  | 1D CNN | 0.9770 | 0.1913 | 해당 없음 |",
        "cnn_protective": "|  | 1D CNN | 0.9355 | 0.2087 | 0.1200 |",
        "cnn_grip": "|  | 1D CNN | 0.9231 | 0.1652 | 0.4375 |",
        "autoencoder_system": "|  | LSTM Autoencoder q95 | 0.0460 | 0.1304 | 해당 없음 |",
        "autoencoder_protective": "|  | LSTM Autoencoder q95 | 0.0645 | 0.1304 | 0.0000 |",
        "autoencoder_grip": "|  | LSTM Autoencoder q95 | 0.0000 | 0.1304 | 0.1042 |",
        "window_pr_auc_rf": "`System_Failure` 0.8137, `ProtectiveStop` 0.8557, `GripLost` 0.7267",
        "window_pr_auc_cnn": "1D CNN의 0.6879, 0.7659, 0.5888",
        "window_pr_auc_autoencoder": "각각 0.2228, 0.1629, 0.0834",
        "pre_failure_f1": "`System_Failure` 0.0920, `ProtectiveStop` 0.0867, `GripLost` 0.3058",
        "pre_failure_grip_recall": "평균 recall은 0.2248",
        "dataset_hash": EXPECTED_DATASET_SHA256,
    }
    for claim_id, snippet in required_snippets.items():
        check_equal(rows, f"report_claim_{claim_id}", REPORT_PATH.name, True, snippet in report)


def write_results(frame: pd.DataFrame) -> None:
    ensure_output_dir()
    csv_path = OUTPUT_DIR / "17_report_evidence_validation.csv"
    report_path = OUTPUT_DIR / "17_report_evidence_validation.md"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    failures = frame[~frame["passed"]]
    source_summary = (
        frame.groupby("source", as_index=False)
        .agg(checks=("check_id", "size"), passed=("passed", "sum"))
    )
    source_summary["failed"] = source_summary["checks"] - source_summary["passed"]
    lines = [
        "# 17 보고서 근거 대조",
        "",
        f"- 총 검사: {len(frame)}",
        f"- 통과: {int(frame['passed'].sum())}",
        f"- 실패: {len(failures)}",
        "",
        "## 출처별 결과",
        "",
        markdown_table(source_summary),
        "",
        "## 실패 항목",
        "",
        markdown_table(failures),
        "",
        "## 해석 범위",
        "",
        "- 이 검사는 저장된 CSV와 보고서 핵심 수치의 내부 일관성을 확인한다.",
        "- 모델을 다시 학습하지 않으므로 전체 재현 실험을 대신하지 않는다.",
        "- 전체 재현은 별도 clean worktree에서 CPU·GPU 실행을 다시 수행해 비교한다.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)
    print(f"checks={len(frame)} passed={int(frame['passed'].sum())} failed={len(failures)}")


def main() -> None:
    rows: list[dict] = []
    validate_dataset(rows)
    validate_row_window_results(rows)
    validate_fault_context(rows)
    validate_window_pr_auc(rows)
    validate_pre_failure(rows)
    validate_report_structure(rows)
    validate_report_claims(rows)
    frame = pd.DataFrame(rows)
    write_results(frame)
    if not frame["passed"].all():
        failed_ids = frame.loc[~frame["passed"], "check_id"].tolist()
        raise AssertionError(f"Report evidence validation failed: {failed_ids}")


if __name__ == "__main__":
    main()
