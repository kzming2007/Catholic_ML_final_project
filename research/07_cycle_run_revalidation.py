from __future__ import annotations

import pandas as pd

from common import OUTPUT_DIR, ensure_output_dir, markdown_table


REFERENCE_DIR = OUTPUT_DIR / "raw_cycle_reference"


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def compare_02() -> tuple[pd.DataFrame, pd.DataFrame]:
    old_results = read_csv(REFERENCE_DIR / "02_window_feature_results.csv")
    new_results = read_csv(OUTPUT_DIR / "02_window_feature_results.csv")
    keys = ["target", "window_size", "model"]
    metrics = ["macro_f1", "positive_recall", "positive_f1", "pr_auc"]
    merged = old_results.merge(new_results, on=keys, suffixes=("_raw_cycle", "_cycle_run"))
    merged = merged[merged["model"] == "rf_smote"].copy()
    for metric in metrics:
        merged[f"{metric}_delta"] = merged[f"{metric}_cycle_run"] - merged[f"{metric}_raw_cycle"]

    result_cols = keys.copy()
    for metric in metrics:
        result_cols.extend([f"{metric}_raw_cycle", f"{metric}_cycle_run", f"{metric}_delta"])

    old_summary = read_csv(REFERENCE_DIR / "02_window_dataset_summary.csv")
    new_summary = read_csv(OUTPUT_DIR / "02_window_dataset_summary.csv")
    count_comparison = old_summary.merge(
        new_summary,
        on=["target", "window_size"],
        suffixes=("_raw_cycle", "_cycle_run"),
    )
    count_comparison["windows_removed"] = (
        count_comparison["windows_raw_cycle"] - count_comparison["windows_cycle_run"]
    )
    return merged[result_cols], count_comparison


def compare_03_best() -> pd.DataFrame:
    old_best = read_csv(REFERENCE_DIR / "03_pre_failure_window_best.csv")
    new_best = read_csv(OUTPUT_DIR / "03_pre_failure_window_best.csv")
    old_best = old_best[
        ["target", "window_size", "prediction_horizon", "positive_recall", "positive_precision", "positive_f1", "pr_auc"]
    ].copy()
    new_best = new_best[
        ["target", "window_size", "prediction_horizon", "positive_recall", "positive_precision", "positive_f1", "pr_auc"]
    ].copy()
    return old_best.merge(new_best, on="target", suffixes=("_raw_cycle", "_cycle_run"))


def compare_04() -> pd.DataFrame:
    old_summary = read_csv(REFERENCE_DIR / "04_pre_failure_repeated_split_summary.csv")
    new_summary = read_csv(OUTPUT_DIR / "04_pre_failure_repeated_split_summary.csv")
    keys = ["target", "window_size", "prediction_horizon", "model"]
    metrics = ["runs_with_tp_rate", "positive_recall_mean", "positive_f1_mean", "pr_auc_mean"]
    merged = old_summary.merge(new_summary, on=keys, suffixes=("_raw_cycle", "_cycle_run"))
    cols = keys.copy()
    for metric in metrics:
        cols.extend([f"{metric}_raw_cycle", f"{metric}_cycle_run"])
    return merged[cols]


def compare_05() -> pd.DataFrame:
    old_recommended = read_csv(REFERENCE_DIR / "05_pre_failure_threshold_sensitivity_recommended.csv")
    new_recommended = read_csv(OUTPUT_DIR / "05_pre_failure_threshold_sensitivity_recommended.csv")
    metrics = ["threshold", "positive_recall_mean", "positive_f1_mean", "false_positive_rate_mean"]
    old_recommended = old_recommended[["target"] + metrics].copy()
    new_recommended = new_recommended[["target"] + metrics].copy()
    return old_recommended.merge(new_recommended, on="target", suffixes=("_raw_cycle", "_cycle_run"))


def round_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if any(token in col for token in ["f1", "recall", "precision", "auc", "rate", "threshold", "delta"]):
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out


def format_report(
    comparison_02: pd.DataFrame,
    count_comparison: pd.DataFrame,
    comparison_03: pd.DataFrame,
    comparison_04: pd.DataFrame,
    comparison_05: pd.DataFrame,
) -> str:
    count_cols = [
        "target",
        "window_size",
        "windows_raw_cycle",
        "windows_cycle_run",
        "windows_removed",
        "cycles",
        "cycle_runs",
    ]
    count_display = count_comparison[count_cols].copy()
    repeated = comparison_04.set_index("target")
    grip = repeated.loc["GripLost"]
    protective = repeated.loc["ProtectiveStop"]
    system = repeated.loc["System_Failure"]

    return "\n".join(
        [
            "# 07 Cycle-Run 경계 재검증",
            "",
            "## 목적",
            "",
            "시간상 떨어진 시행에서 동일 `cycle` ID가 재등장하는 문제를 수정하고, 기존 `02-05` 결과가 `cycle_run` 경계에서도 유지되는지 확인한다.",
            "",
            "## Window 수 변화",
            "",
            markdown_table(count_display),
            "",
            "- `windows_removed`는 raw `cycle`로 합쳤을 때 서로 다른 시행 사이에 만들어졌던 경계-crossing window 수다.",
            "",
            "## 02 구간 단위 이상탐지 비교",
            "",
            markdown_table(round_metrics(comparison_02)),
            "",
            "- 단일 group split 결과이므로 delta에는 경계 수정과 test 시행 구성 변화가 함께 반영된다.",
            "- window별 순위가 바뀌면 기존 단일 best 설정을 고정 결론으로 사용하지 않고 block-held-out 또는 반복 검증 결과를 우선한다.",
            "",
            "## 03 Pre-Failure 단일 Split Best 비교",
            "",
            markdown_table(round_metrics(comparison_03)),
            "",
            "- best 조합 자체가 달라질 수 있으므로 이 표는 안정성 진단용이며 직접 성능 개선 주장에 사용하지 않는다.",
            "",
            "## 04 동일 설정 30회 반복 비교",
            "",
            markdown_table(round_metrics(comparison_04)),
            "",
            "- 동일 target/window/horizon을 30회 반복한 결과가 경계 수정 전후 결론 안정성을 판단하는 주 근거다.",
            "",
            "## 05 Threshold 추천 비교",
            "",
            markdown_table(round_metrics(comparison_05)),
            "",
            "## 결론",
            "",
            "- raw `cycle` 경계 문제는 실제 window 구성과 일부 단일 split best 결과를 바꿨으므로 수정이 필요했다.",
            f"- 반복 검증의 positive F1 평균은 `GripLost` {grip['positive_f1_mean_raw_cycle']:.4f} -> {grip['positive_f1_mean_cycle_run']:.4f}, `ProtectiveStop` {protective['positive_f1_mean_raw_cycle']:.4f} -> {protective['positive_f1_mean_cycle_run']:.4f}, `System_Failure` {system['positive_f1_mean_raw_cycle']:.4f} -> {system['positive_f1_mean_cycle_run']:.4f}였다.",
            "- 따라서 `GripLost`가 상대적으로 가장 안정적인 pre-failure 타깃이라는 결론은 유지되지만, 절대 성능은 여전히 낮아 약한 사전 신호로만 해석한다.",
            "- threshold 후보는 `GripLost`와 `System_Failure`에서 0.30을 유지했고, `ProtectiveStop`은 0.15에서 0.20으로 바뀌어 고정값으로 보기 어렵다.",
            "- 이후 결과 보고는 raw `cycle` 버전이 아니라 `cycle_run` 버전을 기준으로 한다.",
            "",
        ]
    )


def main() -> None:
    ensure_output_dir()
    comparison_02, count_comparison = compare_02()
    comparison_03 = compare_03_best()
    comparison_04 = compare_04()
    comparison_05 = compare_05()

    comparison_02.to_csv(OUTPUT_DIR / "07_cycle_run_02_comparison.csv", index=False, encoding="utf-8-sig")
    count_comparison.to_csv(OUTPUT_DIR / "07_cycle_run_window_count_comparison.csv", index=False, encoding="utf-8-sig")
    comparison_03.to_csv(OUTPUT_DIR / "07_cycle_run_03_best_comparison.csv", index=False, encoding="utf-8-sig")
    comparison_04.to_csv(OUTPUT_DIR / "07_cycle_run_04_repeated_comparison.csv", index=False, encoding="utf-8-sig")
    comparison_05.to_csv(OUTPUT_DIR / "07_cycle_run_05_threshold_comparison.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "07_cycle_run_revalidation.md").write_text(
        format_report(comparison_02, count_comparison, comparison_03, comparison_04, comparison_05),
        encoding="utf-8",
    )
    print("Cycle-run 경계 재검증 비교 완료")
    print(OUTPUT_DIR / "07_cycle_run_revalidation.md")


if __name__ == "__main__":
    main()
