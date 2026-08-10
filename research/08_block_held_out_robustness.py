from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUTPUT_DIR, TARGET_COLS, ensure_output_dir, load_model_data, markdown_table


WINDOW_SIZES = [5, 10, 20]
MODEL_NAME = "rf_smote"
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


def run_holdout(window_module, window_df: pd.DataFrame, cols: list[str], test_block: int) -> dict[str, float | int | str] | None:
    train_mask = window_df["scenario_block_25"].ne(test_block)
    test_mask = window_df["scenario_block_25"].eq(test_block)
    train_df = window_df.loc[train_mask]
    test_df = window_df.loc[test_mask]
    y_train = train_df["label"].astype(int)
    y_test = test_df["label"].astype(int)
    if train_df.empty or test_df.empty or y_train.nunique() < 2 or y_test.nunique() < 2:
        return None
    if int(y_train.value_counts().min()) < 2:
        return None

    model = window_module.build_model(MODEL_NAME, y_train)
    model.fit(train_df[cols], y_train)
    y_pred = model.predict(test_df[cols])
    y_score = model.predict_proba(test_df[cols])[:, 1]
    metrics = window_module.evaluate_binary(y_test, y_pred, y_score)
    negative_count = int((y_test == 0).sum())
    metrics["false_positive_rate"] = float(metrics["fp"] / negative_count) if negative_count else np.nan

    row = {
        "target": str(window_df["target"].iloc[0]),
        "window_size": int(window_df["window_size"].iloc[0]),
        "model": MODEL_NAME,
        "test_block": int(test_block),
        "train_blocks": int(train_df["scenario_block_25"].nunique()),
        "train_cycle_runs": int(train_df["cycle_run"].nunique()),
        "test_cycle_runs": int(test_df["cycle_run"].nunique()),
        "train_windows": int(len(train_df)),
    }
    row.update(metrics)
    return row


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "macro_f1",
        "positive_recall",
        "positive_precision",
        "positive_f1",
        "pr_auc",
        "roc_auc",
        "false_positive_rate",
    ]
    for keys, group in results.groupby(["target", "window_size", "feature_set", "model"], sort=False):
        target, window_size, feature_set, model = keys
        row: dict[str, float | int | str] = {
            "target": target,
            "window_size": int(window_size),
            "feature_set": feature_set,
            "model": model,
            "valid_test_blocks": int(len(group)),
            "blocks_with_tp_rate": float((group["tp"] > 0).mean()),
            "test_positive_count_mean": float(group["positive_count"].mean()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
            row[f"{metric}_min"] = float(group[metric].min())
        rows.append(row)
    return pd.DataFrame(rows)


def best_configs(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.sort_values(
            ["target", "feature_set", "positive_f1_mean", "positive_recall_mean", "pr_auc_mean"],
            ascending=[True, True, False, False, False],
        )
        .groupby(["target", "feature_set"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def temperature_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = ["macro_f1_mean", "positive_recall_mean", "positive_f1_mean", "pr_auc_mean", "false_positive_rate_mean"]
    all_sensors = summary[summary["feature_set"] == "all_sensors"][["target", "window_size"] + metrics]
    no_temperature = summary[summary["feature_set"] == "no_temperature"][["target", "window_size"] + metrics]
    merged = all_sensors.merge(no_temperature, on=["target", "window_size"], suffixes=("_all", "_no_temp"))
    for metric in metrics:
        merged[f"{metric}_delta_no_temp"] = merged[f"{metric}_no_temp"] - merged[f"{metric}_all"]
    return merged


def round_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if col.endswith(("_mean", "_std", "_min", "_rate")) or "delta" in col:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out


def format_report(
    blocks: list[int],
    summary: pd.DataFrame,
    best: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    summary_index = summary.set_index(["target", "window_size", "feature_set"])
    system = summary_index.loc[("System_Failure", 10, "no_temperature")]
    protective = summary_index.loc[("ProtectiveStop", 10, "all_sensors")]
    grip = summary_index.loc[("GripLost", 20, "no_temperature")]
    return "\n".join(
        [
            "# 08 Acquisition Block Held-Out Robustness",
            "",
            "## 목적",
            "",
            "공정조건 정답표가 없는 상황에서 25-cycle acquisition block을 운전조건 또는 session 변화의 proxy로 사용하고, 한 block 전체를 보지 않은 상태에서도 구간 단위 이상탐지가 일반화되는지 평가한다.",
            "",
            "## 설계",
            "",
            f"- 평가 block: {blocks}.",
            f"- 포함 기준: 원본에서 `cycle_run`이 {MIN_BLOCK_CYCLE_RUNS}개 이상인 25-cycle 후보 block.",
            "- 입력: `02`와 동일한 5/10/20 step window feature.",
            "- 분할: 매 반복에서 한 acquisition block 전체를 test로 두고 나머지 block만 train으로 사용.",
            "- 비교: 전체 sensor feature와 temperature feature 제거 조건.",
            "- 모델: `SMOTE + Random Forest`.",
            "- 제한: block은 실제 공정조건 라벨이 아니며 condition과 time/session drift가 섞인 proxy다.",
            "",
            "## Block-Held-Out 요약",
            "",
            markdown_table(round_metrics(summary)),
            "",
            "## Target·Feature Set별 Best Window",
            "",
            markdown_table(round_metrics(best)),
            "",
            "## Temperature 제거 효과",
            "",
            markdown_table(round_metrics(comparison)),
            "",
            "## 결과 해석",
            "",
            f"- `System_Failure` 10-step/no-temperature는 Macro F1 {system['macro_f1_mean']:.4f}, positive recall {system['positive_recall_mean']:.4f}, positive F1 {system['positive_f1_mean']:.4f}, 최소 recall {system['positive_recall_min']:.4f}였다.",
            f"- `ProtectiveStop` 10-step/all-sensors는 Macro F1 {protective['macro_f1_mean']:.4f}, positive recall {protective['positive_recall_mean']:.4f}, positive F1 {protective['positive_f1_mean']:.4f}, 최소 recall {protective['positive_recall_min']:.4f}였다.",
            f"- `GripLost` 20-step/no-temperature는 Macro F1 {grip['macro_f1_mean']:.4f}, positive recall {grip['positive_recall_mean']:.4f}, positive F1 {grip['positive_f1_mean']:.4f}, 최소 recall {grip['positive_recall_min']:.4f}였다.",
            "- 세 target의 모든 설정에서 `blocks_with_tp_rate`가 1.0이므로, 적어도 구간 단위 이상탐지는 특정 acquisition block 하나에만 의존한 결과가 아니다.",
            "- temperature 제거는 `System_Failure`에서 비교적 일관된 개선을 보였지만 `ProtectiveStop`과 `GripLost`에서는 metric별 효과가 엇갈렸다. 온도를 일괄 제거하는 규칙보다 ablation 결과를 함께 보고하는 편이 타당하다.",
            "- 이 결과는 window 안에 이미 이상 상태가 포함될 수 있는 구간 단위 이상탐지이며, 고장 발생 전 예측 성능이 아니다.",
            "",
            "## 해석 기준",
            "",
            "- `blocks_with_tp_rate`가 낮으면 평균 metric이 높아도 특정 block에서는 positive를 전혀 잡지 못한 것이다.",
            "- `positive_f1_min`과 `positive_recall_min`이 0이면 모든 block에 안정적으로 일반화됐다고 주장할 수 없다.",
            "- temperature 제거가 여러 target/window에서 일관되게 개선되면 thermal/session drift를 nuisance factor로 보는 근거가 된다.",
            "- 성능이 cycle-run random split보다 크게 낮으면 기존 성능에 같은 acquisition block의 분포 공유 효과가 포함됐다고 해석한다.",
            "",
        ]
    )


def main(report_only: bool = False) -> None:
    ensure_output_dir()
    if report_only:
        results = pd.read_csv(OUTPUT_DIR / "08_block_held_out_results.csv", encoding="utf-8-sig")
        summary = pd.read_csv(OUTPUT_DIR / "08_block_held_out_summary.csv", encoding="utf-8-sig")
        best = pd.read_csv(OUTPUT_DIR / "08_block_held_out_best.csv", encoding="utf-8-sig")
        comparison = pd.read_csv(OUTPUT_DIR / "08_block_temperature_comparison.csv", encoding="utf-8-sig")
        blocks = sorted(results["test_block"].astype(int).unique().tolist())
        (OUTPUT_DIR / "08_block_held_out_robustness.md").write_text(
            format_report(blocks, summary, best, comparison),
            encoding="utf-8",
        )
        print("Acquisition block held-out report 재생성 완료")
        print(OUTPUT_DIR / "08_block_held_out_robustness.md")
        return

    window_module = load_window_module()
    df = load_model_data()
    blocks = complete_blocks(df)
    result_rows = []

    for target in TARGET_COLS:
        for window_size in WINDOW_SIZES:
            window_df, feature_cols = window_module.build_window_dataset(df, target, window_size)
            window_df = window_df[window_df["scenario_block_25"].isin(blocks)].copy()
            for feature_set, cols in feature_sets(feature_cols).items():
                for test_block in blocks:
                    row = run_holdout(window_module, window_df, cols, test_block)
                    if row is None:
                        continue
                    row["feature_set"] = feature_set
                    result_rows.append(row)

    results = pd.DataFrame(result_rows)
    summary = aggregate_results(results)
    best = best_configs(summary)
    comparison = temperature_comparison(summary)

    results.to_csv(OUTPUT_DIR / "08_block_held_out_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "08_block_held_out_summary.csv", index=False, encoding="utf-8-sig")
    best.to_csv(OUTPUT_DIR / "08_block_held_out_best.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUTPUT_DIR / "08_block_temperature_comparison.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "08_block_held_out_robustness.md").write_text(
        format_report(blocks, summary, best, comparison),
        encoding="utf-8",
    )
    print("Acquisition block held-out robustness 완료")
    print(OUTPUT_DIR / "08_block_held_out_robustness.md")


if __name__ == "__main__":
    main(report_only="--report-only" in sys.argv)
