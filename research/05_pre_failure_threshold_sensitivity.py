from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from common import ensure_output_dir, load_model_data, markdown_table


RANDOM_STATE = 42
REPEAT_COUNT = 30
MODEL_NAME = "rf_smote"
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

SELECTED_CONFIGS = [
    {"target": "System_Failure", "window_size": 5, "prediction_horizon": 10},
    {"target": "ProtectiveStop", "window_size": 10, "prediction_horizon": 3},
    {"target": "GripLost", "window_size": 5, "prediction_horizon": 3},
]


def load_pre_failure_module():
    module_path = Path(__file__).with_name("03_pre_failure_window_baseline.py")
    spec = importlib.util.spec_from_file_location("pre_failure_window_baseline", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def valid_group_splits(window_df: pd.DataFrame, repeat_count: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    y = window_df["label"].astype(int)
    groups = window_df["cycle_run"].astype(int)
    splits: list[tuple[int, np.ndarray, np.ndarray]] = []

    seed = RANDOM_STATE
    while len(splits) < repeat_count and seed < RANDOM_STATE + 1000:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_index, test_index = next(splitter.split(window_df, y, groups=groups))
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        has_both_classes = y_train.nunique() == 2 and y_test.nunique() == 2
        has_enough_smote_minority = int(y_train.value_counts().min()) >= 2
        if has_both_classes and has_enough_smote_minority:
            splits.append((seed, train_index, test_index))
        seed += 1

    if len(splits) < repeat_count:
        raise ValueError(f"Only found {len(splits)} valid splits for {repeat_count} requested repeats.")
    return splits


def evaluate_threshold(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    negative_count = int((y_true == 0).sum())

    return {
        "threshold": float(threshold),
        "test_windows": int(len(y_true)),
        "positive_count": int((y_true == 1).sum()),
        "positive_rate": float((y_true == 1).mean()),
        "positive_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "positive_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "false_positive_rate": float(fp / negative_count) if negative_count else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_split(pre_failure_module, window_df: pd.DataFrame, feature_cols: list[str], seed: int, train_index, test_index) -> list[dict[str, float | int | str]]:
    X = window_df[feature_cols]
    y = window_df["label"].astype(int)

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model = pre_failure_module.build_model(MODEL_NAME, y_train)
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]

    rows = []
    for threshold in THRESHOLDS:
        row = {
            "target": str(window_df["target"].iloc[0]),
            "window_size": int(window_df["window_size"].iloc[0]),
            "prediction_horizon": int(window_df["prediction_horizon"].iloc[0]),
            "model": MODEL_NAME,
            "split_seed": int(seed),
            "train_windows": int(len(y_train)),
            "train_positive_count": int((y_train == 1).sum()),
            "train_positive_rate": float((y_train == 1).mean()),
            "train_cycle_runs": int(window_df.iloc[train_index]["cycle_run"].nunique()),
            "test_cycle_runs": int(window_df.iloc[test_index]["cycle_run"].nunique()),
        }
        row.update(evaluate_threshold(y_test, y_score, threshold))
        rows.append(row)
    return rows


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    metric_cols = ["positive_recall", "positive_precision", "positive_f1", "macro_f1", "false_positive_rate"]

    for keys, group in results.groupby(["target", "window_size", "prediction_horizon", "model", "threshold"], sort=False):
        target, window_size, horizon, model, threshold = keys
        row: dict[str, float | int | str] = {
            "target": target,
            "window_size": int(window_size),
            "prediction_horizon": int(horizon),
            "model": model,
            "threshold": float(threshold),
            "valid_splits": int(len(group)),
            "test_positive_count_mean": float(group["positive_count"].mean()),
            "runs_with_tp_rate": float((group["tp"] > 0).mean()),
        }
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
            row[f"{metric}_median"] = float(group[metric].median())
        rows.append(row)

    return pd.DataFrame(rows)


def choose_recommended_thresholds(summary: pd.DataFrame) -> pd.DataFrame:
    # Prefer thresholds that improve positive F1 without pushing false positives too far.
    candidates = summary[summary["false_positive_rate_mean"] <= 0.10].copy()
    if candidates.empty:
        candidates = summary.copy()
    recommended = (
        candidates.sort_values(
            ["target", "positive_f1_mean", "positive_recall_mean", "false_positive_rate_mean"],
            ascending=[True, False, False, True],
        )
        .groupby("target", as_index=False)
        .head(1)
    )
    return recommended


def format_markdown(summary: pd.DataFrame, recommended: pd.DataFrame) -> str:
    summary_display = summary.copy()
    recommended_display = recommended.copy()
    for table in [summary_display, recommended_display]:
        for col in table.columns:
            if col.endswith(("_mean", "_std", "_median", "_rate")) or col == "threshold":
                table[col] = table[col].astype(float).round(4)

    lines = [
        "# 05 Pre-Failure Threshold Sensitivity",
        "",
        "## 범위",
        "",
        f"- Repeats: target별 유효 `cycle_run_group` split {REPEAT_COUNT}회.",
        f"- Model: `{MODEL_NAME}`.",
        f"- Thresholds: {', '.join(str(value) for value in THRESHOLDS)}.",
        "- 목적: default threshold 0.50에서 놓친 pre-failure positive를 threshold 조정으로 어느 정도 회수할 수 있는지 확인한다.",
        "- 추천 threshold는 평균 false positive rate가 0.10 이하인 후보 중 positive F1 평균이 가장 높은 값으로 고른다.",
        "",
        "## 추천 Threshold 요약",
        "",
        markdown_table(recommended_display),
        "",
        "## Threshold별 반복 요약",
        "",
        markdown_table(summary_display),
        "",
        "## 해석 메모",
        "",
        "- threshold를 낮추면 positive recall은 올라갈 수 있지만 false positive도 함께 증가한다.",
        "- pre-failure positive 수가 작기 때문에 threshold 결과도 반복 split 평균과 분산을 함께 봐야 한다.",
        "- `System_Failure`는 통합 타깃이라 threshold를 낮춰도 고장 유형별 해석보다 불안정할 수 있다.",
        "- 추천 threshold는 후속 실험 후보일 뿐이며, 최종 결론에는 false positive 부담을 함께 보고해야 한다.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    out_dir = ensure_output_dir()
    pre_failure_module = load_pre_failure_module()
    df_all = load_model_data()

    result_rows = []
    for config in SELECTED_CONFIGS:
        window_df, feature_cols, _audit = pre_failure_module.build_pre_failure_window_dataset(
            df_all,
            config["target"],
            config["window_size"],
            config["prediction_horizon"],
        )
        if window_df.empty or window_df["label"].nunique() < 2:
            raise ValueError(f"Invalid window dataset for config: {config}")

        for seed, train_index, test_index in valid_group_splits(window_df, REPEAT_COUNT):
            result_rows.extend(run_split(pre_failure_module, window_df, feature_cols, seed, train_index, test_index))

    results = pd.DataFrame(result_rows)
    summary = aggregate_results(results)
    recommended = choose_recommended_thresholds(summary)

    results.to_csv(out_dir / "05_pre_failure_threshold_sensitivity_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "05_pre_failure_threshold_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    recommended.to_csv(out_dir / "05_pre_failure_threshold_sensitivity_recommended.csv", index=False, encoding="utf-8-sig")
    (out_dir / "05_pre_failure_threshold_sensitivity_results.md").write_text(
        format_markdown(summary, recommended),
        encoding="utf-8",
    )

    print("Pre-failure threshold sensitivity 완료")
    print(out_dir / "05_pre_failure_threshold_sensitivity_results.md")


if __name__ == "__main__":
    main()
