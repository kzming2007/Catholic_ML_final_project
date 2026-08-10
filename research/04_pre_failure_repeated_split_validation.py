from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from common import ensure_output_dir, load_model_data, markdown_table


RANDOM_STATE = 42
REPEAT_COUNT = 30
MODEL_NAME = "rf_smote"

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


def run_split(pre_failure_module, window_df: pd.DataFrame, feature_cols: list[str], seed: int, train_index, test_index) -> dict[str, float | int | str]:
    X = window_df[feature_cols]
    y = window_df["label"].astype(int)

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model = pre_failure_module.build_model(MODEL_NAME, y_train)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = pre_failure_module.evaluate_binary(y_test, y_pred, y_score)

    result = {
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
    result.update(metrics)
    return result


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    metric_cols = ["positive_recall", "positive_precision", "positive_f1", "macro_f1", "pr_auc", "roc_auc"]

    for keys, group in results.groupby(["target", "window_size", "prediction_horizon", "model"], sort=False):
        target, window_size, horizon, model = keys
        row: dict[str, float | int | str] = {
            "target": target,
            "window_size": int(window_size),
            "prediction_horizon": int(horizon),
            "model": model,
            "valid_splits": int(len(group)),
            "test_positive_count_mean": float(group["positive_count"].mean()),
            "test_positive_count_min": int(group["positive_count"].min()),
            "test_positive_count_max": int(group["positive_count"].max()),
            "runs_with_tp_rate": float((group["tp"] > 0).mean()),
        }
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
            row[f"{metric}_median"] = float(group[metric].median())
        rows.append(row)

    return pd.DataFrame(rows)


def format_markdown(results: pd.DataFrame, summary: pd.DataFrame) -> str:
    summary_display = summary.copy()
    for col in summary_display.columns:
        if col.endswith(("_mean", "_std", "_median", "_rate")):
            summary_display[col] = summary_display[col].astype(float).round(4)

    result_display_cols = [
        "target",
        "window_size",
        "prediction_horizon",
        "split_seed",
        "test_windows",
        "positive_count",
        "positive_rate",
        "positive_recall",
        "positive_precision",
        "positive_f1",
        "macro_f1",
        "pr_auc",
        "roc_auc",
        "tp",
        "fp",
        "fn",
    ]
    result_display = results[result_display_cols].copy()
    for col in [
        "positive_rate",
        "positive_recall",
        "positive_precision",
        "positive_f1",
        "macro_f1",
        "pr_auc",
        "roc_auc",
    ]:
        result_display[col] = result_display[col].astype(float).round(4)

    lines = [
        "# 04 Pre-Failure Repeated Split Validation",
        "",
        "## 범위",
        "",
        f"- Repeats: target별 유효 `cycle_run_group` split {REPEAT_COUNT}회.",
        f"- Model: `{MODEL_NAME}`.",
        "- Configs: `03_pre_failure_window_baseline.py`에서 default threshold 기준 positive F1이 가장 의미 있었던 target별 대표 조합.",
        "- 목적: 단일 split 결과가 우연인지 확인하고, `System_Failure` 통합 타깃과 개별 고장 타깃의 안정성을 비교한다.",
        "",
        "## 반복 요약",
        "",
        markdown_table(summary_display),
        "",
        "## Split별 결과",
        "",
        markdown_table(result_display),
        "",
        "## 해석 메모",
        "",
        "- `runs_with_tp_rate`는 반복 split 중 positive를 하나 이상 맞힌 비율이다.",
        "- 평균 positive F1과 `runs_with_tp_rate`가 모두 낮으면, 해당 설정은 default threshold 기준 사전 경고 모델로 해석하기 어렵다.",
        "- `System_Failure`는 통합 타깃이라 `ProtectiveStop`과 `GripLost`의 다른 시간 패턴이 섞일 수 있다.",
        "- 반복 검증 이후에도 개별 타깃이 상대적으로 안정적이면, 연구 본문에서는 통합 타깃보다 고장 유형별 pre-failure 분석을 강조한다.",
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
            result_rows.append(run_split(pre_failure_module, window_df, feature_cols, seed, train_index, test_index))

    results = pd.DataFrame(result_rows)
    summary = aggregate_results(results)

    results.to_csv(out_dir / "04_pre_failure_repeated_split_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "04_pre_failure_repeated_split_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "04_pre_failure_repeated_split_results.md").write_text(
        format_markdown(results, summary),
        encoding="utf-8",
    )

    print("Pre-failure repeated split validation 완료")
    print(out_dir / "04_pre_failure_repeated_split_results.md")


if __name__ == "__main__":
    main()
