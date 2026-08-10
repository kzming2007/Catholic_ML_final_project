from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from common import MODEL_FEATURE_COLS, TARGET_COLS, clean_for_model, ensure_output_dir, load_model_data, markdown_table


RANDOM_STATE = 42
WINDOW_SIZES = [5, 10, 20]
PREDICTION_HORIZONS = [3, 5, 10]
MODEL_NAMES = ["rf_plain", "rf_smote"]


@dataclass(frozen=True)
class SplitData:
    name: str
    train_index: np.ndarray
    test_index: np.ndarray
    seed: int


def build_model(model_name: str, y_train: pd.Series):
    if model_name == "rf_plain":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    if model_name == "rf_smote":
        minority_count = int(y_train.value_counts().min())
        if minority_count < 2:
            raise ValueError("SMOTE requires at least two minority samples.")
        k_neighbors = min(5, minority_count - 1)
        return ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def make_feature_names(feature_cols: list[str]) -> list[str]:
    stats = ["mean", "std", "min", "max", "range", "delta", "slope"]
    return [f"{col}__{stat}" for col in feature_cols for stat in stats]


def summarize_window(values: np.ndarray, feature_cols: list[str]) -> dict[str, float]:
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=0)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    ranges = maxs - mins
    deltas = values[-1] - values[0]
    slopes = deltas / max(len(values) - 1, 1)

    row: dict[str, float] = {}
    for idx, col in enumerate(feature_cols):
        row[f"{col}__mean"] = float(means[idx])
        row[f"{col}__std"] = float(stds[idx])
        row[f"{col}__min"] = float(mins[idx])
        row[f"{col}__max"] = float(maxs[idx])
        row[f"{col}__range"] = float(ranges[idx])
        row[f"{col}__delta"] = float(deltas[idx])
        row[f"{col}__slope"] = float(slopes[idx])
    return row


def build_pre_failure_window_dataset(
    df: pd.DataFrame,
    target_col: str,
    window_size: int,
    prediction_horizon: int,
) -> tuple[pd.DataFrame, list[str], dict[str, int | float | str]]:
    clean = clean_for_model(df, MODEL_FEATURE_COLS, target_col).copy()
    clean["Timestamp"] = pd.to_datetime(clean["Timestamp"], errors="coerce")
    clean = clean.dropna(subset=["Timestamp"]).sort_values(["cycle_run", "Timestamp"]).reset_index(drop=True)

    window_rows: list[dict[str, object]] = []
    skipped_post_trigger = 0
    skipped_short_cycle = 0
    cycle_runs_with_event = 0
    cycle_runs_without_event = 0
    cycle_runs_with_pre_event_window = 0

    for cycle_run, group in clean.groupby("cycle_run", sort=True):
        group = group.reset_index(drop=True)
        if len(group) < window_size:
            skipped_short_cycle += 1
            continue

        cycle = int(group["cycle"].iloc[0])
        cycle_occurrence = int(group["cycle_occurrence"].iloc[0])

        feature_values = group[MODEL_FEATURE_COLS].to_numpy(dtype=float)
        target_values = group[target_col].astype(int).to_numpy()
        timestamp_values = group["Timestamp"].astype(str).to_numpy()

        positive_steps = np.flatnonzero(target_values == 1)
        first_positive_step = int(positive_steps[0]) if len(positive_steps) else None
        if first_positive_step is None:
            cycle_runs_without_event += 1
        else:
            cycle_runs_with_event += 1

        cycle_run_has_pre_event_window = False
        for start in range(0, len(group) - window_size + 1):
            end = start + window_size
            window_end_step = end - 1

            if first_positive_step is not None and window_end_step >= first_positive_step:
                skipped_post_trigger += 1
                continue

            steps_to_event = None
            label = 0
            if first_positive_step is not None:
                steps_to_event = first_positive_step - window_end_step
                label = int(1 <= steps_to_event <= prediction_horizon)
                cycle_run_has_pre_event_window = True

            row = summarize_window(feature_values[start:end], MODEL_FEATURE_COLS)
            row.update(
                {
                    "target": target_col,
                    "window_size": int(window_size),
                    "prediction_horizon": int(prediction_horizon),
                    "cycle": cycle,
                    "cycle_run": int(cycle_run),
                    "cycle_occurrence": cycle_occurrence,
                    "scenario_block_25": int((cycle - 1) // 25 + 1),
                    "window_start_step": int(start),
                    "window_end_step": int(window_end_step),
                    "window_start_timestamp": timestamp_values[start],
                    "window_end_timestamp": timestamp_values[window_end_step],
                    "first_positive_step": first_positive_step,
                    "steps_to_event": steps_to_event,
                    "label": label,
                }
            )
            window_rows.append(row)

        if cycle_run_has_pre_event_window:
            cycle_runs_with_pre_event_window += 1

    window_df = pd.DataFrame(window_rows)
    feature_names = make_feature_names(MODEL_FEATURE_COLS)
    audit = {
        "target": target_col,
        "window_size": int(window_size),
        "prediction_horizon": int(prediction_horizon),
        "cycle_runs_with_event": int(cycle_runs_with_event),
        "cycle_runs_without_event": int(cycle_runs_without_event),
        "cycle_runs_with_pre_event_window": int(cycle_runs_with_pre_event_window),
        "skipped_short_cycle": int(skipped_short_cycle),
        "skipped_post_trigger_windows": int(skipped_post_trigger),
    }
    return window_df, feature_names, audit


def make_cycle_group_split(df: pd.DataFrame) -> SplitData:
    y = df["label"].astype(int)
    groups = df["cycle_run"].astype(int)

    for seed in range(RANDOM_STATE, RANDOM_STATE + 200):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_index, test_index = next(splitter.split(df, y, groups=groups))
        if y.iloc[train_index].nunique() == 2 and y.iloc[test_index].nunique() == 2:
            return SplitData("cycle_run_group", train_index, test_index, seed)

    raise ValueError("Could not create a cycle-run group split with both classes in train/test.")


def evaluate_binary(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics: dict[str, float | int] = {
        "test_windows": int(len(y_true)),
        "positive_count": int((y_true == 1).sum()),
        "positive_rate": float((y_true == 1).mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "positive_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "positive_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if y_score is not None and y_true.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
    return metrics


def run_one(window_df: pd.DataFrame, feature_cols: list[str], split: SplitData, model_name: str) -> dict[str, float | int | str]:
    X = window_df[feature_cols]
    y = window_df["label"].astype(int)

    X_train = X.iloc[split.train_index]
    X_test = X.iloc[split.test_index]
    y_train = y.iloc[split.train_index]
    y_test = y.iloc[split.test_index]

    model = build_model(model_name, y_train)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    result = {
        "target": str(window_df["target"].iloc[0]),
        "window_size": int(window_df["window_size"].iloc[0]),
        "prediction_horizon": int(window_df["prediction_horizon"].iloc[0]),
        "split": split.name,
        "split_seed": int(split.seed),
        "model": model_name,
        "train_windows": int(len(y_train)),
        "train_positive_count": int((y_train == 1).sum()),
        "train_positive_rate": float((y_train == 1).mean()),
        "train_cycle_runs": int(window_df.iloc[split.train_index]["cycle_run"].nunique()),
        "test_cycle_runs": int(window_df.iloc[split.test_index]["cycle_run"].nunique()),
        "train_cycle_ids": int(window_df.iloc[split.train_index]["cycle"].nunique()),
        "test_cycle_ids": int(window_df.iloc[split.test_index]["cycle"].nunique()),
    }
    result.update(evaluate_binary(y_test, y_pred, y_score))
    return result


def summarize_window_dataset(window_df: pd.DataFrame, audit: dict[str, int | float | str]) -> dict[str, int | float | str]:
    summary = dict(audit)
    summary.update(
        {
            "windows": int(len(window_df)),
            "cycle_runs": int(window_df["cycle_run"].nunique()),
            "cycle_ids": int(window_df["cycle"].nunique()),
            "positive_windows": int(window_df["label"].sum()),
            "positive_rate": float(window_df["label"].mean()),
        }
    )
    return summary


def make_best_summary(results: pd.DataFrame) -> pd.DataFrame:
    best = (
        results[results["model"] == "rf_smote"]
        .sort_values(
            ["target", "positive_f1", "positive_recall", "pr_auc", "macro_f1"],
            ascending=[True, False, False, False, False],
        )
        .groupby("target", as_index=False)
        .head(1)
    )
    return best[
        [
            "target",
            "window_size",
            "prediction_horizon",
            "model",
            "test_windows",
            "positive_count",
            "positive_rate",
            "macro_f1",
            "positive_recall",
            "positive_precision",
            "positive_f1",
            "pr_auc",
            "roc_auc",
        ]
    ].copy()


def format_results_markdown(results: pd.DataFrame, summary: pd.DataFrame, best: pd.DataFrame) -> str:
    summary_display = summary.copy()
    for col in ["positive_rate"]:
        summary_display[col] = summary_display[col].astype(float).round(4)

    display_cols = [
        "target",
        "window_size",
        "prediction_horizon",
        "model",
        "test_windows",
        "positive_count",
        "positive_rate",
        "accuracy",
        "macro_f1",
        "positive_recall",
        "positive_precision",
        "positive_f1",
        "roc_auc",
        "pr_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    result_display = results[display_cols].copy()
    for col in [
        "positive_rate",
        "accuracy",
        "macro_f1",
        "positive_recall",
        "positive_precision",
        "positive_f1",
        "roc_auc",
        "pr_auc",
    ]:
        result_display[col] = result_display[col].astype(float).round(4)

    best_display = best.copy()
    for col in ["positive_rate", "macro_f1", "positive_recall", "positive_precision", "positive_f1", "pr_auc", "roc_auc"]:
        best_display[col] = best_display[col].astype(float).round(4)

    lines = [
        "# 03 Pre-Failure Window Baseline",
        "",
        "## 범위",
        "",
        "- Focus: 고장 발생 전 예측에 가까운 분류 실험.",
        "- Target: `System_Failure`, `ProtectiveStop`, `GripLost`.",
        "- Window: 시간상 연속한 `cycle_run` 경계를 넘지 않는 5, 10, 20 step sliding window.",
        "- Prediction horizon: first positive step 이전 3, 5, 10 step.",
        "- Label: window가 first positive step 전에 끝나고, 그 종료 시점이 horizon 안에 있으면 positive.",
        "- Exclusion: first positive step 이후 또는 그 시점을 포함하는 window는 모두 제외.",
        "- Feature: mean, std, min, max, range, first-last delta, simple slope.",
        "- Split: 같은 `cycle_run`이 train/test에 동시에 들어가지 않는 `cycle_run_group` split.",
        "- Models: class weighting을 적용한 Random Forest, SMOTE + Random Forest.",
        "",
        "## Window 데이터 요약",
        "",
        markdown_table(summary_display),
        "",
        "## Best 결과 요약",
        "",
        markdown_table(best_display),
        "",
        "## 전체 결과",
        "",
        markdown_table(result_display),
        "",
        "## 해석 메모",
        "",
        "- 이 실험은 이미 `True`가 된 이후의 센서값을 입력에서 제외한다.",
        "- 원본 `cycle` ID가 재등장하더라도 first positive 탐색과 window 생성은 각 `cycle_run` 안에서만 수행한다.",
        "- 따라서 `02_window_feature_baseline.py`의 구간 단위 이상탐지 결과와 직접 성능 비교하지 않는다.",
        "- positive window는 고장 직전 horizon 안의 구간이며, normal cycle과 event 이전의 먼 구간은 negative로 둔다.",
        "- 데이터가 작고 positive window가 제한되므로, 단일 split 결과는 후속 반복 split 또는 cross-validation으로 확인해야 한다.",
        "- default threshold에서 positive recall이 0에 가까운 조합은, ROC-AUC나 PR-AUC가 높더라도 실제 경고 모델로 해석하지 않는다.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    out_dir = ensure_output_dir()
    df_all = load_model_data()

    result_rows = []
    summary_rows = []
    for target in TARGET_COLS:
        for window_size in WINDOW_SIZES:
            for horizon in PREDICTION_HORIZONS:
                window_df, feature_cols, audit = build_pre_failure_window_dataset(df_all, target, window_size, horizon)
                if window_df.empty or window_df["label"].nunique() < 2:
                    continue

                summary_rows.append(summarize_window_dataset(window_df, audit))
                split = make_cycle_group_split(window_df)
                for model_name in MODEL_NAMES:
                    result_rows.append(run_one(window_df, feature_cols, split, model_name))

    results = pd.DataFrame(result_rows)
    summary = pd.DataFrame(summary_rows)
    best = make_best_summary(results)

    results.to_csv(out_dir / "03_pre_failure_window_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "03_pre_failure_window_summary.csv", index=False, encoding="utf-8-sig")
    best.to_csv(out_dir / "03_pre_failure_window_best.csv", index=False, encoding="utf-8-sig")
    (out_dir / "03_pre_failure_window_results.md").write_text(
        format_results_markdown(results, summary, best),
        encoding="utf-8",
    )

    print("Pre-failure window baseline 완료")
    print(out_dir / "03_pre_failure_window_results.md")


if __name__ == "__main__":
    main()
