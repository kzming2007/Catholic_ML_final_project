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
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from common import MODEL_FEATURE_COLS, TARGET_COLS, clean_for_model, ensure_output_dir, load_model_data, markdown_table


RANDOM_STATE = 42


@dataclass(frozen=True)
class SplitData:
    name: str
    train_index: np.ndarray
    test_index: np.ndarray


def build_model(model_name: str):
    if model_name == "rf_plain":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    if model_name == "rf_smote":
        return ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
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


def make_splits(df: pd.DataFrame, target_col: str) -> list[SplitData]:
    y = df[target_col].astype(int)
    groups = df["cycle"].astype(int)

    random_train, random_test = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    group_train, group_test = next(gss.split(df, y, groups=groups))

    return [
        SplitData("random_stratified", random_train, random_test),
        SplitData("cycle_group", group_train, group_test),
    ]


def evaluate_binary(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, float | int]:
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    metrics: dict[str, float | int] = {
        "test_rows": int(len(y_true)),
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


def run_one(df: pd.DataFrame, target_col: str, split: SplitData, model_name: str) -> dict[str, float | int | str]:
    X = df[MODEL_FEATURE_COLS]
    y = df[target_col].astype(int)

    X_train = X.iloc[split.train_index]
    X_test = X.iloc[split.test_index]
    y_train = y.iloc[split.train_index]
    y_test = y.iloc[split.test_index]

    model = build_model(model_name)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    result = {
        "target": target_col,
        "split": split.name,
        "model": model_name,
        "train_rows": int(len(y_train)),
        "train_positive_count": int((y_train == 1).sum()),
        "train_positive_rate": float((y_train == 1).mean()),
        "test_cycles": int(df.iloc[split.test_index]["cycle"].nunique()),
        "train_cycles": int(df.iloc[split.train_index]["cycle"].nunique()),
    }
    result.update(evaluate_binary(y_test, y_pred, y_score))
    return result


def format_results_markdown(results: pd.DataFrame) -> str:
    display_cols = [
        "target",
        "split",
        "model",
        "test_rows",
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
    rounded = results[display_cols].copy()
    for col in ["positive_rate", "accuracy", "macro_f1", "positive_recall", "positive_precision", "positive_f1", "roc_auc", "pr_auc"]:
        rounded[col] = rounded[col].astype(float).round(4)

    lines = [
        "# 01 Baseline 재현",
        "",
        "## 범위",
        "",
        "- Feature set: 원본 센서 feature 19개 + current-speed power feature 6개 + absolute current sum.",
        "- Models: class weighting을 적용한 Random Forest, SMOTE + Random Forest.",
        "- Splits: row 단위 stratified random split, cycle 기준 group split.",
        "- Targets: System_Failure, ProtectiveStop, GripLost.",
        "",
        "## 결과",
        "",
        markdown_table(rounded),
        "",
        "## 해석 메모",
        "",
        "- random split보다 cycle group split에서 성능이 낮아지면 row 단위 random split이 낙관적이었을 가능성을 시사한다.",
        "- `System_Failure`는 넓은 이상탐지 타깃으로 유용하지만, 고장 유형별 패턴을 보려면 ProtectiveStop과 GripLost를 별도로 분석해야 한다.",
        "- 모든 타깃이 불균형하므로 accuracy보다 PR-AUC와 positive-class recall을 더 중요하게 본다.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    out_dir = ensure_output_dir()
    df_all = load_model_data()

    rows = []
    for target in TARGET_COLS:
        df = clean_for_model(df_all, MODEL_FEATURE_COLS, target).reset_index(drop=True)
        splits = make_splits(df, target)
        for split in splits:
            for model_name in ["rf_plain", "rf_smote"]:
                rows.append(run_one(df, target, split, model_name))

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "01_baseline_results.csv", index=False, encoding="utf-8-sig")
    (out_dir / "01_baseline_results.md").write_text(format_results_markdown(results), encoding="utf-8")

    print("Baseline 재현 완료")
    print(out_dir / "01_baseline_results.md")


if __name__ == "__main__":
    main()
