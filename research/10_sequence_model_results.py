from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from common import OUTPUT_DIR, ensure_output_dir, markdown_table


DEEP_PREDICTION_PATH = OUTPUT_DIR / "10_sequence_window_predictions.csv"
RF_PREDICTION_PATH = OUTPUT_DIR / "09_event_level_window_predictions.csv"


def binary_metrics(y_true: pd.Series, prediction: pd.Series, score: pd.Series) -> dict[str, float]:
    return {
        "window_macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "window_positive_precision": float(precision_score(y_true, prediction, zero_division=0)),
        "window_positive_recall": float(recall_score(y_true, prediction, zero_division=0)),
        "window_positive_f1": float(f1_score(y_true, prediction, zero_division=0)),
        "window_pr_auc": float(average_precision_score(y_true, score)),
    }


def cycle_results(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["target", "model", "seed", "test_block", "cycle_run"]
    for group_keys, group in predictions.groupby(keys, sort=False):
        target, model, seed, test_block, cycle_run = group_keys
        positive_mask = group["label"].eq(1)
        detected_mask = positive_mask & group["prediction"].eq(1)
        true_event = bool(positive_mask.any())
        rows.append(
            {
                "target": target,
                "model": model,
                "seed": int(seed),
                "test_block": int(test_block),
                "cycle": int(group["cycle"].iloc[0]),
                "cycle_run": int(cycle_run),
                "true_event_cycle": int(true_event),
                "event_detected_on_positive_window": int(detected_mask.any()) if true_event else np.nan,
                "normal_cycle_false_alarm": int(group["prediction"].any()) if not true_event else np.nan,
                "window_count": int(len(group)),
                "predicted_positive_window_count": int(group["prediction"].sum()),
            }
        )
    return pd.DataFrame(rows)


def block_results(predictions: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["target", "model", "seed", "test_block"]
    for group_keys, group in predictions.groupby(keys, sort=False):
        target, model, seed, test_block = group_keys
        cycle_group = cycles[
            cycles["target"].eq(target)
            & cycles["model"].eq(model)
            & cycles["seed"].eq(seed)
            & cycles["test_block"].eq(test_block)
        ]
        event_cycles = cycle_group[cycle_group["true_event_cycle"].eq(1)]
        normal_cycles = cycle_group[cycle_group["true_event_cycle"].eq(0)]
        detected_events = int(event_cycles["event_detected_on_positive_window"].sum())
        false_alarm_cycles = int(normal_cycles["normal_cycle_false_alarm"].sum())
        row = {
            "target": target,
            "model": model,
            "seed": int(seed),
            "test_block": int(test_block),
            "event_cycle_count": int(len(event_cycles)),
            "detected_event_cycle_count": detected_events,
            "event_cycle_recall": float(detected_events / len(event_cycles)) if len(event_cycles) else np.nan,
            "normal_cycle_count": int(len(normal_cycles)),
            "false_alarm_cycle_count": false_alarm_cycles,
            "normal_cycle_false_alarm_rate": float(false_alarm_cycles / len(normal_cycles)) if len(normal_cycles) else np.nan,
        }
        row.update(binary_metrics(group["label"], group["prediction"], group["score"]))
        rows.append(row)
    return pd.DataFrame(rows)


def seed_results(predictions: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["target", "model", "seed"], sort=False):
        target, model, seed = keys
        block_group = blocks[
            blocks["target"].eq(target) & blocks["model"].eq(model) & blocks["seed"].eq(seed)
        ]
        event_total = int(block_group["event_cycle_count"].sum())
        detected_total = int(block_group["detected_event_cycle_count"].sum())
        normal_total = int(block_group["normal_cycle_count"].sum())
        false_alarm_total = int(block_group["false_alarm_cycle_count"].sum())
        row = {
            "target": target,
            "model": model,
            "seed": int(seed),
            "test_blocks": int(block_group["test_block"].nunique()),
            "event_cycle_count": event_total,
            "event_cycle_recall": float(detected_total / event_total) if event_total else np.nan,
            "event_cycle_recall_min_block": float(block_group["event_cycle_recall"].min()),
            "normal_cycle_count": normal_total,
            "normal_cycle_false_alarm_rate": float(false_alarm_total / normal_total) if normal_total else np.nan,
            "normal_cycle_false_alarm_rate_max_block": float(block_group["normal_cycle_false_alarm_rate"].max()),
        }
        row.update(binary_metrics(group["label"], group["prediction"], group["score"]))
        rows.append(row)
    return pd.DataFrame(rows)


def model_summary(seeds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "event_cycle_recall",
        "event_cycle_recall_min_block",
        "normal_cycle_false_alarm_rate",
        "normal_cycle_false_alarm_rate_max_block",
        "window_macro_f1",
        "window_positive_precision",
        "window_positive_recall",
        "window_positive_f1",
        "window_pr_auc",
    ]
    for keys, group in seeds.groupby(["target", "model"], sort=False):
        target, model = keys
        row: dict[str, float | int | str] = {
            "target": target,
            "model": model,
            "seeds": int(group["seed"].nunique()),
            "test_blocks_per_seed": int(group["test_blocks"].min()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def load_predictions() -> pd.DataFrame:
    deep = pd.read_csv(DEEP_PREDICTION_PATH, encoding="utf-8-sig")
    deep["seed"] = deep["seed"].astype(int)

    rf = pd.read_csv(RF_PREDICTION_PATH, encoding="utf-8-sig")
    rf = rf[rf["feature_set"].eq("all_sensors")].copy()
    rf["model"] = "rf_window_features"
    rf["seed"] = 42
    keep = [
        "target", "model", "seed", "test_block", "cycle", "cycle_run", "cycle_occurrence",
        "scenario_block_25", "window_start_step", "window_end_step", "label", "score", "prediction",
    ]
    return pd.concat([deep[keep], rf[keep]], ignore_index=True)


def round_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    return out


def format_report(summary: pd.DataFrame, training_runs: pd.DataFrame) -> str:
    columns = [
        "target", "model", "seeds", "event_cycle_recall_mean", "event_cycle_recall_min_block_mean",
        "normal_cycle_false_alarm_rate_mean", "normal_cycle_false_alarm_rate_max_block_mean",
        "window_macro_f1_mean", "window_positive_f1_mean", "window_pr_auc_mean",
    ]
    deep_runs = training_runs[training_runs["model"].isin(["1d_cnn", "lstm"])]
    return "\n".join(
        [
            "# 10 Sequence Model Comparison",
            "",
            "## 고정 설계",
            "",
            "- 입력: `cycle_run` 경계를 넘지 않는 10-step × 26-feature sequence.",
            "- 외부 평가: 9개 후보 block을 한 번씩 test로 두는 held-out 평가.",
            "- 내부 검증: outer train block 중 다음 순서의 1개 block으로 epoch를 선택한 뒤, 8개 outer train block 전체로 해당 epoch만큼 재학습.",
            "- 모델: 고정 소형 1D CNN, 단층 LSTM, 비교 기준 `SMOTE + Random Forest` window summary.",
            "- Deep learning: class-weighted BCE, Adam, threshold `score > 0.50`, seeds 42/43/44.",
            "- 전체 sensor를 주 설정으로 사용하며 test block은 scaling, epoch 선택, threshold 선택에 사용하지 않는다.",
            "- Window 지표는 seed별로 9개 held-out prediction을 합쳐 계산한 뒤 seed 평균을 낸다. Random Forest는 고정 1회 결과다.",
            "",
            "## 결과 요약",
            "",
            markdown_table(round_metrics(summary[columns])),
            "",
            "## 실행 범위",
            "",
            f"- Deep learning 학습 run: {len(deep_runs)}개.",
            f"- PyTorch: {deep_runs['torch_version'].iloc[0]}, CUDA: {deep_runs['cuda_version'].iloc[0]}, device: {deep_runs['device'].iloc[0]}.",
            f"- 총 학습 시간: {deep_runs['elapsed_seconds'].sum():.1f}초.",
            "",
            "## 결론",
            "",
            "- Random Forest는 세 target 모두에서 두 deep learning 모델보다 정상 cycle 오경보율이 낮고, pooled window positive F1과 PR-AUC가 높았다.",
            "- 1D CNN과 LSTM도 높은 event cycle recall을 보였지만, 정상 cycle 오경보가 누적되어 Random Forest를 대체할 근거는 확인되지 않았다.",
            "- 현재 규모의 공개 데이터에서는 짧은 구간의 통계적 요약과 tree ensemble이 효과적인 기준선이라는 결과로 해석한다.",
            "- 후속 확장은 모델 규모 확대보다 오류가 집중된 held-out block과 false alarm 구간을 먼저 분석하는 것이 타당하다.",
            "",
            "## 해석 제한",
            "",
            "- Random Forest는 10-step을 통계량으로 정제한 입력이고, 1D CNN/LSTM은 동일 구간의 step 순서를 직접 입력받는다.",
            "- 이 실험은 이상이 이미 포함된 window의 구간 단위 탐지 비교이며 pre-failure 예측이 아니다.",
            "- 모델 구조와 학습 설정은 사전 고정했으며 test 결과를 이용한 architecture 또는 threshold 선택은 수행하지 않는다.",
            "- 후보 block은 실제 공정조건 정답표가 아니라 cycle 번호 기반 proxy다.",
            "- `09` 보고서의 window 지표는 block별 평균이고 이 표는 held-out prediction pooled 지표이므로 수치 집계 방식이 다르다. Random Forest 원본 prediction은 동일하다.",
            "",
        ]
    )


def main() -> None:
    ensure_output_dir()
    predictions = load_predictions()
    cycles = cycle_results(predictions)
    blocks = block_results(predictions, cycles)
    seeds = seed_results(predictions, blocks)
    summary = model_summary(seeds)
    training_runs = pd.read_csv(OUTPUT_DIR / "10_sequence_training_runs.csv", encoding="utf-8-sig")

    cycles.to_csv(OUTPUT_DIR / "10_sequence_cycle_results.csv", index=False, encoding="utf-8-sig")
    blocks.to_csv(OUTPUT_DIR / "10_sequence_block_results.csv", index=False, encoding="utf-8-sig")
    seeds.to_csv(OUTPUT_DIR / "10_sequence_seed_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "10_sequence_model_summary.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "10_sequence_model_comparison.md").write_text(
        format_report(summary, training_runs),
        encoding="utf-8",
    )
    print(OUTPUT_DIR / "10_sequence_model_comparison.md")


if __name__ == "__main__":
    main()
