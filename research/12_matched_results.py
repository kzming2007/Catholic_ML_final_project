from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from common import OUTPUT_DIR, ensure_output_dir, markdown_table


RF_PATH = OUTPUT_DIR / "12_matched_rf_window_predictions.csv"
TORCH_PATH = OUTPUT_DIR / "12_matched_torch_window_predictions.csv"
PRIMARY_VARIANTS = ["rf_19_raw_window", "1d_cnn_19_raw", "lstm_autoencoder_19_raw_q95"]
VARIANT_ORDER = [
    "rf_19_raw_window",
    "1d_cnn_19_raw",
    "lstm_autoencoder_19_raw_q90",
    "lstm_autoencoder_19_raw_q95",
    "lstm_autoencoder_19_raw_q97.5",
]


def variant_name(row: pd.Series) -> str:
    if row["model"] != "lstm_autoencoder_19_raw":
        return str(row["model"])
    quantile = float(row["threshold_quantile"])
    label = "97.5" if math.isclose(quantile, 0.975) else str(int(round(quantile * 100)))
    return f"lstm_autoencoder_19_raw_q{label}"


def load_predictions() -> pd.DataFrame:
    rf = pd.read_csv(RF_PATH, encoding="utf-8-sig")
    torch_predictions = pd.read_csv(TORCH_PATH, encoding="utf-8-sig", low_memory=False)
    predictions = pd.concat([rf, torch_predictions], ignore_index=True, sort=False)
    predictions["threshold_quantile"] = pd.to_numeric(predictions["threshold_quantile"], errors="coerce")
    predictions["model_variant"] = predictions.apply(variant_name, axis=1)
    predictions["seed"] = predictions["seed"].astype(int)
    predictions["label"] = predictions["label"].astype(int)
    predictions["prediction"] = predictions["prediction"].astype(int)
    return predictions


def binary_metrics(group: pd.DataFrame) -> dict[str, float]:
    return {
        "window_macro_f1": float(f1_score(group["label"], group["prediction"], average="macro", zero_division=0)),
        "window_positive_precision": float(precision_score(group["label"], group["prediction"], zero_division=0)),
        "window_positive_recall": float(recall_score(group["label"], group["prediction"], zero_division=0)),
        "window_positive_f1": float(f1_score(group["label"], group["prediction"], zero_division=0)),
        "window_pr_auc": float(average_precision_score(group["label"], group["score"])),
    }


def seed_cycle_results(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["target", "model_variant", "seed", "test_block", "cycle", "cycle_run", "cycle_occurrence"]
    for group_keys, group in predictions.groupby(keys, sort=False):
        target, model_variant, seed, test_block, cycle, cycle_run, cycle_occurrence = group_keys
        positive = group["label"].eq(1)
        rows.append(
            {
                "target": target,
                "model_variant": model_variant,
                "seed": int(seed),
                "test_block": int(test_block),
                "cycle": int(cycle),
                "cycle_run": int(cycle_run),
                "cycle_occurrence": int(cycle_occurrence),
                "true_event_cycle": int(positive.any()),
                "cycle_alert": int(group["prediction"].any()),
                "event_detected": int((positive & group["prediction"].eq(1)).any()),
            }
        )
    return pd.DataFrame(rows)


def seed_block_results(predictions: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["target", "model_variant", "seed", "test_block"]
    for group_keys, group in predictions.groupby(keys, sort=False):
        target, variant, seed, test_block = group_keys
        cycle_group = cycles[
            cycles["target"].eq(target)
            & cycles["model_variant"].eq(variant)
            & cycles["seed"].eq(seed)
            & cycles["test_block"].eq(test_block)
        ]
        events = cycle_group[cycle_group["true_event_cycle"].eq(1)]
        normal = cycle_group[cycle_group["true_event_cycle"].eq(0)]
        detected = int(events["event_detected"].sum())
        false_alarms = int(normal["cycle_alert"].sum())
        row = {
            "target": target,
            "model_variant": variant,
            "seed": int(seed),
            "test_block": int(test_block),
            "event_cycle_count": int(len(events)),
            "detected_event_cycle_count": detected,
            "event_cycle_recall": float(detected / len(events)) if len(events) else np.nan,
            "normal_cycle_count": int(len(normal)),
            "false_alarm_cycle_count": false_alarms,
            "normal_cycle_false_alarm_rate": float(false_alarms / len(normal)) if len(normal) else np.nan,
        }
        row.update(binary_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def seed_summary(predictions: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["target", "model_variant", "seed"], sort=False):
        target, variant, seed = keys
        block_group = blocks[
            blocks["target"].eq(target)
            & blocks["model_variant"].eq(variant)
            & blocks["seed"].eq(seed)
        ]
        event_total = int(block_group["event_cycle_count"].sum())
        detected_total = int(block_group["detected_event_cycle_count"].sum())
        normal_total = int(block_group["normal_cycle_count"].sum())
        false_alarm_total = int(block_group["false_alarm_cycle_count"].sum())
        row = {
            "target": target,
            "model_variant": variant,
            "seed": int(seed),
            "test_blocks": int(block_group["test_block"].nunique()),
            "event_cycle_count": event_total,
            "event_cycle_recall": float(detected_total / event_total),
            "event_cycle_recall_min_block": float(block_group["event_cycle_recall"].min()),
            "normal_cycle_count": normal_total,
            "normal_cycle_false_alarm_rate": float(false_alarm_total / normal_total),
            "normal_cycle_false_alarm_rate_max_block": float(
                block_group["normal_cycle_false_alarm_rate"].max()
            ),
        }
        row.update(binary_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def model_seed_summary(seeds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "event_cycle_recall", "event_cycle_recall_min_block", "normal_cycle_false_alarm_rate",
        "normal_cycle_false_alarm_rate_max_block", "window_macro_f1", "window_positive_precision",
        "window_positive_recall", "window_positive_f1", "window_pr_auc",
    ]
    for (target, variant), group in seeds.groupby(["target", "model_variant"], sort=False):
        row: dict[str, float | int | str] = {
            "target": target,
            "model_variant": variant,
            "seeds": int(group["seed"].nunique()),
            "test_blocks_per_seed": int(group["test_blocks"].min()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def consensus_cycles(seed_cycles: pd.DataFrame) -> pd.DataFrame:
    keys = ["target", "model_variant", "test_block", "cycle", "cycle_run", "cycle_occurrence"]
    consensus = (
        seed_cycles.groupby(keys, sort=False)
        .agg(
            true_event_cycle=("true_event_cycle", "first"),
            seed_count=("seed", "nunique"),
            seeds_with_cycle_alert=("cycle_alert", "sum"),
            seeds_detecting_event=("event_detected", "sum"),
        )
        .reset_index()
    )
    consensus["consensus_required"] = np.where(consensus["seed_count"].eq(1), 1, 2)
    consensus["consensus_cycle_alert"] = (
        consensus["seeds_with_cycle_alert"] >= consensus["consensus_required"]
    ).astype(int)
    consensus["consensus_event_detected"] = (
        consensus["true_event_cycle"].eq(1)
        & (consensus["seeds_detecting_event"] >= consensus["consensus_required"])
    ).astype(int)
    return consensus


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return center - margin, center + margin


def consensus_block_summary(cycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, variant, test_block), group in cycles.groupby(
        ["target", "model_variant", "test_block"], sort=False
    ):
        events = group[group["true_event_cycle"].eq(1)]
        normal = group[group["true_event_cycle"].eq(0)]
        detected = int(events["consensus_event_detected"].sum())
        false_alarms = int(normal["consensus_cycle_alert"].sum())
        rows.append(
            {
                "target": target,
                "model_variant": variant,
                "test_block": int(test_block),
                "event_cycle_count": int(len(events)),
                "detected_event_cycle_count": detected,
                "event_cycle_recall": float(detected / len(events)) if len(events) else np.nan,
                "normal_cycle_count": int(len(normal)),
                "false_alarm_cycle_count": false_alarms,
                "normal_cycle_false_alarm_rate": float(false_alarms / len(normal)) if len(normal) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def consensus_summary(cycles: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, variant), group in cycles.groupby(["target", "model_variant"], sort=False):
        events = group[group["true_event_cycle"].eq(1)]
        normal = group[group["true_event_cycle"].eq(0)]
        detected = int(events["consensus_event_detected"].sum())
        false_alarms = int(normal["consensus_cycle_alert"].sum())
        recall_low, recall_high = wilson_interval(detected, len(events))
        far_low, far_high = wilson_interval(false_alarms, len(normal))
        block_group = blocks[blocks["target"].eq(target) & blocks["model_variant"].eq(variant)]
        rows.append(
            {
                "target": target,
                "model_variant": variant,
                "seed_count": int(group["seed_count"].max()),
                "event_cycle_count": int(len(events)),
                "detected_event_cycle_count": detected,
                "event_cycle_recall": float(detected / len(events)),
                "event_cycle_recall_ci95_low": recall_low,
                "event_cycle_recall_ci95_high": recall_high,
                "event_cycle_recall_min_block": float(block_group["event_cycle_recall"].min()),
                "normal_cycle_count": int(len(normal)),
                "false_alarm_cycle_count": false_alarms,
                "normal_cycle_false_alarm_rate": float(false_alarms / len(normal)),
                "normal_cycle_false_alarm_rate_ci95_low": far_low,
                "normal_cycle_false_alarm_rate_ci95_high": far_high,
                "normal_cycle_false_alarm_rate_max_block": float(
                    block_group["normal_cycle_false_alarm_rate"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def pairwise_cycle_summary(cycles: pd.DataFrame) -> pd.DataFrame:
    primary = cycles[cycles["model_variant"].isin(PRIMARY_VARIANTS)].copy()
    index = ["target", "test_block", "cycle", "cycle_run", "cycle_occurrence", "true_event_cycle"]
    alert_pivot = primary.pivot(index=index, columns="model_variant", values="consensus_cycle_alert").reset_index()
    detect_pivot = primary.pivot(
        index=index, columns="model_variant", values="consensus_event_detected"
    ).reset_index()
    rows = []
    for target in primary["target"].unique():
        for variant in PRIMARY_VARIANTS[1:]:
            normal = alert_pivot[
                alert_pivot["target"].eq(target) & alert_pivot["true_event_cycle"].eq(0)
            ]
            rf_error = normal[PRIMARY_VARIANTS[0]].eq(1)
            model_error = normal[variant].eq(1)
            events = detect_pivot[
                detect_pivot["target"].eq(target) & detect_pivot["true_event_cycle"].eq(1)
            ]
            rf_miss = events[PRIMARY_VARIANTS[0]].eq(0)
            model_miss = events[variant].eq(0)
            rows.extend(
                [
                    {
                        "target": target,
                        "comparison_model": variant,
                        "error_type": "false_alarm",
                        "eligible_cycles": int(len(normal)),
                        "rf_error_count": int(rf_error.sum()),
                        "comparison_error_count": int(model_error.sum()),
                        "shared_error_count": int((rf_error & model_error).sum()),
                        "comparison_only_error_count": int((~rf_error & model_error).sum()),
                        "rf_only_error_count": int((rf_error & ~model_error).sum()),
                    },
                    {
                        "target": target,
                        "comparison_model": variant,
                        "error_type": "event_miss",
                        "eligible_cycles": int(len(events)),
                        "rf_error_count": int(rf_miss.sum()),
                        "comparison_error_count": int(model_miss.sum()),
                        "shared_error_count": int((rf_miss & model_miss).sum()),
                        "comparison_only_error_count": int((~rf_miss & model_miss).sum()),
                        "rf_only_error_count": int((rf_miss & ~model_miss).sum()),
                    },
                ]
            )
    return pd.DataFrame(rows)


def round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_float_dtype(result[column]):
            result[column] = result[column].round(4)
    return result


def comparison_statement(target_summary: pd.DataFrame, variant: str) -> str:
    indexed = target_summary.set_index("model_variant")
    rf = indexed.loc["rf_19_raw_window"]
    comparison = indexed.loc[variant]
    rf_better_or_equal = (
        rf["event_cycle_recall"] >= comparison["event_cycle_recall"]
        and rf["normal_cycle_false_alarm_rate"] <= comparison["normal_cycle_false_alarm_rate"]
    )
    comparison_better_or_equal = (
        comparison["event_cycle_recall"] >= rf["event_cycle_recall"]
        and comparison["normal_cycle_false_alarm_rate"] <= rf["normal_cycle_false_alarm_rate"]
    )
    if rf_better_or_equal and not comparison_better_or_equal:
        return "Random Forest가 recall과 오경보 기준에서 기술적으로 우세했다."
    if comparison_better_or_equal and not rf_better_or_equal:
        return f"{variant}가 recall과 오경보 기준에서 기술적으로 우세했다."
    return "Recall과 오경보가 엇갈려 단순 우위를 정할 수 없다."


def format_report(
    consensus: pd.DataFrame,
    model_seeds: pd.DataFrame,
    pairwise: pd.DataFrame,
    torch_runs: pd.DataFrame,
    rf_runs: pd.DataFrame,
) -> str:
    primary = consensus[consensus["model_variant"].isin(PRIMARY_VARIANTS)].copy()
    sensitivity = consensus[consensus["model_variant"].str.startswith("lstm_autoencoder")].copy()
    seed_columns = [
        "target", "model_variant", "seeds", "event_cycle_recall_mean",
        "normal_cycle_false_alarm_rate_mean", "window_macro_f1_mean", "window_positive_f1_mean",
        "window_pr_auc_mean",
    ]
    conclusions = []
    for target in ["System_Failure", "ProtectiveStop", "GripLost"]:
        target_summary = primary[primary["target"].eq(target)]
        conclusions.append(
            f"- `{target}` RF 대비 1D CNN: {comparison_statement(target_summary, '1d_cnn_19_raw')}"
        )
        conclusions.append(
            f"- `{target}` RF 대비 LSTM Autoencoder q95: "
            f"{comparison_statement(target_summary, 'lstm_autoencoder_19_raw_q95')}"
        )
    ae_runs = torch_runs[torch_runs["model"].eq("lstm_autoencoder_19_raw")]

    return "\n".join(
        [
            "# 12 동일 센서 입력 기반 LSTM Autoencoder 후속 비교",
            "",
            "## 고정 설계",
            "",
            "- 사전 고정 문서: `research/2026-08-23_lstm_autoencoder_preregistration.md`.",
            "- 공통 입력: `cycle_run` 경계를 넘지 않는 10-step×19개 원본 센서.",
            "- 모델: SMOTE+Random Forest window summary, supervised 1D CNN, normal-only LSTM Autoencoder.",
            "- Outer 평가: 9개 candidate block을 한 번씩 test로 사용.",
            "- Autoencoder primary threshold: calibration 정상 cycle 최대 reconstruction error의 95th percentile.",
            "- Deep learning cycle consensus: 3개 seed 중 2개 이상.",
            "",
            "## Primary consensus 결과",
            "",
            markdown_table(round_frame(primary)),
            "",
            "## Seed 평균 window 결과",
            "",
            markdown_table(round_frame(model_seeds[model_seeds["model_variant"].isin(PRIMARY_VARIANTS)][seed_columns])),
            "",
            "## Autoencoder threshold 민감도",
            "",
            markdown_table(round_frame(sensitivity)),
            "",
            "## Random Forest와 cycle 오류 겹침",
            "",
            markdown_table(pairwise),
            "",
            "## 사전 고정 해석 규칙 적용",
            "",
            *conclusions,
            "- 위 우세 판단은 recall과 오경보의 방향만 본 기술적 Pareto 비교이며 통계적 유의성을 뜻하지 않는다.",
            "",
            "## 실행 범위",
            "",
            f"- Random Forest run: {len(rf_runs)}개.",
            f"- 1D CNN run: {len(torch_runs[torch_runs['model'].eq('1d_cnn_19_raw')])}개.",
            f"- LSTM Autoencoder run: {len(ae_runs)}개.",
            f"- Autoencoder calibration 정상 cycle 수 범위: {int(ae_runs['calibration_normal_cycles'].min())}-"
            f"{int(ae_runs['calibration_normal_cycles'].max())}개.",
            f"- PyTorch: {torch_runs['torch_version'].iloc[0]}, CUDA: {torch_runs['cuda_version'].iloc[0]}, "
            f"device: {torch_runs['device'].iloc[0]}.",
            f"- 기록된 전체 학습 시간: "
            f"{torch_runs['elapsed_seconds'].sum() + rf_runs['elapsed_seconds'].sum():.1f}초.",
            "",
            "## 해석 제한",
            "",
            "- Window 4,035개는 겹치므로 독립 표본 수가 아니다. Cycle-level n은 202개다.",
            "- Wilson interval은 cycle 비율의 불확실성을 나타내지만 block/session 상관을 제거하지 않는다.",
            "- 9개 block은 실제 공정조건 정답이 아니라 수집 구간 proxy다.",
            "- 동일 데이터의 기존 결과를 이미 확인했으므로 독립 외부 검증이 아니라 사전 고정한 내부 후속 비교다.",
            "- Autoencoder sensitivity threshold 중 가장 좋은 값을 primary 결과로 교체하지 않는다.",
            "",
        ]
    )


def validate(predictions: pd.DataFrame, consensus: pd.DataFrame, model_summary: pd.DataFrame) -> None:
    if predictions.duplicated(
        ["target", "model_variant", "seed", "test_block", "cycle_run", "window_start_step"]
    ).any():
        raise AssertionError("Duplicate prediction keys.")
    if not predictions.groupby(["target", "model_variant", "seed"]).size().eq(4035).all():
        raise AssertionError("Each target/model/seed must contain 4,035 held-out predictions.")
    if not consensus.groupby(["target", "model_variant"]).size().eq(202).all():
        raise AssertionError("Each target/model consensus must contain 202 cycles.")
    if not model_summary.groupby("target")["model_variant"].nunique().eq(5).all():
        raise AssertionError("Each target must contain all five model variants.")


def main() -> None:
    ensure_output_dir()
    predictions = load_predictions()
    seed_cycles = seed_cycle_results(predictions)
    seed_blocks = seed_block_results(predictions, seed_cycles)
    seeds = seed_summary(predictions, seed_blocks)
    model_seeds = model_seed_summary(seeds)
    consensus = consensus_cycles(seed_cycles)
    consensus_blocks = consensus_block_summary(consensus)
    consensus_models = consensus_summary(consensus, consensus_blocks)
    pairwise = pairwise_cycle_summary(consensus)
    validate(predictions, consensus, consensus_models)

    seed_cycles.to_csv(OUTPUT_DIR / "12_matched_seed_cycle_results.csv", index=False, encoding="utf-8-sig")
    seed_blocks.to_csv(OUTPUT_DIR / "12_matched_seed_block_results.csv", index=False, encoding="utf-8-sig")
    seeds.to_csv(OUTPUT_DIR / "12_matched_seed_results.csv", index=False, encoding="utf-8-sig")
    model_seeds.to_csv(OUTPUT_DIR / "12_matched_model_seed_summary.csv", index=False, encoding="utf-8-sig")
    consensus.to_csv(OUTPUT_DIR / "12_matched_consensus_cycle_results.csv", index=False, encoding="utf-8-sig")
    consensus_blocks.to_csv(OUTPUT_DIR / "12_matched_consensus_block_results.csv", index=False, encoding="utf-8-sig")
    consensus_models.to_csv(OUTPUT_DIR / "12_matched_consensus_summary.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(OUTPUT_DIR / "12_matched_pairwise_cycle_errors.csv", index=False, encoding="utf-8-sig")

    torch_runs = pd.read_csv(OUTPUT_DIR / "12_matched_torch_runs.csv", encoding="utf-8-sig")
    rf_runs = pd.read_csv(OUTPUT_DIR / "12_matched_rf_runs.csv", encoding="utf-8-sig")
    report_path = OUTPUT_DIR / "12_matched_lstm_autoencoder_comparison.md"
    report_path.write_text(
        format_report(consensus_models, model_seeds, pairwise, torch_runs, rf_runs), encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
