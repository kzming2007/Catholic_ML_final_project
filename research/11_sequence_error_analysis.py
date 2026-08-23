from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUTPUT_DIR, ensure_output_dir, markdown_table


DEEP_PATH = OUTPUT_DIR / "10_sequence_window_predictions.csv"
RF_PATH = OUTPUT_DIR / "09_event_level_window_predictions.csv"
CACHE_DIR = Path(__file__).with_name(".sequence_cache")
MANIFEST_PATH = CACHE_DIR / "10_sequence_manifest.json"
DEEP_MODELS = ["1d_cnn", "lstm"]
MODEL_ORDER = ["rf_window_features", *DEEP_MODELS]
WINDOW_KEYS = [
    "target",
    "test_block",
    "cycle",
    "cycle_run",
    "cycle_occurrence",
    "window_start_step",
    "window_end_step",
]


def load_predictions() -> pd.DataFrame:
    deep = pd.read_csv(DEEP_PATH, encoding="utf-8-sig")
    rf = pd.read_csv(RF_PATH, encoding="utf-8-sig")
    rf = rf[rf["feature_set"].eq("all_sensors")].copy()
    rf["model"] = "rf_window_features"
    rf["seed"] = 42

    columns = WINDOW_KEYS + ["model", "seed", "label", "score", "prediction"]
    predictions = pd.concat([rf[columns], deep[columns]], ignore_index=True)
    predictions["seed"] = predictions["seed"].astype(int)
    predictions["prediction"] = predictions["prediction"].astype(int)
    predictions["label"] = predictions["label"].astype(int)
    return predictions


def aggregate_windows(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = WINDOW_KEYS + ["model"]
    windows = (
        predictions.groupby(keys, sort=False)
        .agg(
            label=("label", "first"),
            seed_count=("seed", "nunique"),
            seeds_predicting_positive=("prediction", "sum"),
            score_mean=("score", "mean"),
            score_std=("score", lambda values: float(values.std(ddof=0))),
        )
        .reset_index()
    )
    windows["consensus_required"] = np.where(windows["seed_count"].eq(1), 1, 2)
    windows["consensus_prediction"] = (
        windows["seeds_predicting_positive"] >= windows["consensus_required"]
    ).astype(int)
    return windows


def aggregate_cycles(predictions: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    seed_cycle_keys = ["target", "model", "test_block", "cycle", "cycle_run", "cycle_occurrence", "seed"]
    seed_cycles = []
    for keys, group in predictions.groupby(seed_cycle_keys, sort=False):
        target, model, test_block, cycle, cycle_run, cycle_occurrence, seed = keys
        positive = group["label"].eq(1)
        seed_cycles.append(
            {
                "target": target,
                "model": model,
                "test_block": int(test_block),
                "cycle": int(cycle),
                "cycle_run": int(cycle_run),
                "cycle_occurrence": int(cycle_occurrence),
                "seed": int(seed),
                "true_event_cycle": int(positive.any()),
                "cycle_alert": int(group["prediction"].any()),
                "event_detected": int((positive & group["prediction"].eq(1)).any()),
            }
        )
    seed_cycles = pd.DataFrame(seed_cycles)

    cycle_keys = ["target", "model", "test_block", "cycle", "cycle_run", "cycle_occurrence"]
    cycles = (
        seed_cycles.groupby(cycle_keys, sort=False)
        .agg(
            true_event_cycle=("true_event_cycle", "first"),
            seed_count=("seed", "nunique"),
            seeds_with_cycle_alert=("cycle_alert", "sum"),
            seeds_detecting_event=("event_detected", "sum"),
        )
        .reset_index()
    )
    cycles["consensus_required"] = np.where(cycles["seed_count"].eq(1), 1, 2)
    cycles["consensus_false_alarm"] = (
        cycles["true_event_cycle"].eq(0)
        & (cycles["seeds_with_cycle_alert"] >= cycles["consensus_required"])
    ).astype(int)
    cycles["consensus_event_detected"] = (
        cycles["true_event_cycle"].eq(1)
        & (cycles["seeds_detecting_event"] >= cycles["consensus_required"])
    ).astype(int)
    cycles["consensus_event_miss"] = (
        cycles["true_event_cycle"].eq(1) & cycles["consensus_event_detected"].eq(0)
    ).astype(int)

    consensus_counts = (
        windows.groupby(cycle_keys, sort=False)
        .agg(
            window_count=("consensus_prediction", "size"),
            consensus_positive_window_count=("consensus_prediction", "sum"),
            mean_window_score=("score_mean", "mean"),
            max_window_score=("score_mean", "max"),
        )
        .reset_index()
    )
    return cycles.merge(consensus_counts, on=cycle_keys, how="left", validate="one_to_one")


def summarize_models(cycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), group in cycles.groupby(["target", "model"], sort=False):
        normal = group[group["true_event_cycle"].eq(0)]
        events = group[group["true_event_cycle"].eq(1)]
        false_alarms = int(normal["consensus_false_alarm"].sum())
        detected = int(events["consensus_event_detected"].sum())
        rows.append(
            {
                "target": target,
                "model": model,
                "seed_count": int(group["seed_count"].max()),
                "consensus_required": int(group["consensus_required"].max()),
                "normal_cycle_count": int(len(normal)),
                "consensus_false_alarm_cycle_count": false_alarms,
                "consensus_false_alarm_rate": float(false_alarms / len(normal)),
                "event_cycle_count": int(len(events)),
                "consensus_detected_event_count": detected,
                "consensus_event_recall": float(detected / len(events)),
            }
        )
    summary = pd.DataFrame(rows)
    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    return summary.sort_values(["target", "model"]).assign(model=lambda frame: frame["model"].astype(str))


def summarize_blocks(cycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model, test_block), group in cycles.groupby(["target", "model", "test_block"], sort=False):
        normal = group[group["true_event_cycle"].eq(0)]
        events = group[group["true_event_cycle"].eq(1)]
        false_alarms = int(normal["consensus_false_alarm"].sum())
        detected = int(events["consensus_event_detected"].sum())
        rows.append(
            {
                "target": target,
                "model": model,
                "test_block": int(test_block),
                "normal_cycle_count": int(len(normal)),
                "consensus_false_alarm_cycle_count": false_alarms,
                "consensus_false_alarm_rate": float(false_alarms / len(normal)) if len(normal) else np.nan,
                "event_cycle_count": int(len(events)),
                "consensus_detected_event_count": detected,
                "consensus_event_recall": float(detected / len(events)) if len(events) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_seed_consistency(cycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    deep = cycles[cycles["model"].isin(DEEP_MODELS)]
    for (target, model), group in deep.groupby(["target", "model"], sort=False):
        normal = group[group["true_event_cycle"].eq(0)]
        events = group[group["true_event_cycle"].eq(1)]
        row: dict[str, int | str] = {
            "target": target,
            "model": model,
            "normal_cycle_count": int(len(normal)),
            "event_cycle_count": int(len(events)),
        }
        for count in range(4):
            row[f"normal_cycles_alerted_by_{count}_seeds"] = int(normal["seeds_with_cycle_alert"].eq(count).sum())
            row[f"event_cycles_detected_by_{count}_seeds"] = int(events["seeds_detecting_event"].eq(count).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def overlap_tables(cycles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = ["target", "test_block", "cycle", "cycle_run", "cycle_occurrence", "true_event_cycle"]
    flags = cycles.pivot(index=index, columns="model", values=["consensus_false_alarm", "consensus_event_miss"])
    flags.columns = [f"{metric}__{model}" for metric, model in flags.columns]
    flags = flags.reset_index()

    detail_rows = []
    summary_rows = []
    for error_type, metric in [("false_alarm", "consensus_false_alarm"), ("event_miss", "consensus_event_miss")]:
        eligible = flags[flags["true_event_cycle"].eq(0 if error_type == "false_alarm" else 1)].copy()
        model_columns = {model: f"{metric}__{model}" for model in MODEL_ORDER}
        for model, column in model_columns.items():
            eligible[column] = eligible[column].fillna(0).astype(int)
        eligible["error_type"] = error_type
        eligible["rf_error"] = eligible[model_columns["rf_window_features"]]
        eligible["cnn_error"] = eligible[model_columns["1d_cnn"]]
        eligible["lstm_error"] = eligible[model_columns["lstm"]]
        detail_rows.append(
            eligible[index + ["error_type", "rf_error", "cnn_error", "lstm_error"]]
        )
        grouped = (
            eligible.groupby(["target", "rf_error", "cnn_error", "lstm_error"], sort=False)
            .size()
            .rename("cycle_count")
            .reset_index()
        )
        grouped.insert(1, "error_type", error_type)
        summary_rows.append(grouped)
    return pd.concat(detail_rows, ignore_index=True), pd.concat(summary_rows, ignore_index=True)


def pairwise_error_summary(overlap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, error_type), group in overlap.groupby(["target", "error_type"], sort=False):
        rf = group["rf_error"].eq(1)
        for model, column in [("1d_cnn", "cnn_error"), ("lstm", "lstm_error")]:
            deep = group[column].eq(1)
            rows.append(
                {
                    "target": target,
                    "error_type": error_type,
                    "deep_model": model,
                    "eligible_cycle_count": int(len(group)),
                    "rf_error_count": int(rf.sum()),
                    "deep_consensus_error_count": int(deep.sum()),
                    "shared_error_count": int((rf & deep).sum()),
                    "deep_only_error_count": int((~rf & deep).sum()),
                    "rf_only_error_count": int((rf & ~deep).sum()),
                }
            )
    return pd.DataFrame(rows)


def sequence_metadata(arrays: np.lib.npyio.NpzFile) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cycle": arrays["cycle"].astype(int),
            "cycle_run": arrays["cycle_run"].astype(int),
            "cycle_occurrence": arrays["cycle_occurrence"].astype(int),
            "test_block": arrays["scenario_block_25"].astype(int),
            "window_start_step": arrays["window_start_step"].astype(int),
            "window_end_step": arrays["window_end_step"].astype(int),
            "source_index": np.arange(len(arrays["y"]), dtype=int),
            "prepared_label": arrays["y"].astype(int),
        }
    )


def sensor_shift_analysis(windows: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    rows = []
    feature_names = manifest["feature_columns"]
    for entry in manifest["targets"]:
        target = entry["target"]
        with np.load(CACHE_DIR / entry["file"]) as arrays:
            x = arrays["X"].astype(np.float64)
            metadata = sequence_metadata(arrays)
        descriptors = np.concatenate([x.mean(axis=1), np.ptp(x, axis=1)], axis=1)
        descriptor_names = [f"{name}__mean" for name in feature_names] + [
            f"{name}__range" for name in feature_names
        ]

        for model in MODEL_ORDER:
            subset = windows[windows["target"].eq(target) & windows["model"].eq(model)].copy()
            merged = subset.merge(
                metadata,
                on=[
                    "test_block",
                    "cycle",
                    "cycle_run",
                    "cycle_occurrence",
                    "window_start_step",
                    "window_end_step",
                ],
                how="left",
                validate="one_to_one",
            )
            if merged["source_index"].isna().any() or not merged["label"].eq(merged["prepared_label"]).all():
                raise AssertionError(f"Sequence metadata mismatch for {target}/{model}.")
            normal_cycle = (
                merged.groupby(["test_block", "cycle_run"])["label"].transform("max").eq(0).to_numpy()
            )
            false_alarm = normal_cycle & merged["consensus_prediction"].eq(1).to_numpy()
            true_negative = normal_cycle & merged["consensus_prediction"].eq(0).to_numpy()
            values = descriptors[merged["source_index"].astype(int).to_numpy()]

            for column_index, descriptor in enumerate(descriptor_names):
                fp_values = values[false_alarm, column_index]
                tn_values = values[true_negative, column_index]
                fp_median = float(np.median(fp_values)) if len(fp_values) else np.nan
                tn_median = float(np.median(tn_values)) if len(tn_values) else np.nan
                tn_q25, tn_q75 = np.quantile(tn_values, [0.25, 0.75]) if len(tn_values) else (np.nan, np.nan)
                tn_iqr = float(tn_q75 - tn_q25)
                robust_shift = float((fp_median - tn_median) / tn_iqr) if tn_iqr > 0 else np.nan
                feature, statistic = descriptor.rsplit("__", 1)
                rows.append(
                    {
                        "target": target,
                        "model": model,
                        "feature": feature,
                        "window_statistic": statistic,
                        "false_alarm_window_count": int(len(fp_values)),
                        "true_negative_window_count": int(len(tn_values)),
                        "false_alarm_median": fp_median,
                        "true_negative_median": tn_median,
                        "true_negative_iqr": tn_iqr,
                        "robust_shift_iqr": robust_shift,
                        "absolute_robust_shift_iqr": abs(robust_shift) if np.isfinite(robust_shift) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_float_dtype(result[column]):
            result[column] = result[column].round(4)
    return result


def format_report(
    model_summary: pd.DataFrame,
    block_summary: pd.DataFrame,
    consistency: pd.DataFrame,
    pairwise: pd.DataFrame,
    sensor_shifts: pd.DataFrame,
) -> str:
    hotspots = block_summary.sort_values(
        ["consensus_false_alarm_rate", "consensus_false_alarm_cycle_count"], ascending=False
    ).head(12)
    shift_rows = []
    for _, group in sensor_shifts.groupby(["target", "model"], sort=False):
        shift_rows.append(group.sort_values("absolute_robust_shift_iqr", ascending=False).head(3))
    top_shifts = pd.concat(shift_rows, ignore_index=True)
    far_lines = []
    consistency_lines = []
    deep_only_lines = []
    for target in ["System_Failure", "ProtectiveStop", "GripLost"]:
        target_models = model_summary[model_summary["target"].eq(target)].set_index("model")
        far_lines.append(
            f"- `{target}` consensus 오경보율: Random Forest "
            f"{target_models.loc['rf_window_features', 'consensus_false_alarm_rate']:.4f}, "
            f"1D CNN {target_models.loc['1d_cnn', 'consensus_false_alarm_rate']:.4f}, "
            f"LSTM {target_models.loc['lstm', 'consensus_false_alarm_rate']:.4f}."
        )
        target_consistency = consistency[consistency["target"].eq(target)].set_index("model")
        consistency_lines.append(
            f"- `{target}`에서 3개 seed 모두 경보한 정상 cycle: 1D CNN "
            f"{int(target_consistency.loc['1d_cnn', 'normal_cycles_alerted_by_3_seeds'])}개, "
            f"LSTM {int(target_consistency.loc['lstm', 'normal_cycles_alerted_by_3_seeds'])}개."
        )
        target_pairwise = pairwise[
            pairwise["target"].eq(target) & pairwise["error_type"].eq("false_alarm")
        ].set_index("deep_model")
        deep_only_lines.append(
            f"- `{target}`에서 Random Forest는 정상 처리했지만 deep learning만 반복 오경보한 cycle: "
            f"1D CNN {int(target_pairwise.loc['1d_cnn', 'deep_only_error_count'])}개, "
            f"LSTM {int(target_pairwise.loc['lstm', 'deep_only_error_count'])}개."
        )
    top_hotspot = hotspots.iloc[0]

    return "\n".join(
        [
            "# 11 시계열 모델 오류 분석",
            "",
            "## 분석 질문",
            "",
            "- 딥러닝 오경보와 미탐이 seed가 바뀌어도 같은 cycle에서 반복되는가?",
            "- 오류가 특정 held-out block에 집중되는가?",
            "- Random Forest와 딥러닝이 같은 cycle에서 오류를 내는가?",
            "- 반복 오경보 window의 센서 요약값은 true-negative window와 기술적으로 어떤 차이를 보이는가?",
            "",
            "## 반복 오류 기준",
            "",
            "- Cycle consensus: 1D CNN/LSTM의 3개 seed 중 2개 이상이 같은 cycle 안에서 한 번 이상 경보하면 반복 cycle 경보로 정의한다.",
            "- Window consensus: 3개 seed 중 2개 이상이 동일 window를 positive로 판단하면 반복 window 경보로 정의한다.",
            "- Random Forest: `09`의 고정 all-sensors prediction 1회를 사용한다.",
            "- Event miss: 실제 positive window가 있는 cycle에서 consensus 기준으로 positive window를 하나도 탐지하지 못한 경우다.",
            "",
            "## Consensus 결과",
            "",
            markdown_table(round_frame(model_summary)),
            "",
            "## 주요 결과",
            "",
            *far_lines,
            *consistency_lines,
            *deep_only_lines,
            f"- 가장 높은 block 오경보율은 `{top_hotspot['target']}` `{top_hotspot['model']}` "
            f"block {int(top_hotspot['test_block'])}의 {top_hotspot['consensus_false_alarm_rate']:.4f}였다.",
            "- 2/3 seed consensus를 적용해도 deep learning의 오경보 격차가 유지되므로, "
            "추가 모델 확대보다 block 변화와 정상 저변동 구간에 대한 오류 원인 분석이 우선이다.",
            "",
            "## 오경보 집중 block 상위 12개",
            "",
            markdown_table(round_frame(hotspots)),
            "",
            "## Seed 일관성",
            "",
            markdown_table(consistency),
            "",
            "## Random Forest 대비 오류 겹침",
            "",
            markdown_table(pairwise),
            "",
            "`deep_only_error_count`가 크면 Random Forest가 틀리지 않은 cycle에서 딥러닝만 반복적으로 틀렸다는 뜻이다.",
            "",
            "## 정상 cycle 내 반복 오경보 window의 센서 차이 상위 3개",
            "",
            markdown_table(
                round_frame(
                    top_shifts[
                        [
                            "target",
                            "model",
                            "feature",
                            "window_statistic",
                            "false_alarm_window_count",
                            "true_negative_window_count",
                            "false_alarm_median",
                            "true_negative_median",
                            "robust_shift_iqr",
                        ]
                    ]
                )
            ),
            "",
            "## 해석 제한",
            "",
            "- sensor shift는 전체 window가 음성인 정상 cycle만 사용한다. 같은 cycle에서 겹치는 window가 다수 생성되므로 독립표본 유의성 검정이 아니라 오류 원인을 찾기 위한 기술통계다.",
            "- `robust_shift_iqr`는 false-alarm median과 true-negative median 차이를 true-negative IQR로 나눈 값이며, 인과적 feature importance가 아니다.",
            "- block은 실제 공정조건 정답이 아니라 수집 순서 기반 proxy이므로 block 집중을 공정조건 효과로 해석하지 않는다.",
            "- 이 분석도 이상이 포함된 window 탐지 결과에 대한 사후 분석이며 pre-failure 성능을 의미하지 않는다.",
            "",
        ]
    )


def validate_outputs(
    predictions: pd.DataFrame,
    windows: pd.DataFrame,
    cycles: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> None:
    expected_models = set(MODEL_ORDER)
    if set(predictions["model"].unique()) != expected_models:
        raise AssertionError("Unexpected model set.")
    if windows.duplicated(WINDOW_KEYS + ["model"]).any():
        raise AssertionError("Duplicate consensus window keys.")
    if cycles.duplicated(["target", "model", "test_block", "cycle_run"]).any():
        raise AssertionError("Duplicate cycle keys.")
    if not model_summary.groupby("target")["model"].nunique().eq(3).all():
        raise AssertionError("Each target must contain all three models.")
    if not model_summary["consensus_false_alarm_rate"].between(0, 1).all():
        raise AssertionError("Invalid false-alarm rate.")
    if not model_summary["consensus_event_recall"].between(0, 1).all():
        raise AssertionError("Invalid event recall.")


def main() -> None:
    ensure_output_dir()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    predictions = load_predictions()
    windows = aggregate_windows(predictions)
    cycles = aggregate_cycles(predictions, windows)
    model_summary = summarize_models(cycles)
    block_summary = summarize_blocks(cycles)
    consistency = summarize_seed_consistency(cycles)
    overlap_detail, overlap_summary = overlap_tables(cycles)
    pairwise = pairwise_error_summary(overlap_detail)
    sensor_shifts = sensor_shift_analysis(windows, manifest)
    validate_outputs(predictions, windows, cycles, model_summary)

    cycles.to_csv(OUTPUT_DIR / "11_error_cycle_details.csv", index=False, encoding="utf-8-sig")
    model_summary.to_csv(OUTPUT_DIR / "11_error_model_summary.csv", index=False, encoding="utf-8-sig")
    block_summary.to_csv(OUTPUT_DIR / "11_error_block_summary.csv", index=False, encoding="utf-8-sig")
    consistency.to_csv(OUTPUT_DIR / "11_seed_consistency_summary.csv", index=False, encoding="utf-8-sig")
    overlap_detail.to_csv(OUTPUT_DIR / "11_error_overlap_cycles.csv", index=False, encoding="utf-8-sig")
    overlap_summary.to_csv(OUTPUT_DIR / "11_error_overlap_summary.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(OUTPUT_DIR / "11_error_pairwise_summary.csv", index=False, encoding="utf-8-sig")
    sensor_shifts.to_csv(OUTPUT_DIR / "11_false_alarm_sensor_shifts.csv", index=False, encoding="utf-8-sig")
    report_path = OUTPUT_DIR / "11_sequence_error_analysis.md"
    report_path.write_text(
        format_report(model_summary, block_summary, consistency, pairwise, sensor_shifts),
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
