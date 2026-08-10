from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from common import (
    BASE_SENSOR_COLS,
    CURRENT_COLS,
    DATASET_PATH,
    OUTPUT_DIR,
    SPEED_COLS,
    TARGET_COLS,
    TEMP_COLS,
    add_targets_and_features,
    ensure_output_dir,
    load_raw_data,
    markdown_table,
)


def target_distribution(df: pd.DataFrame, target: str) -> pd.DataFrame:
    counts = df[target].value_counts(dropna=False).sort_index()
    ratio = df[target].value_counts(dropna=False, normalize=True).sort_index() * 100
    return pd.DataFrame(
        {
            "target": target,
            "value": counts.index.astype(str),
            "count": counts.values,
            "ratio_percent": ratio.round(3).values,
        }
    )


def timestamp_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    diff = ts.diff().dt.total_seconds()
    stats = diff.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).reset_index()
    stats.columns = ["metric", "seconds"]
    common = diff.round(3).value_counts().head(20).reset_index()
    common.columns = ["diff_seconds_rounded", "count"]
    return stats, common


def cycle_audit(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("cycle", dropna=False).agg(
        rows=("cycle", "size"),
        system_failure_count=("System_Failure", lambda x: int((x == 1).sum())),
        protective_stop_count=("ProtectiveStop", lambda x: int((x == True).sum())),
        grip_lost_count=("GripLost", lambda x: int((x == True).sum())),
        first_timestamp=("Timestamp", "first"),
        last_timestamp=("Timestamp", "last"),
    )
    grouped["system_failure_rate"] = (grouped["system_failure_count"] / grouped["rows"]).round(4)
    return grouped.reset_index()


def main() -> None:
    out_dir = ensure_output_dir()
    raw = load_raw_data(DATASET_PATH)
    df = add_targets_and_features(raw)

    missing = df.isna().sum().reset_index()
    missing.columns = ["column", "missing_count"]
    missing["missing_ratio_percent"] = (missing["missing_count"] / len(df) * 100).round(3)

    target_tables = [target_distribution(df, target) for target in TARGET_COLS]
    targets = pd.concat(target_tables, ignore_index=True)

    ts_stats, ts_common = timestamp_audit(df)
    cycles = cycle_audit(df)

    feature_groups = pd.DataFrame(
        [
            {"group": "current", "columns": ", ".join(CURRENT_COLS), "count": len(CURRENT_COLS)},
            {"group": "temperature", "columns": ", ".join(TEMP_COLS), "count": len(TEMP_COLS)},
            {"group": "speed", "columns": ", ".join(SPEED_COLS), "count": len(SPEED_COLS)},
            {"group": "base_sensor", "columns": ", ".join(BASE_SENSOR_COLS), "count": len(BASE_SENSOR_COLS)},
        ]
    )

    missing.to_csv(out_dir / "00_missing_values.csv", index=False, encoding="utf-8-sig")
    targets.to_csv(out_dir / "00_target_distribution.csv", index=False, encoding="utf-8-sig")
    ts_stats.to_csv(out_dir / "00_timestamp_diff_stats.csv", index=False, encoding="utf-8-sig")
    ts_common.to_csv(out_dir / "00_timestamp_common_diffs.csv", index=False, encoding="utf-8-sig")
    cycles.to_csv(out_dir / "00_cycle_summary.csv", index=False, encoding="utf-8-sig")
    feature_groups.to_csv(out_dir / "00_feature_groups.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(DATASET_PATH),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "raw_columns": list(raw.columns),
        "clean_row_count_if_dropna_all": int(raw.dropna().shape[0]),
        "cycle_count": int(df["cycle"].nunique(dropna=True)),
        "cycle_min": int(df["cycle"].min()),
        "cycle_max": int(df["cycle"].max()),
        "notes": [
            "System_Failure is defined as ProtectiveStop OR GripLost.",
            "Rows with missing ProtectiveStop are excluded from model training.",
            "The process hyperparameters in the source paper are not present as dataset columns.",
        ],
    }
    (out_dir / "00_data_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "# 00 데이터 감사",
        "",
        f"- Dataset: `{DATASET_PATH}`",
        f"- 연구용 타깃/파생 feature 추가 후 크기: {df.shape[0]:,} rows x {df.shape[1]:,} columns",
        f"- 원본 기준 `dropna()` 후 행 수: {raw.dropna().shape[0]:,}",
        f"- cycle 범위: {int(df['cycle'].min())} - {int(df['cycle'].max())} ({df['cycle'].nunique()} unique cycles)",
        "",
        "## 타깃 분포",
        "",
        markdown_table(targets),
        "",
        "## Timestamp 간격 통계",
        "",
        markdown_table(ts_stats),
        "",
        "## 주요 Timestamp 간격",
        "",
        markdown_table(ts_common),
        "",
        "## 주의 사항",
        "",
        "- `workload`, `movement speed`, `gripping force`는 논문에 설명된 공정 조건이지만, 공개 CSV의 직접 컬럼은 아니다.",
        "- 과부하나 파지력 원인은 논문 기반 배경으로 설명하고, 본 데이터에서 직접 검증한 라벨처럼 쓰지 않는다.",
        "- row 단위 random split은 시계열 로그의 일반화 성능을 낙관적으로 보일 수 있으므로 cycle 기준 split과 비교한다.",
        "",
    ]
    (out_dir / "00_data_audit_report.md").write_text("\n".join(report), encoding="utf-8")

    print("데이터 감사 완료")
    print(out_dir / "00_data_audit_report.md")


if __name__ == "__main__":
    main()
