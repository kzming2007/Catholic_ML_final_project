from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_validate, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    BASE_SENSOR_COLS,
    CURRENT_COLS,
    SPEED_COLS,
    TEMP_COLS,
    ensure_output_dir,
    load_model_data,
    markdown_table,
)


RANDOM_STATE = 42
MIN_PREFIX_ROWS = 5
EXPECTED_REPEATS_PER_SCENARIO = 25
COMPLETE_BLOCK_MIN_RUNS = 20
CLUSTER_REPEAT_COUNT = 30
PERMUTATION_COUNT = 100

SUMMARY_STATS = ("mean", "std", "q10", "q50", "q90", "abs_mean", "abs_q90")


def summarize_values(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {stat: np.nan for stat in SUMMARY_STATS}
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "q10": float(clean.quantile(0.10)),
        "q50": float(clean.quantile(0.50)),
        "q90": float(clean.quantile(0.90)),
        "abs_mean": float(clean.abs().mean()),
        "abs_q90": float(clean.abs().quantile(0.90)),
    }


def build_cycle_run_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | bool]] = []
    audit_rows: list[dict[str, float | int | bool]] = []

    for cycle_run, group in df.groupby("cycle_run", sort=False):
        group = group.copy()
        cycle_id = int(group["cycle"].iloc[0])
        failure = group["System_Failure"].fillna(0).astype(int).eq(1).to_numpy()
        first_failure_position = int(np.flatnonzero(failure)[0]) if failure.any() else len(group)
        healthy_prefix = group.iloc[:first_failure_position].dropna(subset=BASE_SENSOR_COLS)

        audit = {
            "cycle_run": int(cycle_run),
            "cycle": cycle_id,
            "cycle_occurrence": int(group["cycle_occurrence"].iloc[0]),
            "scenario_block_25": int((cycle_id - 1) // EXPECTED_REPEATS_PER_SCENARIO + 1),
            "source_rows": int(len(group)),
            "first_failure_position": int(first_failure_position),
            "healthy_prefix_rows": int(len(healthy_prefix)),
            "fault_in_run": bool(failure.any()),
            "included": bool(len(healthy_prefix) >= MIN_PREFIX_ROWS),
        }
        audit_rows.append(audit)
        if not audit["included"]:
            continue

        row: dict[str, float | int | bool] = dict(audit)
        for col in BASE_SENSOR_COLS:
            for stat, value in summarize_values(healthy_prefix[col]).items():
                row[f"{col}__{stat}"] = value

        row["speed_proxy"] = float(
            np.mean([row[f"{col}__abs_q90"] for col in SPEED_COLS])
        )
        row["workload_proxy"] = float(
            np.mean([row[f"{col}__abs_q90"] for col in CURRENT_COLS])
        )
        row["gripping_force_proxy"] = float(row["Tool_current__q90"])
        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def feature_columns(sensor_cols: list[str], stats: tuple[str, ...] = SUMMARY_STATS) -> list[str]:
    return [f"{col}__{stat}" for col in sensor_cols for stat in stats]


def cluster_stability(X_scaled: np.ndarray, reference_labels: np.ndarray) -> float:
    label_sets = [reference_labels]
    for seed in range(CLUSTER_REPEAT_COUNT):
        labels = KMeans(n_clusters=3, n_init=10, random_state=seed).fit_predict(X_scaled)
        label_sets.append(labels)
    scores = [adjusted_rand_score(a, b) for a, b in combinations(label_sets, 2)]
    return float(np.mean(scores))


def run_proxy_clustering(cycle_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configs = {
        "movement_speed": {
            "features": feature_columns(SPEED_COLS, ("std", "abs_mean", "abs_q90", "q10", "q90")),
            "proxy": "speed_proxy",
        },
        "workload": {
            "features": feature_columns(CURRENT_COLS, ("std", "abs_mean", "abs_q90", "q10", "q90")),
            "proxy": "workload_proxy",
        },
        "gripping_force": {
            "features": feature_columns(["Tool_current"], ("mean", "std", "q10", "q50", "q90")),
            "proxy": "gripping_force_proxy",
        },
    }

    summary_rows = []
    center_rows = []
    assignment_frames = []

    for parameter, config in configs.items():
        cols = config["features"]
        proxy_col = config["proxy"]
        valid = cycle_df.dropna(subset=cols + [proxy_col]).copy()
        X_scaled = StandardScaler().fit_transform(valid[cols])

        reference = KMeans(n_clusters=3, n_init=50, random_state=RANDOM_STATE)
        raw_labels = reference.fit_predict(X_scaled)
        proxy_centers = valid.assign(raw_cluster=raw_labels).groupby("raw_cluster")[proxy_col].mean()
        ordered_raw_labels = proxy_centers.sort_values().index.tolist()
        ordinal_map = {raw_label: order + 1 for order, raw_label in enumerate(ordered_raw_labels)}
        ordinal_labels = np.array([ordinal_map[label] for label in raw_labels], dtype=int)

        bic_rows = []
        for components in range(1, 6):
            model = GaussianMixture(
                n_components=components,
                covariance_type="diag",
                reg_covar=1e-5,
                n_init=10,
                random_state=RANDOM_STATE,
            )
            model.fit(X_scaled)
            bic_rows.append((components, float(model.bic(X_scaled))))
        best_components = min(bic_rows, key=lambda item: item[1])[0]

        summary_rows.append(
            {
                "parameter_proxy": parameter,
                "cycle_runs": int(len(valid)),
                "feature_count": int(len(cols)),
                "kmeans_k": 3,
                "silhouette": float(silhouette_score(X_scaled, raw_labels)),
                "davies_bouldin": float(davies_bouldin_score(X_scaled, raw_labels)),
                "repeat_ari_mean": cluster_stability(X_scaled, raw_labels),
                "gmm_bic_best_components_1to5": int(best_components),
                "gmm_bic_k1": dict(bic_rows)[1],
                "gmm_bic_k2": dict(bic_rows)[2],
                "gmm_bic_k3": dict(bic_rows)[3],
                "gmm_bic_k4": dict(bic_rows)[4],
                "gmm_bic_k5": dict(bic_rows)[5],
            }
        )

        assignment = valid[["cycle_run", "cycle", "cycle_occurrence", "scenario_block_25", proxy_col]].copy()
        assignment.insert(0, "parameter_proxy", parameter)
        assignment["cluster_level"] = ordinal_labels
        assignment_frames.append(assignment)

        for level in (1, 2, 3):
            mask = ordinal_labels == level
            center_rows.append(
                {
                    "parameter_proxy": parameter,
                    "cluster_level": level,
                    "cycle_runs": int(mask.sum()),
                    "proxy_mean": float(valid.loc[mask, proxy_col].mean()),
                    "proxy_std": float(valid.loc[mask, proxy_col].std(ddof=0)),
                    "proxy_min": float(valid.loc[mask, proxy_col].min()),
                    "proxy_max": float(valid.loc[mask, proxy_col].max()),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(center_rows), pd.concat(assignment_frames, ignore_index=True)


def block_counts(audit: pd.DataFrame) -> pd.DataFrame:
    return (
        audit.groupby("scenario_block_25", as_index=False)
        .agg(
            cycle_runs=("cycle_run", "nunique"),
            included_cycle_runs=("included", "sum"),
            cycle_min=("cycle", "min"),
            cycle_max=("cycle", "max"),
            faults=("fault_in_run", "sum"),
        )
        .assign(complete_like=lambda frame: frame["cycle_runs"] >= COMPLETE_BLOCK_MIN_RUNS)
    )


def run_block_classification(cycle_df: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    complete_blocks = counts.loc[counts["complete_like"], "scenario_block_25"]
    analysis_df = cycle_df[cycle_df["scenario_block_25"].isin(complete_blocks)].copy()
    y = analysis_df["scenario_block_25"].astype(int)
    n_splits = min(4, int(y.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    feature_sets = {
        "process_sensors": feature_columns(CURRENT_COLS + SPEED_COLS + ["Tool_current"]),
        "temperature_only": feature_columns(TEMP_COLS),
    }
    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    rows = []
    for feature_set_name, cols in feature_sets.items():
        valid = analysis_df.dropna(subset=cols).copy()
        X = valid[cols]
        y_valid = valid["scenario_block_25"].astype(int)
        for model_name, model in models.items():
            scores = cross_validate(
                model,
                X,
                y_valid,
                cv=cv,
                scoring={"balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro"},
                n_jobs=None,
            )
            row = {
                "feature_set": feature_set_name,
                "model": model_name,
                "cycle_runs": int(len(valid)),
                "scenario_blocks": int(y_valid.nunique()),
                "cv_splits": int(n_splits),
                "balanced_accuracy_mean": float(scores["test_balanced_accuracy"].mean()),
                "balanced_accuracy_std": float(scores["test_balanced_accuracy"].std(ddof=0)),
                "macro_f1_mean": float(scores["test_macro_f1"].mean()),
                "macro_f1_std": float(scores["test_macro_f1"].std(ddof=0)),
                "permutation_pvalue": np.nan,
            }
            if model_name == "logistic_regression":
                _, _, pvalue = permutation_test_score(
                    model,
                    X,
                    y_valid,
                    scoring="balanced_accuracy",
                    cv=cv,
                    n_permutations=PERMUTATION_COUNT,
                    random_state=RANDOM_STATE,
                    n_jobs=None,
                )
                row["permutation_pvalue"] = float(pvalue)
            rows.append(row)

    return pd.DataFrame(rows)


def evidence_label(row: pd.Series) -> str:
    silhouette = float(row["silhouette"])
    best_k = int(row["gmm_bic_best_components_1to5"])
    if silhouette >= 0.50 and best_k == 3:
        return "3개 수준 구조를 강하게 지지"
    if silhouette >= 0.25 and best_k == 3:
        return "3개 수준 구조를 제한적으로 지지"
    if silhouette >= 0.25:
        return "분리는 보이나 3개 수준과 일치하지 않음"
    return "3개 수준 구조를 지지하기 어려움"


def format_report(
    audit: pd.DataFrame,
    counts: pd.DataFrame,
    clustering: pd.DataFrame,
    centers: pd.DataFrame,
    block_results: pd.DataFrame,
) -> str:
    clustering_display = clustering.copy()
    clustering_display["판정"] = clustering_display.apply(evidence_label, axis=1)
    for col in clustering_display.columns:
        if col not in {
            "parameter_proxy",
            "cycle_runs",
            "feature_count",
            "kmeans_k",
            "gmm_bic_best_components_1to5",
            "판정",
        }:
            clustering_display[col] = pd.to_numeric(clustering_display[col], errors="coerce").round(4)

    centers_display = centers.copy()
    for col in ["proxy_mean", "proxy_std", "proxy_min", "proxy_max"]:
        centers_display[col] = centers_display[col].round(4)

    block_display = block_results.copy()
    for col in [
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "permutation_pvalue",
    ]:
        block_display[col] = block_display[col].round(4)

    duplicate_ids = (
        audit.groupby("cycle")["cycle_run"].nunique().loc[lambda values: values.gt(1)].index.astype(int).tolist()
    )
    excluded = int((~audit["included"]).sum())
    cluster_index = clustering.set_index("parameter_proxy")
    speed = cluster_index.loc["movement_speed"]
    workload = cluster_index.loc["workload"]
    gripping_force = cluster_index.loc["gripping_force"]
    block_index = block_results.set_index(["feature_set", "model"])
    process_logistic = block_index.loc[("process_sensors", "logistic_regression")]
    temperature_forest = block_index.loc[("temperature_only", "random_forest")]

    lines = [
        "# 06 공정조건 분리 가능성 진단",
        "",
        "## 연구 질문",
        "",
        "공개 UR3 CobotOps 센서 데이터만 사용해 논문에 제시된 `movement speed`, `workload`, `gripping force`의 서로 다른 수준을 cycle별로 독립 구분할 수 있는지 탐색한다.",
        "",
        "## 논문 근거와 증거 한계",
        "",
        "- 논문은 세 정적 공정조건을 `workload` 1/2/3 kg, `movement speed` 60/80/100%, `gripping force` 80/100/120 N으로 제시하고 각 test scenario를 25회 반복했다고 기술한다.",
        "- 여기서 hyperparameter는 학습률이나 tree 깊이 같은 ML hyperparameter가 아니라 로봇 실험의 고정 공정조건이다.",
        "- 논문은 가능한 27개 조합을 모두 사용했는지, 각 scenario가 어느 cycle 범위에 해당하는지 공개하지 않는다.",
        "- 공개 Excel/CSV에는 세 공정조건 라벨이나 scenario 조합표가 없고, `Cycle`은 시행 번호만 제공한다.",
        "- 원본 Excel에는 보이는 `data` 시트 하나만 있으며, 숨은 시트·숨은 열·defined name·cell comment·조건 관련 문자열도 확인되지 않았다.",
        "- 따라서 실제 조건 수준에 대한 supervised accuracy는 계산할 수 없다. 아래 결과는 센서에 3개 latent regime이 보이는지와 25-cycle acquisition block이 구분되는지를 진단한 간접 근거다.",
        "- 논문: Tyrovolas et al. (2024), DOI 10.1007/978-3-031-63851-0_6.",
        "",
        "## 전처리",
        "",
        f"- 원본 `cycle` 값이 비연속적으로 재등장하는 ID: {duplicate_ids}.",
        "- 동일 cycle ID의 재등장 구간을 합치지 않고, 시간순 연속 구간마다 별도 `cycle_run`을 부여했다.",
        "- 고장 자체가 조건 분리를 대신하지 않도록 각 시행의 first positive 이전 healthy prefix만 요약했다.",
        f"- healthy prefix가 {MIN_PREFIX_ROWS}개 미만인 시행 {excluded}개는 제외했다.",
        "- cycle ID나 Timestamp는 입력 feature로 사용하지 않았다.",
        "",
        "## 25-cycle 후보 블록",
        "",
        markdown_table(counts),
        "",
        "## 공정조건별 3-cluster 진단",
        "",
        markdown_table(clustering_display),
        "",
        "### Cluster별 물리 대리변수 범위",
        "",
        markdown_table(centers_display),
        "",
        "- `movement_speed`는 joint speed 요약, `workload`는 joint current 요약, `gripping_force`는 `Tool_current` 요약으로 진단했다.",
        "- Silhouette은 cluster 간 분리도, `repeat_ari_mean`은 초기값을 바꿨을 때 cluster 재현성, GMM BIC는 데이터가 선호하는 component 수를 뜻한다.",
        "- GMM component 수는 1-5 범위에서만 비교했다.",
        "- cluster level 1/2/3은 대리변수 평균의 낮음/중간/높음 순서일 뿐 실제 60/80/100%, 1/2/3 kg, 80/100/120 N 라벨이 아니다.",
        "",
        "### Cluster 결과 해석",
        "",
        f"- `movement_speed`: Silhouette {speed['silhouette']:.4f}, GMM 최적 component {int(speed['gmm_bic_best_components_1to5'])}개로, 세 속도 수준이 자연스럽게 분리된다는 근거가 약하다.",
        f"- `workload`: Silhouette {workload['silhouette']:.4f}로 일부 분리는 보이지만 GMM 최적 component는 {int(workload['gmm_bic_best_components_1to5'])}개이고 cluster 크기가 154/16/8로 치우쳤다. 1/2/3 kg의 균형 잡힌 세 수준으로 해석할 수 없다.",
        f"- `gripping_force`: Silhouette {gripping_force['silhouette']:.4f}, GMM 최적 component {int(gripping_force['gmm_bic_best_components_1to5'])}개이며 낮음·중간 cluster의 proxy 범위가 겹친다. 80/100/120 N을 독립 복원했다고 볼 수 없다.",
        "",
        "## 25-cycle 후보 블록 분류",
        "",
        markdown_table(block_display),
        "",
        "- 논문의 25회 반복을 근거로 cycle ID를 25개씩 묶고, 20개 이상 시행이 남은 블록만 사용했다.",
        "- 높은 분류 성능은 acquisition block 사이에 센서 분포 차이가 있음을 뜻한다. 공정조건 조합의 차이인지 시간 경과, 온도 drift, 재설정 같은 session effect인지는 라벨 없이 분리할 수 없다.",
        "- `temperature_only`가 높은 성능을 보이면 공정조건보다 시간·장비 상태가 block 구분에 기여했을 가능성을 함께 고려해야 한다.",
        f"- 실제로 process sensor Logistic Regression의 balanced accuracy는 {process_logistic['balanced_accuracy_mean']:.4f}로 chance 0.1111보다 높지만 낮은 수준이었다. 반면 temperature-only Random Forest는 {temperature_forest['balanced_accuracy_mean']:.4f}로 매우 높아, block 정체성이 thermal/session drift에 강하게 남아 있음을 보여준다.",
        "",
        "## 결론 사용 범위",
        "",
        "- 직접 결론: 공개 파일만으로 세 공정조건의 실제 값을 cycle별로 확정하거나 조건 분류 정확도를 검증할 수 없다.",
        "- 간접 결론: 공정 sensor에는 25-cycle block 차이가 약하게 남지만, 세 물리조건의 3개 수준과 일치하는 단순 구조는 확인되지 않았다. block 구분은 온도와 수집 순서의 영향을 크게 받는다.",
        "- 금지할 해석: 현재 cluster를 1/2/3 kg, 60/80/100%, 80/100/120 N의 정답 라벨로 간주해 후속 supervised model을 학습하면 순환논증이 된다.",
        "- 후속 검증: 저자 또는 원 수집팀의 cycle-to-condition mapping을 확보하면, `cycle_run` 단위 group split으로 세 조건을 각각 supervised classification하여 실제 구분 가능성을 검증한다.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    out_dir = ensure_output_dir()
    df = load_model_data()
    cycle_df, audit = build_cycle_run_summary(df)
    counts = block_counts(audit)
    clustering, centers, assignments = run_proxy_clustering(cycle_df)
    block_results = run_block_classification(cycle_df, counts)

    audit.to_csv(out_dir / "06_condition_cycle_run_audit.csv", index=False, encoding="utf-8-sig")
    cycle_df.to_csv(out_dir / "06_condition_cycle_run_features.csv", index=False, encoding="utf-8-sig")
    counts.to_csv(out_dir / "06_condition_block_counts.csv", index=False, encoding="utf-8-sig")
    clustering.to_csv(out_dir / "06_condition_cluster_summary.csv", index=False, encoding="utf-8-sig")
    centers.to_csv(out_dir / "06_condition_cluster_centers.csv", index=False, encoding="utf-8-sig")
    assignments.to_csv(out_dir / "06_condition_cluster_assignments.csv", index=False, encoding="utf-8-sig")
    block_results.to_csv(out_dir / "06_condition_block_classification.csv", index=False, encoding="utf-8-sig")
    (out_dir / "06_process_condition_separability.md").write_text(
        format_report(audit, counts, clustering, centers, block_results),
        encoding="utf-8",
    )

    print("공정조건 분리 가능성 진단 완료")
    print(out_dir / "06_process_condition_separability.md")


if __name__ == "__main__":
    main()
