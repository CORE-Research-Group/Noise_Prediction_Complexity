"""
Posthoc XAI analysis for trained CatBoost experiments

What this script does
Loads task datasets created in step 2 from ML_tasks/<Dataset>/exp_<add>/tasks/*.csv
Discovers completed ML experiment folders under ML_experiments/<Dataset>/exp_<add>/<task>
Loads the trained CatBoost model from each split folder
Reconstructs the exact balanced test datasets using the saved row indices from balanced_row_indices.json
Computes explainability analyses for every split
  feature importance using the trained CatBoost model
  SHAP importance using the balanced test dataset
  permutation importance using the balanced test dataset with macro F1 scoring
Writes per split XAI outputs into the corresponding split folders
Aggregates XAI results across all splits and stores summary tables and plots under Results/<Dataset>/exp_<add>/<task>

Classification tasks expected from step 2
Task 1 uses noise_label_task1 and is multi class
Task 2 uses noise_label_task2 and is multi class
Task 3 uses noise_present and is binary with 0 and 1

Notes
Only feature columns with complexity metrics are used. Identifiers such as signal_id and run_id are excluded.
The script does not retrain models. It operates only on stored models and saved split metadata.
Balanced datasets are reconstructed using the saved row indices from balanced_row_indices.json to ensure that SHAP and permutation importance are computed on the exact same data used during evaluation.
Feature importance is computed directly from the trained CatBoost model.
SHAP importance is computed as mean absolute SHAP values across the reconstructed balanced test dataset.
Permutation importance measures the mean drop in macro F1 score when features are permuted.
Aggregated results report the mean and standard deviation of importance values across all splits.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import Pool, CatBoostClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score


# -----------------------------
# Use case selection (0..4)
# 0 Roessler, 1 ECG, 2 Lorenz, 3 Henon, 4 AR1
# -----------------------------
START_USE_CASE = 3  # start here and run upward

USE_CASES = {
    0: {
        "name": "Roessler",
        "data_base_folder": "Roessler_Noise_Exp",
        "folder_suffix": "_roessler_data",
        "file_prefix": "roessler",
    },
    1: {
        "name": "ECG",
        "data_base_folder": "ECG_Noise_Exp",
        "folder_suffix": "_ecg_data",
        "file_prefix": "ecg",
    },
    2: {
        "name": "Lorenz",
        "data_base_folder": "Lorenz_Noise_Exp",
        "folder_suffix": "_lorenz_data",
        "file_prefix": "lorenz",
    },
    3: {
        "name": "Henon",
        "data_base_folder": "Henon_Noise_Exp",
        "folder_suffix": "_henon_data",
        "file_prefix": "henon",
    },
    4: {
        "name": "AR1",
        "data_base_folder": "AR1_Noise_Exp",
        "folder_suffix": "_ar1_data",
        "file_prefix": "ar1",
    },
}

BASE_PROJECT = r"C:\Users\kmallinger\PycharmProjects\Noise_Prediction_Complexity_v2"

ML_EXPERIMENTS_ROOT = os.path.join(BASE_PROJECT, "ML_experiments")
ML_TASKS_ROOT = os.path.join(BASE_PROJECT, "ML_tasks")
RESULTS_ROOT = os.path.join(BASE_PROJECT, "Results")

ADD = "ed10_td1_mc300"

TOPK_FEATURES = 20
DPI = 400

PERM_N_REPEATS = 30
PERM_RANDOM_STATE = 42

SIGNAL_ID_COL = "signal_id"
TARGET_COL = "y"

VERBOSE = True


def log(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    log(f"      load_json {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_balanced_test_row_indices(split_dir: str) -> List[int]:
    path = os.path.join(split_dir, "balanced_row_indices.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing balanced_row_indices.json in {split_dir}")

    info = load_json(path)

    row_indices = info.get("test_row_indices_after_balance")
    if row_indices is None:
        raise ValueError(f"test_row_indices_after_balance missing in {path}")

    return [int(x) for x in row_indices]


def list_dirs(parent: str, prefix: Optional[str] = None) -> List[str]:
    if not os.path.isdir(parent):
        return []
    out: List[str] = []
    for name in os.listdir(parent):
        full = os.path.join(parent, name)
        if os.path.isdir(full):
            if prefix is None or name.startswith(prefix):
                out.append(full)
    out = sorted(out)
    log(f"  list_dirs {parent} prefix {prefix} -> {len(out)} dirs")
    return out


def save_png(fig: plt.Figure, out_path: str) -> None:
    log(f"      save_png {out_path}")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")


def compute_mean_std(df_all: pd.DataFrame) -> pd.DataFrame:
    log(
        f"      compute_mean_std n_rows {len(df_all)} n_features "
        f"{df_all['feature'].nunique() if 'feature' in df_all.columns else 'na'}"
    )
    stats = (
        df_all.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    stats["std"] = stats["std"].fillna(0.0)
    return stats


def plot_importance_mean_std(stats: pd.DataFrame, title: str, xlabel: str, topk: int) -> plt.Figure:
    log(f"      plot_importance_mean_std topk {topk} title {title}")
    df = stats.head(topk).copy().iloc[::-1]
    fig = plt.figure(figsize=(8, 5))
    plt.barh(df["feature"], df["mean"], xerr=df["std"], capsize=4)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    return fig


@dataclass(frozen=True)
class TaskInfo:
    dataset_name: str
    task_key: str
    task_dir: str


def dataset_matches_use_case(dataset_name: str, use_case: int) -> bool:
    if use_case not in USE_CASES:
        raise ValueError(f"use_case must be in {sorted(USE_CASES.keys())}")
    key = str(USE_CASES[use_case]["name"]).lower()
    dn = str(dataset_name).lower()
    if key == "roessler":
        return ("roessler" in dn) or ("rossler" in dn)
    return key in dn


def discover_all_tasks(experiments_root: str, add: str, use_case: int) -> List[TaskInfo]:
    log("============================================================")
    log("Discover tasks")
    log(f"  experiments_root {experiments_root}")
    log(f"  add {add}")
    log(f"  use_case {use_case} name {USE_CASES[use_case]['name']}")

    tasks: List[TaskInfo] = []
    for ds_dir in list_dirs(experiments_root):
        dataset_name = os.path.basename(ds_dir)

        if not dataset_matches_use_case(dataset_name, use_case):
            continue

        exp_dir = os.path.join(ds_dir, f"exp_{add}")
        if not os.path.isdir(exp_dir):
            continue

        for task_dir in list_dirs(exp_dir):
            if os.path.isdir(os.path.join(task_dir, "splits")):
                tasks.append(
                    TaskInfo(
                        dataset_name=dataset_name,
                        task_key=os.path.basename(task_dir),
                        task_dir=task_dir,
                    )
                )

    if not tasks:
        raise FileNotFoundError("No task folders found under ML_experiments for the selected use case")

    log(f"  tasks_found {len(tasks)}")
    for t in tasks:
        log(f"    {t.dataset_name} | {t.task_key}")
    log("============================================================")
    return tasks


def _task_csv_path(dataset_name: str, task_key: str) -> str:
    return os.path.join(
        ML_TASKS_ROOT,
        dataset_name,
        f"exp_{ADD}",
        "tasks",
        f"{task_key}_{ADD}.csv",
    )


def load_task_dataframe(dataset_name: str, task_key: str) -> pd.DataFrame:
    path = _task_csv_path(dataset_name, task_key)
    log(f"  load_task_dataframe dataset {dataset_name} task {task_key}")
    log(f"    path {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing task dataset file {path}")

    df = pd.read_csv(path)
    log(f"    df_shape {df.shape}")

    if SIGNAL_ID_COL not in df.columns:
        raise ValueError(f"Missing {SIGNAL_ID_COL} in {path}")

    tk = task_key.lower()

    if TARGET_COL not in df.columns:
        if "task1" in tk:
            src = "noise_label_task1"
        elif "task2" in tk:
            src = "noise_label_task2"
        elif "task3" in tk:
            src = "noise_present"
        else:
            raise ValueError(f"Cannot infer target for {task_key}")

        if src not in df.columns:
            raise ValueError(f"Missing {src} in {path}")

        df[TARGET_COL] = df[src]
        log(f"    created target column y from {src}")

    return df


NON_FEATURE_COLS = {
    TARGET_COL,
    "split",
    "noise_type",
    "noise_intensity",
    "noise_label_task1",
    "noise_label_task2",
    "noise_present",
    "timestep",
    "window_end_idx",
    "window_start_idx",
    SIGNAL_ID_COL,
}


def get_xy_by_row_indices(
    df: pd.DataFrame,
    row_indices: List[int],
    task_key: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    log(f"      get_xy_by_row_indices task {task_key} row_indices_n {len(row_indices)}")

    if not row_indices:
        raise ValueError("Empty row_indices provided")

    df_sel = df.loc[row_indices].copy()
    log(f"      df_sel_shape {df_sel.shape}")
    if df_sel.empty:
        raise ValueError("Empty selected set after filtering via saved row indices")

    tk = task_key.lower()
    if "task1" in tk:
        y = df_sel["noise_label_task1"]
    elif "task2" in tk:
        y = df_sel["noise_label_task2"]
    elif "task3" in tk:
        y = df_sel["noise_present"]
    else:
        y = df_sel[TARGET_COL]

    drop_cols = [c for c in NON_FEATURE_COLS if c in df_sel.columns]
    X = df_sel.drop(columns=drop_cols, errors="ignore")

    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    if obj_cols:
        log(f"      dropping object cols {obj_cols}")
        X = X.drop(columns=obj_cols, errors="ignore")

    X = X.apply(pd.to_numeric, errors="coerce")
    na_before = int(X.isna().sum().sum())
    if na_before > 0:
        log(f"      X has NaNs {na_before} filling with 0")
        X = X.fillna(0.0)

    log(f"      X_shape {X.shape} y_len {len(y)}")
    return X, y


def catboost_predict_labels(model: CatBoostClassifier, X: pd.DataFrame) -> np.ndarray:
    pred = model.predict(X)
    return np.asarray(pred).reshape(-1)

def compute_feature_importance(model: CatBoostClassifier) -> pd.DataFrame:
    log("        compute_feature_importance start")

    imp = model.get_feature_importance()
    features = list(model.feature_names_)

    if len(features) != len(imp):
        raise ValueError(
            f"Feature name count {len(features)} does not match importance length {len(imp)}"
        )

    out = pd.DataFrame(
        {
            "feature": features,
            "importance": np.asarray(imp).reshape(-1),
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    log("        compute_feature_importance done")
    return out


def compute_shap_importance(model: CatBoostClassifier, X_test: pd.DataFrame) -> pd.DataFrame:
    log(f"        compute_shap_importance X_test_shape {X_test.shape}")

    X_use = X_test
    model_features = model.feature_names_
    if model_features:
        missing = [f for f in model_features if f not in X_use.columns]
        if missing:
            raise ValueError(f"X is missing model features {missing[:10]}")
        X_use = X_use[model_features]
        log(f"        aligned X_use to model feature order n_features {len(model_features)}")

    pool = Pool(X_use)
    sv = model.get_feature_importance(pool, type="ShapValues", thread_count=1)
    sv = np.asarray(sv)
    log(f"        shap_values_shape {sv.shape} ndim {sv.ndim}")

    if sv.ndim == 2:
        imp = np.abs(sv[:, :-1]).mean(axis=0)
    elif sv.ndim == 3:
        if sv.shape[1] == X_use.shape[1] + 1:
            imp = np.abs(sv[:, :-1, :]).mean(axis=0).mean(axis=-1)
        elif sv.shape[2] == X_use.shape[1] + 1:
            imp = np.abs(sv[:, :, :-1]).mean(axis=0).mean(axis=0)
        else:
            raise ValueError(f"Unexpected ShapValues shape {sv.shape}")
    else:
        raise ValueError(f"Unexpected ShapValues ndim {sv.ndim} shape {sv.shape}")

    imp = np.asarray(imp).reshape(-1)
    if imp.shape[0] != X_use.shape[1]:
        raise ValueError(f"Importance length {imp.shape[0]} does not match n_features {X_use.shape[1]}")

    out = pd.DataFrame({"feature": list(X_use.columns), "importance": imp})
    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    log("        compute_shap_importance done")
    return out


def compute_permutation_importance(
    model: CatBoostClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    log(f"        compute_permutation_importance X_test_shape {X_test.shape} y_len {len(y_test)} repeats {n_repeats}")

    def scorer(estimator, X, y):
        y_pred = catboost_predict_labels(estimator, X)
        return f1_score(y, y_pred, average="macro")

    r = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )

    out = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance": r.importances_mean,
            "importance_std_repeat": r.importances_std,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    log("        compute_permutation_importance done")
    return out

def analyze_one_task_feature_importance(task: TaskInfo, add: str) -> bool:
    log("------------------------------------------------------------")
    log(f"Analyze feature importance dataset {task.dataset_name} task {task.task_key}")

    _, out_tables, out_agg, out_plots = task_output_dirs(task.dataset_name, add, task.task_key)

    split_dirs = list_dirs(os.path.join(task.task_dir, "splits"), prefix="split_")
    if not split_dirs:
        raise FileNotFoundError("No split folders found")
    log(f"  split_dirs {len(split_dirs)}")

    fi_rows: List[pd.DataFrame] = []

    for k, sd in enumerate(split_dirs, start=1):
        log(f"    Split {k} of {len(split_dirs)} {sd}")

        model_path = os.path.join(sd, "catboost_model.cbm")
        if not os.path.exists(model_path):
            log("      missing model skipping")
            continue

        model = CatBoostClassifier()
        model.load_model(model_path)
        log("      model loaded")

        log("      Feature importance start")
        df_fi = compute_feature_importance(model)
        df_fi["split"] = k

        df_fi.to_csv(os.path.join(sd, "feature_importance.csv"), index=False)
        fi_rows.append(df_fi[["feature", "importance", "split"]])

        write_running_aggregate(
            rows=fi_rows,
            out_tables=out_tables,
            out_agg=out_agg,
            out_plots=out_plots,
            prefix="feature",
            dataset_name=task.dataset_name,
            task_key=task.task_key,
        )
        log("      Feature importance done")

    finalize_aggregate(
        rows=fi_rows,
        out_tables=out_tables,
        out_agg=out_agg,
        out_plots=out_plots,
        prefix="feature",
        dataset_name=task.dataset_name,
        task_key=task.task_key,
    )

    log(f"Analyze feature importance done dataset {task.dataset_name} task {task.task_key}")
    return bool(fi_rows)


def task_output_dirs(dataset_name: str, add: str, task_key: str) -> Tuple[str, str, str, str]:
    out_root = os.path.join(RESULTS_ROOT, dataset_name, f"exp_{add}", task_key)
    out_tables = os.path.join(out_root, "tables")
    out_agg = os.path.join(out_root, "aggregated_results")
    out_plots = os.path.join(out_root, "plots")
    ensure_dir(out_tables)
    ensure_dir(out_agg)
    ensure_dir(out_plots)
    return out_root, out_tables, out_agg, out_plots


def write_running_aggregate(
    rows: List[pd.DataFrame],
    out_tables: str,
    out_agg: str,
    out_plots: str,
    prefix: str,
    dataset_name: str,
    task_key: str,
) -> None:
    if not rows:
        return

    all_df = pd.concat(rows, ignore_index=True)
    all_path = os.path.join(out_tables, f"{prefix}_importance_all_splits_running.csv")
    all_df.to_csv(all_path, index=False)

    stats = compute_mean_std(all_df)
    stats_path = os.path.join(out_agg, f"{prefix}_importance_mean_std_running.csv")
    stats.to_csv(stats_path, index=False)

    top_path = os.path.join(out_agg, f"{prefix}_importance_top{TOPK_FEATURES}_running.csv")
    stats.head(TOPK_FEATURES).to_csv(top_path, index=False)

    if prefix == "shap":
        xlabel = "mean abs SHAP"
        title = f"{dataset_name} {task_key} SHAP"
    elif prefix == "perm":
        xlabel = "mean macro F1 drop"
        title = f"{dataset_name} {task_key} permutation"
    elif prefix == "feature":
        xlabel = "mean feature importance"
        title = f"{dataset_name} {task_key} feature importance"
    else:
        raise ValueError(f"Unsupported prefix {prefix}")

    fig = plot_importance_mean_std(stats, title=title, xlabel=xlabel, topk=TOPK_FEATURES)
    fig_path = os.path.join(out_plots, f"{dataset_name}_{task_key}_{prefix}_top{TOPK_FEATURES}_running.png")
    save_png(fig, fig_path)
    plt.close(fig)


def finalize_aggregate(
    rows: List[pd.DataFrame],
    out_tables: str,
    out_agg: str,
    out_plots: str,
    prefix: str,
    dataset_name: str,
    task_key: str,
) -> None:
    if not rows:
        return

    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(os.path.join(out_tables, f"{prefix}_importance_all_splits.csv"), index=False)

    stats = compute_mean_std(all_df)
    stats.to_csv(os.path.join(out_agg, f"{prefix}_importance_mean_std.csv"), index=False)
    stats.head(TOPK_FEATURES).to_csv(os.path.join(out_agg, f"{prefix}_importance_top{TOPK_FEATURES}.csv"), index=False)

    if prefix == "shap":
        xlabel = "mean abs SHAP"
        title = f"{dataset_name} {task_key} SHAP"
    elif prefix == "perm":
        xlabel = "mean macro F1 drop"
        title = f"{dataset_name} {task_key} permutation"
    elif prefix == "feature":
        xlabel = "mean feature importance"
        title = f"{dataset_name} {task_key} feature importance"
    else:
        raise ValueError(f"Unsupported prefix {prefix}")

    fig = plot_importance_mean_std(stats, title=title, xlabel=xlabel, topk=TOPK_FEATURES)
    save_png(fig, os.path.join(out_plots, f"{dataset_name}_{task_key}_{prefix}_top{TOPK_FEATURES}.png"))
    plt.close(fig)


def analyze_one_task_shap(task: TaskInfo, add: str) -> bool:
    log("------------------------------------------------------------")
    log(f"Analyze SHAP dataset {task.dataset_name} task {task.task_key}")

    df = load_task_dataframe(task.dataset_name, task.task_key)
    _, out_tables, out_agg, out_plots = task_output_dirs(task.dataset_name, add, task.task_key)

    split_dirs = list_dirs(os.path.join(task.task_dir, "splits"), prefix="split_")
    if not split_dirs:
        raise FileNotFoundError("No split folders found")
    log(f"  split_dirs {len(split_dirs)}")

    shap_rows: List[pd.DataFrame] = []

    for k, sd in enumerate(split_dirs, start=1):
        log(f"    Split {k} of {len(split_dirs)} {sd}")

        model_path = os.path.join(sd, "catboost_model.cbm")
        balanced_idx_path = os.path.join(sd, "balanced_row_indices.json")
        if not os.path.exists(model_path) or not os.path.exists(balanced_idx_path):
            log("      missing model or balanced_row_indices skipping")
            continue

        test_row_indices = load_balanced_test_row_indices(sd)

        model = CatBoostClassifier()
        model.load_model(model_path)
        log("      model loaded")

        X_test, _ = get_xy_by_row_indices(df, row_indices=test_row_indices, task_key=task.task_key)

        log("      SHAP start")
        df_shap = compute_shap_importance(model, X_test)
        df_shap["split"] = k

        df_shap.to_csv(os.path.join(sd, "shap_importance.csv"), index=False)
        shap_rows.append(df_shap[["feature", "importance", "split"]])

        write_running_aggregate(
            rows=shap_rows,
            out_tables=out_tables,
            out_agg=out_agg,
            out_plots=out_plots,
            prefix="shap",
            dataset_name=task.dataset_name,
            task_key=task.task_key,
        )
        log("      SHAP done")

    finalize_aggregate(
        rows=shap_rows,
        out_tables=out_tables,
        out_agg=out_agg,
        out_plots=out_plots,
        prefix="shap",
        dataset_name=task.dataset_name,
        task_key=task.task_key,
    )

    log(f"Analyze SHAP done dataset {task.dataset_name} task {task.task_key}")
    return bool(shap_rows)


def analyze_one_task_perm(task: TaskInfo, add: str) -> bool:
    log("------------------------------------------------------------")
    log(f"Analyze permutation dataset {task.dataset_name} task {task.task_key}")

    df = load_task_dataframe(task.dataset_name, task.task_key)
    _, out_tables, out_agg, out_plots = task_output_dirs(task.dataset_name, add, task.task_key)

    split_dirs = list_dirs(os.path.join(task.task_dir, "splits"), prefix="split_")
    if not split_dirs:
        raise FileNotFoundError("No split folders found")
    log(f"  split_dirs {len(split_dirs)}")

    perm_rows: List[pd.DataFrame] = []

    for k, sd in enumerate(split_dirs, start=1):
        log(f"    Split {k} of {len(split_dirs)} {sd}")

        model_path = os.path.join(sd, "catboost_model.cbm")
        balanced_idx_path = os.path.join(sd, "balanced_row_indices.json")
        if not os.path.exists(model_path) or not os.path.exists(balanced_idx_path):
            log("      missing model or balanced_row_indices skipping")
            continue

        test_row_indices = load_balanced_test_row_indices(sd)

        model = CatBoostClassifier()
        model.load_model(model_path)
        log("      model loaded")

        X_test, y_test = get_xy_by_row_indices(df, row_indices=test_row_indices, task_key=task.task_key)

        log("      Permutation start")
        df_perm = compute_permutation_importance(
            model=model,
            X_test=X_test,
            y_test=y_test,
            n_repeats=PERM_N_REPEATS,
            random_state=PERM_RANDOM_STATE,
        )
        df_perm["split"] = k

        df_perm.to_csv(os.path.join(sd, "permutation_importance.csv"), index=False)
        perm_rows.append(df_perm[["feature", "importance", "split"]])

        write_running_aggregate(
            rows=perm_rows,
            out_tables=out_tables,
            out_agg=out_agg,
            out_plots=out_plots,
            prefix="perm",
            dataset_name=task.dataset_name,
            task_key=task.task_key,
        )
        log("      Permutation done")

    finalize_aggregate(
        rows=perm_rows,
        out_tables=out_tables,
        out_agg=out_agg,
        out_plots=out_plots,
        prefix="perm",
        dataset_name=task.dataset_name,
        task_key=task.task_key,
    )

    log(f"Analyze permutation done dataset {task.dataset_name} task {task.task_key}")
    return bool(perm_rows)


def main(use_case: int) -> None:
    if use_case not in USE_CASES:
        raise ValueError(f"use_case must be one of {sorted(USE_CASES.keys())}, got {use_case}")

    use_case_tag = str(USE_CASES[use_case]["name"]).lower()

    log("============================================================")
    log("XAI posthoc analysis start")
    log(f"BASE_PROJECT {BASE_PROJECT}")
    log(f"ADD {ADD}")
    log(f"ML_EXPERIMENTS_ROOT {ML_EXPERIMENTS_ROOT}")
    log(f"ML_TASKS_ROOT {ML_TASKS_ROOT}")
    log(f"RESULTS_ROOT {RESULTS_ROOT}")
    log(f"use_case {use_case} name {USE_CASES[use_case]['name']}")
    log("============================================================")

    tasks = discover_all_tasks(ML_EXPERIMENTS_ROOT, add=ADD, use_case=use_case)

    summary_rows: List[Dict[str, Any]] = []

    log("============================================================")
    log("Phase 1 feature importance for all tasks")
    log("============================================================")
    for i, t in enumerate(tasks, start=1):
        log("============================================================")
        log(f"Task {i} of {len(tasks)} feature importance {t.dataset_name} {t.task_key}")
        fi_ok = analyze_one_task_feature_importance(t, add=ADD)
        summary_rows.append(
            {
                "dataset": t.dataset_name,
                "task": t.task_key,
                "feature_done": int(bool(fi_ok)),
                "shap_done": 0,
                "perm_done": 0,
            }
        )

        out_global = os.path.join(RESULTS_ROOT, "aggregated_results")
        ensure_dir(out_global)
        out_summary_running = os.path.join(out_global, f"summary_running_{use_case_tag}_{ADD}.csv")
        pd.DataFrame(summary_rows).to_csv(out_summary_running, index=False)

    log("============================================================")
    log("Phase 2 SHAP for all tasks")
    log("============================================================")
    for i, t in enumerate(tasks, start=1):
        log("============================================================")
        log(f"Task {i} of {len(tasks)} SHAP {t.dataset_name} {t.task_key}")
        shap_ok = analyze_one_task_shap(t, add=ADD)

        for r in summary_rows:
            if r["dataset"] == t.dataset_name and r["task"] == t.task_key:
                r["shap_done"] = int(bool(shap_ok))
                break

        out_global = os.path.join(RESULTS_ROOT, "aggregated_results")
        ensure_dir(out_global)
        out_summary_running = os.path.join(out_global, f"summary_running_{use_case_tag}_{ADD}.csv")
        pd.DataFrame(summary_rows).to_csv(out_summary_running, index=False)

    log("============================================================")
    log("Phase 3 permutation for all tasks")
    log("============================================================")
    for i, t in enumerate(tasks, start=1):
        log("============================================================")
        log(f"Task {i} of {len(tasks)} permutation {t.dataset_name} {t.task_key}")
        perm_ok = analyze_one_task_perm(t, add=ADD)

        for r in summary_rows:
            if r["dataset"] == t.dataset_name and r["task"] == t.task_key:
                r["perm_done"] = int(bool(perm_ok))
                break

        out_global = os.path.join(RESULTS_ROOT, "aggregated_results")
        ensure_dir(out_global)
        out_summary_running = os.path.join(out_global, f"summary_running_{use_case_tag}_{ADD}.csv")
        pd.DataFrame(summary_rows).to_csv(out_summary_running, index=False)

    out_global = os.path.join(RESULTS_ROOT, "aggregated_results")
    ensure_dir(out_global)
    out_summary = os.path.join(out_global, f"summary_feature_shap_perm_{use_case_tag}_{ADD}.csv")
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    log(f"Wrote summary {out_summary}")
    log("XAI posthoc analysis done")


if __name__ == "__main__":
    for use_case in range(START_USE_CASE, max(USE_CASES.keys()) + 1):
        main(use_case)