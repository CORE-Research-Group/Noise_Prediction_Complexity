"""
ML experiments analysis post hoc with plots and aggregated tables in PNG and EPS

This script analyzes the outputs produced by the ML experiments runner
CatBoost with repeated group aware train test splits

Input folder structure created by the runner

  ML_experiments/<DATASET_NAME>/exp_<add>/<task_key>/
    splits/split_XXX/
      confusion_matrix.csv
      classification_report.csv
      feature_importance.csv
      shap_importance.csv                  optional
      permutation_importance.csv           optional
      catboost_model.cbm                   optional
      split_ids.json
    aggregate/
      split_metrics.csv
      metrics_summary.json
      confusion_matrix_mean.csv
      confusion_matrix_mean_normalized.csv
      feature_importance_all_splits.csv
      feature_importance_mean_std.csv
      report.txt

What this analysis script does

- Creates a new output root at
  Results/<DATASET_NAME>/exp_<add>/

- For each task
  - Reads per split confusion matrices and computes
      * mean and std for absolute confusion matrix cells
      * mean and std for row normalized confusion matrices where each split is normalized first
  - Reproduces confusion matrix plots
      * per split confusion matrix with absolute counts
      * average confusion matrix with row normalized means
    and saves them as PNG and EPS
  - Produces aggregated confusion matrix plots with variability
      * mean ± std in each cell for the absolute confusion matrix
      * mean ± std in each cell for the row normalized confusion matrix
    and saves them as PNG and EPS
  - Reads per split feature importances and produces
      * top 20 barplot with mean ± std
    and saves it as PNG and EPS
  - Optionally reads per split SHAP importances and permutation importances if available
      * top 20 SHAP importance plot with mean ± std
      * top 20 permutation importance plot with mean ± std
    and saves them as PNG and EPS
  - Produces aggregated tables mainly as CSV
      * split_metrics copied from input
      * metrics_mean_std
      * confusion_matrix_abs_mean
      * confusion_matrix_abs_std
      * confusion_matrix_norm_mean
      * confusion_matrix_norm_std
      * feature_importance_mean_std and top20
      * shap_importance_mean_std and top20 when available
      * permutation_importance_mean_std and top20 when available

- Also creates a single Excel summary across all tasks at
  Results/<DATASET_NAME>/exp_<add>/aggregated_results/summary_all_tasks.xlsx

- Additionally saves all averaged row normalized confusion matrices in a global collection folder
  Results/all_results/confmat_norm_average_all/

Train test split semantics match the runner

- Splits are performed at group level using signal_id and not at row or window level
- signal_id is exactly the run_id as a string such as 1 2 ... 20
- Each split
    1) Collects the unique run_ids present through signal_id
    2) Picks a random test subset of size floor TEST_SIZE times n_runs clamped to a valid range if possible
    3) Uses the complement as training set
    4) Assigns all rows or windows of a run_id fully to either train or test

- Unique combination rule on runner side
    * The runner enforces unique test run subsets across splits
    * The maximum number of unique splits is C n_runs comma n_test

Therefore

- No leakage because the model never sees windows from the same run_id in both train and test
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib

# Keep the backend consistent with the original scripts
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# User settings
# =============================================================================

# Must match experiment runner outputs
# DATASET_NAME = "ECG"
# DATASET_NAME = "Roessler"
# DATASET_NAME = "Lorenz"
# DATASET_NAME = "Henon"
# DATASET_NAME = "AR1"
DATASET_NAME = "AR1"

add = "ed10_td1_mc300"

ML_EXPERIMENTS_ROOT = "ML_experiments"
RESULTS_ROOT = "Results"

# Global folder that collects ALL averaged normalized confusion matrices
ALL_RESULTS_ROOT = os.path.join(RESULTS_ROOT, "all_results")
ALL_CM_NORM_AVG_ROOT = os.path.join(ALL_RESULTS_ROOT, "confmat_norm_average_all")

# Plot settings
DPI = 500
TOPK_FEATURES = 20

# -------------------------------------------------------------------------
# Plot styling switches
# -------------------------------------------------------------------------
FONT_SCALE = 1.5
SHOW_TITLES = False
USE_CUSTOM_COLORS = False

# Custom palette
CUSTOM_PALETTE = ["#E2DBBE", "#D5D6AA", "#9DBBAE", "#769FB6", "#188FA7"]


# =============================================================================
# Helpers plotting and IO
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_png_eps(fig: plt.Figure, out_base_no_ext: str, dpi: int = 200) -> None:
    ensure_dir(os.path.dirname(out_base_no_ext))
    fig.savefig(out_base_no_ext + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base_no_ext + ".eps", bbox_inches="tight")


def save_png_eps_twice(
    fig: plt.Figure,
    out_base_regular: str,
    out_base_global: str,
    dpi: int = 200,
) -> None:
    save_png_eps(fig, out_base_regular, dpi=dpi)
    save_png_eps(fig, out_base_global, dpi=dpi)

def get_system_display_name(dataset_name: str) -> str:
    mapping = {
        "Roessler": "Rössler",
        "Henon": "Hénon",
        "Lorenz": "Lorenz",
        "ECG": "ECG",
        "AR1": "AR1",
    }
    return mapping.get(dataset_name, dataset_name)

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_split_dirs(task_dir: str) -> List[str]:
    splits_dir = os.path.join(task_dir, "splits")
    if not os.path.isdir(splits_dir):
        raise FileNotFoundError(f"Missing splits folder: {splits_dir}")
    subdirs = []
    for name in os.listdir(splits_dir):
        full = os.path.join(splits_dir, name)
        if os.path.isdir(full) and name.startswith("split_"):
            subdirs.append(full)
    if not subdirs:
        raise FileNotFoundError(f"No split_XXX folders found under: {splits_dir}")
    return sorted(subdirs)


def infer_labels_from_cm_df(cm_df: pd.DataFrame) -> List[str]:
    # expects index like true_<label>, columns like pred_<label>
    idx = [str(x) for x in cm_df.index.tolist()]
    cols = [str(x) for x in cm_df.columns.tolist()]
    labels_true = [s.replace("true_", "", 1) if s.startswith("true_") else s for s in idx]
    labels_pred = [s.replace("pred_", "", 1) if s.startswith("pred_") else s for s in cols]
    if labels_true == labels_pred:
        return labels_true

    out = []
    for x in labels_true + labels_pred:
        if x not in out:
            out.append(x)
    return out


# -------------------------------------------------------------------------
# Styling helpers
# -------------------------------------------------------------------------

def _scaled(v: float) -> float:
    return float(v) * float(FONT_SCALE)


def get_cm_cmap():
    if not USE_CUSTOM_COLORS:
        return "Blues"
    return LinearSegmentedColormap.from_list("custom_palette", CUSTOM_PALETTE, N=256)


FI_TITLE_FS = _scaled(14)
FI_LABEL_FS = _scaled(12)
FI_TICK_FS = _scaled(10)
CM_FIXED_TITLE_FS = _scaled(12)
CM_FIXED_AXISLABEL_FS = _scaled(12)

def get_cm_style(n_labels: int, task_name: str) -> Dict[str, float]:
    """
    Dynamically scale figure and fonts so larger confusion matrices
    do not overflow the borders.
    """
    base_tick = _scaled(10)
    base_annot = _scaled(10)


    # Keep only class tick labels larger for task 2
    tick_label_multiplier = 2.0 if "task2" in str(task_name).lower() else 1.0

    # Figure grows with the number of classes
    fig_w = max(6.0, 0.72 * n_labels + 2.8)
    fig_h = max(5.0, 0.60 * n_labels + 2.2)

    if n_labels <= 4:
        tick_fs = base_tick * tick_label_multiplier
        annot_fs = base_annot
    elif n_labels <= 6:
        tick_fs = max(9.0, base_tick * 0.92) * tick_label_multiplier
        annot_fs = max(8.0, base_annot * 0.88)
    elif n_labels <= 8:
        tick_fs = max(8.0, base_tick * 0.82) * tick_label_multiplier
        annot_fs = max(14.0, base_annot * 1.50)
    elif n_labels <= 10:
        tick_fs = max(7.0, base_tick * 0.72) * tick_label_multiplier
        annot_fs = max(12.0, base_annot * 1.25)
    else:
        tick_fs = max(6.0, base_tick * 0.62) * tick_label_multiplier
        annot_fs = max(5.0, base_annot * 0.52)

    return {
        "fig_w": float(fig_w),
        "fig_h": float(fig_h),
        "tick_fs": float(tick_fs),
        "annot_fs": float(annot_fs),
    }
def plot_confusion_matrix_absolute(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    task_name: str,
) -> plt.Figure:
    style = get_cm_style(len(labels), task_name)
    fig, ax = plt.subplots(figsize=(style["fig_w"], style["fig_h"]))
    im = ax.imshow(cm, interpolation="nearest", cmap=get_cm_cmap())
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=style["tick_fs"])

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
    )

    ax.set_xlabel("Predicted label", fontsize=CM_FIXED_AXISLABEL_FS)
    ax.set_ylabel("True label", fontsize=CM_FIXED_AXISLABEL_FS)

    if SHOW_TITLES:
        ax.set_title(title, fontsize=CM_FIXED_TITLE_FS)

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=style["tick_fs"],
    )
    plt.setp(ax.get_yticklabels(), fontsize=style["tick_fs"])

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                fontsize=style["annot_fs"],
                color="white" if cm[i, j] > thresh else "black",
                clip_on=True,
            )

    fig.tight_layout()
    return fig


def plot_confusion_matrix_normalized(
    mean_norm: np.ndarray,
    labels: List[str],
    title: str,
    task_name: str,
) -> plt.Figure:
    style = get_cm_style(len(labels), task_name)
    fig, ax = plt.subplots(figsize=(style["fig_w"], style["fig_h"]))
    im = ax.imshow(mean_norm, interpolation="nearest", cmap=get_cm_cmap())
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=style["tick_fs"])

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
    )

    ax.set_xlabel("Predicted label", fontsize=CM_FIXED_AXISLABEL_FS)
    ax.set_ylabel("True label", fontsize=CM_FIXED_AXISLABEL_FS)

    if SHOW_TITLES:
        ax.set_title(title, fontsize=CM_FIXED_TITLE_FS)

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=style["tick_fs"],
    )
    plt.setp(ax.get_yticklabels(), fontsize=style["tick_fs"])

    for i in range(mean_norm.shape[0]):
        for j in range(mean_norm.shape[1]):
            ax.text(
                j,
                i,
                f"{mean_norm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=style["annot_fs"],
                color="white" if mean_norm[i, j] > 0.5 else "black",
                clip_on=True,
            )

    fig.tight_layout()
    return fig


def plot_confusion_matrix_mean_std(
    mean: np.ndarray,
    std: np.ndarray,
    labels: List[str],
    title: str,
    task_name: str,
    mean_fmt: str,
    std_fmt: str,
    thresh_mode: str = "mean",
) -> plt.Figure:
    style = get_cm_style(len(labels), task_name)

    # mean std needs a bit more room vertically
    fig, ax = plt.subplots(figsize=(style["fig_w"], style["fig_h"] + 0.8))
    im = ax.imshow(mean, interpolation="nearest", cmap=get_cm_cmap())
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=style["tick_fs"])

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
    )

    ax.set_xlabel("Predicted label", fontsize=CM_FIXED_AXISLABEL_FS)
    ax.set_ylabel("True label", fontsize=CM_FIXED_AXISLABEL_FS)

    if SHOW_TITLES:
        ax.set_title(title, fontsize=CM_FIXED_TITLE_FS)

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=style["tick_fs"],
    )
    plt.setp(ax.get_yticklabels(), fontsize=style["tick_fs"])

    if thresh_mode == "mean":
        thresh = float(np.max(mean)) / 2.0 if mean.size > 0 else 0.5
        ref = mean
    else:
        thresh = float(np.max(std)) / 2.0 if std.size > 0 else 0.5
        ref = std

    annot_fs = max(4.5, style["annot_fs"] * 0.88)

    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            txt = f"{mean_fmt.format(float(mean[i, j]))}\n± {std_fmt.format(float(std[i, j]))}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=annot_fs,
                color="white" if float(ref[i, j]) > thresh else "black",
                clip_on=True,
            )

    fig.tight_layout()
    return fig


def plot_feature_importance_mean_std(fi_stats: pd.DataFrame, task_name: str, topk: int = 20) -> plt.Figure:
    df = fi_stats.sort_values("mean", ascending=False).head(topk).copy()
    df = df.iloc[::-1]

    fig = plt.figure(figsize=(8, 5))
    plt.barh(df["feature"], df["mean"], xerr=df["std"], capsize=4)

    plt.xlabel("Feature Importance (mean ± std)", fontsize=FI_LABEL_FS)
    if SHOW_TITLES:
        plt.title(f"Average Feature Importance – {task_name}", fontsize=FI_TITLE_FS)

    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=FI_TICK_FS)

    plt.tight_layout()
    return fig


def plot_importance_mean_std(fi_stats: pd.DataFrame, task_name: str, x_label: str, topk: int = 20) -> plt.Figure:
    df = fi_stats.sort_values("mean", ascending=False).head(topk).copy()
    df = df.iloc[::-1]

    fig = plt.figure(figsize=(8, 5))
    plt.barh(df["feature"], df["mean"], xerr=df["std"], capsize=4)

    plt.xlabel(x_label, fontsize=FI_LABEL_FS)
    if SHOW_TITLES:
        plt.title(f"{task_name}", fontsize=FI_TITLE_FS)

    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=FI_TICK_FS)

    plt.tight_layout()
    return fig


def compute_mean_std(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    stack shape is n_splits, n, m
    returns mean and std with ddof 1 if n_splits > 1 else 0
    """
    n = stack.shape[0]
    mean = stack.mean(axis=0)
    if n > 1:
        std = stack.std(axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)
    return mean, std


def row_normalize(cm: np.ndarray) -> np.ndarray:
    row_sum = cm.sum(axis=1, keepdims=True) + 1e-12
    return cm / row_sum


# =============================================================================
# Task discovery
# =============================================================================

@dataclass(frozen=True)
class TaskInfo:
    task_key: str
    task_dir: str


def discover_tasks(exp_root: str) -> List[TaskInfo]:
    """
    Finds all task folders under
      ML_experiments/<DATASET_NAME>/exp_<add>/
    that contain a splits directory
    """
    if not os.path.isdir(exp_root):
        raise FileNotFoundError(f"Experiment root not found: {exp_root}")

    tasks: List[TaskInfo] = []
    for name in sorted(os.listdir(exp_root)):
        task_dir = os.path.join(exp_root, name)
        if os.path.isdir(task_dir) and os.path.isdir(os.path.join(task_dir, "splits")):
            tasks.append(TaskInfo(task_key=name, task_dir=task_dir))

    if not tasks:
        raise FileNotFoundError(f"No task folders with splits found under: {exp_root}")

    return tasks


# =============================================================================
# Core per task analysis
# =============================================================================

def analyze_one_task(
    task: TaskInfo,
    out_task_dir: str,
) -> Dict[str, Any]:
    """
    Reads per split artifacts and writes

      Results/<DATASET>/exp_<add>/<task_key>/
        plots/
        aggregated_results/
        tables/

    Additionally copies the averaged normalized confusion matrix into

      Results/all_results/confmat_norm_average_all/

    as PNG and EPS
    """
    ensure_dir(out_task_dir)
    ensure_dir(ALL_CM_NORM_AVG_ROOT)

    out_plots = os.path.join(out_task_dir, "plots")
    out_tables = os.path.join(out_task_dir, "tables")
    out_agg = os.path.join(out_task_dir, "aggregated_results")

    ensure_dir(out_plots)
    ensure_dir(out_tables)
    ensure_dir(out_agg)

    split_dirs = list_split_dirs(task.task_dir)

    split_metrics_path = os.path.join(task.task_dir, "aggregate", "split_metrics.csv")
    metrics_df = load_csv(split_metrics_path) if os.path.exists(split_metrics_path) else None
    if metrics_df is not None:
        metrics_df.to_csv(os.path.join(out_tables, "split_metrics.csv"), index=False)

    plot_prefix = f"{DATASET_NAME}_{task.task_key}_"
    system_title = get_system_display_name(DATASET_NAME)

    # -------------------------
    # Confusion matrices
    # -------------------------
    cm_abs_list: List[np.ndarray] = []
    cm_norm_list: List[np.ndarray] = []
    labels: Optional[List[str]] = None

    for k, sd in enumerate(split_dirs, start=1):
        cm_path = os.path.join(sd, "confusion_matrix.csv")
        cm_df = load_csv(cm_path)

        if cm_df.shape[1] >= 2 and str(cm_df.columns[0]).startswith("Unnamed"):
            cm_df = cm_df.set_index(cm_df.columns[0])

        if labels is None:
            labels = infer_labels_from_cm_df(cm_df)

            tk = str(task.task_key).lower()
            if ("task3" in tk or "noise_present" in tk) and len(labels) == 2:
                _labs = [str(x).strip() for x in labels]
                if set(_labs) == {"0", "1"}:
                    mapping = {"0": "no noise", "1": "noise"}
                    labels = [mapping[x] for x in _labs]

        cm_abs = cm_df.to_numpy(dtype=float)
        cm_abs_list.append(cm_abs)

        cm_norm = row_normalize(cm_abs)
        cm_norm_list.append(cm_norm)

        # per split absolute plot saved only once
        fig = plot_confusion_matrix_absolute(
            cm_abs,
            labels=labels,
            title=system_title,
            task_name=task.task_key,
        )
        out_base = os.path.join(out_plots, f"{plot_prefix}confmat_abs_split_{k:03d}")
        save_png_eps(fig, out_base, dpi=DPI)
        plt.close(fig)

    cm_abs_stack = np.stack(cm_abs_list, axis=0)
    cm_norm_stack = np.stack(cm_norm_list, axis=0)

    cm_abs_mean, cm_abs_std = compute_mean_std(cm_abs_stack)
    cm_norm_mean, cm_norm_std = compute_mean_std(cm_norm_stack)

    assert labels is not None
    idx = [f"true_{l}" for l in labels]
    cols = [f"pred_{l}" for l in labels]

    pd.DataFrame(cm_abs_mean, index=idx, columns=cols).to_csv(os.path.join(out_agg, "confusion_matrix_abs_mean.csv"))
    pd.DataFrame(cm_abs_std, index=idx, columns=cols).to_csv(os.path.join(out_agg, "confusion_matrix_abs_std.csv"))
    pd.DataFrame(cm_norm_mean, index=idx, columns=cols).to_csv(os.path.join(out_agg, "confusion_matrix_norm_mean.csv"))
    pd.DataFrame(cm_norm_std, index=idx, columns=cols).to_csv(os.path.join(out_agg, "confusion_matrix_norm_std.csv"))

    # average normalized plot saved in the task folder and also in one global folder
    fig = plot_confusion_matrix_normalized(
        cm_norm_mean,
        labels=labels,
        title=system_title,
        task_name=task.task_key,
    )
    save_png_eps_twice(
        fig,
        os.path.join(out_plots, f"{plot_prefix}confmat_norm_average"),
        os.path.join(ALL_CM_NORM_AVG_ROOT, f"{plot_prefix}confmat_norm_average"),
        dpi=DPI,
    )
    plt.close(fig)

    # absolute mean std plot saved only once
    fig = plot_confusion_matrix_mean_std(
        mean=cm_abs_mean,
        std=cm_abs_std,
        labels=labels,
        title=system_title,
        task_name=task.task_key,
        mean_fmt="{:.0f}",
        std_fmt="{:.2f}",
        thresh_mode="mean",
    )
    out_base = os.path.join(out_plots, f"{plot_prefix}confmat_abs_average_mean_std")
    save_png_eps(fig, out_base, dpi=DPI)
    plt.close(fig)

    # normalized mean std plot saved only once in the task folder
    fig = plot_confusion_matrix_mean_std(
        mean=cm_norm_mean,
        std=cm_norm_std,
        labels=labels,
        title=system_title,
        task_name=task.task_key,
        mean_fmt="{:.2f}",
        std_fmt="{:.3f}",
        thresh_mode="mean",
    )
    save_png_eps(
        fig,
        os.path.join(out_plots, f"{plot_prefix}confmat_norm_average_mean_std"),
        dpi=DPI,
    )
    plt.close(fig)

    # -------------------------
    # Feature importance
    # -------------------------
    fi_all_rows: List[pd.DataFrame] = []
    for k, sd in enumerate(split_dirs, start=1):
        fi_path = os.path.join(sd, "feature_importance.csv")
        fi_df = load_csv(fi_path)

        if "feature" not in fi_df.columns and "Feature" in fi_df.columns:
            fi_df = fi_df.rename(columns={"Feature": "feature"})
        if "importance" not in fi_df.columns and "Importance" in fi_df.columns:
            fi_df = fi_df.rename(columns={"Importance": "importance"})
        if "split" not in fi_df.columns:
            fi_df["split"] = k

        fi_all_rows.append(fi_df)

    fi_all = pd.concat(fi_all_rows, ignore_index=True)
    fi_all.to_csv(os.path.join(out_tables, "feature_importance_all_splits.csv"), index=False)

    fi_stats = (
        fi_all.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fi_stats = fi_stats.sort_values("mean", ascending=False)

    fi_stats.to_csv(os.path.join(out_agg, "feature_importance_mean_std.csv"), index=False)
    fi_stats.head(TOPK_FEATURES).to_csv(os.path.join(out_agg, f"feature_importance_top{TOPK_FEATURES}.csv"), index=False)

    fig = plot_feature_importance_mean_std(fi_stats, task_name=task.task_key, topk=TOPK_FEATURES)
    out_base = os.path.join(out_plots, f"{plot_prefix}feature_importance_top{TOPK_FEATURES}_mean_std")
    save_png_eps(fig, out_base, dpi=DPI)
    plt.close(fig)

    # -------------------------
    # SHAP importance
    # -------------------------
    shap_all_rows: List[pd.DataFrame] = []
    for k, sd in enumerate(split_dirs, start=1):
        p = os.path.join(sd, "shap_importance.csv")
        if not os.path.exists(p):
            continue
        df_shap = load_csv(p)

        if "feature" not in df_shap.columns and "Feature" in df_shap.columns:
            df_shap = df_shap.rename(columns={"Feature": "feature"})
        if "importance" not in df_shap.columns and "Importance" in df_shap.columns:
            df_shap = df_shap.rename(columns={"Importance": "importance"})
        if "split" not in df_shap.columns:
            df_shap["split"] = k

        shap_all_rows.append(df_shap)

    if shap_all_rows:
        shap_all = pd.concat(shap_all_rows, ignore_index=True)
        shap_all.to_csv(os.path.join(out_tables, "shap_importance_all_splits.csv"), index=False)

        shap_stats = (
            shap_all.groupby("feature")["importance"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )

        shap_stats.to_csv(os.path.join(out_agg, "shap_importance_mean_std.csv"), index=False)
        shap_stats.head(TOPK_FEATURES).to_csv(
            os.path.join(out_agg, f"shap_importance_top{TOPK_FEATURES}.csv"),
            index=False,
        )

        fig = plot_importance_mean_std(
            shap_stats,
            task_name=task.task_key,
            x_label="SHAP importance (mean ± std)",
            topk=TOPK_FEATURES,
        )
        out_base = os.path.join(out_plots, f"{plot_prefix}shap_importance_top{TOPK_FEATURES}_mean_std")
        save_png_eps(fig, out_base, dpi=DPI)
        plt.close(fig)

    # -------------------------
    # Permutation importance
    # -------------------------
    perm_all_rows: List[pd.DataFrame] = []
    for k, sd in enumerate(split_dirs, start=1):
        p = os.path.join(sd, "permutation_importance.csv")
        if not os.path.exists(p):
            continue
        df_perm = load_csv(p)

        if "feature" not in df_perm.columns and "Feature" in df_perm.columns:
            df_perm = df_perm.rename(columns={"Feature": "feature"})
        if "importance" not in df_perm.columns and "Importance" in df_perm.columns:
            df_perm = df_perm.rename(columns={"Importance": "importance"})
        if "split" not in df_perm.columns:
            df_perm["split"] = k

        perm_all_rows.append(df_perm)

    if perm_all_rows:
        perm_all = pd.concat(perm_all_rows, ignore_index=True)
        perm_all.to_csv(os.path.join(out_tables, "permutation_importance_all_splits.csv"), index=False)

        perm_stats = (
            perm_all.groupby("feature")["importance"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )

        perm_stats.to_csv(os.path.join(out_agg, "permutation_importance_mean_std.csv"), index=False)
        perm_stats.head(TOPK_FEATURES).to_csv(
            os.path.join(out_agg, f"permutation_importance_top{TOPK_FEATURES}.csv"),
            index=False,
        )

        fig = plot_importance_mean_std(
            perm_stats,
            task_name=task.task_key,
            x_label="Permutation importance (mean ± std)",
            topk=TOPK_FEATURES,
        )
        out_base = os.path.join(out_plots, f"{plot_prefix}permutation_importance_top{TOPK_FEATURES}_mean_std")
        save_png_eps(fig, out_base, dpi=DPI)
        plt.close(fig)

    # -------------------------
    # Metrics summary mean std
    # -------------------------
    summary = {
        "task_key": task.task_key,
        "n_splits": int(len(split_dirs)),
        "labels": ", ".join([str(x) for x in labels]),
    }

    if metrics_df is not None:
        metric_cols = [c for c in ["accuracy", "precision", "recall", "f1"] if c in metrics_df.columns]
        for m in metric_cols:
            vals = metrics_df[m].astype(float).to_numpy()
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            summary[f"{m}_mean"] = mean
            summary[f"{m}_std"] = std

        metrics_summary_df = pd.DataFrame([summary])
        metrics_summary_df.to_csv(os.path.join(out_agg, "metrics_mean_std.csv"), index=False)

    return summary


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    exp_root = os.path.join(ML_EXPERIMENTS_ROOT, DATASET_NAME, f"exp_{add}")
    tasks = discover_tasks(exp_root)

    out_root = os.path.join(RESULTS_ROOT, DATASET_NAME, f"exp_{add}")
    ensure_dir(out_root)

    ensure_dir(ALL_RESULTS_ROOT)
    ensure_dir(ALL_CM_NORM_AVG_ROOT)

    out_global_agg = os.path.join(out_root, "aggregated_results")
    ensure_dir(out_global_agg)

    print("============================================================")
    print("ML experiments analysis post hoc")
    print("------------------------------------------------------------")
    print(f"Input experiments: {exp_root}")
    print(f"Output results:    {out_root}")
    print(f"Global CM folder:  {ALL_CM_NORM_AVG_ROOT}")
    print(f"Tasks found:       {len(tasks)}")
    for t in tasks:
        print(f"  - {t.task_key}")
    print("============================================================")

    all_task_summaries: List[Dict[str, Any]] = []

    for t in tasks:
        print("------------------------------------------------------------")
        print(f"Analyzing task: {t.task_key}")
        out_task_dir = os.path.join(out_root, t.task_key)

        s = analyze_one_task(
            t,
            out_task_dir=out_task_dir,
        )
        all_task_summaries.append(s)

    # Global summary table and Excel only in the regular output
    summary_df = pd.DataFrame(all_task_summaries)
    summary_csv = os.path.join(out_global_agg, "summary_all_tasks.csv")
    summary_df.to_csv(summary_csv, index=False)

    summary_xlsx = os.path.join(out_global_agg, "summary_all_tasks.xlsx")
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="summary", index=False)

    print("============================================================")
    print("Done")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {summary_xlsx}")
    print(f"Collected normalized averages in: {ALL_CM_NORM_AVG_ROOT}")
    print("============================================================")


if __name__ == "__main__":
    main()