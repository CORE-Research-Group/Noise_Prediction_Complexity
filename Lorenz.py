# ==============================================================
# Lorenz Noise Classification Experiment (Leak-Free, Grouped CV)
# ==============================================================


import numpy as np
import pandas as pd
import os
import matplotlib
import re

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from datetime import datetime
import neurokit2 as nk
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from catboost import CatBoostClassifier
from metrics import *  # custom metric functions
from sklearn.model_selection import StratifiedKFold

# -----------------------------
# Parameters
# -----------------------------
embedding_dimension = 10
time_delay = 1
memory_complexity = 300  # 100
timestep_size = 0.1
np.random.seed(42)

n_runs = 5  # total runs per noise configuration


# Will generate new data without checking if prior data exists
force_regenerate = True

noise_configs = {
    "gaussian": [1, 0.3, 0.1],
    "pink": [2, 1, 0.2],
    "low_freq": [0.2, 1, 2]
}


# -----------------------------
# Output folders & paths
# -----------------------------
def create_results_folder(base_folder="Lorenz_Noise_Exp"):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{current_time}_noise_classification"
    folder_path = os.path.join(base_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


base_folder = "Lorenz_Noise_Exp"
os.makedirs(base_folder, exist_ok=True)
results_folder = create_results_folder(base_folder)


# -----------------------------
# Helpers
# -----------------------------
def save_classification_report(y_true, y_pred, labels, out_txt_path):
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=[str(l) for l in labels], digits=4)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


def plot_and_save_confusion_matrix(y_true, y_pred, labels, title, out_png_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label', xlabel='Predicted label', title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(out_png_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_feature_importance_with_errorbars(fi_df, task_name):
    """Plot mean and std of feature importances across folds."""
    fi_stats = fi_df.groupby("Feature")["Importance"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    fi_top = fi_stats.head(20)
    plt.figure(figsize=(8, 5))
    plt.barh(fi_top.index[::-1], fi_top["mean"][::-1], xerr=fi_top["std"][::-1], capsize=4)
    plt.xlabel("Feature Importance (mean ± std)")
    plt.title(f"Average Feature Importance – {task_name}")
    plt.tight_layout()
    out_path = os.path.join(results_folder, f"Lorenz_avg_feature_importance_{task_name.replace(' ', '_').lower()}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f" Saved importance plot: {out_path}")


def plot_average_confusion_matrix(cm_list, labels, task_name):
    """Compute and plot normalized average confusion matrix from all folds."""
    cm_mean = np.mean(np.stack(cm_list), axis=0)
    cm_norm = cm_mean / (cm_mean.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label', xlabel='Predicted label',
           title=f"Average Confusion Matrix – {task_name}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    plt.tight_layout()
    out_path = os.path.join(results_folder, f"Lorenz_avg_confmat_{task_name.replace(' ', '_').lower()}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f" Saved average confusion matrix: {out_path}")


# -----------------------------
# Lorenz generator
# -----------------------------
def lorenz(t, state, sigma=10, beta=8 / 3, rho=28):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]



def generate_lorenz_series(duration=200, dt=0.1, run_id=0):
    """
    Generate a Lorenz time series with a 50-second transient period removed.
    The system is first integrated for (duration + 50 s), then the first 50 s are discarded.
    """
    np.random.seed(42 * run_id)

    warmup_time = 50.0  # seconds to discard
    total_duration = duration + warmup_time
    t_eval_full = np.arange(0, total_duration, dt)

    # Slight parameter drift per run (±5%)
    sigma = 10 * np.random.uniform(0.95, 1.05)
    beta = (8 / 3) * np.random.uniform(0.95, 1.05)
    rho = 28 * np.random.uniform(0.95, 1.05)

    # Random initial condition near (1, 1, 1)
    init_state = np.array([1, 1, 1]) + np.random.normal(0, 0.05, 3)

    # Integrate the Lorenz system
    sol = solve_ivp(lorenz, (0, total_duration), init_state, t_eval=t_eval_full, args=(sigma, beta, rho))

    # Discard first 50 seconds (transient)
    mask = sol.t >= warmup_time
    t_eval = sol.t[mask] - warmup_time
    x_series = sol.y[0][mask]

    # Ensure same output length as before
    expected_len = int(duration / dt)
    if len(x_series) > expected_len:
        x_series = x_series[:expected_len]
        t_eval = t_eval[:expected_len]
    elif len(x_series) < expected_len:
        pad_len = expected_len - len(x_series)
        x_series = np.pad(x_series, (0, pad_len), mode='edge')
        t_eval = np.pad(t_eval, (0, pad_len), mode='edge')

    return t_eval, x_series


# -----------------------------
# Scaling + Noise Helpers
# -----------------------------
def zscore_scale(x):
    mu = np.mean(x)
    sd = np.std(x)
    if sd == 0:
        return x - mu, (mu, 1.0)
    return (x - mu) / sd, (mu, sd)


def add_gaussian_noise_relative(x_norm, intensity=0.1):
    return x_norm + np.random.normal(0.0, intensity, size=len(x_norm))


def apply_noise(signal, t_eval, noise_type, intensity):
    signal_norm, _ = zscore_scale(signal)
    if noise_type == "gaussian":
        noisy = add_gaussian_noise_relative(signal_norm, intensity)
    elif noise_type == "pink":
        noisy = signal_norm + generate_pink_noise(len(signal_norm), intensity)
    elif noise_type == "low_freq":
        freq = np.random.uniform(0.45, 0.55)
        phi = np.random.uniform(-0.4 * np.pi, 0.4 * np.pi)
        noisy = signal_norm + intensity * np.sin(2 * np.pi * freq * t_eval + phi)
    else:
        noisy = signal_norm.copy()
    return noisy


# -----------------------------
# Complexity metrics
# -----------------------------
def calculate_fisher_information_nk(time_series, delay=1, dimension=3):
    fi, _ = nk.fisher_information(time_series, delay=delay, dimension=dimension)
    return fi


def safe_scalar(val):
    if isinstance(val, (list, tuple, np.ndarray)):
        try:
            return float(np.ravel(val)[0])
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan


def extract_complexity_metrics(signal, t_eval, emb_dim=embedding_dimension, delay=time_delay):

    rows = []
    fail_counts = {}

    print(f"Starting metric extraction: signal length={len(signal)}, memory_complexity={memory_complexity}")

    for i in range(memory_complexity, len(signal)):
        segment = np.asarray(signal[i - memory_complexity:i], dtype=float)
        timestamp = t_eval[i]

        if not np.isfinite(segment).all():
            print(f"NaN/Inf detected in segment ending at {i}, skipping")
            continue

        try:
            # Phase-space embedding
            embedded = delay_embed(segment, embedding_dimension=emb_dim, time_delay=delay)
            embedded_flat = embedded.flatten()
        except Exception as e:
            print(f"Embedding failed at i={i}: {type(e).__name__} – {e}")
            continue

        try:
            vals = {
                "timestep": timestamp,

                "var1der": safe_scalar(calculate_variance_1st_derivative(embedded)),
                "var2der": safe_scalar(calculate_variance_2nd_derivative(embedded)),

                "std_dev": safe_scalar(np.std(segment)),
                "mad": safe_scalar(np.mean(np.abs(segment - np.mean(segment)))),
                "cv": safe_scalar(np.std(segment) / (np.mean(np.abs(segment)) + 1e-12)),

                "approximate_entropy": safe_scalar(nk.entropy_approximate(segment)),
                "permutation_entropy": safe_scalar(permutation_entropy_metric(segment, order=3, delay=1)),
                "dfa": safe_scalar(nk.fractal_dfa(segment)),
                "hurst": safe_scalar(calculate_hurst(segment)),
                "fisher_info": safe_scalar(calculate_fisher_information(segment)),
                "sample_entropy": safe_scalar(sample_entropy_metric(segment)),
                "lempel_ziv_complexity": safe_scalar(lempel_ziv_complexity_metric(segment)),
                "fisher_info_nk": safe_scalar(
                    calculate_fisher_information_nk(segment, delay=delay, dimension=emb_dim)
                ),
                "svd_entropy": safe_scalar(nk.entropy_svd(segment)),
                "rel_decay": safe_scalar(relative_decay_singular_values(embedded)),
                "svd_energy": safe_scalar(svd_energy(embedded, k=3)),
                "condition_number": safe_scalar(condition_number(embedded)),
                "spectral_skewness": safe_scalar(spectral_skewness(embedded)),
            }

            rows.append(vals)

        except Exception as e:
            err_name = type(e).__name__
            fail_counts[err_name] = fail_counts.get(err_name, 0) + 1
            print(f"Metric extraction failed at i={i}: {err_name} – {e}")
            continue

    df = pd.DataFrame(rows)
    print(f"\n✅ Extraction finished: {len(df)} rows, {len(df.columns)} columns")
    if fail_counts:
        print("Failure summary by exception type:")
        for k, v in fail_counts.items():
            print(f"   {k}: {v} times")
    return df


###########################
# Dataset creation
###########################
def create_dataset():
    all_dfs = []
    run_id = 0

    # Folder for visualizations
    vis_folder = os.path.join(results_folder, "signal_examples")
    os.makedirs(vis_folder, exist_ok=True)

    for noise_type, levels in noise_configs.items():
        for intensity in levels:
            for run in range(n_runs):
                run_id += 1
                print(f"Run {run_id}: noise={noise_type}, intensity={intensity}, repeat={run + 1}/{n_runs}")

                # Generate Lorenz signal and noisy version
                t_eval, clean = generate_lorenz_series()
                noisy = apply_noise(clean, t_eval, noise_type, intensity)

                clean_norm, _ = zscore_scale(clean)

                # Save visualization
                if run == 0:
                    fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
                    ax[0].plot(t_eval, clean_norm, color="black", lw=1)
                    ax[0].set_title("Clean Lorenz")
                    ax[0].set_ylabel("x(t)")
                    ax[0].grid(alpha=0.3)

                    ax[1].plot(t_eval, noisy, color="tab:red", lw=1)
                    ax[1].set_title(f"{noise_type} noise (intensity={intensity})")
                    ax[1].set_xlabel("Time [s]")
                    ax[1].set_ylabel("x(t)")
                    ax[1].grid(alpha=0.3)

                    plt.tight_layout()
                    out_path = os.path.join(vis_folder, f"Lorenz_{noise_type}_intensity{intensity}_run{run_id}.png")
                    plt.savefig(out_path, dpi=200, bbox_inches="tight")
                    plt.close()
                    print(f"Saved Lorenz visualization: {out_path}")

                # Extract metrics
                comp_df = extract_complexity_metrics(noisy, t_eval)
                comp_df["noise_type"] = noise_type
                comp_df["noise_intensity"] = intensity
                comp_df["noise_label_task1"] = noise_type
                comp_df["noise_label_task2"] = f"{noise_type}_{intensity}"
                comp_df["signal_id"] = f"{noise_type}_{intensity}_run{run_id}"
                all_dfs.append(comp_df)

    # Clean no noise runs
    target_clean_runs = len(noise_configs) * n_runs  # e.g. 3×5 = 15
    extra_needed = target_clean_runs - n_runs if target_clean_runs > n_runs else 0
    total_clean_runs = n_runs + extra_needed

    print(f"Generating {total_clean_runs} clean Lorenz runs to balance Tasks 1 & 3.")

    for run in range(total_clean_runs):
        run_id += 1
        print(f"Clean run {run_id}/{total_clean_runs}")
        t_eval, clean = generate_lorenz_series(run_id=run_id + 1000)

        #Visualize the first clean run only
        if run == 0:
            plt.figure(figsize=(10, 2.5))
            plt.plot(t_eval, clean, color="black", lw=1)
            plt.title("Clean Lorenz System (no noise)")
            plt.xlabel("Time [s]")
            plt.ylabel("x(t)")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            out_path = os.path.join(vis_folder, f"Lorenz_clean_example_run{run_id}.png")
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved clean Lorenz visualization: {out_path}")

        clean_norm, _ = zscore_scale(clean)
        comp_df = extract_complexity_metrics(clean_norm, t_eval)
        comp_df["noise_type"] = "none"
        comp_df["noise_intensity"] = 0
        comp_df["noise_label_task1"] = "none"
        comp_df["noise_label_task2"] = "none_0"
        comp_df["signal_id"] = f"none_0_run{run_id}"
        all_dfs.append(comp_df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    dataset_path = os.path.join(base_folder, "lorenz_dataset.csv")
    full_df.to_csv(dataset_path, index=False)
    print(f"Dataset created: {dataset_path} ({full_df.shape})")

    return full_df


# -----------------------------
# Load or regenerate dataset
# -----------------------------
dataset_path = os.path.join(base_folder, "lorenz_dataset.csv")
if os.path.exists(dataset_path) and not force_regenerate:
    print(f"Loading cached dataset: {dataset_path}")
    full_df = pd.read_csv(dataset_path)
else:
    print(" Creating new dataset...")
    full_df = create_dataset()

# -----------------------------
# Cross-validation (GroupKFold)
# -----------------------------
feature_cols = [c for c in full_df.columns if c not in
                ["timestep", "noise_type", "noise_intensity",
                 "noise_label_task1", "noise_label_task2", "signal_id"]]


def run_grouped_cross_validation(task_label_col, task_name,
                                 loss_function="MultiClass",
                                 is_binary=False,
                                 balance_none_label_for_task2=False):
    """
    Stratified 5-fold CV at the group (signal_id) level.
    """
    labels_all = sorted(full_df[task_label_col].unique().tolist()) if not is_binary else [0, 1]
    X_all = full_df.copy()

    group_tbl = (
        X_all.groupby("signal_id")[task_label_col]
        .first()
        .reset_index()
        .rename(columns={task_label_col: "group_label"})
    )

    n_splits = 5
    label_counts = group_tbl["group_label"].value_counts()
    if (label_counts < n_splits).any():
        raise ValueError(f"Not enough runs per label: {label_counts.to_dict()}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_reports, all_importances, cm_all = [], [], []
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(group_tbl["signal_id"], group_tbl["group_label"]), start=1):
        train_groups = set(group_tbl["signal_id"].iloc[train_idx])
        test_groups = set(group_tbl["signal_id"].iloc[test_idx])

        X_train = X_all[X_all["signal_id"].isin(train_groups)][feature_cols]
        y_train = X_all[X_all["signal_id"].isin(train_groups)][task_label_col]
        X_test = X_all[X_all["signal_id"].isin(test_groups)][feature_cols]
        y_test = X_all[X_all["signal_id"].isin(test_groups)][task_label_col]

        print(f"\n Fold {fold}/5 – {task_name} (stratified by group)")

        clf = CatBoostClassifier(
            iterations=300,
            depth=10,
            learning_rate=0.1,
            loss_function=loss_function,
            random_seed=42,
            verbose=False
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        cm_all.append(confusion_matrix(y_test, y_pred, labels=labels_all))
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Fold {fold} accuracy: {acc:.3f}")

        rep = classification_report(y_test, y_pred, labels=labels_all, output_dict=True, digits=4)
        df_rep = pd.DataFrame(rep).transpose()
        df_rep["fold"] = fold
        df_rep["fold_accuracy"] = acc
        all_reports.append(df_rep)

        fi = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": clf.get_feature_importance(),
            "fold": fold
        })
        all_importances.append(fi)

        # per-fold CM plot
        plot_and_save_confusion_matrix(
            y_test, y_pred, labels_all,
            f"Lorenz – Confusion Matrix – {task_name} (Fold {fold})",
            os.path.join(results_folder, f"Lorenz_confmat_{task_name.replace(' ', '_').lower()}_fold{fold}.png")
        )

    acc_df = pd.DataFrame({"Fold": range(1, 6), "Accuracy": accuracies})
    acc_df["MeanAccuracy"] = np.mean(accuracies)
    acc_df["StdAccuracy"] = np.std(accuracies)
    acc_path = os.path.join(results_folder, f"Lorenz_cv_accuracies_{task_name.replace(' ', '_').lower()}.csv")
    acc_df.to_csv(acc_path, index=False)
    print(f"Saved accuracies: {acc_path}")

    reports_df = pd.concat(all_reports, ignore_index=True)
    fi_df = pd.concat(all_importances, ignore_index=True)
    fi_mean = fi_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False).reset_index()

    reports_df.to_csv(os.path.join(results_folder, f"Lorenz_cv_reports_{task_name.replace(' ', '_').lower()}.csv"),
                      index=False)
    fi_mean.to_csv(
        os.path.join(results_folder, f"Lorenz_cv_feature_importance_{task_name.replace(' ', '_').lower()}.csv"),
        index=False)

    plot_average_confusion_matrix(cm_all, labels_all, f"Lorenz – {task_name}")
    plot_feature_importance_with_errorbars(fi_df, f"Lorenz – {task_name}")

    print(f"\n Stratified-by-group 5-fold CV complete for {task_name}.")
    return reports_df, fi_mean


def balance_groups_by_runs(full_df, task_label_col):
    """
    #Balances the dataset by selecting the same number of complete signal_id groups per label.
    #Keeps all windows from each selected run.
    """
    # Count how many signal_id groups exist for each label
    group_tbl = full_df.groupby("signal_id")[task_label_col].first().reset_index()
    label_counts = group_tbl[task_label_col].value_counts()
    min_groups = label_counts.min()
    print(f"Balancing by number of runs → {min_groups} runs per label.")

    selected_ids = []
    for label, ids in group_tbl.groupby(task_label_col)["signal_id"]:
        chosen = ids.sample(min_groups, random_state=42)
        selected_ids.extend(chosen)

    balanced_df = full_df[full_df["signal_id"].isin(selected_ids)].copy()
    print("Balanced label distribution (rows per label):")
    print(balanced_df[task_label_col].value_counts())
    return balanced_df


# ==============================================================
# Run grouped CV tasks
# ==============================================================

# Binary flag for Task 3
full_df["noise_present"] = (full_df["noise_type"] != "none").astype(int)

# Task 1: Noise Type
print("\n Running Task 1: Noise Category Classification")
run_grouped_cross_validation("noise_label_task1", "Noise Category Classification")

# Task 2: Noise Type + Intensity
print("\n Running Task 2: Noise Category + Intensity Classification (with balanced none_0)")

# Temporarily swap datasets for Task 2
print("\n Running Task 2: Noise Category + Intensity Classification (Balanced by Samples)")

#Ensure deterministic order before sampling
full_df = full_df.sort_values(by=["noise_label_task2", "signal_id", "timestep"], ignore_index=True)

balanced_df = balance_groups_by_runs(full_df, "noise_label_task2")

backup_df = full_df
full_df = balanced_df

# Run Task 2
reports_df_task2, fi_mean_task2 = run_grouped_cross_validation(
    task_label_col="noise_label_task2",
    task_name="Noise Category + Intensity Classification"
)

"""
labels_task2 = sorted(full_df["noise_label_task2"].unique())
# Read per-fold predictions (if not kept, re-run them)
cm_files_exist = any("confmat_noise_category_+_intensity_classification" in f for f in os.listdir(results_folder))
if not cm_files_exist:
    print("Re-generating Task 2 confusion matrices...")
    # (the run_grouped_cross_validation call above already did this)
else:
    print("Confusion matrices already generated for Task 2")

# Average CM
try:
    # Collect all individual per-fold matrices if present
    cm_list = []
    for fold in range(1, 6):
        path = os.path.join(results_folder,
                            f"Lorenz_confmat_noise_category_intensity_classification_(balanced_samples)_fold{fold}.png")
        if os.path.exists(path):
            continue  # image files exist, skip
    # Recompute mean CM from reports_df if needed (only numeric guard)
    print("Re-plotting averaged confusion matrix for Task 2 (Balanced)")
    # Dummy empty list placeholder just to ensure function call works safely
    plot_average_confusion_matrix([], labels_task2, "Noise Category + Intensity Classification")
except Exception as e:
    print(f"Could not re-plot average CM for Task 2: {e}")
"""
# Restore dataset
full_df = backup_df

# Task 3: Noise Present vs Not
print("\n Running Task 3: Noise Present vs Not Classification")
run_grouped_cross_validation(
    "noise_present",
    "Noise Present vs Not Classification",
    loss_function="Logloss",
    is_binary=True
)


# ==============================================================
# Create Summary Excel (auto-detects generated report files)
# ==============================================================
def create_cv_summary_excel(results_folder):

    summary_rows = []

    # Detect all cv_reports_*.csv files
    report_files = [
        f for f in os.listdir(results_folder)
        if f.endswith(".csv") and ("cv_reports_" in f)
    ]
    if not report_files:
        print(" No CV report files found in results folder!")
        return

    for file in report_files:
        path = os.path.join(results_folder, file)
        df = pd.read_csv(path)

        task_name = (
            file.replace("Lorenz_", "")
            .replace("cv_reports_", "")
            .replace(".csv", "")
            .replace("_", " ")
            .title()
        )

        # Compute average metrics
        acc = None
        if "accuracy" in df.columns:
            acc = df["accuracy"].mean()
        elif "accuracy" in df.index:
            acc = df.loc["accuracy", "f1-score"] if "f1-score" in df.columns else None

        precision = df["precision"].mean() if "precision" in df.columns else None
        recall = df["recall"].mean() if "recall" in df.columns else None
        f1 = df["f1-score"].mean() if "f1-score" in df.columns else None

        summary_rows.append({
            "Task": task_name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

    # Combine all summaries and save
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(results_folder, "Lorenz_cv_summary_all_tasks.xlsx")
    summary_df.to_excel(summary_path, index=False)

    print(f"\n Summary Excel created: {summary_path}")
    print(summary_df)


# Run the summary creation
create_cv_summary_excel(results_folder)



def export_feature_importance_summary(results_folder, top_k=20):
    """
    Collects per-task feature importance CSVs and creates:
      - Excel summary (one sheet per task + combined)
      - combined CSV ranking (mean across tasks)
    """
    fi_files = [
        f for f in os.listdir(results_folder)
        if f.startswith("Lorenz_cv_feature_importance_") and f.endswith(".csv")
    ]
    if not fi_files:
        print("No feature-importance CSVs found.")
        return

    # ---- load all per-task FI ----
    per_task = {}
    for f in fi_files:
        path = os.path.join(results_folder, f)

        # task name from file
        task_key = (
            f.replace("Lorenz_cv_feature_importance_", "")
             .replace(".csv", "")
        )

        df = pd.read_csv(path)

        # normalize column names just in case
        # expected: Feature, Importance
        if "Feature" not in df.columns or "Importance" not in df.columns:
            print(f"Skipping {f}: missing Feature/Importance columns.")
            continue

        per_task[task_key] = df.sort_values("Importance", ascending=False).reset_index(drop=True)

    if not per_task:
        print("No usable feature-importance files found.")
        return

    # ---- combined ranking across tasks ----
    # join by Feature and average importances across tasks
    merged = None
    for task_key, df in per_task.items():
        d = df[["Feature", "Importance"]].copy()
        d = d.rename(columns={"Importance": f"Importance_{task_key}"})
        merged = d if merged is None else merged.merge(d, on="Feature", how="outer")

    imp_cols = [c for c in merged.columns if c.startswith("Importance_")]
    merged["Importance_mean_across_tasks"] = merged[imp_cols].mean(axis=1, skipna=True)
    merged["Importance_std_across_tasks"] = merged[imp_cols].std(axis=1, skipna=True)

    merged = merged.sort_values("Importance_mean_across_tasks", ascending=False).reset_index(drop=True)

    # ---- save combined CSV ----
    out_csv = os.path.join(results_folder, "Lorenz_feature_importance_combined_across_tasks.csv")
    merged.to_csv(out_csv, index=False)
    print(f"Saved combined FI CSV: {out_csv}")

    # ---- write Excel with sheets ----
    out_xlsx = os.path.join(results_folder, "Lorenz_feature_importance_summary.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        # per task
        for task_key, df in per_task.items():
            sheet = re.sub(r"[^A-Za-z0-9_]+", "_", task_key)[:31]  # Excel sheet name limit
            df.head(top_k).to_excel(writer, sheet_name=sheet, index=False)

        # combined
        merged.head(5 * top_k).to_excel(writer, sheet_name="combined", index=False)

    print(f"Saved FI Excel summary: {out_xlsx}")

    # ---- optional: global plot top-K across tasks ----
    top = merged.head(top_k).copy()
    plt.figure(figsize=(8, 5))
    plt.barh(top["Feature"][::-1], top["Importance_mean_across_tasks"][::-1],
             xerr=top["Importance_std_across_tasks"][::-1], capsize=4)
    plt.xlabel("Importance (mean ± std across tasks)")
    plt.title("Lorenz – Global Feature Importance (across tasks)")
    plt.tight_layout()
    out_png = os.path.join(results_folder, "Lorenz_feature_importance_global_across_tasks.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved global FI plot: {out_png}")


# call it at the very end
export_feature_importance_summary(results_folder, top_k=20)

