# ==============================================================
# AR(1) Noise Classification Experiment (Grouped CV)
# ==============================================================

import pandas as pd
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from metrics import *  # custom metric functions (delay_embed, Fisher, PE, SampEn, LZ, etc.)
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np

import neurokit2 as nk  # still used for entropy + fisher_information_nk

import sklearn
from packaging import version

# --- sklearn OneHotEncoder patch (kept identical) ---
if version.parse(sklearn.__version__) >= version.parse("1.2"):
    from sklearn.preprocessing import OneHotEncoder

    _original_init = OneHotEncoder.__init__

    def _patched_init(self, *args, **kwargs):
        if "sparse" in kwargs:
            kwargs["sparse_output"] = kwargs.pop("sparse")
        return _original_init(self, *args, **kwargs)

    OneHotEncoder.__init__ = _patched_init


# -----------------------------
# Parameters (kept consistent)
# -----------------------------
embedding_dimension = 10
time_delay = 1
memory_complexity = 300
timestep_size = 0.1  # same as Hénon script for low_freq noise time axis
np.random.seed(42)

n_runs = 5
force_regenerate = True

noise_configs = {
    "gaussian": [1, 0.3, 0.1],
    "pink": [2, 1, 0.2],
    "low_freq": [0.2, 1, 2]
}

# -----------------------------
# AR(1) Generator
# -----------------------------
def generate_ar1_series_variable(
    n_samples=2000,
    run_id=0,
    phi_base=0.85,
    phi_jitter=0.03,
    sigma=1.0,
    burn_in=1000
):
    """
    Generate a reproducible AR(1) process:
        x_t = phi * x_{t-1} + eps_t,  eps_t ~ N(0, sigma^2)

    - phi is mildly varied per run_id (deterministic).
    - burn_in samples are generated and dropped to reach stationarity.
    - returns t_eval and the stationary AR(1) series.
    """

    np.random.seed(42 * run_id)

    # keep AR(1) stationary: |phi| < 1
    phi = phi_base + np.random.uniform(-phi_jitter, phi_jitter)
    phi = float(np.clip(phi, -0.95, 0.95))

    total_len = burn_in + n_samples

    x = 0.0
    xs = np.empty(total_len, dtype=float)
    eps = np.random.normal(0.0, sigma, size=total_len)

    for t in range(total_len):
        x = phi * x + eps[t]
        xs[t] = x

    series = xs[burn_in:]  # stationary segment
    t_eval = np.arange(len(series)) * timestep_size
    return t_eval, series


# -----------------------------
# Helpers (kept identical)
# -----------------------------
def zscore_scale(x):
    mu = np.mean(x)
    sd = np.std(x)
    if sd == 0:
        return x - mu, (mu, 1.0)
    return (x - mu) / sd, (mu, sd)

def add_gaussian_noise_relative(x_norm, intensity=0.1):
    return x_norm + np.random.normal(0.0, intensity, size=len(x_norm))

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


# -----------------------------
# Noise application (kept identical)
# -----------------------------
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
# Complexity metrics (kept identical)
# -----------------------------
def calculate_fisher_information_nk(time_series, delay=1, dimension=3):
    fi, info = nk.fisher_information(time_series, delay=delay, dimension=dimension)
    return fi

def extract_complexity_metrics(signal, t_eval, emb_dim=embedding_dimension, delay=time_delay):
    rows = []
    fail_counts = {}

    print(f"Starting metric extraction: signal length={len(signal)}, memory_complexity={memory_complexity}")

    for i in range(memory_complexity, len(signal)):
        segment = np.asarray(signal[i - memory_complexity:i], dtype=float)
        timestamp = t_eval[i]

        if not np.isfinite(segment).all():
            print(f" NaN/Inf detected in segment ending at {i}, skipping")
            continue

        try:
            embedded = delay_embed(segment, embedding_dimension=emb_dim, time_delay=delay)
        except Exception as e:
            print(f" Embedding failed at i={i}: {type(e).__name__} – {e}")
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
    print(f"\n Extraction finished: {len(df)} rows, {len(df.columns)} columns")
    if fail_counts:
        print(" Failure summary by exception type:")
        for k, v in fail_counts.items():
            print(f"   {k}: {v} times")
    return df


# -----------------------------
# Output folders (renamed)
# -----------------------------
def create_results_folder(base_folder="AR1_Noise_Exp"):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{current_time}_noise_classification"
    folder_path = os.path.join(base_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

results_folder = create_results_folder()
base_folder = "AR1_Noise_Exp"
os.makedirs(base_folder, exist_ok=True)

# -----------------------------
# Metadata (kept identical)
# -----------------------------
metrics_used = [
    "var1der", "hurst", "std_dev", "mad", "var2der",
    "fisher_info", "approximate_entropy", "svd_entropy", "dfa",
    "fisher_info_nk", "rel_decay", "svd_energy", "condition_number",
    "cv", "spectral_skewness", "permutation_entropy",
    "sample_entropy", "lempel_ziv_complexity",
]

params = {
    "embedding_dimension": embedding_dimension,
    "time_delay": time_delay,
    "memory_complexity": memory_complexity,
    "timestep_size": timestep_size,
    "noise_configs": str(noise_configs),
    "n_runs": n_runs,
    "metrics_used": ",".join(metrics_used),
    "ar1_phi_base": 0.7,
    "ar1_phi_jitter": 0.1,
    "ar1_sigma": 1.0,
    "ar1_burn_in": 500
}

param_suffix = f"AR1_ed{embedding_dimension}_td{time_delay}_mc{memory_complexity}_dt{timestep_size}"
dataset_path = os.path.join(base_folder, f"dataset_{param_suffix}.csv")
param_file = os.path.join(base_folder, f"params_{param_suffix}.csv")


# -----------------------------
# Dataset creation (generator swapped)
# -----------------------------
def create_dataset():
    all_dfs = []
    run_counter = 0

    vis_folder = os.path.join(results_folder, "signal_examples")
    os.makedirs(vis_folder, exist_ok=True)

    # --- noisy runs ---
    for noise_type, levels in noise_configs.items():
        for intensity in levels:
            for run in range(n_runs):
                run_counter += 1
                print(f" Run {run_counter}: noise={noise_type}, intensity={intensity}, repeat={run + 1}/{n_runs}")

                # Generate AR(1)
                t_eval, clean = generate_ar1_series_variable(run_id=run_counter)
                clean_norm, _ = zscore_scale(clean)
                noisy = apply_noise(clean, t_eval, noise_type, intensity)

                if run == 0:  # visualize only once per intensity
                    fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
                    ax[0].plot(t_eval, clean_norm, color="black", lw=1)
                    ax[0].set_title("Clean AR(1)")
                    ax[0].set_ylabel("Amplitude")
                    ax[0].grid(alpha=0.3)

                    ax[1].plot(t_eval, noisy, color="tab:red", lw=1)
                    ax[1].set_title(f"{noise_type} noise (intensity={intensity})")
                    ax[1].set_xlabel("Time [a.u.]")
                    ax[1].set_ylabel("Amplitude")
                    ax[1].grid(alpha=0.3)

                    plt.tight_layout()
                    out_path = os.path.join(vis_folder, f"AR1_{noise_type}_intensity{intensity}_run{run_counter}.png")
                    plt.savefig(out_path, dpi=200, bbox_inches="tight")
                    plt.close()
                    print(f"📊 Saved AR(1) visualization: {out_path}")

                # Extract metrics
                comp_df = extract_complexity_metrics(noisy, t_eval)
                comp_df["noise_type"] = noise_type
                comp_df["noise_intensity"] = intensity
                comp_df["noise_label_task1"] = noise_type
                comp_df["noise_label_task2"] = f"{noise_type}_{intensity}"
                comp_df["signal_id"] = f"{noise_type}_{intensity}_run{run_counter}"
                all_dfs.append(comp_df)

    # --- clean runs for balancing (kept identical logic) ---
    target_clean_runs = len(noise_configs) * n_runs
    extra_needed = target_clean_runs - n_runs if target_clean_runs > n_runs else 0
    total_clean_runs = n_runs + extra_needed

    print(f"Generating {total_clean_runs} clean AR(1) runs to balance Tasks 1 & 3.")

    for run in range(total_clean_runs):
        run_counter += 1
        print(f" Clean run {run_counter}/{total_clean_runs}")
        t_eval, clean = generate_ar1_series_variable(run_id=run_counter + 1000)
        clean_norm, _ = zscore_scale(clean)

        if run == 0:
            plt.figure(figsize=(10, 2.5))
            plt.plot(t_eval, clean_norm, color="black", lw=1)
            plt.title("Clean AR(1) (no noise)")
            plt.xlabel("Time [a.u.]")
            plt.ylabel("Amplitude")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            out_path = os.path.join(vis_folder, f"AR1_clean_example_run{run_counter}.png")
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"📊 Saved clean AR(1) visualization: {out_path}")

        comp_df = extract_complexity_metrics(clean_norm, t_eval)
        comp_df["noise_type"] = "none"
        comp_df["noise_intensity"] = 0
        comp_df["noise_label_task1"] = "none"
        comp_df["noise_label_task2"] = "none_0"
        comp_df["signal_id"] = f"none_0_run{run_counter}"
        all_dfs.append(comp_df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(dataset_path, index=False)
    pd.DataFrame([params]).to_csv(param_file, index=False)

    print(f"AR(1) dataset created and saved: {dataset_path} ({full_df.shape})")
    return full_df


################################
# Load or regenerate dataset
################################
dataset_candidates = [
    f for f in os.listdir(base_folder)
    if f.startswith(f"dataset_AR1_ed{embedding_dimension}_td{time_delay}_mc{memory_complexity}_dt")
       and f.endswith(".csv")
]

if dataset_candidates and not force_regenerate:
    dataset_path = os.path.join(base_folder, dataset_candidates[0])
    print(f"Found existing dataset: {dataset_path}")
    full_df = pd.read_csv(dataset_path)
else:
    print(" No matching dataset found, creating from scratch...")
    full_df = create_dataset()


# -----------------------------
# Plotting helpers (kept identical)
# -----------------------------
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
    fi_stats = fi_df.groupby("Feature")["Importance"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    fi_top = fi_stats.head(20)
    plt.figure(figsize=(8, 5))
    plt.barh(fi_top.index[::-1], fi_top["mean"][::-1], xerr=fi_top["std"][::-1], capsize=4)
    plt.xlabel("Feature Importance (mean ± std)")
    plt.title(f"Average Feature Importance – {task_name}")
    plt.tight_layout()
    out_path = os.path.join(results_folder, f"avg_feature_importance_{task_name.replace(' ', '_').lower()}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f" Saved importance plot: {out_path}")

def plot_average_confusion_matrix(cm_list, labels, task_name):
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
    out_path = os.path.join(results_folder, f"avg_confmat_{task_name.replace(' ', '_').lower()}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f" Saved average confusion matrix: {out_path}")


# -----------------------------
# Grouped Cross-Validation (kept identical)
# -----------------------------
exclude_cols = [
    "timestep", "noise_type", "noise_intensity",
    "noise_label_task1", "noise_label_task2", "signal_id"
]
for col in ["run_type", "dataset_split", "train_test_split"]:
    if col in full_df.columns:
        exclude_cols.append(col)

feature_cols = [
    c for c in full_df.columns
    if c not in exclude_cols and pd.api.types.is_numeric_dtype(full_df[c])
]
print(f"Using {len(feature_cols)} numeric feature columns.")

def run_grouped_cross_validation(task_label_col, task_name, loss_function="MultiClass", is_binary=False):
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
        missing = label_counts[label_counts < n_splits]
        raise ValueError(f"Not enough runs per label for {n_splits}-fold CV: {missing.to_dict()}")

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

        fold_accuracy = accuracy_score(y_test, y_pred)
        print(f"Fold {fold} accuracy: {fold_accuracy:.3f}")
        accuracies.append(fold_accuracy)

        fi = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": clf.get_feature_importance(),
            "fold": fold
        })
        all_importances.append(fi)

        plot_and_save_confusion_matrix(
            y_test, y_pred, labels_all,
            f"Confusion Matrix – {task_name} (Fold {fold})",
            os.path.join(results_folder, f"confmat_{task_name.replace(' ', '_').lower()}_fold{fold}.png")
        )

    acc_df = pd.DataFrame({
        "Fold": np.arange(1, len(accuracies) + 1),
        "Accuracy": accuracies,
    })
    acc_df["MeanAccuracy"] = acc_df["Accuracy"].mean()
    acc_df["StdAccuracy"] = acc_df["Accuracy"].std()
    acc_csv_path = os.path.join(results_folder, f"AR1_cv_accuracies_{task_name.replace(' ', '_').lower()}.csv")
    acc_df.to_csv(acc_csv_path, index=False)
    print(f"Saved accuracies: {acc_csv_path}")

    fi_df = pd.concat(all_importances, ignore_index=True)
    fi_mean = fi_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False).reset_index()

    fi_mean.to_csv(os.path.join(results_folder, f"AR1_cv_feature_importance_{task_name.replace(' ', '_').lower()}.csv"),
                   index=False)

    plot_average_confusion_matrix(cm_all, labels_all, f"AR(1) – {task_name}")
    plot_feature_importance_with_errorbars(fi_df, f"AR(1) – {task_name}")

    print(f"\n Stratified-by-group 5-fold CV complete for {task_name}.")
    return fi_mean


# ==============================================================
# Run grouped CV tasks
# ==============================================================

full_df["noise_present"] = (full_df["noise_type"] != "none").astype(int)

print("\n Running Task 1: Noise Category Classification")
run_grouped_cross_validation("noise_label_task1", "Noise Category Classification")

print("\n Running Task 2: Noise Category + Intensity Classification (Lorenz-identical balancing)")

group_tbl = (
    full_df.groupby("signal_id")[["noise_label_task2"]]
    .first()
    .reset_index()
)

label_counts = group_tbl["noise_label_task2"].value_counts()
min_runs = label_counts.min()
print(f"Each label currently has between {label_counts.min()} and {label_counts.max()} runs.")
print(f"   → Using {min_runs} runs per label for balancing.")

keep_ids = (
    group_tbl.groupby("noise_label_task2")["signal_id"]
    .apply(lambda x: x.sample(min_runs, random_state=42))
    .explode()
    .tolist()
)

df_task2_balanced = full_df[full_df["signal_id"].isin(keep_ids)].copy()

backup_df = full_df
full_df = df_task2_balanced

run_grouped_cross_validation("noise_label_task2", "Noise Category + Intensity Classification")

full_df = backup_df

print("\n Running Task 3: Noise Present vs Not Classification")
run_grouped_cross_validation("noise_present", "Noise Present vs Not Classification", loss_function="Logloss", is_binary=True)


# ==============================================================
# Create Summary Excel
# ==============================================================
def create_cv_summary_excel(results_folder):
    summary_rows = []

    report_files = [f for f in os.listdir(results_folder) if f.startswith("AR1_cv_feature_importance_") and f.endswith(".csv")]
    if not report_files:
        print("No AR1 feature importance files found to summarize!")
        return

    # Minimal summary: list tasks detected
    summary_df = pd.DataFrame({"DetectedFiles": report_files})
    summary_path = os.path.join(results_folder, "cv_summary_all_tasks.xlsx")
    summary_df.to_excel(summary_path, index=False)

    print(f"\n Summary Excel created: {summary_path}")
    print(summary_df)

create_cv_summary_excel(results_folder)
