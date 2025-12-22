# ==============================================================
# Rössler Noise Classification Experiment
# ==============================================================

import numpy as np
import pandas as pd
import os
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from datetime import datetime
import neurokit2 as nk
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from metrics import *  # custom metric functions

# -----------------------------
# Parameters
# -----------------------------
embedding_dimension = 10
time_delay = 1
memory_complexity = 300
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
# Output folders
# -----------------------------
def create_results_folder(base_folder="Roessler_Noise_Exp"):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{current_time}_noise_classification"
    folder_path = os.path.join(base_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


base_folder = "Roessler_Noise_Exp"
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
    out_path = os.path.join(results_folder, f"avg_feature_importance_{task_name.replace(' ', '_').lower()}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved importance plot: {out_path}")


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
    out_path = os.path.join(results_folder, f"avg_confmat_{task_name.replace(' ', '_').lower()}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved average confusion matrix: {out_path}")


# -----------------------------
# Rössler generator
# -----------------------------
def rossler(t, state, a=0.1, b=0.3, c=14):
    x, y, z = state
    dx_dt = -y - z
    dy_dt = x + a * y
    dz_dt = b + z * (x - c)
    return [dx_dt, dy_dt, dz_dt]




def generate_roessler_series(duration=200, dt=0.1, run_id=0):
    """
    Generate a Rössler time series with a 50-second transient period removed.
    The system is first integrated for duration + 50 seconds, then the first 50 s are discarded.
    """
    np.random.seed(42 * run_id)

    warmup_time = 50.0  # seconds to discard
    total_duration = duration + warmup_time
    t_eval_full = np.arange(0, total_duration, dt)

    # Slight parameter variations
    a = 0.1 * np.random.uniform(0.95, 1.05)
    b = 0.3 * np.random.uniform(0.95, 1.05)
    c = 14 * np.random.uniform(0.95, 1.05)

    # Slightly random initial conditions near (1,1,1)
    init_state = np.array([1, 1, 1]) + np.random.normal(0, 0.05, 3)

    # Integrate over full duration
    sol = solve_ivp(rossler, (0, total_duration), init_state, t_eval=t_eval_full, args=(a, b, c))

    # Discard the transient part (first 50 s)
    mask = sol.t >= warmup_time
    t_eval = sol.t[mask] - warmup_time
    x_series = sol.y[0][mask]

    # Ensure output matches previous sample length (duration/dt)
    if len(x_series) > int(duration / dt):
        x_series = x_series[:int(duration / dt)]
        t_eval = t_eval[:int(duration / dt)]
    elif len(x_series) < int(duration / dt):
        # pad if tiny rounding mismatch
        pad_len = int(duration / dt) - len(x_series)
        x_series = np.pad(x_series, (0, pad_len), mode='edge')
        t_eval = np.pad(t_eval, (0, pad_len), mode='edge')

    return t_eval, x_series


# -----------------------------
# Noise Helpers
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
                "fisher_info_spectral": safe_scalar(spectral_fisher_information(embedded)),
                "svd_entropy": safe_scalar(nk.entropy_svd(segment, delay=delay, dimension=emb_dim)),
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
        print("Failure summary by exception type:")
        for k, v in fail_counts.items():
            print(f"   {k}: {v} times")
    return df


# -----------------------------
# Dataset creation
# -----------------------------
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

                # Generate Rössler signal and apply noise
                t_eval, clean = generate_roessler_series(run_id=run_id)
                noisy = apply_noise(clean, t_eval, noise_type, intensity)

                clean_norm, _ = zscore_scale(clean)

                if run == 0:
                    fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
                    ax[0].plot(t_eval, clean_norm, color="black", lw=1)
                    ax[0].set_title("Clean Rössler")
                    ax[0].set_ylabel("x(t)")
                    ax[0].grid(alpha=0.3)

                    ax[1].plot(t_eval, noisy, color="tab:red", lw=1)
                    ax[1].set_title(f"{noise_type} noise (intensity={intensity})")
                    ax[1].set_xlabel("Time [s]")
                    ax[1].set_ylabel("x(t)")
                    ax[1].grid(alpha=0.3)

                    plt.tight_layout()
                    out_path = os.path.join(vis_folder, f"Roessler_{noise_type}_intensity{intensity}_run{run_id}.png")
                    plt.savefig(out_path, dpi=200, bbox_inches="tight")
                    plt.close()
                    print(f"Saved Rössler visualization: {out_path}")

                #Extract metrics
                comp_df = extract_complexity_metrics(noisy, t_eval)
                comp_df["noise_type"] = noise_type
                comp_df["noise_intensity"] = intensity
                comp_df["noise_label_task1"] = noise_type
                comp_df["noise_label_task2"] = f"{noise_type}_{intensity}"
                comp_df["signal_id"] = f"{noise_type}_{intensity}_run{run_id}"
                all_dfs.append(comp_df)

    # Balanced for both Task 1 and 2
    clean_dfs = []
    total_clean_runs = len(noise_configs[next(iter(noise_configs))]) * n_runs  # e.g. 3 intensities × 5 = 15
    print(f"Generating {total_clean_runs} clean runs (balanced for Tasks 1 & 2).")

    for run in range(total_clean_runs):
        run_id += 1
        print(f" Clean run {run_id}/{total_clean_runs}")
        t_eval, clean = generate_roessler_series(run_id=run_id + 1000)  # offset for unique seeds

        if run == 0:
            plt.figure(figsize=(10, 2.5))
            plt.plot(t_eval, clean, color="black", lw=1)
            plt.title("Clean Rössler System (no noise)")
            plt.xlabel("Time [s]")
            plt.ylabel("x(t)")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            out_path = os.path.join(vis_folder, f"Roessler_clean_example_run{run_id}.png")
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved clean Rössler visualization: {out_path}")

        clean_norm, _ = zscore_scale(clean)
        comp_df = extract_complexity_metrics(clean_norm, t_eval)
        comp_df["noise_type"] = "none"
        comp_df["noise_intensity"] = 0
        comp_df["noise_label_task1"] = "none"
        comp_df["noise_label_task2"] = "none_0"
        comp_df["signal_id"] = f"none_0_run{run_id}"
        clean_dfs.append(comp_df)

    all_dfs += clean_dfs
    full_df = pd.concat(all_dfs, ignore_index=True)
    dataset_path = os.path.join(base_folder, "roessler_dataset.csv")
    full_df.to_csv(dataset_path, index=False)
    print(f"Dataset created with {total_clean_runs} clean runs: {dataset_path} ({full_df.shape})")

    return full_df


# -----------------------------
# Load or regenerate dataset
# -----------------------------
dataset_path = os.path.join(base_folder, "roessler_dataset.csv")
if os.path.exists(dataset_path) and not force_regenerate:
    print(f"Loading cached dataset: {dataset_path}")
    full_df = pd.read_csv(dataset_path)
else:
    print("Creating new dataset...")
    full_df = create_dataset()

# -----------------------------
# Feature selection
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



def balance_groups_by_runs(full_df, task_label_col):
    """
    Balances the dataset by selecting the same number of complete signal_id groups per label.
    Keeps all windows from each selected run.
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



def run_grouped_cross_validation(task_label_col, task_name,
                                 loss_function="MultiClass",
                                 is_binary=False,
                                 balance_none_label_for_task2=False):
    """
    Stratified 5-fold CV at the group (signal_id) level.
    Automatically balances the clean 'none_0' class for Task 2 if requested.
    """

    X_all = full_df.copy()

    #  balancing for Task 2
    if balance_none_label_for_task2 and "noise_label_task2" in X_all.columns:
        print(f"\n⚖️ Balancing dataset for {task_name} (equal number of runs per label)...")

        # Count minimum number of groups per non-clean label
        min_samples = (
            X_all[X_all["noise_label_task2"] != "none_0"]["signal_id"]
            .value_counts()
            .min()
        )

        # Sample same number of "none_0" groups
        none_subset = X_all[X_all["noise_label_task2"] == "none_0"].sample(
            min_samples, random_state=42
        )

        # Combine with all noisy labels
        X_all = pd.concat([
            X_all[X_all["noise_label_task2"] != "none_0"],
            none_subset
        ], ignore_index=True)

        print(f"Balanced {task_name}: {min_samples} groups per label "
              f"({X_all[task_label_col].nunique()} total labels)")
    else:
        print(f"\n Running {task_name} without balancing adjustment.")

    labels_all = sorted(X_all[task_label_col].unique().tolist()) if not is_binary else [0, 1]

    # Build group table (one label per signal_id)
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

    # Stratified split across groups
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

        rep = classification_report(y_test, y_pred, labels=labels_all, output_dict=True, digits=4)
        fold_df = pd.DataFrame(rep).transpose()
        fold_df["fold"] = fold
        fold_df["fold_accuracy"] = fold_accuracy
        all_reports.append(fold_df)

        # Feature importances
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

    # Save per-fold and mean accuracies
    acc_df = pd.DataFrame({
        "Fold": np.arange(1, len(accuracies) + 1),
        "Accuracy": accuracies,
    })
    acc_df["MeanAccuracy"] = acc_df["Accuracy"].mean()
    acc_df["StdAccuracy"] = acc_df["Accuracy"].std()
    acc_csv_path = os.path.join(results_folder, f"Roessler_cv_accuracies_{task_name.replace(' ', '_').lower()}.csv")
    acc_df.to_csv(acc_csv_path, index=False)
    print(f"Saved accuracies: {acc_csv_path}")

    # Aggregate across folds
    reports_df = pd.concat(all_reports, ignore_index=True)
    fi_df = pd.concat(all_importances, ignore_index=True)
    fi_mean = fi_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False).reset_index()

    reports_df.to_csv(os.path.join(results_folder, f"Roessler_cv_reports_{task_name.replace(' ', '_').lower()}.csv"),
                      index=False)
    fi_mean.to_csv(
        os.path.join(results_folder, f"Roessler_cv_feature_importance_{task_name.replace(' ', '_').lower()}.csv"),
        index=False)

    plot_average_confusion_matrix(cm_all, labels_all, f"Roessler – {task_name}")
    plot_feature_importance_with_errorbars(fi_df, f"Roessler – {task_name}")

    print(f"\n Stratified-by-group 5-fold CV complete for {task_name}.")
    return reports_df, fi_mean


# -----------------------------
# Run grouped CV tasks
# -----------------------------
full_df["noise_present"] = (full_df["noise_type"] != "none").astype(int)

# Task 1: Noise Category
print("\n Running Task 1: Noise Category Classification")
run_grouped_cross_validation(
    task_label_col="noise_label_task1",
    task_name="Noise Category Classification"
)

# Task 2: Noise Category + Intensity

print("\n Running Task 2: Noise Category + Intensity Classification (Balanced by Samples)")

full_df = full_df.sort_values(by=["noise_label_task2", "signal_id", "timestep"], ignore_index=True)

balanced_df = balance_groups_by_runs(full_df, "noise_label_task2")

backup_df = full_df
full_df = balanced_df

run_grouped_cross_validation(
    task_label_col="noise_label_task2",
    task_name="Noise Category + Intensity Classification"
)

full_df = backup_df  # restore original dataset

# Task 3: Noise Present vs Not
print("\n Running Task 3: Noise Present vs Not Classification")
run_grouped_cross_validation(
    task_label_col="noise_present",
    task_name="Noise Present vs Not Classification",
    loss_function="Logloss",
    is_binary=True
)


# ==============================================================
# Create Summary Excel
# ==============================================================

def create_cv_summary_excel(results_folder):
    summary_rows = []

    # detect all Roessler_cv_reports_*.csv files
    report_files = [
        f for f in os.listdir(results_folder)
        if f.startswith("Roessler_cv_reports_") and f.endswith(".csv")
    ]
    if not report_files:
        print("No CV report files found in results folder!")
        print("Files in results_folder:", os.listdir(results_folder))
        return

    for file in sorted(report_files):
        path = os.path.join(results_folder, file)
        df = pd.read_csv(path)

        # task name from file name
        task_name = (
            file.replace("Roessler_cv_reports_", "")
                .replace(".csv", "")
                .replace("_", " ")
                .title()
        )

        # Each fold produced multiple rows; accuracy is repeated in each row via 'fold_accuracy'
        # (you add fold_accuracy to every row), so just take the mean over folds by de-duplicating folds.
        acc = None
        if "fold" in df.columns and "fold_accuracy" in df.columns:
            acc = df.drop_duplicates(subset=["fold"])["fold_accuracy"].mean()

        # Macro avg metrics (more meaningful than averaging over class rows)
        precision = recall = f1 = None
        if "macro avg" in df["Unnamed: 0"].values:
            macro = df[df["Unnamed: 0"] == "macro avg"].iloc[0]
            precision = macro.get("precision", None)
            recall = macro.get("recall", None)
            f1 = macro.get("f1-score", None)
        else:
            # fallback: if label column name differs
            label_col = df.columns[0]
            if "macro avg" in df[label_col].values:
                macro = df[df[label_col] == "macro avg"].iloc[0]
                precision = macro.get("precision", None)
                recall = macro.get("recall", None)
                f1 = macro.get("f1-score", None)

        summary_rows.append({
            "Task": task_name,
            "Accuracy (mean over folds)": acc,
            "Macro Precision": precision,
            "Macro Recall": recall,
            "Macro F1": f1,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(results_folder, "cv_summary_all_tasks.xlsx")
    summary_df.to_excel(summary_path, index=False)

    print(f"\nSummary Excel created: {summary_path}")
    print(summary_df)



# Rerun summary creation
create_cv_summary_excel(results_folder)

