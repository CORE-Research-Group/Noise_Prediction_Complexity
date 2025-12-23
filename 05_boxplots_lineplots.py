# ==============================================================
# Cross-System Global Min–Max Scaling of Complexity Datasets
# Automatic processing for Henon, Roessler, Lorenz, ECG
# More generic dataset discovery (supports roessler_dataset.csv etc.)
# ==============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Global matplotlib font scaling
# --------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 14,
})

# --------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------
base_folders = {
    "Henon": "Henon_Noise_Exp",
    "Lorenz": "Lorenz_Noise_Exp",
    "Roessler": "Roessler_Noise_Exp",
    "ECG": "ECG_Noise_Exp",
}

results_folder = "Global_Comparison"
os.makedirs(results_folder, exist_ok=True)

# --------------------------------------------------------------
# 2. Dataset finder (generic, recursive)
# --------------------------------------------------------------
def iter_csv_files(root_folder, max_depth=2):
    """
    Yield CSV file paths within root_folder up to max_depth subfolder levels.
    max_depth=0 means only the root folder.
    """
    root_folder = os.path.abspath(root_folder)
    root_depth = root_folder.rstrip(os.sep).count(os.sep)

    for dirpath, dirnames, filenames in os.walk(root_folder):
        depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
        if depth > max_depth:
            # do not descend further
            dirnames[:] = []
            continue

        for fn in filenames:
            if fn.lower().endswith(".csv"):
                yield os.path.join(dirpath, fn)

def score_dataset_candidate(path, system_name):
    """
    Score a CSV path as a dataset candidate.
    Higher score = more likely the "right" complexity dataset.
    """
    name = os.path.basename(path).lower()
    sys = system_name.lower()

    score = 0

    # Must have dataset in the filename to be considered in the first place
    if "dataset" not in name:
        return -10_000

    # Strong positive signals
    if name.startswith("dataset_"):
        score += 5
    if f"{sys}_dataset" in name or f"{sys}dataset" in name:
        score += 8
    if sys in name:
        score += 4

    # Negative signals / exclusions
    if "raw_windowed" in name:
        score -= 100
    if "windowed" in name and "raw" in name:
        score -= 50

    # Slight preference for files in the root folder (often your final export lives there)
    parent = os.path.basename(os.path.dirname(path)).lower()
    if parent.endswith("_noise_exp") or parent in {"lorenz_noise_exp", "roessler_noise_exp", "henon_noise_exp", "ecg_noise_exp"}:
        score += 2

    # Size tie-breaker later, but include a small size-based nudge
    try:
        score += min(os.path.getsize(path) / 1_000_000, 5)  # cap at +5
    except OSError:
        pass

    return score

def find_complexity_dataset(folder, system_name, max_depth=2):
    """
    Find the most likely complexity dataset CSV for a system.
    Searches root folder and subfolders up to max_depth.
    """
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return None

    candidates = []
    for p in iter_csv_files(folder, max_depth=max_depth):
        fname = os.path.basename(p).lower()
        if "dataset" not in fname:
            continue
        if "raw_windowed" in fname:
            continue
        candidates.append(p)

    if not candidates:
        print(f"No dataset-like CSV found in {folder}")
        return None

    scored = [(score_dataset_candidate(p, system_name), p) for p in candidates]
    scored.sort(key=lambda x: (x[0], os.path.getsize(x[1]) if os.path.exists(x[1]) else 0), reverse=True)

    best_score, best_path = scored[0]
    print(f"Using dataset for {system_name}: {best_path} (score={best_score:.2f})")
    return best_path

datasets = {
    name: find_complexity_dataset(folder, name, max_depth=2)
    for name, folder in base_folders.items()
}
datasets = {k: v for k, v in datasets.items() if v is not None}

if not datasets:
    raise FileNotFoundError("No complexity datasets found. Check folder paths and filenames.")

# --------------------------------------------------------------
# 3. Load all datasets
# --------------------------------------------------------------
dfs = []
for system, path in datasets.items():
    df = pd.read_csv(path)
    df["system"] = system
    dfs.append(df)
    print(f"Loaded {system}: {df.shape} from {path}")

all_df = pd.concat(dfs, ignore_index=True)
print(f"Combined dataset shape: {all_df.shape}")

# --------------------------------------------------------------
# 4. Identify complexity metrics
# --------------------------------------------------------------
exclude_cols = [
    "timestep", "noise_type", "noise_intensity",
    "noise_label_task1", "noise_label_task2",
    "signal_id", "noise_present", "system"
]

exclude_metrics = {"corr_dim", "higuchi_fd"}

complexity_metrics = [
    c for c in all_df.columns
    if c not in exclude_cols
    and c not in exclude_metrics
    and pd.api.types.is_numeric_dtype(all_df[c])
]

print(f"Complexity metrics ({len(complexity_metrics)}):")
print(", ".join(complexity_metrics))

# --------------------------------------------------------------
# 5. Global min–max scaling
# --------------------------------------------------------------
global_min = all_df[complexity_metrics].min()
global_max = all_df[complexity_metrics].max()

scaled_df = all_df.copy()
scaled_df[complexity_metrics] = (
    scaled_df[complexity_metrics] - global_min
) / (global_max - global_min + 1e-12)

print("Applied global min–max scaling.")

# --------------------------------------------------------------
# 6. Boxplot helper
# --------------------------------------------------------------
def save_boxplot(sub_df, metrics, title, out_path):
    if sub_df.empty:
        return

    plt.figure(figsize=(max(10, len(metrics) * 0.6), 6))
    plt.boxplot(
        [sub_df[m].dropna() for m in metrics],
        labels=metrics,
        patch_artist=True,
        medianprops=dict(color="black")
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Global scaled value (0–1)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved boxplot: {out_path}")

# --------------------------------------------------------------
# 7. Per-system visualizations
# --------------------------------------------------------------
colors = list(plt.cm.tab20.colors)
linestyles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 5, 1, 5))]
markers = ["o", "s", "D", "^", "v", "p", "*", "X", "P", "h"]

for system, sys_df in scaled_df.groupby("system"):
    sys_folder = os.path.join(results_folder, system)
    os.makedirs(sys_folder, exist_ok=True)

    print(f"Processing system: {system}")

    # Task 1: Noise category boxplots
    if "noise_label_task1" in sys_df.columns:
        for cat in sorted(sys_df["noise_label_task1"].dropna().unique()):
            sub = sys_df[sys_df["noise_label_task1"] == cat]
            outp = os.path.join(sys_folder, f"{system}_boxplot_task1_{cat}.png")
            save_boxplot(sub, complexity_metrics, f"{system} - Task 1: {cat}", outp)
    else:
        print(f"Task 1 skipped for {system} (missing noise_label_task1).")

    # Task 2: Mean metric vs noise intensity (per noise type)
    if {"noise_type", "noise_intensity"}.issubset(sys_df.columns):
        noise_types = [n for n in sys_df["noise_type"].dropna().unique() if n != "none"]
        baseline = sys_df[sys_df["noise_type"] == "none"][complexity_metrics].mean().to_dict()

        for nt in noise_types:
            sub = sys_df[sys_df["noise_type"] == nt]
            if sub.empty:
                continue

            means = (
                sub.groupby("noise_intensity")[complexity_metrics]
                .mean()
                .reset_index()
                .sort_values("noise_intensity")
            )

            base_row = {"noise_intensity": 0.0}
            for m in complexity_metrics:
                base_row[m] = baseline.get(m, np.nan)

            means = pd.concat([pd.DataFrame([base_row]), means], ignore_index=True).sort_values("noise_intensity")

            plt.figure(figsize=(10, 6))
            for i, m in enumerate(complexity_metrics):
                plt.plot(
                    means["noise_intensity"], means[m],
                    label=m,
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    marker=markers[i % len(markers)],
                    linewidth=1.3,
                    markersize=4,
                    alpha=0.9
                )

            plt.xlabel("Noise intensity (0 = no noise)")
            plt.ylabel("Global scaled value (0-1)")
            plt.title(f"{system} - Task 2: Mean metrics vs intensity ({nt})")
            plt.grid(alpha=0.35)
            plt.legend(
                fontsize=10,
                ncol=4,
                frameon=False,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.45)
            )
            plt.subplots_adjust(bottom=0.28)

            outp = os.path.join(sys_folder, f"{system}_lines_task2_{nt}.png")
            plt.savefig(outp, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved line plot: {outp}")
    else:
        print(f"Task 2 skipped for {system} (missing noise_type or noise_intensity).")

    # Task 3: Noise vs no-noise boxplots
    if "noise_type" in sys_df.columns:
        tmp = sys_df.copy()
        tmp["noise_present"] = (tmp["noise_type"] != "none").astype(int)

        noise_df = tmp[tmp["noise_present"] == 1]
        clean_df = tmp[tmp["noise_present"] == 0]

        save_boxplot(
            noise_df,
            complexity_metrics,
            f"{system} - Task 3: Noise present",
            os.path.join(sys_folder, f"{system}_boxplot_task3_noise.png")
        )

        save_boxplot(
            clean_df,
            complexity_metrics,
            f"{system} - Task 3: No noise",
            os.path.join(sys_folder, f"{system}_boxplot_task3_no_noise.png")
        )
    else:
        print(f"Task 3 skipped for {system} (missing noise_type).")

# --------------------------------------------------------------
# 8. Save globally scaled dataset
# --------------------------------------------------------------
scaled_path = os.path.join(results_folder, "all_systems_global_minmax_scaled_complexity.csv")
scaled_df.to_csv(scaled_path, index=False)

print(f"Saved globally scaled dataset to {scaled_path}")
print("Global-comparable complexity analysis complete.")
