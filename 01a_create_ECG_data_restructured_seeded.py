"""
ECG data generation (step 1)

Generates clean synthetic ECG, creates noisy variants, saves signals (.npz) + metadata (csv),
and writes example plots (png + eps). No metrics and no ML.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib

# Keep the backend consistent with the original scripts
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import neurokit2 as nk

import random

# -----------------------------
# Seeding helper
# -----------------------------
def reseed_all(seed: int) -> None:
    """Reset as many RNGs as reasonably possible for reproducibility."""
    # NOTE: setting PYTHONHASHSEED at runtime won't affect an already-started interpreter,
    # but keeping it here helps when scripts are launched with a clean process environment.
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))



# -----------------------------
# Configuration
# -----------------------------
@dataclass(frozen=True)
class ECGConfig:
    """
    Configuration container for ECG data generation.

    Parameters
    ----------
    n_samples:
        Number of samples per ECG segment.
    target_rate:
        Sampling rate in Hz (e.g., 500).
    warmup_time:
        Transient/warm-up time in seconds that is simulated and discarded before returning
        the final n_samples segment.
    n_runs:
        Replicates per (noise_type, intensity).
    noise_configs:
        Mapping: noise_type -> list of intensities.
    base_folder:
        Base output folder.
    global_seed:
        Seed for deterministic top-level behavior.
    """

    # signal generation
    n_samples: int = 50000
    target_rate: int = 500
    warmup_time: float = 2000

    # schedule
    n_runs: int = 20

    # noise configs (same keys / values as original)
    noise_configs: Dict[str, List[float]] = None

    # base output folder
    base_folder: str = "ECG_Noise_Exp"

    # deterministic global seed (used only for selection stability)
    global_seed: int = 42

    def __post_init__(self):
        # Provide the default noise configuration if none is explicitly passed.
        if self.noise_configs is None:
            object.__setattr__(
                self,
                "noise_configs",
                {
                    "gaussian": [1, 0.3, 0.1],
                    "pink": [2, 1, 0.2],
                    "low_freq": [0.2, 1, 2],
                },
            )


# -----------------------------
# Folder creation
# -----------------------------
def make_run_root(base_folder: str, tag: str = "data_generation") -> str:
    """
    Create a timestamped output directory.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(base_folder, f"{ts}_{tag}")
    os.makedirs(root, exist_ok=True)
    return root


def ensure_dirs(root: str) -> Dict[str, str]:
    """
    Create and return the required subdirectories for a run.
    """
    paths = {
        "root": root,
        "data": os.path.join(root, "data"),
        "data_clean": os.path.join(root, "data", "clean"),
        "data_noisy": os.path.join(root, "data", "noisy"),
        "figures": os.path.join(root, "figures"),
        "fig_signal_examples": os.path.join(root, "figures", "signal_examples"),
        "meta": os.path.join(root, "metadata"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


# -----------------------------
# ECG generator (same logic as original)
# -----------------------------
def generate_ecg_series_variable(
    n_samples: int = 2000,
    target_rate: int = 500,
    run_id: int = 0,
    warmup_time: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a reproducible synthetic ECG with controlled variation in heart rate and phase shift.
    Each run_id produces a slightly different but deterministic waveform.
    """
    np.random.seed(42 * run_id)

    # Natural variation per run
    heart_rate = np.random.uniform(55, 75)  # bpm variation
    phase_shift = np.random.uniform(-0.02, 0.02)  # seconds of time drift

    warmup_samples = int(round(float(warmup_time) * target_rate))
    duration = int(np.ceil((n_samples + warmup_samples) / target_rate))

    ecg_signal = nk.ecg_simulate(
        duration=duration,
        sampling_rate=target_rate,
        heart_rate=heart_rate,
        noise=0.00
    )

    # apply phase shift
    if abs(phase_shift) > 0:
        t = np.arange(len(ecg_signal)) / target_rate
        t_shifted = t + phase_shift
        interp_func = interp1d(t_shifted, ecg_signal, fill_value="extrapolate")
        ecg_signal = interp_func(t)

    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=target_rate)

    # discard warm-up / transient
    if warmup_samples > 0:
        if len(ecg_cleaned) > warmup_samples:
            ecg_cleaned = ecg_cleaned[warmup_samples:]
        else:
            ecg_cleaned = np.array([], dtype=ecg_cleaned.dtype)

    # pad to fixed length
    if len(ecg_cleaned) >= n_samples:
        ecg_cleaned = ecg_cleaned[:n_samples]
    else:
        pad_len = n_samples - len(ecg_cleaned)
        ecg_cleaned = np.pad(ecg_cleaned, (0, pad_len), mode="constant")

    t_eval = np.arange(len(ecg_cleaned)) / target_rate
    return t_eval, ecg_cleaned


# -----------------------------
# Noise helpers (same behavior)
# -----------------------------
def zscore_scale(x: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Z-score normalize a signal and return (x_norm, (mean, std)).
    """
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd == 0.0:
        return x - mu, (mu, 1.0)
    return (x - mu) / sd, (mu, sd)


def generate_pink_noise(n: int, scale: float) -> np.ndarray:
    """
    Simple 1/f noise via frequency-domain shaping.
    """
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0

    real = np.random.normal(0.0, 1.0, size=len(freqs))
    imag = np.random.normal(0.0, 1.0, size=len(freqs))
    spectrum = real + 1j * imag

    spectrum /= np.sqrt(freqs)

    pink = np.fft.irfft(spectrum, n=n)
    pink = (pink - np.mean(pink)) / (np.std(pink) + 1e-12)
    return scale * pink


def add_gaussian_noise_relative(x_norm: np.ndarray, intensity: float = 0.1) -> np.ndarray:
    """
    Add Gaussian noise to an already-normalized signal.
    """
    return x_norm + np.random.normal(0.0, intensity, size=len(x_norm))


def apply_noise(
    signal: np.ndarray,
    t_eval: np.ndarray,
    noise_type: str,
    intensity: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply noise to a z-score normalized signal.
    Returns (clean_norm, noisy).
    """
    clean_norm, _ = zscore_scale(signal)

    if noise_type == "gaussian":
        noisy = add_gaussian_noise_relative(clean_norm, intensity)
    elif noise_type == "pink":
        noisy = clean_norm + generate_pink_noise(len(clean_norm), intensity)
    elif noise_type == "low_freq":
        freq = np.random.uniform(0.45, 0.55)
        phi = np.random.uniform(-0.4 * np.pi, 0.4 * np.pi)
        noisy = clean_norm + intensity * np.sin(2 * np.pi * freq * t_eval + phi)
    else:
        noisy = clean_norm.copy()

    return clean_norm, noisy


# -----------------------------
# Plotting (same layout; save png + eps)
# -----------------------------
def save_png_eps(fig: plt.Figure, out_basepath_no_ext: str, dpi: int = 200) -> None:
    """
    Save figure to both PNG and EPS.
    """
    fig.savefig(out_basepath_no_ext + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_basepath_no_ext + ".eps", bbox_inches="tight")


def plot_example_clean_vs_noisy(
    t_eval: np.ndarray,
    clean_norm: np.ndarray,
    noisy: np.ndarray,
    noise_type: str,
    intensity: float,
) -> plt.Figure:
    """
    Two-panel example plot: clean (top) and noisy (bottom).
    Matches the original ECG plotting layout (sharex + sharey).
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True, sharey=True)

    ax[0].plot(t_eval, clean_norm, color="black", lw=1)
    ax[0].set_title("Clean ECG (z-scored)")
    ax[0].set_ylabel("Amplitude (z)")
    ax[0].grid(alpha=0.3)

    ax[1].plot(t_eval, noisy, color="tab:red", lw=1)
    ax[1].set_title(f"{noise_type} noise (intensity={intensity}) [on z-scored ECG]")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Amplitude (z)")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_example_clean_only(t_eval: np.ndarray, clean_norm: np.ndarray) -> plt.Figure:
    """
    Single-panel clean example plot (z-scored), matching the original ECG script.
    """
    fig = plt.figure(figsize=(10, 2.5))
    plt.plot(t_eval, clean_norm, color="black", lw=1)
    plt.title("Clean ECG (no noise)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# -----------------------------
# Main generation routine
# -----------------------------
def generate_ecg_dataset(cfg: ECGConfig) -> str:
    """
    Generate and store the full dataset with run_id as the only run identifier.

    Outer loop: run_id = 1..cfg.n_runs
      - generate ONE clean ECG realization per run_id
      - for that same clean signal, generate ALL noisy variants across cfg.noise_configs
    """
    print("[1/6] Start dataset generation")
    print(f"      base_folder={cfg.base_folder}")
    print(f"      n_samples={cfg.n_samples}, target_rate={cfg.target_rate}")
    print(f"      n_runs={cfg.n_runs}")
    print(f"      noise_configs={cfg.noise_configs}")

    reseed_all(cfg.global_seed)
    print(f"[2/6] Set global seed: {cfg.global_seed}")

    root = make_run_root(cfg.base_folder, tag="ecg_data")
    paths = ensure_dirs(root)
    print(f"[3/6] Created run folder: {root}")
    print(f"      data_clean={paths['data_clean']}")
    print(f"      data_noisy={paths['data_noisy']}")
    print(f"      figures={paths['figures']}")
    print(f"      metadata={paths['meta']}")

    with open(os.path.join(paths["meta"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    print("[4/6] Wrote config.json")

    records: List[Dict[str, Any]] = []

    num_noisy_configs = sum(len(levels) for levels in cfg.noise_configs.values())
    total_records = cfg.n_runs * (1 + num_noisy_configs)
    done = 0

    print("[5/6] Generating runs (outer loop: run_id)")
    for run_id in range(1, cfg.n_runs + 1):
        # Reset RNGs per run to avoid cross-run leakage
        reseed_all(cfg.global_seed + run_id)
        # ECG generator uses np.random.seed(42 * run_id) internally
        ecg_run_id = run_id

        print(f"      run_id={run_id:04d} (clean seed basis ecg_run_id={ecg_run_id})")

        t_eval, clean = generate_ecg_series_variable(
            n_samples=cfg.n_samples,
            target_rate=cfg.target_rate,
            run_id=ecg_run_id,
            warmup_time=cfg.warmup_time,
        )

        # -----------------
        # Clean (exact same normalization logic)
        # -----------------
        clean_norm, _ = zscore_scale(clean)

        out_clean = os.path.join(paths["data_clean"], f"run_{run_id:04d}.npz")
        np.savez_compressed(
            out_clean,
            t_eval=t_eval.astype(np.float64),
            clean=clean.astype(np.float64),
            clean_norm=clean_norm.astype(np.float64),
        )

        # Example visualization: only for the first run
        if run_id == 1:
            print("          saving clean example plot")
            fig = plot_example_clean_only(t_eval, clean_norm)
            out_base = os.path.join(paths["fig_signal_examples"], f"ECG_clean_example_run{run_id:04d}")
            save_png_eps(fig, out_base)
            plt.close(fig)

        records.append(
            {
                "split": "clean",
                "noise_type": "none",
                "noise_intensity": 0.0,
                "run_id": run_id,
                "rep": 0,
                "signal_id": str(run_id),
                "npz_path": os.path.relpath(out_clean, root),
                "n_samples": int(cfg.n_samples),
                "target_rate": int(cfg.target_rate),
                "warmup_time": float(cfg.warmup_time),
                "ecg_run_id": int(ecg_run_id),
            }
        )
        done += 1

        # -----------------
        # Noisy variants for the same clean run_id
        # -----------------
        noise_cfg_idx = 0
        for noise_type, levels in cfg.noise_configs.items():
            for intensity in levels:
                # Reset RNGs per (run_id, noise setting) so each variant is independent
                reseed_all(cfg.global_seed + run_id * 10_000 + noise_cfg_idx)
                noise_cfg_idx += 1
                clean_norm2, noisy = apply_noise(clean, t_eval, noise_type, float(intensity))

                out_dir = os.path.join(paths["data_noisy"], noise_type, f"intensity_{intensity}")
                os.makedirs(out_dir, exist_ok=True)
                out_npz = os.path.join(out_dir, f"run_{run_id:04d}.npz")

                np.savez_compressed(
                    out_npz,
                    t_eval=t_eval.astype(np.float64),
                    clean=clean.astype(np.float64),
                    clean_norm=clean_norm2.astype(np.float64),
                    noisy=noisy.astype(np.float64),
                )

                # Example visualization: only for the first run per configuration
                if run_id == 1:
                    print(f"          saving example plot for noise={noise_type}, intensity={intensity}")
                    fig = plot_example_clean_vs_noisy(t_eval, clean_norm2, noisy, noise_type, float(intensity))
                    out_base = os.path.join(
                        paths["fig_signal_examples"],
                        f"ECG_{noise_type}_intensity{intensity}_run{run_id:04d}",
                    )
                    save_png_eps(fig, out_base)
                    plt.close(fig)

                records.append(
                    {
                        "split": "noisy",
                        "noise_type": noise_type,
                        "noise_intensity": float(intensity),
                        "run_id": run_id,
                        "rep": 0,
                        "signal_id": str(run_id),
                        "npz_path": os.path.relpath(out_npz, root),
                        "n_samples": int(cfg.n_samples),
                        "target_rate": int(cfg.target_rate),
                        "warmup_time": float(cfg.warmup_time),
                        "ecg_run_id": int(ecg_run_id),
                    }
                )
                done += 1

        print(f"      progress: {done}/{total_records}")

    print("[6/6] Writing metadata index.csv")
    meta_df = pd.DataFrame.from_records(records)
    meta_csv = os.path.join(paths["meta"], "index.csv")
    meta_df.to_csv(meta_csv, index=False)

    print("Done")
    print(f"  dataset root: {root}")
    print(f"  metadata index: {meta_csv}")
    print(f"  total records: {len(records)} (expected={total_records})")

    return root

if __name__ == "__main__":
    cfg = ECGConfig()
    generate_ecg_dataset(cfg)