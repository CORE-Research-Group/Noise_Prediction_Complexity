"""
AR(1) data generation (step 1)

Generates clean AR(1), creates noisy variants, saves signals (.npz) + metadata (csv),
and writes example plots (png + eps). No metrics and no ML.
"""

from __future__ import annotations

import os
import json
import random
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib

# Keep the backend consistent with the original scripts
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# =============================================================================
# RNG helper (ONLY ADDITION FOR SEED CONTROL)
# =============================================================================
def reseed_all(seed: int) -> None:
    """
    Reset all relevant random number generators to avoid leakage.
    Note: PYTHONHASHSEED is only fully effective if set before Python starts,
    but we still set it here for completeness.
    """
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))


def stable_config_offset(noise_type: str, intensity: float, modulo: int = 1_000_000) -> int:
    """
    Stable, cross-run/process offset for per-(noise_type,intensity) reseeding.
    Avoids Python's randomized built-in hash().
    """
    s = f"{noise_type}|{float(intensity)}".encode("utf-8")
    h = hashlib.md5(s).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False) % int(modulo)


# -----------------------------
# Configuration
# -----------------------------
@dataclass(frozen=True)
class AR1Config:
    """
    Configuration container for AR(1) data generation.
    """

    # signal generation
    n_samples: int = 50000
    timestep_size: float = 0.1
    warmup_time: float = 200.0  # seconds

    # AR(1) parameters
    phi_base: float = 0.85
    phi_jitter: float = 0.03
    sigma: float = 1.0

    # schedule
    n_runs: int = 20

    # noise configs
    noise_configs: Dict[str, List[float]] = None

    # base output folder
    base_folder: str = "AR1_Noise_Exp"

    # deterministic global seed (used only for selection stability)
    global_seed: int = 42

    def __post_init__(self):
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
# AR(1) generator (deterministic per run_id)
# -----------------------------
def generate_ar1_series_variable(
    n_samples: int,
    timestep_size: float,
    run_id: int,
    warmup_time: float,
    phi_base: float,
    phi_jitter: float,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Generate a reproducible AR(1) process:
        x_t = phi * x_{t-1} + eps_t,  eps_t ~ N(0, sigma^2)

    - phi is mildly varied per run_id (deterministic via per-run reseeding).
    - warmup_time seconds are generated and dropped.
    - returns (t_eval, series, info_dict).
    """
    # IMPORTANT: per-run deterministic seed (no leakage across runs)
    reseed_all(42 * int(run_id))

    # keep AR(1) stationary: |phi| < 1
    phi = float(phi_base + np.random.uniform(-phi_jitter, phi_jitter))
    phi = float(np.clip(phi, -0.95, 0.95))

    warmup_samples = int(round(float(warmup_time) / float(timestep_size)))
    total_len = warmup_samples + int(n_samples)

    x = 0.0
    xs = np.empty(total_len, dtype=float)
    eps = np.random.normal(0.0, float(sigma), size=total_len)

    for t in range(total_len):
        x = phi * x + eps[t]
        xs[t] = x

    series = xs[warmup_samples:]
    t_eval = np.arange(len(series), dtype=float) * float(timestep_size)

    info = {
        "phi": float(phi),
        "sigma": float(sigma),
        "warmup_samples": float(warmup_samples),
    }
    return t_eval, series, info


# -----------------------------
# Noise helpers (same behavior as your other generators)
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
    return float(scale) * pink


def add_gaussian_noise_relative(x_norm: np.ndarray, intensity: float = 0.1) -> np.ndarray:
    """
    Add Gaussian noise to an already-normalized signal.
    """
    return x_norm + np.random.normal(0.0, float(intensity), size=len(x_norm))


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
        noisy = add_gaussian_noise_relative(clean_norm, float(intensity))
    elif noise_type == "pink":
        noisy = clean_norm + generate_pink_noise(len(clean_norm), float(intensity))
    elif noise_type == "low_freq":
        freq = np.random.uniform(0.45, 0.55)
        phi = np.random.uniform(-0.4 * np.pi, 0.4 * np.pi)
        noisy = clean_norm + float(intensity) * np.sin(2 * np.pi * freq * t_eval + phi)
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
    sharex + sharey, consistent with the other generators.
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True, sharey=True)

    ax[0].plot(t_eval, clean_norm, color="black", lw=1)
    ax[0].set_title("Clean AR(1) (z-scored)")
    ax[0].set_ylabel("Amplitude (z)")
    ax[0].grid(alpha=0.3)

    ax[1].plot(t_eval, noisy, color="tab:red", lw=1)
    ax[1].set_title(f"{noise_type} noise (intensity={intensity}) [on z-scored AR(1)]")
    ax[1].set_xlabel("Time [a.u.]")
    ax[1].set_ylabel("Amplitude (z)")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_example_clean_only(t_eval: np.ndarray, clean_norm: np.ndarray) -> plt.Figure:
    """
    Single-panel clean example plot (z-scored).
    """
    fig = plt.figure(figsize=(10, 2.5))
    plt.plot(t_eval, clean_norm, color="black", lw=1)
    plt.title("Clean AR(1) (no noise)")
    plt.xlabel("Time [a.u.]")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# -----------------------------
# Main generation routine
# -----------------------------
def generate_ar1_dataset(cfg: AR1Config) -> str:
    """
    Generate and store the full dataset with run_id as the only run identifier.

    Outer loop: run_id = 1..cfg.n_runs
      - generate ONE clean AR(1) realization per run_id
      - for that same clean signal, generate ALL noisy variants across cfg.noise_configs
    """
    print("[1/6] Start dataset generation")
    print(f"      base_folder={cfg.base_folder}")
    print(f"      n_samples={cfg.n_samples}, timestep_size={cfg.timestep_size}")
    print(f"      n_runs={cfg.n_runs}")
    print(f"      noise_configs={cfg.noise_configs}")

    reseed_all(cfg.global_seed)
    print(f"[2/6] Set global seed: {cfg.global_seed}")

    root = make_run_root(cfg.base_folder, tag="ar1_data")
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
        ar1_run_id = run_id

        print(f"      run_id={run_id:04d} ar1_run_id={ar1_run_id}")

        # Clean generation is already reseeded inside generate_ar1_series_variable()
        t_eval, clean, info = generate_ar1_series_variable(
            n_samples=cfg.n_samples,
            timestep_size=cfg.timestep_size,
            run_id=ar1_run_id,
            warmup_time=cfg.warmup_time,
            phi_base=cfg.phi_base,
            phi_jitter=cfg.phi_jitter,
            sigma=cfg.sigma,
        )

        clean_norm, _ = zscore_scale(clean)

        out_clean = os.path.join(paths["data_clean"], f"run_{run_id:04d}.npz")
        np.savez_compressed(
            out_clean,
            t_eval=t_eval.astype(np.float64),
            clean=clean.astype(np.float64),
            clean_norm=clean_norm.astype(np.float64),
        )

        if run_id == 1:
            print("          saving clean example plot")
            fig = plot_example_clean_only(t_eval, clean_norm)
            out_base = os.path.join(paths["fig_signal_examples"], f"AR1_clean_example_run{run_id:04d}")
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
                "timestep_size": float(cfg.timestep_size),
                "warmup_time": float(cfg.warmup_time),
                "ar1_run_id": int(ar1_run_id),
                "ar1_phi": float(info["phi"]),
                "ar1_sigma": float(info["sigma"]),
            }
        )
        done += 1

        # Noise variants: reseed per (run_id, noise_type, intensity)
        run_seed = 42 * int(run_id)
        for noise_type, levels in cfg.noise_configs.items():
            for intensity in levels:
                reseed_all(int(run_seed + stable_config_offset(noise_type, float(intensity))))

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

                if run_id == 1:
                    print(f"          saving example plot for noise={noise_type}, intensity={intensity}")
                    fig = plot_example_clean_vs_noisy(t_eval, clean_norm2, noisy, noise_type, float(intensity))
                    out_base = os.path.join(
                        paths["fig_signal_examples"],
                        f"AR1_{noise_type}_intensity{intensity}_run{run_id:04d}",
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
                        "timestep_size": float(cfg.timestep_size),
                        "warmup_time": float(cfg.warmup_time),
                        "ar1_run_id": int(ar1_run_id),
                        "ar1_phi": float(info["phi"]),
                        "ar1_sigma": float(info["sigma"]),
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
    cfg = AR1Config()
    generate_ar1_dataset(cfg)
