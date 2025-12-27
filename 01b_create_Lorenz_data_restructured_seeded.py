"""
Lorenz data generation (step 1)

Generates clean Lorenz x(t), creates noisy variants, saves signals (.npz) + metadata (csv),
and writes example plots (png + eps). No metrics and no ML.
"""

from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib

# Keep the backend consistent with the original scripts
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# -----------------------------
# RNG helper (NEW)
# -----------------------------
def reseed_all(seed: int) -> None:
    """
    Reset all relevant RNGs to avoid leakage and ensure per-run determinism.
    Note: PYTHONHASHSEED is only fully effective if set before Python starts.
    """
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))


# -----------------------------
# Configuration
# -----------------------------
@dataclass(frozen=True)
class LorenzConfig:
    """
    Configuration container for Lorenz data generation.

    Parameters
    ----------
    duration:
        Duration (seconds) after removing transient warm-up.
    dt:
        Sampling step size (seconds).
    warmup_time:
        Transient duration (seconds) that is simulated but removed.
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
    duration: float = 5000.0
    dt: float = 0.1
    warmup_time: float = 200.0

    # schedule
    n_runs: int = 20

    # noise configs (same keys / values as original)
    noise_configs: Dict[str, List[float]] = None

    # base output folder
    base_folder: str = "Lorenz_Noise_Exp"

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

    Parameters
    ----------
    base_folder:
        Base folder for all Lorenz experiment outputs.
    tag:
        Text label appended to the timestamp.

    Returns
    -------
    root:
        Full path to the created output directory.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(base_folder, f"{ts}_{tag}")
    os.makedirs(root, exist_ok=True)
    return root


def ensure_dirs(root: str) -> Dict[str, str]:
    """
    Create and return the required subdirectories for a run.

    Parameters
    ----------
    root:
        Root directory for a single run.

    Returns
    -------
    paths:
        Dictionary of named paths used by the script.
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
# Lorenz system
# -----------------------------
def lorenz_rhs(
    t: float,
    state: np.ndarray,
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
    rho: float = 28.0,
) -> List[float]:
    """
    Right-hand side of the Lorenz system.
    """
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]


def generate_lorenz_series(cfg: LorenzConfig, run_seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate one clean Lorenz x(t) series.
    - Simulate duration + warmup_time, then discard warmup_time.
    - Randomize parameters and initial conditions per run_seed.
    - Return fixed-length arrays (int(duration/dt)).
    """
    reseed_all(run_seed)

    total_duration = cfg.duration + cfg.warmup_time
    t_eval_full = np.arange(0.0, total_duration, cfg.dt)

    # slight parameter variations
    sigma = 10.0 * np.random.uniform(0.95, 1.05)
    beta = (8.0 / 3.0) * np.random.uniform(0.95, 1.05)
    rho = 28.0 * np.random.uniform(0.95, 1.05)

    # initial conditions near (1, 1, 1)
    init_state = np.array([1.0, 1.0, 1.0]) + np.random.normal(0.0, 0.05, 3)

    sol = solve_ivp(
        lorenz_rhs,
        (0.0, total_duration),
        init_state,
        t_eval=t_eval_full,
        args=(sigma, beta, rho),
    )

    mask = sol.t >= cfg.warmup_time
    t_eval = sol.t[mask] - cfg.warmup_time
    x_series = sol.y[0][mask]

    target_len = int(cfg.duration / cfg.dt)
    if len(x_series) > target_len:
        x_series = x_series[:target_len]
        t_eval = t_eval[:target_len]
    elif len(x_series) < target_len:
        pad_len = target_len - len(x_series)
        x_series = np.pad(x_series, (0, pad_len), mode="edge")
        t_eval = np.pad(t_eval, (0, pad_len), mode="edge")

    meta = {
        "sigma": float(sigma),
        "beta": float(beta),
        "rho": float(rho),
        "init_state": init_state.tolist(),
        "run_seed": int(run_seed),
        "duration": float(cfg.duration),
        "dt": float(cfg.dt),
        "warmup_time": float(cfg.warmup_time),
    }
    return t_eval, x_series, meta


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
        noisy = clean_norm + np.random.normal(0.0, intensity, size=len(clean_norm))
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
    """
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

    fig.tight_layout()
    return fig


def plot_example_clean_only(t_eval: np.ndarray, clean: np.ndarray) -> plt.Figure:
    """
    Single-panel clean example plot.
    """
    fig = plt.figure(figsize=(10, 2.5))
    plt.plot(t_eval, clean, color="black", lw=1)
    plt.title("Clean Lorenz System (no noise)")
    plt.xlabel("Time [s]")
    plt.ylabel("x(t)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# -----------------------------
# Main generation routine
# -----------------------------
def generate_lorenz_dataset(cfg: LorenzConfig) -> str:
    """
    Generate and store the full dataset with run_id as the only run identifier.

    Outer loop: run_id = 1..cfg.n_runs
      - generate ONE clean Lorenz realization per run_id
      - for that same clean signal, generate ALL noisy variants across cfg.noise_configs
    """
    print("[1/6] Start dataset generation")
    print(f"      base_folder={cfg.base_folder}")
    print(f"      duration={cfg.duration}, dt={cfg.dt}, warmup_time={cfg.warmup_time}")
    print(f"      n_runs={cfg.n_runs}")
    print(f"      noise_configs={cfg.noise_configs}")

    reseed_all(cfg.global_seed)
    print(f"[2/6] Set global seed: {cfg.global_seed}")

    root = make_run_root(cfg.base_folder, tag="lorenz_data")
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
        run_seed = 42 * run_id

        print(f"      run_id={run_id:04d} seed={run_seed}")

        # per-run seed (clean generation)
        reseed_all(run_seed)
        t_eval, clean, gen_meta = generate_lorenz_series(cfg, run_seed=run_seed)

        # Clean (exact same normalization logic)
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
            fig = plot_example_clean_only(t_eval, clean)
            out_base = os.path.join(paths["fig_signal_examples"], f"Lorenz_clean_example_run{run_id:04d}")
            save_png_eps(fig, out_base)
            plt.close(fig)

        records.append(
            {
                "split": "clean",
                "noise_type": "none",
                "noise_intensity": 0.0,
                "run_id": run_id,
                "rep": 0,
                "rep_in_config": 0,
                "config_id": "none_0",
                "signal_id": str(run_id),
                "npz_path": os.path.relpath(out_clean, root),
                **gen_meta,
            }
        )
        done += 1

        # Noisy variants for the same run_id
        for noise_type, levels in cfg.noise_configs.items():
            for intensity in levels:
                noise_intensity = float(intensity)
                config_id = f"{noise_type}_{noise_intensity}"

                # per-(run, config) reseed to ensure unique noise draws
                config_seed = int(1_000_000 * run_id + 10_000 * (abs(hash(noise_type)) % 100) + int(noise_intensity * 1000))
                reseed_all(config_seed)

                clean_norm2, noisy = apply_noise(clean, t_eval, noise_type, noise_intensity)

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
                    fig = plot_example_clean_vs_noisy(t_eval, clean, noisy, noise_type, noise_intensity)
                    out_base = os.path.join(
                        paths["fig_signal_examples"],
                        f"Lorenz_{noise_type}_intensity{intensity}_run{run_id:04d}",
                    )
                    save_png_eps(fig, out_base)
                    plt.close(fig)

                records.append(
                    {
                        "split": "noisy",
                        "noise_type": noise_type,
                        "noise_intensity": noise_intensity,
                        "run_id": run_id,
                        "rep": 0,
                        "rep_in_config": 0,
                        "config_id": config_id,
                        "signal_id": str(run_id),
                        "npz_path": os.path.relpath(out_npz, root),
                        **gen_meta,
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
    cfg = LorenzConfig()
    generate_lorenz_dataset(cfg)
