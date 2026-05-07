"""Raman loading, calculations, plotting, and interactive prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import interp1d

from .common import (
    WorkflowError,
    apply_plot_style,
    ask,
    load_json,
    natural_sort_key,
    parse_indices,
    print_numbered,
    prompt_for_path,
    save_json,
)


CUTOFF_WAVENUMBER = 1100
D_BAND_RANGE = (1300, 1400)
G_BAND_RANGE = (1450, 1650)
G_FWHM_RANGE = (1500, 1650)
COLORS = ["#000000", "#7E1FD1", "#26C2FF", "#E16462", "#FFB000", "#6F728C", "#00A86A"]


def baseline_als(y: np.ndarray, lam: float = 1e6, p: float = 0.001, n_iter: int = 10) -> np.ndarray:
    length = len(y)
    D = sp.diags([1, -2, 1], [0, 1, 2], shape=(length - 2, length))
    w = np.ones(length)
    for _ in range(n_iter):
        W = sp.diags(w, 0)
        baseline = spla.spsolve(W + lam * D.T @ D, w * y)
        w = p * (y > baseline) + (1 - p) * (y < baseline)
    return baseline


def calculate_fwhm(wavenumbers: np.ndarray, intensities: np.ndarray) -> float | None:
    if len(wavenumbers) < 2:
        return None
    half_max = np.min(intensities) + 0.5 * (np.max(intensities) - np.min(intensities))
    interp = interp1d(wavenumbers, intensities - half_max, kind="linear", fill_value="extrapolate")
    sign_changes = np.where(np.diff(np.sign(interp(wavenumbers))))[0]
    if len(sign_changes) >= 2:
        return float(wavenumbers[sign_changes[-1]] - wavenumbers[sign_changes[0]])
    return None


def integrate_band_area(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    band_range: tuple[float, float],
    clip_negative: bool = True,
) -> float:
    mask = (wavenumbers >= band_range[0]) & (wavenumbers <= band_range[1])
    x = wavenumbers[mask]
    y = intensities[mask]
    if len(x) < 2:
        return float("nan")
    if clip_negative:
        y = np.clip(y, 0, None)
    return float(np.trapz(y, x))


def load_raman_spectrum(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep=r"\s+", header=None, names=["Wavenumber", "Intensity"], engine="python")
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    return data.sort_values("Wavenumber").reset_index(drop=True)


def find_raman_files(data_folder: Path) -> dict[str, Path]:
    if not data_folder.exists():
        raise WorkflowError(f"Raman folder does not exist: {data_folder}")
    files = sorted(data_folder.glob("*.txt"), key=lambda p: natural_sort_key(p.stem))
    return {path.stem: path for path in files}


def settings_path(data_folder: Path) -> Path:
    return data_folder / ".raman_memory.json"


def analyze_raman_file(path: Path, horizontal_shift: float = 0.0) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = load_raman_spectrum(path)
    data = data[data["Wavenumber"] >= CUTOFF_WAVENUMBER].copy()
    data["Wavenumber"] = data["Wavenumber"] + horizontal_shift
    data["Baseline"] = baseline_als(data["Intensity"].to_numpy(), lam=1e6, p=0.001)
    data["Corrected"] = data["Intensity"] - data["Baseline"]
    corrected_max = data["Corrected"].max()
    data["Normalized"] = data["Corrected"] / corrected_max if corrected_max != 0 else np.nan
    data["Smoothed"] = data["Normalized"].rolling(window=10, center=True).mean()

    g_band = data[(data["Wavenumber"] >= G_FWHM_RANGE[0]) & (data["Wavenumber"] <= G_FWHM_RANGE[1])]
    fwhm_g = calculate_fwhm(g_band["Wavenumber"].to_numpy(), g_band["Corrected"].to_numpy())
    d_band = data[(data["Wavenumber"] >= D_BAND_RANGE[0]) & (data["Wavenumber"] <= D_BAND_RANGE[1])]
    g_peak_band = data[(data["Wavenumber"] >= G_BAND_RANGE[0]) & (data["Wavenumber"] <= G_BAND_RANGE[1])]
    d_max = float(d_band["Corrected"].max()) if not d_band.empty else float("nan")
    g_max = float(g_peak_band["Corrected"].max()) if not g_peak_band.empty else float("nan")
    d_area = integrate_band_area(data["Wavenumber"].to_numpy(), data["Corrected"].to_numpy(), D_BAND_RANGE)
    g_area = integrate_band_area(data["Wavenumber"].to_numpy(), data["Corrected"].to_numpy(), G_BAND_RANGE)
    metrics = {
        "FWHM_G": fwhm_g,
        "D_max_corrected": d_max,
        "G_max_corrected": g_max,
        "D/G_intensity_ratio": d_max / g_max if g_max != 0 else np.nan,
        "D_area": d_area,
        "G_area": g_area,
        "D/G_area_ratio": d_area / g_area if g_area != 0 else np.nan,
    }
    return data, metrics


def choose_raman_samples(sample_paths: dict[str, Path], saved: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    names = list(sample_paths)
    print("\nAvailable Raman samples:")
    print_numbered(names)
    previous = saved.get("indices") or []
    selected = ask("Enter sample indices to plot, comma separated, or x to reuse previous", ",".join(map(str, previous)))
    indices = previous if selected.lower() == "x" and previous else parse_indices(selected, len(names))
    saved["indices"] = indices
    return [names[idx] for idx in indices], saved


def run_interactive_raman() -> None:
    try:
        data_folder = prompt_for_path("raman_data", "Raman data folder")
        sample_paths = find_raman_files(data_folder)
        if not sample_paths:
            raise WorkflowError(f"No Raman .txt files found in {data_folder}")
        saved = load_json(settings_path(data_folder), default={})
        names, saved = choose_raman_samples(sample_paths, saved)
        plot_title = ask("Plot title", saved.get("plot_title") or "Raman")
        saved["plot_title"] = plot_title
        saved_samples = saved.setdefault("samples", {})

        fig, ax = plt.subplots(figsize=(6, 6.5))
        rows = []
        for idx, name in enumerate(names):
            prev = saved_samples.get(name, {})
            default_entry = f"{prev.get('nickname', name)}; {prev.get('offset', 0.0)}; {prev.get('shift', 0.0)}"
            entry = ask(f"Nickname; offset; horizontal shift for {name}", default_entry)
            nickname, offset, shift = _parse_raman_entry(entry, name)
            saved_samples[name] = {"nickname": nickname, "offset": offset, "shift": shift}
            data, metrics = analyze_raman_file(sample_paths[name], horizontal_shift=shift)
            ax.plot(
                data["Wavenumber"],
                data["Smoothed"] + offset,
                label=nickname,
                linewidth=2,
                color=COLORS[idx % len(COLORS)],
            )
            rows.append({"Sample name": nickname, **metrics})

        ax.text(1240, 1.25 + len(names) * 0.25, "D", fontsize=10, color="black")
        ax.text(1635, 1.25 + len(names) * 0.25, "G", fontsize=10, color="black")
        ax.text(2750, 1.05 + len(names) * 0.25, "D'", fontsize=10, color="black")
        ax.set_xlabel("Raman Shift (cm^-1)", fontsize=10)
        ax.set_ylabel("Normalized Intensity", fontsize=10)
        ax.tick_params(axis="y", labelcolor="white", length=0)
        ax.set_title(plot_title)
        ax.legend(fontsize=9)
        apply_plot_style(ax)
        fig.tight_layout()
        plt.show()

        results = pd.DataFrame(rows)
        print("\n=== Raman parameters ===")
        print(results.to_string(index=False))
        save_json(settings_path(data_folder), saved)
    except WorkflowError as exc:
        print(f"\nRaman workflow stopped: {exc}")


def _parse_raman_entry(entry: str, default_name: str) -> tuple[str, float, float]:
    parts = [part.strip() for part in entry.split(";")]
    nickname = parts[0] if len(parts) > 0 and parts[0] else default_name
    offset = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
    shift = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
    return nickname, offset, shift
