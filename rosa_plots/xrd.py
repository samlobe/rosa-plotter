"""XRD loading, fitting, plotting, and interactive prompts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    PROJECT_ROOT,
    WorkflowError,
    apply_plot_style,
    ask,
    ask_yes_no,
    natural_sort_key,
    parse_indices,
    print_numbered,
    prompt_for_path,
    load_json,
    save_json,
)


USE_TOP_N_SI = 2
MIN_ACCEPTED_SNR = 3.0
SI_SEARCH_HALF_WIDTH = 1.0
SI_REFINE_HALF_WIDTH = 0.35
REQUIRE_MIN_POINTS = 10
DEBUG_SI = True
SI_REFS = [("111", 28.440), ("220", 47.303), ("311", 56.120), ("400", 69.128)]
G002_WINDOW = (25.5, 27.5)
EDGE_FRAC = 0.12
LAMBDA_A = 1.5406
STEP_SIZE_DEG = 0.007
COLORS = ["#000000", "#7E1FD1", "#26C2FF", "#E16462", "#FFB000", "#6F728C", "#00A86A"]


@dataclass
class XrdChoice:
    sample_name: str
    nickname: str
    offset: float
    has_si: bool


def memory_path() -> Path:
    local_memory = PROJECT_ROOT / ".xrd_memory.json"
    old_memory = PROJECT_ROOT / "memory.json"
    return local_memory if local_memory.exists() else old_memory


def load_xrd_memory() -> dict[str, Any]:
    data = load_json(memory_path(), default={})
    if not isinstance(data, dict):
        return {}
    data.setdefault("indices", [])
    data.setdefault("sample_names", [])
    data.setdefault("plot_title", "")
    data.setdefault("include_shift_on_legend", False)
    data.setdefault("samples", {})
    return data


def save_xrd_memory(data: dict[str, Any]) -> None:
    save_json(PROJECT_ROOT / ".xrd_memory.json", data)


def read_xrd_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix in {".xls", ".xlsx"}:
            return pd.read_excel(path, header=None)
        return pd.read_csv(path, header=None)
    except ImportError as exc:
        if suffix == ".xls":
            raise WorkflowError(
                "This XRD file is .xls, which needs xlrd. "
                "Run: pip install -r requirements.txt"
            ) from exc
        raise


def load_xrd_samples(data_folder: Path) -> dict[str, pd.DataFrame]:
    if not data_folder.exists():
        raise WorkflowError(f"XRD folder does not exist: {data_folder}")

    samples: dict[str, pd.DataFrame] = {}
    files = sorted(data_folder.iterdir(), key=lambda p: natural_sort_key(p.name))
    for path in files:
        if path.suffix.lower() not in {".csv", ".xls", ".xlsx"}:
            continue
        try:
            raw = read_xrd_table(path)
        except WorkflowError:
            raise
        except Exception as exc:
            print(f"Could not read {path.name}: {exc}")
            continue

        for col in range(0, raw.shape[1], 2):
            if col + 1 >= raw.shape[1]:
                continue
            try:
                name = str(raw.iloc[0, col]).strip()
                angles = pd.to_numeric(raw.iloc[1:, col], errors="coerce")
                intensities = pd.to_numeric(raw.iloc[1:, col + 1], errors="coerce")
                mask = (~angles.isna()) & (~intensities.isna())
                if not name or not mask.any():
                    continue
                data = pd.DataFrame(
                    {
                        "Degree": angles[mask].astype(float),
                        "Intensity": intensities[mask].astype(float),
                    }
                )
                samples[name] = data.sort_values("Degree").reset_index(drop=True)
            except Exception as exc:
                print(f"Error in {path.name}, columns {col}-{col + 1}: {exc}")
    return samples


def _linear_background(x: np.ndarray, y: np.ndarray, edge_frac: float = 0.12):
    n = len(x)
    k = max(2, int(n * edge_frac))
    idx = np.r_[np.arange(k), np.arange(n - k, n)]
    A = np.vstack([x[idx], np.ones_like(x[idx])]).T
    m, b = np.linalg.lstsq(A, y[idx], rcond=None)[0]
    return m * x + b, m, b


def _voigt_profile(x: np.ndarray, amplitude: float, center: float, sigma: float, gamma: float):
    try:
        from scipy.special import wofz

        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    except Exception:
        fG = 2.354820045 * sigma
        fL = 2 * gamma
        f = (
            fL**5
            + 2.69269 * fL**4 * fG
            + 2.42843 * fL**3 * fG**2
            + 4.47163 * fL**2 * fG**3
            + 0.07842 * fL * fG**4
            + fG**5
        ) ** (1 / 5)
        eta = 1.36603 * (fL / f) - 0.47719 * (fL / f) ** 2 + 0.11116 * (fL / f) ** 3
        lor = (1 / np.pi) * (0.5 * f) / ((x - center) ** 2 + (0.5 * f) ** 2)
        gau = (np.sqrt(4 * np.log(2) / np.pi) / f) * np.exp(-4 * np.log(2) * ((x - center) / f) ** 2)
        return amplitude * (eta * lor + (1 - eta) * gau)


def _voigt_fwhm(sigma: float, gamma: float) -> float:
    fG = 2.354820045 * sigma
    fL = 2 * gamma
    return float(0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2))


def _smooth_local(x: np.ndarray, y: np.ndarray, width_deg: float = 0.08) -> np.ndarray:
    dx = np.median(np.diff(x)) if len(x) > 1 else None
    if dx is None or dx <= 0:
        return y
    win = max(5, int(round(width_deg / dx)) | 1)
    win = min(win, len(y) - (1 - len(y) % 2))
    if win < 5:
        return y
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(y, window_length=win, polyorder=min(3, win - 2), mode="interp")
    except Exception:
        pad = win // 2
        ypad = np.r_[y[pad:0:-1], y, y[-2 : -pad - 2 : -1]]
        return np.convolve(ypad, np.ones(win) / win, mode="valid")


def fit_peak_voigt_bg_coarse(
    x_all: np.ndarray,
    y_all: np.ndarray,
    ref_2theta: float,
    coarse_hw: float = 1.0,
    refine_hw: float = 0.35,
    min_pts: int = 15,
    edge_frac: float = 0.15,
) -> dict[str, Any] | None:
    from scipy.optimize import curve_fit

    m_coarse = (x_all >= ref_2theta - coarse_hw) & (x_all <= ref_2theta + coarse_hw)
    if m_coarse.sum() < 5:
        return None
    xc, yc = x_all[m_coarse].astype(float), y_all[m_coarse].astype(float)
    c0 = float(xc[int(np.nanargmax(_smooth_local(xc, yc)))])

    lo, hi = max(c0 - refine_hw, xc[0]), min(c0 + refine_hw, xc[-1])
    mr = (x_all >= lo) & (x_all <= hi)
    if mr.sum() < min_pts:
        return None
    x = x_all[mr].astype(float)
    y = y_all[mr].astype(float)

    k = max(2, int(len(x) * edge_frac))
    idx_edges = np.r_[np.arange(k), np.arange(len(x) - k, len(x))]
    m_init, b_init = np.linalg.lstsq(
        np.vstack([x[idx_edges], np.ones_like(x[idx_edges])]).T,
        y[idx_edges],
        rcond=None,
    )[0]
    ycorr = np.clip(y - (m_init * x + b_init), 0, None)
    amp0 = max(float(ycorr.max()), 1e-9)
    i_pk = int(np.argmax(ycorr))
    half = amp0 / 2.0
    left = np.where(ycorr[:i_pk] < half)[0]
    right = np.where(ycorr[i_pk:] < half)[0]
    fwhm0 = float(x[i_pk + right[0]] - x[left[-1]]) if len(left) and len(right) else 0.25

    def model(x_value, A, x0, sig, gam, m, b):
        return _voigt_profile(x_value, A, x0, sig, gam) + (m * x_value + b)

    p0 = [amp0, c0, max(fwhm0 / 2.3548, 0.01), max(0.5 * (fwhm0 / 2.0), 0.01), m_init, b_init]
    bounds = ([0.0, lo, 1e-4, 1e-4, -np.inf, -np.inf], [np.inf, hi, 5.0, 5.0, np.inf, np.inf])
    try:
        popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=30000)
    except Exception:
        if DEBUG_SI:
            print(f"[Si] Fit failed near {ref_2theta:.3f} deg")
        return None
    A, center, sig, gam, m_bg, b_bg = popt
    perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
    res_edges = y[idx_edges] - (m_bg * x[idx_edges] + b_bg)
    snr = float(A / max(np.std(res_edges), 1e-6))
    return {
        "center_2theta_deg": float(center),
        "center_se_deg": float(perr[1]),
        "fwhm_deg": _voigt_fwhm(sig, gam),
        "amplitude": float(A),
        "snr": snr,
        "xw": x,
        "yw": y,
        "yfit": model(x, *popt),
    }


def calibrate_si_shift_robust(
    df: pd.DataFrame,
    refs=SI_REFS,
    use_top_n: int = USE_TOP_N_SI,
    min_snr: float = MIN_ACCEPTED_SNR,
) -> tuple[float, float, list[dict[str, Any]]]:
    x = df["Degree"].to_numpy()
    y = df["Intensity"].to_numpy()
    found = []
    for hkl, ref in refs:
        res = fit_peak_voigt_bg_coarse(
            x,
            y,
            ref_2theta=ref,
            coarse_hw=SI_SEARCH_HALF_WIDTH,
            refine_hw=SI_REFINE_HALF_WIDTH,
            min_pts=REQUIRE_MIN_POINTS,
        )
        if res is None or res["snr"] < min_snr:
            continue
        res.update({"hkl": hkl, "ref_2theta": ref, "delta": float(ref - res["center_2theta_deg"])})
        found.append(res)
    if not found:
        return 0.0, 0.0, []
    found.sort(key=lambda row: row["amplitude"], reverse=True)
    used = found[: max(1, min(use_top_n, len(found)))]
    w = np.array([1.0 / max(row["center_se_deg"], 1e-6) ** 2 for row in used], dtype=float)
    d = np.array([row["delta"] for row in used], dtype=float)
    shift = float((w * d).sum() / w.sum())
    return shift, float(np.sqrt(1.0 / w.sum())), used


def fit_graphite_002(x: np.ndarray, y: np.ndarray, window=G002_WINDOW) -> dict[str, Any] | None:
    from scipy.optimize import curve_fit

    wmask = (x >= window[0]) & (x <= window[1])
    if not np.any(wmask):
        return None
    xw = x[wmask].astype(float)
    yw = y[wmask].astype(float)
    bg, m_bg, b_bg = _linear_background(xw, yw, edge_frac=EDGE_FRAC)
    ycorr = np.clip(yw - bg, 0, None)
    i_max = int(np.argmax(ycorr))
    amp0 = max(ycorr[i_max], 1e-6)
    c0 = xw[i_max]
    half = amp0 / 2
    left = np.where(ycorr[:i_max] < half)[0]
    right = np.where(ycorr[i_max:] < half)[0]
    fwhm0 = xw[i_max + right[0]] - xw[left[-1]] if len(left) and len(right) else 0.25
    sigma0 = max(fwhm0 / 2.3548, 0.01)
    gamma0 = max(0.5 * (fwhm0 / 2), 0.01)
    try:
        popt, pcov = curve_fit(
            _voigt_profile,
            xw,
            ycorr,
            p0=[amp0, c0, sigma0, gamma0],
            bounds=([0.0, window[0], 1e-4, 1e-4], [np.inf, window[1], 5.0, 5.0]),
            maxfev=20000,
        )
        amp, center, sigma, gamma = popt
        center_se = float(np.sqrt(np.maximum(np.diag(pcov), 0))[1])
    except Exception:
        amp, center, sigma, gamma = amp0, c0, sigma0, gamma0
        snr = max((amp if amp > 0 else 0.0) / np.sqrt(max(yw.max(), 1.0)), 1.0)
        center_se = float(np.sqrt(max(_voigt_fwhm(sigma, gamma), STEP_SIZE_DEG) * STEP_SIZE_DEG) / snr)
    yfit = _voigt_profile(xw, amp, center, sigma, gamma)
    theta_rad = np.deg2rad(center / 2.0)
    return {
        "center_2theta_deg": float(center),
        "center_se_deg": float(center_se),
        "fwhm_deg": _voigt_fwhm(sigma, gamma),
        "amplitude": float(amp),
        "area_int": float(np.trapz(yfit, xw)),
        "d_spacing_A": float(LAMBDA_A / (2.0 * np.sin(theta_rad))),
        "bg_slope": float(m_bg),
        "bg_intercept": float(b_bg),
        "xw": xw,
        "yw": yw,
        "ybg": bg,
        "yfit": yfit + bg,
    }


def choose_xrd_samples(samples: dict[str, pd.DataFrame], memory: dict[str, Any]) -> list[XrdChoice]:
    sample_keys = sorted(samples, key=natural_sort_key)
    print("\nAvailable XRD samples:")
    print_numbered(sample_keys)

    previous = memory.get("indices") or []
    selected = ask("Enter sample indices to plot, comma separated, or x to reuse previous", ",".join(map(str, previous)))
    indices = previous if selected.strip().lower() == "x" and previous else parse_indices(selected, len(sample_keys))

    choices = []
    saved_samples = memory.setdefault("samples", {})
    for idx in indices:
        sample_name = sample_keys[idx]
        prev = saved_samples.get(sample_name, {})
        default_entry = f"{prev.get('nickname', sample_name)}, {prev.get('offset', 0.0)}"
        entry = ask(f"Nickname and vertical offset for {sample_name}", default_entry)
        nickname_part, _, offset_part = entry.partition(",")
        nickname = nickname_part.strip() or sample_name
        offset = float(offset_part.strip() or 0.0)
        has_si_default = bool(prev.get("has_si", False))
        has_si = ask_yes_no(f"Does {nickname} include Si internal standard?", has_si_default)
        choices.append(XrdChoice(sample_name, nickname, offset, has_si))
        saved_samples[sample_name] = {"nickname": nickname, "offset": offset, "has_si": has_si}

    memory["indices"] = indices
    memory["sample_names"] = [choice.sample_name for choice in choices]
    return choices


def plot_xrd(
    samples: dict[str, pd.DataFrame],
    choices: list[XrdChoice],
    plot_title: str,
    include_shift_on_legend: bool,
    output_csv: Path,
) -> pd.DataFrame:
    plt.figure(figsize=(10, 6))
    rows = []
    for idx, choice in enumerate(choices):
        df = samples[choice.sample_name].copy()
        shift, shift_se, used = (0.0, 0.0, [])
        if choice.has_si:
            shift, shift_se, used = calibrate_si_shift_robust(df)
            if used:
                df["Degree"] = df["Degree"] + shift
                print(f"Applied XRD shift {shift:+.4f} deg +/- {shift_se:.4f} to {choice.nickname}")
            else:
                print(f"No usable Si peaks found for {choice.nickname}; no shift applied.")

        df["Intensity"] = df["Intensity"] / max(df["Intensity"].max(), 1e-12) + choice.offset
        label = choice.nickname
        if include_shift_on_legend and abs(shift) > 0:
            label = f"{label}\n(shift={shift:+.3f} deg)"
        color = COLORS[idx % len(COLORS)]
        plt.plot(df["Degree"], df["Intensity"], label=label, linewidth=1.8, color=color)

        fitres = fit_graphite_002(df["Degree"].to_numpy(), (df["Intensity"] - choice.offset).to_numpy())
        row = {
            "Sample name": label,
            "002 2theta (deg)": np.nan,
            "002 2theta SE (deg)": np.nan,
            "FWHM (deg)": np.nan,
            "Area (arb deg)": np.nan,
            "d002 (A)": np.nan,
            "d002 SE (A)": np.nan,
            "BG slope (int/deg)": np.nan,
            "BG intercept (int)": np.nan,
            "Si shift applied (deg)": shift,
            "Si shift SE (deg)": shift_se,
        }
        if fitres is not None:
            plt.plot(fitres["xw"], fitres["yfit"] + choice.offset, linestyle="--", linewidth=2.2, color=color)
            center = fitres["center_2theta_deg"]
            center_se_total = float(np.sqrt(fitres["center_se_deg"] ** 2 + shift_se**2))
            theta_rad = np.deg2rad(center / 2.0)
            d_se = float(fitres["d_spacing_A"] * abs(1.0 / np.tan(theta_rad)) * (np.deg2rad(center_se_total) / 2.0))
            row.update(
                {
                    "002 2theta (deg)": center,
                    "002 2theta SE (deg)": center_se_total,
                    "FWHM (deg)": fitres["fwhm_deg"],
                    "Area (arb deg)": fitres["area_int"],
                    "d002 (A)": fitres["d_spacing_A"],
                    "d002 SE (A)": d_se,
                    "BG slope (int/deg)": fitres["bg_slope"],
                    "BG intercept (int)": fitres["bg_intercept"],
                }
            )
        rows.append(row)

    plt.xlabel("2theta (degrees)", fontsize=14)
    plt.ylabel("Normalized Intensity + Offset", fontsize=13)
    plt.title(plot_title, fontsize=16)
    plt.yticks([], [])
    apply_plot_style(plt.gca())
    plt.xlim([15, 70])
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

    results_df = pd.DataFrame(rows)
    print("\n=== Voigt fit results for graphite (002) ===")
    print(results_df.to_string(index=False))
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"\nSaved results to: {output_csv}")
    return results_df


def run_interactive_xrd() -> None:
    try:
        data_folder = prompt_for_path("xrd_data", "XRD data folder")
        samples = load_xrd_samples(data_folder)
        if not samples:
            raise WorkflowError(f"No XRD samples found in {data_folder}")
        memory = load_xrd_memory()
        choices = choose_xrd_samples(samples, memory)
        plot_title = ask("Plot title", memory.get("plot_title") or "XRD")
        include_shift = ask_yes_no(
            "Include Si shift on legend?",
            bool(memory.get("include_shift_on_legend", False)),
        )
        memory["plot_title"] = plot_title
        memory["include_shift_on_legend"] = include_shift
        save_xrd_memory(memory)
        plot_xrd(samples, choices, plot_title, include_shift, PROJECT_ROOT / "outputs" / "voigt_002_results.csv")
    except WorkflowError as exc:
        print(f"\nXRD workflow stopped: {exc}")
