#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 20:40:55 2025

@author: rosadrianazelaya
"""


# XRD with voigt fit and error bar calcs

import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "Avenir Next",
    "font.weight": "bold",
    "axes.labelweight": "bold"
})
import json

# === CONFIG ===
data_folder = '/Users/rosadrianazelaya/Library/CloudStorage/Box-Box/Halomaterials/Rosa/XRD/XRD_data'
nickname_file = 'sample_nicknames.csv'   # CSV with columns: sample_name,nickname

# ---- Si calibration settings (Cu Kα1 references) ----
# Use the strongest 2 peaks; set USE_TOP_N_SI = 3 if your standard is strong.
USE_TOP_N_SI = 2          # choose 2 (default) or 3
MIN_ACCEPTED_SNR = 5.0    # require decent SNR for a Si peak to be used

# Si reference 2θ (deg, Cu Kα1 = 1.5406 Å) and search windows
CAL_SI_PEAKS = [
    {"hkl": "111", "ref_2theta": 28.440, "window": (27.5, 28.9)},#28.20, 28.70)},
    {"hkl": "220", "ref_2theta": 47.303, "window": (46.7, 47.8)},#47.05, 47.55)},
    {"hkl": "311", "ref_2theta": 56.120, "window": (55.85, 56.45)},
    {"hkl": "400", "ref_2theta": 69.128, "window": (68.88, 69.35)},
]

# Graphite (002) fit settings (Cu Kα1 ~26.37°). Widen if needed.
G002_WINDOW = (25.5, 27.5)
EDGE_FRAC = 0.12

# Instrument wavelength (Cu Kα1), for d-spacing
LAMBDA_A = 1.5406

# Step size of your scans in degrees 2θ (used in fallback SE estimate)
STEP_SIZE_DEG = 0.007

SAVE_RESULTS_CSV = 'voigt_002_results.csv'

# --- Robust Si search settings ---
SI_SEARCH_HALF_WIDTH = 1 #1  # coarse search half-width (deg) around each Si ref
SI_REFINE_HALF_WIDTH = .35 #.35  # refined fit half-width once a local max is found
MIN_ACCEPTED_SNR = 3.0       # was 5.0; relax if your Si is weak
REQUIRE_MIN_POINTS = 10      # require at least this many points in the fit window
DEBUG_SI = True             # set True to print why a peak was rejected
# Si references (Cu Kα1)
SI_REFS = [("111", 28.440), ("220", 47.303), ("311", 56.120), ("400", 69.128)]


# ----------------- Helper: linear background -----------------
def _linear_background(x, y, edge_frac=0.12):
    n = len(x)
    k = max(2, int(n * edge_frac))
    idx = np.r_[np.arange(k), np.arange(n - k, n)]
    A = np.vstack([x[idx], np.ones_like(x[idx])]).T
    coeff, _, _, _ = np.linalg.lstsq(A, y[idx], rcond=None)
    m, b = coeff[0], coeff[1]
    return m*x + b, m, b

# ----------------- Voigt model -----------------
def _voigt_profile(x, amplitude, center, sigma, gamma):
    try:
        from scipy.special import wofz
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))
    except Exception:
        # pseudo-Voigt fallback
        fG = 2.354820045 * sigma
        fL = 2 * gamma
        f = (fL**5 + 2.69269*fL**4*fG + 2.42843*fL**3*fG**2 +
             4.47163*fL**2*fG**3 + 0.07842*fL*fG**4 + fG**5) ** (1/5)
        eta = 1.36603*(fL/f) - 0.47719*(fL/f)**2 + 0.11116*(fL/f)**3
        lor = (1/np.pi) * (0.5*f) / ((x - center)**2 + (0.5*f)**2)
        gau = (np.sqrt(4*np.log(2)/np.pi) / f) * np.exp(-4*np.log(2)*((x-center)/f)**2)
        return amplitude * (eta*lor + (1-eta)*gau)

def _voigt_fwhm(sigma, gamma):
    fG = 2.354820045 * sigma
    fL = 2 * gamma
    return 0.5346*fL + np.sqrt(0.2166*fL**2 + fG**2)

# ----------------- Generic Voigt fit for a single peak -----------------
def fit_peak_voigt(x_all, y_all, window, edge_frac=EDGE_FRAC):
    """Fit a single peak in 'window' with Voigt + linear background.
    Returns dict with center, center_se, fwhm, amp, area, bg, yfit, SNR, etc.
    """
    from scipy.optimize import curve_fit

    mask = (x_all >= window[0]) & (x_all <= window[1])
    if not np.any(mask):
        return None
    x = x_all[mask].astype(float)
    y = y_all[mask].astype(float)

    bg, m_bg, b_bg = _linear_background(x, y, edge_frac=edge_frac)
    ycorr = y - bg
    ycorr = np.clip(ycorr, 0, None)

    # Initial guesses
    i_max = int(np.argmax(ycorr))
    amp0 = max(ycorr[i_max], 1e-9)
    c0 = x[i_max]
    half = amp0 / 2
    left = np.where(ycorr[:i_max] < half)[0]
    right = np.where(ycorr[i_max:] < half)[0]
    if len(left) > 0 and len(right) > 0:
        fwhm0 = (x[i_max + right[0]] - x[left[-1]])
    else:
        fwhm0 = 0.25
    sigma0 = max(fwhm0 / 2.3548, 0.01)
    gamma0 = max(0.5*(fwhm0/2), 0.01)

    # Fit
    try:
        popt, pcov = curve_fit(
            _voigt_profile, x, ycorr,
            p0=[amp0, c0, sigma0, gamma0],
            bounds=([0.0, window[0], 1e-4, 1e-4],
                    [np.inf, window[1], 5.0, 5.0]),
            maxfev=20000
        )
        amp, center, sigma, gamma = popt
        perr = np.sqrt(np.maximum(np.diag(pcov), 0))
        center_se = float(perr[1])
    except Exception:
        # crude fallback: keep guesses; large SE
        amp, center, sigma, gamma = amp0, c0, sigma0, gamma0
        center_se = 0.5 * (window[1] - window[0])

    fwhm = _voigt_fwhm(sigma, gamma)
    yfit = _voigt_profile(x, amp, center, sigma, gamma)
    area = float(np.trapz(yfit, x))

    # SNR (height over Poisson at peak position; robust proxy)
    noise = np.sqrt(max(y.max(), 1.0))
    snr = (amp if amp > 0 else 0.0) / (noise if noise > 0 else 1.0)

    return {
        "center_2theta_deg": float(center),
        "center_se_deg": float(center_se),
        "fwhm_deg": float(fwhm),
        "amplitude": float(amp),
        "area_int": area,
        "bg_slope": float(m_bg),
        "bg_intercept": float(b_bg),
        "xw": x,
        "yw": y,
        "ybg": bg,
        "yfit": yfit + bg,
        "snr": float(snr),
    }

def _smooth_local(x, y, width_deg=0.08):
    # Use the globally imported numpy as np
    dx = np.median(np.diff(x)) if len(x) > 1 else None
    if dx is None or dx <= 0:
        return y
    win = max(5, int(round(width_deg / dx)) | 1)   # odd window
    win = min(win, len(y) - (1 - len(y) % 2))
    if win < 5:
        return y
    try:
        from scipy.signal import savgol_filter
        poly = min(3, win - 2)
        return savgol_filter(y, window_length=win, polyorder=poly, mode="interp")
    except Exception:
        # Moving-average fallback using the global np
        pad = win // 2
        ypad = np.r_[y[pad:0:-1], y, y[-2:-pad-2:-1]]
        ker = np.ones(win) / win
        ys = np.convolve(ypad, ker, mode='valid')
        return ys


def fit_peak_voigt_bg_coarse(x_all, y_all, ref_2theta,
                             coarse_hw=1.0, refine_hw=0.35,
                             min_pts=15, edge_frac=0.15):
    """
    2-stage: coarse local-max search near ref_2theta, then Voigt + linear background fit.
    Returns dict with center, center_se, fwhm, amplitude, snr, and fit arrays; or None.
    """
    import numpy as np
    from scipy.optimize import curve_fit

    # --- coarse window around the reference ---
    m_coarse = (x_all >= ref_2theta - coarse_hw) & (x_all <= ref_2theta + coarse_hw)
    if m_coarse.sum() < 5:
        return None
    xc, yc = x_all[m_coarse].astype(float), y_all[m_coarse].astype(float)
    yc_s = _smooth_local(xc, yc, width_deg=0.08)
    i_max = int(np.nanargmax(yc_s))
    c0 = float(xc[i_max])

    # --- refine window around local max ---
    lo, hi = max(c0 - refine_hw, xc[0]), min(c0 + refine_hw, xc[-1])
    mr = (x_all >= lo) & (x_all <= hi)
    if mr.sum() < min_pts:
        return None
    x = x_all[mr].astype(float)
    y = y_all[mr].astype(float)

    # initial background from edges
    k = max(2, int(len(x) * edge_frac))
    idx_edges = np.r_[np.arange(k), np.arange(len(x) - k, len(x))]
    A_bg = np.vstack([x[idx_edges], np.ones_like(x[idx_edges])]).T
    m_init, b_init = np.linalg.lstsq(A_bg, y[idx_edges], rcond=None)[0]

    ycorr = y - (m_init * x + b_init)
    ycorr = np.clip(ycorr, 0, None)
    amp0 = max(float(ycorr.max()), 1e-9)

    # crude FWHM guess
    half = amp0 / 2.0
    i_pk = int(np.argmax(ycorr))
    left = np.where(ycorr[:i_pk] < half)[0]
    right = np.where(ycorr[i_pk:] < half)[0]
    if len(left) and len(right):
        fwhm0 = float(x[i_pk + right[0]] - x[left[-1]])
    else:
        fwhm0 = 0.25
    sigma0 = max(fwhm0 / 2.3548, 0.01)
    gamma0 = max(0.5 * (fwhm0 / 2.0), 0.01)

    # model: Voigt + linear background
    def _voigt_profile(x, amplitude, center, sigma, gamma):
        try:
            from scipy.special import wofz
            z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
            return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))
        except Exception:
            fG = 2.354820045 * sigma
            fL = 2 * gamma
            f = (fL**5 + 2.69269*fL**4*fG + 2.42843*fL**3*fG**2 +
                 4.47163*fL**2*fG**3 + 0.07842*fL*fG**4 + fG**5) ** (1/5)
            eta = 1.36603*(fL/f) - 0.47719*(fL/f)**2 + 0.11116*(fL/f)**3
            lor = (1/np.pi) * (0.5*f) / ((x - center)**2 + (0.5*f)**2)
            gau = (np.sqrt(4*np.log(2)/np.pi) / f) * np.exp(-4*np.log(2)*((x-center)/f)**2)
            return amplitude * (eta*lor + (1-eta)*gau)

    def model(x, A, x0, sig, gam, m, b):
        return _voigt_profile(x, A, x0, sig, gam) + (m * x + b)

    p0 = [amp0, c0, sigma0, gamma0, m_init, b_init]
    bounds = ([0.0, lo, 1e-4, 1e-4, -np.inf, -np.inf],
              [np.inf, hi,    5.0,  5.0,  np.inf,  np.inf])

    try:
        popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=30000)
        A, center, sig, gam, m_bg, b_bg = popt
        perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        center_se = float(perr[1])
    except Exception:
        if DEBUG_SI: print(f"[Si] Fit failed near {ref_2theta:.3f}°")
        return None

    # SNR from edge residuals (background noise estimate)
    yfit = model(x, *popt)
    if len(idx_edges) >= 4:
        res_edges = y[idx_edges] - (m_bg * x[idx_edges] + b_bg)
        noise = np.std(res_edges)
    else:
        noise = np.sqrt(max(y.max(), 1.0))
    snr = float(A / max(noise, 1e-6))

    # FWHM
    fG = 2.354820045 * sig
    fL = 2.0 * gam
    fwhm = 0.5346 * fL + np.sqrt(0.2166 * fL * fL + fG * fG)

    return {
        "center_2theta_deg": float(center),
        "center_se_deg": float(center_se),
        "fwhm_deg": float(fwhm),
        "amplitude": float(A),
        "snr": snr,
        "xw": x, "yw": y, "yfit": yfit
    }

def calibrate_si_shift_robust(df,
                              refs=SI_REFS,
                              use_top_n=USE_TOP_N_SI,
                              min_snr=MIN_ACCEPTED_SNR):
    """Return (shift_to_add_deg, shift_se_deg, used_details). shift = ref - meas."""
    x = df['Degree'].to_numpy()
    y = df['Intensity'].to_numpy()

    found = []
    for hkl, ref in refs:
        res = fit_peak_voigt_bg_coarse(
            x, y, ref_2theta=ref,
            coarse_hw=SI_SEARCH_HALF_WIDTH,
            refine_hw=SI_REFINE_HALF_WIDTH,
            min_pts=REQUIRE_MIN_POINTS
        )
        if res is None:
            if DEBUG_SI: print(f"[Si] {hkl} @ {ref:.3f}°: not found (coarse/refine window)")
            continue
        if res["snr"] < min_snr:
            if DEBUG_SI: print(f"[Si] {hkl}: SNR {res['snr']:.1f} < {min_snr}")
            continue
        delta = ref - res["center_2theta_deg"]
        res.update({"hkl": hkl, "ref_2theta": ref, "delta": float(delta)})
        found.append(res)

    if not found:
        return 0.0, 0.0, []

    # take strongest peaks by amplitude
    found.sort(key=lambda r: r["amplitude"], reverse=True)
    used = found[:max(1, min(use_top_n, len(found)))]

    # weighted mean by 1/SE^2 of centers
    w = np.array([1.0 / max(u["center_se_deg"], 1e-6)**2 for u in used], dtype=float)
    d = np.array([u["delta"] for u in used], dtype=float)
    W = float(w.sum())
    shift = float((w * d).sum() / W)
    shift_se = float(np.sqrt(1.0 / W))

    return shift, shift_se, used


# ----------------- Graphite (002) Voigt fit with center SE -----------------
def fit_graphite_002(x, y, window=G002_WINDOW):
    from scipy.optimize import curve_fit

    wmask = (x >= window[0]) & (x <= window[1])
    if not np.any(wmask):
        return None
    xw = x[wmask].astype(float)
    yw = y[wmask].astype(float)

    bg, m_bg, b_bg = _linear_background(xw, yw, edge_frac=EDGE_FRAC)
    ycorr = yw - bg
    ycorr = np.clip(ycorr, a_min=0, a_max=None)

    i_max = int(np.argmax(ycorr))
    amp0 = max(ycorr[i_max], 1e-6)
    c0 = xw[i_max]
    half = amp0 / 2
    left = np.where(ycorr[:i_max] < half)[0]
    right = np.where(ycorr[i_max:] < half)[0]
    if len(left) > 0 and len(right) > 0:
        fwhm0 = (xw[i_max + right[0]] - xw[left[-1]])
    else:
        fwhm0 = 0.25
    sigma0 = max(fwhm0 / 2.3548, 0.01)
    gamma0 = max(0.5*(fwhm0/2), 0.01)

    try:
        popt, pcov = curve_fit(
            _voigt_profile, xw, ycorr,
            p0=[amp0, c0, sigma0, gamma0],
            bounds=([0.0, window[0], 1e-4, 1e-4],
                    [np.inf, window[1], 5.0, 5.0]),
            maxfev=20000
        )
        amp, center, sigma, gamma = popt
        perr = np.sqrt(np.maximum(np.diag(pcov), 0))
        center_se = float(perr[1])
    except Exception:
        amp, center, sigma, gamma = amp0, c0, sigma0, gamma0
        # Fallback SE using FWHM & step size & rough SNR
        fwhm_guess = _voigt_fwhm(sigma, gamma)
        snr = (amp if amp > 0 else 0.0) / (np.sqrt(max(yw.max(), 1.0)))
        snr = max(snr, 1.0)
        center_se = float(np.sqrt(max(fwhm_guess, STEP_SIZE_DEG) * STEP_SIZE_DEG) / snr)

    fwhm = _voigt_fwhm(sigma, gamma)
    yfit = _voigt_profile(xw, amp, center, sigma, gamma)
    area = np.trapz(yfit, xw)

    # d-spacing (n=1)
    theta_rad = np.deg2rad(center / 2.0)
    d_A = LAMBDA_A / (2.0 * np.sin(theta_rad))

    return {
        'center_2theta_deg': float(center),
        'center_se_deg': float(center_se),
        'fwhm_deg': float(fwhm),
        'amplitude': float(amp),
        'area_int': float(area),
        'd_spacing_A': float(d_A),
        'bg_slope': float(m_bg),
        'bg_intercept': float(b_bg),
        'xw': xw,
        'yw': yw,
        'ybg': bg,
        'yfit': yfit + bg
    }

# === LOAD nicknames, parse files (unchanged from your script) ===
nicknames = {}
if os.path.exists(nickname_file):
    try:
        df_nick = pd.read_csv(nickname_file)
        nicknames = dict(zip(df_nick['sample_name'], df_nick['nickname']))
    except Exception:
        pass

# === PARSE ALL FILES ===
samples = {}
for fname in os.listdir(data_folder):
    if fname.endswith(('.csv', '.xls', '.xlsx')):
        fpath = os.path.join(data_folder, fname)
        try:
            if fname.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(fpath, header=None)
            else:
                df = pd.read_csv(fpath, header=None)

            for col in range(0, df.shape[1], 2):
                try:
                    name = df.iloc[0, col]
                    angles = pd.to_numeric(df.iloc[1:, col], errors='coerce')
                    intensities = pd.to_numeric(df.iloc[1:, col + 1], errors='coerce')
                    mask = (~angles.isna()) & (~intensities.isna())
                    data = pd.DataFrame({'Degree': angles[mask].astype(float),
                                         'Intensity': intensities[mask].astype(float)})
                    data = data.sort_values('Degree').reset_index(drop=True)
                    samples[name] = data
                except Exception as e:
                    print(f"Error in file {fname}, columns {col}-{col+1}: {e}")
        except Exception as e:
            print(f"Could not read file {fname}: {e}")

# === DISPLAY SAMPLE OPTIONS ===
print("\n📄 Available Samples:")
sample_keys = list(samples.keys())
for i, name in enumerate(sample_keys):
    nickname = nicknames.get(name, "???")
    print(f"{i}: {name} → {nickname}")

def ask(question):
    print(f"\n>>> {question}", flush=True)
    return input("> ")

# === USER SELECTION ===
selected = ask("Enter sample indices to plot, comma separated. Example: 67 or 2,5")
indices = [int(i.strip()) for i in selected.split(',') if i.strip() != ""]

# === CUSTOMIZE PLOT TITLE ===
plot_title = ask("Enter plot title")
include_shift_on_legend = ask("Include shift on legend? (y/n)").strip().lower()

# === COLLECT NICKNAMES, OFFSETS, AND OPTIONAL SI-BASED SHIFTS ===
custom_nicknames = {}
offsets = {}
shifts_applied = {}

# === APPLY Si calibration (new) & collect offsets ===
custom_nicknames = {}
offsets = {}
shifts_applied = {}
shifts_se = {}

for i in indices:
    key = sample_keys[i]
    current_nickname = nicknames.get(key, "")
    entry = ask(f"Enter nickname and vertical offset for '{key}' [{current_nickname}, 0.0]").strip()
    if entry:
        parts = entry.split(',')
        nickname = parts[0].strip() if parts[0] else key
        v_offset = float(parts[1]) if len(parts) > 1 and parts[1].strip() else 0.0
    else:
        nickname = current_nickname or key
        v_offset = 0.0
    custom_nicknames[key] = nickname
    offsets[key] = v_offset

    has_si = ask(f"Does '{nickname}' include Si internal standard? (y/n)").strip().lower()
    df = samples[key].copy()
    if has_si == 'y':
        shift, shift_se, used = calibrate_si_shift_robust(df)
        if used:
            df['Degree'] = df['Degree'] + shift
            samples[key] = df
            shifts_applied[key] = shift
            shifts_se[key] = shift_se
            used_txt = ", ".join([f"Si({r['hkl']}): meas={r['center_2theta_deg']:.4f}°, ref={r['ref_2theta']:.3f}°"
                                  for r in used])
            print(f"  → Applied weighted shift {shift:+.4f}° ± {shift_se:.4f}° using {len(used)} peak(s): {used_txt}")
        else:
            shifts_applied[key] = 0.0
            shifts_se[key] = 0.0
            print("  ! Si peaks not usable (weak or missing); no shift applied.")
    else:
        shifts_applied[key] = 0.0
        shifts_se[key] = 0.0

# === OPTION TO SAVE UPDATED NICKNAMES ===
save = ask("Save these nicknames to file for future use? (y/n)").lower().strip()
if save == 'y':
    all_nicknames = {**nicknames, **custom_nicknames}
    df_save = pd.DataFrame(all_nicknames.items(), columns=['sample_name', 'nickname'])
    df_save.to_csv(nickname_file, index=False)
    print(f"✅ Nicknames saved to {nickname_file}")

# === VOIGT FIT + PLOTTING ===
plt.figure(figsize=(10, 6))

colors = ['#000000', '#7E1FD1','#26C2FF',  '#E16462', '#FFB000','#6F728C', '#00A86A' ]
rows = []

for idx, i in enumerate(indices):
    key = sample_keys[i]
    df = samples[key].copy()

    # Normalize for plotting + stack
    df['Intensity'] = df['Intensity'] / max(df['Intensity'].max(), 1e-12) + offsets[key]

    label = custom_nicknames[key]
    if abs(shifts_applied.get(key, 0.0)) > 0:
        if include_shift_on_legend == 'y':
            label = f"{label} \n(Δ2θ={shifts_applied[key]:+.3f}°)"
        else:
            label = f"{label}"

    plt.plot(df['Degree'], df['Intensity'], label=label, linewidth=1.8, color=colors[idx % len(colors)])

    # Fit graphite (002) on de-offset, normalized trace
    fitres = fit_graphite_002(df['Degree'].to_numpy(),
                              (df['Intensity'] - offsets[key]).to_numpy(),
                              window=G002_WINDOW)
    if fitres is not None:
        xw = fitres['xw']
        yfit_plot = fitres['yfit'] + offsets[key]
        plt.plot(xw, yfit_plot, linestyle='--', linewidth=2.2, color=colors[idx % len(colors)])

        center = fitres['center_2theta_deg']
        center_se_stat = fitres['center_se_deg']
        # combine with Si shift uncertainty in quadrature
        si_se = shifts_se.get(key, 0.0)
        center_se_total = float(np.sqrt(center_se_stat**2 + si_se**2))

        # d-spacing and uncertainty
        theta_rad = np.deg2rad(center / 2.0)
        d_A = fitres['d_spacing_A']
        sigma_theta = np.deg2rad(center_se_total) / 2.0  # rad
        d_se = float(d_A * abs(1.0/np.tan(theta_rad)) * sigma_theta)

        rows.append({
            'Sample name': label,
            '002 2θ (deg)': center,
            '002 2θ SE (deg)': center_se_total,
            'FWHM (deg)': fitres['fwhm_deg'],
            'Area (arb·deg)': fitres['area_int'],
            'd002 (Å)': d_A,
            'd002 SE (Å)': d_se,
            'BG slope (int/deg)': fitres['bg_slope'],
            'BG intercept (int)': fitres['bg_intercept'],
            'Si shift applied (deg)': shifts_applied.get(key, 0.0),
            'Si shift SE (deg)': si_se
        })
    else:
        rows.append({
            'Sample name': label,
            '002 2θ (deg)': np.nan,
            '002 2θ SE (deg)': np.nan,
            'FWHM (deg)': np.nan,
            'Area (arb·deg)': np.nan,
            'd002 (Å)': np.nan,
            'd002 SE (Å)': np.nan,
            'BG slope (int/deg)': np.nan,
            'BG intercept (int)': np.nan,
            'Si shift applied (deg)': shifts_applied.get(key, 0.0),
            'Si shift SE (deg)': shifts_se.get(key, 0.0)
        })

plt.xlabel('2θ (degrees)', fontsize=14)
plt.ylabel('Normalized Intensity + Offset', fontsize=13)
plt.title(plot_title, fontsize=16)
plt.xticks(fontsize=13)
plt.yticks([], [])
for spp in plt.gca().spines.values():
    spp.set_linewidth(2)
plt.tick_params(width=2)
plt.xlim([15, 70])
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# === RESULTS TABLE ===
results_df = pd.DataFrame(rows,
    columns=['Sample name','002 2θ (deg)','002 2θ SE (deg)','FWHM (deg)',
             'Area (arb·deg)','d002 (Å)','d002 SE (Å)',
             'BG slope (int/deg)','BG intercept (int)',
             'Si shift applied (deg)','Si shift SE (deg)'])
print("\n=== Voigt fit results for graphite (002) ===")
print(results_df.to_string(index=False))

if SAVE_RESULTS_CSV:
    try:
        results_df.to_csv(SAVE_RESULTS_CSV, index=False)
        print(f"\n✅ Saved results to: {SAVE_RESULTS_CSV}")
    except Exception as e:
        print(f"\n⚠️ Could not save results CSV: {e}")

##### ^^ XRD w Voigt Fit and Si peak ^^ #####


#%% Raman peak area

#######################################################
#################### R A M A N ########################
#######################################################

import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d
import scipy.sparse as sp
import scipy.sparse.linalg as spla

mpl.rcParams.update({
    "font.family": "Avenir Next",
    "font.weight": "bold",
    "axes.labelweight": "bold"
})

# === SETTINGS ===
data_folder = '/Users/rosadrianazelaya/Library/CloudStorage/Box-Box/Halomaterials/Rosa/Raman/For Plotting'
cutoff_wavenumber = 1100  # cm⁻¹
settings_file = os.path.join(data_folder, 'plot_settings.json')

# Band ranges for calculations
D_BAND_RANGE = (1300, 1400)
G_BAND_RANGE = (1450, 1650)
G_FWHM_RANGE = (1500, 1650)

def baseline_als(y, lam=1e6, p=0.001, n_iter=10):
    """
    Asymmetric least-squares baseline correction.
    """
    L = len(y)
    D = sp.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    w = np.ones(L)
    for _ in range(n_iter):
        W = sp.diags(w, 0)
        Z = W + lam * D.T @ D
        baseline = spla.spsolve(Z, w * y)
        w = p * (y > baseline) + (1 - p) * (y < baseline)
    return baseline

def calculate_fwhm(wavenumbers, intensities):
    min_intensity = np.min(intensities)
    max_intensity = np.max(intensities)
    peak_height = max_intensity - min_intensity
    half_max = min_intensity + 0.5 * peak_height

    interp = interp1d(
        wavenumbers,
        intensities - half_max,
        kind='linear',
        fill_value='extrapolate'
    )
    sign_changes = np.where(np.diff(np.sign(interp(wavenumbers))))[0]

    if len(sign_changes) >= 2:
        left = wavenumbers[sign_changes[0]]
        right = wavenumbers[sign_changes[-1]]
        return right - left
    else:
        return None

def get_StN(key, wnms, intens):
    peak_range = [1500, 1650]
    noise_range = [1800, 1900]

    sig = intens[(wnms > peak_range[0]) & (wnms < peak_range[1])].max()
    noise_slice = intens[(wnms > noise_range[0]) & (wnms < noise_range[1])]
    noise_mean = noise_slice.mean()
    noise_std = noise_slice.std(ddof=1)
    StN = (sig - noise_mean) / noise_std
    print(f"Signal to noise of G band for {key} is {StN:.3f}")
    return StN

def integrate_band_area(wavenumbers, intensities, band_range, clip_negative=True):
    """
    Integrate area under a Raman band using baseline-corrected intensities.
    By default, negative values are clipped to zero after baseline subtraction.
    """
    mask = (wavenumbers >= band_range[0]) & (wavenumbers <= band_range[1])
    x = wavenumbers[mask]
    y = intensities[mask]

    if len(x) < 2:
        return np.nan

    if clip_negative:
        y = np.clip(y, 0, None)

    area = np.trapz(y, x)
    return area

# === LOAD FILES ===
all_files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]
sample_paths = {os.path.splitext(f)[0]: os.path.join(data_folder, f) for f in all_files}

# === LOAD PREVIOUS SETTINGS IF AVAILABLE ===
if os.path.exists(settings_file):
    with open(settings_file, 'r') as f:
        saved_settings = json.load(f)
else:
    saved_settings = {}

# === DISPLAY AVAILABLE FILES ===
print("\nAvailable Raman Samples:")
for i, name in enumerate(sample_paths):
    print(f"{i}: {name}")

# === USER INPUTS ===
if 'indices' in saved_settings:
    print(f"Previously selected sample indices: {saved_settings['indices']}")
selected = input("Enter indices of samples to plot (comma separated, or x to reuse previous): ")
if selected.strip().lower() == 'x' and 'indices' in saved_settings:
    indices = saved_settings['indices']
else:
    indices = [int(i.strip()) for i in selected.split(',')]
    saved_settings['indices'] = indices

if 'plot_title' in saved_settings:
    print(f"Previous title: {saved_settings['plot_title']}")
plot_title = input("Enter plot title (or x to reuse previous): ")
if plot_title.strip().lower() == 'x' and 'plot_title' in saved_settings:
    plot_title = saved_settings['plot_title']
else:
    saved_settings['plot_title'] = plot_title

# D/G/G' label shift
dg_shift_x = saved_settings.get('dg_x_shift', 0.0)
dg_shift_y = saved_settings.get('dg_y_shift', 0.0)
print(f"Previous D/G x-shift: {dg_shift_x}, y-shift: {dg_shift_y}")
dg_shift_input = input("Enter x and y shift for D/G labels separated by comma (or x to reuse previous): ")
if dg_shift_input.strip().lower() == 'x':
    dg_x_shift = dg_shift_x
    dg_y_shift = dg_shift_y
else:
    dg_x_shift, dg_y_shift = [float(v.strip()) for v in dg_shift_input.split(',')]
    saved_settings['dg_x_shift'] = dg_x_shift
    saved_settings['dg_y_shift'] = dg_y_shift

dg_x_default = {'D': 1240, 'G': 1635, "D'": 2750}
dg_y_default = 1.25 + len(indices) * 0.25

offsets = {}
shifts = {}
nicknames = {}

saved_samples = saved_settings.get('samples', {})

for idx in indices:
    key = list(sample_paths.keys())[idx]
    print(f"\nSample: {key}")
    if key in saved_samples:
        prev = saved_samples[key]
        print(f"  Previous values -> Nickname: {prev['nickname']}, Offset: {prev['offset']}, Shift: {prev['shift']}")
    entry = input("  Enter nickname; offset; horizontal shift (or x to reuse previous): ")

    if entry.strip().lower() == 'x' and key in saved_samples:
        nickname = saved_samples[key]['nickname']
        offset = saved_samples[key]['offset']
        shift = saved_samples[key]['shift']
    else:
        parts = [x.strip() for x in entry.split(';')]
        nickname = parts[0] if len(parts) > 0 else key
        offset = float(parts[1]) if len(parts) > 1 else 0.0
        shift = float(parts[2]) if len(parts) > 2 else 0.0
        saved_samples[key] = {'nickname': nickname, 'offset': offset, 'shift': shift}

    nicknames[key] = nickname
    offsets[key] = offset
    shifts[key] = shift

saved_settings['samples'] = saved_samples

# Save updated settings
with open(settings_file, 'w') as f:
    json.dump(saved_settings, f, indent=2)

# === PLOT SETUP ===
plt.figure(figsize=(6, 6.5))
colors = ['#000000', '#7E1FD1', '#26C2FF', '#E16462', '#FFB000', '#6F728C', '#00A86A']

# === RESULTS TABLE ===
Raman_parameters = pd.DataFrame(columns=[
    'Sample name',
    'FWHM_G',
    'D_max_corrected',
    'G_max_corrected',
    'D/G_intensity_ratio',
    'D_area',
    'G_area',
    'D/G_area_ratio'
])

# === PROCESS AND PLOT ===
for i, idx in enumerate(indices):
    key = list(sample_paths.keys())[idx]
    path = sample_paths[key]

    data = pd.read_csv(path, sep='\t', header=None, names=['Wavenumber', 'Intensity'])
    data = data[data['Wavenumber'] >= cutoff_wavenumber].copy()
    data['Wavenumber'] = data['Wavenumber'] + shifts[key]

    # --- baseline subtraction ---
    base = baseline_als(data['Intensity'].values, lam=1e6, p=0.001)
    data['Baseline'] = base
    data['Corrected'] = data['Intensity'] - data['Baseline']

    # Optional: signal-to-noise on corrected spectrum
    # get_StN(key, data['Wavenumber'].values, data['Corrected'].values)

    # --- plotting trace (normalized + smoothed only for visualization) ---
    corrected_max = data['Corrected'].max()
    if corrected_max == 0:
        data['Normalized'] = np.nan
    else:
        data['Normalized'] = data['Corrected'] / corrected_max

    data['Smoothed'] = data['Normalized'].rolling(window=10, center=True).mean()

    offset = offsets[key]
    plt.plot(
        data['Wavenumber'],
        data['Smoothed'] + offset,
        label=nicknames[key],
        linewidth=2,
        color=colors[i % len(colors)]
    )

    # --- FWHM of G band ---
    g_band_fwhm = data[
        (data['Wavenumber'] >= G_FWHM_RANGE[0]) &
        (data['Wavenumber'] <= G_FWHM_RANGE[1])
    ]

    fwhm_g = calculate_fwhm(
        g_band_fwhm['Wavenumber'].values,
        g_band_fwhm['Corrected'].values
    )

    if fwhm_g is not None:
        print(f"FWHM of G band for {key}: {fwhm_g:.2f} cm⁻¹")

    # --- D/G peak height ratio from corrected spectrum ---
    d_band = data[
        (data['Wavenumber'] >= D_BAND_RANGE[0]) &
        (data['Wavenumber'] <= D_BAND_RANGE[1])
    ]
    g_band = data[
        (data['Wavenumber'] >= G_BAND_RANGE[0]) &
        (data['Wavenumber'] <= G_BAND_RANGE[1])
    ]

    if not d_band.empty and not g_band.empty:
        d_max = d_band['Corrected'].max()
        g_max = g_band['Corrected'].max()
        d_to_g_ratio = d_max / g_max if g_max != 0 else np.nan
        print(f"D/G intensity ratio for {key}: {d_to_g_ratio:.3f}")
    else:
        d_max = np.nan
        g_max = np.nan
        d_to_g_ratio = np.nan

    # --- D/G area ratio from corrected spectrum ---
    d_area = integrate_band_area(
        data['Wavenumber'].values,
        data['Corrected'].values,
        D_BAND_RANGE,
        clip_negative=True
    )
    g_area = integrate_band_area(
        data['Wavenumber'].values,
        data['Corrected'].values,
        G_BAND_RANGE,
        clip_negative=True
    )

    d_to_g_area_ratio = d_area / g_area if g_area != 0 else np.nan
    print(f"D/G area ratio for {key}: {d_to_g_area_ratio:.3f}")

    # --- store results ---
    Raman_parameters.loc[len(Raman_parameters)] = [
        key,
        fwhm_g,
        d_max,
        g_max,
        d_to_g_ratio,
        d_area,
        g_area,
        d_to_g_area_ratio
    ]

print("\nThe data above is compiled in Raman_parameters")
print(Raman_parameters)

# === D/G/G' LABELS ===
dg_labels = {
    'D': dg_x_default['D'] + dg_x_shift,
    'G': dg_x_default['G'] + dg_x_shift,
    "D'": dg_x_default["D'"] + dg_x_shift
}
dg_y = dg_y_default + dg_y_shift

plt.text(dg_labels['D'], dg_y, 'D', fontsize=10, color='black')
plt.text(dg_labels['G'], dg_y, 'G', fontsize=10, color='black')
plt.text(dg_labels["D'"], dg_y - 0.2, "D'", fontsize=10, color='black')

# === FINAL FORMATTING ===
plt.xlabel('Raman Shift (cm⁻¹)', fontsize=10)
plt.ylabel('Normalized Intensity', fontsize=10)
plt.xticks(fontsize=10)
plt.tick_params(axis='y', labelcolor='white', length=0)
plt.title(plot_title)
plt.legend(fontsize=9)

ax = plt.gca()
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params(direction='in', width=2)

plt.tight_layout()
plt.show()

#### ^^^ Raman ^^^ #######

#%% Mass Spec Calibs

#######################################################
######### M A S S   S P E C   C A L I B S #############
#######################################################

"""
RGA plot + time-window averaging

Dependencies
------------
    pip install pandas matplotlib
"""

from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "Avenir",
    "font.weight": "bold",
    "axes.labelweight": "bold"
})

# ------------------------------------------------------------------
# 1)  EDIT ME: file location
# ------------------------------------------------------------------
FILEPATH = Path(r"/Users/rosadrianazelaya/Library/CloudStorage/Box-Box/Halomaterials/Rosa/Experiment Data/Calibration-data/9-12-25-cal-ms.txt")

# ------------------------------------------------------------------
# 2)  EDIT ME: list of (start_h, end_h) tuples
#     Example: analyse 0–0.5 h and 2.5–3 h
# ------------------------------------------------------------------
TIME_WINDOWS = [
    # (start_hour, end_hour),
    (0.32, 0.67),
    (0.69, 0.95),
    (0.98, 1.28),
    (1.32, 1.59),
    (1.62, 1.87),
    (1.92, 2.18),
    (2.23, 2.48),
    (2.52, 2.78),
    (2.82, 3.08),
    (3.12, 3.42),
    (3.48, 3.98),
]

# ------------------------------------------------------------------
# 3)  Locate header line that begins “Elapsed Time (s)”
# ------------------------------------------------------------------
with FILEPATH.open("r", encoding="utf-8", errors="ignore") as f:
    for idx, line in enumerate(f):
        if line.strip().startswith("Elapsed Time"):
            header_row = idx
            break
    else:
        raise RuntimeError("Header row starting with 'Elapsed Time (s)' not found.")

# ------------------------------------------------------------------
# 4)  Read the data table
# ------------------------------------------------------------------
df = pd.read_csv(
    FILEPATH,
    skiprows=header_row,      # skip preamble
    header=0,                 # first remaining line is the header
    sep=",",
    engine="python",
    skipinitialspace=True,
)

# Convert seconds → hours
df["Hours"] = df["Elapsed Time (s)"] / 3600.0

# List pressure columns (everything except time & Hours)
pressure_cols = [c for c in df.columns if c not in ("Elapsed Time (s)", "Hours")]

# ------------------------------------------------------------------
# 5)  Plot all traces
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
for col in pressure_cols:
    plt.plot(df["Hours"], df[col], label=col)

plt.yscale("log", base=10)
plt.xlabel("Hours since start")
plt.ylabel("Partial pressure (Torr)")
plt.title("RGA partial pressures vs. time")
plt.legend(fontsize="small", ncol=2)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 6)  Average each time window, if any are provided
# ------------------------------------------------------------------
if TIME_WINDOWS:
    avg_dict = {}
    for start_h, end_h in TIME_WINDOWS:
        mask = (df["Hours"] >= start_h) & (df["Hours"] <= end_h)
        window_label = f"{start_h:g}–{end_h:g} h"
        avg_dict[window_label] = df.loc[mask, pressure_cols].mean()

    avg_df = pd.DataFrame(avg_dict).T  # rows = windows, cols = gases

    # Nicely formatted display
    print("\nAverage partial pressures (Torr):")
    print(avg_df.to_string(float_format=lambda x: f"{x:.3e}"))

    # Uncomment to export the table
    # avg_df.to_csv("rga_average_pressures.csv")

##### ^^^ Mass Spec Calibs ^^^ ####


#%% Plotting MS and calculating conversion

#######################################################
############## M A S S   S P E C   E X P ##############
#######################################################

"""
Plot RGA pressures versus time (log-10 scale, hours since start).
"""

from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "Avenir",
    "font.weight": "bold",
    "axes.labelweight": "bold"
})
import numpy as np
from scipy.optimize import curve_fit

# ------------------------------------------------------------------
# 1)  EDIT THESE LINES ONLY ↴
# ------------------------------------------------------------------
#FILEPATH = Path(r"/Users/rosadrianazelaya/Library/CloudStorage/Box-Box/Halogens/Experiment Data/5-20-25-C2R3-PreCarb-3/ms-calibration.txt")
#Folder_path = r"/Users/rosadrianazelaya/Library/CloudStorage/Box-Box/Halogens/Experiment Data/8-5-25-C3R11-Fe-Syn-Beads-3/"
# Folder_path = r"/Users/rosadrianazelaya/Library/CloudStorage/Box-Box/Halomaterials/Rosa/Experiment Data/9-29-25-C3R20-Fe-Chips-MCS-4/"
# Folder_path = r"/Users/rosadrianazelaya/Library/CloudStorage/Box-Box/Halomaterials/Rosa/Experiment Data/7-1-25-C3R4-Fe-Chips-2/"
Folder_path = r"/Users/rosadrianazelaya/Library/CloudStorage/Box-Box/Halomaterials/Rosa/Experiment Data/11-4-2-C4R1/"
# Folder_path = r"/Users/rosadrianazelaya/Library/CloudStorage/Box-Box/Halomaterials/Rosa/Experiment Data/8-20-25 Ar cooking/"

FILEPATH = Path(Folder_path + r"C4R1-ms.txt")
t_heat_up = 0 #hours 
#  C3R4 = 0.3432 C3R12 = 0.069 
# C3R14 = 0.3816  C3R16 = 0.1781 C3R17 = 0.85 C3R18 = 0.2543 C3R19 = 0.2301 C3R20 = 0.1933



vertical_lines = [
    ]

CH4_flow = 6 #sccm 
exp_end_point = 6 # hours
end_hr = 6 #for plot
HtoC_starting = 0.1/6
#C3R20 = 60; 6.89; 6.89
# ------------------------------------------------------------------
# 2)  Locate the header row that starts with “Elapsed Time (s)”
# ------------------------------------------------------------------
with FILEPATH.open("r", encoding="utf-8", errors="ignore") as f:
    for idx, line in enumerate(f):
        if line.strip().startswith("Elapsed Time"):
            header_row = idx
            break
    else:
        raise RuntimeError(
            "Could not find a line starting with 'Elapsed Time (s)'.\n"
            "Is this the correct file and path?"
        )

# ------------------------------------------------------------------
# 3)  Read the data table into a DataFrame
# ------------------------------------------------------------------
df = pd.read_csv(
    FILEPATH,
    skiprows=header_row,     # skip everything before column names
    header=0,                # first remaining line is the header
    sep=",",                 # comma-separated
    engine="python",
    skipinitialspace=True,   # trim the leading spaces after commas
)

# ------------------------------------------------------------------
# 4)  Convert elapsed time (s) → hours since start
# ------------------------------------------------------------------
df["Hours"] = df["Elapsed Time (s)"] / 3600.0 - t_heat_up

# Identify the pressure columns (everything except time & Hours)
pressure_cols = [c for c in df.columns if c not in ("Elapsed Time (s)", "Hours")]

# ------------------------------------------------------------------
# 5)  Plot each species on the same log-scaled axis
# ------------------------------------------------------------------
# ─── Plot-style helper ──────────────────────────────────────
def _style(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2)
      

plt.figure(figsize=(10, 6))

for col in pressure_cols:
    plt.plot(df["Hours"], df[col], label=col)
    

plt.yscale("log", base=10)
plt.xlabel("Hours since start")
plt.ylabel("Partial pressure (Torr)")
plt.title("RGA partial pressures vs. time")
plt.legend(fontsize="small", ncol=2)
plt.tight_layout()
_style(plt.gca())
plt.show()


# CONVERSION 
#calibration prepped 9-12-25 in Calibration sheet
m = 4.195034927
b = - 0.041611609

ms_HtoC_ratio = df['Hydrogen (Torr)'] / df['Methane (Torr)']
flow_HtoC_ratio = (ms_HtoC_ratio - b ) / m

# Define your tolerance
tolerance = 0.005

# Find the index where time is closest to zero within the tolerance
close_to_zero = df[abs(df["Hours"]) <= tolerance]

if not close_to_zero.empty:
    # Take the first (and likely only) valid match
    target_index = close_to_zero.index[0]
    flow_ratio_starting = flow_HtoC_ratio.loc[target_index]
    print(f"Closest time: {df.loc[target_index, 'Hours']}, Flow ratio: {flow_ratio_starting}")
else:
    print("No time value close enough to 0 found.")


#preset starting ratio
Methane_conversion = (flow_HtoC_ratio - HtoC_starting) / (flow_HtoC_ratio + 2) * 100 


#using the flow ratio at t=0
# Methane_conversion = (flow_HtoC_ratio - flow_ratio_starting) / (flow_HtoC_ratio + 2) * 100
df["Methane conversion"] = Methane_conversion

MC_C3R4 = Methane_conversion

plt.figure(figsize=(10, 6))

conv_time = df["Hours"] 
Time_C3R4 = conv_time

mask = conv_time.between(0, exp_end_point)
time_subset = conv_time[mask].reset_index(drop=True)
conv_subset = Methane_conversion[mask].reset_index(drop=True)

plt.plot(conv_time, Methane_conversion)
_style(plt.gca())
plt.xlabel("Hours since start")
plt.ylabel(" % Methane converted")
plt.xlim([0,end_hr])

CH4_converted = CH4_flow * Methane_conversion/100 
Mass_C = CH4_converted/1000 / 22.4 * 12

close_to_end = df[abs((df["Hours"] - exp_end_point)) <= tolerance]
target_index_2 = close_to_end.index[0]
Avg_mass_C = Mass_C[target_index:target_index_2].mean()

endpoints = [0,conv_time.iloc[target_index_2]]
C_deposited = Avg_mass_C * 60 * (endpoints[1] - endpoints[0])

print(str(C_deposited) + ' g of C deposited')

### mass spec ##

#%% Pressure plots

#######################################################
################## P R E S S U R E ####################
#######################################################


# ────--------------- EDIT THESE FOUR LINES --------------- #
csv_path      = Folder_path + "C3R24-P.csv"
t_mass_spec   = 50/3600   # h elapsed before you started logging RGA / mass-spec data
t_preheat     = 0.1933  # h between RGA start and reactor heat-up
header_lines  = 10     # metadata lines to skip before the first Time,Value row
# ─────────────────────────────────────────────────────── #

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "Avenir Next",
    "font.weight": "bold",
    "axes.labelweight": "bold"
})
from pathlib import Path

# ─── Load and groom the data ────────────────────────────
df = pd.read_csv(Path(csv_path).expanduser(), skiprows=header_lines)
df.columns = [  'inlet_time', 'inlet_barg','outlet_time', 'outlet_barg']

df['inlet_time']  = pd.to_datetime(df['inlet_time'])
# df['outlet_time'] = pd.to_datetime(df['outlet_time'])

df['inlet_h']  = (df['inlet_time']  - df['inlet_time'].iloc[0]).dt.total_seconds() / 3600
# df['outlet_h'] = (df['outlet_time'] - df['outlet_time'].iloc[0]).dt.total_seconds() / 3600

shift = t_mass_spec + t_preheat        # align with mass-spec and set t = 0 at heat-up
df['inlet_h']  -= shift
# df['outlet_h'] -= shift

# ─── Helper for consistent 2-pt borders/ticks ───────────
def _style(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2)

# ─── Inlet pressure plot ────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(df['inlet_h'], df['inlet_barg'], color='#1c96ea', linewidth=2)
plt.xlabel('Time (h)')
plt.ylabel('Feed P [barg]')
# plt.title('Inlet Pressure')
_style(plt.gca())

# plt.plot([1.15,1.15],[0,2], color = '#000000', linewidth = 2, label = 'CCl4 introduced')
# plt.legend()

plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()

# # ─── Outlet pressure plot ───────────────────────────────
# plt.figure(figsize=(8, 4))
# plt.plot(df['outlet_h'], df['outlet_barg'], color='#10ba8c', linewidth=2)
# plt.xlabel('Time since reactor heat-up (h)')
# plt.ylabel('Outlet pressure (barg)')
# plt.title('Outlet Pressure')
# _style(plt.gca())
# plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
# plt.show()



#%%Pico log temp

#######################################################
############## T E M P E R A T U R E ##################
#######################################################


# ─────────── EDIT THESE FOUR LINES ─────────────────────────────────────────── #
csv_path      = Folder_path + r'C4R1-T.csv'   # full path to temperature CSV
t_mass_spec   = 0/60 + 0/3600    # h elapsed before mass-spec logging began
# C3R4 = 18;38 C3R14 = 15;49 C3R16 = 2;40 C3R18 = 3;7 C3R19 = 3;3 C3R20 = 4;53
# C3R26 = 1;48
t_preheat     = 0   # h between mass-spec start and reactor heat-up
#  C3R4 = 0.4775 C3R12 = 0.069 
# C3R14 = 0.3816  C3R16 = 0.1781 C3R17 = 0.85 C3R18 = 0.2543 C3R19 = 0.2301 C3R20 = 0.1933
# C3R26 = 0.1056
header_lines  = 0       # lines of boiler-plate *before* the header row
temp_verticals = []
# ───────────────────────────────────────────────────────────── #

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "Avenir",
    "font.weight": "bold",
    "axes.labelweight": "bold"
})
import re
from pathlib import Path

df = pd.read_csv(Path(csv_path).expanduser(), skiprows=header_lines)

# ─── Discover column names ───────────────────────────────────
def _first_match(pattern, cols):
    for c in cols:
        if re.search(pattern, c, re.I):
            return c
    return None

time_col = _first_match(r'\btime\b', df.columns)            # standard case
if time_col is None:                                        # fallback: first column
    time_col = df.columns[0]

wb_col = _first_match(r'(water|bath).*a(?:vg|ve)', df.columns)
in_col = _first_match(r'(inside|reactor).*a(?:vg|ve)', df.columns)

if not wb_col or not in_col:
    raise ValueError(
        "Couldn’t locate average-temperature columns.\n"
        f"Found columns: {df.columns.tolist()}\n"
        "Expected one header containing ‘bath’ or ‘water’ + ‘Ave/Avg’,\n"
        "and another containing ‘inside’ or ‘reactor’ + ‘Ave/Avg’."
    )

print(f"Detected ➜ Time: '{time_col}', Water bath: '{wb_col}', Inside reactor: '{in_col}'")

# ─── Convert timestamps to hours since log start ─────────────
# Try datetime first; fall back to numeric if necessary
try:
    df[time_col] = pd.to_datetime(df[time_col])
    df['t_h'] = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds() / 3600
except Exception:
    df['t_h'] = (df[time_col] - df[time_col].iloc[0]) / 3600  # assume already numbers

# Align time so t = 0 at reactor heat-up
df['t_h'] += (t_mass_spec - t_preheat)


# ─── Plot-style helper ──────────────────────────────────────
def _style(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2)
dfT = df

# # Water-bath temperature
# plt.figure(figsize=(8, 4))
# plt.plot(dfT['t_h'], dfT[wb_col], color='#124ff0', linewidth=2)
# plt.xlabel('Time since reactor heat-up (h)')
# plt.ylabel('Water-bath temperature (°C)')
# plt.title('Water-Bath Temperature')
# _style(plt.gca())
# plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
# plt.show()

# Inside-reactor temperature
plt.figure(figsize=(8, 4))
plt.plot(dfT['t_h'], dfT[in_col], color='#f01212', linewidth=2)
plt.xlabel('Time since reactor heat-up (h)')
plt.ylabel('Inside-reactor temperature (°C)')
plt.title('Inside-Reactor Temperature')
_style(plt.gca())
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()


#%% Temp and methane conversion

#######################################################
############## T E M P   &  C O N V ##################
#######################################################

figtp, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(conv_time, Methane_conversion, color='#1c96ea', lw=2)
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Methane conversion %')
ax1.grid(True, ls='--', lw=0.5, alpha=0.5)
ax1.set_ylim([0,40])

ax2 = ax1.twinx()

ax2.plot(dfT['t_h'], dfT[in_col], color='#f01212', lw=1)
ax2.set_ylabel('Inside-reactor temperature (°C)')

ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)

for x, lbl in temp_verticals:
    add_event(ax1, x, lbl)

_style(ax2)

_style(ax1)
ax1.set_ylim([0.4,1.50])
ax2.set_ylim([940,970])
ax1.set_xlim([7.1, 7.9])
figtp.tight_layout()
plt.show()


#%% COmparing conversions

#######################################################
##############  C O N V     C O M P  ##################
#######################################################

figCC, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(Time_C3R12, MC_C3R12, color='#1c96ea', lw=2)
ax1.plot(Time_C3R16, MC_C3R16, color='#b300ff', lw=2)
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Methane conversion %')
ax1.grid(True, ls='--', lw=0.5, alpha=0.5)
ax1.set_ylim([0,40])

# ax2 = ax1.twinx()

# ax2.plot(dfT['t_h'], dfT[in_col], color='#f01212', lw=1)
# ax2.set_ylabel('Inside-reactor temperature (°C)')

ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)

# for x, lbl in temp_verticals:
#     add_event(ax1, x, lbl)

# _style(ax2)

_style(ax1)
# ax2.set_ylim([1000,1120])
ax1.set_xlim([0,4])
figtp.tight_layout()
plt.show()


colors = ['#000000', '#7E1FD1','#26C2FF',  '#E16462', '#FFB000','#6F728C', '#00A86A' ]

## ^^ Comparing conversions ^^ ##
#%% TGA

#######################################################
######################## T G A ########################
#######################################################

#!/usr/bin/env python3
"""
TGA Data Processing & **Gaussian Peak Fitting**
=============================================
*Change log (rev 3)*
-------------------
* Dropped the simple peak‑detection heuristics.  We now **fit Gaussian
  functions** to each hump in *Deriv. Weight (%/°C) vs. Temperature* and label
  the fitted peak centres (μ) on the plot.
* Uses SciPy’s `find_peaks` to get rough starting positions, then
  `curve_fit` → `A·exp(-(T-μ)²/(2σ²)) + B`.  If SciPy is missing, the script
  falls back to the old local‑max method.
* Each fitted μ is written on the plot (in the curve’s colour).  Overlapping
  labels shift sideways to stay readable and inside the axes.

Paste this entire cell into Spyder/Jupyter and run.  No `main()` needed.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "Avenir",
    "font.weight": "bold",
    "axes.labelweight": "bold"
})
try:
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

# ───────────────────────── Configuration ───────────────────────────────────── #

MAIN_DIR = Path("/Users/rosadrianazelaya/Library/CloudStorage/Box-Box/Halomaterials/Rosa/TGA")
CACHE_FILE = MAIN_DIR / ".tga_cache.json"
COLORS = ['#000000', '#7E1FD1','#26C2FF',  '#E16462', '#FFB000','#6F728C', '#00A86A' ]

# ─────────────────────── File & Cache Helpers ─────────────────────────────── #

def read_text_lines(path: Path):
    raw = path.read_bytes()
    enc = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw[:100] else "utf-8"
    try:
        txt = raw.decode(enc)
    except UnicodeDecodeError:
        txt = raw.decode("latin-1", errors="ignore")
    return txt.splitlines()


def find_txt_files(base_dir: Path):
    return sorted(base_dir.rglob("*.txt"), key=lambda p: p.name.lower())


def load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"last_selection": [], "nicknames": {}, "title": ""}


def save_cache(obj):
    CACHE_FILE.write_text(json.dumps(obj, indent=2))

# ───────────────────── TGA Parsing Helpers ────────────────────────────────── #

def parse_tga_header(lines: List[str], path: Path):
    sig_names, start = [], None
    sig_pat = re.compile(r"^Sig\d+\t(?P<name>.+)")
    for idx, raw in enumerate(lines):
        ln = raw.lstrip("\ufeff")
        if (m := sig_pat.match(ln)):
            sig_names.append(m.group("name").strip())
        elif ln.strip().lower().replace(" ", "") == "startofdata":
            start = idx + 1
            break
    if start is None:
        raise ValueError(f"StartOfData not found in {path.name}")
    return sig_names, start


def read_tga_file(path: Path):
    lines = read_text_lines(path)
    cols, idx = parse_tga_header(lines, path)
    data = [ln.split("\t") for ln in lines[idx:] if ln.strip()]
    return pd.DataFrame(data, columns=cols).apply(pd.to_numeric, errors="coerce")

# ────────────────────────── User I/O Helpers ──────────────────────────────── #

def prompt_indices(max_i: int):
    while True:
        try:
            out = [int(x) for x in input("Indices to plot (comma‑sep): ").split(',') if x.strip()]
        except ValueError:
            print("Numbers only."); continue
        if all(0 <= i < max_i for i in out):
            return out
        print(f"Out of range (0‑{max_i-1}).")

# ─────────────────────────── Peak Fitting  ────────────────────────────────── #

# def _gaussian(x, A, mu, sigma, B):
#     return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + B


# def fit_gaussians(x: np.ndarray, y: np.ndarray) -> list[float]:
#     """Return sorted list of Gaussian peak centres μ (°C)."""
#     if not _HAVE_SCIPY:
#         return []
#     pk, _ = find_peaks(y, prominence=0.01 * (y.max() - y.min()))
#     centres = []
#     for idx in pk:
#         lo, hi = max(0, idx - 25), min(len(x) - 1, idx + 25)
#         xw, yw = x[lo:hi], y[lo:hi]
#         if len(xw) < 5:  # need enough points
#             continue
#         p0 = [yw.max() - yw.min(), x[idx], (xw.max() - xw.min()) / 6, yw.min()]
#         try:
#             popt, _ = curve_fit(_gaussian, xw, yw, p0=p0, maxfev=5000)
#             centres.append(popt[1])        # μ is 2nd parameter
#         except RuntimeError:
#             continue
#     return sorted({round(mu) for mu in centres})  # dedupe & sort
def _g(x,A,mu,sigma,B): return A*np.exp(-(x-mu)**2/(2*sigma**2))+B

def fit_params(x: np.ndarray, y: np.ndarray):
    if not _HAVE_SCIPY: return []
    pk,_ = find_peaks(y, prominence=0.01*(y.max()-y.min()))
    params=[]
    for idx in pk:
        lo,hi=max(0,idx-25),min(len(x)-1,idx+25)
        if hi-lo<5: continue
        xw,yw=x[lo:hi],y[lo:hi]
        p0=[yw.max()-yw.min(), x[idx], (xw.max()-xw.min())/6, yw.min()]
        try:
            if p0[1] > 50:
                params.append(tuple(curve_fit(_g,xw,yw,p0=p0,maxfev=5000)[0]))
        except RuntimeError: pass
    return sorted(params, key=lambda p:p[1])

def centres(params): return sorted({round(p[1]) for p in params})


# ───────────────────────────── Execution ──────────────────────────────────── #

cache = load_cache()
files = find_txt_files(MAIN_DIR)
if not files:
    sys.exit("No .txt files found.")

names = [p.stem for p in files]
print("\nSamples:")
for i, n in enumerate(names):
    print(f"[{i:2d}] {n}")
print()
sel = cache["last_selection"] if cache["last_selection"] and input(f"Reuse {cache['last_selection']}? (y/n): ").lower().startswith('y') else prompt_indices(len(files))
cache["last_selection"] = sel

nick = {}
for i in sel:
    dflt = cache["nicknames"].get(names[i], names[i])
    nick[i] = input(f"Nickname for {names[i]} [{dflt}]: ").strip() or dflt
    cache["nicknames"][names[i]] = nick[i]

save_cache(cache)

dfs = {}
for i in sel:
    dfs[names[i]] = read_tga_file(files[i])
    
# ─────────── Add DTG and 2DTG columns to each loaded dataframe ─────────── #

def _first_match(columns, *needles):
    s = [c for c in columns if any(n in c.lower() for n in needles)]
    return s[0] if s else None

for name, df in dfs.items():
    print("working on " + name)
    # Try to discover the relevant columns
    tempcol = "Temperature (°C)"#_first_match(df.columns, "Temperature (°C)")   # Temperature
    d1col = "Deriv. Weight (%/°C)"#_first_match(df.columns, "Deriv. Weight (%/°C)")    # Existing DTG
    wcol  = "Weight (mg)"#_first_match(df.columns, "Weight (mg)")          # Raw Weight

    # if tempcol is None:
    #     continue  # can't proceed without temperature

    # Ensure sorted by temperature (important for gradients)
    # df.sort_values(tempcol, inplace=True, ignore_index=True)

    T = df[tempcol].to_numpy()

    if d1col is not None:
        # You already have DTG: compute 2DTG = d^2(Weight)/dT^2
        d1 = df[d1col].to_numpy()
        df["2DTG (d²Weight/dT²)"] = np.gradient(d1, T, edge_order=2)
    elif wcol is not None:
        # No DTG present: compute DTG, then 2DTG
        W = df[wcol].to_numpy()
        df["DTG (dWeight/dT)"] = np.gradient(W, T, edge_order=2)
        df["2DTG (d²Weight/dT²)"] = np.gradient(df["DTG (dWeight/dT)"].to_numpy(), T, edge_order=2)
    # else: if neither d1col nor wcol could be found, we can't make 2DTG

# [read_tga_file(files[i]) for i in sel]

cols = dfs[names[sel[0]]].columns.tolist()
print("\nVariables:")
for j, c in enumerate(cols):
    print(f"[{j:2d}] {c}")
print()

x_var = cols[int(input("x‑var index: "))]
y_var = cols[int(input("y‑var index: "))]

def_title = cache.get("title", f"{y_var} vs {x_var}")
plot_title = input(f"Title [{def_title}]: ").strip() or def_title
cache["title"] = plot_title
save_cache(cache)

# ──────────── Plot ──────────── #
fig, ax = plt.subplots()
colors = ['#000000', '#7E1FD1', '#26C2FF',  '#E16462', '#FFB000','#6F728C', '#00A86A' ]
used_x = []

text_y = 0.60            # start just below the legend (axes fraction)
line_height = 0.05       # vertical step for each sample
# for df, i, c in zip(dfs, sel, colors):
#     ax.plot(df[x_var], df[y_var], color=c, label=nick[i])

#     if _HAVE_SCIPY and y_var.lower().startswith('deriv') and 'temp' in x_var.lower():
#         mus = fit_gaussians(df[x_var].values, df[y_var].values)
#         if mus:
#             label_str = ", ".join([f"T{k+1} = {t:.0f} °C" for k, t in enumerate(mus)])
#             ax.text(.02, text_y, label_str, transform=ax.transAxes,
#                     color=c, ha='left', va='top', fontsize=9)
#             text_y -= line_height
# ax.set_xlabel(x_var)
# ax.set_ylabel(y_var)
# ax.set_title(plot_title)
# for sp in ax.spines.values():
#     sp.set_linewidth(2)
# ax.tick_params(width=2)
# ax.legend(loc='upper left')

dict_tga = {}

for i, c in zip(sel, colors):
    df = dfs[names[i]]

    if y_var.lower().startswith('weight') and 'temp' in x_var.lower():
        df[y_var] = df[y_var]/(df[y_var][0]) * 100
    ax.plot(df[x_var],df[y_var], color=c ,lw=2,label= nick[i])
    dict_tga[nick[i]] = df[[x_var, y_var]].copy()
    
    if y_var.lower().startswith('2dtg') and 'temp' in x_var.lower():
        df['2dtg avg'] = df[y_var].rolling(window=200, center=True).mean()
        ax.plot(df[x_var],df['2dtg avg'], color=c ,lw=1,label= nick[i] + ' avg')
        m = (df["Temperature (°C)"] > 200) & (df["2dtg avg"] > 1e-3)
        T_on = df.loc[m, "Temperature (°C)"].min() if m.any() else None  # None if not found
        T_on = round(T_on)
        m2 = (df["Temperature (°C)"] > 400) & (df["2dtg avg"] < -1e-3)
        T_off = df.loc[m2, "Temperature (°C)"].iloc[-1] if m.any() else None  # last match in order
        T_off = round(T_off)
        
        print(nick[i] + " T_on " + str(T_on) + " and T_off " + str(T_off))
    
    if _HAVE_SCIPY and y_var.lower().startswith('deriv') and 'temp' in x_var.lower():
        pars=fit_params(df[x_var].values,df[y_var].values)

        # if pars:
        #     lab=", ".join(f"T{k+1} = {t} °C" for k,t in enumerate(centres(pars)))
        #     ax.text(.02,text_y,lab,transform=ax.transAxes,color=c,ha='left',va='top',fontsize=9)
        #     text_y -= line_height
ax.legend(loc='upper left')
if y_var.lower().startswith('weight'):
    y_var = 'Weight Percent (%)'
    ax.legend(loc='lower left')
    

ax.set_xlabel(x_var); ax.set_ylabel(y_var); ax.set_title(plot_title)
for spp in ax.spines.values(): spp.set_linewidth(2)
ax.tick_params(width=2)

fig.tight_layout()
plt.show()

######## ^^ TGA ^^ ########
#%% TGA/DTG plots

#######################################################
################### T G A / D T G #####################
#######################################################


import json
import re
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "Avenir Next",
    "font.weight": "bold",
    "axes.labelweight": "bold"
})
try:
    from scipy.signal import find_peaks, savgol_filter
    from scipy.optimize import curve_fit
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

# ───────────────────────── Configuration ───────────────────────────────────── #

MAIN_DIR = Path("/Users/rosadrianazelaya/Library/CloudStorage/Box-Box/Halomaterials/Rosa/TGA")
CACHE_FILE = MAIN_DIR / ".tga_cache.json"
COLORS = ['#000000', '#7E1FD1','#26C2FF',  '#E16462', '#FFB000','#6F728C', '#00A86A' ]
transparencies = [1,1,1,1,1,1,1]

\
# ── Samples that need smoothing: add the file stem names (no .txt) here ──────── #
# Example: SMOOTH_SAMPLES = ["sample_03_run2", "sample_07_highburn"]
SMOOTH_SAMPLES = ['CM3-Post Gr-CO2', 'C3R13 CO2 mix'
    # "your_sample_name_here",
]

SMOOTH_HALF_WINDOW = 20   # rolling average uses this many points on each side (weight smoothing)

# ── Savitzky-Golay derivative settings (used for DTG on SMOOTH_SAMPLES only) ─ #
# SG_WINDOW  : number of points in the fitting window — must be odd and > SG_POLYORDER
#              larger  → smoother DTG, but softens sharp peaks
#              smaller → noisier DTG, but preserves sharp features better
#              good starting range for TGA: 21–51
# SG_POLYORDER : degree of the polynomial fitted inside each window
#              2 or 3 works well for most TGA curves; 4 if you have narrow peaks
SG_WINDOW    = 501   # must be odd
SG_POLYORDER = 2

# ─────────────────────── File & Cache Helpers ─────────────────────────────── #

def read_text_lines(path: Path):
    raw = path.read_bytes()
    enc = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw[:100] else "utf-8"
    try:
        txt = raw.decode(enc)
    except UnicodeDecodeError:
        txt = raw.decode("latin-1", errors="ignore")
    return txt.splitlines()


def find_txt_files(base_dir: Path):
    return sorted(base_dir.rglob("*.txt"), key=lambda p: p.name.lower())


def load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"last_selection": [], "nicknames": {}, "title": ""}


def save_cache(obj):
    CACHE_FILE.write_text(json.dumps(obj, indent=2))

# ───────────────────── TGA Parsing Helpers ────────────────────────────────── #

def parse_tga_header(lines: List[str], path: Path):
    sig_names, start = [], None
    sig_pat = re.compile(r"^Sig\d+\t(?P<name>.+)")
    for idx, raw in enumerate(lines):
        ln = raw.lstrip("\ufeff")
        if (m := sig_pat.match(ln)):
            sig_names.append(m.group("name").strip())
        elif ln.strip().lower().replace(" ", "") == "startofdata":
            start = idx + 1
            break
    if start is None:
        raise ValueError(f"StartOfData not found in {path.name}")
    return sig_names, start


def read_tga_file(path: Path):
    lines = read_text_lines(path)
    cols, idx = parse_tga_header(lines, path)
    data = [ln.split("\t") for ln in lines[idx:] if ln.strip()]
    return pd.DataFrame(data, columns=cols).apply(pd.to_numeric, errors="coerce")

# ────────────────────────── User I/O Helpers ──────────────────────────────── #

def prompt_indices(max_i: int):
    while True:
        try:
            out = [int(x) for x in input("Indices to plot (comma‑sep): ").split(',') if x.strip()]
        except ValueError:
            print("Numbers only."); continue
        if all(0 <= i < max_i for i in out):
            return out
        print(f"Out of range (0‑{max_i-1}).")


# ───────────────────── Smoothing Helpers ──────────────────────────────────── #

def rolling_average(arr: np.ndarray, half_win: int) -> np.ndarray:
    """
    Centred rolling average with a shrinking window near the edges.
    Each output point i is the mean of arr[max(0, i-half_win) : i+half_win+1].
    The number of output points equals the number of input points.
    """
    n = len(arr)
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - half_win)
        hi = min(n, i + half_win + 1)
        out[i] = arr[lo:hi].mean()
    return out


def fix_temperature_monotonic(T: np.ndarray) -> np.ndarray:
    """
    Find every window [A, B] where temperature is not strictly increasing,
    then replace those points with a linearly spaced sequence from T[A] to T[B].
    Multiple non-monotonic windows are handled in a single pass.
    """
    T = T.copy()
    n = len(T)
    i = 1
    while i < n:
        if T[i] <= T[i - 1]:
            # Found the start of a non-monotonic window at index i-1 (point A)
            A = i - 1
            # Scan forward to find point B: first index after A where the series
            # climbs back above T[A] and stays increasing to the next point
            B = i + 1
            while B < n - 1:
                if T[B] > T[A] and T[B + 1] > T[B]:
                    break
                B += 1
            # Safety: if we hit the end without finding a recovery, just go to end
            if B >= n:
                B = n - 1
            # Replace A..B (inclusive) with a linear ramp
            num_pts = B - A + 1
            T[A:B + 1] = np.linspace(T[A], T[B], num_pts)
            # Resume scanning from B
            i = B + 1
        else:
            i += 1
    return T


def smooth_sample(df: pd.DataFrame, tempcol: str, wcol: str,
                  half_win: int, sg_window: int, sg_polyorder: int) -> pd.DataFrame:
    """
    Apply rolling average to weight data, fix temperature monotonicity,
    then recompute DTG (dWeight%/dT) using a Savitzky-Golay derivative.

    SG fits a polynomial of degree `sg_polyorder` to each window of `sg_window`
    consecutive points and analytically differentiates it, so the slope estimate
    is based on many points at once — far less prone to spikes than np.gradient.

    Returns a copy of df with updated columns.
    """
    df = df.copy()

    # # ── Step 3: smooth weight with rolling average ───────────────────────── #
    W_raw = df[wcol].to_numpy(dtype=float)
    W_smooth = df[wcol].to_numpy(dtype=float)
    W_smooth = rolling_average(W_raw, half_win)
    df[wcol] = W_smooth
    

    # ── Step 4: fix temperature so it is strictly increasing ─────────────── #
    T_raw = df[tempcol].to_numpy(dtype=float)
    T_fixed = fix_temperature_monotonic(T_raw)
    df[tempcol] = T_fixed

    # ── Step 5: recompute DTG via Savitzky-Golay derivative ──────────────── #
    W_pct = W_smooth / W_smooth[0] * 100   # weight percent

    # savgol_filter with deriv=1 returns d(W_pct)/d(index).
    # We divide by d(T)/d(index) (also smoothed by SG) to get d(W_pct)/dT.
    # Both arrays use the same window so the division is point-wise consistent.
    # Ensure window is odd and large enough for the polynomial order.
    win = sg_window if sg_window % 2 == 1 else sg_window + 1   # force odd
    win = max(win, sg_polyorder + 2 if (sg_polyorder + 2) % 2 == 1 else sg_polyorder + 3)

    dW_didx = savgol_filter(W_pct,   window_length=win, polyorder=sg_polyorder, deriv=1)
    dT_didx = savgol_filter(T_fixed, window_length=win, polyorder=sg_polyorder, deriv=1)

    # Guard against near-zero dT (shouldn't happen after fix, but just in case)
    dT_didx = np.where(np.abs(dT_didx) < 1e-6, np.nan, dT_didx)
    dtg = -dW_didx / dT_didx

    df["Deriv. Weight (%/°C)"] = dtg

    return df


# ───────────────────────────── Execution ──────────────────────────────────── #

cache = load_cache()
files = find_txt_files(MAIN_DIR)
if not files:
    sys.exit("No .txt files found.")

names = [p.stem for p in files]
print("\nSamples:")
for i, n in enumerate(names):
    print(f"[{i:2d}] {n}")
print()
sel = cache["last_selection"] if cache["last_selection"] and input(f"Reuse {cache['last_selection']}? (y/n): ").lower().startswith('y') else prompt_indices(len(files))
cache["last_selection"] = sel

nick = {}
for i in sel:
    dflt = cache["nicknames"].get(names[i], names[i])
    nick[i] = input(f"Nickname for {names[i]} [{dflt}]: ").strip() or dflt
    cache["nicknames"][names[i]] = nick[i]

save_cache(cache)

dfs = {}
for i in sel:
    dfs[names[i]] = read_tga_file(files[i])
    
# ─────────── Add DTG and 2DTG columns to each loaded dataframe ─────────── #

def _first_match(columns, *needles):
    s = [c for c in columns if any(n in c.lower() for n in needles)]
    return s[0] if s else None

for name, df in dfs.items():
    print("working on " + name)
    tempcol = "Temperature (°C)"
    d1col   = "Deriv. Weight (%/°C)"
    wcol    = "Weight (mg)"

    # df.sort_values(tempcol, inplace=True, ignore_index=True)

    # ── Step 2: check if this sample needs smoothing ──────────────────────── #
    if name in SMOOTH_SAMPLES:
        print(f"  → applying smoothing to '{name}'")
        df = smooth_sample(df, tempcol, wcol, SMOOTH_HALF_WINDOW, SG_WINDOW, SG_POLYORDER)
        dfs[name] = df          # store the smoothed version back

    T = df[tempcol].to_numpy()

    # if d1col is not None and d1col in df.columns:
    #     d1 = df[d1col].to_numpy()
    #     df["2DTG (d²Weight/dT²)"] = np.gradient(d1, T, edge_order=2)
    # elif wcol is not None and wcol in df.columns:
    #     W = df[wcol].to_numpy()
    #     df["DTG (dWeight/dT)"] = np.gradient(W, T, edge_order=2)
    #     df["2DTG (d²Weight/dT²)"] = np.gradient(df["DTG (dWeight/dT)"].to_numpy(), T, edge_order=2)


cols = dfs[names[sel[0]]].columns.tolist()

x_var   = cols[1]
y_var   = cols[2]
y_var_2 = cols[7]

def_title = cache.get("title", f"{y_var} vs {x_var}")
plot_title = input(f"Title [{def_title}]: ").strip() or def_title
cache["title"] = plot_title
save_cache(cache)

# ──────────── Plot ──────────── #
fig, ax = plt.subplots()
colors = ['#000000', '#7E1FD1', '#26C2FF',  '#E16462', '#FFB000','#6F728C', '#00A86A' ]

dict_tga = {}

for i, c, alph in zip(sel, colors, transparencies):
    df = dfs[names[i]]

    # ── Step 6: plot uses the (possibly smoothed) data already in df ──────── #
    df[y_var] = df[y_var] / df[y_var].iloc[0] * 100
    ax.plot(df[x_var], df[y_var], color=c, lw=2, label=nick[i], alpha = alph)
    dict_tga[nick[i]] = df[[x_var, y_var]].copy()

    if _HAVE_SCIPY and y_var.lower().startswith('deriv') and 'temp' in x_var.lower():
        pars = fit_params(df[x_var].values, df[y_var].values)


# if y_var.lower().startswith('weight'):
#     y_var = 'Weight Percent (%)'
#     ax.legend(loc='lower left')
ax.legend(loc='center left') 

ax2 = ax.twinx()

for i, c, alph in zip(sel, colors, transparencies):
    df = dfs[names[i]]
    
    new_alph = alph*0.6
    ax2.plot(df[x_var], df[y_var_2], color=c, lw=2, label=nick[i], linestyle="--", alpha=new_alph)
    dict_tga[nick[i]] = df[[x_var, y_var_2]].copy()

    # if _HAVE_SCIPY and y_var_2.lower().startswith('deriv') and 'temp' in x_var.lower():
    #     pars = fit_params(df[x_var].values, df[y_var_2].values)

y_var_2 = 'DTG (%/°C)'
x_var = 'Weight (%)'

ax.set_xlabel(x_var);  ax.set_ylabel(y_var);  ax.set_title(plot_title)
ax2.set_ylabel(y_var_2)
for spp in ax.spines.values():  spp.set_linewidth(2)
ax.tick_params(width=2)
for spp in ax2.spines.values(): spp.set_linewidth(2)
ax2.tick_params(width=2)

ax.set(xlim=(500, 1250), ylim=(0, 105))
ax2.set(xlim=(500, 1250), ylim=(0, 0.75))


fig.tight_layout()
plt.show()



#%% TGA comparison temps 


##############################################################
##################  TGA Temps comparison  ####################
##############################################################


import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd, ast
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# 1) Read from clipboard -> df_tgas
df_tgas = pd.read_clipboard(sep='\t').rename(columns=str.strip)

# 2) Convert bracketed strings like "[518, 622, 766]" -> list[int]
for c in df_tgas.columns[1:]:
    df_tgas[c] = df_tgas[c].apply(
        lambda s: None if pd.isna(s) or str(s).strip()=='' 
        else [int(x) for x in ast.literal_eval(str(s))]
    )

df_tgas2 = pd.DataFrame({"FV":[[513, 676, 755]],"GV":[[515, 674, 758]],"Loose":[[514, 663, 761]]}) #new sample
# df_tgas2.columns = ["FV", "GV", "Loose"]

# 3) Make x numeric
df_tgas['x'] = pd.to_numeric(df_tgas['x'], errors='coerce')


# base hues (cycled by H2:CH4 ratio row)
base_hues = ['#b300ff', '#2ea7e4', '#ff0000', '#ff7f95', '#007200']
markers = {'FV': 'D',  'GV': 'o',  'Loose': 's'}  # square



def tint(hex_color, f):
    """Lighten by mixing with white. f in [0,1]; 0=no change, 1=white."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    return mcolors.to_hex(1 - (1 - rgb) * f)

def shade(hex_color, f):
    """Darken by mixing with black. f in [0,1]; 0=no change, 1=black."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    return mcolors.to_hex(rgb * (1 - f))

# how much to darken/lighten each sample type
role_mix = {'FV': -0.35, 'GV': 0.0, 'Loose': +0.50}  # tweak to taste

def role_color(base_hex, role):
    f = role_mix[role]
    if f > 0:   # lighten
        return tint(base_hex, f)
    if f < 0:   # darken
        return shade(base_hex, -f)
    return base_hex


# column-specific x offsets brightness (FV darker, GV normal, Loose lighter)
x_offset  = {'FV': 0.00, 'GV': 0.02, 'Loose': 0.04}


cols = [c for c in ['FV', 'GV', 'Loose'] if c in df_tgas.columns]
cols2 = [c for c in ['FV', 'GV', 'Loose'] if c in df_tgas2.columns] #new sample



# 1) categorical positions for each ratio (keeps one continuous axes)
ratios = df_tgas['x'].astype(float).tolist()           # [0.1, 0.2222..., 1.0]
xpos   = {r:i for i, r in enumerate(ratios)}           # 0,1,2


# Which sample columns exist
cols = [c for c in ['FV','GV','Loose'] if c in df_tgas.columns]


# ---- categorical x positions (keep your real ratios only as labels)
ratios = pd.unique(df_tgas['x'].astype(float))        # e.g., [0.1, 0.222..., 1.0]
xpos   = {float(r): i for i, r in enumerate(ratios)}  # map ratio -> 0,1,2

# offsets within each category (FV darker, GV mid, Loose light)
x_off_cat = {'FV': -0.12, 'GV': 0.00, 'Loose': +0.12}

fig, ax = plt.subplots(figsize=(6, 6))

for i, row in df_tgas.iterrows():
    base = base_hues[i % len(base_hues)]
    x_cat = xpos[float(row['x'])]
    for col in cols:
        vals = row[col]
        if not isinstance(vals, (list, tuple)) or not vals:
            continue
        ys = sorted(float(v) for v in vals)
        x  = x_cat + x_off_cat[col]                  # <- categorical x
        c  = role_color(base, col)
        ax.plot([x]*len(ys), ys, marker=markers[col], ms=7, lw=1.2, color=c)

for i, row in df_tgas2.iterrows(): #new sample
    base = base_hues[i % len(base_hues)]
    x_cat = 0.35
    for col in cols:
        vals = row[col]
        if not isinstance(vals, (list, tuple)) or not vals:
            continue
        ys = sorted(float(v) for v in vals)
        x  = x_cat + x_off_cat[col]                  # <- categorical x
        c  = role_color(base, col)
        ax.plot([x]*len(ys), ys, marker=markers[col], ms=7, lw=1.2,linestyle = "--", color=c)
    

# axes, ticks, labels
ax.set_xlim(-0.5, len(ratios)-0.5)
ax.set_xticks(range(len(ratios)))
ax.set_xticklabels([f"{r:.3g}" for r in ratios])      # shows 0.1, 0.222, 1
ax.set_xlabel('H₂:CH₄ ratio in feed')
ax.set_ylabel('Temperature (°C)')
ax.spines["right"].set_linewidth(2.0)
ax.spines["left"].set_linewidth(2.0)
ax.spines["bottom"].set_linewidth(2.0)
ax.spines["top"].set_linewidth(2.0)
ax.tick_params(width=2)
ax.grid(True, alpha=0.25)

# ---- draw decorative x-axis break between the 2nd and 3rd categories
i_break = 1                      # between tick 1 and 2 (0-indexed)
xb = i_break + 0.5               # halfway position
d  = 0.015
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
ax.plot([xb-0.05, xb+0.05], [-d, +d], transform=trans, color='k', lw=1.2, clip_on=False)
ax.plot([xb-0.05, xb+0.05], [1-d, 1+d], transform=trans, color='k', lw=1.2, clip_on=False)

# ---- draw decorative x-axis break between the 2nd and 3rd categories
i_break = 1                      # between tick 1 and 2 (0-indexed)
xb = i_break + 0.6               # halfway position
d  = 0.015
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
ax.plot([xb-0.05, xb+0.05], [-d, +d], transform=trans, color='k', lw=1.2, clip_on=False)
ax.plot([xb-0.05, xb+0.05], [1-d, 1+d], transform=trans, color='k', lw=1.2, clip_on=False)

# legend: grayscale only
legend_handles = [
    Line2D([0], [0], color='#333333', marker='D', lw=1.2, ms=7, label='FV'),
    Line2D([0], [0], color='#777777', marker='o', lw=1.2, ms=7, label='GV'),
    Line2D([0], [0], color='#BBBBBB', marker='s', lw=1.2, ms=7, label='Loose'),
]
ax.legend(handles=legend_handles, loc='center right',        # anchor point of the legend box
          bbox_to_anchor=(0.98, 0.20), frameon=True)


fig.tight_layout()
plt.show()

#%%

fig, ax = plt.subplots(figsize=(6.5, 4.2))

for i, row in df_tgas.iterrows():
    base = base_hues[i % len(base_hues)]
    x_base = float(row['x'])
    for col in cols:
        vals = row[col]
        if not isinstance(vals, (list, tuple)) or len(vals) == 0 or pd.isna(vals).any():
            continue
        ys = sorted(float(v) for v in vals)
        x = x_base + x_offset[col]
        c = role_color(base, col)  # col is 'FV'/'GV'/'Loose'
        # plot three diamonds and connect them (vertical line at given x)
        ax.plot([x]*len(ys), ys, marker='D', ms=7, lw=1.2, color=c)

# axes/labels
ax.set_xlabel('H₂:CH₄ ratio in feed')
ax.set_ylabel('Temperature (°C)')
ax.set_xticks(df_tgas['x'].to_list())
ax.set_xlim(min(df_tgas['x']) - 0.05, max(df_tgas['x']) + 0.10)
ax.grid(True, alpha=0.2)

# legend: show mapping using grayscale only
legend_handles = [
    Line2D([0], [0], color='#333333', marker='D', lw=1.2, ms=7, label='FV (dark)'),
    Line2D([0], [0], color='#777777', marker='D', lw=1.2, ms=7, label='GV (medium)'),
    Line2D([0], [0], color='#BBBBBB', marker='D', lw=1.2, ms=7, label='Loose (light)'),
]
ax.legend(handles=legend_handles, title='Sample type', loc='best', frameon=True)

fig.tight_layout()
plt.show()


#%% Comparison Raman and XRD

#######################################################
################## RAMAN AND XRD ######################
#######################################################


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Fonts
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
})

# Load Excel again
raw = pd.read_excel("/Users/rosadrianazelaya/Documents/Erics Lab/Rosa Data 8-18.xlsx", sheet_name="Sheet1")

# ---- Parse averages block (rows 2,3,4) ----
avg = raw.loc[[2,3,4]].copy()

dg_map  = {"D/G": "FV", "Unnamed: 3": "GV", "Unnamed: 4": "Loose"}
dog_map = {"DOG": "FV", "Unnamed: 6": "GV", "Unnamed: 7": "Loose"}

records = []
for _, r in avg.iterrows():
    x_val = r["Averages"]
    x_lab = f"{x_val:.2g}" if x_val != 0.222222 else "0.22"
    for fine in ["FV", "GV", "Loose"]:
        dg_col  = [k for k,v in dg_map.items()  if v==fine][0]
        dog_col = [k for k,v in dog_map.items() if v==fine][0]
        records.append({
            "x_pos": {"0.1":0, "0.22":1, "1":2}[x_lab],
            "x_lab": x_lab,
            "fine": fine,
            "DG": float(r[dg_col]),
            "DOG": float(r[dog_col]),
        })
tidy = pd.DataFrame.from_records(records)

# ## Adding new data point #########################
new = raw.loc[[5]].copy()

records2 = []
for _, r in new.iterrows():
    x_val2 = r["Averages"]
    x_lab2 = f"{x_val2:.2g}" if x_val2 != 0.222222 else "0.22"
    for fine in ["FV", "GV", "Loose"]:
        dg_col2  = [k for k,v in dg_map.items()  if v==fine][0]
        dog_col2 = [k for k,v in dog_map.items() if v==fine][0]
        records2.append({
            "x_pos": {"0.11":0.1}[x_lab2],
            "x_lab": x_lab2,
            "fine": fine,
            "DG": float(r[dg_col2]),
            "DOG": float(r[dog_col2]),
        })
tidy2 = pd.DataFrame.from_records(records2)
####################################################



# ---- Parse error bars block for D/G (rows 9,10,11) ----
err_rows = [10,11,12]#[9,10,11]
err_new = [13]
err_tbl = raw.loc[err_rows, ["Averages","D/G","Unnamed: 3","Unnamed: 4"]].copy()
err_tbl.columns = ["x", "FV", "GV", "Loose"]
err_tbl["x_label"] = err_tbl["x"].map(lambda v: f"{v:.2g}" if v != 0.222222 else "0.22")
err_long = err_tbl.melt(id_vars=["x_label"], value_vars=["FV","GV","Loose"], var_name="fine", value_name="err_diff")
# Merge into tidy
tidy = tidy.merge(err_long, left_on=["x_lab","fine"], right_on=["x_label","fine"], how="left").drop(columns=["x_label"])

## Adding a new one ##########
err_rows2 = [13]
err_tbl2 = raw.loc[err_rows2, ["Averages","D/G","Unnamed: 3","Unnamed: 4"]].copy()
err_tbl2.columns = ["x", "FV", "GV", "Loose"]
err_tbl2["x_label"] = err_tbl2["x"].map(lambda v: f"{v:.2g}" if v != 0.222222 else "0.22")
err_long2 = err_tbl2.melt(id_vars=["x_label"], value_vars=["FV","GV","Loose"], var_name="fine", value_name="err_diff")
tidy2 = tidy2.merge(err_long2, left_on=["x_lab","fine"], right_on=["x_label","fine"], how="left").drop(columns=["x_label"])
####################################################

# Visuals
size_by_fine = {"FV": 240, "GV": 150, "Loose": 70}   # bubble areas
purple = "#6a0dad"
blue   = "#007acc"
purple2 = "#A14FDB"
blue2 = "#50ADEB"
white = "#ffffff" 
offset = 0.1

fig, ax1 = plt.subplots(figsize=(7.4,7))

# Make ax1 background transparent so ax2 right spine stays visible
ax1.patch.set_alpha(0.0)

# D/G (purple) with offsets and error bars
for _, row in tidy.iterrows():
    x = row["x_pos"] - offset
    if row["x_lab"] == "1" and row["fine"] == "Loose":
        x -= 0.06  # extra nudge
    if not pd.isna(row["err_diff"]):
        yerr = float(row["err_diff"]) / 2.0
        ax1.errorbar(x, row["DG"], yerr=yerr, fmt="none",
                     ecolor=purple, elinewidth=2.0, capsize=5, capthick=2.0, zorder=2)
    ax1.scatter(x, row["DG"],
                s=size_by_fine[row["fine"]],
                color=purple, alpha=0.95, marker="o", zorder=4)

## Adding new point ###########################################
for _, row in tidy2.iterrows():
    x = row["x_pos"] - offset
    if row["x_lab"] == "1" and row["fine"] == "Loose":
        x -= 0.06  # extra nudge
    if not pd.isna(row["err_diff"]):
        yerr = float(row["err_diff"]) / 2.0
        ax1.errorbar(x, row["DG"], yerr=yerr, fmt="none",
                     ecolor=purple2, elinewidth=2.0, capsize=5, capthick=2.0, zorder=2)
    ax1.scatter(x, row["DG"],
                s=size_by_fine[row["fine"]],
                color=purple2, alpha=0.95, marker="o", zorder=5)
####################################################

ax1.set_xlabel("H$_2$ : CH$_4$ in feed")
ax1.set_ylabel("D/G", color=purple)
ax1.tick_params(axis='y', colors=purple)
ax1.spines["left"].set_color(purple)
ax1.spines["right"].set_linewidth(2.0)
ax1.spines["left"].set_linewidth(2.0)
ax1.spines["top"].set_linewidth(2.0)
ax1.spines["bottom"].set_linewidth(2.0)
ax1.tick_params(width=2)
ax1.set_xticks([0,1,2]); ax1.set_xticklabels(["0.1","0.22","1"])
ax1.set_ylim(0, 0.6)

# Add diagonal x-axis break marks between 0.22 (x=1) and 1 (x=2)
for pos in [1.45, 1.55]:
    ax1.plot([pos-0.035, pos+0.035], [-0.02, 0.02],
             transform=ax1.get_xaxis_transform(), color="black",
             linewidth=2.0, clip_on=False)

# DOG (blue) right (points only, no error bars provided)
ax2 = ax1.twinx()
for fine, g in tidy.groupby("fine"):
    ax2.scatter(g["x_pos"]+offset, g["DOG"],
                s=g["fine"].map(size_by_fine),
                color=blue, alpha=0.95, marker="o", zorder=3)
    
### Adding new point ##################
for fine, g in tidy2.groupby("fine"):
    ax2.scatter(g["x_pos"]+offset, g["DOG"],
                s=g["fine"].map(size_by_fine),
                color=blue2, alpha=0.95, marker="o", zorder=3)
############################################

ax2.set_ylabel("DOG (%)", color=blue)
ax2.tick_params(axis='y', colors=blue)
ax2.tick_params(width=2)
ax2.spines["right"].set_color(blue)
ax2.spines["right"].set_linewidth(2.0)
ax2.set_ylim(90, 97)

# Draw ax2 above so the right spine is visible
ax2.set_zorder(ax1.get_zorder() + 1)

# Legend bottom-left
legend_handles = [Line2D([0],[0], marker='o', linestyle='',
                         markersize=(size_by_fine[k]**0.5)/1.2,
                         color='black', label=f"{k} fines")
                  for k in ["FV","GV","Loose"]]
ax1.legend(handles=legend_handles, loc="lower left", frameon=False)

ax1.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)
fig.tight_layout()

# png = "/mnt/data/dual_axes_bubbles_errors_breaks_repeat.png"
# svg = "/mnt/data/dual_axes_bubbles_errors_breaks_repeat.svg"
# plt.savefig(png, dpi=300, bbox_inches="tight")
# plt.savefig(svg, bbox_inches="tight")
plt.show()

# png, svg



#%%

#!/usr/bin/env python3
"""
TGA Data Processing + Gaussian Overlay (rev 7)
---------------------------------------------
• Raw derivative curve  → solid black, 2 pt  
• Each fitted Gaussian  → dashed line in the sample’s colour  
• Peak temperatures     → “T1 = …, T2 = …” listed to the right, colour‑matched
Requires SciPy; if SciPy isn’t installed, only the black curve appears.
"""

import json, re, sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

# ───────── Configuration ───────── #
MAIN_DIR = Path("/Users/rosadrianazelaya/Library/CloudStorage/Box-Box/Halogens/TGA")
CACHE_FILE = MAIN_DIR / ".tga_cache.json"
COLORS = ['#b300ff', '#2ea7e4', '#ff0000', '#ff7f95', '#007200', '#7848ff']

# ───── File helpers ───── #
def read_text_lines(path: Path):
    raw = path.read_bytes()
    enc = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw[:80] else "utf-8"
    try:
        txt = raw.decode(enc)
    except UnicodeDecodeError:
        txt = raw.decode("latin-1", errors="ignore")
    return txt.splitlines()

def find_txt_files(base: Path):
    return sorted(base.rglob("*.txt"), key=lambda p: p.name.lower())

def load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"last_selection": [], "nicknames": {}, "title": ""}

def save_cache(obj): CACHE_FILE.write_text(json.dumps(obj, indent=2))

# ───── TGA parsing ───── #
def parse_tga_header(lines: List[str], path: Path):
    sigs, start = [], None
    pat = re.compile(r"^Sig\d+\t(?P<name>.+)")
    for idx, raw in enumerate(lines):
        ln = raw.lstrip("\ufeff")
        if (m := pat.match(ln)):
            sigs.append(m.group("name").strip())
        elif ln.strip().lower().replace(" ", "") == "startofdata":
            start = idx + 1
            break
    if start is None:
        raise ValueError(f"StartOfData not found in {path.name}")
    return sigs, start

def read_tga(path: Path):
    lines = read_text_lines(path)
    cols, idx = parse_tga_header(lines, path)
    data = [ln.split("\t") for ln in lines[idx:] if ln.strip()]
    return pd.DataFrame(data, columns=cols).apply(pd.to_numeric, errors="coerce")

# ───── UI helper ───── #
def prompt_indices(max_i: int):
    while True:
        try:
            out = [int(x) for x in input("Indices to plot (comma‑sep): ").split(',') if x.strip()]
        except ValueError:
            print("Numbers only."); continue
        if all(0 <= i < max_i for i in out): return out
        print(f"Out of range (0‑{max_i-1}).")

# ───── Gaussian fitting ───── #
def _g(x,A,mu,sigma,B): return A*np.exp(-(x-mu)**2/(2*sigma**2))+B

def fit_params(x: np.ndarray, y: np.ndarray):
    if not _HAVE_SCIPY: return []
    pk,_ = find_peaks(y, prominence=0.01*(y.max()-y.min()))
    params=[]
    for idx in pk:
        lo,hi=max(0,idx-25),min(len(x)-1,idx+25)
        if hi-lo<5: continue
        xw,yw=x[lo:hi],y[lo:hi]
        p0=[yw.max()-yw.min(), x[idx], (xw.max()-xw.min())/6, yw.min()]
        try:
            params.append(tuple(curve_fit(_g,xw,yw,p0=p0,maxfev=5000)[0]))
        except RuntimeError: pass
    return sorted(params, key=lambda p:p[1])

def centres(params): return sorted({round(p[1]) for p in params})

# ───────────── Execution ───────────── #
cache = load_cache()
files = find_txt_files(MAIN_DIR)
if not files: sys.exit("No .txt files found.")

names=[p.stem for p in files]
print("\nSamples:")
for i,n in enumerate(names): print(f"[{i:2d}] {n}")
sel=cache["last_selection"] if cache["last_selection"] and \
     input(f"Reuse {cache['last_selection']}? (y/n): ").lower().startswith('y') \
     else prompt_indices(len(files))
cache["last_selection"]=sel

nick={}
for i in sel:
    d=cache["nicknames"].get(str(i),names[i])
    nick[i]=input(f"Nickname for {names[i]} [{d}]: ").strip() or d
    cache["nicknames"][str(i)]=nick[i]
save_cache(cache)

dfs=[read_tga(files[i]) for i in sel]
cols=dfs[0].columns.tolist()
print("\nVariables:")
for j,c in enumerate(cols): print(f"[{j:2d}] {c}")
x_var=cols[int(input("x‑var index: "))]
y_var=cols[int(input("y‑var index: "))]
title=input(f"Title [{cache.get('title','') or y_var+' vs '+x_var}]: ").strip() or \
      cache.get('title', y_var+' vs '+x_var)
cache["title"]=title; save_cache(cache)

fig,ax=plt.subplots()
text_y,dy=0.90,0.05
col_iter=iter(COLORS)

for df,i in zip(dfs,sel):
    ax.plot(df[x_var],df[y_var],color='black',lw=2,label=f"{nick[i]} data")
    pars=fit_params(df[x_var].values,df[y_var].values)
    c=next(col_iter)
    if pars:
        xg=df[x_var].values
        for p in pars: ax.plot(xg,_g(xg,*p),ls='--',color=c,lw=1.5)
        ax.plot([],[],ls='--',color=c,label=f"{nick[i]} fits")  # legend handle
        lab=", ".join(f"T{k+1} = {t} °C" for k,t in enumerate(centres(pars)))
        ax.text(.02,text_y,lab,transform=ax.transAxes,color=c,
                ha='left',va='top',fontsize=9)
        text_y-=dy

ax.set_xlabel(x_var); ax.set_ylabel(y_var); ax.set_title(title)
for sp in ax.spines.values(): sp.set_linewidth(2)
ax.tick_params(width=2)
ax.legend(loc='upper left')
fig.tight_layout()
plt.show()


#%% #################
#########################################################################
#########################################################################
#########################################################################
#########################################################################



#%% Pressure plots (inlet & outlet)
# ----------------------------------------------------------
csv_path      = r'/Users/rosadrianazelaya/Library/CloudStorage/Box-Box/Halogens/Experiment Data/6-3-25-C2R5-PreCarb-2/pressure-attempt-1.csv'
t_mass_spec   = 0#- 31/60   # h before RGA logging
t_preheat     = 14/60 + 18/3600             # h between RGA start & heat-up
header_lines  = 11
pressure_verticals = vertical_lines.copy()   # reuse or create a new list
# ---------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Load and groom the data ────────────────────────────
df = pd.read_csv(Path(csv_path).expanduser(), skiprows=header_lines)
df.columns = [ 'outlet_time', 'outlet_barg', 'inlet_time', 'inlet_barg']

df['inlet_time']  = pd.to_datetime(df['inlet_time'])
df['outlet_time'] = pd.to_datetime(df['outlet_time'])

df['inlet_h']  = (df['inlet_time']  - df['inlet_time'].iloc[0]).dt.total_seconds() / 3600
df['outlet_h'] = (df['outlet_time'] - df['outlet_time'].iloc[0]).dt.total_seconds() / 3600

shift = t_mass_spec + t_preheat        # align with mass-spec and set t = 0 at heat-up
df['inlet_h']  -= shift
df['outlet_h'] -= shift

# ─── Helper for consistent 2-pt borders/ticks ───────────
def _style(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2)

# ─── Inlet pressure plot ────────────────────────────────

dfP = pd.read_csv(Path(csv_path), skiprows=header_lines,
                  names=['outlet_time','outlet_barg','inlet_time','inlet_barg'])
dfP['inlet_time']  = pd.to_datetime(dfP['inlet_time'])
dfP['outlet_time'] = pd.to_datetime(dfP['outlet_time'])

dfP['inlet_h']  = (dfP['inlet_time']  - dfP['inlet_time'].iloc[0]).dt.total_seconds()/3600
dfP['outlet_h'] = (dfP['outlet_time'] - dfP['outlet_time'].iloc[0]).dt.total_seconds()/3600
shift = t_mass_spec + t_preheat
dfP[['inlet_h','outlet_h']] -= shift

# ─── Inlet pressure ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(dfP['inlet_h'], dfP['inlet_barg'], color='#1c96ea', lw=2)
ax.set_xlabel('Time since reactor heat-up (h)')
ax.set_ylabel('Inlet pressure (barg)')
ax.set_title('Inlet Pressure')
ax.grid(True, ls='--', lw=0.5, alpha=0.5)

for x, lbl in pressure_verticals:
    add_event(ax, x, lbl)

_style(ax)
fig.tight_layout()
plt.show()

# ─── Outlet pressure ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(dfP['outlet_h'], dfP['outlet_barg'], color='#10ba8c', lw=2)
ax.set_xlabel('Time since reactor heat-up (h)')
ax.set_ylabel('Outlet pressure (barg)')
ax.set_title('Outlet Pressure')
ax.grid(True, ls='--', lw=0.5, alpha=0.5)

for x, lbl in pressure_verticals:
    add_event(ax, x, lbl)

_style(ax)
fig.tight_layout()
plt.show()


#%% Pico-log temperatures
# ----------------------------------------------------------
csv_path      = r'/Users/rosadrianazelaya/Library/CloudStorage/Box-Box/Halogens/Experiment Data/6-3-25-C2R5-PreCarb-2/temp-attempt-1.csv'
t_mass_spec   = 0
t_preheat     = 0.08
header_lines  = 0
temp_verticals = vertical_lines.copy()
# ---------------------------------------------------------

dfT = pd.read_csv(Path(csv_path), skiprows=header_lines)

def _first_match(pattern, cols):
    for c in cols:
        if re.search(pattern, c, flags=re.I):
            return c
    return None

time_col = _first_match(r'\btime\b', dfT.columns) or dfT.columns[0]
wb_col   = _first_match(r'(water|bath).*a(?:vg|ve)', dfT.columns)
in_col   = _first_match(r'(inside|reactor).*a(?:vg|ve)', dfT.columns)
if not wb_col or not in_col:
    raise ValueError(f"Temp columns not found – got {dfT.columns.tolist()}")

# convert time → hours
try:
    dfT[time_col] = pd.to_datetime(dfT[time_col])
    dfT['t_h'] = (dfT[time_col] - dfT[time_col].iloc[0]).dt.total_seconds()/3600
except Exception:
    dfT['t_h'] = (dfT[time_col] - dfT[time_col].iloc[0]) / 3600
dfT['t_h'] -= (t_mass_spec + t_preheat)

# ─── Water-bath temp ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(dfT['t_h'], dfT[wb_col], color='#124ff0', lw=2)
ax.set_xlabel('Time since reactor heat-up (h)')
ax.set_ylabel('Water-bath temperature (°C)')
ax.set_title('Water-Bath Temperature')
ax.grid(True, ls='--', lw=0.5, alpha=0.5)

for x, lbl in temp_verticals:
    add_event(ax, x, lbl)

_style(ax)
fig.tight_layout()
plt.show()

# ─── Inside-reactor temp ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(dfT['t_h'], dfT[in_col], color='#f01212', lw=2)
ax.set_xlabel('Time since reactor heat-up (h)')
ax.set_ylabel('Inside-reactor temperature (°C)')
ax.set_title('Inside-Reactor Temperature')
ax.grid(True, ls='--', lw=0.5, alpha=0.5)

for x, lbl in temp_verticals:
    add_event(ax, x, lbl)

_style(ax)
fig.tight_layout()
plt.show()

#%% Temp and pressure

# ─── Inlet pressure ──────────────────────────────────────
figtp, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(dfP['inlet_h'], dfP['inlet_barg'], color='#1c96ea', lw=2)
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Inlet pressure (barg)')
ax1.grid(True, ls='--', lw=0.5, alpha=0.5)
ax1.set_ylim([0,0.3])

ax2 = ax1.twinx()

ax2.plot(dfT['t_h'], dfT[in_col], color='#f01212', lw=1)
ax2.set_ylabel('Inside-reactor temperature (°C)')

for x, lbl in temp_verticals:
    add_event(ax1, x, lbl)

_style(ax2)

_style(ax1)
fig.tight_layout()
plt.show()
