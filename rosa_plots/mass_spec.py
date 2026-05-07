"""Mass-spec, pressure, and temperature helper workflows."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import WorkflowError, apply_plot_style, ask, prompt_for_path


def find_elapsed_time_header(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle):
            if line.strip().startswith("Elapsed Time"):
                return idx
    raise WorkflowError(f"Could not find an 'Elapsed Time' header in {path}")


def load_rga_file(path: Path, t_heat_up: float = 0.0) -> pd.DataFrame:
    header_row = find_elapsed_time_header(path)
    df = pd.read_csv(
        path,
        skiprows=header_row,
        header=0,
        sep=",",
        engine="python",
        skipinitialspace=True,
    )
    if "Elapsed Time (s)" not in df.columns:
        raise WorkflowError(f"Expected an 'Elapsed Time (s)' column in {path.name}")
    df["Hours"] = df["Elapsed Time (s)"] / 3600.0 - t_heat_up
    return df


def pressure_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in ("Elapsed Time (s)", "Hours")]


def average_time_windows(df: pd.DataFrame, windows: list[tuple[float, float]]) -> pd.DataFrame:
    cols = pressure_columns(df)
    averages = {}
    for start_h, end_h in windows:
        mask = (df["Hours"] >= start_h) & (df["Hours"] <= end_h)
        averages[f"{start_h:g}-{end_h:g} h"] = df.loc[mask, cols].mean()
    return pd.DataFrame(averages).T


def methane_conversion(
    df: pd.DataFrame,
    *,
    calibration_m: float,
    calibration_b: float,
    starting_h_to_c: float,
) -> pd.Series:
    for col in ("Hydrogen (Torr)", "Methane (Torr)"):
        if col not in df.columns:
            raise WorkflowError(f"Missing required mass-spec column: {col}")
    ms_h_to_c = df["Hydrogen (Torr)"] / df["Methane (Torr)"]
    flow_h_to_c = (ms_h_to_c - calibration_b) / calibration_m
    return (flow_h_to_c - starting_h_to_c) / (flow_h_to_c + 2) * 100


def load_pressure_csv(
    path: Path,
    *,
    header_lines: int,
    t_mass_spec: float,
    t_preheat: float,
    include_outlet: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(path.expanduser(), skiprows=header_lines)
    if include_outlet:
        df.columns = ["outlet_time", "outlet_barg", "inlet_time", "inlet_barg"]
        df["outlet_time"] = pd.to_datetime(df["outlet_time"])
        df["outlet_h"] = (df["outlet_time"] - df["outlet_time"].iloc[0]).dt.total_seconds() / 3600
    else:
        df.columns = ["inlet_time", "inlet_barg", "outlet_time", "outlet_barg"]
    df["inlet_time"] = pd.to_datetime(df["inlet_time"])
    df["inlet_h"] = (df["inlet_time"] - df["inlet_time"].iloc[0]).dt.total_seconds() / 3600
    shift = t_mass_spec + t_preheat
    df["inlet_h"] -= shift
    if include_outlet:
        df["outlet_h"] -= shift
    return df


def load_temperature_csv(path: Path, *, header_lines: int, t_mass_spec: float, t_preheat: float) -> pd.DataFrame:
    df = pd.read_csv(path.expanduser(), skiprows=header_lines)
    time_col = _first_match(df.columns, "time") or df.columns[0]
    wb_col = _first_match(df.columns, "water", "bath")
    in_col = _first_match(df.columns, "inside", "reactor")
    if not wb_col or not in_col:
        raise WorkflowError(f"Could not locate water bath and reactor temperature columns in {path.name}")
    try:
        df[time_col] = pd.to_datetime(df[time_col])
        df["t_h"] = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds() / 3600
    except Exception:
        df["t_h"] = (df[time_col] - df[time_col].iloc[0]) / 3600
    df["t_h"] += t_mass_spec - t_preheat
    df.attrs["time_col"] = time_col
    df.attrs["water_bath_col"] = wb_col
    df.attrs["inside_reactor_col"] = in_col
    return df


def _first_match(columns, *needles: str) -> str | None:
    needles = [needle.lower() for needle in needles]
    for col in columns:
        lowered = str(col).lower()
        if any(needle in lowered for needle in needles):
            return col
    return None


def run_interactive_mass_spec() -> None:
    try:
        path = prompt_for_path("mass_spec_file", "Mass-spec .txt file")
        t_heat_up = float(ask("Heat-up time to subtract, hours", "0"))
        df = load_rga_file(path, t_heat_up=t_heat_up)
        cols = pressure_columns(df)
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in cols:
            ax.plot(df["Hours"], df[col], label=col)
        ax.set_yscale("log", base=10)
        ax.set_xlabel("Hours since start")
        ax.set_ylabel("Partial pressure (Torr)")
        ax.set_title("RGA partial pressures vs. time")
        ax.legend(fontsize="small", ncol=2)
        apply_plot_style(ax)
        fig.tight_layout()
        plt.show()

        if {"Hydrogen (Torr)", "Methane (Torr)"}.issubset(df.columns):
            m = float(ask("Calibration slope m", "4.195034927"))
            b = float(ask("Calibration intercept b", "-0.041611609"))
            h_to_c = float(ask("Starting H:C feed ratio", str(0.1 / 6)))
            df["Methane conversion"] = methane_conversion(
                df,
                calibration_m=m,
                calibration_b=b,
                starting_h_to_c=h_to_c,
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df["Hours"], df["Methane conversion"])
            ax.set_xlabel("Hours since start")
            ax.set_ylabel("% Methane converted")
            apply_plot_style(ax)
            fig.tight_layout()
            plt.show()
    except WorkflowError as exc:
        print(f"\nMass-spec workflow stopped: {exc}")
