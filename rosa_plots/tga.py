"""TGA parsing, smoothing helpers, plotting, and interactive prompts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


COLORS = ["#000000", "#7E1FD1", "#26C2FF", "#E16462", "#FFB000", "#6F728C", "#00A86A"]


def read_text_lines(path: Path) -> list[str]:
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw[:100] else "utf-8"
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="ignore")
    return text.splitlines()


def parse_tga_header(lines: list[str], path: Path) -> tuple[list[str], int]:
    sig_names: list[str] = []
    start = None
    sig_pat = re.compile(r"^Sig\d+\t(?P<name>.+)")
    for idx, raw in enumerate(lines):
        line = raw.lstrip("\ufeff")
        match = sig_pat.match(line)
        if match:
            sig_names.append(match.group("name").strip())
        elif line.strip().lower().replace(" ", "") == "startofdata":
            start = idx + 1
            break
    if start is None:
        raise WorkflowError(f"StartOfData not found in {path.name}")
    if not sig_names:
        raise WorkflowError(f"No Sig columns found in {path.name}")
    return sig_names, start


def read_tga_file(path: Path) -> pd.DataFrame:
    lines = read_text_lines(path)
    cols, start_idx = parse_tga_header(lines, path)
    data = [line.split("\t") for line in lines[start_idx:] if line.strip()]
    return pd.DataFrame(data, columns=cols).apply(pd.to_numeric, errors="coerce")


def find_tga_files(data_folder: Path) -> list[Path]:
    if not data_folder.exists():
        raise WorkflowError(f"TGA folder does not exist: {data_folder}")
    return sorted(data_folder.rglob("*.txt"), key=lambda p: natural_sort_key(p.stem))


def rolling_average(arr: np.ndarray, half_win: int) -> np.ndarray:
    out = np.empty(len(arr))
    for idx in range(len(arr)):
        lo = max(0, idx - half_win)
        hi = min(len(arr), idx + half_win + 1)
        out[idx] = arr[lo:hi].mean()
    return out


def fix_temperature_monotonic(T: np.ndarray) -> np.ndarray:
    T = T.copy()
    idx = 1
    while idx < len(T):
        if T[idx] <= T[idx - 1]:
            start = idx - 1
            end = idx + 1
            while end < len(T) - 1:
                if T[end] > T[start] and T[end + 1] > T[end]:
                    break
                end += 1
            end = min(end, len(T) - 1)
            T[start : end + 1] = np.linspace(T[start], T[end], end - start + 1)
            idx = end + 1
        else:
            idx += 1
    return T


def add_tga_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    temp_col = _first_col(df, "Temperature")
    weight_col = _first_col(df, "Weight (mg)")
    deriv_col = _first_col(df, "Deriv. Weight")
    if temp_col is None:
        return df
    T = df[temp_col].to_numpy()
    if deriv_col is not None:
        df["2DTG (d2Weight/dT2)"] = np.gradient(df[deriv_col].to_numpy(), T, edge_order=2)
    elif weight_col is not None:
        df["DTG (dWeight/dT)"] = np.gradient(df[weight_col].to_numpy(), T, edge_order=2)
        df["2DTG (d2Weight/dT2)"] = np.gradient(df["DTG (dWeight/dT)"].to_numpy(), T, edge_order=2)
    return df


def _first_col(df: pd.DataFrame, needle: str) -> str | None:
    needle = needle.lower()
    for col in df.columns:
        if needle in col.lower():
            return col
    return None


def run_interactive_tga() -> None:
    try:
        data_folder = prompt_for_path("tga_data", "TGA data folder")
        files = find_tga_files(data_folder)
        if not files:
            raise WorkflowError(f"No TGA .txt files found in {data_folder}")
        cache_path = data_folder / ".tga_cache.json"
        cache = load_json(cache_path, default={})
        names = [path.stem for path in files]
        print("\nAvailable TGA samples:")
        print_numbered(names)
        previous = cache.get("last_selection") or []
        selected = ask("Indices to plot, comma separated, or x to reuse previous", ",".join(map(str, previous)))
        indices = previous if selected.lower() == "x" and previous else parse_indices(selected, len(files))
        cache["last_selection"] = indices

        dfs = {names[idx]: add_tga_derived_columns(read_tga_file(files[idx])) for idx in indices}
        columns = list(next(iter(dfs.values())).columns)
        print("\nVariables:")
        print_numbered(columns)
        x_idx = int(ask("x-var index", "1"))
        y_idx = int(ask("y-var index", "2"))
        x_var = columns[x_idx]
        y_var = columns[y_idx]
        title = ask("Title", cache.get("title") or f"{y_var} vs {x_var}")
        cache["title"] = title

        fig, ax = plt.subplots()
        nicknames = cache.setdefault("nicknames", {})
        for plot_idx, sample_idx in enumerate(indices):
            name = names[sample_idx]
            default_nick = nicknames.get(name, name)
            nickname = ask(f"Nickname for {name}", default_nick)
            nicknames[name] = nickname
            df = dfs[name].copy()
            y_label = y_var
            if y_var.lower().startswith("weight"):
                df[y_var] = df[y_var] / df[y_var].iloc[0] * 100
                y_label = "Weight Percent (%)"
            ax.plot(df[x_var], df[y_var], color=COLORS[plot_idx % len(COLORS)], lw=2, label=nickname)

        ax.set_xlabel(x_var)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend(loc="best")
        apply_plot_style(ax)
        fig.tight_layout()
        plt.show()
        save_json(cache_path, cache)
    except WorkflowError as exc:
        print(f"\nTGA workflow stopped: {exc}")
