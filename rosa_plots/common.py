"""Shared helpers for the plotting workflows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "paths.json"


def natural_sort_key(value: object) -> list[object]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    ]


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {} if default is None else default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def load_paths() -> dict[str, str]:
    data = load_json(CONFIG_PATH, default={})
    return data if isinstance(data, dict) else {}


def save_paths(data: dict[str, str]) -> None:
    save_json(CONFIG_PATH, data)


def ask(question: str, default: str | None = None) -> str:
    if default not in (None, ""):
        print(f"\n>>> {question} [{default}]", flush=True)
    else:
        print(f"\n>>> {question}", flush=True)
    answer = input("> ").strip()
    return answer if answer else (default or "")


def ask_yes_no(question: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    answer = ask(f"{question} ({suffix})").strip().lower()
    if not answer:
        return default
    return answer.startswith("y")


def parse_indices(text: str, max_count: int | None = None) -> list[int]:
    indices = [int(part.strip()) for part in text.split(",") if part.strip()]
    if max_count is not None:
        bad = [idx for idx in indices if idx < 0 or idx >= max_count]
        if bad:
            raise ValueError(f"Out-of-range index/indices: {bad}")
    return indices


def print_numbered(items: Iterable[str]) -> None:
    for idx, item in enumerate(items):
        print(f"{idx}: {item}")


def prompt_for_path(
    key: str,
    prompt: str,
    *,
    must_exist: bool = True,
    default: str | None = None,
) -> Path:
    paths = load_paths()
    current = default or paths.get(key, "")
    while True:
        answer = ask(prompt, current)
        path = Path(answer).expanduser()
        if not must_exist or path.exists():
            paths[key] = str(path)
            save_paths(paths)
            return path
        print(f"That path does not exist: {path}")


def apply_plot_style(ax: Any) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2)


class WorkflowError(RuntimeError):
    """Error shown cleanly to the user during an interactive workflow."""
