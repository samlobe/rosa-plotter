#!/usr/bin/env python3
"""Friendly launcher for Rosa's plotting workflows."""

from rosa_plots.mass_spec import run_interactive_mass_spec
from rosa_plots.raman import run_interactive_raman
from rosa_plots.tga import run_interactive_tga
from rosa_plots.xrd import run_interactive_xrd


WORKFLOWS = {
    "1": ("XRD", run_interactive_xrd),
    "2": ("Raman", run_interactive_raman),
    "3": ("TGA", run_interactive_tga),
    "4": ("Mass Spec", run_interactive_mass_spec),
    "5": ("Pressure/Temperature", run_interactive_mass_spec),
}


def build_menu() -> str:
    lines = ["\nWhat do you want to work on?"]
    for key, (name, _) in WORKFLOWS.items():
        lines.append(f"  {key}. {name}")
    lines.append("  x. Exit")
    return "\n".join(lines)


def main() -> None:
    while True:
        print(build_menu())
        choice = input("> ").strip().lower()
        if choice in {"x", "q", "quit", "exit"}:
            print("Bye.")
            return
        workflow = WORKFLOWS.get(choice)
        if workflow is None:
            print("Please choose one of the menu numbers, or x to exit.")
            continue
        _, runner = workflow
        runner()


if __name__ == "__main__":
    main()
