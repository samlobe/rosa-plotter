# Research Plotter

This folder contains a small local plotting package for prompt-driven research
figure workflows.

## Get The Code

There are two easy ways to get this project from GitHub.

### Option 1: Download ZIP

This is the simplest if you are new to GitHub.

1. Open the GitHub repository page in your browser.
2. Click the green **Code** button.
3. Click **Download ZIP**.
4. Unzip the downloaded file.
5. Open the unzipped folder in VSCode.

This gives you the files, but it does not set up git syncing. If a version is updated later, you can download the ZIP again.

### Option 2: Clone From The Terminal

If git is already installed:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

## Run

Install dependencies once:

```bash
pip install -r requirements.txt
```

Set local data folders:

```bash
cp config/paths.example.json config/paths.json
```

Then edit `config/paths.json` so the folders point at the local data folders.
This file is ignored by git because each computer can have different local paths.

Then start the prompt menu:

```bash
python plotter.py
```

## Where Data Comes From

The workflows read folder defaults from `config/paths.json`. For example, XRD
uses `xrd_data`, Raman uses `raman_data`, and TGA uses `tga_data`.

If a configured folder does not exist, the workflow asks for the correct folder
and saves the answer back to `config/paths.json` for next time. Once the folder
is correct, the loaders scan all matching files in that folder:

- XRD: `.csv`, `.xls`, `.xlsx`
- Raman: `.txt`
- TGA: `.txt` files in the TA Instruments `Sig...` / `StartOfData` format

The package is meant to scan normal local data folders, not only the small
example files in this repo. The examples are here so tests can confirm the
parsers still understand the file formats.

## Add A New Workflow

1. Add a module in `rosa_plots/`, such as `new_analysis.py`.
2. Put the interactive entrypoint in `run_interactive_new_analysis()`.
3. Reuse helpers from `rosa_plots/common.py` for prompts, JSON memory, paths, and plot styling.
4. Register it in `plotter.py`.
5. Optional but helpful: add one tiny fixture file and a pytest smoke test.

One-off group-meeting figures can start in `rosa_plots/manual_figures.py`. If
one becomes reused, promote it to its own module.

### Example: Add A Pressure + Temperature Plot

Imagine the same group-meeting figure keeps coming up: inlet pressure and
inside-reactor temperature on the same time axis. Once that figure becomes
reused, it can become its own workflow.

First, make a new file:

```text
rosa_plots/pressure_temperature.py
```

Inside that file, the main function should be named something obvious:

```python
from pathlib import Path

import matplotlib.pyplot as plt

from rosa_plots.common import apply_plot_style, ask, prompt_for_path
from rosa_plots.mass_spec import load_pressure_csv, load_temperature_csv


def run_interactive_pressure_temperature():
    pressure_path = prompt_for_path("pressure_file", "Pressure CSV file")
    temp_path = prompt_for_path("temperature_file", "Temperature CSV file")

    pressure_header_lines = int(ask("Pressure header lines", "10"))
    temp_header_lines = int(ask("Temperature header lines", "0"))
    t_mass_spec = float(ask("Mass-spec start offset, hours", "0"))
    t_preheat = float(ask("Preheat offset, hours", "0"))

    pressure = load_pressure_csv(
        Path(pressure_path),
        header_lines=pressure_header_lines,
        t_mass_spec=t_mass_spec,
        t_preheat=t_preheat,
    )
    temp = load_temperature_csv(
        Path(temp_path),
        header_lines=temp_header_lines,
        t_mass_spec=t_mass_spec,
        t_preheat=t_preheat,
    )

    inside_col = temp.attrs["inside_reactor_col"]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(pressure["inlet_h"], pressure["inlet_barg"], color="#1c96ea")
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Inlet pressure (barg)")

    ax2 = ax1.twinx()
    ax2.plot(temp["t_h"], temp[inside_col], color="#f01212")
    ax2.set_ylabel("Inside-reactor temperature (deg C)")

    apply_plot_style(ax1)
    apply_plot_style(ax2)
    fig.tight_layout()
    plt.show()
```

Then add it to `plotter.py`:

```python
from rosa_plots.pressure_temperature import run_interactive_pressure_temperature

WORKFLOWS = {
    "1": ("XRD", run_interactive_xrd),
    "2": ("Raman", run_interactive_raman),
    "3": ("TGA", run_interactive_tga),
    "4": ("Mass Spec", run_interactive_mass_spec),
    "5": ("Pressure/Temperature", run_interactive_pressure_temperature),
}
```

That is all “add a module” means: make a new Python file for one reusable
recipe, give it one function the menu can call, and add that function to the
menu.

Tests are optional for quick plots, but useful for workflows that parse a file
format you do not want to accidentally break. A test can stay tiny: with one
small pressure CSV and one small temperature CSV in the repo, it could check
that both files load and produce the expected time columns. It does not need to
inspect whether the plot looks perfect.
