# Rosa Plotter

This folder now has a small local plotting package next to the original
`generating_plots.py`. The original file is intentionally left untouched as the
archive/reference version.

## Run

Install dependencies once:

```bash
pip install -r requirements.txt
```

Set local data folders:

```bash
cp config/paths.example.json config/paths.json
```

Then edit `config/paths.json` so the folders point at Rosa's local Box folders.
This file is ignored by git because each computer can have different paths.

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

So yes, the package is meant to work on Rosa's dozens of local files, not just
the small examples in this repo. The examples are here so tests can confirm the
parsers still understand the file formats.

## Add A New Workflow

1. Add a module in `rosa_plots/`, such as `new_analysis.py`.
2. Put the interactive entrypoint in `run_interactive_new_analysis()`.
3. Reuse helpers from `rosa_plots/common.py` for prompts, JSON memory, paths, and plot styling.
4. Register it in `plotter.py`.
5. Add one tiny fixture file and a pytest smoke test.

One-off group-meeting figures can start in `rosa_plots/manual_figures.py`. If
one becomes reused, promote it to its own module.
