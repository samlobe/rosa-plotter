from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from rosa_plots.common import PROJECT_ROOT, WorkflowError, load_json, save_json
from rosa_plots.raman import load_raman_spectrum
from rosa_plots.tga import parse_tga_header, read_text_lines, read_tga_file
from rosa_plots.xrd import load_xrd_memory, load_xrd_samples


def test_raman_loader_reads_two_numeric_columns():
    df = load_raman_spectrum(PROJECT_ROOT / "C5R3-SF-2.txt")

    assert list(df.columns) == ["Wavenumber", "Intensity"]
    assert len(df) > 10
    assert df["Wavenumber"].is_monotonic_increasing


def test_tga_loader_reads_sig_header_and_numeric_data():
    path = PROJECT_ROOT / "C3R6_loose_fines.txt"
    lines = read_text_lines(path)
    columns, start_idx = parse_tga_header(lines, path)
    df = read_tga_file(path)

    assert start_idx > 0
    assert columns[:3] == ["Time (min)", "Temperature (°C)", "Weight (mg)"]
    assert "Deriv. Weight (%/°C)" in df.columns
    assert len(df) > 10
    assert df["Temperature (°C)"].notna().any()


def test_xrd_loader_reads_xls_when_xlrd_is_available():
    if importlib.util.find_spec("xlrd") is None:
        pytest.skip("xlrd is not installed in this environment")

    samples = load_xrd_samples(PROJECT_ROOT)

    assert samples
    name, df = next(iter(samples.items()))
    assert name
    assert list(df.columns) == ["Degree", "Intensity"]
    assert len(df) > 10


def test_xrd_loader_gives_clear_xlrd_message_when_missing():
    if importlib.util.find_spec("xlrd") is not None:
        pytest.skip("xlrd is installed in this environment")

    with pytest.raises(WorkflowError, match="needs xlrd"):
        load_xrd_samples(PROJECT_ROOT)


def test_xrd_memory_preserves_existing_memory_shape():
    memory = load_xrd_memory()

    assert "indices" in memory
    assert "samples" in memory
    assert "plot_title" in memory
    assert "C5R3-SF" in memory["samples"]


def test_json_round_trip(tmp_path):
    path = tmp_path / "memory.json"
    data = {"indices": [1, 2], "samples": {"sample": {"offset": 1.0}}}
    save_json(path, data)

    assert load_json(path) == data
