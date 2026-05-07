from __future__ import annotations

import importlib

import plotter


def test_import_workflow_modules():
    for module_name in [
        "rosa_plots.common",
        "rosa_plots.xrd",
        "rosa_plots.raman",
        "rosa_plots.tga",
        "rosa_plots.mass_spec",
        "rosa_plots.manual_figures",
    ]:
        importlib.import_module(module_name)


def test_launcher_menu_is_built_without_running_workflows():
    menu = plotter.build_menu()

    assert "XRD" in menu
    assert "Raman" in menu
    assert "TGA" in menu
    assert "Mass Spec" in menu
    assert "Exit" in menu
