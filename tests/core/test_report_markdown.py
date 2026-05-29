"""Tests for ``render_markdown_table`` and ``Report.preview.md``."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.core.reporting import Report, render_markdown_table


def test_render_markdown_table_basic_shape() -> None:
    df = pd.DataFrame({"variant": ["full", "no_bridge"], "mase": [0.9995, 0.9976], "n": [10, 12]})
    md = render_markdown_table(df, title="Arch", digits=3)
    lines = md.strip().splitlines()
    assert lines[0] == "# Arch"
    assert lines[2].startswith("| variant | mase | n |")
    assert lines[3].strip() == "| --- | --- | --- |"
    assert "0.999" in lines[4] or "1.000" in lines[4]


def test_render_markdown_handles_nan_and_none() -> None:
    df = pd.DataFrame({"a": [1.0, math.nan, None], "b": ["x", "y", "z"]})
    md = render_markdown_table(df)
    assert md.count("—") == 2


def test_report_writes_preview_md_when_csv_present(tmp_path: Path) -> None:
    df = pd.DataFrame({"variant": ["full"], "mase": [0.95]})
    report = Report(name="my_table", csv=df, md_title="My table")

    captured: dict[str, str] = {}

    def fake_log_artifacts(local_dir: str, artifact_path: str | None = None) -> None:
        layout = Path(local_dir) / "my_table"
        md_path = layout / "preview.md"
        assert md_path.exists(), f"preview.md missing under {layout}"
        captured["md"] = md_path.read_text()
        captured["files"] = ",".join(sorted(p.name for p in layout.iterdir()))

    with patch("src.core.reporting.report.mlflow.log_artifacts", side_effect=fake_log_artifacts):
        report.log_to_mlflow()

    assert "# My table" in captured["md"]
    assert "| variant | mase |" in captured["md"]
    assert "summary.csv" in captured["files"]
    assert "summary.json" in captured["files"]
