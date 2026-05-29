"""``Report`` dataclass — one manuscript output owned by one experiment.

Each ``Report`` is logged as a directory of artifacts under the
currently active MLflow run at the path:

    artifacts/reports/<name>/
        summary.json    canonical structured payload (machine-readable)
        preview.tex     optional single-tabular LaTeX block
        preview.md      preview rendered from ``csv`` as a markdown
                        table; MLflow's web UI renders this inline so
                        ablation tables are visible without download
        preview.png     optional rendered figure
        summary.csv     optional flat view for spreadsheets

The ``summary.json`` is the source of truth for downstream compilers;
the other files are preview material for the MLflow UI and manual
inspection during research.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import mlflow
import pandas as pd


@dataclass(frozen=True)
class Report:
    """One self-contained manuscript output (table or figure)."""

    name: str
    summary: dict[str, Any] = field(default_factory=dict)
    rendered_tex: str | None = None
    png_bytes: bytes | None = None
    csv: pd.DataFrame | None = None
    extra_files: dict[str, bytes] = field(default_factory=dict)
    md_title: str | None = None
    md_digits: int = 4

    def log_to_mlflow(self) -> None:
        """Write the report to the currently active MLflow run.

        Layout (binding contract; downstream compilers depend on it):

            artifacts/reports/<name>/summary.json
            artifacts/reports/<name>/preview.tex   (if rendered_tex set)
            artifacts/reports/<name>/preview.md    (if csv set)
            artifacts/reports/<name>/preview.png   (if png_bytes set)
            artifacts/reports/<name>/summary.csv   (if csv set)
            artifacts/reports/<name>/<filename>    (per entry in extra_files)
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / self.name
            tmp_path.mkdir(parents=True, exist_ok=True)
            (tmp_path / "summary.json").write_text(
                json.dumps(self.summary, indent=2, default=_json_default)
            )
            if self.rendered_tex is not None:
                (tmp_path / "preview.tex").write_text(self.rendered_tex)
            if self.png_bytes is not None:
                (tmp_path / "preview.png").write_bytes(self.png_bytes)
            if self.csv is not None:
                self.csv.to_csv(tmp_path / "summary.csv", index=False)
                (tmp_path / "preview.md").write_text(
                    render_markdown_table(
                        self.csv, title=self.md_title or self.name, digits=self.md_digits
                    )
                )
            for fname, blob in self.extra_files.items():
                (tmp_path / fname).write_bytes(blob)
            mlflow.log_artifacts(str(tmp_path.parent), artifact_path="reports")


def render_markdown_table(df: pd.DataFrame, *, title: str | None = None, digits: int = 4) -> str:
    """Render a ``DataFrame`` as a GitHub-flavoured markdown table.

    Used by :class:`Report` to produce ``preview.md`` artifacts that
    MLflow renders inline in the UI. Numeric cells are rounded to
    ``digits``; ``NaN`` is displayed as ``—``.
    """
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    align = "| " + " | ".join(["---"] * len(cols)) + " |"
    body: list[str] = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(_fmt_cell(v, digits=digits) for v in row.tolist()) + " |")
    lines: list[str] = []
    if title:
        lines.extend([f"# {title}", ""])
    lines.extend([header, align, *body])
    return "\n".join(lines) + "\n"


def _fmt_cell(value: Any, *, digits: int) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        return f"{value:.{digits}f}"
    if isinstance(value, (int,)):
        return str(value)
    return str(value)


def _json_default(obj: Any) -> Any:
    """Best-effort JSON encoder for numpy / pandas scalars in summaries."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"unserialisable type: {type(obj).__name__}")
