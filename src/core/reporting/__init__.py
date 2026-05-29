"""Shared reporting abstractions.

Concrete experiments produce ``Report`` instances in
``produce_reports()``; ``BaseExperiment.on_parent_close`` logs them as
MLflow artifacts under ``reports/<name>/`` on the parent run. The
post-hoc ``compilers`` package then stitches summaries from multiple
parent runs into publication-ready merged tables without re-deriving
any numbers.
"""

from src.core.reporting.io import download_summary_artifact
from src.core.reporting.report import Report, render_markdown_table

__all__ = ["Report", "download_summary_artifact", "render_markdown_table"]
