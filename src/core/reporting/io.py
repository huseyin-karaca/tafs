"""Helpers for reading ``Report`` artifacts from finished MLflow runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow


def download_summary_artifact(
    run_id: str,
    report_name: str,
    tracking_uri: str | None = None,
) -> dict[str, Any] | None:
    """Read ``artifacts/reports/<report_name>/summary.json`` from a parent run.

    Returns ``None`` if the artifact is missing so compilers can render
    placeholder cells instead of crashing on partial runs.
    """
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    artifact_path = f"reports/{report_name}/summary.json"
    try:
        local_path = client.download_artifacts(run_id, artifact_path)
    except Exception:
        return None
    p = Path(local_path)
    if not p.exists():
        return None
    return json.loads(p.read_text())
