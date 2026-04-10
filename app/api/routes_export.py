"""Routes for exporting finished datasets with cluster / outlier labels.

When the user is done interacting with the dashboard, they can hit
GET /api/export/csv/<session_id> to download a CSV containing:
    - all original numeric columns (pre-normalization)
    - cluster_label      (integer; -1 for outliers)
    - is_outlier         (boolean)
    - rscore, lscore, simscore, tscore  (optional, controlled by ?include_scores)

The download triggers a browser file-save dialog via Content-Disposition.
"""
from io import StringIO

from flask import Blueprint, request, Response, current_app
import numpy as np
import pandas as pd

from app.api.errors import NotFoundError, ValidationError


bp = Blueprint("export", __name__, url_prefix="/api/export")


@bp.route("/csv/<session_id>", methods=["GET"])
def export_csv(session_id):
    """Download the labeled dataset as a CSV file."""
    session_service = current_app.config["SESSION_SERVICE"]
    state = session_service.get(session_id)
    if state is None:
        raise NotFoundError(f"Session {session_id} not found")

    if state.current_clusters is None:
        raise ValidationError(
            "Cannot export: no clustering results available. "
            "Run clustering at least once before exporting."
        )

    # Optional scores column (default: include them)
    include_scores = request.args.get("include_scores", "true").lower() == "true"

    # Start from the raw (pre-normalized) dataset if we have it; fall back to
    # the normalized dataset for older sessions that predate the raw field.
    if state.raw_dataset is not None:
        base_df = state.raw_dataset.copy()
    else:
        base_df = state.dataset.copy()

    # Append cluster and outlier labels
    base_df["cluster_label"] = state.current_clusters.astype(int)
    base_df["is_outlier"] = state.current_outliers.astype(bool)

    # Append scores if requested and available
    if include_scores and state.current_scores:
        for key in ("rscore", "lscore", "simscore", "tscore"):
            arr = state.current_scores.get(key)
            if arr is not None:
                base_df[key] = np.asarray(arr)

    # Build CSV body
    buf = StringIO()
    base_df.to_csv(buf, index_label="point_id")
    csv_body = buf.getvalue()

    # Build a sensible download filename
    stem = state.source_filename or "dataset"
    if stem.endswith(".csv"):
        stem = stem[:-4]
    download_name = f"{stem}_labeled.csv"

    return Response(
        csv_body,
        mimetype="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{download_name}"',
            "Cache-Control": "no-store",
        },
    )


@bp.route("/summary/<session_id>", methods=["GET"])
def export_summary(session_id):
    """Return a JSON summary of what the export will contain (for the UI)."""
    session_service = current_app.config["SESSION_SERVICE"]
    state = session_service.get(session_id)
    if state is None:
        raise NotFoundError(f"Session {session_id} not found")

    ready = state.current_clusters is not None
    n_clusters = 0
    if ready and len(state.current_clusters) > 0:
        valid = state.current_clusters[state.current_clusters >= 0]
        n_clusters = int(max(valid) + 1) if len(valid) > 0 else 0

    base_df = state.raw_dataset if state.raw_dataset is not None else state.dataset
    return {
        "ready": ready,
        "n_rows": len(base_df),
        "n_original_columns": len(base_df.columns),
        "columns": list(base_df.columns) + ["cluster_label", "is_outlier"],
        "n_clusters": n_clusters,
        "n_outliers": int(sum(state.current_outliers)) if state.current_outliers is not None else 0,
        "n_constraints_applied": len(state.constraints_history),
    }
