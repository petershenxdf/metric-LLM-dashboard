"""Routes for data upload and sample management."""
import os
import uuid
from pathlib import Path

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from app.api.errors import ValidationError, NotFoundError
from app.infrastructure.data.factory import get_loader_for_file


bp = Blueprint("data", __name__, url_prefix="/api/data")


@bp.route("/upload", methods=["POST"])
def upload_dataset():
    """Upload a CSV (or other supported file). Creates a new session."""
    if "file" not in request.files:
        raise ValidationError("No file provided")
    file = request.files["file"]
    if not file.filename:
        raise ValidationError("Empty filename")

    config = current_app.config["APP_CONFIG"]
    session_service = current_app.config["SESSION_SERVICE"]

    filename = secure_filename(file.filename)
    save_path = Path(config.upload_folder) / f"{uuid.uuid4().hex}_{filename}"
    file.save(save_path)

    try:
        loader = get_loader_for_file(str(save_path))
        raw_df, df = loader.load_both(str(save_path))
        warnings = loader.validate(df)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise ValidationError(f"Failed to load file: {e}")

    session_id = session_service.create_session(
        dataset=df,
        source_filename=filename,
        raw_dataset=raw_df,
    )

    return jsonify({
        "session_id": session_id,
        "n_points": len(df),
        "n_features": df.shape[1],
        "feature_names": df.columns.tolist(),
        "warnings": warnings,
    })


@bp.route("/samples", methods=["GET"])
def list_sample_datasets():
    config = current_app.config["APP_CONFIG"]
    samples_dir = config.samples_folder
    if not samples_dir.exists():
        return jsonify({"samples": []})
    samples = []
    for f in sorted(samples_dir.iterdir()):
        if f.suffix.lower() in (".csv",):
            samples.append({"name": f.stem, "filename": f.name})
    return jsonify({"samples": samples})


@bp.route("/load_sample", methods=["POST"])
def load_sample():
    body = request.get_json() or {}
    filename = body.get("filename")
    if not filename:
        raise ValidationError("Missing filename")

    config = current_app.config["APP_CONFIG"]
    session_service = current_app.config["SESSION_SERVICE"]

    sample_path = config.samples_folder / secure_filename(filename)
    if not sample_path.exists():
        raise NotFoundError(f"Sample {filename} not found")

    loader = get_loader_for_file(str(sample_path))
    raw_df, df = loader.load_both(str(sample_path))
    warnings = loader.validate(df)

    session_id = session_service.create_session(
        dataset=df,
        source_filename=filename,
        raw_dataset=raw_df,
    )

    return jsonify({
        "session_id": session_id,
        "n_points": len(df),
        "n_features": df.shape[1],
        "feature_names": df.columns.tolist(),
        "warnings": warnings,
    })


@bp.route("/info/<session_id>", methods=["GET"])
def get_dataset_info(session_id):
    session_service = current_app.config["SESSION_SERVICE"]
    state = session_service.get(session_id)
    if state is None:
        raise NotFoundError(f"Session {session_id} not found")
    return jsonify({
        "session_id": session_id,
        "n_points": len(state.dataset),
        "n_features": state.dataset.shape[1],
        "feature_names": state.dataset.columns.tolist(),
        "source_filename": state.source_filename,
    })
