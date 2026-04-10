"""Routes for inspecting debug dumps.

These endpoints let you look at dumped iterations without stopping the app.
Useful for a quick sanity check from the browser, or for feeding data into
a Jupyter notebook.
"""
import base64
import io

import numpy as np
from flask import Blueprint, jsonify, current_app

from app.api.errors import NotFoundError, ValidationError


bp = Blueprint("debug", __name__, url_prefix="/api/debug")


@bp.route("/iterations/<session_id>", methods=["GET"])
def list_iterations(session_id):
    """List all dumped iterations for a session, with their meta info."""
    recorder = current_app.config.get("DEBUG_RECORDER")
    if recorder is None or not recorder.enabled:
        return jsonify({
            "enabled": False,
            "iterations": [],
            "message": "Debug dumps are disabled. Set DEBUG_DUMP_ENABLED=true in .env to enable.",
        })

    iterations = recorder.list_iterations(session_id)
    return jsonify({
        "enabled": True,
        "session_id": session_id,
        "n_iterations": len(iterations),
        "iterations": iterations,
    })


@bp.route("/iteration/<session_id>/<int:iter_num>", methods=["GET"])
def get_iteration(session_id, iter_num):
    """Return a full iteration payload, with numpy arrays base64-encoded.

    The arrays are returned as base64-encoded npy blobs so the payload stays
    valid JSON. Clients that want raw arrays should use debug_tools.load_iteration
    on the file system instead.
    """
    recorder = current_app.config.get("DEBUG_RECORDER")
    if recorder is None or not recorder.enabled:
        raise ValidationError("Debug dumps are disabled")

    try:
        payload = recorder.load_iteration(session_id, iter_num)
    except FileNotFoundError:
        raise NotFoundError(f"Iteration {iter_num} not found for session {session_id}")

    # Encode numpy arrays as base64 npy blobs for JSON transport
    encoded = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            encoded[key] = {
                "__ndarray__": True,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data_b64": _ndarray_to_b64(value),
            }
        elif isinstance(value, dict) and all(isinstance(v, np.ndarray) for v in value.values()):
            encoded[key] = {
                k: {
                    "__ndarray__": True,
                    "dtype": str(v.dtype),
                    "shape": list(v.shape),
                    "data_b64": _ndarray_to_b64(v),
                }
                for k, v in value.items()
            }
        else:
            encoded[key] = value

    return jsonify(encoded)


@bp.route("/clear/<session_id>", methods=["POST"])
def clear_session_dumps(session_id):
    recorder = current_app.config.get("DEBUG_RECORDER")
    if recorder is None:
        return jsonify({"status": "no_recorder"})
    recorder.clear_session(session_id)
    return jsonify({"status": "cleared"})


def _ndarray_to_b64(arr: np.ndarray) -> str:
    """Serialize a numpy array to a base64 string (npy format)."""
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")
