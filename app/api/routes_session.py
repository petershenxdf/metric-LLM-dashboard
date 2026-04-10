"""Routes for session lifecycle management."""
from flask import Blueprint, request, jsonify, current_app

from app.api.errors import ValidationError, NotFoundError


bp = Blueprint("session", __name__, url_prefix="/api/session")


@bp.route("/state/<session_id>", methods=["GET"])
def get_session_state(session_id):
    session_service = current_app.config["SESSION_SERVICE"]
    state = session_service.get(session_id)
    if state is None:
        raise NotFoundError(f"Session {session_id} not found")
    return jsonify(state.to_summary_dict())


@bp.route("/reset", methods=["POST"])
def reset_session():
    body = request.get_json() or {}
    session_id = body.get("session_id")
    if not session_id:
        raise ValidationError("Missing session_id")

    session_service = current_app.config["SESSION_SERVICE"]
    session_service.reset(session_id)
    return jsonify({"status": "reset"})


@bp.route("/delete", methods=["POST"])
def delete_session():
    body = request.get_json() or {}
    session_id = body.get("session_id")
    if not session_id:
        raise ValidationError("Missing session_id")

    session_service = current_app.config["SESSION_SERVICE"]
    session_service.delete(session_id)
    return jsonify({"status": "deleted"})
