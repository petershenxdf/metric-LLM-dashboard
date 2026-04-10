"""Routes for submitting structured constraints (and undo)."""
from flask import Blueprint, request, jsonify, current_app

from app.api.errors import ValidationError, NotFoundError
from app.domain.constraints.schemas import constraint_from_dict


bp = Blueprint("feedback", __name__, url_prefix="/api/feedback")


@bp.route("/submit", methods=["POST"])
def submit_constraint():
    body = request.get_json() or {}
    session_id = body.get("session_id")
    constraint_dict = body.get("constraint")

    if not session_id:
        raise ValidationError("Missing session_id")
    if not constraint_dict:
        raise ValidationError("Missing constraint")

    try:
        constraint = constraint_from_dict(constraint_dict)
    except Exception as e:
        raise ValidationError(f"Invalid constraint: {e}")

    feedback = current_app.config["FEEDBACK_SERVICE"]
    result = feedback.apply_constraint(session_id, constraint)
    return jsonify(result)


@bp.route("/undo", methods=["POST"])
def undo_last():
    body = request.get_json() or {}
    session_id = body.get("session_id")
    if not session_id:
        raise ValidationError("Missing session_id")

    feedback = current_app.config["FEEDBACK_SERVICE"]
    result = feedback.undo_last(session_id)
    return jsonify(result)


@bp.route("/list/<session_id>", methods=["GET"])
def list_constraints(session_id):
    session_service = current_app.config["SESSION_SERVICE"]
    state = session_service.get(session_id)
    if state is None:
        raise NotFoundError(f"Session {session_id} not found")

    return jsonify({
        "constraints": [c.to_dict() for c in state.constraints_history]
    })
