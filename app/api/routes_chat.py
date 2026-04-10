"""Routes for the chatbox interaction."""
from flask import Blueprint, request, jsonify, current_app

from app.api.errors import ValidationError, NotFoundError


bp = Blueprint("chat", __name__, url_prefix="/api/chat")


@bp.route("/message", methods=["POST"])
def send_message():
    body = request.get_json() or {}
    session_id = body.get("session_id")
    user_text = body.get("text", "").strip()
    selected_ids = body.get("selected_ids", [])

    if not session_id:
        raise ValidationError("Missing session_id")
    if not user_text:
        raise ValidationError("Empty message")

    chat_service = current_app.config["CHAT_SERVICE"]
    result = chat_service.process_message(session_id, user_text, selected_ids)
    return jsonify(result)


@bp.route("/history/<session_id>", methods=["GET"])
def get_history(session_id):
    session_service = current_app.config["SESSION_SERVICE"]
    state = session_service.get(session_id)
    if state is None:
        raise NotFoundError(f"Session {session_id} not found")
    return jsonify({"history": state.chat_history})
