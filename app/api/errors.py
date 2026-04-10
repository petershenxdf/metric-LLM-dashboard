"""Centralized error handling for all routes."""
from flask import jsonify
import traceback


class APIError(Exception):
    def __init__(self, message: str, status_code: int = 400, payload: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload or {}

    def to_dict(self):
        return {"error": self.message, **self.payload}


class NotFoundError(APIError):
    def __init__(self, message: str = "Not found"):
        super().__init__(message, status_code=404)


class ValidationError(APIError):
    def __init__(self, message: str, payload: dict = None):
        super().__init__(message, status_code=400, payload=payload)


def register_error_handlers(app):
    @app.errorhandler(APIError)
    def handle_api_error(err):
        return jsonify(err.to_dict()), err.status_code

    @app.errorhandler(404)
    def handle_404(err):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(413)
    def handle_413(err):
        return jsonify({"error": "Uploaded file too large"}), 413

    @app.errorhandler(Exception)
    def handle_unexpected(err):
        if app.debug:
            traceback.print_exc()
        return jsonify({"error": str(err), "type": type(err).__name__}), 500
