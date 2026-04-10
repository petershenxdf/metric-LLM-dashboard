"""Flask application factory."""
from flask import Flask, render_template
from flask_cors import CORS

from config.config import Config
from app.api import (
    routes_data,
    routes_cluster,
    routes_chat,
    routes_feedback,
    routes_session,
    routes_debug,
    routes_export,
    errors,
)
from app.services.session_service import SessionService
from app.services.pipeline_service import PipelineService
from app.services.chat_service import ChatService
from app.services.feedback_service import FeedbackService
from app.infrastructure.llm.factory import create_llm_client
from app.infrastructure.storage.base import SessionStore
from app.infrastructure.storage.memory_store import InMemorySessionStore
from app.infrastructure.storage.pickle_store import PickleSessionStore
from app.infrastructure.debug.logger import configure_logging, get_logger
from app.infrastructure.debug.debug_recorder import DebugRecorder


def _create_session_store(config: Config) -> SessionStore:
    if config.storage_backend == "pickle":
        return PickleSessionStore(config.pickle_store_path)
    return InMemorySessionStore()


def create_app(config: Config) -> Flask:
    # Configure logging first so every subsystem logs through the same root
    configure_logging(level=config.log_level)
    logger = get_logger("app")
    logger.info("Creating Flask app: llm=%s, storage=%s, debug_dumps=%s",
                config.llm_provider, config.storage_backend, config.debug_dump_enabled)

    app = Flask(
        __name__,
        static_folder="../static",
        template_folder="../templates",
    )
    app.config["SECRET_KEY"] = config.secret_key
    app.config["MAX_CONTENT_LENGTH"] = config.max_upload_size_mb * 1024 * 1024
    app.config["APP_CONFIG"] = config

    CORS(app)

    # Build dependency graph (manual DI container)
    session_store = _create_session_store(config)
    llm_client = create_llm_client(config)
    debug_recorder = DebugRecorder(
        enabled=config.debug_dump_enabled,
        dump_dir=config.debug_dump_dir,
    )

    session_service = SessionService(session_store)
    pipeline_service = PipelineService(session_service, config, debug_recorder=debug_recorder)
    chat_service = ChatService(session_service, llm_client, config)
    feedback_service = FeedbackService(session_service, pipeline_service, config)

    # Attach to app context for routes to access
    app.config["SESSION_SERVICE"] = session_service
    app.config["PIPELINE_SERVICE"] = pipeline_service
    app.config["CHAT_SERVICE"] = chat_service
    app.config["FEEDBACK_SERVICE"] = feedback_service
    app.config["DEBUG_RECORDER"] = debug_recorder

    # Register blueprints
    app.register_blueprint(routes_data.bp)
    app.register_blueprint(routes_cluster.bp)
    app.register_blueprint(routes_chat.bp)
    app.register_blueprint(routes_feedback.bp)
    app.register_blueprint(routes_session.bp)
    app.register_blueprint(routes_debug.bp)
    app.register_blueprint(routes_export.bp)

    errors.register_error_handlers(app)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/health")
    def health():
        return {
            "status": "ok",
            "llm_available": llm_client.is_available(),
            "debug_dumps_enabled": debug_recorder.enabled,
        }

    return app
