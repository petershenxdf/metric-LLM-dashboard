"""Central configuration. All tunable parameters live here.

Reads from environment variables (loaded from .env) with sensible defaults.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # Flask
    secret_key: str = "dev-secret-key"
    flask_debug: bool = True
    host: str = "127.0.0.1"
    port: int = 5000

    # LLM
    llm_provider: str = "ollama"          # ollama | openai
    llm_model: str = "mistral-small3.1:latest"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""
    llm_temperature: float = 0.1
    llm_timeout: int = 60

    # Storage
    storage_backend: str = "memory"       # memory | pickle
    pickle_store_path: str = "./data/sessions"

    # Algorithm defaults
    ssdbcodi_min_pts: int = 3
    ssdbcodi_alpha: float = 0.4
    ssdbcodi_beta: float = 0.4
    ssdbcodi_k_outliers: int = 10
    itml_gamma: float = 1.0
    triplet_lr: float = 0.01

    # Upload
    max_upload_size_mb: int = 20
    upload_folder: str = "./data/uploads"

    # Debug / logging
    debug_dump_enabled: bool = False
    debug_dump_dir: str = "./data/debug"
    log_level: str = "INFO"

    # Paths derived from project root
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    samples_folder: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "samples")
    prompts_folder: Path = field(default_factory=lambda: PROJECT_ROOT / "config" / "prompts")


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


def load_config() -> Config:
    """Load configuration from .env (if present) and environment variables."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    cfg = Config(
        secret_key=os.getenv("SECRET_KEY", "dev-secret-key"),
        flask_debug=_bool("FLASK_DEBUG", True),
        host=os.getenv("HOST", "127.0.0.1"),
        port=_int("PORT", 5000),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        llm_model=os.getenv("LLM_MODEL", "mistral-small3.1:latest"),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_temperature=_float("LLM_TEMPERATURE", 0.1),
        llm_timeout=_int("LLM_TIMEOUT", 60),
        storage_backend=os.getenv("STORAGE_BACKEND", "memory"),
        pickle_store_path=os.getenv("PICKLE_STORE_PATH", "./data/sessions"),
        ssdbcodi_min_pts=_int("SSDBCODI_MIN_PTS", 3),
        ssdbcodi_alpha=_float("SSDBCODI_ALPHA", 0.4),
        ssdbcodi_beta=_float("SSDBCODI_BETA", 0.4),
        ssdbcodi_k_outliers=_int("SSDBCODI_K_OUTLIERS", 10),
        itml_gamma=_float("ITML_GAMMA", 1.0),
        triplet_lr=_float("TRIPLET_LR", 0.01),
        max_upload_size_mb=_int("MAX_UPLOAD_SIZE_MB", 20),
        upload_folder=os.getenv("UPLOAD_FOLDER", "./data/uploads"),
        debug_dump_enabled=_bool("DEBUG_DUMP_ENABLED", False),
        debug_dump_dir=os.getenv("DEBUG_DUMP_DIR", "./data/debug"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

    # Make sure folders exist
    Path(cfg.upload_folder).mkdir(parents=True, exist_ok=True)
    if cfg.debug_dump_enabled:
        Path(cfg.debug_dump_dir).mkdir(parents=True, exist_ok=True)
    if cfg.storage_backend == "pickle":
        Path(cfg.pickle_store_path).mkdir(parents=True, exist_ok=True)

    return cfg
