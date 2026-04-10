"""Structured logger setup for the dashboard.

All modules in the project can get a named logger via:

    from app.infrastructure.debug.logger import get_logger
    logger = get_logger(__name__)   # or a custom name
    logger.info("hello")

The root logger 'ssdbcodi' is configured once at app startup via
configure_logging(level). Log level is controlled by the LOG_LEVEL env var.
"""
import logging
import sys


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure the root 'ssdbcodi' logger. Idempotent -- safe to call twice."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger("ssdbcodi")
    root_logger.setLevel(numeric_level)
    root_logger.propagate = False

    # Avoid duplicate handlers if this module is reloaded
    if root_logger.handlers:
        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    root_logger.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a named logger under the 'ssdbcodi' root.

    Passing a dotted module path like 'ssdbcodi.pipeline' works; passing
    a bare name like 'pipeline' also works -- we prefix with 'ssdbcodi.' if
    the caller forgot.
    """
    if not name.startswith("ssdbcodi"):
        name = f"ssdbcodi.{name}"
    return logging.getLogger(name)
