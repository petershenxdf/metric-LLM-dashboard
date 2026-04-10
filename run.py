"""Application entry point. Run with: python run.py"""
import os
from app import create_app
from config.config import load_config


def main():
    config = load_config()
    app = create_app(config)
    app.run(
        host=config.host,
        port=config.port,
        debug=config.flask_debug,
    )


if __name__ == "__main__":
    main()
