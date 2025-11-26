import logging
import logging.handlers
from pathlib import Path
from config.settings import settings

def setup_logging():
    """Setup application logging with sane DEBUG/INFO defaults."""

    # Ensure logs directory exists
    settings.LOGS_DIR.mkdir(exist_ok=True)

    # Resolve effective levels
    is_debug = bool(getattr(settings, "DEBUG", False))
    try:
        configured_level = getattr(logging, str(settings.LOG_LEVEL).upper())
    except Exception:
        configured_level = logging.INFO
    root_level = logging.DEBUG if is_debug else configured_level

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # File handlers
    app_handler = logging.handlers.RotatingFileHandler(
        settings.LOGS_DIR / 'app.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    app_handler.setFormatter(file_formatter)
    app_handler.setLevel(logging.DEBUG if is_debug else logging.INFO)

    error_handler = logging.handlers.RotatingFileHandler(
        settings.LOGS_DIR / 'error.log',
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    error_handler.setFormatter(file_formatter)
    error_handler.setLevel(logging.ERROR)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG if is_debug else logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)
    root_logger.handlers.clear()
    root_logger.addHandler(app_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    # Quiet noisy third-party loggers unless in DEBUG
    noisy_loggers = [
        'sqlalchemy.engine',
        'sqlalchemy.pool',
        'uvicorn',
        'uvicorn.error',
        'uvicorn.access',
        'httpx',
        'qdrant_client',
        'transformers',
        'sentence_transformers',
        'llama_cpp',
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.DEBUG if is_debug else logging.WARNING)

    # App-specific loggers
    app_loggers = [
        'notebooklm',
        'src.processing',
        'src.rag',
        'src.serving',
        'src.frontend',
    ]
    for name in app_loggers:
        logging.getLogger(name).setLevel(root_level)

    return root_logger
