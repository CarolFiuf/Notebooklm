import logging
import logging.handlers
from pathlib import Path
from config.config import settings

def setup_logging():
    """Setup application logging"""
    
    # Ensure logs directory exists
    settings.LOGS_DIR.mkdir(exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handlers
    app_handler = logging.handlers.RotatingFileHandler(
        settings.LOGS_DIR / 'app.log',
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    app_handler.setFormatter(file_formatter)
    app_handler.setLevel(logging.INFO)
    
    error_handler = logging.handlers.RotatingFileHandler(
        settings.LOGS_DIR / 'error.log',
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    error_handler.setFormatter(file_formatter)
    error_handler.setLevel(logging.ERROR)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO if settings.DEBUG else logging.WARNING)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    root_logger.addHandler(app_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    # Setup specific loggers
    loggers = [
        'notebooklm',
        'src.processing',
        'src.rag', 
        'src.serving',
        'src.frontend'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    return root_logger