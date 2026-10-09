# src/utils/logger.py

import logging
from pathlib import Path


# =========================================================
# LOGGING CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE = LOG_DIR / "app.log"


# =========================================================
# LOG FORMAT
# =========================================================

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)


# =========================================================
# LOGGER FACTORY
# =========================================================

def get_logger(
    name: str = "rag_chatbot",
) -> logging.Logger:
    """
    Create or retrieve a configured application logger.

    Args:
        name:
            Logger name.

    Returns:
        logging.Logger:
            Configured logger instance.
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers when Streamlit reloads
    if logger.handlers:
        return logger

    logger.setLevel(
        logging.INFO
    )

    formatter = logging.Formatter(
        LOG_FORMAT
    )

    # -----------------------------------------------------
    # File handler
    # -----------------------------------------------------

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )

    file_handler.setLevel(
        logging.INFO
    )

    file_handler.setFormatter(
        formatter
    )

    # -----------------------------------------------------
    # Console handler
    # -----------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setLevel(
        logging.INFO
    )

    console_handler.setFormatter(
        formatter
    )

    # -----------------------------------------------------
    # Add handlers
    # -----------------------------------------------------

    logger.addHandler(
        file_handler
    )

    logger.addHandler(
        console_handler
    )

    # Prevent messages from being duplicated
    # by the root logger.
    logger.propagate = False

    return logger


# =========================================================
# DEFAULT APPLICATION LOGGER
# =========================================================

logger = get_logger()


# =========================================================
# CONVENIENCE FUNCTIONS
# =========================================================

def log_info(message: str) -> None:
    """
    Log an informational message.
    """

    logger.info(message)


def log_warning(message: str) -> None:
    """
    Log a warning message.
    """

    logger.warning(message)


def log_error(
    message: str,
    exc_info: bool = False,
) -> None:
    """
    Log an error message.

    Args:
        message:
            Error description.

        exc_info:
            Include exception traceback if True.
    """

    logger.error(
        message,
        exc_info=exc_info,
    )


def log_exception(
    message: str,
) -> None:
    """
    Log an exception with its traceback.
    """

    logger.exception(
        message
    )

