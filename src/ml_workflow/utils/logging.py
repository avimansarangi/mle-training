import logging
import sys
from pathlib import Path


from typing import Optional

def get_logger(
    name: str,
    log_level: str = "INFO",
    log_path: Optional[str] = None,
    console_log: bool = True,
):
    """
    Create and configure a logger.

    Parameters
    ----------
    name : str
        Logger name.
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    log_path : str or None
        Optional file path to write logs.
    console_log : bool
        Whether to log to console.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level.upper())
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    if console_log:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
