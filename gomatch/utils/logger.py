import logging
from typing import Optional

from .typing import PathT


def get_logger(
    level: str = "INFO", log_path: Optional[PathT] = None, name: Optional[str] = None
) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        # Initialize the logger
        level = logging.__dict__[level]  # TODO: fix this
        logger.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s|%(name)s|%(levelname)s]: %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        # Add console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_path:
            # Add log file handler
            fh = logging.FileHandler(log_path)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger
