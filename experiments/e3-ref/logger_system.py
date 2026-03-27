from __future__ import annotations

import logging


def _get_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("e3_ref")


def log_msg(level: str, message: str) -> None:
    logger = _get_logger()
    method = getattr(logger, level.lower(), logger.info)
    method(message)
