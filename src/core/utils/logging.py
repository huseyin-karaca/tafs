"""Unified loguru-frontend logging.

Both papers (and every third-party dep that uses stdlib :mod:`logging`)
emit through a single ``loguru`` sink. Hydra's own job-logging is silenced
by clearing the root handlers; an ``InterceptHandler`` redirects every
stdlib record to loguru with the original caller's frame depth so the
``file:line`` annotation is correct.
"""

from __future__ import annotations

import logging
import sys
import warnings

from loguru import logger

# Logger prefixes that are noisy at INFO and provide little signal for our
# experiments. Demoted to WARNING so the terminal stays readable.
_NOISY_LOGGERS = (
    "matplotlib",
    "PIL",
    "urllib3",
    "filelock",
    "fsspec",
    "git",
    "h5py",
)

# Substrings whose presence in a log message marks it as boilerplate
# (Lightning startup banners, the litlogger tip, etc.). Filtered out by
# the loguru sink instead of demoting the entire logger, so any genuinely
# useful message from those modules still gets through.
_MESSAGE_BLOCKLIST = (
    "litlogger",
    "Tip:",
    "TPU available",
    "GPU available",
    "HPU available",
    "Loading `train_dataloader`",
    "Trainer.fit` stopped",
    "LOCAL_RANK:",
)

# Library-emitted warnings that are cosmetic and clutter the terminal.
# Suppressed via ``warnings.filterwarnings`` rather than the logging sink
# because they go through ``warnings.warn``, not the logging module.
_WARNING_BLOCKLIST = (
    "enable_nested_tensor is True",
    "isinstance.treespec, LeafSpec.",
    "does not have many workers",
    "filesystem tracking backend",
    "GPU available but not used",
    "MPS available but not used",
)


def _message_filter(record: dict) -> bool:
    """Return False for messages whose text matches the blocklist."""
    msg = record["message"]
    return not any(needle in msg for needle in _MESSAGE_BLOCKLIST)


class _InterceptHandler(logging.Handler):
    """Bridge stdlib :mod:`logging` records into loguru.

    Lightning / mlflow / hydra all use stdlib loggers. We translate each
    record by looking up the matching loguru level and stepping the call
    stack back to the original ``logging.<level>(...)`` callsite so the
    line numbers in the formatted output are useful.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk the frame stack back through the logging module.
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_unified_logging(level: str = "INFO", fmt: str | None = None) -> None:
    """Install loguru as the single logging sink.

    Safe to call multiple times; the loguru sinks are reset on each call.
    """
    logger.remove()
    if fmt is None:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
    logger.add(sys.stderr, level=level, format=fmt, colorize=True, filter=_message_filter)

    # Route stdlib through the intercept handler and silence noisy children.
    logging.root.handlers = [_InterceptHandler()]
    logging.root.setLevel(level)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Suppress cosmetic library warnings that come through ``warnings.warn``.
    for needle in _WARNING_BLOCKLIST:
        warnings.filterwarnings("ignore", message=f".*{needle}.*")
