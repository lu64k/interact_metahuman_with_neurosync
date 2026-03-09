import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(threadName)s | %(message)s"
)


class _ConsoleNoiseFilter(logging.Filter):
    """Reduce noisy logs in terminal while keeping full file logs."""

    _logger_name = "utils.stt.Ali_voicer_rc"
    _message_prefixes = (
        "[聚合]",
        "[时间基准]",
        "收到完整句子:",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != self._logger_name:
            return True
        message = record.getMessage()
        for prefix in self._message_prefixes:
            if message.startswith(prefix):
                return False
        return True


def setup_logging(
    level: int = logging.INFO,
    log_dir: str = "logs",
    log_file: str = "app.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Configure root logger once with console + rotating file handlers."""
    root_logger = logging.getLogger()

    # Avoid duplicate handlers when app modules are reloaded.
    if getattr(root_logger, "_metahuman_logging_configured", False):
        return

    root_logger.setLevel(level)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    full_log_path = log_path / log_file

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_ConsoleNoiseFilter())

    file_handler = RotatingFileHandler(
        filename=str(full_log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger._metahuman_logging_configured = True

    # Keep noisy third-party logs manageable by default.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


def parse_log_level(default: int = logging.INFO) -> int:
    """Parse APP_LOG_LEVEL env var to a logging level, fallback to default."""
    level_name = os.getenv("APP_LOG_LEVEL", "").strip().upper()
    if not level_name:
        return default
    return getattr(logging, level_name, default)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get logger by name, defaults to root logger when name is None."""
    return logging.getLogger(name)
