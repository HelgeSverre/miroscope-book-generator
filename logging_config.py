"""
Comprehensive logging configuration for the book generator project.
Provides detailed logging with proper formatters, handlers, and library logger configuration.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class LoggingConfig:
    """Centralized logging configuration for the book generator."""

    # Default logger names to configure
    LIBRARY_LOGGERS = [
        "lilypad",
        "httpx",
        "mirascope",
        "openai",
        "lilypad.otel-debug",
        "opentelemetry",
        "lilypad._opentelemetry._opentelemetry_openai",
    ]

    def __init__(
        self,
        log_dir: str = "logs",
        log_level: str = "INFO",
        console_level: str = "WARNING",
        include_libraries: bool = True,
    ):
        """
        Initialize logging configuration.

        Args:
            log_dir: Directory for log files
            log_level: Level for file logging (DEBUG, INFO, WARNING, ERROR)
            console_level: Level for console output
            include_libraries: Whether to configure library loggers
        """
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
        self.console_level = getattr(logging, console_level.upper())
        self.include_libraries = include_libraries

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(exist_ok=True)

        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"book_generator_{timestamp}.log"
        self.library_log_file = self.log_dir / f"libraries_{timestamp}.log"

    def get_detailed_formatter(self) -> logging.Formatter:
        """Get detailed formatter for file logging."""
        return logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-30s | %(funcName)-25s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def get_simple_formatter(self) -> logging.Formatter:
        """Get simple formatter for console output."""
        return logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
        )

    def setup_root_logger(self) -> None:
        """Configure the root logger with file and console handlers."""
        # Remove any existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers = []
        root_logger.setLevel(
            logging.DEBUG
        )  # Capture everything, filter at handler level

        # File handler for application logs
        app_file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        app_file_handler.setLevel(self.log_level)
        app_file_handler.setFormatter(self.get_detailed_formatter())

        # Console handler for important messages only
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(self.get_simple_formatter())

        # Add handlers to root logger
        root_logger.addHandler(app_file_handler)
        root_logger.addHandler(console_handler)

    def setup_library_loggers(self) -> None:
        """Configure library loggers to separate file."""
        if not self.include_libraries:
            return

        # Create a separate handler for library logs
        lib_file_handler = logging.FileHandler(self.library_log_file, encoding="utf-8")
        lib_file_handler.setLevel(logging.DEBUG)  # Capture all library logs
        lib_file_handler.setFormatter(self.get_detailed_formatter())

        for logger_name in self.LIBRARY_LOGGERS:
            try:
                lib_logger = logging.getLogger(logger_name)
                # Remove existing handlers to avoid duplication
                lib_logger.handlers = []
                lib_logger.addHandler(lib_file_handler)
                lib_logger.setLevel(logging.DEBUG)
                lib_logger.propagate = False  # Don't propagate to root logger
            except Exception as e:
                logging.debug(f"Could not configure logger '{logger_name}': {e}")

    def configure(self) -> None:
        """Apply the complete logging configuration."""
        self.setup_root_logger()
        self.setup_library_loggers()

        # Log configuration details
        logger = logging.getLogger(__name__)
        logger.info("=" * 80)
        logger.info("Logging system initialized")
        logger.info(f"Application log: {self.log_file}")
        if self.include_libraries:
            logger.info(f"Library log: {self.library_log_file}")
        logger.info(f"Log level (file): {logging.getLevelName(self.log_level)}")
        logger.info(f"Log level (console): {logging.getLevelName(self.console_level)}")
        logger.info("=" * 80)


class ProgressLogger:
    """Helper class for structured progress logging."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_times: Dict[str, datetime] = {}

    def start_operation(
        self, operation_id: str, description: str, **extra: Any
    ) -> None:
        """Log the start of an operation with timing."""
        self.start_times[operation_id] = datetime.now()
        self.logger.info(f"[START] {description}", extra=extra)

    def update_progress(self, operation_id: str, message: str, **extra: Any) -> None:
        """Log progress update for an operation."""
        if operation_id in self.start_times:
            elapsed = (datetime.now() - self.start_times[operation_id]).total_seconds()
            extra["elapsed_seconds"] = elapsed
        self.logger.info(f"[PROGRESS] {message}", extra=extra)

    def complete_operation(
        self, operation_id: str, description: str, **extra: Any
    ) -> None:
        """Log the completion of an operation with duration."""
        if operation_id in self.start_times:
            duration = (datetime.now() - self.start_times[operation_id]).total_seconds()
            extra["duration_seconds"] = duration
            self.logger.info(
                f"[COMPLETE] {description} (took {duration:.2f}s)", extra=extra
            )
            del self.start_times[operation_id]
        else:
            self.logger.info(f"[COMPLETE] {description}", extra=extra)

    def log_error(
        self, operation_id: str, error: Exception, context: str, **extra: Any
    ) -> None:
        """Log an error with structured context."""
        extra.update(
            {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
            }
        )
        if operation_id in self.start_times:
            elapsed = (datetime.now() - self.start_times[operation_id]).total_seconds()
            extra["elapsed_seconds"] = elapsed
        self.logger.error(
            f"[ERROR] {context}: {type(error).__name__}: {error}", extra=extra
        )

    def log_summary(self, title: str, stats: Dict[str, Any]) -> None:
        """Log a formatted summary of statistics."""
        self.logger.info(f"[SUMMARY] {title}")
        for key, value in stats.items():
            self.logger.info(f"  - {key}: {value}")


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: Optional[str] = None,
    console_level: Optional[str] = None,
    include_libraries: bool = True,
) -> ProgressLogger:
    """
    Convenience function to setup logging with environment variable support.

    Environment variables:
        LOG_DIR: Directory for log files (default: "logs")
        LOG_LEVEL: File logging level (default: "INFO")
        CONSOLE_LOG_LEVEL: Console logging level (default: "WARNING")
        LOG_LIBRARIES: Whether to log library output (default: "true")

    Returns:
        ProgressLogger instance for structured logging
    """
    # Get configuration from environment or use defaults
    log_dir = log_dir or os.getenv("LOG_DIR", "logs")
    log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
    console_level = console_level or os.getenv("CONSOLE_LOG_LEVEL", "WARNING")
    include_libraries = (
        include_libraries and os.getenv("LOG_LIBRARIES", "true").lower() == "true"
    )

    # Create and apply configuration
    config = LoggingConfig(
        log_dir=log_dir,
        log_level=log_level,
        console_level=console_level,
        include_libraries=include_libraries,
    )
    config.configure()

    # Return a progress logger for the main application
    return ProgressLogger(logging.getLogger("book_generator"))
