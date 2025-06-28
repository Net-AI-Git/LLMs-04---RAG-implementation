"""
Shared logging configuration with colored output and custom exceptions for all modules.
"""
import logging

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        record.msg = f"{log_color}{record.msg}{self.COLORS['RESET']}"
        return super().format(record)


def configure_logging(debug_mode: bool = False):
    """
    Configure the root logger for the entire application.

    This function should be called once at the start of the application.

    Args:
        debug_mode (bool): If True, logs will be printed to the console.
                           If False, logs will be silenced by default.
    """
    # Get the root logger, which is the parent of all other loggers.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # Set the minimum level to capture.

    # Remove any existing handlers to prevent duplicate logs.
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    if debug_mode:
        # In debug mode, add a handler that prints to the console with colors.
        console_handler = logging.StreamHandler()
        colored_formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(colored_formatter)
        root_logger.addHandler(console_handler)
    else:
        # In normal (non-debug) mode, add a NullHandler.
        # This handler effectively discards all log messages.
        root_logger.addHandler(logging.NullHandler())


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module. It will inherit the root configuration.

    Args:
        module_name (str): Name of the module (usually __name__)

    Returns:
        logging.Logger: Logger instance for the module.
    """
    return logging.getLogger(module_name)


# Custom Exceptions (remain unchanged)
class DocumentProcessingError(Exception):
    """Raised when document processing fails."""
    pass

class DatabaseError(Exception):
    """Raised when database operations fail."""
    pass

class EmbeddingGenerationError(Exception):
    """Raised when embedding generation fails."""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass

class SearchError(Exception):
    """Raised when search operations fail."""
    pass

class DatabaseSearchError(Exception):
    """Raised when database search operations fail."""
    pass

class EmbeddingError(Exception):
    """Raised when embedding creation fails."""
    pass
