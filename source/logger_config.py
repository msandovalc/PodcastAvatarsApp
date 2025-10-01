import logging
from termcolor import colored
import sys


class ColoredFormatter(logging.Formatter):
    """
    A custom logging formatter that adds color to log messages based on their level.

    This class provides a visually distinct way to differentiate between different
    types of log output (e.g., DEBUG, INFO, ERROR) in the console.
    """

    FORMATS = {
        logging.DEBUG: colored("%(asctime)s - %(levelname)s - %(message)s", "blue"),
        logging.INFO: colored("%(asctime)s - %(levelname)s - %(message)s", "white"),
        logging.WARNING: colored("%(asctime)s - %(levelname)s - %(message)s", "yellow"),
        logging.ERROR: colored("%(asctime)s - %(levelname)s - %(message)s", "red"),
        logging.CRITICAL: colored("%(asctime)s - %(levelname)s - %(message)s", "red", attrs=["bold"]),
    }

    def format(self, record):
        """
        Formats a log record, applying the color based on its level.

        Args:
            record (logging.LogRecord): The log record to be formatted.

        Returns:
            str: The formatted and colored log message.
        """
        # Get the color format string for the record's log level
        log_fmt = self.FORMATS.get(record.levelno, colored("%(asctime)s - %(levelname)s - %(message)s", "white"))

        # Create a standard formatter with the selected format and format the record
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(name=__name__, log_level=logging.INFO, log_file=None):
    """
    Configures a logger with color formatting for the console and optionally for a file.

    This function sets up a logger instance, adds a console handler with a custom
    colored formatter, and can optionally add a file handler for persistent logs.
    It also checks for existing handlers to prevent adding duplicates.

    Args:
        name (str): The name of the logger. Defaults to the name of the calling module.
        log_level (int): The minimum logging level to capture (e.g., logging.DEBUG, logging.INFO).
                         Defaults to logging.INFO.
        log_file (str, optional): The path to a file where logs should also be written.
                                  Defaults to None, in which case no file logging is performed.

    Returns:
        logging.Logger: The configured logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Check if handlers already exist to prevent adding them multiple times, which
    has_console_handler = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_console_handler:
        # Prevent propagation to the root logger to avoid message duplication.
        logger.propagate = False

        # --- NEW CODE: Force UTF-8 encoding for the console handler ---
        try:
            # For modern Python versions, reconfigure the stream
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
            # Fallback for older versions or non-reconfigurable streams
            elif sys.stdout.encoding.lower() != 'utf-8':
                sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        except Exception:
            # Handle cases where re-assigning stdout is not possible (e.g., some IDEs)
            pass
        # --- END OF NEW CODE ---

        # Handler for the console (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        # Apply the custom colored formatter to the console handler
        colored_formatter = ColoredFormatter()
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)

    # Optional file handler
    if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        file_handler = logging.FileHandler(log_file)
        # Standard formatter for file logs (no colors)
        formatter_file = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(formatter_file)
        logger.addHandler(file_handler)

    return logger
