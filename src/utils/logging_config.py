"""
Logging configuration for aspect extraction experiments.
Provides file and console logging with proper formatting.
"""

import logging
import os
import sys
from datetime import datetime


class CustomFormatter(logging.Formatter):
    """Custom formatter with file names and timestamps."""
    
    def format(self, record):
        # Add filename to the record
        if hasattr(record, 'pathname'):
            filename = os.path.basename(record.pathname)
        else:
            filename = "unknown.py"
        
        # Create formatted message
        log_format = f"%(asctime)s [%(levelname)s] - {filename} : %(message)s"
        formatter = logging.Formatter(
            log_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        return formatter.format(record)


def setup_logging(
    experiment_name: str,
    logs_dir: str = "results/logs",
    log_level: str = "INFO",
    console_output: bool = True
) -> logging.Logger:
    """
    Setup logging for experiment with file and console output.
    
    Args:
        experiment_name: Name of the experiment
        logs_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        console_output: Whether to output to console as well
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    experiment_logs_dir = os.path.join(logs_dir, experiment_name)
    os.makedirs(experiment_logs_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{timestamp}_{experiment_name}.log"
    log_filepath = os.path.join(experiment_logs_dir, log_filename)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create custom formatter
    formatter = CustomFormatter()
    
    # File handler with immediate flush
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Log initial info
    logger.info(f"Logging initialized for experiment: {experiment_name}")
    logger.info(f"Log file: {log_filepath}")
    logger.info(f"Log level: {log_level}")
    
    return logger


def get_experiment_logger(experiment_name: str) -> logging.Logger:
    """
    Get logger for specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"experiment.{experiment_name}")


class LogProgress:
    """Context manager for logging progress with auto-flush."""
    
    def __init__(self, logger: logging.Logger, message: str):
        self.logger = logger
        self.message = message
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.message}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed: {self.message} (took {duration})")
        else:
            self.logger.error(f"Failed: {self.message} (took {duration}) - {exc_val}")
        
        # Force flush all handlers
        for handler in self.logger.handlers:
            handler.flush()


# Example usage
if __name__ == "__main__":
    # Test logging setup
    logger = setup_logging("test_experiment", "test_logs")
    
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test progress logging
    with LogProgress(logger, "sample training process"):
        import time
        time.sleep(1)
        logger.info("Processing batch 1/10")
        time.sleep(1)
        logger.info("Processing batch 5/10")
    
    print("✅ Logging test completed!")