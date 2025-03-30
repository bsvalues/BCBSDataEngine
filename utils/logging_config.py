"""
Centralized logging configuration for the BCBS_Values application.

This module provides a standardized logging setup that can be used across all
components of the application for consistent log formatting and handling.
"""
import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

def setup_logging(
    log_level=logging.INFO,
    log_file="etl_pipeline.log",
    console_level=logging.INFO,
    file_level=logging.DEBUG,
    module_name=None
):
    """
    Configure logging for the application with both console and file handlers.
    
    Args:
        log_level (int or str): The overall logging level to use (default: logging.INFO)
        log_file (str): The name of the log file (default: 'etl_pipeline.log')
        console_level (int or str): The logging level for console output (default: logging.INFO)
        file_level (int or str): The logging level for file output (default: logging.DEBUG)
        module_name (str): Optional name for the logger (default: None, uses root logger)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Convert string log levels to numeric values if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    if isinstance(console_level, str):
        console_level = getattr(logging, console_level.upper(), logging.INFO)
    if isinstance(file_level, str):
        file_level = getattr(logging, file_level.upper(), logging.DEBUG)
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a log file with timestamp if a specific module name is provided
    if module_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{module_name}_{timestamp}.log"
    
    # Get the full path to the log file
    log_path = log_dir / log_file
    
    # Create a logger (use module name if provided, otherwise root logger)
    logger = logging.getLogger(module_name)
    
    # Clear any existing handlers to avoid duplicates when reconfiguring
    if logger.handlers:
        logger.handlers.clear()
    
    # Set the overall logging level
    logger.setLevel(log_level)
    
    # Create formatters for different verbosity levels
    detailed_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB max file size
        backupCount=5,              # Keep up to 5 backup files
        encoding="utf-8"
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Configure additional log colors for console if colorlog is available
    # Note: colorlog is optional, standard formatting will be used if not available
    try:
        # Try to import colorlog module (may not be installed)
        colorlog_available = False
        try:
            import colorlog
            colorlog_available = True
        except ImportError:
            # If colorlog is not available, just use standard formatting
            logger.debug("colorlog module not available, using standard formatting")
            colorlog_available = False
            
        # Only apply color formatting if colorlog is available    
        if colorlog_available:
            color_formatter = colorlog.ColoredFormatter(
                "%(log_color)s[%(asctime)s] [%(levelname)s] - %(message)s%(reset)s",
                datefmt="%H:%M:%S",
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(color_formatter)
            logger.debug("Color logging enabled")
    except Exception as e:
        # Fall back to standard formatting if any error occurs
        logger.debug(f"Error setting up color logging: {str(e)}")
    
    # Log the configuration details
    logger.debug(f"Logging initialized: console={logging.getLevelName(console_level)}, "
                f"file={logging.getLevelName(file_level)}, path={log_path}")
    
    return logger

def get_etl_logger(component="etl"):
    """
    Get a logger specifically configured for ETL operations.
    This logger includes component-specific formatting and logs to a timestamped file.
    
    Args:
        component (str): ETL component name (e.g., 'pacs_import', 'data_validation')
    
    Returns:
        logging.Logger: Configured logger instance
    """
    return setup_logging(
        log_level=logging.DEBUG,
        module_name=f"etl_pipeline",
        console_level=logging.INFO,
        file_level=logging.DEBUG
    )

def get_api_logger():
    """
    Get a logger specifically configured for API operations.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    return setup_logging(
        log_level=logging.INFO,
        log_file="api.log",
        console_level=logging.WARNING,  # Less verbose console output for API
        file_level=logging.INFO,
        module_name="api"
    )

def get_valuation_logger():
    """
    Get a logger specifically configured for valuation operations.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    return setup_logging(
        log_level=logging.DEBUG,
        log_file="valuation.log",
        console_level=logging.INFO,
        file_level=logging.DEBUG,
        module_name="valuation"
    )

# Additional helper functions for specific log types

def log_data_operation(logger, operation, source, record_count, duration_ms=None, success=True):
    """
    Log a standardized data operation message.
    
    Args:
        logger (logging.Logger): The logger to use
        operation (str): Operation name (e.g., 'extract', 'transform', 'load')
        source (str): Data source name (e.g., 'PACS', 'MLS')
        record_count (int): Number of records processed
        duration_ms (float, optional): Duration in milliseconds
        success (bool): Whether the operation was successful
    """
    status = "SUCCESS" if success else "FAILURE"
    duration_info = f", duration={duration_ms:.2f}ms" if duration_ms else ""
    logger.info(f"{operation.upper()} {status}: source={source}, records={record_count}{duration_info}")
    
    # Add structured log entry for potential log analysis tools
    if logger.isEnabledFor(logging.DEBUG):
        import json
        structured_log = {
            "operation": operation,
            "source": source,
            "status": status.lower(),
            "record_count": record_count,
            "timestamp": datetime.now().isoformat()
        }
        if duration_ms:
            structured_log["duration_ms"] = duration_ms
        
        logger.debug(f"STRUCTURED_LOG: {json.dumps(structured_log)}")

def log_validation_result(logger, validation_results):
    """
    Log validation results in a standardized format.
    
    Args:
        logger (logging.Logger): The logger to use
        validation_results (dict): Validation results dictionary
    """
    if validation_results["validation_passed"]:
        logger.info(f"VALIDATION PASSED: {validation_results['valid_records']}/{validation_results['total_records']} "
                   f"records valid ({validation_results['valid_records']/validation_results['total_records']*100:.1f}%)")
    else:
        logger.warning(f"VALIDATION ISSUES: {validation_results['invalid_records']}/{validation_results['total_records']} "
                      f"records have issues ({validation_results['invalid_records']/validation_results['total_records']*100:.1f}%)")
    
    # Log detailed validation results at debug level
    for check_type, result in validation_results["validation_results"].items():
        if result["count"] > 0:
            logger.debug(f"Validation issue '{check_type}': {result['count']} occurrences")
            
            # Log first few details at debug level for diagnostics
            if result["details"] and logger.isEnabledFor(logging.DEBUG):
                for i, detail in enumerate(result["details"][:5]):  # Log up to 5 examples
                    logger.debug(f"  Detail {i+1}: {detail}")
                
                if len(result["details"]) > 5:
                    logger.debug(f"  ... and {len(result['details']) - 5} more issues")