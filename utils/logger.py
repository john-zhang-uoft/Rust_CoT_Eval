import logging
from typing import Optional

logger = logging.getLogger("rust_cot_eval")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def debug(msg: str, verbose: bool = False) -> None:
    """Log debug message if verbose is enabled"""
    if verbose:
        logger.debug(msg)

def info(msg: str, verbose: bool = False) -> None:
    """Log info message if verbose is enabled"""
    if verbose:
        logger.info(msg)

def warning(msg: str) -> None:
    """Log warning message"""
    logger.warning(msg)

def error(msg: str) -> None:
    """Log error message"""
    logger.error(msg)

def set_verbose_mode(verbose: bool = False) -> None:
    """Set logger level based on verbose mode"""
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING) 