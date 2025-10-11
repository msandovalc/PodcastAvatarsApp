import logging
from logging.handlers import RotatingFileHandler
import os
import torch
import subprocess
from .logger_config import setup_logger

# --- Configuration ---
logger = setup_logger(__name__)



def get_cpu_count():
    """
    Get the number of available CPU cores for multithreading.

    Returns:
        int: Number of CPU cores, defaulting to 4 if detection fails.
    """
    try:
        return max(1, os.cpu_count() or 4)
    except Exception as e:
        logger.warning(f"Error detecting CPU count: {e}", exc_info=True)
        return 4


def check_cuda():
    """
    Check if CUDA is available and FFmpeg supports NVENC.

    Returns:
        bool: True if CUDA and NVENC are available, False otherwise.
    """
    logger = logging.getLogger(__name__)
    try:
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            logger.warning("CUDA is not available in PyTorch", exc_info=True)
            return False
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        logger.info(f"CUDA detected: {gpu_name}, PyTorch CUDA version: {cuda_version}")

        # Check FFmpeg NVENC support
        result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        nvenc_supported = "h264_nvenc" in result.stdout
        if not nvenc_supported:
            logger.warning("FFmpeg does not support NVENC", exc_info=True)
            return False

        logger.info("CUDA and NVENC are supported")
        return True
    except Exception as e:
        logger.warning(f"Error checking CUDA/NVENC: {e}", exc_info=True)
        return False