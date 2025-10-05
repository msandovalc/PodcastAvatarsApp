# MuseTalk/scripts/utils_io.py
"""
I/O utility functions for MuseTalk project.

Includes:
- osmakedirs: Create multiple directories if they don't exist.
- video2imgs: Convert a video into individual image frames.
"""

import os
import logging
import cv2

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def osmakedirs(path_list):
    """
    Create multiple directories if they don't already exist.

    Args:
        path_list (list[str] or list[Path]): List of directory paths to create.
    """
    try:
        for path in path_list:
            os.makedirs(path, exist_ok=True)
            logging.info(f"[INFO] Directory created or already exists: {path}")
    except Exception as e:
        logging.error(f"[ERROR] Failed to create directories: {e}")
        raise


def video2imgs(vid_path, save_path, ext='.png', cut_frame=1e7):
    """
    Convert a video file into individual image frames.

    Args:
        vid_path (str or Path): Path to the input video file.
        save_path (str or Path): Path to the folder where frames will be saved.
        ext (str, optional): Extension/format for saved images. Default is '.png'.
        cut_frame (int, optional): Maximum number of frames to extract. Default is 1e7.

    Raises:
        FileNotFoundError: If the input video file does not exist.
        Exception: For other errors during video processing.
    """
    try:
        if not os.path.exists(vid_path):
            raise FileNotFoundError(f"Video file does not exist: {vid_path}")

        os.makedirs(save_path, exist_ok=True)  # Ensure save directory exists
        logging.info(f"[INFO] Starting video frame extraction: {vid_path}")

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            raise Exception(f"Failed to open video file: {vid_path}")

        count = 0
        while True:
            if count > cut_frame:
                logging.info(f"[INFO] Reached cut_frame limit: {cut_frame}")
                break

            ret, frame = cap.read()
            if not ret:
                logging.info(f"[INFO] Video frames extraction completed. Total frames: {count}")
                break

            frame_file = os.path.join(save_path, f"{count:08d}{ext}")
            cv2.imwrite(frame_file, frame)
            count += 1

            if count % 100 == 0:
                logging.info(f"[INFO] Extracted {count} frames...")

        cap.release()
        logging.info(f"[INFO] Finished extracting frames to: {save_path}")

    except Exception as e:
        logging.error(f"[ERROR] Failed during video2imgs: {e}")
        raise
