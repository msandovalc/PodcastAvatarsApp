import cv2
import subprocess
import numpy as np
import os
from pathlib import Path
import time
from .logger_config import setup_logger

# --- Configuration ---
logger = setup_logger(__name__)

class VideoProcessor:
    """Handles video processing tasks such as frame extraction and audio extraction."""

    def __init__(self, video_path, use_cuda=False):
        """
        Initialize the video processor with the input video.

        Args:
            video_path (str): Path to the input video.
            use_cuda (bool): Whether to use CUDA for hardware acceleration.

        Raises:
            FileNotFoundError: If the video file does not exist.
        """
        # Setup logging
        self.logger = logger
        self.video_path = video_path
        self.use_cuda = use_cuda
        self.cap = None
        try:
            if not os.path.exists(video_path):
                self.logger.error(f"Video file not found: {video_path}", exc_info=True)
                raise FileNotFoundError(f"Video file not found: {video_path}")
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video: {video_path}", exc_info=True)
                raise ValueError(f"Failed to open video: {video_path}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.frame_count / self.fps if self.fps > 0 else 0
            self.logger.info(f"Video loaded: {video_path}, duration: {self.duration}s, fps: {self.fps}")
            Path("temp").mkdir(exist_ok=True)
        except Exception as e:
            self.logger.error(f"Error initializing video processor: {e}", exc_info=True)
            raise

    def extract_audio(self):
        """
        Extract audio from the video to a WAV file using FFmpeg.

        Returns:
            str: Path to the extracted audio file.

        Raises:
            subprocess.CalledProcessError: If FFmpeg audio extraction fails.
        """
        try:
            output_path = "temp_audio.wav"
            cmd = ["ffmpeg", "-i", self.video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                   "-threads", "1", output_path, "-y"]
            if self.use_cuda:
                cmd.insert(1, "-hwaccel")
                cmd.insert(2, "cuda")
            self.logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)  # Added timeout
            self.logger.info(f"Audio extracted to: {output_path}")
            return output_path
        except subprocess.TimeoutExpired:
            self.logger.error("FFmpeg audio extraction timed out", exc_info=True)
            raise
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg audio extraction failed: {e.stderr}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Error extracting audio: {e}", exc_info=True)
            raise

    def get_frame(self, timestamp):
        """
        Extract a frame at the specified timestamp using OpenCV.

        Args:
            timestamp (float): Time in seconds.

        Returns:
            numpy.ndarray: Frame as an RGB image.

        Raises:
            ValueError: If frame cannot be read.
        """
        try:
            frame_number = int(timestamp * self.fps)
            if frame_number >= self.frame_count:
                self.logger.warning(f"Timestamp {timestamp}s exceeds video duration {self.duration}s, using last frame")
                frame_number = self.frame_count - 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error(f"Failed to read frame at {timestamp}s", exc_info=True)
                raise ValueError(f"Failed to read frame at {timestamp}s")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        except Exception as e:
            self.logger.error(f"Error extracting frame at {timestamp}s: {e}", exc_info=True)
            raise

    def get_frame_ffmpeg(self, timestamp):
        """
        Extract a frame at the specified timestamp using FFmpeg.

        Args:
            timestamp (float): Time in seconds.

        Returns:
            numpy.ndarray: Frame as an RGB image.

        Raises:
            ValueError: If FFmpeg fails to extract the frame.
        """
        try:
            # Cap timestamp to video duration
            if timestamp >= self.duration:
                self.logger.warning(
                    f"Timestamp {timestamp}s exceeds video duration {self.duration}s, capping to {self.duration - 0.1}s")
                timestamp = max(0, self.duration - 0.1)
            elif timestamp < 0:
                self.logger.warning(f"Timestamp {timestamp}s is negative, setting to 0.0s")
                timestamp = 0.0

            # Unique output path to avoid file locks
            output_path = f"temp/temp_frame_{timestamp:.3f}_{int(time.time() * 1000)}.jpg"
            cmd = [
                "ffmpeg", "-ss", str(timestamp), "-i", self.video_path,
                "-vframes", "1", "-pix_fmt", "yuv420p", "-vf", "select=eq(n\,0)",
                "-q:v", "2", "-strict", "unofficial", "-threads", "1", output_path, "-y"
            ]
            self.logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                self.logger.error(f"FFmpeg frame extraction failed at {timestamp}s: {result.stderr}", exc_info=True)
                raise ValueError(f"FFmpeg failed to extract frame at {baseline}s: {result.stderr}")

            if not os.path.exists(output_path):
                self.logger.error(f"FFmpeg output file not created at {timestamp}s: {output_path}", exc_info=True)
                raise ValueError(f"FFmpeg output file not created: {output_path}")

            frame = cv2.imread(output_path)
            if frame is None:
                self.logger.error(f"Failed to load frame from {output_path}", exc_info=True)
                raise ValueError(f"Failed to load frame from {output_path}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Attempt to remove temporary file with retries
            for _ in range(3):
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                        self.logger.debug(f"Removed temporary frame file: {output_path}")
                        break
                    except Exception as e:
                        self.logger.warning(f"Failed to remove temporary frame file {output_path}: {e}", exc_info=True)
                        time.sleep(0.1)
                else:
                    break
            return frame
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg frame extraction failed at {timestamp}s: {e.stderr}", exc_info=True)
            self.logger.info(f"Falling back to OpenCV for frame extraction at {timestamp}s")
            return self.get_frame(timestamp)
        except Exception as e:
            self.logger.error(f"Error extracting frame with FFmpeg at {timestamp}s: {e}", exc_info=True)
            self.logger.info(f"Falling back to OpenCV for frame extraction at {timestamp}s")
            return self.get_frame(timestamp)

    def close(self):
        """
        Release video resources.
        """
        if self.cap:
            self.cap.release()
            self.logger.info("Video resources released")