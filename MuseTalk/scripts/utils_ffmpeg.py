# MuseTalk/scripts/utils_ffmpeg.py
"""
FFmpeg utility module for handling video operations.
Provides methods to encode frames into videos, combine audio,
and monitor progress in real time.

Author: Juan
Project: MuseTalk
"""
import shutil
import subprocess
from pathlib import Path
import sys
import re
import os
from typing import Union
import logging
from MuseTalk.scripts.exceptions import FFmpegExecutionError, FFmpegNotFoundError

logger = logging.getLogger(__name__)


class FFmpegUtils:
    """
    Utility class containing helper methods for video encoding,
    audio merging, and command execution using FFmpeg.
    """

    @staticmethod
    def ensure_ffmpeg_in_path(ffmpeg_path: Union[str, Path], try_add_to_path: bool = True) -> None:
        """
        Ensure `ffmpeg` binary is available (either already in PATH or inside ffmpeg_path).
        If not found and try_add_to_path is True, attempts to prepend the provided path
        to the process PATH and re-check.

        Args:
            ffmpeg_path (Union[str, Path]): Path to ffmpeg binary or to a directory containing ffmpeg.
            try_add_to_path (bool, optional): If True, will add directory to PATH when ffmpeg is found there.
                                               Defaults to True.

        Raises:
            FFmpegNotFoundError: If ffmpeg cannot be located or executed after attempts.
        """
        try:
            # If ffmpeg already in PATH, nothing to do
            which_ffmpeg = shutil.which("ffmpeg")
            if which_ffmpeg:
                logger.info(f"ffmpeg already available at: {which_ffmpeg}")
                return

            p = Path(ffmpeg_path)

            # Candidate resolution: if user gave a dir, look for ffmpeg inside
            candidate_executables = []
            if p.is_dir():
                candidate_executables = [p / "ffmpeg", p / "ffmpeg.exe"]
            else:
                # if p is a file, treat it as candidate executable
                candidate_executables = [p]

            found_exec = None
            for cand in candidate_executables:
                if cand.exists() and cand.is_file():
                    # On some systems executability bit might be irrelevant; still check access
                    try:
                        if os.access(str(cand), os.X_OK) or os.name == "nt":
                            found_exec = str(cand)
                            break
                    except Exception:
                        # best-effort: accept file if it exists
                        found_exec = str(cand)
                        break

            # If found an executable file, optionally add its parent dir to PATH
            if found_exec:
                ffmpeg_dir = str(Path(found_exec).parent)
                if try_add_to_path:
                    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
                    logger.info(f"Prepended ffmpeg dir to PATH: {ffmpeg_dir}")
                    # recheck
                    if shutil.which("ffmpeg"):
                        logger.info(f"ffmpeg located after updating PATH: {shutil.which('ffmpeg')}")
                        return
                else:
                    # If we found binary but not adding to PATH, just verify it executes
                    return

            # If path pointed to directory but no explicit binary found, try adding directory to PATH then check
            if p.exists() and p.is_dir() and try_add_to_path:
                os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
                if shutil.which("ffmpeg"):
                    logger.info(f"ffmpeg found after adding directory to PATH: {shutil.which('ffmpeg')}")
                    return

            # Nothing worked -> raise
            raise FFmpegNotFoundError(
                f"ffmpeg not found in PATH and not found at provided location: {ffmpeg_path}"
            )

        except FFmpegNotFoundError:
            raise
        except Exception as e:
            logger.exception("Unexpected error while ensuring ffmpeg in PATH")
            raise FFmpegNotFoundError(f"Unexpected error verifying ffmpeg: {e}") from e
    # ---------------------------------------------------------------------

    @staticmethod
    def build_command(
        avatar_path: Path,
        fps: int = 30,
        use_cuda: bool = True,
        input_pattern: str = "%08d.png",
        output_file: str = "temp.mp4",
    ) -> str:
        """
        Builds the FFmpeg command to encode a sequence of frames into a video.

        Args:
            avatar_path (Path): Directory containing frame images to be processed.
            fps (int, optional): Frames per second for the output video. Defaults to 30.
            use_cuda (bool, optional): Enables GPU acceleration using CUDA if True. Defaults to True.
            input_pattern (str, optional): File naming pattern for input frames (e.g., '%08d.png'). Defaults to '%08d.png'.
            output_file (str, optional): Name of the resulting video file. Defaults to 'temp.mp4'.

        Returns:
            str: Complete FFmpeg command string ready for execution.
        """
        output_path = avatar_path / output_file
        tmp_folder = avatar_path / "tmp"

        cmd = (
            f"ffmpeg -y -v warning "
            f"{'-hwaccel cuda' if use_cuda else ''} "
            f"-r {fps} -f image2 -i {tmp_folder}/{input_pattern} "
            f"-vcodec {'h264_nvenc' if use_cuda else 'libx264'} "
            f"-pix_fmt yuv420p -crf 18 {output_path}"
        )
        return cmd

    # ---------------------------------------------------------------------

    @staticmethod
    def combine_audio(
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        use_cuda: bool = True,
    ) -> str:
        """
        Builds the FFmpeg command to merge an audio track with a video file.

        Args:
            video_path (Path): Path to the input video file.
            audio_path (Path): Path to the input audio file.
            output_path (Path): Destination path for the output video with audio merged.
            use_cuda (bool, optional): Enables GPU acceleration if True. Defaults to True.

        Returns:
            str: Complete FFmpeg command string ready for execution.
        """
        cmd = (
            f"ffmpeg -y -v warning "
            f"{'-hwaccel cuda' if use_cuda else ''} "
            f"-i {audio_path} -i {video_path} "
            f"-c:v copy -c:a aac -b:a 192k {output_path}"
        )
        return cmd

    # ---------------------------------------------------------------------

    @staticmethod
    def run_ffmpeg(cmd: str, show_progress: bool = False, req_dur: float = 0.0):
        """
        Executes an FFmpeg command with optional progress bar for real-time monitoring.

        Args:
            cmd (str): Full FFmpeg command to execute.
            show_progress (bool, optional): Enables a live progress bar in console output. Defaults to False.
            req_dur (float, optional): Expected duration of the media in seconds, used for progress calculation. Defaults to 0.0.

        Raises:
            FFmpegExecutionError: Raised if FFmpeg returns a non-zero exit code or fails to execute.
        """
        try:
            logger.info(f"[INFO] Running FFmpeg command: {cmd}")

            if show_progress and req_dur > 0:
                command = cmd.split()
                command.extend(["-progress", "pipe:1"])
                bar_length = 30

                with subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                ) as process:
                    for line in process.stdout:
                        match = re.search(r"out_time_ms=(\d+)", line)
                        if match:
                            elapsed_ms = int(match.group(1))
                            elapsed_seconds = elapsed_ms / 1_000_000.0
                            progress_percent = min(100.0, (elapsed_seconds / req_dur) * 100.0)
                            filled_chars = int(bar_length * progress_percent // 100)
                            empty_chars = bar_length - filled_chars
                            progress_bar = (
                                "Progress [" + "█" * filled_chars + " " * empty_chars + "]"
                            )
                            print(f"\r{progress_bar} {progress_percent:.2f}%", end="")
                            sys.stdout.flush()

                    process.wait()
                    print("\rProgress [██████████████████████████████] 100.00% - Done!\n")
                    if process.returncode != 0:
                        raise subprocess.CalledProcessError(process.returncode, process.args)
            else:
                subprocess.run(cmd, shell=True, check=True)

            logger.info(f"[INFO] ✅ FFmpeg executed successfully.")

        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] FFmpeg failed: {e}")
            raise FFmpegExecutionError(f"FFmpeg failed: {e}")
