import os
import subprocess
import re
import sys
import torch
import uuid
from pathlib import Path
from tqdm import tqdm
import time
from .subtitle_generator import *
from .logger_config import setup_logger

# --- Configuration ---
logger = setup_logger(__name__)


class ClipGenerator:
    """Generates video clips from selected segments with embedded subtitles."""

    def __init__(self, video_path, output_dir="output_clips", subtitle_style=None, use_background=False, background_color="black"):
        """
        Initialize the clip generator.

        Args:
            video_path (str): Path to the input video.
            output_dir (str): Directory to save generated clips.
            subtitle_style (dict): Configuration for subtitle styling.
            use_background (bool): Whether to apply a background to subtitles.
            background_color (str): Background color for subtitles ("black" or "white").
        """
        try:
            # Setup logging
            self.logger = logger
            self.video_path = os.path.abspath(video_path)
            self.output_dir = os.path.abspath(output_dir)
            Path(self.output_dir).mkdir(exist_ok=True)
            self.subtitle_generator = SubtitleGenerator(style_config=subtitle_style)
            self.use_cuda = torch.cuda.is_available()
            self.use_background = use_background
            # Validate background_color
            valid_colors = ["black", "white"]
            if background_color.lower() not in valid_colors:
                self.logger.warning("Invalid background_color '%s'. Defaulting to 'black'.", background_color)
                self.background_color = "black"
            else:
                self.background_color = background_color.lower()
            self.logger.info("Initialized ClipGenerator with video: %s, output_dir: %s, CUDA available: %s, use_background: %s, background_color: %s",
                             self.video_path, self.output_dir, self.use_cuda, self.use_background, self.background_color)
            if self.use_cuda:
                try:
                    self.logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
                    # Verify FFmpeg NVENC support
                    result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, check=True)
                    if "h264_nvenc" not in result.stdout:
                        self.logger.warning("FFmpeg does not support h264_nvenc. Falling back to CPU encoding.")
                        self.use_cuda = False
                except Exception as e:
                    self.logger.error("Error checking FFmpeg NVENC support: %s", e, exc_info=True)
                    self.use_cuda = False
        except Exception as e:
            self.logger.error("Error initializing ClipGenerator: %s", e, exc_info=True)
            raise

    def _calculate_clip_duration(self, start_time, all_segments, min_duration, max_duration):
        """
        Calculate the duration of a clip to complete an idea, within min_duration and max_duration.

        Args:
            start_time (float): Start time of the clip in seconds.
            all_segments (list): List of all transcription segments with start, end, and text.
            min_duration (int): Minimum duration of the clip in seconds.
            max_duration (int): Maximum duration of the clip in seconds.

        Returns:
            float: Calculated duration of the clip in seconds.
        """
        try:
            self.logger.debug("Calculating duration for start_time=%ss, min_duration=%ss, max_duration=%ss", 
                             start_time, min_duration, max_duration)
            end_time = start_time + max_duration
            relevant_segments = []
            for i, seg in enumerate(all_segments):
                seg_start = seg["start"]
                seg_end = seg.get("end", all_segments[i + 1]["start"] if i + 1 < len(all_segments) else end_time)
                if seg_start >= start_time and seg_start <= end_time:
                    relevant_segments.append({"end": seg_end, "text": seg["text"]})
                if seg_end > end_time:
                    break

            if not relevant_segments:
                self.logger.warning("No segments found within %ss to %ss, using max_duration %ss",
                                  start_time, end_time, max_duration)
                return max_duration

            # Include all segments that end within bounds, even if before min_duration
            valid_ends = [seg["end"] for seg in relevant_segments if seg["end"] <= end_time]
            if not valid_ends:
                self.logger.warning("No valid segment ends within %ss to %ss, using max_duration %ss",
                                  start_time, end_time, max_duration)
                return max_duration

            # Prefer segments that end with punctuation
            preferred_ends = [
                seg["end"] for seg in relevant_segments
                if seg["end"] <= end_time and seg["text"].strip().endswith((".", "!", "?"))
            ]
            clip_end = max(preferred_ends or valid_ends)
            duration = clip_end - start_time
            if duration < min_duration:
                self.logger.warning("Calculated duration %ss is less than min_duration %ss, using max_duration %ss",
                                  duration, min_duration, max_duration)
                duration = max_duration
            self.logger.debug("Calculated clip duration: %ss (start=%ss, end=%ss)", duration, start_time, clip_end)
            return duration
        except Exception as e:
            self.logger.error("Error calculating clip duration: %s", e, exc_info=True)
            return max_duration

    def _run_ffmpeg_with_progress(self, ffmpeg_cmd, clip_duration, clip_index):
        """
        Run FFmpeg command with a custom progress bar showing percentage and block characters.

        Args:
            ffmpeg_cmd (list): FFmpeg command to execute.
            clip_duration (float): Duration of the clip in seconds.
            clip_index (int): Index of the clip being processed.

        Returns:
            bool: True if FFmpeg succeeds, False otherwise.
        """
        try:
            # Add progress output to FFmpeg command
            ffmpeg_cmd = ["ffmpeg", "-y", "-progress", "pipe:2"] + ffmpeg_cmd[1:]

            # Now, add the rest of your command to the list
            # ffmpeg_cmd.extend(ffmpeg_cmd[1:])

            self.logger.info("Starting FFmpeg for clip %s: %s", clip_index, " ".join(ffmpeg_cmd))

            # Start FFmpeg process
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace"
            )

            # Initialize progress bar parameters
            bar_length = 30  # Length of the progress bar in characters
            last_progress = 0

            while process.poll() is None:
                line = process.stderr.readline()
                if line:
                    self.logger.debug("FFmpeg output: %s", line.strip())
                    match = re.search(r"out_time_ms=(\d+)", line)
                    if match:
                        elapsed_ms = int(match.group(1)) // 1000  # Convert microseconds to milliseconds
                        elapsed_seconds = elapsed_ms / 1000  # Convert to seconds
                        if clip_duration > 0:
                            progress_percent = min(100, (elapsed_seconds / clip_duration) * 100)
                            # Create the progress bar string
                            filled_chars = int(bar_length * progress_percent / 100)
                            empty_chars = bar_length - filled_chars
                            # Create the progress bar string
                            progress_bar = "Progress [" + "█" * filled_chars + " " * empty_chars + "]"
                            # Print the progress on the same line
                            # self.logger.info("%s %.2f%%", progress_bar, progress_percent)
                            sys.stdout.write(f"\r{progress_bar} {progress_percent:.2f}%")
                            sys.stdout.flush()
                            last_progress = elapsed_ms

                # Sleep briefly to avoid overloading
                import time
                time.sleep(0.01)

            # Read any remaining output
            error_output = ""
            for line in process.stderr:
                self.logger.debug("FFmpeg output: %s", line.strip())
                error_output += line
                match = re.search(r"out_time_ms=(\d+)", line)
                if match:
                    elapsed_ms = int(match.group(1)) // 1000
                    if elapsed_ms > last_progress:
                        elapsed_seconds = elapsed_ms / 1000
                        progress_percent = min(100, (elapsed_seconds / clip_duration) * 100)
                        filled_chars = int(bar_length * progress_percent / 100)
                        empty_chars = bar_length - filled_chars
                        progress_bar = "Progress [" + "█" * filled_chars + " " * empty_chars + "]"
                        # self.logger.info("%s %.2f%%", progress_bar, progress_percent)
                        sys.stdout.write(f"\r{progress_bar} {progress_percent:.2f}%")
                        sys.stdout.flush()
                        last_progress = elapsed_ms

            # Log final progress and success message
            self.logger.info("Progress [%s] 100.00%% - Process completed successfully!", "█" * bar_length)
            sys.stdout.write("\n")
            sys.stdout.flush()

            # Check FFmpeg exit code
            if process.returncode != 0:
                self.logger.error("FFmpeg failed for clip %s: %s", clip_index, error_output)
                return False

            self.logger.debug("FFmpeg completed for clip %s", clip_index)
            return True
        except Exception as e:
            self.logger.error("Error running FFmpeg for clip %s: %s", clip_index, e, exc_info=True)
            return False

    # def _run_ffmpeg_with_progress(self, ffmpeg_cmd, clip_duration, clip_index):
    #     """
    #     Run FFmpeg command with a custom progress bar showing percentage and block characters.
    #
    #     Args:
    #         ffmpeg_cmd (list): FFmpeg command to execute.
    #         clip_duration (float): Duration of the clip in seconds.
    #         clip_index (int): Index of the clip being processed.
    #
    #     Returns:
    #         bool: True if FFmpeg succeeds, False otherwise.
    #     """
    #     try:
    #         # Add progress and debug output to FFmpeg command
    #         ffmpeg_cmd = ["ffmpeg", "-y", "-v", "debug", "-hide_banner", "-progress", "pipe:2"] + ffmpeg_cmd[1:]
    #         self.logger.info("Starting FFmpeg for clip %s: %s", clip_index, " ".join(ffmpeg_cmd))
    #         print(f"Starting FFmpeg for clip {clip_index}")
    #
    #         # Start FFmpeg process
    #         process = subprocess.Popen(
    #             ffmpeg_cmd,
    #             stdout=subprocess.PIPE,
    #             stderr=subprocess.PIPE,
    #             text=True,
    #             bufsize=1,
    #             universal_newlines=True,
    #             encoding="utf-8",
    #             errors="replace"
    #         )
    #
    #         # Initialize progress bar parameters
    #         last_progress = 0
    #
    #         # Use tqdm for a clean progress bar
    #         with tqdm(total=clip_duration, desc=f"Generating clip {clip_index}", unit="s", leave=True) as pbar:
    #             while process.poll() is None:
    #                 line = process.stderr.readline()
    #                 if line:
    #                     # Log all stderr output for debugging.
    #                     self.logger.debug("FFmpeg output: %s", line.strip())
    #
    #                     match = re.search(r"out_time_ms=(\d+)", line)
    #                     if match:
    #                         # Convert microseconds to seconds
    #                         elapsed_seconds = int(match.group(1)) / 1000000
    #                         if clip_duration > 0:
    #                             # Update the progress bar
    #                             pbar.n = min(elapsed_seconds, clip_duration)
    #                             pbar.refresh()
    #                             last_progress = int(match.group(1))
    #
    #                 # Sleep briefly to avoid overloading
    #                 time.sleep(0.01)
    #
    #         # Read any remaining output after the process exits
    #         error_output = process.stderr.read()
    #         if process.returncode != 0:
    #             self.logger.error("FFmpeg failed for clip %s: %s", clip_index, error_output)
    #             print(f"Error: FFmpeg failed for clip {clip_index}: {error_output}")
    #             return False
    #
    #         self.logger.info("Progress bar reached 100.00% - Process completed successfully!")
    #         self.logger.debug("FFmpeg completed for clip %s", clip_index)
    #         print(f"FFmpeg completed for clip {clip_index}")
    #         return True
    #     except Exception as e:
    #         self.logger.error("Error running FFmpeg for clip %s: %s", clip_index, e, exc_info=True)
    #         print(f"Error running FFmpeg for clip {clip_index}: {e}")
    #         return False

    def _get_subtitle_background_color(self, segment_text):
        """
        Determine subtitle background color based on configuration.
        
        Args:
            segment_text (list): List of subtitle texts for the clip (unused in current implementation).
        
        Returns:
            str or None: FFmpeg-compatible color string (e.g., 'black' or 'white') if background is used, else None.
        """
        try:
            if not self.use_background:
                self.logger.debug("Background disabled, no background color applied")
                return None
            self.logger.debug("Background enabled, using color: %s", self.background_color)
            return self.background_color
        except Exception as e:
            self.logger.error("Error determining subtitle background color: %s", e, exc_info=True)
            return None
    
    def generate_clips(self, selected_segments, min_duration=60, max_duration=90, all_segments=None):
        """
        Generate video clips from selected segments with embedded subtitles and variable duration.

        Args:
            selected_segments (list): List of selected segments with start time and text.
            min_duration (int): Minimum duration of each clip in seconds.
            max_duration (int): Maximum duration of the clip in seconds.
            all_segments (list): List of all transcription segments for SRT generation.

        Returns:
            list: List of dicts with 'clip_path', 'srt_content', 'segment_text', and 'metadata' for each generated clip.
        """
        if all_segments is None:
            all_segments = selected_segments
        clip_data = []
        try:
            self.logger.debug("Generating %s clips with duration between %ss and %ss",
                            len(selected_segments), min_duration, max_duration)

            for i, segment in enumerate(selected_segments):
                srt_path = None
                try:
                    start_time = segment["start"]
                    self.logger.info("Processing clip %s with start_time=%ss", i + 1, start_time)

                    # Calculate dynamic clip duration
                    clip_duration = self._calculate_clip_duration(start_time, all_segments, min_duration, max_duration)
                    output_path = os.path.abspath(os.path.join(self.output_dir, f"short_{uuid.uuid4()}.mp4"))
                    srt_path = os.path.join(self.output_dir, f"temp_subtitle_{i + 1}.srt")

                    self.logger.debug("Generating clip %s from %ss to %ss at %s",
                                     i + 1, start_time, start_time + clip_duration, output_path)

                    # Step 1: Generate SRT file using SubtitleGenerator
                    self.subtitle_generator.create_srt_file(all_segments, start_time, clip_duration, srt_path)

                    # Read SRT content (not strictly needed for FFmpeg, but good for data)
                    srt_content = ""
                    try:
                        with open(srt_path, "r", encoding="utf-8") as f:
                            srt_content = f.read()
                        self.logger.debug("Read SRT content for clip %s: %s", i + 1, srt_path)
                    except Exception as e:
                        self.logger.error("Error reading SRT file %s: %s", srt_path, e, exc_info=True)
                        srt_content = ""

                    # Collect segment text
                    segment_text = [
                        seg["text"] for seg in all_segments
                        if seg["start"] >= start_time and seg["start"] <= start_time + clip_duration
                    ]
                    self.logger.debug("Collected segment text for clip %s: %s", i + 1, segment_text)

                    # Prepare subtitles path for FFmpeg
                    srt_path_abs = os.path.abspath(srt_path)
                    srt_path_ffmpeg = srt_path_abs.replace("\\", "/").replace(":", "\\:")
                    self.logger.info("Formatted subtitles path for FFmpeg: %s", srt_path_ffmpeg)

                    # Get base subtitle style and append dynamic background color
                    force_style = self.subtitle_generator.get_ffmpeg_style()
                    self.logger.info("FFmpeg subtitle style: %s", force_style)
                    background_color = self._get_subtitle_background_color(segment_text)
                    if background_color:
                        force_style += f",BackColour={background_color}"
                    self.logger.info("FFmpeg subtitle style: %s", force_style)

                    # Step 2: Use FFmpeg to perform all operations in one go
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-hwaccel", "cuda" if self.use_cuda else "auto",
                        "-ss", str(start_time),
                        "-i", self.video_path,
                        "-t", str(clip_duration),
                        "-filter_complex",
                        f"[0:v]crop='if(gt(ih,iw/0.5625),iw,ih*0.5625)':ih,scale='if(gt(iw,ih/0.5625),iw,ih*0.5625)':ih:force_original_aspect_ratio=increase,scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,subtitles='{srt_path_ffmpeg}':force_style='{force_style}'[final_v];[0:a]asetpts=PTS-STARTPTS[final_a]",
                        "-map", "[final_v]",
                        "-map", "[final_a]",
                    ]
                    if self.use_cuda:
                        ffmpeg_cmd += ["-c:v", "h264_nvenc", "-preset", "p7", "-b:v", "5000k"]
                        self.logger.debug("Using NVIDIA CUDA with h264_nvenc for clip %s", i + 1)
                    else:
                        ffmpeg_cmd += ["-c:v", "libx264", "-preset", "fast", "-cq", "23", "-b:v", "5000k"]
                        self.logger.debug("Using CPU with libx264 for clip %s", i + 1)
                    ffmpeg_cmd += ["-c:a", "aac", "-b:a", "192k", output_path]

                    self.logger.debug("Preparing FFmpeg command for clip %s: %s", i + 1, " ".join(ffmpeg_cmd))
                    self.logger.debug("--- Full FFmpeg Command for Debugging ---")
                    self.logger.debug("%s", " ".join(ffmpeg_cmd))
                    self.logger.debug("-----------------------------------------")

                    # Run FFmpeg with progress bar
                    success = self._run_ffmpeg_with_progress(ffmpeg_cmd, clip_duration, i + 1)
                    if not success:
                        continue

                    # Verify output
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        self.logger.info("Generated clip with subtitles: %s", output_path)
                        clip_data.append({
                            "clip_path": output_path,
                            "segment_text": segment_text,
                            "metadata": {
                                "video_title": os.path.basename(self.video_path),
                                "book_title": "Unknown Book Title"  # Placeholder
                            }
                        })
                    else:
                        self.logger.error("Clip %s was not created or is empty", output_path)

                    # # Clean up SRT file
                    # try:
                    #     os.remove(srt_path)
                    #     self.logger.debug("Removed temporary SRT file: %s", srt_path)
                    # except Exception as e:
                    #     self.logger.warning("Failed to remove temporary SRT file %s: %s", srt_path, e)

                except Exception as e:
                    self.logger.error("Error generating clip %s at %ss: %s", i + 1, start_time, e, exc_info=True)
                    continue

            self.logger.info("Generated %s clips: %s", len(clip_data), [d["clip_path"] for d in clip_data])
            return clip_data

        except Exception as e:
            self.logger.error("Error in generate_clips: %s", e, exc_info=True)
            raise