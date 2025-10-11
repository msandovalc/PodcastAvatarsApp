import os
from pathlib import Path
from .logger_config import setup_logger

# --- Configuration ---
logger = setup_logger(__name__)

class SubtitleGenerator:
    """Handles generation of SRT subtitle files from transcription segments."""

    def __init__(self, style_config=None):
        """
        Initialize the subtitle generator.

        Args:
            style_config (dict): Configuration for subtitle styling (e.g., font, size, color).
        """
        try:
            # Setup logging
            self.logger = logger
            self.style_config = style_config or {
                "FontName": "Arial",
                "FontSize": 12,
                "PrimaryColour": "&H00FFFFFF",
                "BorderStyle": 1,
                "Outline": 1,
                "Alignment": 2,  # Center-bottom
                "MarginV": 100
            }
            self.logger.debug("Initialized SubtitleGenerator with style_config: %s", self.style_config)
            self.logger.info("Initialized SubtitleGenerator")
        except Exception as e:
            self.logger.error("Error initializing SubtitleGenerator: %s", e, exc_info=True)
            raise

    def _format_srt_time(self, seconds):
        """Format seconds to SRT time format (HH:MM:SS,MMM)."""
        try:
            if seconds < 0:
                seconds = 0
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")
        except Exception as e:
            self.logger.error("Error formatting SRT time for %s seconds: %s", seconds, e, exc_info=True)
            return "00:00:00,000"

    def create_srt_file(self, segments, start_time, duration, srt_path):
        """
        Create an SRT file for subtitles from transcription segments.

        Args:
            segments (list): List of transcription segments with start, end, and text.
            start_time (float): Start time of the clip in seconds.
            duration (float): Duration of the clip in seconds.
            srt_path (str): Path to save the SRT file.

        Returns:
            str: Path to the created SRT file.
        """
        try:
            srt_content = ""
            end_time = start_time + duration
            relevant_segments = []
            for i, seg in enumerate(segments):
                seg_start = seg["start"]
                seg_end = seg.get("end", segments[i + 1]["start"] if i + 1 < len(segments) else end_time)
                if seg_start >= start_time and seg_start <= end_time:
                    seg_with_end = seg.copy()
                    seg_with_end["end"] = min(seg_end, end_time)
                    relevant_segments.append(seg_with_end)
                if seg_end > end_time:
                    break

            if not relevant_segments:
                self.logger.error("No segments found within %ss to %ss for SRT file %s", start_time, end_time, srt_path)
                raise ValueError("No relevant segments for SRT")

            self.logger.debug("Filtered %s segments for %ss to %ss", len(relevant_segments), start_time, end_time)
            for seg in relevant_segments:
                self.logger.debug("Segment: start=%s, end=%s, text=%s", seg['start'], seg['end'], seg['text'])

            relevant_segments.sort(key=lambda x: x["start"])

            for i, segment in enumerate(relevant_segments, 1):
                text = segment["text"].strip()
                if not text:
                    self.logger.warning("Empty text in segment at %ss, skipping", segment['start'])
                    continue
                start = max(segment["start"] - start_time, 0)
                end = min(segment["end"] - start_time, duration)
                if end <= start:
                    self.logger.warning("Invalid timing for segment at %ss: start=%s, end=%s, skipping",
                                      segment['start'], start, end)
                    continue
                start_srt = self._format_srt_time(start)
                end_srt = self._format_srt_time(end)
                srt_content += f"{i}\n{start_srt} --> {end_srt}\n{text}\n\n"

            if not srt_content:
                self.logger.error("No valid SRT content generated for %s", srt_path)
                raise ValueError("No valid SRT content generated")

            Path(srt_path).parent.mkdir(exist_ok=True)
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            self.logger.info("SRT file created: %s", srt_path)
            return srt_path
        except Exception as e:
            self.logger.error("Error creating SRT file %s: %s", srt_path, e, exc_info=True)
            raise

    def get_ffmpeg_style(self):
        """Return FFmpeg-compatible subtitle style string."""
        try:
            style = ",".join(f"{k}={v}" for k, v in self.style_config.items())
            self.logger.debug("Generated FFmpeg subtitle style: %s", style)
            return style
        except Exception as e:
            self.logger.error("Error generating FFmpeg subtitle style: %s", e, exc_info=True)
            return "FontName=Arial,FontSize=12,PrimaryColour=&H00FFFFFF,BorderStyle=1,Outline=1,Alignment=2,MarginV=100"