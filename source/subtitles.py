import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from logger_config import setup_logger
from tqdm import tqdm
import torch
from moviepy.editor import VideoFileClip
import whisper

# ---------------- Configuration ----------------
logging = setup_logger(__name__)

# ---------------- Constants for Directory Paths ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
PODCAST_DIR = BASE_DIR / "podcast_content"
TEMP_DIR = PODCAST_DIR / "temp"
SUBTITLES_DIR = PODCAST_DIR / "subtitles"

# ---------------- Device & FFmpeg Setup ----------------
if torch.cuda.is_available():
    device = "cuda"
    base_cmd = ["ffmpeg", "-y", "-hwaccel", "cuda"]
    codec_cmd = ['-c:v', 'h264_nvenc']
    logging.info(f"[+] Device: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    base_cmd = ["ffmpeg", "-y"]
    codec_cmd = ['-c:v', 'libx264']
    logging.info(f"[+] Device: {device}")


class SubtitleGenerator:
    """
    Generates subtitles with karaoke and typewriter effects for video files.
    Requires FFmpeg to be installed and available in the system PATH.
    """

    def __init__(self, video_path: str, output_dir: str = str(SUBTITLES_DIR)):
        """
        Initializes the subtitle generator.

        Args:
            video_path (str): Path to the input video file.
            output_dir (str): Directory to save output files.
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.audio_path = self.output_dir / f"{self.video_path.stem}_audio.mp3"
        self.transcription_data = None

    def _extract_audio(self):
        """Extracts the audio track from the video using MoviePy."""
        print(f"Extracting audio from '{self.video_path}'...")
        video_clip = VideoFileClip(str(self.video_path))
        video_clip.audio.write_audiofile(str(self.audio_path), logger=None)
        print("Audio extracted successfully.")

    def _transcribe_audio(self):
        """Transcribes audio to text with word-level timestamps using Whisper."""
        print("Transcribing audio. This may take several minutes...")
        model = whisper.load_model("base")  # You can choose "small" or "large" for higher accuracy
        result = model.transcribe(str(self.audio_path), word_timestamps=True)
        self.transcription_data = result['segments']
        print("Transcription completed.")

    def _format_time(self, seconds: float) -> str:
        """Formats seconds into ASS timestamp (h:mm:ss.cs)."""
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:05.2f}"

    def _generate_ass_file_karaoke(self, ass_path: Path):
        """
        Generates a karaoke-style ASS subtitle file with word highlighting.

        Each word is highlighted with a temporary style change, simulating a karaoke effect.

        Styles:
            - Default: White text with black outline.
            - Highlight: White text with black outline, no background rectangle.

        Args:
            ass_path (Path): Path to save the generated .ass subtitle file.
        """
        with open(ass_path, "w", encoding="utf-8") as f:
            # ---------------- Script Info ----------------
            f.write("[Script Info]\n")
            f.write("ScriptType: v4.00+\n")
            f.write("Collisions: Normal\n")
            f.write("PlayResX: 1280\n")
            f.write("PlayResY: 720\n")
            f.write("WrapStyle: 0\n\n")

            # ---------------- Styles ----------------
            f.write("[V4+ Styles]\n")
            f.write(
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
                "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            )
            # Default style: white text with black outline
            f.write(
                "Style: Default,Candara-Bold,35,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
                "0,0,0,0,100,100,0,0,1,2,2,2,10,10,20,1\n"
            )
            # Highlight style: white text, black outline, no background rectangle
            f.write(
                "Style: Highlight,Candara-Bold,35,&H00FFFFFF,&H00000000,&H00000000,&H00000000,"
                "0,0,0,0,100,100,0,0,1,2,2,2,10,10,20,1\n"
            )
            f.write("\n")

            # ---------------- Events ----------------
            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

            for segment in self.transcription_data:
                start_time = self._format_time(segment['start'])
                end_time = self._format_time(segment['end'])
                text = ""

                for word in segment['words']:
                    word_text = word['word'].strip()
                    if not word_text:
                        continue
                    # Duration of the word in centiseconds
                    duration = int((word['end'] - word['start']) * 100)
                    # \k = karaoke timing, \rHighlight = temporary style change
                    text += f"{{\\k{duration}\\rHighlight}}{word_text} "

                # Write dialogue line with Default style, Highlight triggers for each word
                f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")

        print(f"Karaoke ASS file generated at: {ass_path}")

    def _generate_ass_file_typewriter(self, ass_path: Path):
        """
        Generates an ASS subtitle file with a typewriter effect.

        Each line appears progressively over its duration, simulating a typing effect.
        True per-letter animation is complex; uses \k tags for timing per character.

        Style:
            - Typewriter: Candara-Bold, size 35, white text, black outline.

        Args:
            ass_path (Path): Path to save the generated .ass subtitle file.
        """
        with open(ass_path, "w", encoding="utf-8") as f:
            # ---------------- Script Info ----------------
            f.write("[Script Info]\n")
            f.write("ScriptType: v4.00+\n")
            f.write("Collisions: Normal\n")
            f.write("PlayResX: 1280\n")
            f.write("PlayResY: 720\n\n")

            # ---------------- Styles ----------------
            f.write("[V4+ Styles]\n")
            f.write(
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
                "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            )
            # Typewriter style: white text with black outline
            f.write(
                "Style: Typewriter,Candara-Bold,35,&H00FFFFFF,&H00000000,&H00000000,&H00000000,"
                "0,0,0,0,100,100,0,0,1,2,2,2,10,10,20,1\n"
            )
            f.write("\n")

            # ---------------- Events ----------------
            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

            for segment in self.transcription_data:
                start_time = self._format_time(segment['start'])
                end_time = self._format_time(segment['end'])
                full_text = segment['text'].strip()

                # Duration per letter in centiseconds
                segment_duration_cs = int((segment['end'] - segment['start']) * 100)
                per_letter_duration = max(1, segment_duration_cs // max(len(full_text), 1))

                # Build typewriter text using \k tags
                typewriter_text = ""
                for letter in full_text:
                    typewriter_text += f"{{\\k{per_letter_duration}}}{letter}"

                f.write(f"Dialogue: 0,{start_time},{end_time},Typewriter,,0,0,0,,{typewriter_text}\n")

        print(f"Typewriter ASS file generated at: {ass_path}")

    def _run_ffmpeg(self, ass_path: Path, output_filename: str):
        """
        Runs FFmpeg to embed the ASS subtitles into the video.

        Args:
            ass_path (Path): Path to the .ass subtitle file.
            output_filename (str): Output video filename.
        """
        output_video_path = self.output_dir / output_filename
        formatted_subtitles_path = str(ass_path).replace("\\", "/").replace(":", "\\:")

        logging.info(f"Formatted subtitles path for FFmpeg: {formatted_subtitles_path}")

        command = base_cmd + [
            "-fflags", "+genpts", "-err_detect", "ignore_err",
            "-i", str(self.video_path).replace("\\", "/"),
            "-vf", f"ass='{formatted_subtitles_path}'",
            "-c:a", "copy",
            "-fps_mode", "passthrough",
            "-y",
        ] + codec_cmd + [
            "-preset", "fast",
            "-b:a", "192k",
            "-ar", "48000",
            "-movflags", "+faststart",
            "-bufsize", "10M",
            str(output_video_path).replace("\\", "/")
        ]

        logging.info(f"Running FFmpeg: {' '.join(command)}")

        try:
            # Set check=True to raise CalledProcessError on failure
            result = subprocess.run(command, capture_output=True, text=True, check=True)

            # If successful, print success message and any warnings from stdout/stderr
            print(f"Video with subtitles saved at: {output_video_path}")

            # Log FFmpeg's normal output (useful for debugging warnings)
            if result.stdout:
                logging.info(f"FFmpeg stdout:\n{result.stdout.strip()}")
            if result.stderr:
                logging.warning(f"FFmpeg stderr (Warnings/Info):\n{result.stderr.strip()}")

        except subprocess.CalledProcessError as e:
            print(f"Return code: {e.returncode}")
            print(f"FFmpeg stdout:\n{e.stdout}")
            print(f"FFmpeg stderr:\n{e.stderr}")
            raise

    def generate_karaoke_subtitles(self, output_filename: str = "video_karaoke.mp4"):
        """Generates karaoke subtitles and embeds them in the video."""
        self._extract_audio()
        self._transcribe_audio()
        ass_path = self.output_dir / f"{self.video_path.stem}_karaoke.ass"
        self._generate_ass_file_karaoke(ass_path)
        # self._run_ffmpeg(ass_path, output_filename)
        print("Karaoke subtitle process completed.")

    def generate_typewriter_subtitles(self, output_filename: str = "video_typewriter.mp4") -> str:
        """Generates typewriter subtitles and embeds them in the video."""
        self._extract_audio()
        self._transcribe_audio()
        ass_path = self.output_dir / f"{self.video_path.stem}_typewriter.ass"
        self._generate_ass_file_typewriter(ass_path)
        # self._run_ffmpeg(ass_path, output_filename)
        print("Typewriter subtitle process completed.")

        return str(ass_path)


if __name__ == "__main__":
    # ---------------- Example Usage ----------------
    video_to_process = r"F:\Development\PyCharm\Projects\PodcastAvatarsApp\podcast_content\temp\f1ed75be-c11b-4d73-a76c-2722d3560086.mp4"

    if not os.path.exists(video_to_process):
        print(f"Error: Video file '{video_to_process}' not found.")
    else:
        generator = SubtitleGenerator(video_to_process)

        # Generate karaoke subtitles
        print("\n--- Generating karaoke subtitles ---")
        generator.generate_karaoke_subtitles()

        # Generate typewriter subtitles
        print("\n--- Generating typewriter subtitles ---")
        generator.generate_typewriter_subtitles()
