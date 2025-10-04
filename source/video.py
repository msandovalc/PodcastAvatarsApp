import os
import uuid
import gc
import math
import ffmpeg
import requests
import srt_equalizer
import assemblyai as aai
from uuid import uuid4
import srt
from datetime import timedelta
import re

from pathlib import Path
from moviepy.editor import *
from termcolor import colored
from dotenv import load_dotenv
from datetime import timedelta
from typing import List, Optional, Tuple
from moviepy.video.fx.all import crop
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.tools.subtitles import SubtitlesClip
import concurrent.futures
from logger_config import setup_logger

import subprocess
import time
import shlex
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm
import torch

# --- Configuration ---
logging = setup_logger(__name__)

# Load environment variables
dotenv_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

ASSEMBLY_AI_API_KEY = os.getenv("ASSEMBLY_AI_API_KEY")

# --- Constants for Directory Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "audiobook_content" / "temp"
SUBTITLES_DIR = BASE_DIR / "audiobook_content" / "subtitles"

# Get device
if torch.cuda.is_available():
    device = "cuda"
    # base_cmd = ["ffmpeg", "-y", '-hwaccel', 'cuda']
    # base_cmd = ["ffmpeg", "-y", '-hwaccel', 'cuda', "-loglevel", "debug"]
    base_cmd = ["ffmpeg", "-y", '-hwaccel', 'cuda']
    codec_cmd = ['-c:v', 'h264_nvenc']
    logging.info(f"[+] Device: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    base_cmd = ["ffmpeg", "-y"]
    codec_cmd = ['-c:v', 'libx264']
    logging.info(f"[+] Device: {device}")


def get_video_duration(path):
    """
    Use ffprobe to get the duration of a video file in seconds.
    """
    probe_command = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        path
    ]
    try:
        result = subprocess.run(probe_command, capture_output=True, text=True, check=True)
        if result.returncode != 0:
            logging.error(f"ffprobe error: {result.stderr}")
            raise Exception(f"ffprobe failed with error: {result.stderr}")
        duration_str = result.stdout.strip()
        try:
            duration = float(duration_str)
            logging.debug(f"ffprobe returned duration: {duration}")
            return duration
        except ValueError:
            logging.error(f"ffprobe returned invalid duration: {duration_str}")
            raise ValueError(f"Invalid duration format: {duration_str}")
    except Exception as e:
        logging.warning(f"Could not get duration for {path}: {e}")
        return 0


def generate_ffmpeg_command(image_path: str, output_path: str, duration: float, effect: str, zoom_speed: float = 0.0005,
                            max_zoom: float = 1.5) -> Optional[List[str]]:
    """
    Generates an FFmpeg command to apply visual effects to a single image and convert it to a short video.

    Args:
        image_path (str): Path to the input image file.
        output_path (str): Path to save the output video file.
        duration (float): Duration of the output video in seconds.
        effect (str): The visual effect to apply ('zoom_in', 'zoom_out', 'fade_in', 'fade_out',
                       'smoothleft', 'smoothright', 'vertopen', 'vertclose').
        zoom_speed (float): Speed of the zoom effect (default: 0.0005).
        max_zoom (float): Maximum zoom level for zoom out effect (default: 1.5).

    Returns:
        Optional[List[str]]: A list containing the FFmpeg command, or None if an error occurred.
    """
    logging.info(f"Generating FFmpeg command for image: {image_path}, effect: {effect}, duration: {duration}s")

    try:
        # Validate input paths
        if not os.path.exists(image_path):
            logging.error(f"Error: Image file not found at {image_path}")
            return None
        if not output_path:
            logging.error("Error: Output path cannot be empty")
            return None

        # Retrieve image dimensions using ffprobe
        probe_command = ['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', image_path]
        probe_result = subprocess.run(probe_command, capture_output=True, text=True, check=True)
        width_str, height_str = probe_result.stdout.strip().split(',')
        width, height = int(width_str), int(height_str)
        logging.debug(f"Image dimensions: width={width}, height={height}")

        # Calculate frame counts based on assumed frame rates for effects
        zoom_frames = round(duration * 60)
        fade_frames = round(duration * 24)

        # # Fixed output video dimensions
        # output_size = '1080x1920'
        # resize_width = 1080
        # resize_height = 1920

        # Set final resolution for YouTube (16:9 aspect ratio)
        output_size = '1920x1080'
        final_width, final_height = 1920, 1080

        x_center, y_center = 'iw/2', 'ih/2'

        # command = ['ffmpeg', '-y', "-fflags", "+genpts", "-err_detect", "ignore_err", '-loop', '1',
        #            '-i', image_path, '-t', str(duration)]

        vf_options = []
        # Apply the specified effect
        if effect == 'zoom_in':
            vf_options = [
                '-vf',
                f"scale=2000:-1,zoompan=z='zoom+{zoom_speed}':x={x_center}-(iw/zoom/2):y={y_center}-(ih/zoom/2):d={zoom_frames}:s={output_size}:fps=60"
            ]
        elif effect == 'zoom_out':
            vf_options = [
                '-vf',
                f"scale=2000:-1,zoompan=z='if(lte(zoom,1.0),{max_zoom},max(1.001,zoom-{zoom_speed}))':x={x_center}-(iw/zoom/2):y={y_center}-(ih/zoom/2):d={zoom_frames}:s={output_size}:fps=60"
            ]
        elif effect == 'fade_in':
            if fade_frames > 0:
                vf_options = [
                    '-vf',
                    f"scale={output_size},format=yuv420p,fade=in:st=0:d={fade_frames / 24}"
                ]
            else:
                logging.warning("fade_in duration is too short, skipping effect.")
        elif effect == 'fade_out':
            fade_start = max(0, duration - fade_frames / 24)
            if fade_frames > 0:
                vf_options = [
                    '-vf',
                    f"scale={output_size},format=yuv420p,fade=out:st={fade_start}:d={min(fade_frames / 24, duration)}"
                ]
            else:
                logging.warning("fade_out duration is too short, skipping effect.")
        elif effect == 'smoothleft':
            smooth_duration = duration
            fps = 24
            vf_options = [
                '-vf',
                f"crop={1080}:ih:x='if(between(t,0,{smooth_duration}),(iw-1080)*(t/{smooth_duration}),0)',"
                f"scale={output_size},format=yuv420p,fps={fps}"
            ]
        elif effect == 'smoothright':
            smooth_duration = duration
            smooth_factor = 1.5
            fps = 60
            vf_options = [
                '-vf',
                f"crop={1080}:ih:x='if(between(t,0,{smooth_duration}),(iw-1080)*(t/{smooth_duration})/{smooth_factor},0)',"
                f"scale={output_size},format=yuv420p,fps={fps}"
            ]
        elif effect == 'vertopen':
            smooth_duration = duration
            fps = 24
            smooth_factor = 1.5
            scale_percent = 1.0
            vf_options = [
                '-vf',
                f"scale={width * scale_percent}:{height * scale_percent},"
                f"crop={1080}:1920:0:'if(between(t,0,{smooth_duration}),floor((ih - 1920) * (t/{smooth_duration})/{smooth_factor}), (ih - {height}))',"
                f"scale={output_size},format=yuv420p,fps={fps}"
            ]
        elif effect == 'vertclose':
            smooth_duration = duration
            fps = 24
            smooth_factor = 1.5
            scale_percent = 1.0
            vf_options = [
                '-vf',
                f"scale={width * scale_percent}:{height * scale_percent},"
                f"crop={1080}:1920:0:'if(between(t,0,{smooth_duration}),floor((ih - 1920) - (ih - 1920)*(t/{smooth_duration})/{smooth_factor}), (ih - {height}))',"
                f"scale={output_size},format=yuv420p,fps={fps}"
            ]
        else:
            raise ValueError(
                f"Invalid effect type: {effect}. Supported effects are 'zoom_in', 'zoom_out', 'fade_in', 'fade_out', 'smoothleft', 'smoothright', 'vertopen', 'vertclose'.")

        # Final encoding settings
        command = (base_cmd + [
            "-fflags", "+genpts",
            "-err_detect", "ignore_err",
            '-loop', '1',
            '-i', image_path,
            '-t', str(duration),
        ] + vf_options +
                   codec_cmd + [
                       "-preset", "fast",
                       '-r', '30', "-bufsize",
                       "10M", "-an",
                       output_path
                   ])

        # Final encoding settings
        # command = (base_cmd + hw_accel_options +[
        #     "-fflags", "+genpts",
        #     "-err_detect", "ignore_err",
        #     '-loop', '1',
        #     '-i', image_path,
        #     '-t', str(duration),
        # ] + codec_cmd + [
        #                "-preset", "fast",
        #                '-r', '30', "-bufsize",
        #                "10M", "-an",
        #                output_path
        #            ])

        return command

    except subprocess.CalledProcessError as e:
        logging.error(f"Error running ffprobe: {e}")
        return None
    except ValueError as e:
        logging.error(f"Invalid parameter: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during command generation: {e}")
        return None


def save_video(video_url: str, directory: str = "../temp", timeout: int = 30) -> str:
    """
    Saves only the first 60 seconds (~20MB) of a video from a given URL with a progress bar.

    Args:
        video_url (str): The URL of the video to save.
        directory (str): The directory to save the video to.
        timeout (int): Maximum time (in seconds) to wait for a response.

    Returns:
        str: The path to the saved video.
    """
    video_id = uuid.uuid4()
    video_path = Path(__file__).resolve().parent.parent / "temp" / f"{video_id}.mp4"
    Path(video_path).parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        print(f"[+] Starting partial video download from URL: {video_url}")

        chunk_size = 1024 * 1024  # 1MB chunks
        max_download_size = 30_000_000  # ~30MB
        headers = {"Range": f"bytes=0-{max_download_size - 1}"}  # Corrected header

        start_time = time.time()

        with session.get(video_url, headers=headers, stream=True, timeout=timeout) as response:
            response.raise_for_status()

            # Use `min(content-length, max_download_size)` to avoid overestimating
            total_size = min(int(response.headers.get("content-length", max_download_size)), max_download_size)

            with open(video_path, "wb") as f, tqdm(
                    total=total_size, unit="B", unit_scale=True, desc="Downloading", leave=True
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
                        if f.tell() >= max_download_size:
                            break  # Stop after downloading ~60s

        elapsed_time = time.time() - start_time
        print(f"[+] Video downloaded in {elapsed_time:.2f} seconds and saved to {video_path}")

        return str(video_path)

    except requests.exceptions.Timeout:
        logging.error("Download timed out.")
        raise ValueError("The video download took too long and timed out.")

    except requests.exceptions.RequestException as e:
        logging.error(f"Error during video download: {str(e)}")
        raise ValueError(f"Failed to download video from URL: {video_url}")

    except IOError as e:
        logging.error(f"Error saving the video to {video_path}: {str(e)}")
        raise ValueError(f"Failed to save video to path: {video_path}")

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise


def __generate_subtitles_assemblyai(audio_path: str, voice: str) -> Tuple[Optional[str], str]:
    """
    Generates subtitles from a given audio file using AssemblyAI, with error handling and logging.

    Args:
        audio_path (str): The path to the audio file to generate subtitles from.
        voice (str): The voice code, used to determine the language for transcription.

    Returns:
        Tuple[Optional[str], str]: A tuple containing the generated subtitles (as a string) and the status
                                     of the processing ('completed' if successful, 'error' otherwise).
    """
    logging.info(f"Starting subtitle generation for: {audio_path} in language: {voice} (AssemblyAI)")
    language_mapping = {
        "br": "pt",
        "id": "en",  # AssemblyAI doesn't have Indonesian
        "jp": "ja",
        "kr": "ko",
    }

    lang_code = language_mapping.get(voice, voice)
    aai.settings.api_key = ASSEMBLY_AI_API_KEY
    transcript = None
    subtitles = None
    status = 'pending'

    try:
        config = aai.TranscriptionConfig(language_code=lang_code)
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio_path)
        status = transcript.status
        logging.info(f"Transcription status (ID: {transcript.id}): {status}")

        if status == 'completed':
            subtitles = transcript.export_subtitles_srt()
            logging.info(f"Subtitles generated successfully for (ID: {transcript.id}).")
        elif status == 'error':
            logging.error(f"Transcription (ID: {transcript.id}) finished with error. Details: {transcript.error}")
        else:
            logging.warning(
                f"Transcription (ID: {transcript.id}) is not in 'completed' or 'error' state. Current status: {status}")

    except aai.TranscriptError as e:  # Capture TranscriptError for transcription-specific errors
        logging.error(f"AssemblyAI Transcript error during transcription: {e}")
        status = 'error'
    except aai.AssemblyAIError as e:  # Capture the base AssemblyAIError for other API-related issues
        logging.error(f"AssemblyAI API error during transcription: {e}")
        status = 'error'
    except Exception as e:
        logging.error(f"An unexpected error occurred during subtitle generation: {e}")
        status = 'error'

    return subtitles, status


def __generate_subtitles_locally(sentences: List[str], audio_clips: List[AudioFileClip]) -> str:
    """
    Generates subtitles from a given audio file and returns the path to the subtitles.

    Args:
        sentences (List[str]): all the sentences said out loud in the audio clips
        audio_clips (List[AudioFileClip]): all the individual audio clips which will make up the final audio track
    Returns:
        str: The generated subtitles
    """

    def convert_to_srt_time_format(total_seconds):
        # Convert total seconds to the SRT time format: HH:MM:SS,mmm
        if total_seconds == 0:
            return "0:00:00,0"
        return str(timedelta(seconds=total_seconds)).rstrip('0').replace('.', ',')

    start_time = 0
    subtitles = []

    for i, (sentence, audio_clip) in enumerate(zip(sentences, audio_clips), start=1):
        duration = audio_clip.duration
        end_time = start_time + duration

        # Format: subtitle index, start time --> end time, sentence
        subtitle_entry = f"{i}\n{convert_to_srt_time_format(start_time)} --> {convert_to_srt_time_format(end_time)}\n{sentence}\n"
        subtitles.append(subtitle_entry)

        start_time += duration  # Update start time for the next subtitle

    return "\n".join(subtitles)


def generate_subtitles(audio_path: str, sentences: List[str] = None, audio_clips: List[AudioFileClip] = None,
                       voice: str = "es") -> str:
    """
    Generates subtitles from a given audio file and returns the path to the subtitles.

    Args:
        audio_path (str): The path to the audio file to generate subtitles from.
        sentences (List[str]): All the sentences spoken in the audio clips.
        audio_clips (List[AudioFileClip]): All the individual audio clips that make up the final audio track.

    Returns:
        str: The path to the generated subtitles, or None on error.
    """

    def _equalize_subtitles(srt_path: str, max_chars: int = 40) -> None:
        """
        Equalizes subtitle timings and line lengths in an SRT file.

        This function reads the SRT file with proper UTF-8 encoding, then
        parses, modifies, and re-composes the subtitles, ensuring that
        long lines are split for better readability. It avoids encoding
        issues by handling all file I/O with explicit UTF-8 encoding.

        Args:
            srt_path (str): Path to the SRT file.
            max_chars (int): Maximum characters per subtitle line.
        """
        logging.info(f"Starting to equalize subtitles for: {srt_path}")
        try:
            # Read the file with the correct UTF-8 encoding
            with open(srt_path, 'r', encoding='utf-8') as f:
                raw_srt = f.read()

            # Use the 'srt' library to parse the content
            subtitles = list(srt.parse(raw_srt))
            new_subtitles = []

            for sub in subtitles:
                # Check for illegal blank lines or multiple newlines
                content = sub.content.strip()

                if not content:
                    continue

                lines = []
                current_line = ""
                words = content.split()

                for word in words:
                    if len(current_line) + len(word) + 1 <= max_chars:
                        if current_line:
                            current_line += " "
                        current_line += word
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)

                # Re-join the lines with a newline separator
                new_content = "\n".join(lines)

                # Create a new subtitle object with the modified content
                new_sub = srt.Subtitle(
                    index=sub.index,
                    start=sub.start,
                    end=sub.end,
                    content=new_content,
                    proprietary=sub.proprietary
                )
                new_subtitles.append(new_sub)

            # Re-index the subtitles just in case
            re_indexed_subtitles = srt.sort_and_reindex(new_subtitles)

            # Re-compose the SRT file content from the new subtitle objects
            equalized_srt = srt.compose(re_indexed_subtitles)

            # Write the equalized content back to the file
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(equalized_srt)

            logging.info("Subtitles equalized successfully.")

        except FileNotFoundError:
            logging.error(f"Error: The file {srt_path} was not found.")
        except Exception as e:
            logging.error(f"An unexpected error occurred during subtitle equalization: {e}")

    subtitles_path = str(SUBTITLES_DIR / f"{uuid.uuid4()}.srt")  # Corrected path handling
    Path(subtitles_path).parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    logging.info(f"Subtitles will be saved to: {subtitles_path}")  #added logging

    subtitles_content = None
    status = 'pending'

    if ASSEMBLY_AI_API_KEY:  # Check for non-empty string
        logging.info("Using AssemblyAI for subtitle generation.")
        subtitles_content, status = __generate_subtitles_assemblyai(audio_path, voice)
        if status != 'completed':
            logging.error(f"AssemblyAI failed to generate subtitles. Status: {status}")
            return None  #early return
    else:
        logging.info("Using local engine for subtitle generation.")
        subtitles_content = __generate_subtitles_locally(sentences,
                                                         audio_clips)  # Removed the conditional block that prevented local generation
        status = 'completed'  #set status

    if not subtitles_content:  # Check if subtitles_content is None or empty
        logging.error("No subtitles were generated.")
        return ""  # Return None on error

    try:
        with open(subtitles_path, "w", encoding="utf-8", errors='replace') as file:  # added errors='replace'
            file.write(subtitles_content)
        _equalize_subtitles(subtitles_path)
        logging.info(f"Subtitles saved and equalized at: {subtitles_path}")
        return subtitles_path
    except Exception as e:
        logging.error(f"Error writing or equalizing subtitles: {e}")
        return ""


def __process_clip(video_path: str, max_duration: int, req_dur: float, max_clip_duration: int) -> 'VideoFileClip':
    """
    Processes a single video clip: loads, crops, resizes, and checks duration limits.

    Args:
        video_path (str): Path to the video file.
        max_duration (int): Maximum allowed duration of the combined video.
        req_dur (float): The calculated required duration per clip.
        max_clip_duration (int): Maximum duration for each clip.

    Returns:
        VideoFileClip: The processed video clip.
    """
    try:
        print(colored(f"[+] Processing video: {video_path}", "cyan"))
        clip = VideoFileClip(video_path)
        clip = clip.without_audio()

        print(colored(f"[+] Clip duration: {clip.duration}", "cyan"))
        print(f"req_dur: {req_dur}, max_clip_duration: {max_clip_duration}")

        # Check if clip is longer than the remaining duration (no need to use tot_dur here)
        if max_duration < clip.duration:
            clip = clip.subclip(0, max_duration)
        elif req_dur < clip.duration:
            clip = clip.subclip(0, req_dur)

        clip = clip.set_fps(30)

        # Resize the clip if necessary
        if round((clip.w / clip.h), 4) < 0.5625:
            clip = crop(clip, width=clip.w, height=round(clip.w / 0.5625), x_center=clip.w / 2, y_center=clip.h / 2)
        else:
            clip = crop(clip, width=round(0.5625 * clip.h), height=clip.h, x_center=clip.w / 2, y_center=clip.h / 2)

        clip = clip.resize((1080, 1920))

        # Limit the clip duration
        if clip.duration > max_clip_duration:
            clip = clip.subclip(0, max_clip_duration)

        print(colored(f"[+] Clip processed. Duration: {clip.duration:.2f}s", "green"))

        return clip
    except Exception as e:
        print(colored(f"[-] ERROR processing {video_path}: {e}", "red"))
        return None


def process_single_video(video_path: str, req_dur: float, max_duration: int) -> Optional[str]:
    """
    Processes a single video: trims, crops, resizes using FFmpeg.

    Args:
        video_path (str): Path to the input video.
        req_dur (float): Desired clip duration in seconds.
        max_duration (int): Maximum allowed duration for all clips combined.

    Returns:
        Optional[str]: Path to the processed temporary video clip, or None if processing failed.
    """
    logging.info(f"Processing video: {video_path}")

    # Generate a unique temporary output path
    while True:
        temp_clip_path = str(TEMP_DIR / f"{uuid.uuid4()}_{int(time.time() * 1000)}.mp4")
        if not os.path.exists(temp_clip_path):
            break

    try:
        # Get video dimensions
        probe_command = ['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', video_path]
        probe_result = subprocess.run(probe_command, capture_output=True, text=True, check=True)
        width_str, height_str = probe_result.stdout.strip().split(',')
        width, height = int(width_str), int(height_str)
        logging.debug(f"Original video dimensions: width={width}, height={height}")

        # Calculate crop parameters to maintain 9:16 aspect ratio
        target_aspect = 9 / 16
        current_aspect = width / height

        if current_aspect > target_aspect:
            crop_width = int(height * target_aspect)
            crop_x = (width - crop_width) // 2
            crop_height, crop_y = height, 0
        else:
            crop_height = int(width / target_aspect)
            crop_y = (height - crop_height) // 2
            crop_width, crop_x = width, 0

        # Set final resolution
        final_width, final_height = 1080, 1920

        command = base_cmd + [
            "-fflags", "+genpts", "-err_detect", "ignore_err",
            "-i", video_path,
        ]

        if req_dur is not None:
            command += [
                "-t", str(req_dur),
                "-force_key_frames", f"expr:gte(t,n_forced*{req_dur})",
            ]
        # Build FFmpeg command
        command += [
                       "-vf",
                       f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},fps=30,scale={final_width}:{final_height}",
                   ] + codec_cmd + [
                       "-preset", "fast",
                       "-r", "30",
                       "-bufsize", "10M",
                       "-an",  # remove audio
                       temp_clip_path
                   ]

        logging.info(f"Running FFmpeg: {' '.join(command)}")
        subprocess.run(command, check=True)

        # Validate resulting clip duration
        processed_duration = get_video_duration(temp_clip_path)
        logging.info(f"Processed clip {temp_clip_path}, duration: {processed_duration:.2f}s")

        return temp_clip_path

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            logging.error(f"ffprobe stderr: {e.stderr}")
            logging.error(f"ffprobe stdout: {e.stdout}")
            logging.error(f"ffprobe returncode: {e.returncode}")
            logging.error(f"ffprobe cmd: {e.cmd}")
    except FileNotFoundError:
        logging.error("FFmpeg not found. Make sure it’s installed and in PATH.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        gc.collect()
        if not os.path.exists(temp_clip_path) or os.path.getsize(temp_clip_path) == 0:
            if os.path.exists(temp_clip_path):
                os.remove(temp_clip_path)

    return None


def process_single_video_one_element(
        video_path: str,
        req_dur: float,
        show_progress: bool = False
) -> Optional[str]:
    """
    Processes a single video: resizes and loops to the required duration for a YouTube format.
    Optionally displays a real-time progress bar.

    Args:
        video_path (str): Path to the input video.
        req_dur (float): Desired clip duration in seconds.
        show_progress (bool): If True, a progress bar will be displayed.

    Returns:
        Optional[str]: Path to the processed temporary video clip, or None if processing failed.
    """
    logging.info(f"Processing video: {video_path}")

    # Generate a unique temporary output path
    temp_clip_path = str(TEMP_DIR / f"{uuid.uuid4()}_{int(time.time() * 1000)}.mp4")

    try:
        # Get video dimensions using ffprobe
        probe_command = ['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', video_path]
        probe_result = subprocess.run(probe_command, capture_output=True, text=True, check=True)
        width_str, height_str = probe_result.stdout.strip().split(',')
        width, height = int(width_str), int(height_str)
        logging.debug(f"Original video dimensions: width={width}, height={height}")

        # Set final resolution for YouTube (16:9 aspect ratio)
        final_width, final_height = 1920, 1080

        # Build FFmpeg command. The crop logic has been removed.
        command = base_cmd + [
            "-fflags", "+genpts", "-err_detect", "ignore_err",
            "-stream_loop", "-1",  # Loops the input infinitely.
            "-i", video_path,
            "-t", str(req_dur),
            # The `-vf` now only includes scaling to the new dimensions
            "-vf", f"scale={final_width}:{final_height},fps=30",
        ] + codec_cmd + [
                      "-preset", "fast",
                      "-r", "30",
                      "-bufsize", "10M",
                      "-an",  # remove audio
                      temp_clip_path
                  ]

        # Add the progress pipe for reading output if enabled
        if show_progress:
            command.extend(["-progress", "pipe:1"])

        logging.info(f"Executing FFmpeg command: {' '.join(command)}")
        start_time = time.time()

        if show_progress:
            bar_length = 30  # Define the length of the progress bar

            # Use Popen to read the output in real-time
            with subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stdout and stderr
                    text=True,
                    bufsize=1,
                    universal_newlines=True
            ) as process:
                for line in process.stdout:
                    # Use a regex to find the `out_time_ms` value in the progress output
                    match = re.search(r"out_time_ms=(\d+)", line)
                    if match:
                        elapsed_ms = int(match.group(1))
                        elapsed_seconds = elapsed_ms / 1000000.0

                        if req_dur > 0:
                            progress_percent = min(100.0, (elapsed_seconds / req_dur) * 100.0)

                            # Calculate the number of filled and empty characters
                            filled_chars = int(bar_length * progress_percent // 100)
                            empty_chars = bar_length - filled_chars

                            # Create the progress bar string
                            progress_bar = "Progress [" + "█" * filled_chars + " " * empty_chars + "]"

                            # Print the progress on the same line, overwriting the previous one
                            print(f"\r{progress_bar} {progress_percent:.2f}%", end="")
                            sys.stdout.flush()

                process.wait()

                # Print a final newline and a success message
                print("\rProgress [██████████████████████████████] 100.00% - Process completed successfully!")
                print()

                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, process.args)
        else:
            # Simple execution without progress tracking
            subprocess.run(command, check=True)

        end_time = time.time()
        processing_time = end_time - start_time
        logging.info(f"Successfully processed clip: {temp_clip_path}")
        logging.info(f"Processing time: {processing_time:.2f} seconds.")

        # Validate resulting clip duration
        processed_duration = get_video_duration(temp_clip_path)
        logging.info(f"Processed clip {temp_clip_path}, duration: {processed_duration:.2f}s")

        return temp_clip_path

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            logging.error(f"ffprobe stderr: {e.stderr}")
            logging.error(f"ffprobe stdout: {e.stdout}")
            logging.error(f"ffprobe returncode: {e.returncode}")
            logging.error(f"ffprobe cmd: {e.cmd}")
    except FileNotFoundError:
        logging.error("FFmpeg not found. Make sure it's installed and in PATH.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        # A simple cleanup logic to prevent issues if processing fails early.
        if not os.path.exists(temp_clip_path) or os.path.getsize(temp_clip_path) == 0:
            if os.path.exists(temp_clip_path):
                os.remove(temp_clip_path)

    return None


# def combine_videos(video_paths: List[str], max_duration: int, max_clip_duration: int, threads: int = 2) -> str:
#     """
#     Combines a list of videos into one video using FFmpeg's concat demuxer.
#
#     Args:
#         video_paths (List[str]): A list of paths to the videos to combine.
#         max_duration (int): The maximum duration of the combined video in seconds.
#         max_clip_duration (int): The maximum duration to consider from each input clip in seconds.
#         threads (int): The maximum number of concurrent threads to use for processing individual clips.
#
#     Returns:
#         str: The path to the combined video, or an empty string if an error occurred.
#     """
#
#     combined_video_path = str(TEMP_DIR / f"{uuid.uuid4()}.mp4")
#     logging.info(
#         f"Combining videos. Output path: {combined_video_path}, Max duration: {max_duration}s, Max clip duration: {max_clip_duration}s, Threads: {threads}")
#
#     valid_video_paths = [os.path.normpath(path) for path in video_paths if
#                          isinstance(path, str) and path.lower().endswith(
#                              (".mp4", ".avi", ".mov", ".mkv")) and os.path.exists(path)]
#
#     if not valid_video_paths:
#         logging.error("Error: No valid video files found to combine.")
#         return ""
#
#     num_videos = len(valid_video_paths)
#     req_dur = max_duration / num_videos if num_videos > 0 else 0
#     logging.info(f"Number of valid videos: {num_videos}, Required duration per clip: {req_dur:.2f}s")
#
#     processed_clips = [None] * num_videos  # Pre-allocate list to store results in order
#     futures_with_indices = []
#
#     total_duration = 0
#
#     with concurrent.futures.ThreadPoolExecutor(max_workers=min(2, threads)) as executor:
#         for i, video_path in enumerate(valid_video_paths):
#             clip_duration_to_process = min(req_dur, max_clip_duration)
#             future = executor.submit(process_single_video, video_path, clip_duration_to_process, max_duration)
#             futures_with_indices.append((i, future))
#
#         completed_futures = concurrent.futures.as_completed(dict(futures_with_indices).values())
#         futures_dict = dict(futures_with_indices)  # Create a dictionary for quick lookup
#
#         for future in completed_futures:
#             for original_index, f in futures_with_indices:
#                 if f is future:
#                     try:
#                         processed_clip_path = future.result()
#                         if processed_clip_path:
#                             processed_clips[original_index] = processed_clip_path
#                             duration = get_video_duration(processed_clip_path)
#                             if duration > 0:
#                                 total_duration += duration
#                                 logging.info(
#                                     f"Processed and stored clip at index {original_index}: {processed_clip_path}, duration: {duration:.2f}s")
#                             else:
#                                 logging.warning(
#                                     f"Could not determine duration of processed clip at index {original_index}: {processed_clip_path}")
#                         else:
#                             logging.warning(f"Processing for video at index {original_index} failed.")
#                     except Exception as e:
#                         logging.error(f"Exception during video processing for index {original_index}: {e}")
#                     break
#
#     logging.info(f"Final combined video duration: {total_duration:.2f}s")
#
#     final_processed_clips = [clip for clip in processed_clips if clip is not None]
#
#     logging.info(f"Generated file list for final_processed_clips: {len(final_processed_clips)}")
#
#     if not final_processed_clips:
#         logging.error("Error: No processed video clips to combine.")
#         return ""
#
#     # Create a file list for FFmpeg's concat demuxer
#     file_list_path = str(TEMP_DIR / "file_list.txt")
#
#     try:
#         with open(file_list_path, "w", encoding="utf-8") as f:
#             for clip_path in final_processed_clips:
#                 f.write(f"file '{clip_path}'\n")
#
#         logging.info(f"Generated file list for concatenation: {file_list_path}")
#
#         # Build FFmpeg command to concatenate clips listed in 'file_list_path'
#         ffmpeg_cmd = (
#                 base_cmd + [
#             "-fflags", "+genpts",
#             "-err_detect", "ignore_err",
#             "-f", "concat",
#             "-safe", "0",
#             "-i", file_list_path
#         ] +
#                 codec_cmd + [
#                     "-preset", "fast",
#                     "-cq:v", "19",
#                     "-pix_fmt", "yuv420p",
#                     combined_video_path
#                 ]
#         )
#
#         logging.info(f"Executing final FFmpeg command for combining videos: {ffmpeg_cmd}")
#         subprocess.run(ffmpeg_cmd, check=True)
#         logging.info(f"Successfully combined videos at: {combined_video_path}")
#
#         # # Clean up temporary processed clips
#         # for clip_path in processed_clips:
#         #     try:
#         #         os.remove(clip_path)
#         #         logging.debug(f"Removed temporary clip: {clip_path}")
#         #     except OSError as e:
#         #         logging.warning(f"Could not remove temporary clip {clip_path}: {e}")
#         try:
#             os.remove(file_list_path)
#             logging.debug(f"Removed temporary file list: {file_list_path}")
#         except OSError as e:
#             logging.warning(f"Could not remove temporary file list {file_list_path}: {e}")
#
#         final_duration = get_video_duration(combined_video_path)
#         logging.info(f"Final combined video duration: {final_duration:.2f} seconds")
#
#         return combined_video_path
#
#     except FileNotFoundError:
#         logging.error("Error: FFmpeg not found. Ensure it's installed and in your PATH.")
#         return ""
#     except subprocess.CalledProcessError as e:
#         logging.error(f"FFmpeg error during video concatenation: {e}")
#         return ""
#     except Exception as e:
#         logging.error(f"An unexpected error occurred during video combination: {e}")
#         return ""
#     finally:
#         gc.collect()


def combine_videos(video_paths: List[str],
                   max_duration: int,
                   max_clip_duration: int,
                   threads: int = 2,
                   consider_duration: bool = True) -> str:
    """
    Combines a list of videos into one video using FFmpeg's concat demuxer.

    Args:
        video_paths (List[str]): A list of paths to the videos to combine.
        max_duration (int): The maximum duration of the combined video in seconds.
        max_clip_duration (int): The maximum duration to consider from each input clip in seconds.
        threads (int): The maximum number of concurrent threads to use for processing individual clips.
        consider_duration (bool): Whether to consider max_duration and max_clip_duration in processing.

    Returns:
        str: The path to the combined video, or an empty string if an error occurred.
    """

    combined_video_path = str(TEMP_DIR / f"{uuid.uuid4()}.mp4")
    logging.info(
        f"Combining videos. Output path: {combined_video_path}, "
        f"Max duration: {max_duration}s, Max clip duration: {max_clip_duration}s, "
        f"Threads: {threads}, Consider duration: {consider_duration}"
    )

    valid_video_paths = [os.path.normpath(path) for path in video_paths if
                         isinstance(path, str) and path.lower().endswith(
                             (".mp4", ".avi", ".mov", ".mkv")) and os.path.exists(path)]

    if not valid_video_paths:
        logging.error("Error: No valid video files found to combine.")
        return ""

    num_videos = len(valid_video_paths)

    if consider_duration:
        req_dur = max_duration / num_videos if num_videos > 0 else 0
        logging.info(f"Number of valid videos: {num_videos}, Required duration per clip: {req_dur:.2f}s")
    else:
        req_dur = None
        logging.info(f"Number of valid videos: {num_videos}, Duration calculations are skipped.")

    processed_clips = [None] * num_videos  # Pre-allocate list to store results in order
    futures_with_indices = []

    total_duration = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(2, threads)) as executor:
        for i, video_path in enumerate(valid_video_paths):
            clip_duration_to_process = min(req_dur, max_clip_duration) if consider_duration else None
            future = executor.submit(process_single_video, video_path, clip_duration_to_process,
                                     max_duration if consider_duration else None)
            futures_with_indices.append((i, future))

        completed_futures = concurrent.futures.as_completed(dict(futures_with_indices).values())
        futures_dict = dict(futures_with_indices)  # Create a dictionary for quick lookup

        for future in completed_futures:
            for original_index, f in futures_with_indices:
                if f is future:
                    try:
                        processed_clip_path = future.result()
                        if processed_clip_path:
                            processed_clips[original_index] = processed_clip_path
                            if consider_duration:
                                duration = get_video_duration(processed_clip_path)
                                if duration > 0:
                                    total_duration += duration
                                    logging.info(
                                        f"Processed and stored clip at index {original_index}: {processed_clip_path}, duration: {duration:.2f}s")
                                else:
                                    logging.warning(
                                        f"Could not determine duration of processed clip at index {original_index}: {processed_clip_path}")
                        else:
                            logging.warning(f"Processing for video at index {original_index} failed.")
                    except Exception as e:
                        logging.error(f"Exception during video processing for index {original_index}: {e}")
                    break

    if consider_duration:
        logging.info(f"Final combined video duration: {total_duration:.2f}s")

    final_processed_clips = [clip for clip in processed_clips if clip is not None]

    logging.info(f"Generated file list for final_processed_clips: {len(final_processed_clips)}")

    if not final_processed_clips:
        logging.error("Error: No processed video clips to combine.")
        return ""

    # Create a file list for FFmpeg's concat demuxer
    file_list_path = str(TEMP_DIR / "file_list.txt")

    try:
        with open(file_list_path, "w", encoding="utf-8") as f:
            for clip_path in final_processed_clips:
                f.write(f"file '{clip_path}'\n")

        logging.info(f"Generated file list for concatenation: {file_list_path}")

        # Build FFmpeg command to concatenate clips listed in 'file_list_path'
        ffmpeg_cmd = (
                base_cmd + [
            "-fflags", "+genpts",
            "-err_detect", "ignore_err",
            "-f", "concat",
            "-safe", "0",
            "-i", file_list_path
        ] +
                codec_cmd + [
                    "-preset", "fast",
                    "-cq:v", "19",
                    "-pix_fmt", "yuv420p",
                    combined_video_path
                ]
        )

        logging.info(f"Executing final FFmpeg command for combining videos: {ffmpeg_cmd}")
        subprocess.run(ffmpeg_cmd, check=True)
        logging.info(f"Successfully combined videos at: {combined_video_path}")

        try:
            os.remove(file_list_path)
            logging.debug(f"Removed temporary file list: {file_list_path}")
        except OSError as e:
            logging.warning(f"Could not remove temporary file list {file_list_path}: {e}")

        final_duration = get_video_duration(combined_video_path)
        logging.info(f"Final combined video duration: {final_duration:.2f} seconds")

        return combined_video_path

    except FileNotFoundError:
        logging.error("Error: FFmpeg not found. Ensure it's installed and in your PATH.")
        return ""
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error during video concatenation: {e}")
        return ""
    except Exception as e:
        logging.error(f"An unexpected error occurred during video combination: {e}")
        return ""
    finally:
        gc.collect()


def create_video_from_images(image_folder, audio_path, threads: int) -> []:
    try:
        # Crear una carpeta temporal para los clips generados
        # output_temp_folder = "../temp/temp_clips"
        # os.makedirs(output_temp_folder, exist_ok=True)

        output_temp_folder = Path(__file__).resolve().parent.parent / "temp"
        # Creates the directory
        Path(output_temp_folder).parent.mkdir(parents=True, exist_ok=True)
        output_temp_folder = str(output_temp_folder)

        image_files = sorted([
            os.path.normpath(os.path.join(image_folder, img))
            for img in os.listdir(image_folder)
            if img.endswith(('png', 'jpg', 'jpeg', 'jfif'))
        ])
        # print("Images: ", image_files)
        audio_clip = AudioFileClip(audio_path)
        duration_per_image = audio_clip.duration / len(image_files)

        clips = []
        clips_path = []
        for img in image_files:
            print("Image: ", img)
            clip = ImageSequenceClip([img], durations=[duration_per_image])
            clip = clip.set_duration(duration_per_image).fx(lambda c: c.resize(height=720))

            # Guardar el clip temporalmente para revisión
            temp_clip_filename = os.path.normpath(
                os.path.join(output_temp_folder, f"temp_clip_{os.path.splitext(os.path.basename(img))[0]}.mp4"))
            clip.write_videofile(temp_clip_filename, codec='libx264', fps=24)  # Guardar el clip temporal

            # Agregar el clip a la lista de clips
            clips.append(clip)
            clips_path.append(temp_clip_filename)

        print(f"Total de clips cargados: {len(clips)}")  # Verificar cuántos clips hay

        return clips_path
    except Exception as e:
        print(colored(f"[-] Error creating video from images: {e}", "red"))
        return None


def create_video_from_images_ffmpeg(
        image_path: str,
        req_dur: float,
        effect: str,
        threads: int,
        show_progress: bool = False
) -> Optional[str]:
    """
    Creates a short video clip from a single image using FFmpeg, applying a specified visual effect.
    Optionally displays a real-time progress bar.

    Args:
        image_path (str): Path to the input image file.
        req_dur (float): The requested duration of the output video in seconds.
        effect (str): The visual effect to apply.
        threads (int): The number of threads to use (currently not directly used by FFmpeg).
        show_progress (bool): If True, a progress bar will be displayed.

    Returns:
        Optional[str]: Path to the generated video clip, or None if an error occurred.
    """
    logging.info(f"Starting video creation from image: {image_path}, effect: {effect}")

    video_path = str(TEMP_DIR / f"{uuid.uuid4()}.mp4")

    # Validate input image path
    if not isinstance(image_path, str) or not image_path.lower().endswith((".png", ".webp", ".jpg", ".jpeg", ".jfif")):
        logging.error(f"Error: Invalid image file path: {image_path}")
        return None

    # Generate FFmpeg command with or without the progress flag
    ffmpeg_command = generate_ffmpeg_command(image_path, video_path, req_dur, effect)

    if show_progress:
        # Add the progress pipe for reading output
        ffmpeg_command.extend(["-progress", "pipe:1"])

    if not ffmpeg_command:
        logging.error("Failed to generate FFmpeg command.")
        return None

    logging.info(f"Executing FFmpeg command: {' '.join(ffmpeg_command)}")
    start_time = time.time()

    try:
        if show_progress:
            bar_length = 30  # Define the length of the progress bar

            # Use Popen to read the output in real-time
            with subprocess.Popen(
                    ffmpeg_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stdout and stderr
                    text=True,
                    bufsize=1,
                    universal_newlines=True
            ) as process:
                for line in process.stdout:
                    # Use a regex to find the `out_time_ms` value in the progress output
                    match = re.search(r"out_time_ms=(\d+)", line)
                    if match:
                        elapsed_ms = int(match.group(1))
                        elapsed_seconds = elapsed_ms / 1000000.0

                        if req_dur > 0:
                            progress_percent = min(100.0, (elapsed_seconds / req_dur) * 100.0)

                            # Calculate the number of filled and empty characters
                            filled_chars = int(bar_length * progress_percent // 100)
                            empty_chars = bar_length - filled_chars

                            # Create the progress bar string
                            progress_bar = "Progress [" + "█" * filled_chars + " " * empty_chars + "]"

                            # Print the progress on the same line, overwriting the previous one
                            print(f"\r{progress_bar} {progress_percent:.2f}%", end="")
                            sys.stdout.flush()

                process.wait()

                # Print a final newline and a success message
                print("\rProgress [██████████████████████████████] 100.00% - Process completed successfully!")
                print()

                if process.returncode != 0:
                    # If there's an error, FFmpeg will return a non-zero code
                    raise subprocess.CalledProcessError(process.returncode, process.args)

        else:
            # Simple execution without progress tracking
            subprocess.run(ffmpeg_command, check=True)

        end_time = time.time()
        processing_time = end_time - start_time
        logging.info(f"Successfully generated video clip: {video_path}")
        logging.info(f"Processing time: {processing_time:.2f} seconds.")
        return video_path

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg command failed with error: {e}")
        return None
    except FileNotFoundError:
        logging.error("Error: FFmpeg not found. Ensure it is installed and in your system's PATH.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during FFmpeg execution: {e}")
        return None


def generate_video(combined_video_path: str, tts_path: str, subtitles_path: str, cover_image_path: str,
                   end_image_path: str, logo_path: str, music_path: str, subtitles_position: str, threads: int,
                   text_color: str) -> str:
    """
    Generates the final video by adding audio, subtitles, cover image, end image, and logo.

    Args:
        combined_video_path (str): Path to the combined video without audio or overlays.
        tts_path (str): Path to the text-to-speech audio file.
        subtitles_path (str): Path to the SRT subtitle file.
        cover_image_path (str): Path to the cover image file.
        end_image_path (str): Path to the end image file.
        logo_path (str): Path to the logo image file.
        subtitles_position (str): Comma-separated string defining the horizontal and vertical position of subtitles (e.g., "center,bottom").
        threads (int): Number of threads to use for video processing (currently not directly used in the FFmpeg command).
        text_color (str): Color of the subtitles text (not directly used in the FFmpeg command).

    Returns:
        str: The path to the final generated video file.
    """
    output_path = str(TEMP_DIR)
    output_temp_folder = str(TEMP_DIR / f"output_{uuid.uuid4()}.mp4")

    logging.info(f"Starting final video generation. Output path: {output_path}")

    try:
        # Validate input file paths
        if not all(os.path.exists(path) for path in
                   [combined_video_path, tts_path, subtitles_path, cover_image_path, end_image_path, logo_path]):
            logging.error("Error: One or more input files not found.")
            if not os.path.exists(combined_video_path):
                logging.error(f"Error: Combined video file not found at: {combined_video_path}")
                print(f"Combined video file not found.")
            elif not os.path.exists(tts_path):
                logging.error(f"Error: TTS audio file not found at: {tts_path}")
                print(f"TTS audio file not found.")
            elif not os.path.exists(subtitles_path):
                logging.error(f"Error: Subtitles file not found at: {subtitles_path}")
                print(f"Subtitles file not found.")
            elif not os.path.exists(cover_image_path):
                logging.error(f"Error: Cover image file not found at: {cover_image_path}")
                print(f"Cover image file not found.")
            elif not os.path.exists(end_image_path):
                logging.error(f"Error: End image file not found at: {end_image_path}")
                print(f"End image file not found.")
            elif not os.path.exists(logo_path):
                logging.error(f"Error: Logo file not found at: {logo_path}")
                print(f"Logo file not found.")
            elif not os.path.exists(music_path):
                logging.error(f"Error: Logo file not found at: {music_path}")
                print(f"Logo file not found.")
            return None
        else:
            logging.info("All required files were found.")

        # Prepare subtitles path for FFmpeg
        formatted_subtitles_path = subtitles_path.replace("\\", "/").replace(":", "\\:")
        logging.info(f"Formatted subtitles path for FFmpeg: {formatted_subtitles_path}")

        # Probe the combined video for duration
        try:
            # Validate resulting clip duration
            fade_out_start = get_video_duration(combined_video_path)
            fade_in_duration = 0.5
            fade_out_duration = 1
            logging.debug(f"Combined video duration: {fade_out_start} seconds.")
        except Exception as e:
            logging.error(f"Error probing combined video: {e}")

        logo_start_x = 50
        logo_start_y = 200

        # Define subtitle styling for FFmpeg
        force_style = (
            "FontName=Candara-Bold,FontSize=12,PrimaryColour=&H00FFFFFF,"
            "BorderStyle=1,Outline=1,Alignment=2,"
            "MarginV=100"
        )
        logging.info(f"FFmpeg subtitle style: {force_style}")

        # The number of the music input will be 5, as we have 5 inputs now (0-4)
        filter_complex = (
            f"[2:v]scale=1080:1920[img_scaled];"  # Scale front image (Input 2)
            f"[3:v]scale=1080:1920[end_image_scaled];"  # Scale end image (Input 3)
            f"[4:v]scale=width=500:height=-1[logo_scaled];"  # Scale logo (Input 4)
            f"[0:v][img_scaled]overlay=0:0:enable='lte(t,{fade_in_duration})'[overlayed];"  # Overlay front image at the start
            f"[overlayed]subtitles='{formatted_subtitles_path}':force_style='{force_style}'[final];"  # Add subtitles
            f"[final]fade=t=out:st={fade_out_start - 0.5}:d={fade_out_duration}[final_out];"  # Apply fade-out at 129.5 seconds
            f"[final_out][end_image_scaled]overlay=0:0:enable='gte(t,{fade_out_start - 0.5})'[temp_output];"  # Add end image after 129.5 seconds
            f"[temp_output][logo_scaled]overlay="
            f"'if(lt(t,{fade_in_duration}),{logo_start_x},if(lt(t,{fade_out_start - 0.5}),(W-w)/2,{logo_start_x})):"
            f"if(lt(t,{fade_in_duration}),{logo_start_y},if(lt(t,{fade_out_start - 0.5}),(H-h)/2,{logo_start_y}))'"
            f"[video_with_logo];"  # Output from all video filters

            # --- Audio Mixing Filters ---
            f"[1:a]volume=1.0[voice_over];"  # Voice-over audio (Input 1), no volume change
            f"[5:a]volume=0.2[background_music];"  # Background music (Input 5), reduced to 20%
            f"[voice_over][background_music]amix=inputs=2:duration=first[audio_mix]"  # Mix the two audio tracks
        )
        logging.info(f"FFmpeg filter_complex style: {filter_complex}")

        # # The number of the music input will be 5, as we have 5 inputs now (0-4)
        # filter_complex = (
        #     f"[2:v]scale=1080:1920[img_scaled];"  # Scale front image (Input 2)
        #     f"[3:v]scale=1080:1920[end_image_scaled];"  # Scale end image (Input 3)
        #     f"[4:v]scale=width=500:height=-1[logo_scaled];"  # Scale logo (Input 4)
        #     f"[0:v][img_scaled]overlay=0:0:enable='lte(t,{fade_in_duration})'[overlayed];"  # Overlay front image at the start
        #     f"[overlayed]fade=t=out:st={fade_out_start - 0.5}:d={fade_out_duration}[final_out];"  # Apply fade-out at 129.5 seconds
        #     f"[final_out][end_image_scaled]overlay=0:0:enable='gte(t,{fade_out_start - 0.5})'[temp_output];"  # Add end image after 129.5 seconds
        #     f"[temp_output][logo_scaled]overlay="
        #     f"'if(lt(t,{fade_in_duration}),{logo_start_x},if(lt(t,{fade_out_start - 0.5}),(W-w)/2,{logo_start_x})):"
        #     f"if(lt(t,{fade_in_duration}),{logo_start_y},if(lt(t,{fade_out_start - 0.5}),(H-h)/2,{logo_start_y}))'"
        #     f"[video_with_logo];"  # Output from all video filters
        #
        #     # --- Audio Mixing Filters ---
        #     f"[1:a]volume=1.0[voice_over];"  # Voice-over audio (Input 1), no volume change
        #     f"[5:a]volume=0.2[background_music];"  # Background music (Input 5), reduced to 20%
        #     f"[voice_over][background_music]amix=inputs=2:duration=first[audio_mix]"  # Mix the two audio tracks
        # )
        # logging.info(f"FFmpeg filter_complex style: {filter_complex}")

        # Build FFmpeg command for final video composition

        command = (
                base_cmd + [
            "-r", "30",  # Set frame rate
            "-i", combined_video_path,  # Input 0: Video
            "-i", tts_path,  # Input 1: TTS (voice-over)
            "-i", cover_image_path,  # Input 2: Cover image
            "-i", end_image_path,  # Input 3: End image
            "-i", logo_path,  # Input 4: Logo
            "-i", music_path,  # Input 5: Background music
            "-filter_complex", filter_complex,  # Filters
            "-map", "[video_with_logo]",  # Use video from filter
            "-map", "[audio_mix]",  # Use mixed audio from filter
        ] + codec_cmd + [
                    "-preset", "fast",  # Encoding speed
                    "-cq", "23",  # Constant quality
                    "-b:v", "5000k",  # Bitrate for video
                    "-c:a", "aac",  # Audio codec
                    "-b:a", "192k",  # Bitrate for audio
                    output_temp_folder  # Output file path
                ]
        )

        logging.info(f"Executing FFmpeg command: {' '.join(command)}")

        # Execute the FFmpeg command
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            logging.info(f"FFmpeg command executed successfully. Return code: {result.returncode}")

            if result.returncode == 0:
                logging.info(f"Final video created successfully and saved to {output_temp_folder}.")

        except subprocess.CalledProcessError as e:
            error_message = f"FFmpeg failed with CalledProcessError: {e}. Command: {e.cmd}.  stderr: {e.stderr if e.stderr else 'No stderr'}. stdout: {e.stdout if e.stdout else 'No stdout'}"
            logging.error(error_message)
            raise ValueError(error_message)
        except FileNotFoundError:
            error_message = "Error: FFmpeg not found. Ensure it is installed and in your system's PATH."
            logging.error(error_message)
            raise FileNotFoundError(error_message)
        except Exception as e:
            error_message = f"An unexpected error occurred during final video processing: {e}"
            logging.error(error_message)
            raise Exception(error_message)

    except ValueError as ve:
        logging.error(f"Value error during video generation: {ve}")
    except FileNotFoundError as fnf_error:
        logging.error(f"File not found error: {fnf_error}")
    except Exception as e:
        logging.error(f"An unexpected error occurred in generate_video: {e}")

    finally:
        # Release resources
        gc.collect()

    return output_temp_folder


def generate_video_custom(combined_video_path: str, tts_path: str, music_path: str, threads: int,
                          subtitles_path: str = None, subtitles_position: str = "center,center",
                          cover_image_path: str = None, end_image_path: str = None,
                          logo_path: str = None, show_progress: bool = False) -> str:
    """
    Generates the final video by optionally adding audio, subtitles, cover image, end image, and logo.
    Includes an option to show a progress bar by streaming FFmpeg's output.

    Args:
        combined_video_path (str): Path to the combined video without audio or overlays.
        tts_path (str): Path to the text-to-speech audio file.
        subtitles_path (str): Path to the SRT subtitle file.
        cover_image_path (str): Path to the cover image file.
        end_image_path (str): Path to the end image file.
        logo_path (str): Path to the logo image file.
        subtitles_position (str): Comma-separated string defining the horizontal and vertical position of subtitles (e.g., "center,bottom").
        threads (int): Number of threads to use for video processing (currently not directly used in the FFmpeg command).
        show_progress (bool): If True, enables real-time progress logging.

    Returns:
        str: The path to the final generated video file.
    """
    output_temp_folder = str(TEMP_DIR / f"output_{uuid.uuid4()}.mp4")

    logging.info(f"Starting final video generation. Output path: {output_temp_folder}")

    # Validate mandatory input file paths
    if not all(os.path.exists(path) for path in [combined_video_path, tts_path, music_path]):
        logging.error("Error: One or more required input files not found.")
        return None

    # Determine video duration for fade-out effect
    try:
        video_duration = get_video_duration(combined_video_path)
        fade_in_duration = 0.5
        fade_out_duration = 1
        logging.debug(f"Combined video duration: {video_duration} seconds.")
    except Exception as e:
        logging.error(f"Error probing combined video: {e}")
        return None

    # Build FFmpeg command inputs and filters dynamically
    inputs = [
        "-r", "30",
        "-i", combined_video_path,  # Input 0: Video
        "-i", tts_path,  # Input 1: TTS (voice-over)
        "-i", music_path,  # Input 2: Background music
    ]
    filter_parts = []
    current_input_index = 3

    # Start the video filter chain. This stream label will be used for all subsequent video filters.
    output_video_stream = "[video_with_filters]"

    # Scale the base video to 1920x1080 for YouTube and set the correct aspect ratio
    filter_parts.append("[0:v]scale=1920:1080,setsar=1:1[base_video]")
    last_video_stream = "[base_video]"

    # Add optional inputs and filters
    if cover_image_path and os.path.exists(cover_image_path):
        inputs.extend(["-i", cover_image_path])
        filter_parts.append(f"[{current_input_index}:v]scale=1920:1080[img_scaled]")
        filter_parts.append(
            f"[{last_video_stream}][img_scaled]overlay=0:0:enable='lte(t,{fade_in_duration})'{output_video_stream}")
        last_video_stream = output_video_stream
        current_input_index += 1

    if end_image_path and os.path.exists(end_image_path):
        inputs.extend(["-i", end_image_path])
        filter_parts.append(f"[{current_input_index}:v]scale=1920:1080[end_image_scaled]")
        filter_parts.append(
            f"[{last_video_stream}]fade=t=out:st={video_duration - 0.5}:d={fade_out_duration}[final_out]")
        filter_parts.append(
            f"[final_out][end_image_scaled]overlay=0:0:enable='gte(t,{video_duration - 0.5})'{output_video_stream}")
        last_video_stream = output_video_stream
        current_input_index += 1

    if logo_path and os.path.exists(logo_path):
        inputs.extend(["-i", logo_path])
        logo_start_x = 50
        logo_start_y = 200
        filter_parts.append(f"[{current_input_index}:v]scale=width=500:height=-1[logo_scaled]")
        # Conditional overlay to move the logo
        logo_overlay_filter = (
            f"[{last_video_stream}][logo_scaled]overlay="
            f"'if(lt(t,{fade_in_duration}),{logo_start_x},if(lt(t,{video_duration - 0.5}),(W-w)/2,{logo_start_x})):"
            f"if(lt(t,{fade_in_duration}),{logo_start_y},if(lt(t,{video_duration - 0.5}),(H-h)/2,{logo_start_y}))'{output_video_stream}"
        )
        filter_parts.append(logo_overlay_filter)
        last_video_stream = output_video_stream
        current_input_index += 1

    if subtitles_path and os.path.exists(subtitles_path):
        formatted_subtitles_path = subtitles_path.replace("\\", "/").replace(":", "\\:")

        # Mapping from human-readable position to FFmpeg alignment value
        alignment_map = {
            "center,center": "5",
            "center,bottom": "2",
            "center,top": "8",
            "left,bottom": "1",
            "right,bottom": "3",
            "left,top": "7",
            "right,top": "9"
        }
        alignment = alignment_map.get(subtitles_position.lower(), "5")

        force_style = (
            "FontName=Candara-Bold,FontSize=12,PrimaryColour=&H00FFFFFF,"
            f"BorderStyle=1,Outline=1,Alignment={alignment},MarginV=100"
        )
        subtitles_filter = f"[{last_video_stream}]subtitles='{formatted_subtitles_path}':force_style='{force_style}'{output_video_stream}"
        filter_parts.append(subtitles_filter)
        last_video_stream = output_video_stream

    # Audio mixing filter
    filter_parts.append("[1:a]volume=1.0[voice_over]")
    filter_parts.append("[2:a]volume=0.2[background_music]")
    filter_parts.append("[voice_over][background_music]amix=inputs=2:duration=first[audio_mix]")

    filter_complex = ";".join(filter_parts)
    logging.info(f"FFmpeg filter_complex: {filter_complex}")

    # Build the final command
    command = (
            base_cmd + inputs +
            ["-threads", str(threads)] +
            ["-filter_complex", filter_complex] +
            ["-map", last_video_stream] +  # Use the last valid video stream
            ["-map", "[audio_mix]"] +
            codec_cmd + [
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "192k",
                output_temp_folder
            ]
    )

    if show_progress:
        command.extend(["-progress", "pipe:1"])

    logging.info(f"Executing FFmpeg command: {' '.join(command)}")

    # Execute the command and track progress if enabled
    try:
        if show_progress:
            bar_length = 30  # Define the length of the progress bar
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  universal_newlines=True,
                                  encoding='utf-8', errors='replace') as process:
                for line in process.stdout:
                    match = re.search(r"out_time_ms=(\d+)", line)
                    if match:
                        elapsed_ms = int(match.group(1))
                        elapsed_seconds = elapsed_ms / 1000000

                        if video_duration > 0:
                            progress_percent = min(100, (elapsed_seconds / video_duration) * 100)

                            # Calculate the number of filled and empty characters
                            filled_chars = int(bar_length * progress_percent // 100)
                            empty_chars = bar_length - filled_chars

                            # Create the progress bar string
                            progress_bar = "Progress [" + "█" * filled_chars + " " * empty_chars + "]"

                            # Print the progress on the same line, overwriting the previous one
                            print(f"\r{progress_bar} {progress_percent:.2f}%", end="")
                            sys.stdout.flush()

                process.wait()

                # Print a final newline and a success message
                print("\rProgress [██████████████████████████████] 100.00% - ¡Process completed succesfully!")
                print()

                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, process.args)

        else:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            if result.returncode == 0:
                logging.info(f"Final video created successfully and saved to {output_temp_folder}.")

    except subprocess.CalledProcessError as e:
        error_message = f"FFmpeg failed with error: {e.stderr if hasattr(e, 'stderr') else e}"
        logging.error(error_message)
        raise ValueError(error_message)
    except FileNotFoundError:
        error_message = "Error: FFmpeg not found."
        logging.error(error_message)
        raise FileNotFoundError(error_message)

    return output_temp_folder


if __name__ == "__main__":

    audio_clip = Path(__file__).resolve().parent.parent / "temp" / "6a4abc75-507d-43cb-910a-b40334faa07a.mp3"
    audio_path = Path(__file__).resolve().parent.parent / "temp" / "b47c88f1-bd2d-444d-a46a-402a44d16282.mp3"

    output_path = str(TEMP_DIR / f"{uuid4()}.mp3")
    combined_video_path = str(
        Path(
            __file__).resolve().parent.parent / "audiobook_content" / "temp" / "ed71fc4b-e5da-4587-b5d8-8b945de39283_1755370194516.mp4")
    image_folder = Path(__file__).resolve().parent.parent / "audiobook_content" / "images"
    tts_path = str(
        Path(__file__).resolve().parent.parent / "audiobook_content" / "audio" / "Capitulo_1_MI_HISTORIA.mp3")
    image_path = str(Path(__file__).resolve().parent.parent / "audiobook_content" / "images" / "atomic_habits.png")
    subtitles_path = str(
        Path(
            __file__).resolve().parent.parent / "audiobook_content" / "temp" / "e5fb9605-cc28-4160-ab77-0b9546ff08c1.srt")
    cover_image_path = str(Path(
        __file__).resolve().parent.parent / "audiobook_content" / "cover" / "image_content_front.png")
    image_back_path = str(
        Path(__file__).resolve().parent.parent / "audiobook_content" / "cover" / "image_content_back_text.png")
    image_logo_path = str(
        Path(__file__).resolve().parent.parent / "audiobook_content" / "cover" / "image_logo_content.png")
    music_path = str(
        Path(
            __file__).resolve().parent.parent / "audiobook_content" / "music" / "432Hz_postive_energy_break_chains_and_emotions.mp3")

    # Verifica si la carpeta existe. Si no, la crea.
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        print(f"Directorio temporal creado: {TEMP_DIR}")

    n_threads = 2
    subtitles_position = "center,center"
    text_color = "# fff"

    # subtitles_path = generate_subtitles(audio_path=audio_path, sentences=script, audio_clips=audio_clips, voice="es")
    #

    # subtitles_path = generate_subtitles(audio_path=tts_path, voice="es")

    #
    # effects = ['zoom_in', 'zoom_out', 'zoom_in', 'zoom_out', 'zoom_in', 'zoom_out', 'zoom_in', 'zoom_out',
    #            'zoom_in', 'zoom_out', 'zoom_in', 'zoom_out', 'zoom_in', 'zoom_out', 'zoom_in', 'zoom_out']
    #
    # image_clips_path = []
    # index = 0
    # threads = 4
    #
    # # image_files = sorted([
    # #     os.path.normpath(os.path.join(image_folder, img))
    # #     for img in os.listdir(image_folder)
    # #     if img.endswith(('png', 'jpg', 'jpeg', 'jfif'))
    # # ])
    # #
    # # for image_path in image_files[:5]:
    # #     logging.info(f"Image:: {image_path}")
    # #     image_path_output = create_video_from_images_ffmpeg(image_path=image_path, effect=effects[index], threads=4)
    # #     index += 1
    # #     image_clips_path.append(image_path_output)
    #
    # temp_audio = AudioFileClip(tts_path)
    #
    # logging.info(f"Starting video creation from image: {image_path}, effect: {effects[index]}, duration: {temp_audio.duration}")
    # image_path_output = create_video_from_images_ffmpeg(image_path=image_path, req_dur=10,
    #                                                     effect=effects[index], threads=4)
    #
    # logging.info(f"Starting single video creation for one element: {image_path}, duration: {temp_audio.duration}")
    # combined_video_path_v2 = process_single_video_one_element(video_path=image_path_output, req_dur=temp_audio.duration)
    #
    # logging.debug(colored(text=f"[+] Video combination path: {combined_video_path_v2}", color="green"))
    #
    # final_video_path = generate_video(combined_video_path=combined_video_path_v2, tts_path=tts_path,
    #                                   subtitles_path=subtitles_path, cover_image_path=cover_image_path,
    #                                   end_image_path=image_back_path, logo_path=image_logo_path,
    #                                   music_path=music_path, threads=n_threads,
    #                                   subtitles_position=subtitles_position, text_color=text_color)
