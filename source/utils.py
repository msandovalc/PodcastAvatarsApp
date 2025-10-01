import os
import sys
import json
import random
from logger_config import setup_logger
import zipfile
import requests
import re
import chardet
from pathlib import Path
from uuid import uuid4
from termcolor import colored
from pydub import AudioSegment
from pydub.silence import detect_silence
from typing import List, Dict, Any
import pytz
import time
import datetime
from typing import Optional


# Configure logging for the module
logging = setup_logger(__name__)

# The maximum byte length for a single text conversion (TTS API limit)
TEXT_BYTE_LIMIT = 200


def clean_dir(path: str) -> None:
    """
    Removes all files from a specified directory.

    This function first checks if the directory exists and creates it if not.
    Then, it iterates through all files in the directory and removes them.

    Args:
        path (str): The path to the directory to be cleaned.

    Returns:
        None
    """
    try:
        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(path):
            os.mkdir(path)
            logging.info(f"Created directory: {path}")

        # Iterate over each file and remove it
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            os.remove(file_path)
            logging.info(f"Removed file: {file_path}")

        logging.info(colored(f"Cleaned {path} directory", "green"))
    except Exception as e:
        logging.error(f"Error occurred while cleaning directory {path}: {str(e)}")


def fetch_songs(zip_url: str) -> None:
    """
    Downloads a zip file of songs from a URL, unzips it, and places the songs
    in a `../Songs` directory.

    This function checks if the songs directory already exists. If it does, it
    skips the download to avoid re-downloading.

    Args:
        zip_url (str): The URL to the zip file containing the songs.

    Returns:
        None
    """
    try:
        logging.info(colored(f" => Fetching songs...", "magenta"))

        files_dir = "../Songs"
        # Check if the songs are already downloaded
        if not os.path.exists(files_dir):
            os.mkdir(files_dir)
            logging.info(colored(f"Created directory: {files_dir}", "green"))
        else:
            logging.info("Songs directory already exists. Skipping download.")
            return

        # Download the zip file from the provided URL
        response = requests.get(zip_url)

        # Save the downloaded zip file
        with open("../Songs/songs.zip", "wb") as file:
            file.write(response.content)

        # Unzip the file to the songs directory
        with zipfile.ZipFile("../Songs/songs.zip", "r") as file:
            file.extractall("../Songs")

        # Remove the temporary zip file
        os.remove("../Songs/songs.zip")

        logging.info(colored(" => Downloaded Songs to ../Songs.", "green"))

    except Exception as e:
        logging.error(colored(f"Error occurred while fetching songs: {str(e)}", "red"))


def choose_random_song() -> str:
    """
    Selects a random song file from the `../Songs` directory.

    Returns:
        str: The full path to the chosen song file. Returns None if an error occurs.
    """
    try:
        # List all files in the songs directory
        songs = os.listdir("../Songs")
        # Choose a random file from the list
        song = random.choice(songs)
        logging.info(colored(f"Chose song: {song}", "green"))
        return f"../Songs/{song}"
    except Exception as e:
        logging.error(colored(f"Error occurred while choosing random song: {str(e)}", "red"))
        return None


def check_env_vars() -> None:
    """
    Verifies that all required environment variables are set.

    This function checks for the presence and non-empty values of specific
    environment variables. If any are missing, it logs an error and exits the program.

    Raises:
        SystemExit: If any of the required environment variables are not set.
    """
    try:
        required_vars = ["PEXELS_API_KEY", "TIKTOK_SESSION_ID", "IMAGEMAGICK_BINARY"]
        # Find all required variables that are either missing or have empty values
        missing_vars = [var for var in required_vars if os.getenv(var) is None or not os.getenv(var)]

        if missing_vars:
            missing_vars_str = ", ".join(missing_vars)
            logging.error(colored(f"The following environment variables are missing: {missing_vars_str}", "red"))
            logging.error(colored("Please consult 'EnvironmentVariables.md' for instructions on how to set them.", "yellow"))
            sys.exit(1)  # Aborts the program
    except Exception as e:
        logging.error(f"Error occurred while checking environment variables: {str(e)}")
        sys.exit(1)  # Aborts the program if an unexpected error occurs


def convert_to_utf8(file_path: str) -> str:
    """
    Detects the encoding of a file and converts it to UTF-8 if it's not already.

    This is particularly useful for handling subtitle files (.srt) that might have
    various encodings. The function reads the file using the detected encoding
    and then writes it back, forcing UTF-8.

    Args:
        file_path (str): Path to the file to be processed.

    Returns:
        str: The path of the processed subtitle file (same as the input path).
    """
    # Detect the file encoding by reading a sample of the raw data
    with open(file_path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"]

    logging.info(f"Detected encoding: {encoding}")

    # If the file is already UTF-8, no conversion is needed
    if encoding and encoding.lower() == "utf-8":
        logging.info("The file is already in UTF-8. No conversion needed.")
        return file_path

    # Read the file using the detected encoding
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        content = f.read()

    # Write the content back in UTF-8 encoding
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    logging.info("File successfully converted to UTF-8.")
    return file_path


def download_image_by_data(image_data: bytes, download_folder: str) -> str:
    """
    Saves image byte data to a file in a specified folder.

    The function generates a unique filename for the image and ensures the
    target directory exists before saving the file.

    Args:
        image_data (bytes): The raw byte data of the image.
        download_folder (str): The path to the folder where the image will be saved.

    Returns:
        str: The full path to the downloaded image file. Returns None if an error occurs.
    """
    try:
        logging.info(f"Starting the download image from Together")

        # Ensure the download folder exists
        Path(download_folder).mkdir(parents=True, exist_ok=True)

        # Create a unique filename for the new image
        image_name = f"generated_image_{uuid4()}.png"

        # Set the full download path
        download_path = os.path.join(download_folder, image_name)

        # Write image to the file in binary mode
        with open(download_path, 'wb') as file:
            file.write(image_data)

        logging.info(f"Image successfully downloaded and saved to: {download_path}")

        return download_path

    except requests.exceptions.RequestException as e:
        logging.error(f"Error while downloading the image: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None


def split_sentences(script: str, max_length: int = TEXT_BYTE_LIMIT) -> List[str]:
    """
    Splits a script into sentences based on punctuation and a maximum length constraint.

    The function uses regular expressions to find sentence endings and then
    further splits sentences that are longer than `max_length` while respecting
    word boundaries.

    Args:
        script (str): The input text to be split into sentences.
        max_length (int): The maximum allowed length for each sentence.

    Returns:
        List[str]: A list of processed sentences. Returns an empty list if an error occurs.
    """
    try:
        # Split the script based on punctuation and spacing
        sentences = re.split(r'(?<=\?)(?=\s)(?=[A-Z¿¡])|(?<=[.?!])\s+(?=[A-Z¿¡])|\s*,\s*,\s*(?=\w)', script)

        processed_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Handle sentences that are too long by splitting them word-by-word
            parts = []
            while len(sentence) > max_length - 1:
                # Find the last space before the max length limit
                split_index = sentence.rfind(' ', 0, max_length)
                if split_index == -1:
                    # If no space is found, split at the max length
                    split_index = max_length

                parts.append(sentence[:split_index].strip())
                sentence = sentence[split_index:].strip()

            if sentence:
                parts.append(sentence)

            processed_sentences.extend(parts)

        return processed_sentences

    except Exception as e:
        logging.error(f"Error splitting sentences: {e}")
        return []


def clean_text_punctuation(text: str) -> str:
    """
    Cleans and normalizes a text string to make it suitable for TTS (Text-to-Speech).

    This function performs several cleaning operations:
    - Replaces newlines with spaces.
    - Normalizes multiple whitespaces to a single space.
    - Removes double punctuation like '?.' or '!.'
    - Removes unwanted characters while keeping Spanish accents and punctuation.

    Args:
        text (str): The raw input text.

    Returns:
        str: The cleaned and normalized text.
    """
    try:
        logging.debug("Starting text cleaning process.")

        # Replace all newline characters with a single space to flatten the text
        text = text.replace('\n', ' ')

        # Replace multiple consecutive whitespace characters with a single space and trim
        text = re.sub(r'\s+', ' ', text).strip()

        # Replace '?.', '!.', '?!.' or similar with just '?' or '!'
        text = re.sub(r'\?\.|!\.', lambda m: m.group(0)[0], text)

        # Remove any characters that are not:
        # - word characters
        # - spaces
        # - Spanish accents and punctuation (¡ ¿ , . : ? !)
        text = re.sub(r"[^\w\sñÑáéíóúüÁÉÍÓÚÜ,.:?!¡¿]", "", text)

        logging.debug("Text cleaning complete.")
        return text

    except Exception as e:
        logging.error(f"Error cleaning text: {e}")
        return text  # Return the original text in case of error


def split_long_sentences(chapter_text: str, sentence_limit_bytes: int = 900) -> List[str]:
    """
    Splits text into sentences based on a byte limit, respecting word boundaries
    and preserving UTF-8 integrity.

    This function intelligently breaks down long sentences and paragraphs into
    smaller chunks suitable for API calls, ensuring no sentence exceeds the
    byte limit while maintaining readability.

    Args:
        chapter_text (str): The input text content of a chapter.
        sentence_limit_bytes (int): The maximum byte size for each sentence.

    Returns:
        List[str]: A list of sentences, each within the byte limit.
    """
    # Helper function to get byte length of a string
    def b_len(s: str) -> int:
        return len(s.encode("utf-8"))

    # Helper function for a hard, byte-safe split (used only for very long tokens)
    def hard_split_preserving_utf8(s: str, limit: int) -> List[str]:
        out = []
        rest = s
        while b_len(rest) > limit:
            chunk_bytes = rest.encode("utf-8")[:limit]
            chunk = chunk_bytes.decode("utf-8", errors="ignore")
            out.append(chunk)
            rest = rest[len(chunk):].lstrip()
        if rest:
            out.append(rest)
        return out

    # Helper function to accumulate tokens into word-aware segments
    def wrap_segment_wordwise(seg: str, limit: int) -> List[str]:
        seg = seg.strip()
        if not seg:
            return []

        if b_len(seg) <= limit:
            return [seg]

        tokens = re.findall(r'\S+\s*', seg)
        out: List[str] = []
        cur = ""

        for tok in tokens:
            tok_b = b_len(tok)
            if tok_b > limit:
                pieces = hard_split_preserving_utf8(tok, limit)
                for i, piece in enumerate(pieces):
                    piece = piece.strip()
                    if not piece:
                        continue
                    if cur.strip():
                        if not cur.rstrip().endswith(('.', '!', '?', ';', '…')):
                            cur = cur.rstrip() + ';'
                        out.append(cur.strip())
                        cur = ""
                    if i < len(pieces) - 1:
                        out.append(piece + ';')
                    else:
                        cur = piece + ' '
                continue

            if b_len(cur + tok) <= limit:
                cur += tok
            else:
                cs = cur.strip()
                if cs:
                    if not cs.endswith(('.', '!', '?', ';', '…')):
                        cs += ';'
                    out.append(cs)
                cur = tok

        cs = cur.strip()
        if cs:
            out.append(cs)
        return out

    # Split text by strong sentence boundaries and double newlines
    parts = re.split(r'(?:[\.!?;…]+[\s\n]+|\n{2,})', chapter_text)
    sentences: List[str] = []
    for seg in parts:
        seg = seg.strip()
        if not seg:
            continue
        sentences.extend(wrap_segment_wordwise(seg, sentence_limit_bytes))

    return sentences


def split_text_into_chunks(text_content: str, max_bytes: int = 4500) -> List[str]:
    """
    Splits a large text block into smaller chunks based on a maximum byte limit,
    while maintaining word boundaries.

    This function iterates through the words of the text, accumulating them into
    a `current_chunk` until adding the next word would exceed the `max_bytes`
    limit. It is a reliable method to handle text for APIs with byte constraints.

    Args:
        text_content (str): The full text to be split.
        max_bytes (int): The maximum byte size for each chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    words = text_content.split()

    for word in words:
        # Check the byte size of the current chunk plus the next word.
        word_to_add = (" " + word).strip()

        # Check if the potential new chunk exceeds the byte limit
        if len((current_chunk + word_to_add).encode('utf-8')) > max_bytes:
            # If it exceeds, save the current chunk and start a new one
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = word
        else:
            # Otherwise, add the word to the current chunk
            current_chunk += (" " + word_to_add)

    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def trim_silence(audio_segment: AudioSegment, silence_thresh: int = -40, min_silence_len: int = 100) -> AudioSegment:
    """
    Trims leading and trailing silence from an AudioSegment object.

    This function uses pydub's `detect_silence` to find quiet parts at the beginning
    and end of an audio segment and removes them.

    Args:
        audio_segment (AudioSegment): The pydub AudioSegment object to be trimmed.
        silence_thresh (int): The silence threshold in dBFS (Decibels relative to Full Scale).
                              Defaults to -40 dBFS.
        min_silence_len (int): The minimum length of silence in milliseconds. Defaults to 100 ms.

    Returns:
        AudioSegment: The trimmed AudioSegment object.
    """
    # Detect silent ranges in the audio
    silent_ranges = detect_silence(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    if not silent_ranges:
        return audio_segment  # Return original if no silence is detected

    start_trim = 0
    end_trim = len(audio_segment)

    # Trim leading silence if it exists
    if silent_ranges[0][0] == 0:
        start_trim = silent_ranges[0][1]

    # Trim trailing silence if it exists
    if silent_ranges[-1][1] == len(audio_segment):
        end_trim = silent_ranges[-1][0]

    # Return the trimmed audio segment
    return audio_segment[start_trim:end_trim]


def publish_at_youtube_format_datetime(days: int = 0, hours: int = 0, minutes: int = 0, delta: bool = True) -> datetime.datetime:
    """
    Calculates a future datetime for YouTube video scheduling, in UTC.

    This function gets the current time in a specified local timezone (`Etc/GMT-6`),
    converts it to UTC, and then adds a time delta specified by the arguments.

    Args:
        days (int): The number of days to add to the current time. Defaults to 0.
        hours (int): The number of hours to add. Defaults to 0.
        minutes (int): The number of minutes to add. Defaults to 0.
        delta (bool): A flag to indicate whether to add the delta to the current time.
                      Defaults to True.

    Returns:
        datetime.datetime: The resulting datetime object in UTC. Returns None on error.
    """
    try:
        # Get local time and convert it to UTC
        local_tz = pytz.timezone('Etc/GMT-6')
        now_local = datetime.datetime.now(local_tz)
        now_utc = now_local.astimezone(pytz.utc)

        # Calculate the time delta
        time_delta = datetime.timedelta(days=days, hours=hours, minutes=minutes)

        # Add the delta if the flag is true
        if delta:
            publish_at_utc = now_utc + time_delta
        else:
            # This case seems to be an error in the original logic, as it would
            # return the timedelta object itself, not a datetime.
            # A more logical behavior would be to return now_utc if delta is false.
            publish_at_utc = now_utc

        # Round seconds and remove microseconds for a clean format
        publish_at_utc = publish_at_utc.replace(microsecond=0)

        logging.info(f"Calculated publish_at (UTC): {publish_at_utc}")
        return publish_at_utc
    except Exception as e:
        logging.error(f"Error calculating publish_at: {e}")
        return None


def calculate_publish_datetime(target_weekday: str, target_time: str, timezone: str = "America/Mexico_City") -> datetime.datetime:
    """
    Calculate the next publish datetime based on the given weekday and time.

    :param target_weekday: Target weekday (e.g., 'Friday', 'Monday').
    :param target_time: Target time in HH:MM format (24-hour format).
    :param timezone: Target timezone (default: America/Mexico_City).
    :return: Datetime object with the calculated publish time.
    """
    try:
        # Get current time in the desired timezone
        tz = pytz.timezone(timezone)
        now = datetime.datetime.now(tz)
        logging.info(f"Current datetime in timezone {timezone}: {now}")

        # Map weekday names to Python's weekday index (Monday=0 ... Sunday=6)
        weekdays = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }

        target_weekday = target_weekday.lower()
        if target_weekday not in weekdays:
            logging.error(f"Invalid weekday provided: {target_weekday}")
            raise ValueError("Invalid weekday name. Please use: Monday ... Sunday")

        target_weekday_index = weekdays[target_weekday]
        logging.debug(f"Target weekday index: {target_weekday_index}")

        # Parse target time (HH:MM)
        try:
            target_hour, target_minute = map(int, target_time.split(":"))
        except ValueError:
            logging.error(f"Invalid time format provided: {target_time}")
            raise ValueError("Invalid time format. Use HH:MM in 24-hour format")

        logging.debug(f"Target time: {target_hour}:{target_minute:02d}")

        # Calculate days until next target weekday
        days_ahead = target_weekday_index - now.weekday()
        if days_ahead < 0 or (days_ahead == 0 and (now.hour, now.minute) >= (target_hour, target_minute)):
            days_ahead += 7

        logging.info(f"Days until next {target_weekday.capitalize()}: {days_ahead}")

        # Construct the target datetime
        publish_datetime = now + datetime.timedelta(days=days_ahead)
        publish_datetime = publish_datetime.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)

        logging.info(f"Calculated publish datetime: {publish_datetime}")

        return publish_datetime

    except Exception as e:
        logging.error(f"Error while calculating publish datetime: {e}")
        raise


# Keep track of the last published datetime per schedule
_last_published_dates = {}


def get_next_publish_datetime(
        schedule_name: str,
        day: str,
        target_time: str,
        start_date: Optional[str] = None,
        timezone: str = "America/Mexico_City",
        target_week: bool = True,
        target_day: bool = False,
        days_ahead_jump: int = 1,
        interval_hours: Optional[int] = None
) -> datetime.datetime:
    """
    Calculates the next publish datetime for a given schedule based on weekly or same-day scheduling.

    This function keeps track of the last published date and time for each schedule.
    It supports either weekly scheduling (target_week=True) on the specified day,
    or same-day scheduling (target_day=True) on the current day with time increments.
    The returned datetime is converted to UTC for YouTube scheduling.

    Args:
        schedule_name (str): Unique identifier for this publishing schedule.
        day (str): Target weekday (e.g., 'Friday', 'Monday') for weekly scheduling.
                   Ignored for date calculation when target_day=True.
        target_time (str): Target start time in HH:MM 24-hour format.
        start_date (Optional[str]): A specific date in 'YYYY-MM-DD' format to start the calculation from.
                                    This is only used if the schedule_name is not already in the cache.
        timezone (str): Local timezone for scheduling (default: America/Mexico_City).
        target_week (bool): If True, schedule weekly on the specified day (default: True).
        target_day (bool): If True, schedule on the current day with time increments.
        interval_hours (Optional[int]): Interval in hours for multiple same-day publications (e.g., 2 for every 2 hours).
        days_ahead_jump (int): Number of days to jump when no slots are available (default=1).

    Returns:
        datetime.datetime: Next publish datetime in UTC.

    Raises:
        ValueError: If target_week and target_day are both True or both False, or if inputs are invalid.
    """
    try:
        tz = pytz.timezone(timezone)

        weekdays = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
        }

        # Validate exclusivity of target_week and target_day
        if target_week == target_day:
            logging.error("Exactly one of target_week or target_day must be True")
            raise ValueError("Exactly one of target_week or target_day must be True.")

        # Validate day (only for target_week)
        day_lower = day.lower()
        if day_lower not in weekdays:
            logging.error(f"Invalid day provided: {day}")
            raise ValueError("Invalid day name. Use Monday, Tuesday, ..., Sunday.")

        target_day_index = weekdays[day_lower]

        # Validate target_time
        try:
            target_hour, target_minute = map(int, target_time.split(":"))
            if not (0 <= target_hour <= 23 and 0 <= target_minute <= 59):
                raise ValueError
        except ValueError:
            logging.error(f"Invalid time format provided: {target_time}")
            raise ValueError("Invalid time format. Use HH:MM in 24-hour format.")

        # Validate interval_hours for target_day
        if target_day and interval_hours is not None:
            if not isinstance(interval_hours, int) or interval_hours < 1:
                logging.error(f"Invalid interval_hours: {interval_hours}")
                raise ValueError("interval_hours must be a positive integer.")

        # Determine the starting point for the calculation
        current_time = datetime.datetime.now(tz)
        if schedule_name in _last_published_dates:
            last_published = _last_published_dates[schedule_name]
        # Otherwise, if a start_date is provided, use it for the initial calculation.
        elif start_date:
            start_dt_naive = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            # Localize the start date to the correct timezone
            last_published = tz.localize(start_dt_naive.replace(hour=0, minute=0, second=0, microsecond=0))
        # If neither of the above, fall back to the current time.
        else:
            last_published = current_time

        logging.info(f"Using start datetime for '{schedule_name}': {last_published}")

        # Initialize next_hour and next_minute to avoid UnboundLocalError
        next_hour, next_minute = target_hour, target_minute

        # Same-day scheduling (on current day)
        if target_day:
            # Use the date from last_published, or current date if it's the first call
            target_date = last_published.date()
            if last_published.date() < current_time.date():
                target_date = current_time.date()
                days_ahead = 7  # Move to next week if last_published is in the past
            else:
                days_ahead = 0

            # Check if we need to increment time slots
            if interval_hours:
                current_minutes = last_published.hour * 60 + last_published.minute
                start_minutes = target_hour * 60 + target_minute
                # Generate possible times for the day
                possible_times = []
                current_hour, current_minute = target_hour, target_minute
                while current_hour < 24:
                    possible_times.append((current_hour, current_minute))
                    current_hour += interval_hours
                # Find the next available time
                future_times = [
                    (h, m) for h, m in possible_times
                    if (h * 60 + m) > current_minutes
                ]
                if not future_times:
                    # No future times today
                    if interval_hours == 24:
                        # Special case: publish every 24 hours → move just 1 day ahead
                        days_ahead = 1
                    else:
                        days_ahead = days_ahead_jump
                    target_date = last_published.date() + datetime.timedelta(days=days_ahead)
                    next_hour, next_minute = target_hour, target_minute
                else:
                    # Use the next available time today
                    next_hour, next_minute = min(future_times, key=lambda t: t[0] * 60 + t[1])
            else:
                # Single time slot, check if it's in the future
                current_minutes = last_published.hour * 60 + last_published.minute
                target_minutes = target_hour * 60 + target_minute
                if target_minutes <= current_minutes:
                    days_ahead = 7  # Move to next week if time has passed
                    target_date = last_published.date() + datetime.timedelta(days=days_ahead)
                next_hour, next_minute = target_hour, target_minute

            # Set next_publish_local with the correct date and time
            next_publish_local = tz.localize(
                datetime.datetime.combine(target_date, datetime.time(next_hour, next_minute))
            )
        else:
            # Weekly scheduling (on specified day)
            days_ahead = target_day_index - last_published.weekday()
            if days_ahead <= 0:
                days_ahead += 7

            next_publish_local = last_published + datetime.timedelta(days=days_ahead)
            next_publish_local = next_publish_local.replace(
                hour=target_hour, minute=target_minute, second=0, microsecond=0
            )

            if next_publish_local <= last_published:
                next_publish_local += datetime.timedelta(weeks=1)

        # Convert to UTC
        next_publish_utc = next_publish_local.astimezone(pytz.utc)

        # Update last published date for the next call
        _last_published_dates[schedule_name] = next_publish_local

        logging.info(f"Next publish local datetime for '{schedule_name}' ({timezone}): {next_publish_local}")
        logging.info(f"Next publish utc datetime for '{schedule_name}' (UTC): {next_publish_utc}")
        return next_publish_utc

    except Exception as e:
        logging.error(f"Error calculating next publish datetime: {e}")
        raise


def convert_datetime(utc_datetime: datetime.datetime) -> dict:
    """
    Converts a UTC datetime to Mexico City time and returns a dictionary of parameters
    for the get_next_publish_datetime function, dynamically derived from the converted datetime.

    Args:
        utc_datetime (datetime.datetime): The UTC datetime to convert (e.g., 2025-09-01 01:00:00+00:00).

    Returns:
        dict: Dictionary containing parameters for get_next_publish_datetime, including
              schedule_name, day, target_time, start_date, target_week, target_day,
              interval_hours, timezone, and converted_datetime for reference.

    Raises:
        ValueError: If the input datetime is not timezone-aware.
    """
    try:
        # Validate input datetime
        if utc_datetime.tzinfo is None:
            logging.error("Input datetime must be timezone-aware")
            raise ValueError("Input datetime must be timezone-aware")

        # Convert UTC datetime to Mexico City time
        target_tz = pytz.timezone("America/Mexico_City")
        local_dt = utc_datetime.astimezone(target_tz)
        logging.info(f"Converted UTC datetime {utc_datetime} to America/Mexico_City: {local_dt}")

        # Dynamically derive parameters
        day = local_dt.strftime("%A").lower()  # e.g., "sunday"
        target_time = local_dt.strftime("%H:%M")  # e.g., "19:00"
        start_date = local_dt.strftime("%Y-%m-%d")  # e.g., "2025-08-31"

        # Prepare dictionary with parameters
        params = {
            "day": day,
            "target_time": target_time,
            "start_date": start_date,
            "timezone": "America/Mexico_City",
            "converted_datetime": local_dt
        }

        return params

    except Exception as e:
        logging.error(f"Error in convert_datetime: {e}")
        raise


def clean_video_title(local_title: str) -> str:
    """
    Cleans a video title by performing three operations:
    1. Replaces '- <number>' (e.g., '- 1') with ': '.
    2. Removes the fixed string ' (Spanish Edition)'.
    3. Truncates the title to a maximum of 100 characters.

    Args:
        local_title (str): The original title string.

    Returns:
        str: The cleaned and truncated title string.
    """
    # 1. Use regex to find and replace the pattern "- <number>"
    # with a colon and a single space.
    # The pattern r' - \d+' looks for:
    #   - ' ' (a space)
    #   - '-' (a hyphen)
    #   - ' ' (a space)
    #   - '\d+' (one or more digits)
    cleaned_title = re.sub(r' - \d+', ':', local_title)

    # 2. Use a simple string replace to remove the fixed text
    cleaned_title = cleaned_title.replace(" (Spanish Edition)", "")

    # Remove any leading or trailing whitespace that might have been created
    cleaned_title = cleaned_title.strip()

    # 3. Check if the title exceeds 100 characters. If so, truncate it.
    if len(cleaned_title) > 100:
        cleaned_title = cleaned_title[:100]

    return cleaned_title


def sanitize_filename(filename):
    # Elimina caracteres no válidos para nombres de archivo en Windows
    # Se reemplazan por un guion bajo para mantener la legibilidad
    return re.sub(r'[\\/:*?"<>|¿]', '_', filename)


def is_kaggle():
    """
    Detects if the current environment is Kaggle.
    Checks for the existence of the /kaggle/working directory.
    """
    return os.path.exists("/kaggle/working")


if __name__ == "__main__":

    # # Split script into sentences
    # script = """
    #     El Enemigo Silencioso Desenmascarado: La Verdad Profunda Detrás del Miedo y Cómo Reclamar tu Poder Innato:
    #
    #     Detente por un momento y observa con atención las sutiles maneras en que el miedo se infiltra en tu día a
    #     día. A menudo lo disfrazamos de precaución, de instinto de supervivencia, pero ¿y si te dijera que esta
    #     emoción aparentemente protectora es, en realidad, una de las ilusiones más astutas que oscurecen la radiante
    #     verdad de tu ser?. El temor, esa sensación opresiva que te paraliza ante lo desconocido, se presenta como un
    #     escudo, pero en su núcleo, actúa como una barrera invisible, una prisión mental que te aísla de la abundancia
    #     ilimitada, la salud vibrante y la alegría profunda que son tu derecho de nacimiento inherente. Desvelemos
    #     juntos las capas de esta emoción limitante, explorando sus raíces y comprendiendo su verdadero impacto en tu
    #     potencial. ¿Cuáles son esos miedos específicos que sientes que te atan, que te impiden dar el siguiente paso
    #     hacia tus sueños? Reflexiona profundamente sobre esas áreas donde la ansiedad te detiene y comparte tu
    #     introspección honesta en los comentarios. Juntos, podemos iluminar estas sombras y comenzar el viaje
    #     transformador hacia una vida basada en la confianza inquebrantable y el amor incondicional. ¡Dale like a este
    #     video si estás absolutamente listo para liberarte de las garras del miedo y sígueme para descubrir las
    #     herramientas y la sabiduría para reclamar tu poder innato y vivir una vida sin limitaciones!
    #
    #     """
    #
    # logging.info(f"Script Input: {script}")
    #
    # script_threated = clean_text_punctuation(script)
    #
    # logging.info(f"Script cleaned: {script_threated}")

    # schedule = "weekly_friday_10am"
    #
    # for i in range(10):
    #     publish_at = get_next_publish_datetime(
    #         schedule_name=schedule,
    #         day="Friday",
    #         target_time="10:00",
    #         start_date="2025-08-31",
    #         target_week=True,
    #         target_day=False
    #     )
    #
    #     print(f"Video {i + 1} scheduled: {schedule} for: {publish_at}")
    #
    #
    # schedule = "friday_multi_10am"
    #
    # for i in range(10):
    #     publish_at = get_next_publish_datetime(
    #         schedule_name=schedule,
    #         day="Friday",
    #         target_time="10:00",
    #         start_date="2025-08-31",
    #         target_week=False,
    #         target_day=True,
    #         interval_hours=2
    #     )
    #
    #     print(f"Video {i + 1} scheduled: {schedule} for: {publish_at}")
    #

    # Example usage
    utc_dt = datetime.datetime(2025, 9, 15, 10, 0, tzinfo=pytz.UTC)
    params = convert_datetime(utc_dt)
    print(f"Parameters for scheduling: {params}")

    logging.info(f"Generated params[day]: {params['day']} - "
                 f"params[target_time]: {params['target_time']} - "
                 f"params[start_date]: {params['start_date']}")

    schedule = "Test"
    publish_at = get_next_publish_datetime(
        schedule_name=schedule,
        day=params["day"],
        target_time=params["target_time"],
        start_date=params["start_date"],
        timezone=params["timezone"],
        target_week=False,
        target_day=True,
        interval_hours=1
    )

    print(f"Video scheduled: {schedule} for: {publish_at}")


