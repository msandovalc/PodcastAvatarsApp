import os
import uuid
import psutil
import shutil
import pandas as pd
# from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any
from google_tts import GCPTextToSpeechManager
from logger_config import setup_logger
from gpt_langchain import ContentGenerator
from excelhandler import ExcelHandler
from utils import *
from youtube import *
from OpusClipApp import *
from instagram_api import InstagramAPI
from cloudinary_handler import *
from video import *
from subtitles import *
from image_handler import *
from muse_talk_wrapper import *
from MuseTalk.scripts.realtime_inference import run_musetalk_inference, musetalk_inference_reloaded

# --- Configuration ---
logger = setup_logger(__name__)

dotenv_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# --- Constants for Directory Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
PODCAST_DIR = BASE_DIR / "podcast_content"
IMG_DIR = PODCAST_DIR / "images"
OUTPUT_DIR = PODCAST_DIR / "output"
OUTPUT_SHORTS_DIR = OUTPUT_DIR / "shorts"
TEMP_DIR = PODCAST_DIR / "temp"
AUDIO_DIR = PODCAST_DIR / "audio"
SUBTITLES_DIR = PODCAST_DIR / "subtitles"
TRANSCRIPT_DIR = PODCAST_DIR / "transcriptions"
ANALYSES_DIR = PODCAST_DIR / "analyses"
AVATARS_DIR = PODCAST_DIR / "avatars"
EXCEL_FILE_DIR = PODCAST_DIR / "docs" / "video_podcast_content.xlsx"
CLIENT_SECRETS_FILE = BASE_DIR / "credentials" / "client_secret_471046779868-8ocft10nuip214911m6cpul8no9urcrg.apps.googleusercontent.com.json"
MUSETALK_CONFIG_PATH = BASE_DIR / "MuseTalk" / "configs" / "inference" / "realtime.yaml"

# --- Constants for Environment Variable Keys ---
REQUIRED_KEYS = [
    "ASSEMBLY_AI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GCS_BUCKET_NAME",
]

# --- Initialize Global Objects (Consider managing these within a class) ---
tts_generator = None
excel_handler = None
content_generator = None
gcp_manager = None
instagram_api = None
cloudinary_uploader = None
youtube_uploader = None
enable_youtube = False
enable_instagram = False


def initialize_app():
    """Initializes global objects and configurations, checking for required API keys."""
    global tts_generator, excel_handler, content_generator, gcp_manager, instagram_api, cloudinary_uploader, youtube_uploader

    app_initialized = True

    logging.info("Initializing application...")

    # Check for required API keys and configuration paths
    for key in REQUIRED_KEYS:
        value = os.getenv(key)
        if not value:
            logging.error(f"Required environment variable '{key}' is not set or empty. Application cannot start.")
            app_initialized = False

    excel_file_path_str = str(EXCEL_FILE_DIR) if EXCEL_FILE_DIR else os.getenv("EXCEL_FILE_DIR")
    if excel_file_path_str:
        try:
            excel_handler = ExcelHandler(excel_file_path_str)
            logging.info(f"Excel handler initialized with path: {excel_file_path_str}")
        except Exception as e:
            logging.error(f"Error initializing Excel handler with path '{excel_file_path_str}': {e}")
            excel_handler = None
            app_initialized = False
    else:
        logging.error("EXCEL_FILE_PATH is not defined. Excel functionality will be unavailable.")
        app_initialized = False

    # Initialize ContentGenerator
    content_generator = ContentGenerator()

    # 2. Initialize the GCP Text-to-Speech manager (no bucket used for text)
    gcp_manager = GCPTextToSpeechManager(bucket_name=os.getenv("GCS_BUCKET_NAME"))

    # Initialize CloudinaryUploader
    cloudinary_uploader = CloudinaryUploader()

    instagram_api = InstagramAPI()

    youtube_uploader = YouTubeUploader(client_secret_file=CLIENT_SECRET_FILE, channels=CHANNELS)

    # Example: authenticate one channel
    youtube, channel_id = youtube_uploader.get_authenticated_service_by_channel(
        "Mente de Campeón: El Poder de Tu Pensamiento")
    if youtube:
        print(f"✅ Channel ready: {channel_id}")

    # Define the local output directory for audio files
    PODCAST_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SHORTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if app_initialized:
        logging.info("Application initialization completed successfully.")
        return True
    else:
        logging.error("Application initialization failed due to missing or invalid configurations.")
        return False


def initialize_directories():
    """Initializes and cleans temporary and subtitle directories."""
    logging.info("Initializing directories.")
    # clean_dir(str(TEMP_DIR))
    # clean_dir(str(SUBTITLES_DIR))
    # clean_dir(str(AUDIO_DIR))
    # clean_dir(str(OUTPUT_DIR))

    # logging.info(f"Cleaned temporary directory: {TEMP_DIR}")
    # logging.info(f"Cleaned subtitles directory: {SUBTITLES_DIR}")
    # logging.info(f"Cleaned subtitles directory: {AUDIO_DIR}")
    # logging.info(f"Cleaned subtitles directory: {OUTPUT_DIR}")


def debug():
    # audio_paths = [
    #     {'speaker': 'Speaker 1',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_1.wav'},
    #     {'speaker': 'Speaker 2',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_2.wav'},
    #     {'speaker': 'Speaker 1',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_3.wav'},
    #     {'speaker': 'Speaker 2',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_4.wav'},
    #     {'speaker': 'Speaker 1',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_5.wav'},
    #     {'speaker': 'Speaker 2',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_6.wav'},
    #     {'speaker': 'Speaker 1',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_7.wav'},
    #     {'speaker': 'Speaker 2',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_8.wav'},
    #     {'speaker': 'Speaker 1',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_9.wav'},
    #     {'speaker': 'Speaker 2',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_10.wav'},
    #     {'speaker': 'Speaker 1',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_11.wav'},
    #     {'speaker': 'Speaker 1',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_13.wav'},
    #     {'speaker': 'Speaker 2',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_14.wav'},
    #     {'speaker': 'Speaker 1',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_15.wav'},
    #     {'speaker': 'Speaker 2',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_16.wav'},
    #     {'speaker': 'Speaker 1',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_17.wav'},
    #     {'speaker': 'Speaker 2',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_18.wav'},
    #     {'speaker': 'Speaker 1',
    #      'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_19.wav'}
    # ]

    # combination_paths = [
    #     {'avatar_id': 'avator_1',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_1/vid_output/audio_0.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_1.wav'},
    #     {'avatar_id': 'avator_2',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_2/vid_output/audio_0.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_2.wav'},
    #     {'avatar_id': 'avator_1',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_1/vid_output/audio_1.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_3.wav'},
    #     {'avatar_id': 'avator_2',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_2/vid_output/audio_1.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_4.wav'},
    #     {'avatar_id': 'avator_1',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_1/vid_output/audio_2.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_5.wav'},
    #     {'avatar_id': 'avator_2',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_2/vid_output/audio_2.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_6.wav'},
    #     {'avatar_id': 'avator_1',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_1/vid_output/audio_3.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_7.wav'},
    #     {'avatar_id': 'avator_2',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_2/vid_output/audio_3.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_8.wav'},
    #     {'avatar_id': 'avator_1',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_1/vid_output/audio_4.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_9.wav'},
    #     {'avatar_id': 'avator_2',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_2/vid_output/audio_4.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_10.wav'},
    #     {'avatar_id': 'avator_1',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_1/vid_output/audio_5.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_11.wav'},
    #     {'avatar_id': 'avator_2',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_2/vid_output/audio_5.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_12.wav'},
    #     {'avatar_id': 'avator_1',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_1/vid_output/audio_6.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_13.wav'},
    #     {'avatar_id': 'avator_2',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_2/vid_output/audio_6.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_14.wav'},
    #     {'avatar_id': 'avator_1',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_1/vid_output/audio_7.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_15.wav'},
    #     {'avatar_id': 'avator_2',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_2/vid_output/audio_7.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_16.wav'},
    #     {'avatar_id': 'avator_1',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_1/vid_output/audio_8.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_17.wav'},
    #     {'avatar_id': 'avator_1',
    #      'output_path': './MuseTalk/results/v15/avatars/avator_1/vid_output/audio_9.mp4',
    #      'original_audio': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_18.wav'}
    # ]

    # processed_clips = [
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\5d8a919f-7442-49be-9f37-aa2212fdc221_1759807443487.mp4',
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\12097b19-4e89-4f53-af3d-8b1efd6c651c_1759807446477.mp4',
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\6b9e3dcf-117c-4eb4-bd2b-058d5dfc7878_1759807449851.mp4',
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\5d339d36-10b8-4b04-832f-d1a728614b5d_1759807454088.mp4',
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\8be0b2c5-7932-4491-ad50-75e8a616cef4_1759807457109.mp4',
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\0e615ff0-5381-4d7b-9b25-41fe31d50066_1759807459649.mp4',
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\02ea4db7-209d-4be3-99c9-572df270aba2_1759807462644.mp4',
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\37eb35d6-f091-4b8f-ba20-c8447a2b9306_1759807465460.mp4',
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\55a729a7-f88e-4604-9e49-1f5bbabd8fff_1759807468500.mp4',
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\f1923307-6c04-4b77-a330-3b32c5f71a68_1759807470906.mp4',
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\e3ff1264-f984-4824-8a29-44ec7c63e5b8_1759807473186.mp4',
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\a8171f31-7777-4b2d-9c88-cebe84272bba_1759807476293.mp4',
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\4f946c5b-745b-4350-b0bb-96e2e31e8df3_1759807478553.mp4',
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\415de267-91f5-4bf3-ac77-3f8b177fb937_1759807482201.mp4',
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\240bcee0-67e6-463d-8f3c-547802124a87_1759807485156.mp4',
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\82f38f84-ceb6-4fcc-9f03-334cefb31699_1759807487659.mp4',
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\29f34f46-e1e0-472a-88e9-29958c882d7d_1759807490061.mp4',
    #     'F:\\Development\\PyCharm\\Projects\\PodcastAvatarsApp\\podcast_content\\temp\\0e48beb1-40c2-4aa6-8166-fd2ab4df677f_1759807492527.mp4']

    # chapter_path = chapter["path"]
    # chapter_index = chapter["index"]
    # chapter_text = chapter["text"]
    # chapter_title = f"Capitulo {chapter_index} - {chapter['title']}"
    # chapter_name = chapter["filename"].replace(".txt", "")
    # book_title = chapter["book_title"]
    # local_title = clean_video_title(f"Capitulo {chapter_index} - {chapter['title']} | {book_title}")

    # Debug options
    # final_video_path = str(OUTPUT_DIR / "output_72dd96ae-a6b7-4c71-aa72-921340fcd81f.mp4")
    # audio_path = str(AUDIO_DIR / "Chapter_9_SIGNIFICADO METAFÍSICO DE LOS DIEZ MANDAMIENTOS DEMOISÉS.wav")

    # logging.info("Generating cover image.")
    # image_output_path = str(TEMP_DIR / f"image_thumbail_{uuid.uuid4()}.png")
    # creates_images(hook_text=local_title, image_input_path=image_path,
    #                image_logo_path=image_logo_path, image_output_path=image_output_path, font_size=150)
    # logging.info(f"Cover image generated at: {image_output_path}")

    # subtitles_path = generate_subtitles(audio_path=audio_path, voice=lang_code)
    # convert_to_utf8(subtitles_path)
    # logging.info(f"Subtitles generated at: {subtitles_path}")

    # logger.info(f"Starting video creation from image: {image_path}, effect: {effects[0]}, duration: {temp_audio.duration}")
    # image_path_output = create_video_from_images_ffmpeg(image_path=image_path, req_dur=10, effect=effects[0],
    #                                                     threads=n_threads, show_progress=True)

    # # Generate a random integer between 120 and 300 (2 to 5 minutes)
    # sleep_duration_seconds = random.randint(120, 300)
    #
    # logging.info(f"Sleep time for : {sleep_duration_seconds} secs")
    #
    # # Pause the program for the random duration
    # time.sleep(sleep_duration_seconds)

    pass


def process_and_generate_audio(chapter_data: dict[str, Any], chapter_text: str, audio_output_path: str) -> list[dict]:
    """
    Processes a list of chapter dictionaries and generates audio for each chapter
    directly from local text files, saving the audio in a local folder.
    """

    # 3. Iterate through the ordered list to process each chapter
    try:
        chapter_name = chapter_data.get('Title')

        # # Define the local path for the output audio
        # # output_gcs_uri = f"gs://{gcp_manager.bucket_name}/audio/{chapter_name}.mp3"

        logger.info(f"Processing chapter: {chapter_name}")

        audio_output_paths = []

        logger.info(f"audio_output_path path: {audio_output_path}")

        # Synthesize audio directly from the local text
        audio_output_paths = gcp_manager.synthesize_audio_from_text(
            text_content=chapter_text,
            local_audio_path=audio_output_path,
            individual_audios=True
        )

        logger.info("Audiobook generation process completed successfully!")

        return audio_output_paths

    except Exception as e:
        logger.critical(f"An error occurred during audiobook generation: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")
        return ""


def process_video(video_path, output_dir=OUTPUT_SHORTS_DIR, num_clips=None, min_duration=60, max_duration=90,
                  transcription_dir=TRANSCRIPT_DIR, analysis_dir=ANALYSES_DIR,
                  subtitle_style=None, book_title=None):
    """
    Process a video to generate short clips with subtitles, with variable duration between min_duration and max_duration.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory to save generated clips.
        num_clips (int, optional): Number of clips to generate. If None, generate clips for all valid segments.
        min_duration (int): Minimum duration of each clip in seconds.
        max_duration (int): Maximum duration of each clip in seconds.
        transcription_dir (str): Directory to save/load transcription.
        analysis_dir (str): Directory to save/load analysis.
        subtitle_style (dict): Configuration for subtitle styling.
        book_title (str, optional): Title of the book to include in clip metadata.

    Returns:
        list: List of dicts with 'path', 'text', 'srt_content', and 'book_title' for each generated clip.
    """
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
    Path(log_dir).mkdir(exist_ok=True)
    log_file = os.path.join(log_dir, "opus_clip_clone.log")

    # setup_logging(log_file)

    try:
        logger.info("Starting processing for video: %s", video_path)
        use_cuda = torch.cuda.is_available()
        logger.info("Using %s CPU cores, CUDA available: %s", os.cpu_count(), use_cuda)
        if use_cuda:
            logger.debug("CUDA device: %s, memory allocated: %.2f MB",
                         torch.cuda.get_device_name(0), torch.cuda.memory_allocated(0) / 1024 ** 2)
        # Derive paths based on video filename
        video_basename = os.path.basename(video_path)
        audio_basename = f"temp_audio_{video_basename}.wav"
        transcription_path = os.path.join(transcription_dir, f"transcription_{video_basename}.json")
        analysis_path = os.path.join(analysis_dir, f"analysis_{video_basename}.json")
        Path(transcription_dir).mkdir(exist_ok=True)
        Path(analysis_dir).mkdir(exist_ok=True)

        # Initialize components
        logger.info("Initializing VideoProcessor")
        video_processor = VideoProcessor(video_path, use_cuda=True)
        logger.info("Initializing AudioTranscriber")
        audio_transcriber = AudioTranscriber(use_cuda=True)
        logger.info("Initializing ContentAnalyzer")
        content_analyzer = ContentAnalyzer(use_cuda=True, max_workers=1)
        logger.info("Initializing ClipGenerator")
        clip_generator = ClipGenerator(video_path, output_dir=output_dir, subtitle_style=subtitle_style,
                                       use_background=True, background_color="white")

        # Extract audio
        logger.info("Extracting audio")
        audio_path = video_processor.extract_audio()

        # Check for existing transcription
        logger.info("Checking for transcription at: %s", transcription_path)
        segments = None
        if os.path.exists(transcription_path):
            try:
                with open(transcription_path, "r", encoding="utf-8") as f:
                    transcription_data = json.load(f)
                if isinstance(transcription_data, list):
                    segments = transcription_data
                    logger.warning("Legacy transcription format detected at %s, converting to new format",
                                   transcription_path)
                    transcription_data = {"audio_path": audio_path, "segments": segments}
                    with open(transcription_path, "w", encoding="utf-8") as f:
                        json.dump(transcription_data, f, indent=4)
                    logger.info("Converted transcription to new format at %s", transcription_path)
                elif not isinstance(transcription_data, dict):
                    logger.warning("Invalid transcription format at %s: not a dict or list", transcription_path)
                elif (os.path.normpath(transcription_data.get("audio_path", "")) == audio_path and
                      isinstance(transcription_data.get("segments"), list) and
                      len(transcription_data["segments"]) > 0):
                    segments = transcription_data["segments"]
                    logger.info("Reusing existing transcription with %s segments from %s",
                                len(segments), transcription_path)
                else:
                    logger.warning("Invalid or mismatched transcription file at %s: missing audio_path or segments",
                                   transcription_path)
            except Exception as e:
                logger.error("Error loading transcription file %s: %s", transcription_path, e, exc_info=True)

        # Transcribe if no valid segments
        if segments is None:
            logger.info("Starting transcription")
            segments = audio_transcriber.transcribe(audio_path, save_path=transcription_path)
            logger.info("Transcription completed, %s segments found", len(segments))

        # Check for existing analysis
        logger.info("Checking for analysis at: %s", analysis_path)
        scores = None
        if os.path.exists(analysis_path):
            try:
                with open(analysis_path, "r", encoding="utf-8") as f:
                    analysis_data = json.load(f)
                if (isinstance(analysis_data, dict) and
                        os.path.normpath(analysis_data.get("video_path", "")) == video_path and
                        isinstance(analysis_data.get("segments"), list) and
                        len(analysis_data["segments"]) == len(segments) and
                        all(a.get("start") == b["start"] and a.get("text") == b["text"] and
                            isinstance(a.get("score"), (int, float))
                            for a, b in zip(analysis_data["segments"], segments))):
                    scores = [s["score"] for s in analysis_data["segments"]]
                    logger.info("Reusing existing analysis with %s scores from %s", len(scores), analysis_path)
                    logger.info("Scores: %s", scores)
                else:
                    logger.info("Analysis file mismatch or invalid scores, re-running analysis")
            except Exception as e:
                logger.error("Error loading analysis file %s: %s", analysis_path, e, exc_info=True)

        # Analyze content if no valid scores
        if scores is None:
            logger.info("Analyzing video content")
            scores = content_analyzer.analyze_segments_batch(video_processor, segments, use_ffmpeg_frames=True,
                                                             save_path=analysis_path)
            logger.info("Analysis completed, %s scores generated", len(scores))
            logger.info("Scores: %s", scores)

        # Select top segments
        logger.info("Selecting top segments")
        valid_scores = [(i, score) for i, score in enumerate(scores) if score is not None]
        if not valid_scores:
            logger.error("No valid scores available for segment selection")
            raise ValueError("No valid scores available for segment selection")
        num_clips_to_generate = len(valid_scores) if num_clips is None else min(num_clips, len(valid_scores))
        top_indices = sorted(valid_scores, key=lambda x: x[1], reverse=True)[:num_clips_to_generate]
        selected_segments = [segments[i] for i, _ in top_indices]
        logger.info("Selected %s clips for generation: %s", len(selected_segments),
                    [s['start'] for s in selected_segments])
        for i, seg in enumerate(selected_segments, 1):
            logger.info("Selected segment %s: start=%s, text=%s", i, seg['start'], seg['text'])

        # Generate clips
        logger.info("Generating clips with variable duration between %ss and %ss", min_duration, max_duration)
        clip_data = clip_generator.generate_clips(selected_segments, min_duration=min_duration,
                                                  max_duration=max_duration,
                                                  all_segments=segments)
        # Add book title to clip data
        for clip in clip_data:
            clip["metadata"]["book_title"] = book_title if book_title else "Untitled Book"
        logger.info("Generated %s clips with metadata", len(clip_data))

        # Cleanup
        logger.info("Closing video processor")
        video_processor.close()

        logger.info("Processing completed successfully")
        return clip_data

    except Exception as e:
        logger.error("Error processing video: %s", e, exc_info=True)
        raise
    finally:
        if 'video_processor' in locals():
            video_processor.close()


def create_podcast():
    logger.info("Starting the main application for long audio generation.")

    # subtitles_path = str(SUBTITLES_DIR / "f1ed75be-c11b-4d73-a76c-2722d3560086_typewriter.ass")
    cover_image_path = str(PODCAST_DIR / "cover" / "image_content_front.png")
    image_back_path = str(PODCAST_DIR / "cover" / "image_content_back_text.png")
    image_logo_path = str(PODCAST_DIR / "cover" / "image_logo_content.png")
    image_output_path = str(TEMP_DIR / f"image_thumbail_{uuid.uuid4()}.png")
    music_path = str(PODCAST_DIR / "music" / "432Hz_postive_energy_break_chains_and_emotions.mp3")

    n_threads = 2
    subtitles_position = "center,bottom"
    language = "Spanish"

    # video_url = "https://youtu.be/ba6I7IFAMaI?si=ihTAlY9zaAD7_b_Y"
    # output_video_folder = str(TEMP_DIR / "output_6c4ee41e-db26-4b7e-906b-e427c486f44d.mp4")

    try:

        # Store the start time
        start_time = time.time()

        logger.info("Starting the video generation process.")
        initialize_directories()

        excel_data = excel_handler.read_data(sheet_name='Content')

        if not excel_data:
            logger.error("No data found in the Excel file.")
            return False

        subtitle_style = {
            "FontName": "Candara-Bold",
            "FontSize": 10,
            "PrimaryColour": "&H00FFFFFF",
            "BorderStyle": 1,
            "Outline": 2,
            "Alignment": 2,
            "MarginV": 60
        }

        for chapter in excel_data:

            if chapter.get('Enabled') in (0, False):
                logger.info(f"Content Enabled: {chapter.get('Enabled')}")
                continue

            chapter_name = chapter.get('Title')
            chapter_text = chapter.get('Chapter')
            book_title = chapter.get("Book title")
            thumbnail_image = chapter.get('Thumbnail image')
            video_cover = chapter.get('Video cover')

            if pd.isna(thumbnail_image):
                logger.warning(f"Missing Thumbnail image: {thumbnail_image} for chapter: {chapter_name}")
                continue

            if pd.isna(video_cover):
                logger.warning(f"Missing Video cover: {video_cover} for chapter: {chapter_name}")
                continue

            image_path = str(IMG_DIR / chapter.get('Thumbnail image'))
            image_path_output = str(IMG_DIR / chapter.get('Video cover'))

            # Define the local path for the output audio
            local_audio_path = str(AUDIO_DIR / f"{sanitize_filename(book_title)}")
            logger.info(f"local_audio_path: {local_audio_path}")

            image_output_path = str(TEMP_DIR / f"image_content_front_{uuid4()}.png")

            logger.info("Generating script for hook.")
            hook_text = content_generator.generate_video_cover_hook(video_subject=chapter_text[:500], language=language,
                                                                    custom_prompt="")
            logger.info(f"Generated hook text: {hook_text}")

            logger.info("Generating cover image.")
            creates_images(hook_text=hook_text, image_input_path=cover_image_path,
                           image_logo_path=image_logo_path, image_output_path=image_output_path)
            logger.info(f"Cover image generated at: {image_output_path}")

            logger.info("Generating intro and reflection for YouTube upload.")
            introduction, reflection = content_generator.generate_intro_and_reflection(chapter_text[:500], language)

            script = f"{introduction}\n\n{chapter_text}\n\n{reflection}"

            audio_paths = process_and_generate_audio(chapter_data=chapter, chapter_text=chapter_text,
                                                     audio_output_path=local_audio_path)

            logger.info(f"audio_paths: {audio_paths}")

            if not audio_paths:
                logging.error("TTS processing failed. Skipping video generation for this topic.")
                return

            manager = MuseTalkConfigManager(str(MUSETALK_CONFIG_PATH))
            manager.update_avatar_audio_clips_from_segments(segments=audio_paths,
                                                            preparation=True,
                                                            video_avatar_1_path=str(AVATARS_DIR / "avatar_01.mp4"),
                                                            video_avatar_2_path=str(AVATARS_DIR / "avatar_02.mp4"),
                                                            )

            manager.save_yaml()  # Overwrites original YAML

            try:
                logger.info("[INFO] Starting MuseTalk inference from class method...")
                video_paths = run_musetalk_inference()
                # video_paths = musetalk_inference_reloaded()
                logger.info(f"[INFO] Inference finished successfully from class method.{video_paths}")

            except Exception as e:
                logger.info(f"[ERROR] Inference failed in class: {e}")
                raise

            combine_start = time.time()

            ordered_paths = build_ordered_output_list(video_paths)

            logger.info(colored(text=f"[+] ordered_paths: {ordered_paths}", color="green"))

            output_video_path = str(TEMP_DIR / f"{uuid.uuid4()}.mp4")

            processed_clips = []

            for video_path in ordered_paths:
                clip = process_single_video(video_path)
                if clip:
                    processed_clips.append(clip)

            if processed_clips is None:
                logging.error("No processed clips to combine.")
                return False

            logger.info(colored(text=f"[+] processed_clips: {processed_clips}", color="green"))

            combine_video_path = combine_videos_with_xfade(video_paths=processed_clips, output_path=output_video_path)

            if not combine_video_path:
                logger.critical("Could not create the background video because FFmpeg failed. Aborting the process.")
                return  # or raise Exception("Failed to create the background video")
            else:
                logger.info(colored(text=f"[+] Video combine_video_path: {combine_video_path}", color="green"))

            generator = SubtitleGenerator(combine_video_path)

            print("\n--- Generating typewriter subtitles ---")
            logger.info(colored(text=f"[+] --- Generating typewriter subtitles ---", color="green"))
            subtitles_path = generator.generate_typewriter_subtitles()

            final_video_path = generate_video_audio_included(combined_video_path=combine_video_path,
                                                             music_path=music_path,
                                                             threads=n_threads,
                                                             subtitles_path=subtitles_path,
                                                             subtitles_position=subtitles_position,
                                                             cover_image_path=image_output_path,
                                                             end_image_path=image_back_path,
                                                             logo_path=image_logo_path,
                                                             show_progress=True
                                                             )

            if not final_video_path:
                logger.critical("Could not create the background video because FFmpeg failed. Aborting the process.")
                return  # or raise Exception("Failed to create the background video")
            else:
                logger.debug(colored(text=f"[+] Video final_video_path: {final_video_path}", color="green"))

            combine_end = time.time()
            logging.info(
                f"Video combination took {combine_end - combine_start:.2f} seconds. final_video_path: {final_video_path}")

            logging.info("Generating metadata for YouTube upload.")
            title, description, keywords = content_generator.generate_metadata_shorts(book_title,
                                                                                      chapter_text[:500],
                                                                                      language)

            if not enable_youtube:
                logger.warning("Youtube upload is NOT Enabled")
                continue

            # Example: Schedule for Saturday at 9:00 AM, default timezone
            schedule = "weekly_sunday_5am"
            publish_video_at = get_next_publish_datetime(schedule_name=schedule, day='sunday', target_time='5:00',
                                                         start_date='2025-11-08', target_week=True, target_day=False)

            metadata = {
                'title': title,
                'description': f"{chapter_name}\n\n{description}\n\n",
                'keywords': ",".join(keywords),
                'publish_at': publish_video_at,
            }

            logger.info(f"Generated metadata: {metadata}")

            logger.info("Uploading video to YouTube.")

            # New call to capture the results
            video_url, upload_status = youtube_uploader.handle_video_upload(video_path=final_video_path,
                                                                            thumbail_path=image_path,
                                                                            metadata=metadata)

            # You can use these values to display the information
            if upload_status == "Success":
                print(f"The video was uploaded successfully. URL: {video_url}")
            else:
                print(f"The video could not be uploaded. Status: {upload_status}")
                return

            output_video_folder = os.path.join(OUTPUT_DIR, os.path.basename(final_video_path))
            shutil.move(final_video_path, output_video_folder)
            logger.info(f"File moved successfully to: {output_video_folder}")

            # Save the data in excel file
            excel_handler.write_data_by_title(title=chapter_name, sheet_name='Content', copy_text=description,
                                              video_path=output_video_folder, schedule=publish_video_at)

            logger.info(f"Starting to create Short Videos!")

            result = process_video(
                video_path=output_video_folder,
                output_dir=str(OUTPUT_SHORTS_DIR),
                num_clips=5,
                min_duration=60,
                max_duration=90,
                subtitle_style=subtitle_style,
                book_title=book_title
            )

            logger.info(f"result -->: {result}")

            # Prepare data for write_data
            data_to_write = []
            successful_uploads = 0
            failed_uploads = 0

            for index, clip in enumerate(result, 1):

                title = clip['metadata']['book_title']
                logging.info(f"Generating metadata for YouTube upload. {title}")

                title, description, keywords = content_generator.generate_metadata_shorts(title,
                                                                                          clip['segment_text'],
                                                                                          language)
                logger.info(f"Generated metadata title: {title} description:  {description} keywords:  {keywords}")

                # Example: Schedule for Friday at 10:00 AM, default timezone
                schedule = "sunday_5am"
                logger.info(f"Generated sunday_5am: {schedule}")
                params = convert_datetime(publish_video_at)
                logger.info(f"Generated params[day]: {params['day']} - "
                             f"params[target_time]: {params['target_time']} - "
                             f"params[start_date]: {params['start_date']}")

                publish_short_at = get_next_publish_datetime(schedule_name=schedule,
                                                             day=params["day"],
                                                             target_time=params["target_time"],
                                                             start_date=params["start_date"],
                                                             target_week=False,
                                                             target_day=True,
                                                             interval_hours=24)

                metadata = {
                    'title': title,
                    'description': f"{video_url}\n\n{description}\n\n",
                    'keywords': ",".join(keywords),
                    'publish_at': publish_short_at,
                }

                logger.info("Uploading Short Video to YouTube.")

                logger.info(f"Prepared video path: {clip['clip_path']}")

                # New call to capture the results
                short_video_url, short_pload_status = youtube_uploader.handle_video_upload(video_path=clip['clip_path'],
                                                                                           thumbail_path=image_path,
                                                                                           metadata=metadata)

                # You can use these values to display the information
                if short_pload_status == "Success":
                    logger.info(f"The video was uploaded successfully. URL: {video_url}")
                else:
                    logger.info(f"The video could not be uploaded. Status: {short_pload_status}")
                    return

                # Ensure publish_short_at is timezone-unaware
                if publish_short_at.tzinfo is not None:
                    logger.debug(f"Removing timezone from publish_short_at: {publish_short_at}")
                    publish_short_at = publish_short_at.replace(tzinfo=None)

                entry = {
                    'Id': str(uuid.uuid4()),
                    'Book title': title,
                    'Title': chapter_name,
                    'Chapter': "N/A",
                    'Copywriting': description,
                    'Schedule': publish_short_at,
                    'Output path': clip['clip_path'],
                }
                data_to_write.append(entry)
                logger.info(f"Prepared video: {chapter_name} for {publish_short_at}")

                if not enable_instagram:
                    logger.warning("Instagram upload is NOT Enabled")
                    continue

                # Start Uploading Instagram Reel Video
                logging.info("Uploading Instagram Reel.")

                public_id = f"reel_{index}_{int(time.time())}"
                video_url = cloudinary_uploader.upload_video(clip['clip_path'], public_id=public_id)

                if not video_url:
                    logger.error("Cloudinary upload failed.")
                    failed_uploads += 1
                    continue

                # For scheduled publishing
                container_id = instagram_api.create_scheduled_container(caption=description,
                                                                        video_url=video_url,
                                                                        publish_time=int(publish_short_at.timestamp()))

                if container_id:
                    successful_uploads += 1
                    logger.info(f"✅ Instagram Reel Video {index} scheduled successfully!")
                else:
                    failed_uploads += 1
                    logger.error(f"❌ Complete failure for Instagram Reel {index}.")

                # Wait between requests
                time.sleep(15)

            logger.info(f"Summary: Successful: {successful_uploads}, Failed: {failed_uploads}")

            # Write all entries to Shorts sheet
            if data_to_write:
                excel_handler.write_data(data=data_to_write, sheet_name='Shorts')
                logging.info(f"Scheduled {len(data_to_write)} videos in Shorts sheet.")
            else:
                logging.warning("No videos scheduled. All chapters were skipped or Content sheet is empty.")

        # Store the end time
        end_time = time.time()

        # Calculate the total time
        elapsed_time = end_time - start_time

        # Print the results
        logger.info("Task completed!")
        logger.info(f"The process took {elapsed_time:.2f} seconds to run.")

    except Exception as e:
        logger.critical(f"An unexpected error occurred in main: {e}", exc_info=True)


if __name__ == "__main__":

    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Script directory: {os.path.dirname(__file__)}")

    n_threads = psutil.cpu_count(logical=True)
    logger.info(f"Your system can use {n_threads} threads based on available CPU resources.")

    try:

        logging.info("Starting the video generation process.")
        initialize_directories()

        if not initialize_app():
            logger.info(colored("[-] Failed to initialize the application. Check logs for errors.", "red"))
            exit(1)

        enable_youtube = True
        enable_instagram = False

        create_podcast()

    except Exception as e:
        logger.error(f"[ERROR] Inference failed in class: {e}")
        raise
