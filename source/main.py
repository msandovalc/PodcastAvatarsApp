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
# from youtube import *
# from OpusClipApp import *
# from instagram_api import InstagramAPI
from video import *

from muse_talk_wrapper import *
from MuseTalk.scripts.realtime_inference import run_musetalk_inference


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


def initialize_app():
    """Initializes global objects and configurations, checking for required API keys."""
    global tts_generator, excel_handler, content_generator, gcp_manager, instagram_api
    tts_generator = None
    excel_handler = None
    content_generator = None
    gcp_manager = None
    instagram_api = None
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

    # Define the local output directory for audio files
    local_audio_dir = AUDIO_DIR

    # 1. Clean the local audio folder if it already exists
    # if local_output_dir.exists():
    #     for file in local_output_dir.iterdir():
    #         if file.is_file():
    #             os.remove(file)
    #     logger.info(f"Local folder '{local_output_dir}' content cleared.")

    local_audio_dir.mkdir(parents=True, exist_ok=True)

    if app_initialized:
        logging.info("Application initialization completed successfully.")
        return True
    else:
        logging.error("Application initialization failed due to missing or invalid configurations.")
        return False


def initialize_directories():
    """Initializes and cleans temporary and subtitle directories."""
    logging.info("Initializing directories.")
    clean_dir(str(TEMP_DIR))
    clean_dir(str(SUBTITLES_DIR))
    # clean_dir(str(AUDIO_DIR))
    # clean_dir(str(OUTPUT_DIR))

    logging.info(f"Cleaned temporary directory: {TEMP_DIR}")
    logging.info(f"Cleaned subtitles directory: {SUBTITLES_DIR}")
    # logging.info(f"Cleaned subtitles directory: {AUDIO_DIR}")
    # logging.info(f"Cleaned subtitles directory: {OUTPUT_DIR}")


def debug():
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


# def process_video(video_path, output_dir=OUTPUT_SHORTS_DIR, num_clips=None, min_duration=60, max_duration=90,
#                   transcription_dir=TRANSCRIPT_DIR, analysis_dir=ANALYSES_DIR,
#                   subtitle_style=None, book_title=None):
#     """
#     Process a video to generate short clips with subtitles, with variable duration between min_duration and max_duration.
#
#     Args:
#         video_path (str): Path to the input video.
#         output_dir (str): Directory to save generated clips.
#         num_clips (int, optional): Number of clips to generate. If None, generate clips for all valid segments.
#         min_duration (int): Minimum duration of each clip in seconds.
#         max_duration (int): Maximum duration of each clip in seconds.
#         transcription_dir (str): Directory to save/load transcription.
#         analysis_dir (str): Directory to save/load analysis.
#         subtitle_style (dict): Configuration for subtitle styling.
#         book_title (str, optional): Title of the book to include in clip metadata.
#
#     Returns:
#         list: List of dicts with 'path', 'text', 'srt_content', and 'book_title' for each generated clip.
#     """
#     log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
#     Path(log_dir).mkdir(exist_ok=True)
#     log_file = os.path.join(log_dir, "opus_clip_clone.log")
#
#     # setup_logging(log_file)
#
#     try:
#         logger.info("Starting processing for video: %s", video_path)
#         use_cuda = torch.cuda.is_available()
#         logger.info("Using %s CPU cores, CUDA available: %s", os.cpu_count(), use_cuda)
#         if use_cuda:
#             logger.debug("CUDA device: %s, memory allocated: %.2f MB",
#                          torch.cuda.get_device_name(0), torch.cuda.memory_allocated(0) / 1024 ** 2)
#         # Derive paths based on video filename
#         video_basename = os.path.basename(video_path)
#         audio_basename = f"temp_audio_{video_basename}.wav"
#         transcription_path = os.path.join(transcription_dir, f"transcription_{video_basename}.json")
#         analysis_path = os.path.join(analysis_dir, f"analysis_{video_basename}.json")
#         Path(transcription_dir).mkdir(exist_ok=True)
#         Path(analysis_dir).mkdir(exist_ok=True)
#
#         # Initialize components
#         logger.info("Initializing VideoProcessor")
#         video_processor = VideoProcessor(video_path, use_cuda=True)
#         logger.info("Initializing AudioTranscriber")
#         audio_transcriber = AudioTranscriber(use_cuda=True)
#         logger.info("Initializing ContentAnalyzer")
#         content_analyzer = ContentAnalyzer(use_cuda=True, max_workers=1)
#         logger.info("Initializing ClipGenerator")
#         clip_generator = ClipGenerator(video_path, output_dir=output_dir, subtitle_style=subtitle_style,
#                                        use_background=True, background_color="white")
#
#         # Extract audio
#         logger.info("Extracting audio")
#         audio_path = video_processor.extract_audio()
#
#         # Check for existing transcription
#         logger.info("Checking for transcription at: %s", transcription_path)
#         segments = None
#         if os.path.exists(transcription_path):
#             try:
#                 with open(transcription_path, "r", encoding="utf-8") as f:
#                     transcription_data = json.load(f)
#                 if isinstance(transcription_data, list):
#                     segments = transcription_data
#                     logger.warning("Legacy transcription format detected at %s, converting to new format",
#                                    transcription_path)
#                     transcription_data = {"audio_path": audio_path, "segments": segments}
#                     with open(transcription_path, "w", encoding="utf-8") as f:
#                         json.dump(transcription_data, f, indent=4)
#                     logger.info("Converted transcription to new format at %s", transcription_path)
#                 elif not isinstance(transcription_data, dict):
#                     logger.warning("Invalid transcription format at %s: not a dict or list", transcription_path)
#                 elif (os.path.normpath(transcription_data.get("audio_path", "")) == audio_path and
#                       isinstance(transcription_data.get("segments"), list) and
#                       len(transcription_data["segments"]) > 0):
#                     segments = transcription_data["segments"]
#                     logger.info("Reusing existing transcription with %s segments from %s",
#                                 len(segments), transcription_path)
#                 else:
#                     logger.warning("Invalid or mismatched transcription file at %s: missing audio_path or segments",
#                                    transcription_path)
#             except Exception as e:
#                 logger.error("Error loading transcription file %s: %s", transcription_path, e, exc_info=True)
#
#         # Transcribe if no valid segments
#         if segments is None:
#             logger.info("Starting transcription")
#             segments = audio_transcriber.transcribe(audio_path, save_path=transcription_path)
#             logger.info("Transcription completed, %s segments found", len(segments))
#
#         # Check for existing analysis
#         logger.info("Checking for analysis at: %s", analysis_path)
#         scores = None
#         if os.path.exists(analysis_path):
#             try:
#                 with open(analysis_path, "r", encoding="utf-8") as f:
#                     analysis_data = json.load(f)
#                 if (isinstance(analysis_data, dict) and
#                         os.path.normpath(analysis_data.get("video_path", "")) == video_path and
#                         isinstance(analysis_data.get("segments"), list) and
#                         len(analysis_data["segments"]) == len(segments) and
#                         all(a.get("start") == b["start"] and a.get("text") == b["text"] and
#                             isinstance(a.get("score"), (int, float))
#                             for a, b in zip(analysis_data["segments"], segments))):
#                     scores = [s["score"] for s in analysis_data["segments"]]
#                     logger.info("Reusing existing analysis with %s scores from %s", len(scores), analysis_path)
#                     logger.info("Scores: %s", scores)
#                 else:
#                     logger.info("Analysis file mismatch or invalid scores, re-running analysis")
#             except Exception as e:
#                 logger.error("Error loading analysis file %s: %s", analysis_path, e, exc_info=True)
#
#         # Analyze content if no valid scores
#         if scores is None:
#             logger.info("Analyzing video content")
#             scores = content_analyzer.analyze_segments_batch(video_processor, segments, use_ffmpeg_frames=True,
#                                                              save_path=analysis_path)
#             logger.info("Analysis completed, %s scores generated", len(scores))
#             logger.info("Scores: %s", scores)
#
#         # Select top segments
#         logger.info("Selecting top segments")
#         valid_scores = [(i, score) for i, score in enumerate(scores) if score is not None]
#         if not valid_scores:
#             logger.error("No valid scores available for segment selection")
#             raise ValueError("No valid scores available for segment selection")
#         num_clips_to_generate = len(valid_scores) if num_clips is None else min(num_clips, len(valid_scores))
#         top_indices = sorted(valid_scores, key=lambda x: x[1], reverse=True)[:num_clips_to_generate]
#         selected_segments = [segments[i] for i, _ in top_indices]
#         logger.info("Selected %s clips for generation: %s", len(selected_segments),
#                     [s['start'] for s in selected_segments])
#         for i, seg in enumerate(selected_segments, 1):
#             logger.info("Selected segment %s: start=%s, text=%s", i, seg['start'], seg['text'])
#
#         # Generate clips
#         logger.info("Generating clips with variable duration between %ss and %ss", min_duration, max_duration)
#         clip_data = clip_generator.generate_clips(selected_segments, min_duration=min_duration,
#                                                   max_duration=max_duration,
#                                                   all_segments=segments)
#         # Add book title to clip data
#         for clip in clip_data:
#             clip["metadata"]["book_title"] = book_title if book_title else "Untitled Book"
#         logger.info("Generated %s clips with metadata", len(clip_data))
#
#         # Cleanup
#         logger.info("Closing video processor")
#         video_processor.close()
#
#         logger.info("Processing completed successfully")
#         return clip_data
#
#     except Exception as e:
#         logger.error("Error processing video: %s", e, exc_info=True)
#         raise
#     finally:
#         if 'video_processor' in locals():
#             video_processor.close()
#
#

def create_podcast():

    logger.info("Starting the main application for long audio generation.")

    # subtitles_path = str(TEMP_DIR / "e5fb9605-cc28-4160-ab77-0b9546ff08c1.srt")
    cover_image_path = str(PODCAST_DIR / "cover" / "image_content_front.png")
    image_back_path = str(PODCAST_DIR / "cover" / "image_content_back_text.png")
    image_logo_path = str(PODCAST_DIR / "cover" / "image_logo_content.png")
    image_output_path = str(TEMP_DIR / f"image_thumbail_{uuid.uuid4()}.png")
    music_path = str(PODCAST_DIR / "music" / "432Hz_postive_energy_break_chains_and_emotions.mp3")

    # final_video_path = str(OUTPUT_DIR / "output_a0ad19dc-dfb9-4f18-b1bd-92c8f4bbae71.mp4")
    # video_url = "https://youtu.be/eoFidzaJhwU"

    n_threads = 2
    subtitles_position = "center,center"
    language = "Spanish"

    try:

        subtitle_style = {
            "FontName": "Candara-Bold",
            "FontSize": 10,
            "PrimaryColour": "&H00FFFFFF",
            "BorderStyle": 1,
            "Outline": 2,
            "Alignment": 2,
            "MarginV": 60
        }

        # Store the start time
        start_time = time.time()

        logging.info("Starting the video generation process.")
        initialize_directories()

        excel_data = excel_handler.read_data(sheet_name='Content')

        if not excel_data:
            logging.error("No data found in the Excel file.")
            return False

        for chapter in excel_data:

            if chapter.get('Enabled') in (0, False):
                logging.info(f"Excel Enabled: {chapter.get('Enabled')}")
                continue

            chapter_name = chapter.get('Title')
            chapter_text = chapter.get('Chapter')
            book_title = chapter.get("Book title")

            # Sanitize Excel values before using them as paths
            thumbnail_image = chapter.get('Thumbnail image')
            video_cover = chapter.get('Video cover')

            if pd.isna(thumbnail_image):
                logging.warning(f"Missing Thumbnail image: {thumbnail_image} for chapter: {chapter_name}")
                continue

            if pd.isna(video_cover):
                logging.warning(f"Missing Video cover: {video_cover} for chapter: {chapter_name}")
                continue

            image_path = str(IMG_DIR / chapter.get('Thumbnail image'))
            image_path_output = str(IMG_DIR / chapter.get('Video cover'))

            # Define the local path for the output audio
            local_audio_path = str(AUDIO_DIR / f"{sanitize_filename(book_title)}")
            logging.info(f"local_audio_path: {local_audio_path}")

            # logging.info("Generating intro and reflection for YouTube upload.")
            # introduction, reflection = content_generator.generate_intro_and_reflection(chapter_text[:500], language)
            #
            # script = f"{introduction}\n\n{chapter_text}\n\n{reflection}"

            # audio_paths = process_and_generate_audio(chapter_data=chapter, chapter_text=chapter_text,
            #                                         audio_output_path=local_audio_path)

            audio_paths = [
                {'speaker': 'Speaker 1',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_1.wav'},
                {'speaker': 'Speaker 2',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_2.wav'},
                {'speaker': 'Speaker 1',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_3.wav'},
                {'speaker': 'Speaker 2',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_4.wav'},
                {'speaker': 'Speaker 1',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_5.wav'},
                {'speaker': 'Speaker 2',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_6.wav'},
                {'speaker': 'Speaker 1',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_7.wav'},
                {'speaker': 'Speaker 2',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_8.wav'},
                {'speaker': 'Speaker 1',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_9.wav'},
                {'speaker': 'Speaker 2',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_10.wav'},
                {'speaker': 'Speaker 1',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_11.wav'},
                {'speaker': 'Speaker 1',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_13.wav'},
                {'speaker': 'Speaker 2',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_14.wav'},
                {'speaker': 'Speaker 1',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_15.wav'},
                {'speaker': 'Speaker 2',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_16.wav'},
                {'speaker': 'Speaker 1',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_17.wav'},
                {'speaker': 'Speaker 2',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 2_segment_18.wav'},
                {'speaker': 'Speaker 1',
                 'audio_path': '/kaggle/working/PodcastAvatarsApp/podcast_content/audio/Hábitos Atómicos en el deporte/Speaker 1_segment_19.wav'}
            ]


            if not audio_paths:
                logging.error("TTS processing failed. Skipping video generation for this topic.")
                return

            manager = MuseTalkConfigManager(str(MUSETALK_CONFIG_PATH))
            manager.update_avatar_audio_clips_from_segments(audio_paths, preparation=True)
            manager.save_yaml()  # Overwrites original YAML

            try:
                print("[INFO] Starting MuseTalk inference from class method...")
                video_paths = run_musetalk_inference()
                print(f"[INFO] Inference finished successfully from class method.{video_paths}")

            except Exception as e:
                print(f"[ERROR] Inference failed in class: {e}")
                raise

            combine_start = time.time()
            output_paths = [r["output_path"] for r in video_paths]

            logger.info(colored(text=f"[+] Video combination path: {output_paths}", color="green"))

            # combined_video_path = combine_videos(video_paths=output_paths, max_duration=temp_audio.duration,
            #                                      max_clip_duration=max_clip_duration, threads=n_threads)
            # combine_end = time.time()
            # logging.info(
            #     f"Video combination took {combine_end - combine_start:.2f} seconds. Combined video path: {combined_video_path}")




            #
            # temp_audio = AudioFileClip(audio_path)
            #
            # logger.info(
            #     f"Starting single video creation for one element: {image_path_output}, duration: {temp_audio.duration}")
            # combined_video_path = process_single_video_one_element(video_path=image_path_output,
            #                                                        req_dur=temp_audio.duration,
            #                                                        show_progress=True)
            #
            # if not combined_video_path:
            #     logger.critical("Could not create the background video because FFmpeg failed. Aborting the process.")
            #     # Here you can terminate the execution in a controlled manner
            #     return  # or raise Exception("Failed to create the background video")
            # else:
            #     logger.debug(colored(text=f"[+] Video combination path: {combined_video_path}", color="green"))
            #
            # final_video_path = generate_video_custom(combined_video_path=combined_video_path, tts_path=audio_path,
            #                                          music_path=music_path, threads=n_threads,
            #                                          subtitles_position=subtitles_position, show_progress=True)
            #
            # if not final_video_path:
            #     logger.critical("Could not create the background video because FFmpeg failed. Aborting the process.")
            #     # Here you can terminate the execution in a controlled manner
            #     return  # or raise Exception("Failed to create the background video")
            # else:
            #     logger.debug(colored(text=f"[+] Video combination path: {final_video_path}", color="green"))
            #
            # logging.info("Generating metadata for YouTube upload.")
            # title, description, keywords = content_generator.generate_metadata_shorts(book_title,
            #                                                                           chapter_text[:500],
            #                                                                           language)
            #
            # # Example: Schedule for Saturday at 9:00 AM, default timezone
            # schedule = "weekly_tuesday_5am"
            # publish_video_at = get_next_publish_datetime(schedule_name=schedule, day='tuesday', target_time='5:00',
            #                                              start_date='2025-09-29', target_week=True, target_day=False)
            #
            # metadata = {
            #     'title': title,
            #     'description': f"{chapter_name}\n\n{description}\n\n",
            #     'keywords': ",".join(keywords),
            #     'publish_at': publish_video_at,
            # }
            #
            # logging.info(f"Generated metadata: {metadata}")
            #
            # logging.info("Uploading video to YouTube.")
            #
            # # New call to capture the results
            # video_url, upload_status = handle_video_upload(final_video_path, image_path, metadata)
            #
            # # You can use these values to display the information
            # if upload_status == "Success":
            #     print(f"The video was uploaded successfully. URL: {video_url}")
            # else:
            #     print(f"The video could not be uploaded. Status: {upload_status}")
            #
            # output_video_folder = os.path.join(OUTPUT_DIR, os.path.basename(final_video_path))
            # shutil.move(final_video_path, output_video_folder)
            # logging.info(f"File moved successfully to: {output_video_folder}")
            #
            # # Save the data in excel file
            # excel_handler.write_data_by_title(title=chapter_name, sheet_name='Content', copy_text=description,
            #                                   video_path=output_video_folder, schedule=publish_video_at)
            #
            # logging.info(f"Starting to create Short Videos!")
            #
            # result = process_video(
            #     video_path=output_video_folder,
            #     output_dir=str(OUTPUT_SHORTS_DIR),
            #     num_clips=5,
            #     min_duration=60,
            #     max_duration=90,
            #     subtitle_style=subtitle_style,
            #     book_title=book_title
            # )
            #
            # # Prepare data for write_data
            # data_to_write = []
            # successful_uploads = 0
            # failed_uploads = 0
            #
            # for index, clip in enumerate(result, 1):
            #
            #     title = clip['metadata']['book_title']
            #     print(f"Generating metadata for YouTube upload. {title}")
            #     title, description, keywords = content_generator.generate_metadata_shorts(title,
            #                                                                               clip['segment_text'],
            #                                                                               language)
            #     print(f"Generated metadata title: {title} description:  {description} keywords:  {keywords}")
            #
            #     # Example: Schedule for Friday at 10:00 AM, default timezone
            #     schedule = "tuesday_5am"
            #     logging.info(f"Generated tuesday_5am: {schedule}")
            #     params = convert_datetime(publish_video_at)
            #     logging.info(f"Generated params[day]: {params['day']} - "
            #                  f"params[target_time]: {params['target_time']} - "
            #                  f"params[start_date]: {params['start_date']}")
            #
            #     publish_short_at = get_next_publish_datetime(schedule_name=schedule,
            #                                                  day=params["day"],
            #                                                  target_time=params["target_time"],
            #                                                  start_date=params["start_date"],
            #                                                  target_week=False,
            #                                                  target_day=True,
            #                                                  interval_hours=24)
            #
            #     metadata = {
            #         'title': title,
            #         'description': f"{video_url}\n\n{description}\n\n",
            #         'keywords': ",".join(keywords),
            #         'publish_at': publish_short_at,
            #     }
            #
            #     logging.info("Uploading Short Video to YouTube.")
            #
            #     # New call to capture the results
            #     short_video_url, short_pload_status = handle_video_upload(clip['clip_path'], image_path, metadata)
            #
            #     # You can use these values to display the information
            #     if short_pload_status == "Success":
            #         print(f"The video was uploaded successfully. URL: {video_url}")
            #     else:
            #         print(f"The video could not be uploaded. Status: {short_pload_status}")
            #
            #     # Ensure publish_short_at is timezone-unaware
            #     if publish_short_at.tzinfo is not None:
            #         logging.debug(f"Removing timezone from publish_short_at: {publish_short_at}")
            #         publish_short_at = publish_short_at.replace(tzinfo=None)
            #
            #     entry = {
            #         'Id': str(uuid.uuid4()),
            #         'Book title': title,
            #         'Title': chapter_name,
            #         'Chapter': "N/A",
            #         'Copywriting': description,
            #         'Schedule': publish_short_at,
            #         'Output path': clip['clip_path'],
            #     }
            #     data_to_write.append(entry)
            #     logging.info(f"Prepared video: {chapter_name} for {publish_short_at}")
            #
            #
            #     # Start Uploading Instagram Reel Video
            #     logging.info("Uploading Instagram Reel.")
            #
            #     public_id = f"reel_{index}_{int(time.time())}"
            #     video_url = cloudinary_uploader.upload_video(clip['clip_path'], public_id=public_id)
            #
            #     if not video_url:
            #         logger.error("Cloudinary upload failed.")
            #         failed_uploads += 1
            #         continue
            #
            #     # For scheduled publishing
            #     container_id = instagram_api.create_scheduled_container(caption=description,
            #                                                             video_url=video_url,
            #                                                             publish_time=int(publish_short_at.timestamp()))
            #
            #     if container_id:
            #         successful_uploads += 1
            #         logger.info(f"✅ Instagram Reel Video {index} scheduled successfully!")
            #     else:
            #         # Fallback: Publish immediately
            #         if instagram_api.publish_immediately(video_url):
            #             successful_uploads += 1
            #             logger.warning(f"⚠️ Immediate publication for Instagram Reel {index}.")
            #         else:
            #             failed_uploads += 1
            #             logger.error(f"❌ Complete failure for Instagram Reel {index}.")
            #
            #     # Wait between requests
            #     time.sleep(15)
            #
            # logger.info(f"Summary: Successful: {successful_uploads}, Failed: {failed_uploads}")
            #
            # # Write all entries to Shorts sheet
            # if data_to_write:
            #     excel_handler.write_data(data=data_to_write, sheet_name='Shorts')
            #     logging.info(f"Scheduled {len(data_to_write)} videos in Shorts sheet.")
            # else:
            #     logging.warning("No videos scheduled. All chapters were skipped or Content sheet is empty.")

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

    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(__file__)}")

    n_threads = psutil.cpu_count(logical=True)
    print(f"Your system can use {n_threads} threads based on available CPU resources.")

    try:

        logging.info("Starting the video generation process.")
        initialize_directories()

        if not initialize_app():
            print(colored("[-] Failed to initialize the application. Check logs for errors.", "red"))
            exit(1)

        create_podcast()

        # print("[INFO] Starting MuseTalk inference from class method...")
        # results_list = run_musetalk_inference()
        # results_list = inferense_v2()
        # print(f"[INFO] Inference finished successfully from class method.: {results_list}")

    except Exception as e:
        print(f"[ERROR] Inference failed in class: {e}")
        raise