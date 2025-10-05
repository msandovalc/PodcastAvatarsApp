# MuseTalk/scripts/config.py
import os
from pathlib import Path
from dataclasses import dataclass

# ---------------- Project Root ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------- Directories ----------------
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"
TEMP_DIR = PROJECT_ROOT / "temp"
PODCAST_DIR = PROJECT_ROOT / "podcast_content"
AUDIO_DIR = PODCAST_DIR / "audio"

# ---------------- Models ----------------
MUSE_UNET_CONFIG = MODELS_DIR / "musetalk/musetalk.json"
MUSE_UNET_MODEL = MODELS_DIR / "musetalk/pytorch_model.bin"
WHISPER_DIR = MODELS_DIR / "whisper"
VAE_TYPE = "sd-vae"
FFMPEG_PATH = PROJECT_ROOT / "ffmpeg-4.4-amd64-static"

# ---------------- Defaults ----------------
DEFAULT_VERSION = "v15"
DEFAULT_GPU_ID = 0
DEFAULT_FPS = 25
DEFAULT_BATCH_SIZE = 20
DEFAULT_AUDIO_PADDING_LEFT = 2
DEFAULT_AUDIO_PADDING_RIGHT = 2
DEFAULT_EXTRA_MARGIN = 10
DEFAULT_LEFT_CHEEK = 90
DEFAULT_RIGHT_CHEEK = 90
DEFAULT_SKIP_SAVE_IMAGES = False

# ---------------- System / Limits ----------------
MAX_WORKERS = 4           # Number of threads used in MuseTalkRunner
MAX_CLIP_DURATION = 120.0 # Max clip duration in seconds

# ---------------- Notes ----------------
# CONFIG is imported throughout MuseTalk scripts as the global configuration.
# Modify constants above to adjust model paths or runtime parameters.

# ---------------- Ensure directories ----------------
for path in [RESULTS_DIR, TEMP_DIR]:
    os.makedirs(path, exist_ok=True)

@dataclass
class Config:
    version: str = DEFAULT_VERSION
    gpu_id: int = DEFAULT_GPU_ID
    fps: int = DEFAULT_FPS
    batch_size: int = DEFAULT_BATCH_SIZE
    audio_padding_length_left: int = DEFAULT_AUDIO_PADDING_LEFT
    audio_padding_length_right: int = DEFAULT_AUDIO_PADDING_RIGHT
    extra_margin: int = DEFAULT_EXTRA_MARGIN
    left_cheek_width: int = DEFAULT_LEFT_CHEEK
    right_cheek_width: int = DEFAULT_RIGHT_CHEEK
    skip_save_images: bool = DEFAULT_SKIP_SAVE_IMAGES

    ffmpeg_path: Path = FFMPEG_PATH
    vae_type: str = VAE_TYPE
    unet_config: Path = MUSE_UNET_CONFIG
    unet_model_path: Path = MUSE_UNET_MODEL
    whisper_dir: Path = WHISPER_DIR
    results_dir: Path = RESULTS_DIR
    temp_dir: Path = TEMP_DIR
    podcast_dir: Path = PODCAST_DIR
    audio_dir: Path = AUDIO_DIR
    configs_dir: Path = CONFIGS_DIR

CONFIG = Config()
