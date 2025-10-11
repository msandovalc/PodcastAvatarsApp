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
MAX_WORKERS = 4           # Maximum threads for MuseTalkRunner
MAX_CLIP_DURATION = 120.0 # Maximum clip duration in seconds

# ---------------- Ensure directories exist ----------------
for path in [RESULTS_DIR, TEMP_DIR]:
    os.makedirs(path, exist_ok=True)

@dataclass
class Config:
    # ---------------- General / Model Configs ----------------
    version: str = DEFAULT_VERSION          # MuseTalk version (v1, v15)
    ffmpeg_path: Path = FFMPEG_PATH         # Path to ffmpeg binary
    gpu_id: int = DEFAULT_GPU_ID            # GPU device index
    vae_type: str = VAE_TYPE                # VAE type for latent generation

    # ---------------- Path Configs ----------------
    unet_config: Path = MUSE_UNET_CONFIG    # Path to UNet config JSON
    unet_model_path: Path = MUSE_UNET_MODEL # Path to UNet PyTorch model
    whisper_dir: Path = WHISPER_DIR         # Path to Whisper model
    inference_config_path: str = None       # Optional path to inference config YAML
    bbox_shift: int = 0                     # Shift applied to face bounding boxes
    extra_margin: int = DEFAULT_EXTRA_MARGIN# Extra margin added to bounding boxes

    # ---------------- Output / I/O ----------------
    result_dir: Path = RESULTS_DIR          # Directory to save results
    output_vid_name: str = None             # Output video filename

    # ---------------- Audio / Time Params ----------------
    fps: int = DEFAULT_FPS                  # Frames per second for output video
    audio_padding_length_left: int = DEFAULT_AUDIO_PADDING_LEFT   # Padding left for audio clips
    audio_padding_length_right: int = DEFAULT_AUDIO_PADDING_RIGHT # Padding right for audio clips

    # ---------------- Processing / Flags ----------------
    batch_size: int = DEFAULT_BATCH_SIZE   # Batch size for processing frames
    use_saved_coord: bool = False          # Flag to reuse previously saved coordinates
    saved_coord: bool = False              # Indicates if saved coordinates exist
    skip_save_images: bool = DEFAULT_SKIP_SAVE_IMAGES # Skip saving intermediate images

    # ---------------- Face Parsing / Blending ----------------
    parsing_mode: str = 'jaw'              # Mode used for face parsing / blending
    left_cheek_width: int = DEFAULT_LEFT_CHEEK   # Width of left cheek for mask
    right_cheek_width: int = DEFAULT_RIGHT_CHEEK # Width of right cheek for mask

    # ---------------- Directories (Paths) ----------------
    results_dir: Path = RESULTS_DIR
    temp_dir: Path = TEMP_DIR
    configs_dir: Path = CONFIGS_DIR
    podcast_dir: Path = PODCAST_DIR
    audio_dir: Path = AUDIO_DIR


# ---------------- Global config instance ----------------
CONFIG = Config()

