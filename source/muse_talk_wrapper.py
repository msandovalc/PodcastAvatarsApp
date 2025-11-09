import yaml
from pathlib import Path
from collections import OrderedDict
from typing import List
from dotenv import load_dotenv
from logger_config import setup_logger


# --- Configuration ---
logger = setup_logger(__name__)

dotenv_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Get the project root (adjust according to your repo structure)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = str(PROJECT_ROOT / "MuseTalk" / "data" / "video"/ "yongen.mp4")
YAML_PATH = str(PROJECT_ROOT / "MuseTalk" / "configs" / "inference"/ "realtime.yaml")

class MuseTalkConfigManager:
    """
    Manager for MuseTalk inference YAML configurations.
    Allows loading, updating avatars, and saving YAML files with logging and error handling.
    """

    def __init__(self, yaml_path: str):
        self.yaml_path = Path(yaml_path)
        self.config = {}
        self._load_yaml()

    def _load_yaml(self):
        """Load YAML configuration from file, with error handling."""
        try:
            if not self.yaml_path.exists():
                raise FileNotFoundError(f"YAML file not found: {self.yaml_path}")
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
            logger.info(f"YAML configuration loaded successfully from: {self.yaml_path}")
        except Exception as e:
            logger.error(f"Failed to load YAML: {e}")
            raise

    def update_avatar_audio_clips_from_segments(self, segments: list[dict],
                                                preparation: bool = True,
                                                bbox_shift: int = 5,
                                                video_avatar_1_path: str = VIDEO_PATH,
                                                # Robust change: Default to None for the second avatar path
                                                video_avatar_2_path: str | None = None,
                                                output_yaml: str = YAML_PATH):
        """
        Updates MuseTalk YAML avatars automatically based on podcast segments.

        This function supports both single-avatar and dual-avatar configurations.
        If video_avatar_2_path is None or identical to video_avatar_1_path, all speakers
        will be mapped to the single video defined by video_avatar_1_path.

        Args:
            segments (list[dict]): List of transcription segments including 'speaker' and 'audio_path'.
            preparation (bool): MuseTalk preparation flag.
            bbox_shift (int): MuseTalk bounding box shift value.
            video_avatar_1_path (str): The video path for the first avatar (Speaker 1).
            video_avatar_2_path (str | None): The video path for the second avatar (Speaker 2). Defaults to None for single-avatar mode.
            output_yaml (str): Path to the output YAML configuration file.
        """
        try:
            # === Step 1: Clear YAML / config ===
            if hasattr(self, "config") and isinstance(self.config, dict):
                logger.info("Clearing existing YAML configuration...")
                self.config.clear()
            else:
                self.config = {}

            # Ensure the YAML file itself is also cleared (truncate)
            if Path(output_yaml).exists():
                with open(output_yaml, "w", encoding="utf-8") as f:
                    f.truncate(0)
                logger.info(f"YAML file content cleared: {output_yaml}")

            # === Step 2: Process new segments ===

            # Determine if a dual-avatar setup is intended (only if path 2 is explicitly set and different)
            is_dual_avatar = (video_avatar_2_path is not None) and (video_avatar_1_path != video_avatar_2_path)

            # Define the final path for Avatar 2. If single-avatar mode, this defaults to Path 1.
            final_video_avatar_2_path = str(video_avatar_2_path) if is_dual_avatar else str(video_avatar_1_path)

            for segment in segments:
                speaker = segment.get("speaker")
                audio_path = segment.get("audio_path")

                # Validation
                if not speaker or not audio_path or not Path(audio_path).exists():
                    reason = []
                    if not speaker: reason.append("Missing speaker")
                    if not audio_path:
                        reason.append("Missing audio_path")
                    elif not Path(audio_path).exists():
                        reason.append("Audio file does not exist")
                    logger.warning(f"Skipping invalid segment: {segment} | Reasons: {', '.join(reason)}")
                    continue

                # Normalize speaker -> avator_X key (e.g., "Speaker 1" -> "avator_1")
                speaker_str = str(speaker)
                speaker_num = ''.join(c for c in speaker_str if c.isdigit()) or "1"
                avatar_key = f"avator_{speaker_num}"

                # --- Conditional Video Path Selection ---

                if avatar_key == "avator_1":
                    # Avatar 1 always uses Path 1
                    selected_video_path = str(video_avatar_1_path)

                elif avatar_key == "avator_2":
                    # Avatar 2 uses the dynamically determined path (Path 2 if dual, Path 1 if single)
                    selected_video_path = final_video_avatar_2_path

                else:
                    # Any other avatar (e.g., avator_3) defaults to Path 1
                    selected_video_path = str(video_avatar_1_path)

                # Initialize avatar configuration if it doesn't exist
                if avatar_key not in self.config:
                    self.config[avatar_key] = {
                        "preparation": preparation,
                        "bbox_shift": bbox_shift,
                        "video_path": selected_video_path,
                        "audio_clips": {}
                    }

                # Append audio preserving order
                audio_clips = self.config[avatar_key]["audio_clips"]
                clip_index = len(audio_clips)
                audio_clips[f"audio_{clip_index}"] = str(audio_path)

            # === Step 3: Save updated YAML ===
            with open(output_yaml, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, sort_keys=False, allow_unicode=True)

            logger.info(f"YAML successfully saved to {output_yaml}")

        except Exception as e:
            logger.error(f"Failed to update avatars from segments: {e}")
            raise

    def save_yaml(self, output_path: str = None):
        """
        Saves the current configuration to a YAML file with logging.

        Args:
            output_path (str): Optional path to save; defaults to original YAML path.
        """
        try:
            path_to_save = Path(output_path) if output_path else self.yaml_path
            with open(path_to_save, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"YAML saved successfully at: {path_to_save}")
        except Exception as e:
            logger.error(f"Failed to save YAML to {output_path or self.yaml_path}: {e}")
            raise
