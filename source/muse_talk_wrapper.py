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

    import yaml
    from pathlib import Path
    import logging

    logger = logging.getLogger(__name__)

    import yaml
    from pathlib import Path
    import logging

    logger = logging.getLogger(__name__)

    def update_avatar_audio_clips_from_segments(self, segments: list[dict],
                                                preparation: bool = True,
                                                bbox_shift: int = 5,
                                                video_path: str = "data/video/yongen.mp4",
                                                output_yaml: str = "inference.yaml"):
        """
        Updates MuseTalk YAML avatars automatically based on podcast segments.

        Paths for video_path and audio_clips are always wrapped in double quotes.
        """
        try:
            for segment in segments:
                speaker = segment.get("speaker")
                audio_path = segment.get("audio_path")

                # Validation
                if not speaker or not audio_path or not Path(audio_path).exists():
                    reason = []
                    if not speaker:
                        reason.append("Missing speaker")
                    if not audio_path:
                        reason.append("Missing audio_path")
                    elif not Path(audio_path).exists():
                        reason.append("Audio file does not exist")
                    logger.warning(f"Skipping invalid segment: {segment} | Reasons: {', '.join(reason)}")
                    continue

                # Normalize speaker -> avator_X
                speaker_str = str(speaker)
                speaker_num = ''.join(c for c in speaker_str if c.isdigit()) or "1"
                avatar_key = f"avator_{speaker_num}"

                # Init avatar if missing
                if avatar_key not in self.config:
                    self.config[avatar_key] = {
                        "preparation": preparation,
                        "bbox_shift": bbox_shift,
                        "video_path": f"\"{video_path}\"",  # wrap in quotes
                        "audio_clips": {}
                    }

                # Append audio preserving order
                audio_clips = self.config[avatar_key]["audio_clips"]
                clip_index = len(audio_clips)
                audio_clips[f"audio_{clip_index}"] = f"\"{audio_path}\""  # wrap in quotes

            # Dump YAML normally (no representer hacks)
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
