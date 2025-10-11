import whisper
import logging
import os
import json
from pathlib import Path
from tqdm import tqdm
import torch
from .logger_config import setup_logger

# --- Configuration ---
logger = setup_logger(__name__)

class AudioTranscriber:
    """Handles audio transcription using Whisper model."""

    def __init__(self, model_size="base", use_cuda=True, language="es"):
        """
        Initialize the audio transcriber with Whisper model.

        Args:
            model_size (str): Whisper model size (e.g., 'base', 'small').
            use_cuda (bool): Whether to use CUDA for transcription.
            language (str): Language code for transcription (e.g., 'es' for Spanish).
        """
        try:
            # Setup logging
            self.logger = logger
            self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
            self.language = language
            self.logger.debug("Loading Whisper model '%s' on %s", model_size, self.device)
            self.model = whisper.load_model(model_size, device=self.device)
            self.logger.info("Whisper model '%s' loaded on %s", model_size, self.device)
            if self.device == "cuda":
                self.logger.debug("CUDA memory allocated: %.2f MB", torch.cuda.memory_allocated(0) / 1024 ** 2)
        except Exception as e:
            self.logger.error("Failed to load Whisper model: %s", e, exc_info=True)
            raise

    def transcribe(self, audio_path, save_path=None):
        """
        Transcribe audio file and return segments.

        Args:
            audio_path (str): Path to the audio file.
            save_path (str): Optional path to save the transcription.

        Returns:
            list: List of transcription segments with start, end, and text.
        """
        try:
            audio_path = os.path.normpath(audio_path)
            if save_path is None:
                save_path = os.path.normpath(
                    "temp/transcriptions/transcription_" + os.path.basename(audio_path) + ".json")
            Path(save_path).parent.mkdir(exist_ok=True)

            self.logger.debug("Checking for transcription at: %s", save_path)
            if os.path.exists(save_path):
                try:
                    with open(save_path, "r", encoding="utf-8") as f:
                        transcription_data = json.load(f)
                    if not transcription_data.get("audio_path"):
                        self.logger.warning("Transcription file missing 'audio_path': %s", save_path)
                    elif os.path.normpath(transcription_data["audio_path"]) != audio_path:
                        self.logger.warning(
                            "Audio path mismatch: expected '%s', got '%s'", audio_path, transcription_data["audio_path"])
                    elif not isinstance(transcription_data.get("segments"), list):
                        self.logger.warning("Invalid segments in transcription file: %s", save_path)
                    elif len(transcription_data["segments"]) == 0:
                        self.logger.warning("Empty segments in transcription file: %s", save_path)
                    else:
                        self.logger.info(
                            "Reusing existing transcription with %s segments from %s",
                            len(transcription_data["segments"]), save_path)
                        return transcription_data["segments"]
                except Exception as e:
                    self.logger.error("Error loading transcription file %s: %s", save_path, e, exc_info=True)
                    self.logger.warning("Retranscribing due to invalid transcription file")

            self.logger.debug("Starting transcription for audio: %s", audio_path)
            result = self.model.transcribe(audio_path, verbose=False, language=self.language)
            segments = [{"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()} for seg in result["segments"]]

            self.logger.debug("Saving transcription to: %s", save_path)
            transcription_data = {"audio_path": audio_path, "segments": segments}
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(transcription_data, f, indent=4)
            self.logger.info("Transcription saved to: %s", save_path)

            return segments
        except Exception as e:
            self.logger.error("Error transcribing audio: %s", e, exc_info=True)
            raise