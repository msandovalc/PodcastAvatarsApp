import os
import io
import time
import logging
import re
from pathlib import Path
from typing import List, Dict
from pydub import AudioSegment
from pydub.silence import split_on_silence
from google.cloud import texttospeech_v1 as tts
from google.cloud import storage
from google.api_core import exceptions
from logger_config import setup_logger
from utils import split_text_into_chunks

logger = setup_logger(__name__)

class GCPTextToSpeechManager:
    """
    Modernized Google Cloud TTS manager with multi-voice podcast support,
    long-text chunking, retry logic, and silence trimming.
    """

    MAX_RETRIES = 5
    RETRY_DELAY_SECONDS = 2

    VOICE_MAPPING = {
        "Speaker1": "es-US-Chirp3-HD-Laomedeia",
        "Speaker2": "es-US-Chirp3-HD-Algenib",
        "Speaker": "es-US-Chirp3-HD-Laomedeia"
    }

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name

        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path or not Path(credentials_path).exists():
            raise FileNotFoundError(
                "GCP credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS in .env"
            )
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

        self.tts_client = tts.TextToSpeechClient()
        self.storage_client = storage.Client()
        logger.info(f"✅ Initialized TTS client with bucket: {bucket_name}")

    def synthesize_long_audio(self, text: str, output_path: str, voice_name="es-US-Chirp3-HD-Algenib") -> str:
        """Synthesize long text by splitting into chunks."""
        chunks = split_text_into_chunks(text, max_bytes=4500)
        logger.info(f"DEBUG: {len(chunks)} chunks created for TTS")

        combined_audio = AudioSegment.empty()
        language_code = '-'.join(voice_name.split('-')[:2])
        voice = tts.VoiceSelectionParams(language_code=language_code, name=voice_name)
        audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

        for i, chunk_text in enumerate(chunks, start=1):
            input_text = tts.SynthesisInput(text=chunk_text)
            for attempt in range(self.MAX_RETRIES):
                try:
                    response = self.tts_client.synthesize_speech(
                        input=input_text,
                        voice=voice,
                        audio_config=audio_config
                    )
                    break
                except exceptions.InternalServerError as e:
                    if attempt < self.MAX_RETRIES - 1:
                        delay = self.RETRY_DELAY_SECONDS * (2 ** attempt)
                        logger.warning(f"Attempt {attempt+1} failed: {e}, retrying in {delay:.2f}s")
                        time.sleep(delay)
                    else:
                        logger.error(f"Final attempt failed for chunk {i}")
                        raise

            audio_segment = AudioSegment.from_file(io.BytesIO(response.audio_content),
                                                   format="raw", frame_rate=24000,
                                                   channels=1, sample_width=2)
            combined_audio += self._trim_audio_silence(audio_segment)

        combined_audio.export(output_path, format="wav")
        logger.info(f"Combined long audio exported to {output_path}")
        return output_path

    def _parse_podcast_script(self, text_script: str) -> List[Dict[str, str]]:
        """Parse podcast script with speaker labels into segments."""
        conversation = []
        lines = [line.strip() for line in text_script.strip().split('\n') if line.strip()]
        current_speaker, current_text = None, ""
        speaker_regex = re.compile(r"^(Speaker\d*):(.*)")

        for line in lines:
            match = speaker_regex.match(line)
            if match:
                if current_speaker:
                    conversation.append({"text": current_text.strip(),
                                         "voice_name": self.VOICE_MAPPING.get(current_speaker, "es-US-Chirp3-HD-Algenib")})
                current_speaker, current_text = match.group(1).strip(), match.group(2).strip()
            elif current_speaker:
                current_text += " " + line.strip()

        if current_speaker and current_text:
            conversation.append({"text": current_text.strip(),
                                 "voice_name": self.VOICE_MAPPING.get(current_speaker, "es-US-Chirp3-HD-Algenib")})
        return conversation

    def _trim_audio_silence(self, audio: AudioSegment) -> AudioSegment:
        """Trim leading/trailing silence."""
        non_silent = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
        if not non_silent:
            return AudioSegment.silent(duration=0)
        trimmed = non_silent[0]
        for seg in non_silent[1:]:
            trimmed += seg
        return trimmed

    def synthesize_podcast_audio(self, script: str, output_path: str,
                                 speaking_rate=1.0, pitch=0.0) -> str:
        """Generate multi-voice podcast audio."""
        conversation = self._parse_podcast_script(script)
        combined_audio = AudioSegment.empty()

        for i, segment in enumerate(conversation):
            text, voice_name = segment.get("text"), segment.get("voice_name")
            if not text or not voice_name:
                logger.warning(f"Skipping segment {i+1} due to missing data")
                continue

            language_code = '-'.join(voice_name.split('-')[:2])
            voice = tts.VoiceSelectionParams(language_code=language_code, name=voice_name)
            audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16,
                                           speaking_rate=speaking_rate, pitch=pitch)
            input_text = tts.SynthesisInput(text=text)

            for attempt in range(self.MAX_RETRIES):
                try:
                    response = self.tts_client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
                    break
                except exceptions.InternalServerError as e:
                    if attempt < self.MAX_RETRIES - 1:
                        delay = self.RETRY_DELAY_SECONDS * (2 ** attempt)
                        logger.warning(f"Attempt {attempt+1} failed for segment {i+1}, retrying in {delay:.2f}s")
                        time.sleep(delay)
                    else:
                        logger.error(f"Final attempt failed for segment {i+1}")
                        raise

            audio_segment = AudioSegment.from_file(io.BytesIO(response.audio_content),
                                                   format="raw", frame_rate=24000,
                                                   channels=1, sample_width=2)
            trimmed = self._trim_audio_silence(audio_segment)
            if i == 0:
                combined_audio = trimmed.fade_in(10)
            else:
                combined_audio = combined_audio.append(trimmed, crossfade=50)

        combined_audio.export(output_path, format="wav")
        logger.info(f"Podcast audio exported to {output_path}")
        return output_path

    def synthesize_audio_from_text(self, text: str, output_path: str) -> str:
        """Decide method automatically based on script type."""
        is_podcast = re.search(r"^\s*Speaker\d*:", text, re.MULTILINE)
        if is_podcast:
            logger.info("Detected podcast script, using multi-voice synthesis")
            return self.synthesize_podcast_audio(text, output_path)
        else:
            logger.info("Detected long-form text, using single-voice synthesis")
            return self.synthesize_long_audio(text, output_path)

    def download_audio_from_gcs(self, gcs_uri: str, local_path: str):
        """Download file from GCS."""
        try:
            bucket_name = gcs_uri.split('/')[2]
            blob_name = '/'.join(gcs_uri.split('/')[3:])
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded GCS audio {gcs_uri} to {local_path}")
        except Exception as e:
            logger.error(f"GCS download error: {e}")
            raise

    def delete_gcs_folder(self, gcs_uri: str):
        """Delete all files in a GCS 'folder'."""
        try:
            path_parts = gcs_uri.replace("gs://", "").split("/", 1)
            bucket_name, prefix = path_parts[0], path_parts[1]
            bucket = self.storage_client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                blob.delete()
                logger.info(f"Deleted {blob.name} from GCS")
        except Exception as e:
            logger.error(f"GCS delete folder error: {e}")
            raise
