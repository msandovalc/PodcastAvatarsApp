import os
import logging
import io
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import texttospeech_v1
from google.cloud.texttospeech_v1 import types
from google.cloud import storage
from google.api_core import exceptions
from logger_config import setup_logger
from pydub import AudioSegment, silence
from pydub.silence import split_on_silence, detect_silence
from typing import List, Dict, Any
from utils import trim_silence, split_long_sentences, split_text_into_chunks


# --- Configuration ---
logger = setup_logger(__name__)

dotenv_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)


# --- Constants for Directory Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
PODCAST_DIR = BASE_DIR / "podcast_content"
AUDIO_DIR = PODCAST_DIR / "audio"

class GCPTextToSpeechManager:
    """
    Manages interactions with Google Cloud Text-to-Speech and Google Cloud Storage.
    """
    MAX_RETRIES = 5  # Maximum number of times to retry API calls
    RETRY_DELAY_SECONDS = 2  # Initial delay in seconds before the first retry
    
    # Pre-defined voices for podcast speakers
    # Feel free to change these voices to suit your needs
    VOICE_MAPPING = {
        "Speaker 1": "es-US-Chirp3-HD-Laomedeia",  # Female Voice
        "Speaker 2": "es-US-Chirp3-HD-Algenib",  # Male Voice
        "Speaker": "es-US-Chirp3-HD-Laomedeia"   # Default Voice (e.g., for monologues)
    }

    def __init__(self, bucket_name: str, voice_name: str = "es-US-Chirp3-HD-Algenib"):
        """
        Initializes the Google Cloud Text-to-Speech and Storage clients.
        
        Args:
            bucket_name (str): The name of the GCS bucket.
            voice_name (str): The name of the voice for audio synthesis.
        """
        google_gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
        gcp_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if not gcp_credentials_path:
            logger.error("GCP credentials path not configured. "
                         "Set GOOGLE_APPLICATION_CREDENTIALS in .env")
            raise FileNotFoundError("GCP credentials path not configured.")

        # Convert relative path to absolute if necessary
        credentials_path = Path(gcp_credentials_path)
        if not credentials_path.is_absolute():
            # Dynamically resolve project root (assumes 'ebookContentApp' is your project folder name)
            project_root = Path(__file__).resolve()
            while project_root.name != "PodcastAvatarsApp" and project_root.parent != project_root:
                project_root = project_root.parent
            credentials_path = project_root / credentials_path

        # Validate that the credentials file exists
        if not credentials_path.exists():
            logger.error(f"GCP credentials file not found: {credentials_path}"
                         "Set GOOGLE_APPLICATION_CREDENTIALS in .env and ensure the file exists.")
            raise FileNotFoundError(f"GCP credentials file not found: {credentials_path}")

        # Set the environment variable so Google SDK picks it up
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
        logger.info(f"✅ Resolved credentials path: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")

        self.tts_client = texttospeech_v1.TextToSpeechClient()
        self.storage_client = storage.Client()
        self.bucket_name = google_gcs_bucket_name

        if not self.bucket_name:
            logger.error("GCS bucket name not configured. Ensure GCS_BUCKET_NAME is in .env")
            raise ValueError("GCS bucket name not found.")

        logger.info(f"GCP Text-to-Speech and Storage clients initialized. Bucket: {self.bucket_name}")

    def synthesize_long_audio(self, text_content: str, local_audio_path: str,
                              voice_name: str = "es-US-Chirp3-HD-Algenib",
                              audio_encoding: types.AudioEncoding = types.AudioEncoding.MP3):
        """
        Synthesizes long-form audio by splitting the text into byte-safe chunks
        and combining the resulting audio.

        Args:
            text_content (str): The input text to be synthesized.
            local_audio_path (str): The local file path to save the combined audio.
            voice_name (str): The name of the voice to use for synthesis.
            audio_encoding (types.AudioEncoding): The audio encoding for the output.

        Returns:
            str: The local file path of the generated audio file.
        """
        try:

            # es-US-Chirp3-HD-Laomedeia, es-US-Chirp3-HD-Gacrux --> Female voice
            # es-US-Chirp3-HD-Algenib --> Male voice
            # Step 1: Split the text into safe chunks based on byte size.
            # This function is designed to avoid the API's sentence-length limit.
            chunks = split_text_into_chunks(text_content, max_bytes=4500)

            logger.info(f"DEBUG: Generated {len(chunks)} chunks for TTS")
            for i, chunk in enumerate(chunks, start=1):
                logger.info(f"DEBUG: Chunk {i} length={len(chunk.encode('utf-8'))} bytes, start='{chunk[:100]}...'")

            # Step 2: Prepare voice and audio config
            language_code_from_voice = '-'.join(voice_name.split('-')[:2])
            voice = types.VoiceSelectionParams(
                language_code=language_code_from_voice,
                name=voice_name
            )
            audio_config = types.AudioConfig(
                audio_encoding=audio_encoding.LINEAR16,
                speaking_rate=1.0,
                pitch=0.0
            )

            # Step 3: Synthesize each chunk
            audio_contents: List[bytes] = []
            for i, chunk_text in enumerate(chunks, start=1):
                logger.info(f"Processing chunk {i}/{len(chunks)} ({len(chunk_text.encode('utf-8'))} bytes)")

                # input_text = types.SynthesisInput(ssml=ssml_chunk)
                input_text = types.SynthesisInput(text=chunk_text)

                # --- START: Retry mechanism for API call ---
                for attempt in range(self.MAX_RETRIES):
                    try:
                        response = self.tts_client.synthesize_speech(
                            input=input_text,
                            voice=voice,
                            audio_config=audio_config
                        )
                        # If the call succeeds, break the retry loop
                        break
                    except exceptions.InternalServerError as e:
                        if attempt < self.MAX_RETRIES - 1:
                            retry_delay = self.RETRY_DELAY_SECONDS * (2 ** attempt)
                            logger.warning(
                                f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay:.2f} seconds...")
                            time.sleep(retry_delay)
                        else:
                            # Re-raise the exception after all retries are exhausted
                            logger.error(f"Final attempt failed after {self.MAX_RETRIES} retries.")
                            raise

                audio_contents.append(response.audio_content)
                logger.info(f"  Chunk {i} audio content length: {len(response.audio_content)} bytes")

            # Step 4: Combine audio chunks
            combined = AudioSegment.empty()
            for content in audio_contents:
                if not content:
                    logger.warning("Skipping an empty audio chunk.")
                    continue
                # audio = AudioSegment.from_mp3(io.BytesIO(content))
                # Use from_file and specify the format as "raw" for LINEAR16
                audio = AudioSegment.from_file(io.BytesIO(content), format="raw", frame_rate=24000, channels=1,
                                               sample_width=2)
                combined += audio

            # Add this line to increase the volume of the entire combined audio
            # The value (e.g., 5.0) is in decibels (dB). A positive value increases volume.
            # You may need to experiment to find the perfect value.
            # boost_in_db = 5.0
            # combined = combined.apply_gain(boost_in_db)

            # logger.info(f"All chunks combined. Total audio length: {len(combined)} ms.")
            # combined.export(local_audio_path, format="mp3")
            # logger.info(f"Combined audio exported locally: {local_audio_path}")

            logger.info(f"All chunks combined. Total audio length: {len(combined)} ms.")
            # Export the combined audio to a .wav file
            combined.export(local_audio_path, format="wav")
            logger.info(f"Combined audio exported locally: {local_audio_path}")

            return local_audio_path

        except Exception as e:
            logger.error(f"Error during audio synthesis: {e}")
            raise

    def _parse_podcast_script(self, text_script: str) -> List[Dict[str, str]]:
        """
        Parses a text script with speaker labels into a list of dictionaries.

        Args:
            text_script (str): The raw text script with speaker labels (e.g., 'Speaker1: ...').

        Returns:
            List[Dict[str, str]]: A list of dictionaries, each with 'text' and 'voice_name'.
        """
        conversation_list = []
        # Split the text by lines and clean up empty lines
        lines = [line.strip() for line in text_script.strip().split('\n') if line.strip()]

        current_speaker = None
        current_text = ""

        # Regular expression to find speaker labels
        # Matches 'Speaker' followed by any characters until a colon, then a space
        # speaker_regex = re.compile(r"^(Speaker\d*):(.*)")
        speaker_regex = re.compile(r"^\s*(Speaker\s*\d+):\s*(.*)")

        for line in lines:
            clean_line = line.strip()
            match = speaker_regex.match(clean_line)
            # logger.info(f"Checking line after clean: {repr(clean_line)} -> match: {match}")
            if match:
                # If a new speaker is found, save the previous segment
                if current_speaker is not None:
                    voice_name = self.VOICE_MAPPING.get(current_speaker, "es-US-Chirp3-HD-Algenib")
                    conversation_list.append({
                        "speaker": current_speaker,
                        "text": current_text.strip(),
                        "voice_name": voice_name
                    })

                # Start a new segment for the new speaker
                current_speaker = match.group(1).strip()
                current_text = match.group(2).strip()
            elif current_speaker:
                # Continue adding text to the current speaker's segment
                current_text += " " + line.strip()

        # Add the last segment after the loop finishes
        if current_speaker is not None and current_text:
            voice_name = self.VOICE_MAPPING.get(current_speaker, "es-US-Chirp3-HD-Algenib")
            conversation_list.append({
                "speaker": current_speaker,
                "text": current_text.strip(),
                "voice_name": voice_name
            })

        logger.info(f"Print conversation_list: {conversation_list}")

        return conversation_list

    def _trim_audio_silence(self, audio_segment: AudioSegment) -> AudioSegment:
        """
        Trims leading and trailing silence from an AudioSegment using a more reliable method.
        
        This function is now more robust against edge cases where the audio does not
        start or end with silence.
        
        Args:
            audio_segment (AudioSegment): The audio segment to trim.

        Returns:
            AudioSegment: The trimmed audio segment.
        """
        # Split audio on silence. This returns a list of non-silent audio segments.
        # The min_silence_len and silence_thresh parameters can be adjusted.
        non_silent_chunks = split_on_silence(
            audio_segment,
            min_silence_len=500,  # A 500ms silence duration to consider for a split
            silence_thresh=-40  # A -40 dBFS threshold to consider something as silence
        )

        if not non_silent_chunks:
            # If the entire audio is silent, return an empty segment or the original
            return AudioSegment.silent(duration=0)

        # Concatenate the non-silent chunks back together
        # The first chunk is the start, then append the rest
        trimmed_audio = non_silent_chunks[0]
        for chunk in non_silent_chunks[1:]:
            trimmed_audio += chunk

        return trimmed_audio

    def synthesize_podcast_segments(self, conversation_script: str, output_dir: str,
                                    audio_encoding: types.AudioEncoding = types.AudioEncoding.MP3,
                                    speaking_rate: float = 1.0, pitch: float = 0.0) -> list[dict]:
        """
        Generates separate audio files per podcast segment with speaker info.
        Returns a list of dictionaries containing speaker and path for MuseTalk integration.

        Returns:
            List[Dict[str, str]]: Each dict contains 'speaker' and 'path' keys.
        """
        try:
            conversation = self._parse_podcast_script(conversation_script)
            os.makedirs(output_dir, exist_ok=True)
            segment_paths = []

            logger.info(f"Conversation parsed with {len(conversation)} segments.")

            for i, segment in enumerate(conversation):
                text = segment.get("text")
                voice_name = segment.get("voice_name")
                speaker = segment.get("speaker")  # <- Assuming your updated _parse_podcast_script includes 'speaker'

                if not text or not voice_name or not speaker:
                    logger.warning(f"Skipping segment {i + 1} due to missing text, voice_name, or speaker.")
                    continue

                logger.info(f"Processing segment {i + 1} | Speaker: {speaker} | Voice: {voice_name}")

                language_code = '-'.join(voice_name.split('-')[:2])
                voice = types.VoiceSelectionParams(
                    language_code=language_code,
                    name=voice_name
                )
                audio_config = types.AudioConfig(
                    audio_encoding=types.AudioEncoding.LINEAR16,
                    speaking_rate=speaking_rate,
                    pitch=pitch
                )
                input_text = types.SynthesisInput(text=text)

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
                            retry_delay = self.RETRY_DELAY_SECONDS * (2 ** attempt)
                            logger.warning(
                                f"Attempt {attempt + 1} failed for segment {i + 1}: {e}. Retrying in {retry_delay:.2f}s...")
                            time.sleep(retry_delay)
                        else:
                            logger.error(f"Final attempt failed for segment {i + 1}.")
                            raise

                audio = AudioSegment.from_file(io.BytesIO(response.audio_content),
                                               format="raw", frame_rate=24000, channels=1, sample_width=2)
                trimmed_audio = self._trim_audio_silence(audio)

                segment_path = os.path.join(output_dir, f"{speaker}_segment_{i + 1}.wav")
                trimmed_audio.export(segment_path, format="wav")
                logger.info(f"Segment {i + 1} exported: {segment_path}")

                # Append dict with speaker info and path
                segment_paths.append({
                    "speaker": speaker,
                    "path": segment_path
                })

            return segment_paths

        except Exception as e:
            logger.error(f"Error during podcast segment synthesis: {e}")
            raise

    def synthesize_podcast_audio(self, conversation_script: str, local_audio_path: str,
                                 audio_encoding: types.AudioEncoding = types.AudioEncoding.MP3,
                                 speaking_rate: float = 1.0, pitch: float = 0.0):
        """
        Synthesizes a multi-voice audio from a structured conversation, ideal for podcasts.

        This method now accepts a raw text script and parses it internally.

        Args:
            conversation_script (str): The raw text script of the podcast conversation.
            local_audio_path (str): The local file path to save the combined audio.
            audio_encoding (types.AudioEncoding): The audio encoding for the output.
            speaking_rate (float): Overall speaking rate for the audio.
            pitch (float): Overall pitch for the audio.

        Returns:
            str: The local file path of the generated audio file.
        """
        try:
            # Parse the raw script into the required list format
            conversation = self._parse_podcast_script(conversation_script)

            combined_audio = AudioSegment.empty()

            for i, segment in enumerate(conversation):
                text = segment.get("text")
                voice_name = segment.get("voice_name")

                if not text or not voice_name:
                    logger.warning(f"Skipping segment {i+1} due to missing text or voice_name.")
                    continue

                logger.info(f"Processing segment {i+1} with voice '{voice_name}'.")

                language_code = '-'.join(voice_name.split('-')[:2])
                voice = types.VoiceSelectionParams(
                    language_code=language_code,
                    name=voice_name
                )
                audio_config = types.AudioConfig(
                    audio_encoding=types.AudioEncoding.LINEAR16,
                    speaking_rate=speaking_rate,
                    pitch=pitch
                )
                input_text = types.SynthesisInput(text=text)

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
                            retry_delay = self.RETRY_DELAY_SECONDS * (2 ** attempt)
                            logger.warning(
                                f"Attempt {attempt + 1} failed for segment {i+1}: {e}. Retrying in {retry_delay:.2f} seconds...")
                            time.sleep(retry_delay)
                        else:
                            logger.error(f"Final attempt failed for segment {i+1} after {self.MAX_RETRIES} retries.")
                            raise

                audio = AudioSegment.from_file(io.BytesIO(response.audio_content), format="raw", frame_rate=24000, channels=1, sample_width=2)

                # Trim silence from the start of the audio segment
                trimmed_audio = self._trim_audio_silence(audio)

                # Use a small crossfade to ensure a smooth transition
                crossfade_duration = 50 # milliseconds
                fade_in_duration = 10 # milliseconds
                # FIX: Handle the first segment with a fade-in, and all others with a crossfade
                if i == 0:
                    combined_audio = trimmed_audio.fade_in(fade_in_duration)
                else:
                    combined_audio = combined_audio.append(trimmed_audio, crossfade=crossfade_duration)
                logger.info(f"Segment {i+1} appended. Total duration: {combined_audio.duration_seconds:.2f}s")
            
            logger.info(f"All segments combined. Total audio length: {len(combined_audio)} ms.")
            combined_audio.export(local_audio_path, format="wav")
            logger.info(f"Combined audio exported locally: {local_audio_path}")

            return local_audio_path

        except Exception as e:
            logger.error(f"Error during podcast audio synthesis: {e}")
            raise

    def synthesize_audio_from_text(self, text_content: str, local_audio_path: str, individual_audios: bool=False) -> list[dict]:
        """
        Analyzes the text format and synthesizes audio using the appropriate method.

        Args:
            text_content (str): The raw text to be synthesized.
            local_audio_path (str): The local file path to save the generated audio.

        Returns:
            str: The local file path of the generated audio file.
        """

        segment_paths = []
        text_content = text_content.strip()

        # Check if the text contains a speaker label pattern
        # is_podcast_script = re.search(r"^\s*Speaker\d*:", text_content, re.MULTILINE)
        is_podcast_script = re.search(r"^\s*Speaker\s*\d+:", text_content, re.MULTILINE)

        if is_podcast_script:
            if individual_audios:
                logger.info("Detected podcast script format. Using multi-voice synthesis with individual audios.")
                segment_paths = self.synthesize_podcast_segments(text_content, local_audio_path)
            else:
                logger.info("Detected podcast script format. Using multi-voice synthesis with one audio.")
                # segment_paths = self.synthesize_podcast_audio(text_content, local_audio_path)
        else:
            logger.info("Detected long-form book format. Using single-voice synthesis.")
            # segment_paths = self.synthesize_long_audio(text_content, local_audio_path)

        return segment_paths

    def download_audio_from_gcs(self, gcs_uri: str, local_path: str):
        """Downloads a blob from GCS to a local path."""
        try:
            bucket_name = gcs_uri.split('/')[2]
            blob_name = '/'.join(gcs_uri.split('/')[3:])
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            logger.info(f"Audio downloaded from GCS: {gcs_uri} to {local_path}")
        except Exception as e:
            logger.error(f"Error downloading audio from GCS: {e}")
            raise

    def delete_gcs_folder(self, gcs_uri: str):
        """Deletes all files within a GCS 'folder'."""
        try:
            # We need the bucket name and the folder prefix.
            path_parts = gcs_uri.replace("gs://", "").split("/", 1)
            bucket_name = path_parts[0]
            # The prefix is everything after the bucket name
            prefix = path_parts[1]

            bucket = self.storage_client.bucket(bucket_name)

            # Use a `list_blobs` with the prefix to get all files in the "folder"
            # It's better to explicitly use `prefix=`.
            blobs = bucket.list_blobs(prefix=prefix)

            for blob in blobs:
                blob.delete()
                logger.info(f"File deleted from GCS: {blob.name}")
        except Exception as e:
            logger.error(f"Error deleting files from GCS: {e}")
            raise


if __name__ == "__main__":

    # # IMPORTANTE: Cambia "your-gcs-bucket-name" por el nombre real de tu bucket de GCS
    # tts_manager = GCPTextToSpeechManager(bucket_name=os.getenv("GCS_BUCKET_NAME"))
    #
    # # --- EJEMPLO 1: Usando formato de podcast ---
    # print("\nStarting podcast audio generation from script example...")
    #
    # podcast_script = """
    # CAPÍTULO 1
    # Speaker1: ¡Bienvenidos a "Principios y Hábitos para una Vida Efectiva"! Soy Carla, y me fascina la arquitectura de una vida con sentido, una vida construida sobre la base sólida de los principios. Hoy, quiero que pensemos en Stephen Covey como nuestro arquitecto, la mente maestra detrás de los planos. Él nos enseña que una vida efectiva no es un accidente, sino el resultado de vivir de adentro hacia afuera, de alinear nuestras acciones con verdades fundamentales que no cambian.
    # Speaker2: Y yo soy Diego, el que toma esos planos y los hace realidad. Mi enfoque está en la ciencia de los pequeños ladrillos, esos sistemas y hábitos que construyen lo que el arquitecto diseñó. Pienso en James Clear como nuestro ingeniero de hábitos. Su enfoque en la mejora del 1% cada día nos demuestra que la transformación no es un evento mágico, sino el efecto acumulado de innumerables decisiones pequeñas y consistentes.
    # Speaker1: Juntos, vamos a unir estos dos mundos, el del ser y el del hacer. ¿Alguna vez te has sentido como un barco a la deriva en un océano de tareas, redes sociales y distracciones? Eso es lo que pasa cuando solo hacemos, pero no sabemos por qué hacemos. Nos invita a la "ética de la personalidad". Nos venden atajos, trucos, hacks de productividad. "Usa esta app y serás exitoso". "Aprende este tip y conseguirás más en menos tiempo". No estoy diciendo que esas herramientas no sirvan, pero son la punta del iceberg. Covey nos lo advirtió hace décadas: la verdadera efectividad no viene de la imagen pública o de las técnicas superficiales. Él nos invita a un cambio "de adentro hacia afuera", a un proceso que comienza por reescribir nuestros propios guiones.
    # Speaker2: Tienes toda la razón, Carla. La gente se cansa porque no ve resultados. Empieza el año con una lista de propósitos, se siente motivada, pero a las tres semanas el gimnasio ya no es tan atractivo y la dieta se ha ido al traste. ¿Por qué? Porque se enfocaron en la meta, no en el sistema. Y esto es algo que Clear explica con una claridad asombrosa. Él nos dice que "los resultados duraderos no vienen de una sola acción masiva, sino de un cúmulo de pequeñas decisiones diarias". No es un sprint, es una maratón. Piénsalo así: si mejoraras solo un 1% cada día, al final del año serías 37 veces mejor. La consistencia, y no la intensidad, es lo que importa a largo plazo.
    # """
    # output_podcast_path = "podcast_episode.wav"

    try:
        # generated_audio_paths = tts_manager.synthesize_audio_from_text(podcast_script, output_podcast_path,True)
        # print(f"Podcast audio successfully generated and saved to: {generated_audio_paths}")

        audio_paths = [
            {"speaker": "Speaker1", "audio_path": str(AUDIO_DIR / "segment_1.wav")},
            {"speaker": "Speaker2", "audio_path": str(AUDIO_DIR / "segment_2.wav")},
            {"speaker": "Speaker1", "audio_path": str(AUDIO_DIR / "segment_3.wav")},
            {"speaker": "Speaker2", "audio_path": str(AUDIO_DIR / "segment_4.wav")},
        ]

    except Exception as e:
        print(f"Failed to generate podcast audio: {e}")


    # # --- EJEMPLO 2: Usando formato de libro ---
    # print("\nStarting long-form audio generation from book script example...")
    #
    # book_script = """
    # CAPÍTULO 1
    # El Derecho a Ser Rico
    # CUALQUIER COSA QUE PUEDA SER DICHA A FAVOR DE LA POBREZA, la verdad está en que no es posible vivir una vida realmente completa y exitosa a menos que uno sea rico. Nadie puede alcanzar su punto más alto en su talento o desarrollo del alma, a menos que esa persona tenga mucho dinero, porque para desplegar el alma y desarrollar el talento, se deben tener muchas cosas para usar, y no se pueden tener esas cosas a menos que se tenga el dinero para comprarlas.
    # Una persona desarrolla su mente, alma y cuerpo al hacer uso de las cosas, y la sociedad está tan organizada que el hombre debe tener dinero para convertirse en poseedor de las cosas. Así pues, la base de todo avance debe estar en la ciencia de volverse rico.
    # El objeto de toda la vida es el desarrollo y cada cosa viviente tiene el inalienable derecho a todo el desarrollo que sea capaz de alcanzar.
    # El derecho de una persona a la vida significa su derecho a tener el uso libre e ilimitado de todas las cosas que sean necesarias para su total desarrollo mental, espiritual y físico, o en otras palabras, su derecho a ser rico.
    # En este libro, no hablaré de riquezas en una forma figurativa. Ser realmente rico no significa estar satisfecho o contento con poco.
    # Nadie debe estar satisfecho con poco si se puede ser capaz de usar y disfrutar más. El propósito de la naturaleza es el avance y el desarrollo de la vida y todo el mundo debería tener todo lo que pueda contribuir al poder, elegancia, belleza y riquezas de la vida.
    # """
    # output_book_path = "book_chapter.wav"
    #
    # try:
    #     generated_audio_path = tts_manager.synthesize_audio_from_text(book_script, output_book_path)
    #     print(f"Book chapter audio successfully generated and saved to: {generated_audio_path}")
    # except Exception as e:
    #     print(f"Failed to generate book chapter audio: {e}")