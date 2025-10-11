import cloudinary
import cloudinary.uploader
from pathlib import Path
from dotenv import load_dotenv
import os
from logger_config import setup_logger

# --- Configuration ---
logger = setup_logger(__name__)

class CloudinaryUploader:
    def __init__(self, env_path: str = ".env"):
        """
        Initializes Cloudinary using environment variables from a .env file.
        :param env_path: Path to the .env file.
        """
        # Setup logging
        self.logger = logger

        try:
            # Load environment variables
            load_dotenv(dotenv_path=env_path)

            # Get credentials from environment
            cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
            api_key = os.getenv("CLOUDINARY_API_KEY")
            api_secret = os.getenv("CLOUDINARY_API_SECRET")

            # Validate configuration
            if not all([cloud_name, api_key, api_secret]):
                raise ValueError("Missing Cloudinary credentials in .env file.")

            # Configure Cloudinary
            cloudinary.config(
                cloud_name=cloud_name,
                api_key=api_key,
                api_secret=api_secret,
                secure=True
            )

            self.logger.info("✅ Cloudinary configured successfully from .env file.")

        except Exception as e:
            self.logger.error(f"❌ Failed to configure Cloudinary: {e}")
            raise

    def upload_video(self, local_video_path: str, public_id: str = "uploaded_reel") -> str:
        """
        Uploads a local video file to Cloudinary and returns its public URL.
        :param local_video_path: Full path to the local .mp4 video file.
        :param public_id: Public identifier name for the uploaded file (without extension).
        :return: Public URL of the uploaded video (str). Returns "" if failed.
        """
        try:
            video_file = Path(local_video_path)

            # Validate file existence
            if not video_file.exists():
                raise FileNotFoundError(f"Video file not found: {local_video_path}")

            self.logger.info(f"🚀 Uploading video to Cloudinary: {local_video_path}")

            # Upload the video in chunks (resumable)
            response = cloudinary.uploader.upload_large(
                str(video_file),
                resource_type="video",
                public_id=public_id,
                chunk_size=6000000,  # 6 MB per chunk
                eager=[{"format": "mp4"}],  # Ensure it's mp4 format
                overwrite=True
            )

            # Extract the secure URL from the response
            video_url = response.get("secure_url")

            if video_url:
                self.logger.info(f"✅ Video uploaded successfully. Public URL: {video_url}")
                print(f"\n✅ Video uploaded successfully. Public URL:\n{video_url}")
                return video_url
            else:
                raise Exception("Cloudinary response missing secure_url.")

        except Exception as e:
            self.logger.error(f"❌ Error during video upload: {e}")
            return ""
