import os
import time
from datetime import datetime, timedelta
import pytz
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
import threading
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import json
from cloudinary_handler import *
from excelhandler import ExcelHandler
from logger_config import setup_logger

# --- Configuration ---
logger = setup_logger(__name__)

# Load environment variables
dotenv_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

IG_ACCESS_TOKEN = os.getenv("IG_ACCESS_TOKEN")  # Must be EAASj7KCB...
INSTAGRAM_ACCOUNT_ID = os.getenv("INSTAGRAM_ACCOUNT_ID")  # 17841476203857454

# Lock for file access
file_lock = threading.Lock()

# --- Constants for Directory Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
BOOK_DIR = BASE_DIR / "audiobook_content"
DOCS_DIR = BOOK_DIR / "docs"
IMG_DIR = BOOK_DIR / "images"
OUTPUT_DIR = BOOK_DIR / "output"
OUTPUT_SHORTS_DIR = OUTPUT_DIR / "shorts"
EXCEL_FILE_PATH = DOCS_DIR / "video_content.xlsx"


class InstagramAPI:
    """A class to handle interactions with the Instagram Graph API."""
    def __init__(self):
        self.access_token = IG_ACCESS_TOKEN
        self.account_id = INSTAGRAM_ACCOUNT_ID
        self.api_version = "v22.0"  # Updated to v22.0
        self.base_graph_url = f"https://graph.facebook.com/{self.api_version}"

        if not self.access_token or not self.account_id:
            raise ValueError("Access token or account ID is missing.")

        logger.info("✅ Instagram API configured...")
        logger.info(f"Account ID: {self.account_id}")

    def _make_api_request(self, method: str, url: str, data: dict = None, params: dict = None, timeout: int = 90) -> requests.Response:
        """
        Helper method to make API requests with robust retry logic.
        Supports sending data in both the request body (json) and query string (params).
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)

        headers = {}

        try:
            logger.debug(f"Requesting: {method} {url}")
            logger.debug(f"Data: {json.dumps(data, indent=2) if data else 'No data'}")
            logger.debug(f"Params: {params}")

            response = session.request(
                method,
                url,
                data=data,
                params=params,
                headers=headers,
                timeout=timeout
            )

            logger.debug(f"Response: {response.status_code} - {response.text}")

            if response.status_code >= 400:
                try:
                    error_data = response.json()
                    logger.error(f"API Error: {error_data.get('error', {}).get('message', 'Unknown error')}")
                    logger.error(f"Error Code: {error_data.get('error', {}).get('code', 'Unknown')}")
                    logger.error(f"Error Type: {error_data.get('error', {}).get('type', 'Unknown')}")
                except json.JSONDecodeError:
                    logger.error(f"Error Response: {response.text}")
                raise requests.HTTPError(response.text, response=response)

            response.raise_for_status() # Raise an exception for bad status codes
            return response
        except requests.exceptions.RequestException as e:
            logger.exception(f"API request failed: {e}")
            raise

    def create_scheduled_container(self, caption: str, video_url: str, publish_time: int) -> Optional[str]:
        """Creates a media container for a scheduled Reel and publishes it."""
        # Step 1: Create media container
        url = f"{self.base_graph_url}/{self.account_id}/media"
        params = {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "access_token": self.access_token
        }
        try:
            logger.info(f"Creating container for video: {video_url}")
            response = self._make_api_request("POST", url, params=params)
            container_data = response.json()
            if "id" not in container_data:
                logger.error(f"Error creating container: {container_data}")
                return None

            container_id = container_data["id"]
            logger.info(f"Container created: {container_id}")

            # Step 2: Poll container status
            status_url = f"https://graph.facebook.com/{self.api_version}/{container_id}"
            status_params = {"fields": "status_code,id", "access_token": self.access_token}
            for _ in range(60):  # Max 10 minutes
                status_response = self._make_api_request("GET", status_url, params=status_params)
                status_data = status_response.json()
                logger.info(f"Status: {status_data}")
                if status_data.get("status_code") == "FINISHED":
                    break
                elif status_data.get("status_code") == "ERROR":
                    logger.error(f"Container error: {status_data}")
                    return None
                time.sleep(10)

            if status_data.get("status_code") != "FINISHED":
                logger.error(f"Container did not reach FINISHED status: {status_data}")
                return None

            # Step 3: Publish container
            publish_url = f"{self.base_graph_url}/{self.account_id}/media_publish"
            publish_params = {
                "creation_id": container_id,
                "access_token": self.access_token,
                "publish_time": publish_time
            }
            publish_response = self._make_api_request("POST", publish_url, params=publish_params)
            publish_data = publish_response.json()
            if "id" in publish_data:
                logger.info(f"✅ Reel scheduled successfully: {publish_data['id']}")
                return publish_data["id"]
            else:
                logger.error(f"Error publishing: {publish_data}")
                return None

        except Exception as e:
            logger.error(f"Error creating container: {e}")
            return None

    def publish_immediately(self, video_url: str) -> bool:
        """Publishes a video immediately (fallback method)."""
        logger.info(f"Publishing immediately: {video_url}")
        params = {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": self.caption,
            "access_token": self.access_token
        }
        try:
            url = f"{self.base_graph_url}/{self.account_id}/media"
            response = self._make_api_request("POST", url, params=params)
            container_data = response.json()
            if "id" not in container_data:
                logger.error(f"Error creating container: {container_data}")
                return False

            container_id = container_data["id"]
            logger.info(f"Container created: {container_id}")

            status_url = f"https://graph.facebook.com/{self.api_version}/{container_id}"
            status_params = {"fields": "status_code,id", "access_token": self.access_token}
            for _ in range(60):
                status_response = self._make_api_request("GET", status_url, params=status_params)
                status_data = status_response.json()
                logger.info(f"Status: {status_data}")
                if status_data.get("status_code") == "FINISHED":
                    break
                elif status_data.get("status_code") == "ERROR":
                    logger.error(f"Container error: {status_data}")
                    return False
                time.sleep(10)

            if status_data.get("status_code") != "FINISHED":
                logger.error(f"Container did not reach FINISHED status: {status_data}")
                return False

            publish_url = f"{self.base_graph_url}/{self.account_id}/media_publish"
            publish_params = {
                "creation_id": container_id,
                "access_token": self.access_token
            }
            publish_response = self._make_api_request("POST", publish_url, params=publish_params)
            publish_data = publish_response.json()
            if "id" in publish_data:
                logger.info(f"✅ Reel published immediately: {publish_data['id']}")
                return True
            else:
                logger.error(f"Error publishing: {publish_data}")
                return False
        except Exception as e:
            logger.error(f"Error publishing immediately: {e}")
            return False

def debug_direct_api_call(video_url: str, caption: str, publish_time: int):
    """Function to test the API directly without classes."""
    url = f"https://graph.facebook.com/v22.0/{INSTAGRAM_ACCOUNT_ID}/media"
    params = {
        "media_type": "REELS",
        "video_url": video_url,
        "caption": caption,
        "access_token": IG_ACCESS_TOKEN
    }
    try:
        logger.info(f"Testing API directly with video: {video_url}")
        response = requests.post(url, params=params)
        logger.info(f"Status: {response.status_code}, Response: {response.text}")
        create_data = response.json()

        if "id" not in create_data:
            logger.error(f"Error creating container: {create_data}")
            return False

        container_id = create_data["id"]
        logger.info(f"Container created: {container_id}")

        status_url = f"https://graph.facebook.com/v22.0/{container_id}"
        status_params = {"fields": "status_code,id", "access_token": IG_ACCESS_TOKEN}
        for _ in range(60):
            status_response = requests.get(status_url, params=status_params)
            status_data = status_response.json()
            logger.info(f"Status: {status_data}")
            if status_data.get("status_code") == "FINISHED":
                break
            elif status_data.get("status_code") == "ERROR":
                logger.error(f"Container error: {status_data}")
                return False
            time.sleep(10)

        if status_data.get("status_code") != "FINISHED":
            logger.error(f"Container did not reach FINISHED status: {status_data}")
            return False

        publish_url = f"https://graph.facebook.com/v22.0/{INSTAGRAM_ACCOUNT_ID}/media_publish"
        publish_params = {
            "creation_id": container_id,
            "access_token": IG_ACCESS_TOKEN,
            "publish_time": publish_time
        }
        publish_response = requests.post(publish_url, params=publish_params)
        publish_data = publish_response.json()
        logger.info(f"Publish response: {publish_data}")
        if "id" in publish_data:
            logger.info(f"✅ Reel scheduled: {publish_data['id']}")
            return True
        else:
            logger.error(f"Error publishing: {publish_data}")
            return False
    except Exception as e:
        logger.error(f"Direct API test failed: {e}")
        return False


def bulk_upload_reels():
    from excelhandler import ExcelHandler

    logger.info("Starting bulk reel upload...")
    excel_handler = ExcelHandler(str(EXCEL_FILE_PATH))
    data = excel_handler.read_data(sheet_name="Shorts")

    if not data:
        logger.error("No data found in the Excel file.")
        return

    cloudinary_uploader = CloudinaryUploader()
    successful_uploads = 0
    failed_uploads = 0

    # Configure a specific time for scheduling (in UTC)
    local_tz = pytz.timezone('America/Mexico_City')
    specific_date = datetime(2025, 10, 1, 9, 0, 0)  # Matches SCHEDULED_PUBLISH_TIME
    publish_time = local_tz.localize(specific_date).astimezone(pytz.UTC)
    scheduled_ts = int(publish_time.timestamp())

    logger.info(f"Scheduled time: {publish_time} (UNIX: {scheduled_ts})")

    # # Test with a sample video first
    # test_result = debug_direct_api_call(
    #     video_url="https://res.cloudinary.com/doc1jntmu/video/upload/v1754514309/reel_test.mp4",
    #     # video_url="https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4",  # Uncomment to test with alternative video
    #     caption="Direct API programming test",
    #     publish_time=scheduled_ts
    # )
    #
    # if test_result:
    #     logger.info("✅ Direct API test successful.")
    # else:
    #     logger.error("❌ Direct API test failed.")

    for index, info in enumerate(data, 1):

        if info.get('Enabled') in (0, False):
            logger.info(f"Excel Enabled: {info.get('Enabled')}")
            continue

        final_video_path = info.get('Output path')
        caption = info.get('Copywriting', 'Test Reel')

        if not final_video_path or not os.path.exists(final_video_path):
            logger.error(f"Video not found: {final_video_path}")
            failed_uploads += 1
            continue

        try:
            # Upload to Cloudinary
            public_id = f"reel_{index}_{int(time.time())}"
            video_url = cloudinary_uploader.upload_video(final_video_path, public_id=public_id)

            if not video_url:
                logger.error("Cloudinary upload failed.")
                failed_uploads += 1
                continue

            # For scheduled publishing
            api = InstagramAPI()
            container_id = api.create_scheduled_container(caption=caption, video_url=video_url, publish_time=scheduled_ts)

            if container_id:
                successful_uploads += 1
                logger.info(f"✅ Video {index} scheduled successfully!")
                print(f"✅ Video {index} scheduled for {publish_time.astimezone(local_tz)}")
            else:
                # Fallback: Publish immediately
                if api.publish_immediately(video_url):
                    successful_uploads += 1
                    logger.warning(f"⚠️ Immediate publication for video {index}.")
                    print(f"⚠️ Video {index} published immediately.")
                else:
                    failed_uploads += 1
                    logger.error(f"❌ Complete failure for video {index}.")

        except Exception as e:
            logger.exception(f"Error processing video {index}: {e}")
            failed_uploads += 1
            print(f"❌ Video {index} failed: {str(e)}")

        # Wait between requests
        time.sleep(15)

    logger.info(f"Summary: Successful: {successful_uploads}, Failed: {failed_uploads}")
    print(f"\n✅ Bulk upload complete: {successful_uploads} successful, {failed_uploads} failed.")


if __name__ == "__main__":
    try:
        bulk_upload_reels()
    except Exception as e:
        logger.exception("Critical error in bulk upload.")
        print(f"❌ Critical error: {str(e)}")