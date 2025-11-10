import os
import sys
import time
import datetime
import pytz
import random
import httplib2
from pathlib import Path

# Import libraries for Google API interactions.
import googleapiclient.discovery
import googleapiclient.http
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.auth.exceptions import RefreshError
from gpt_langchain import ContentGenerator
from logger_config import setup_logger
from utils import is_kaggle, get_next_publish_datetime

# --- Configuration ---
logging = setup_logger(__name__)

# Explicitly tell the underlying HTTP transport library not to retry, since
# the application handles the retry logic itself.
httplib2.RETRIES = 1

# Maximum number of times to retry before giving up.
MAX_RETRIES = 10

# Always retry when these specific exceptions are raised.
RETRIABLE_EXCEPTIONS = (httplib2.HttpLib2Error, IOError, httplib2.ServerNotFoundError)

# Always retry when an apiclient.errors.HttpError with one of these status
# codes is raised.
RETRIABLE_STATUS_CODES = [500, 502, 503, 504]

# The CLIENT_SECRETS_FILE variable specifies the name of a file that contains
# the OAuth 2.0 information for this application.
CLIENT_SECRET_FILE = str((Path(__file__).resolve().parent.parent / "credentials" /
                           "client_secret_471046779868-8ocft10nuip214911m6cpul8no9urcrg.apps.googleusercontent.com.json"))

# This OAuth 2.0 access scope allows an application to upload files to the
# authenticated user's YouTube channel. The `youtube` scope is needed for
# creating and managing playlists.
SCOPES = ['https://www.googleapis.com/auth/youtube.upload',
          'https://www.googleapis.com/auth/youtube',
          'https://www.googleapis.com/auth/youtubepartner']
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# This variable defines a message to display if the CLIENT_SECRETS_FILE is
# missing.
MISSING_CLIENT_SECRETS_MESSAGE = f"""
WARNING: Please configure OAuth 2.0

To make this sample run you will need to populate the client_secrets.json file
found at:

{os.path.abspath(os.path.join(os.path.dirname(__file__), CLIENT_SECRET_FILE))}

with information from the API Console
https://console.cloud.google.com/

For more information about the client_secrets.json file format, please visit:
https://developers.google.com/api-client-library/python/guide/aaa_client_secrets
"""

# Valid privacy statuses for a YouTube video.
VALID_PRIVACY_STATUSES = ("public", "private", "unlisted")

CHANNELS = {
    "El Universo del Audiolibro": "credentials_audiolibro.json",
    "Panuel: La Voz del Audiolibro": "credentials_lavozdelaudiolibro.json",
    "Mente de Campeón: El Poder de Tu Pensamiento": "credentials_mente_de_campeon.json",
}

# --- Constants for Directory Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
BOOK_DIR = BASE_DIR / "podcast_content"
OUTPUT_DIR = BOOK_DIR / "output"
EXCEL_FILE_PATH = BOOK_DIR / "docs" / "video_podcast_content.xlsx"
IMG_DIR = BOOK_DIR / "images"
CREDENTIALS_DIR = BASE_DIR / "credentials"


class YouTubeUploader:

    def __init__(self, client_secret_file: str, channels: dict, credentials_dir: str = CREDENTIALS_DIR):
        """
        :param client_secret_file: Path to client_secret.json (from GCP app).
        :param channels: Dictionary mapping channel_name → credentials filename.
        :param credentials_dir: Directory where credentials are stored.
        """
        self.client_secret_file = Path(client_secret_file)
        self.channels = channels
        self.credentials_dir = Path(credentials_dir)
        self.youtube = None
        self.channelId = None

        if not self.client_secret_file.exists():
            raise FileNotFoundError(f"Client secrets file not found: {self.client_secret_file}")

        self.credentials_dir.mkdir(parents=True, exist_ok=True)


    def get_authenticated_service(self):
        """
        Handles the authentication flow using the modern google-auth libraries.

        This function checks for existing credentials in 'credentials.json', refreshes them
        if necessary, or initiates a new OAuth 2.0 flow to get them. This ensures the
        script can consistently access the user's YouTube account.

        - In local environment: uses InstalledAppFlow with browser.
        - In Kaggle: uses credentials.json previously obtained locally.

        Returns:
            any: An authenticated YouTube service object ready for API calls.
        """

        credentials = None
        channel_id = None

        if is_kaggle():
            # Path to the credentials.json uploaded to Kaggle
            creds_file_path = str(Path(__file__).resolve().parent.parent / "credentials" / "credentials.json")

            if not os.path.exists(creds_file_path):
                logging.error(
                    f"Credentials JSON not found at {creds_file_path}. Upload your local credentials.json to Kaggle."
                )
                raise FileNotFoundError("Please upload your local credentials.json to Kaggle.")

            # Load credentials from the JSON
            creds = Credentials.from_authorized_user_file(creds_file_path, SCOPES)
            logging.info("Using Kaggle credentials.json for authentication.")

            # Refresh the token if expired
            try:
                if creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                    logging.info("Token refreshed successfully")
            except RefreshError:
                logging.error("Refresh token invalid or expired. Please update the credentials.json with a valid token.")
                raise

            # Build YouTube API client
            youtube = googleapiclient.discovery.build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, credentials=creds)

            # Verify access to at least one channel
            try:
                channels_response = youtube.channels().list(mine=True, part="id,snippet").execute()
                if not channels_response.get('items'):
                    logging.error(
                        "No channels found. Kaggle credentials may be incorrect, token expired, or scopes insufficient."
                    )
                    raise Exception("No channels found. Cannot proceed with video upload.")
                else:
                    channel_id = channels_response['items'][0]['id']
                    logging.info(f"Channel ID detected: {channel_id}")
            except googleapiclient.errors.HttpError as e:
                logging.error(f"Error accessing YouTube API: {e}")
                raise Exception("Failed to verify channel access. Check your credentials and scopes.")

        else:
            # Local environment: normal OAuth flow with browser
            # Check if a saved credentials file exists.
            if os.path.exists('../credentials/credentials.json'):
                credentials = Credentials.from_authorized_user_file('../credentials/credentials.json', SCOPES)

            # If there are no (valid) credentials available, prompt the user to log in.
            if not credentials or not credentials.valid:
                if credentials and credentials.expired and credentials.refresh_token:
                    # If credentials exist but are expired, use the refresh token to get a new one.
                    credentials.refresh(Request())
                else:
                    # No credentials exist, so start a new OAuth flow.
                    # The flow will open a browser window for the user to authenticate.
                    flow = InstalledAppFlow.from_client_secrets_file(
                        CLIENT_SECRET_FILE, SCOPES)
                    credentials = flow.run_local_server(port=0)

                # Save the new credentials for future runs.
                with open('../credentials/credentials.json', 'w') as token_file:
                    token_file.write(credentials.to_json())

            # Build and return the authenticated YouTube service object.
            youtube = googleapiclient.discovery.build(
                YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, credentials=credentials)

            # Verify channel
            channels_response = youtube.channels().list(mine=True, part="id,snippet").execute()
            if not channels_response.get('items'):
                logging.error("No channels found. Check your local credentials and scopes.")
                raise Exception("No channels found. Cannot proceed with video upload.")
            else:
                channel_id = channels_response['items'][0]['id']
                logging.info(f"Channel ID detected: {channel_id}")

        return youtube, channel_id

    def get_authenticated_service_by_channel(self, channel_name: str):
        """
        Authenticate and return a YouTube API client + channel_id for the given channel.
        """
        if channel_name not in self.channels:
            logging.error(f"Channel '{channel_name}' not found in channels dictionary.")
            return None, None

        credentials_file = self.credentials_dir / self.channels[channel_name]
        credentials = None

        # Load saved credentials
        if os.path.exists(credentials_file):
            credentials = Credentials.from_authorized_user_file(credentials_file, SCOPES)

        # Refresh or run OAuth flow if necessary
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                try:
                    credentials.refresh(Request())
                    logging.info(f"Refreshed credentials for {channel_name}")
                except Exception as e:
                    logging.error(f"Error refreshing credentials for {channel_name}: {e}")
                    credentials = None

            if not credentials or not credentials.valid:
                logging.info(f"Starting OAuth flow for {channel_name}...")
                flow = InstalledAppFlow.from_client_secrets_file(str(self.client_secret_file), SCOPES)
                credentials = flow.run_local_server(port=0)

            # Save credentials
            with open(credentials_file, "w") as token:
                token.write(credentials.to_json())
                logging.info(f"Saved credentials for {channel_name} → {credentials_file}")

        try:

            # Build and return the authenticated YouTube service object.
            youtube = googleapiclient.discovery.build(
                YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, credentials=credentials)

            # Verify channel
            channels_response = youtube.channels().list(mine=True, part="id,snippet").execute()
            if not channels_response.get('items'):
                logging.error("No channels found. Check your local credentials and scopes.")
                raise Exception("No channels found. Cannot proceed with video upload.")
            else:
                channel_id = channels_response['items'][0]['id']
                logging.info(f"Channel ID detected: {channel_id}")

            self.youtube = youtube
            self.channelId = channel_id

            logging.info(f"Authenticated to channel '{channel_name}' ({channel_id})")

            return youtube, channel_id
        except Exception as e:
            logging.error(f"Failed to authenticate {channel_name}: {e}")
            return None, None


    def initialize_upload(self, youtube: any, options: dict, publish_at: datetime.datetime = None, is_premiere: bool = False):
        """
        Initializes a video upload request to YouTube.

        This function prepares the video's metadata (title, description, tags, etc.)
        and creates a `videos.insert` request, which can then be executed
        to upload the video file.

        Args:
            youtube (any): The authenticated YouTube service object.
            options (dict): A dictionary containing video metadata, including:
                'title' (str): The video's title.
                'description' (str): The video's description.
                'keywords' (str): A comma-separated string of video tags.
                'category' (str): The YouTube category ID for the video.
                'privacyStatus' (str): The video's privacy status ('public', 'private', or 'unlisted').
                'file' (str): The local path to the video file to upload.
            publish_at (datetime.datetime, optional): The scheduled publish time in UTC.
                                                     Defaults to None.
            is_premiere (bool, optional): Whether to schedule the video as a Premiere.
                                          Defaults to False.

        Returns:
            googleapiclient.http.MediaFileUpload: The resumable upload request object.
        """
        tags = None
        if options.get('keywords'):
            tags = options['keywords'].split(",")

        # Define the body of the API request with snippet and status information.
        body = {
            'snippet': {
                'title': options.get('title'),
                'description': options.get('description'),
                'tags': tags,
                'categoryId': options.get('category'),
                'liveBroadcastContent': 'upcoming'
            },
            'status': {
                'privacyStatus': options.get('privacyStatus'),
                'madeForKids': False,  # Declares that the video is not made for kids.
                'selfDeclaredMadeForKids': False
            }
        }

        # Add the scheduled publish time if a 'publishAt' date is provided.
        if publish_at:
            # Format the time in RFC 3339 format with 'Z' for UTC.
            publish_at_utc = publish_at.astimezone(pytz.utc)
            body['status']['publishAt'] = publish_at_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

        # If the video is set to be a premiere, override the privacyStatus to 'private'
        # and ensure a publishAt time is set as required by the YouTube API for premieres.
        if is_premiere and publish_at:
            body['status']['privacyStatus'] = 'private'

        # Create the `videos.insert` request with the prepared body and media file.
        insert_request = youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=googleapiclient.http.MediaFileUpload(options['file'], chunksize=-1, resumable=True)
        )

        return self.resumable_upload(insert_request)


    def resumable_upload(self, insert_request: googleapiclient.http.MediaFileUpload):
        """
        Implements an exponential backoff strategy to resume a failed video upload.

        This function attempts to upload a video in chunks and handles retriable
        HTTP errors and exceptions by sleeping for an exponentially increasing
        amount of time before retrying.

        Args:
            insert_request (googleapiclient.http.MediaFileUpload): The resumable upload request object.

        Returns:
            dict: The API response from the successful upload, including the video's ID.

        Raises:
            Exception: If the maximum number of retries is exceeded without a successful upload.
            HttpError: For non-retriable HTTP errors.
        """
        response = None
        error = None
        retry = 0
        while response is None:
            try:
                logging.info("Starting file upload...")
                # Upload the next chunk of the video file.
                status, response = insert_request.next_chunk()
                if status:
                    logging.info(f"Uploaded {status.resumable_progress * 100:.0f}% of the file.")
                if response and 'id' in response:
                    logging.info(f"Video id '{response['id']}' was successfully uploaded.")
                    return response
            except HttpError as e:
                # Check for retriable status codes.
                if e.resp.status in RETRIABLE_STATUS_CODES:
                    error = f"A retriable HTTP error {e.resp.status} occurred:\n{e.content}"
                else:
                    # If the error is not retriable, re-raise it immediately.
                    raise
            except RETRIABLE_EXCEPTIONS as e:
                # Handle other retriable exceptions.
                error = f"A retriable error occurred: {e}"

            if error is not None:
                logging.error(f"Error occurred: {error}")
                retry += 1
                if retry > MAX_RETRIES:
                    raise Exception("No longer attempting to retry.")

                # Calculate sleep time with exponential backoff and add random jitter.
                # Jitter helps to prevent a thundering herd of retries.
                max_sleep = 2 ** retry
                sleep_seconds = random.random() * max_sleep
                logging.info(f"Sleeping for {sleep_seconds:.2f} seconds before retrying...")
                time.sleep(sleep_seconds)


    def upload_video(self, youtube: any, video_path: str, title: str, description: str, category: str, keywords: str,
                     privacy_status: str, publish_at: datetime.datetime = None, is_premiere: bool = False):
        """
        A high-level function to handle the entire video upload process.

        This function first retrieves the channel ID, then calls `initialize_upload`
        to prepare the request, and handles any potential HTTP errors during the upload.

        Args:
            youtube (any): The authenticated YouTube service object.
            video_path (str): The local path to the video file.
            title (str): The title of the video.
            description (str): The video description.
            category (str): The video category ID.
            keywords (str): A comma-separated string of keywords/tags.
            privacy_status (str): The privacy status of the video ('public', 'private', 'unlisted').
            publish_at (datetime.datetime, optional): The scheduled publish time in UTC.
                                                     Defaults to None.
            is_premiere (bool, optional): Whether to schedule the video as a Premiere.

        Returns:
            dict: The response from the successful upload.

        Raises:
            HttpError: If an unrecoverable HTTP error occurs during the upload.
        """
        try:
            # Prepare options dictionary
            options = {
                'file': video_path,
                'title': title,
                'description': description,
                'category': category,
                'keywords': keywords,
                'privacyStatus': privacy_status
            }

            # If privacyStatus is not private but a publish_at is provided, set it to private
            if publish_at and privacy_status != "private":
                options['privacyStatus'] = "private"

            # Initialize and execute the video upload process
            video_response = self.initialize_upload(
                youtube,
                options=options,
                publish_at=publish_at,
                is_premiere=is_premiere
            )
            return video_response
        except HttpError as e:
            logging.error(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
            # Handle specific authentication errors by attempting to re-authenticate.
            if e.resp.status in [401, 403]:
                logging.info("Re-authenticating and retrying...")
                # Note: This will prompt the user for re-authentication if necessary.
                youtube, _ = self.get_authenticated_service()
                video_response = self.initialize_upload(youtube, options, publish_at, is_premiere)
                return video_response
            else:
                raise e


    def set_video_thumbnail(self, youtube: any, video_id: str, thumbnail_path: str):
        """
        Sets a custom thumbnail for a YouTube video using a local image file.

        Args:
            youtube (any): The authenticated YouTube Data API service object.
            video_id (str): The ID of the YouTube video to which the thumbnail will be applied.
            thumbnail_path (str): The file path to the thumbnail image (JPG or PNG).

        Returns:
            dict: The API response if successful, otherwise None.
        """
        try:
            # Check if the thumbnail file exists before attempting to upload.
            if not os.path.exists(thumbnail_path):
                logging.error(f"Error: Thumbnail file not found at {thumbnail_path}")
                return None

            # Determine the correct MIME type based on the file extension.
            if thumbnail_path.lower().endswith(('.jpg', '.jpeg')):
                mimetype = 'image/jpeg'
            elif thumbnail_path.lower().endswith('.png'):
                mimetype = 'image/png'
            else:
                logging.error("Error: Invalid thumbnail file type. Must be JPG or PNG.")
                return None

            # Create the media file upload object.
            media = googleapiclient.http.MediaFileUpload(thumbnail_path, mimetype=mimetype)

            # Create the request to set the video thumbnail.
            request = youtube.thumbnails().set(
                videoId=video_id,
                media_body=media
            )

            # Execute the request.
            response = request.execute()

            logging.info(f"Successfully set thumbnail for video ID: {video_id}")
            return response

        except HttpError as e:
            logging.error(f"HTTP Error setting thumbnail: {e}")
            # Provide detailed error messages for common HTTP status codes.
            if e.resp.status == 400:
                logging.error("Error 400: Bad Request. Check video ID and thumbnail format.")
            elif e.resp.status == 401:
                logging.error("Error 401: Unauthorized. Check your authentication credentials.")
            elif e.resp.status == 403:
                logging.error("Error 403: Forbidden. You don't have permission to modify this video.")
            elif e.resp.status == 404:
                logging.error(f"Error 404: Video ID '{video_id}' not found.")
            elif e.resp.status == 405:
                logging.error("Error 405: Method Not Allowed. Check your API request.")
            elif e.resp.status == 500:
                logging.error("Error 500: Internal Server Error. Try again later.")
            else:
                logging.error(f"Unexpected HTTP error: {e.resp.status}")
            return None

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return None


    def create_playlist(self, youtube: any, title: str) -> str:
        """
        Creates a new YouTube playlist with the specified title.

        Args:
            youtube (any): The authenticated YouTube Data API service object.
            title (str): The title for the new playlist.

        Returns:
            str: The ID of the newly created playlist.
        """
        try:
            logging.info(f"Creating new playlist with title: '{title}'...")
            request_body = {
                'snippet': {
                    'title': title,
                    'description': f"Playlist for the video '{title}'.",
                    'privacyStatus': 'unlisted'
                },
                'status': {
                    'privacyStatus': 'unlisted'
                }
            }

            insert_request = youtube.playlists().insert(
                part="snippet,status",
                body=request_body
            )

            response = insert_request.execute()
            playlist_id = response['id']
            logging.info(f"Successfully created playlist. Playlist ID: {playlist_id}")
            return playlist_id

        except HttpError as e:
            logging.error(f"HTTP Error creating playlist: {e}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred while creating the playlist: {e}")
            return None


    def find_or_create_playlist(self, youtube: any, title: str) -> str:
        """
        Finds a playlist by title for the current user. If it doesn't exist, it creates it.

        Args:
            youtube (any): The authenticated YouTube Data API service object.
            title (str): The title of the playlist to find or create.

        Returns:
            str: The ID of the existing or newly created playlist.
        """
        try:
            logging.info(f"Searching for playlist with title: '{title}'...")
            # Search for a playlist by title for the authenticated user.
            search_request = youtube.playlists().list(
                part="id,snippet",
                mine=True,
                q=title
            )
            search_response = search_request.execute()

            # Check if any playlists were found.
            # It's better to iterate to find an exact match in case of partial matches.
            for playlist in search_response['items']:
                if playlist['snippet']['title'].strip().lower() == title.strip().lower():
                    logging.info(f"Existing playlist found! ID: {playlist['id']}")
                    return playlist['id']

            logging.info("No playlist with that exact name was found.")
            # If no exact match found, create a new one.
            return self.create_playlist(youtube, title)

        except HttpError as e:
            logging.error(f"An HTTP error occurred while searching for the playlist: {e}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred while searching for the playlist: {e}")
            return None


    def add_video_to_playlist(self, youtube: any, video_id: str, playlist_id: str):
        """
        Adds a video to an existing YouTube playlist.

        Args:
            youtube (any): The authenticated YouTube Data API service object.
            video_id (str): The ID of the video to add.
            playlist_id (str): The ID of the playlist to add the video to.

        Returns:
            dict: The API response from the playlist item insertion.
        """
        try:
            logging.info(f"Adding video '{video_id}' to playlist '{playlist_id}'...")
            # Create the request body for the playlist item.
            body = {
                'snippet': {
                    'playlistId': playlist_id,
                    'resourceId': {
                        'kind': 'youtube#video',
                        'videoId': video_id
                    }
                }
            }

            # Insert the video into the playlist.
            insert_request = youtube.playlistItems().insert(
                part="snippet",
                body=body
            )

            response = insert_request.execute()
            logging.info(f"Video '{video_id}' successfully added to playlist '{playlist_id}'.")
            return response

        except HttpError as e:
            logging.error(f"An HTTP error occurred while adding the video to the playlist:\n{e.content}")
            # Provide specific error messages for common playlist-related errors.
            if e.resp.status == 404:
                logging.error("Error 404: The specified playlist ID was not found.")
            elif e.resp.status == 403:
                logging.error("Error 403: Forbidden. You don't have permission to modify this playlist.")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred while adding the video to the playlist: {e}")
            return None


    def handle_video_upload(self, video_path: str, thumbail_path: str, metadata: dict) -> tuple[str, str]:
        """
        Handles the complete video and thumbnail upload workflow.

        This function is the main entry point for the upload process. It retrieves the
        authenticated YouTube service, performs the video upload, and then
        sets the thumbnail. It also checks for the existence of the client secrets file.

        Args:
            video_path (str): The local path to the video file.
            thumbail_path (str): The local path to the thumbnail image file.
            metadata (dict): A dictionary containing video metadata (title, description, etc.),
                             including an optional 'publish_at' datetime object.

        Returns:
            tuple: A tuple containing the YouTube URL (str) and the status of the upload ('Success' or 'Failed').
            :param self:
        """
        # Get the authenticated YouTube service.
        # youtube, channel_id = self.get_authenticated_service_by_channel("El Universo del Audiolibro")
        # get_authenticated_service()

        if self.youtube is None:
            print(f"✅ Youtube credentials are not ready: {self.youtube}")
            return "", ""

        client_secrets_file = os.path.abspath(CLIENT_SECRET_FILE)
        if not os.path.exists(client_secrets_file):
            logging.error("Client secrets file missing. YouTube upload will be skipped.")
            logging.error("Please download the client_secret.json from Google Cloud Platform and store it inside the "
                          "/Backend directory.")
            return "", "Failed: Client secrets file not found"

        # Use the publish_at from the metadata if it exists, otherwise calculate a new one.
        if 'publish_at' in metadata:
            future_publish_at = metadata['publish_at']
        else:
            # Calculate a future publish time (at least 15 minutes from now for a Premiere).
            future_publish_at = datetime.datetime.utcnow().replace(tzinfo=pytz.utc) + datetime.timedelta(minutes=15)

        # Print the scheduled publish time for debugging and verification purposes.
        logging.info(f"The scheduled publishAt from metadata is: {metadata.get('publish_at')}")
        logging.info(f"The scheduled publishAt time is: {future_publish_at.isoformat()}")

        try:
            # Upload the video and get the response, which includes the new video's ID.
            video_response = self.upload_video(
                youtube=self.youtube,
                video_path=os.path.abspath(video_path),
                title=metadata.get('title'),
                description=metadata.get('description'),
                category="24",  # 24: Entertainment, 27: Education
                keywords=metadata.get('keywords'),
                privacy_status="private",
                publish_at=future_publish_at,
                is_premiere=True
            )

            # If the video upload was successful, proceed to set the thumbnail.
            if video_response and 'id' in video_response:
                video_id = video_response['id']
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                logging.info(f"Uploaded video ID: {video_id}")

                # # 1. Find or create the playlist based on the video's title.
                # playlist_title = metadata.get('playlist_title')
                # playlist_id = find_or_create_playlist(youtube, playlist_title)
                #
                # if playlist_id:
                #     # 2. Add the video to the newly found or created playlist.
                #     add_video_to_playlist(youtube, video_id, playlist_id)

                # 3. Set the video's thumbnail.
                thumbnail_response = self.set_video_thumbnail(self.youtube, video_id, thumbail_path)
                logging.info(f"Thumbnail upload response: {thumbnail_response}")

                return video_url, "Success"
            else:
                logging.error("The video upload failed or the video ID was not found.")
                return "", "Failed: Upload failed or video ID not found"

        except HttpError as e:
            logging.error(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
            return "None", f"Failed: HTTP Error {e.resp.status}"
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return "None", f"Failed: {e}"


def bulk_upload_reels():
    from excelhandler import ExcelHandler

    logging.info("Starting bulk reel upload...")
    excel_handler = ExcelHandler(str(EXCEL_FILE_PATH))
    data = excel_handler.read_data(sheet_name="Content")

    # Initialize ContentGenerator
    content_generator = ContentGenerator()
    youtube_uploader = YouTubeUploader(client_secret_file=CLIENT_SECRET_FILE, channels=CHANNELS)

    # Example: authenticate one channel
    youtube, channel_id = youtube_uploader.get_authenticated_service_by_channel("Mente de Campeón: El Poder de Tu Pensamiento")
    if youtube:
        print(f"✅ Channel ready: {channel_id}")

    if not data:
        logging.error("No data found in the Excel file.")
        return

    successful_uploads = 0
    failed_uploads = 0

    for index, info in enumerate(data, 1):

        if info.get('Enabled') in (0, False):
            logging.info(f"Excel Enabled: {info.get('Enabled')}")
            continue

        chapter_name = info.get('Title')
        final_video_path = info.get('Output path')
        caption = info.get('Copywriting', 'Test Reel')
        chapter_text = info.get('Chapter')
        book_title = info.get("Book title")
        image_path = str(IMG_DIR / info.get('Thumbnail image'))
        image_path_output = str(IMG_DIR / info.get('Video cover'))

        if not final_video_path or not os.path.exists(final_video_path):
            logging.error(f"Video not found: {final_video_path}")
            failed_uploads += 1
            continue

        try:

            logging.info("Generating metadata for YouTube upload.")
            title, description, keywords = content_generator.generate_metadata_shorts(book_title,
                                                                                      chapter_text[:500],
                                                                                      "Spanish")

            # Example: Schedule for Saturday at 9:00 AM, default timezone
            schedule = "weekly_monday_5am"
            publish_video_at = get_next_publish_datetime(schedule_name=schedule, day='monday', target_time='5:00',
                                                         start_date='2025-09-29', target_week=True, target_day=False)

            metadata = {
                'title': title,
                'description': f"{chapter_name}\n\n{description}\n\n",
                'keywords': ",".join(keywords),
                'publish_at': publish_video_at,
            }

            logging.info(f"Generated metadata: {metadata}")

            logging.info("Uploading video to YouTube.")

            # New call to capture the results
            video_url, upload_status = youtube_uploader.handle_video_upload(video_path=final_video_path,
                                                                            thumbail_path=image_path,
                                                                            metadata=metadata)

            # You can use these values to display the information
            if upload_status == "Success":
                logging.info(f"✅ Video {index} scheduled uploaded successfully!  URL: {video_url}")
            else:
                logging.error(f"❌ Complete failure for video {index}. Status: {upload_status}")

        except Exception as e:
            logging.exception(f"Error processing video {index}: {e}")
            failed_uploads += 1
            print(f"❌ Video {index} failed: {str(e)}")

        # Wait between requests
        time.sleep(15)

    logging.info(f"Summary: Successful: {successful_uploads}, Failed: {failed_uploads}")
    print(f"\n✅ Bulk upload complete: {successful_uploads} successful, {failed_uploads} failed.")


if __name__ == "__main__":
    logging.info("__main__")

    # try:
    #
    #     # CHANNELS = {
    #     #     "El Universo del Audiolibro": "credentials1.json",
    #     #     "Canal Metafísico": "credentials2.json",
    #     #     "Reflexiones Diarias": "credentials3.json",
    #     #     "Audiolibros Premium": "credentials4.json",
    #     # }
    #     #
    #     # # Path to your client_secret.json
    #     # CLIENT_SECRET_FILE = Path(__file__).resolve().parent / "credentials" / "client_secret.json"
    #
    #     # yt_auth = YouTubeMultiChannelAuth(client_secret_file=CLIENT_SECRET_FILE, channels=CHANNELS)
    #     #
    #     # # Example: authenticate one channel
    #     # youtube, channel_id = yt_auth.get_authenticated_service("El Universo del Audiolibro")
    #     # if youtube:
    #     #     print(f"✅ Channel ready: {channel_id}")
    #
    # bulk_upload_reels()
    # except Exception as e:
    #     logging.exception("Critical error in bulk upload.")
    #     print(f"❌ Critical error: {str(e)}")

    #
    # from excelhandler import ExcelHandler
    #
    # logging.info("Generating metadata")
    #
    # # final_video_path = str((Path(__file__).resolve().parent.parent / "output" /
    # #                       "output_a3e84ee4-a39d-4764-9edd-f436f906e0ff.mp4"))
    #
    # thumbail_path = str((Path(__file__).resolve().parent.parent / "images" /
    #                       "image_content_front.png"))
    #
    #
    # subject = """
    #     ¿Y si la vida que siempre has soñado ya te pertenece? Reclama tu derecho a la abundancia y descubre cómo vivirla plenamente.
    # """
    #
    # script = """
    #     La abundancia es un derecho. Tienes derecho a vivir una vida de abundancia y felicidad. Nuestro método te enseña a reclamar tu derecho a la abundancia y a vivir la vida que deseas. Deja de lado la escasez y la limitación, y comienza a vivir la vida que te mereces. La abundancia es tu derecho, reclámala.
    #
    #     Imagina que estás viviendo una vida donde todo es posible, donde no hay límites ni restricciones. Imagina que tienes todo lo que necesitas y deseas, y que estás viviendo una vida de plena satisfacción y felicidad. Esto es lo que la abundancia puede ofrecerte.
    #
    #     Pero la abundancia no es solo algo que se te da, es algo que debes reclamar. Debes tomar conciencia de que tienes derecho a vivir una vida de abundancia y felicidad, y debes estar dispuesto a hacer lo que sea necesario para lograrlo. Nuestro método te enseña a hacer esto de manera efectiva, para que puedas vivir la vida que deseas y reclamar tu derecho a la abundancia.
    #
    #     La escasez y la limitación son solo creencias y patrones de pensamiento que te impiden vivir la vida que deseas. No son la realidad, sino solo una percepción de la realidad. Puedes cambiar esta percepción y comenzar a ver la abundancia y la posibilidad en tu vida. Puedes comenzar a creer en ti mismo y en tus capacidades, y a tomar acción para lograr tus objetivos.
    #
    #     Al reclamar tu derecho a la abundancia, puedes comenzar a vivir la vida que te mereces. Puedes comenzar a experimentar una mayor abundancia de dinero, de tiempo, de energía y de recursos. Puedes comenzar a vivir una vida de plena satisfacción y felicidad, y a disfrutar de todo lo que la vida tiene que ofrecer.
    #
    #     Recuerda que la abundancia es un derecho que te pertenece. No es algo que debes pedir permiso para tener, sino algo que debes reclamar como tuyo. Nuestro método te enseña a hacer esto de manera efectiva, para que puedas vivir la vida que deseas y reclamar tu derecho a la abundancia. Deja de lado la escasez y la limitación, y comienza a vivir la vida que te mereces. La abundancia es tu derecho, reclámala.
    #
    #     La abundancia es un estado de conciencia que puedes elegir cada día. Puedes elegir ver la abundancia y la posibilidad en tu vida, o puedes elegir ver la escasez y la limitación. La elección es tuya. Pero recuerda que la abundancia es tu derecho, y que debes reclamarla para vivir la vida que te mereces. Así que comienza a reclamar tu derecho a la abundancia hoy mismo, y comienza a vivir la vida que siempre has soñado. La abundancia es posible, y está al alcance de tu mano.
    #
    #     Si esto resonó contigo, es porque estás listo para más; imagina todo lo que podemos descubrir juntos. Sígueme y sigamos creciendo juntos.
    #
    # """
    #
    # # Initialize ContentGenerator
    # content_generator = ContentGenerator()
    #
    # youtube_uploader = YouTubeUploader()
    #
    # ai_model = "gemini-2.0-flash"
    # language_mx = "Spanish"
    #
    # final_video_path = str((Path(__file__).resolve().parent.parent / "audiobook_content" / "output" /
    #                       "output_43e080b7-fc3c-43d3-9d5d-a78b287cc582.mp4"))
    #
    # thumbail_path = str((Path(__file__).resolve().parent.parent / "audiobook_content" / "images" /
    #                       "la_ciencia_de_hacerse_rico_v2.png"))
    #
    # title, description, keywords = content_generator.generate_metadata_shorts(subject, script, language_mx)
    #
    # metadata = {
    #     'title': title,
    #     'description': description,
    #     'keywords': ",".join(keywords),
    # }
    #
    # print("[+] Uploading video to YouTube")
    # youtube_uploader.handle_video_upload(final_video_path, thumbail_path, metadata)

    #
    # excel_handler = ExcelHandler(str(EXCEL_FILE_PATH))
    # logging.info(f"Excel handler initialized with path: {EXCEL_FILE_PATH}")
    #
    # # Read data from the specified sheet
    # data = excel_handler.read_data(sheet_name="Histories")
    # if not data:
    #     logging.error("No data found in the Excel file.")
    #     exit()
    #
    # # Get the path for the thumbnail image
    # thumbnail_path = str((Path(__file__).resolve().parent.parent / "images" / "image_content_front.png"))
    #
    # # Iterate through the data and upload each video
    # for info in data:
    #
    #     final_video_path = info.get('Video output path')
    #     keywords = "Leadership"
    #     metadata = {
    #         'title': info.get('Subject'),
    #         'description': info.get('Copywrite'),
    #         'keywords': ",".join(keywords),
    #     }
    #
    #     if os.path.exists(final_video_path):
    #         logging.info("Uploading video to YouTube")
    #         handle_video_upload(final_video_path, thumbail_path, metadata)
    #     else:
    #         logging.error(f"The file does NOT exist: {final_video_path}")
