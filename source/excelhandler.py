from pathlib import Path
import pandas as pd
from logger_config import setup_logger
import os
from typing import List, Dict, Optional
import uuid
import time
from datetime import datetime
import pytz

from utils import get_next_publish_datetime, convert_datetime

# --- Configuration ---
logging = setup_logger(__name__)


class ExcelHandler:
    def __init__(self, file_path: str):
        """
        Initialize the class with the Excel file path.
        :param file_path: Path to the Excel file.
        """
        self.file_path = file_path
        self.sheets = {}  # Dictionary to store DataFrames for each sheet

        # Define valid sheets
        self.valid_sheets = ['Content', 'Shorts']

        # Define columns for each sheet
        self.columns_map = {
            'Content': ['Id', 'Book title', 'Title', 'Chapter', 'Copywriting', 'Schedule', 'Thumbnail image', 'Video cover', 'Output path', 'Enabled'],
            'Shorts': ['Id', 'Book title', 'Title', 'Chapter', 'Copywriting', 'Schedule', 'Output path', 'Enabled']
        }

        # Define dtypes for each sheet (all as string/object)
        # Schedule will be converted later when needed
        self.dtypes_map = {
            'Content': {col: 'object' for col in self.columns_map['Content']},
            'Shorts': {col: 'object' for col in self.columns_map['Shorts']}
        }

        self._initialize_excel_file()

    def _initialize_excel_file(self):
        """
        Check if Excel file exists, load it or create a new one with required sheets.
        """
        try:
            if os.path.exists(self.file_path):
                logging.info(f"Loading Excel file: {self.file_path}...")
                with pd.ExcelFile(self.file_path) as xls:
                    logging.info(f"Available sheets in Excel: {xls.sheet_names}")
                    for sheet_name in self.valid_sheets:
                        if sheet_name in xls.sheet_names:
                            logging.info(f"Loading sheet: {sheet_name}")
                            self.sheets[sheet_name] = pd.read_excel(
                                xls,
                                sheet_name=sheet_name,
                                dtype=self.dtypes_map[sheet_name]
                            )
                            # Ensure all required columns exist
                            for col in self.columns_map[sheet_name]:
                                if col not in self.sheets[sheet_name].columns:
                                    logging.warning(f"Column {col} missing in {sheet_name}. Adding with NA.")
                                    self.sheets[sheet_name][col] = pd.NA
                            logging.info(f"Sheet {sheet_name} loaded with {len(self.sheets[sheet_name])} rows.")
                        else:
                            logging.warning(f"Sheet {sheet_name} not found in Excel. Creating empty DataFrame.")
                            self.sheets[sheet_name] = pd.DataFrame(columns=self.columns_map[sheet_name]).astype(self.dtypes_map[sheet_name])
                logging.info("Excel file loaded successfully.")
            else:
                logging.warning(f"Excel file does not exist at {self.file_path}. Creating a new one.")
                for sheet_name in self.valid_sheets:
                    self.sheets[sheet_name] = pd.DataFrame(columns=self.columns_map[sheet_name]).astype(self.dtypes_map[sheet_name])

                with pd.ExcelWriter(self.file_path, engine='openpyxl') as writer:
                    for sheet_name in self.valid_sheets:
                        self.sheets[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
                logging.info(f"Created new Excel file at {self.file_path} with {', '.join(self.valid_sheets)} sheets.")
        except Exception as e:
            logging.error(f"Error initializing Excel file: {e}")
            raise

    def read_data(self, sheet_name: str, convert_schedule: bool = False) -> List[Dict]:
        """
        Read data from the specified sheet.
        :param sheet_name: Name of the sheet (e.g., 'Content', 'Shorts')
        :param convert_schedule: If True, convert 'Schedule' column to datetime.
        :return: List of dictionaries with row data.
        """
        try:
            if sheet_name not in self.valid_sheets:
                raise ValueError(f"Invalid sheet name: {sheet_name}. Use one of {self.valid_sheets}.")

            df = self.sheets.get(sheet_name)
            if df is None or df.empty:
                logging.warning(f"The {sheet_name} sheet is empty or not initialized.")
                return []

            # Optionally convert Schedule column to datetime
            if convert_schedule and 'Schedule' in df.columns:
                try:
                    df['Schedule'] = pd.to_datetime(df['Schedule'], errors='coerce')
                except Exception as e:
                    logging.warning(f"Could not convert 'Schedule' to datetime: {e}")

            logging.info(f"Reading data from {sheet_name} sheet. Found {len(df)} rows.")
            return df.to_dict(orient='records')
        except Exception as e:
            logging.error(f"Error reading data from {sheet_name} sheet: {e}")
            return []

    def write_data(self, data: List[Dict], sheet_name: str):
        """
        Write new data to the specified sheet.
        :param data: List of dictionaries containing data to write.
        :param sheet_name: Name of the sheet to write to.
        """
        try:
            if sheet_name not in self.valid_sheets:
                raise ValueError(f"Invalid sheet name: {sheet_name}. Use one of {self.valid_sheets}.")

            if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
                raise ValueError("Data must be a list of dictionaries.")

            logging.info(f"Writing {len(data)} rows to {sheet_name} sheet...")
            new_df = pd.DataFrame(data)

            # Ensure all required columns exist
            for col in self.columns_map[sheet_name]:
                if col not in new_df.columns:
                    new_df[col] = pd.NA

            # Append new rows
            self.sheets[sheet_name] = pd.concat([self.sheets[sheet_name], new_df], ignore_index=True)

            # Save back to Excel
            with pd.ExcelWriter(self.file_path, engine='openpyxl', mode='w') as writer:
                for sheet in self.valid_sheets:
                    self.sheets[sheet].to_excel(writer, sheet_name=sheet, index=False)

            logging.info(f"Data written successfully to {sheet_name} sheet in {self.file_path}")
        except Exception as e:
            logging.error(f"Error writing data to {sheet_name} sheet: {e}")
            raise

    def write_data_by_title(self, title: str, sheet_name: str, book_title: str = None, chapter: str = None,
                            copy_text: str = None, schedule: str = None, video_path: str = None,
                            thumbnail_image: str = None, video_cover: str = None, enabled: bool = None):
        """
        Updates or adds an entry to the specified sheet for a specific Title.
        Only non-empty fields are updated. Handles NaN/None and removes timezones from datetime.

        Args:
            title (str): The topic to search for in the 'Title' column (Required).
            sheet_name (str): Name of the sheet to update (e.g., 'Content', 'Shorts').
            book_title (str, optional): Text for the 'Book title' column. Defaults to None.
            chapter (str, optional): Text for the 'Chapter' column. Defaults to None.
            copy_text (str, optional): Text for the 'Copywriting' column. Defaults to None.
            schedule (str or datetime, optional): Text or datetime for the 'Schedule' column. Defaults to None.
            video_path (str, optional): Text for the 'Output path' column. Defaults to None.
            thumbnail_image (str, optional): Text for the 'Thumbnail image' column (Content only). Defaults to None.
            video_cover (str, optional): Text for the 'Video cover' column (Content only). Defaults to None.
            enabled (bool, optional): Value for the 'Enabled' column. Defaults to None.
        """
        try:
            if sheet_name not in self.valid_sheets:
                raise ValueError(f"Invalid sheet name: {sheet_name}. Use one of {self.valid_sheets}.")

            if self.sheets.get(sheet_name) is None:
                logging.warning(f"{sheet_name} sheet not initialized. Creating empty DataFrame.")
                self.sheets[sheet_name] = pd.DataFrame(columns=self.columns_map[sheet_name]).astype(self.dtypes_map[sheet_name])
                logging.info(f"Initialized empty {sheet_name} DataFrame.")

            logging.info(f"Processing Title '{title}' in {sheet_name} sheet...")

            # Convert schedule safely to timezone-unaware datetime if provided
            processed_schedule = None
            if schedule:
                if isinstance(schedule, str):
                    try:
                        # Convert the string to a timezone-unaware datetime object
                        # This is a safe way to handle various datetime string formats
                        processed_schedule = pd.to_datetime(schedule).tz_localize(None)
                        logging.info("Converted schedule string to timezone-unaware datetime object.")
                    except Exception as e:
                        logging.warning(f"Could not convert schedule string to datetime: {e}. Keeping as string.")
                        processed_schedule = schedule
                elif isinstance(schedule, datetime):
                    processed_schedule = schedule.replace(tzinfo=None) if schedule.tzinfo else schedule

            # Find row index for the given title
            title_index = self.sheets[sheet_name][self.sheets[sheet_name]['Title'] == title].index

            if not title_index.empty:
                # Update existing row
                updates = {
                    'Book title': book_title,
                    'Chapter': chapter,
                    'Copywriting': copy_text,
                    'Schedule': processed_schedule,
                    'Output path': video_path,
                    'Enabled': enabled
                }
                if sheet_name == 'Content':
                    updates.update({
                        'Thumbnail image': thumbnail_image,
                        'Video cover': video_cover
                    })

                # Apply updates only for non-empty values
                for col, val in updates.items():
                    if val is not None and val != "":
                        self.sheets[sheet_name].loc[title_index, col] = val

                logging.info(f"Updated existing entry for Title '{title}' in {sheet_name} sheet.")
            else:
                # Add new row
                logging.info(f"Title '{title}' not found in {sheet_name} sheet. Adding new entry.")
                new_entry = {
                    'Id': str(uuid.uuid4()),
                    'Book title': book_title,
                    'Title': title,
                    'Chapter': chapter,
                    'Copywriting': copy_text,
                    'Schedule': processed_schedule if processed_schedule is not None else pd.NaT,
                    'Output path': video_path,
                    'Enabled': enabled
                }
                if sheet_name == 'Content':
                    new_entry['Thumbnail image'] = thumbnail_image
                    new_entry['Video cover'] = video_cover

                # Centralize datetime cleaning for new entries
                if isinstance(new_entry['Schedule'], datetime) and new_entry['Schedule'].tzinfo is not None:
                    new_entry['Schedule'] = new_entry['Schedule'].replace(tzinfo=None)

                new_df = pd.DataFrame([new_entry]).astype(self.dtypes_map[sheet_name])
                self.sheets[sheet_name] = pd.concat([self.sheets[sheet_name], new_df], ignore_index=True)

            # Ensure 'Schedule' column is datetime and timezone-unaware
            if 'Schedule' in self.sheets[sheet_name].columns:
                self.sheets[sheet_name]['Schedule'] = pd.to_datetime(self.sheets[sheet_name]['Schedule'], errors='coerce')

            # Save all sheets to Excel
            with pd.ExcelWriter(self.file_path, engine='openpyxl', mode='w') as writer:
                for sheet in self.valid_sheets:
                    self.sheets[sheet].to_excel(writer, sheet_name=sheet, index=False)

            logging.info(f"Data save operation completed for Title '{title}' in {sheet_name} sheet.")
        except Exception as e:
            logging.error(f"Error modifying {sheet_name} sheet for Title '{title}': {e}")
            raise

if __name__ == "__main__":
    """
    Main function orchestrating the reading and writing of data to the Excel file.
    """
    try:

        # WriteDataTest()

        # --- Constants for Directory Paths ---
        BASE_DIR = Path(__file__).resolve().parent.parent
        BOOK_DIR = BASE_DIR / "audiobook_content"
        OUTPUT_DIR = BOOK_DIR / "output"
        EXCEL_PATH = BOOK_DIR / "docs" / "video_podcast_content.xlsx"

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Initialize Excel handler
        excel_handler = ExcelHandler(str(EXCEL_PATH))

        # Example: Read data from both sheets
        for sheet_name in ['Content']:
            excel_data = excel_handler.read_data(sheet_name=sheet_name)

            if not excel_data:
                logging.info(f"No data found in the {sheet_name} sheet.")
                continue

            logging.info(f"Printing first 10 items from {sheet_name} sheet:")

            for chapter in excel_data[-5:]:  # Limit to first 10 entries
                if sheet_name == 'Content':
                    print(
                        "************* New register **************\n"
                        f"Id: {chapter.get('Id')}\n"
                        f"Book title: {chapter.get('Book title')}\n"
                        f"Title: {chapter.get('Title')}\n"
                        f"Chapter: {chapter.get('Chapter')}\n"
                        f"Copywriting: {chapter.get('Copywriting')}\n"
                        f"Schedule: {chapter.get('Schedule')}\n"
                        f"Thumbnail image: {chapter.get('Thumbnail image')}\n"
                        f"Video cover: {chapter.get('Video cover')}\n"
                        f"Output path: {chapter.get('Output path')}\n"
                        f"Enabled: {chapter.get('Enabled')}\n"
                    )

                    # Sanitize Excel values before using them as paths
                    thumbnail_image = chapter.get('Thumbnail image')
                    video_cover = chapter.get('Video cover')

                    if pd.isna(thumbnail_image):
                        logging.warning(f"Missing Thumbnail image: {thumbnail_image} for chapter: {chapter.get('Chapter')}")
                        continue

                    if pd.isna(video_cover):
                        logging.warning(f"Missing Video cover: {video_cover} for chapter: {chapter.get('Chapter')}")
                        continue

                    image_path = str(BOOK_DIR / chapter.get('Thumbnail image'))
                    logging.info(f"image_path: {image_path}")

                else:  # Shorts
                    print(
                        "************* New register **************\n"
                        f"Id: {chapter.get('Id')}\n"
                        f"Book title: {chapter.get('Book title')}\n"
                        f"Title: {chapter.get('Title')}\n"
                        f"Chapter: {chapter.get('Chapter')}\n"
                        f"Copywriting: {chapter.get('Copywriting')}\n"
                        f"Schedule: {chapter.get('Schedule')}\n"
                        f"Output path: {chapter.get('Output path')}\n"
                        f"Enabled: {chapter.get('Enabled')}\n"
                    )
        # # Example: Add new data to Content sheet
        # new_content = [{
        #     'Id': str(uuid.uuid4()),
        #     'Book title': 'Sample Book',
        #     'Title': 'The Power of Thought',
        #     'Chapter': 'Chapter 1',
        #     'Copywriting': 'Inspirational content',
        #     'Schedule': '',
        #     'Output path': str(OUTPUT_DIR / f"output_{uuid.uuid4()}.mp4"),
        #     'Enabled': True
        # }]
        # excel_handler.write_data(new_content, sheet_name='Shorts')
        #
        # # Example: Add or update entry in Shorts sheet
        # excel_handler.write_data_by_title(
        #     sheet_name='Shorts',
        #     title='Short Video 1',
        #     book_title='Sample Book',
        #     chapter='Short Clip',
        #     copy_text='Quick inspiration',
        #     schedule='2025-08-31 10:00:00',
        #     video_path=str(OUTPUT_DIR / f"short_{uuid.uuid4()}.mp4"),
        #     enabled=True
        # )

    except Exception as e:
        logging.error(f"Error in process: {e}")
