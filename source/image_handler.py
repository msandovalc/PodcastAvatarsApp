import os
import logging
import uuid

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
logging.basicConfig(level=logging.INFO)


class ImageTextEditor:
    def __init__(self, image_path):
        """
        Initializes the class with the given image.
        """
        self.image = Image.open(image_path)
        self.draw = ImageDraw.Draw(self.image)
        self.width, self.height = self.image.size
        logging.info("Image loaded successfully. Dimensions: %dx%d", self.width, self.height)

    def add_text(self, text, position, font_path="Candaraz.ttf", font_size=45, color=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0)):
        """
        Adds centered text to the image.
        :param text: Text to add
        :param font_path: Path to the font file (ensure it exists)
        :param font_size: Font size
        :param color: Text color in RGB format
        :param stroke_width: Width of the stroke
        :param stroke_fill: Stroke color in RGB format
        """
        try:
            font = ImageFont.truetype(font_path, font_size)
            logging.info("Font loaded successfully.")
        except IOError:
            font = ImageFont.load_default()
            logging.warning("Font not found, using default font.")

        max_width = self.width - 50  # Ensure text box fits within image width
        lines = []
        words = text.split()  # Split text into words
        line = ""

        for word in words:
            test_line = f"{line} {word}".strip()
            bbox = self.draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                line = test_line
            else:
                lines.append(line)
                line = word
        lines.append(line)

        text_height = sum(self.draw.textbbox((0, 0), line, font=font)[3] for line in lines)
        y_start = (self.height - text_height) // 2

        # Draw background rectangle (background color)
        background_color = (0, 0, 0)  # Set your desired background color here (RGB)
        padding = 10  # Add padding around the text in the background

        for line in lines:
            bbox = self.draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x_position = (self.width - text_width) // 2

            # Draw rectangle for background
            self.draw.rectangle(
                ((x_position - padding, y_start), (x_position + text_width + padding, y_start + bbox[3])),
                # Rectangle size as tuple
                fill=background_color
            )

            self.draw.text((x_position, y_start), line, font=font, fill=color, stroke_width=stroke_width,
                           stroke_fill=stroke_fill)
            y_start += bbox[3]

        logging.info("Text added to image with centered alignment.")

    def add_logo(self, logo_path, position='top-center', scale=1.0):
        """
        Adds a logo image to the main image at the specified position.
        :param logo_path: Path to the logo image
        :param position: Position for the logo ('top-center', 'top-left', 'top-right', 'bottom-center', etc.)
        :param scale: Scaling factor for the logo size
        """
        try:
            logo = Image.open(logo_path)
            logo_width, logo_height = logo.size
            new_size = (int(logo_width * scale), int(logo_height * scale))
            logo = logo.resize(new_size, Image.LANCZOS)

            # Calculate position based on input parameter
            if position == 'top-center':
                x_position = (self.width - logo.size[0]) // 2
                y_position = 80  # You can adjust this value to control the vertical distance from the top
            elif position == 'top-left':
                x_position = 10  # Left margin
                y_position = 10  # Top margin
            elif position == 'top-right':
                x_position = self.width - logo.size[0] - 10  # Right margin
                y_position = 10  # Top margin
            elif position == 'bottom-center':
                x_position = (self.width - logo.size[0]) // 2
                y_position = self.height - logo.size[1] - 10  # Bottom margin
            elif position == 'bottom-left':
                x_position = 10  # Left margin
                y_position = self.height - logo.size[1] - 10  # Bottom margin
            elif position == 'bottom-right':
                x_position = self.width - logo.size[0] - 10  # Right margin
                y_position = self.height - logo.size[1] - 10  # Bottom margin
            else:
                raise ValueError(
                    "Invalid position specified. Use one of the following: 'top-center', 'top-left', 'top-right', 'bottom-center', 'bottom-left', 'bottom-right'")

            # Paste the logo into the image
            self.image.paste(logo, (x_position, y_position), logo if logo.mode == 'RGBA' else None)
            logging.info(f"Logo added successfully at {position}.")
        except IOError:
            logging.error("Failed to load logo image.")
        except ValueError as e:
            logging.error(str(e))

    def save_image(self, output_path):
        """
        Saves the edited image to the specified path.
        """
        self.image.save(output_path)
        logging.info("Image saved successfully at %s", output_path)


def creates_images(hook_text: str, image_input_path: str, image_logo_path: str, image_output_path: str,
                   font_size: int = 45) -> str:

    editor = ImageTextEditor(image_input_path)
    editor.add_text(hook_text, (50, 50), font_size=font_size, color=(255, 255, 255), stroke_width=1,
                    stroke_fill=(0, 0, 0))

    editor.add_logo(image_logo_path, position='top-center', scale=0.7)

    editor.save_image(image_output_path)

    return ""


# Example usage
if __name__ == "__main__":
    # # Output folder
    # images_folder_path = str((Path(__file__).resolve().parent.parent / "images"))
    # image_input_path = images_folder_path + "\\image_content_front.png"
    # image_input_back_path = images_folder_path + "\\image_content_back.png"
    # image_logo_path = images_folder_path + "\\image_logo_content.png"
    # image_output_path = images_folder_path + f"\\image_content_front_{uuid.uuid4()}.png"
    # image_output_back_path = images_folder_path + f"\\image_content_back_{uuid.uuid4()}.png"



    script = """
    No olvides seguirme...
    """

    # creates_images(script, image_input_back_path, image_logo_path, image_output_back_path, 80)
    # editor.add_text(script, (50, 50), font_size=40, color=(255, 255, 255), stroke_width=1,
    #                 stroke_fill=(0, 0, 0))
    #
    # editor.add_logo(image_logo_path, position='top-center', scale=0.7)
    #
    # editor.save_image(image_output_path)
