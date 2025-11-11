import os
import sys
import json
import re
import uuid
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from termcolor import colored
from logger_config import setup_logger
import wave

# --- Configuration ---
logging = setup_logger(__name__)

# Load environment variables
dotenv_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Set environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# genai.Client(api_key=GOOGLE_API_KEY) # Crucial for direct genai calls

# Voice mapping for Gemini TTS (update with actual voices from Google AI Studio)
VOICE_MAP = {
    "en_us_001": "Zephyr",  # Placeholder: Replace with actual Gemini TTS voice (e.g., English voice)
    "en_us_002": "Puck",
    "en_us_003": "Charon",
    "en_us_004": "Kore",
    "en_us_005": "Fenrir",
    "en_us_006": "Leda",
    "en_us_007": "Orus",
    "en_us_008": "Aoede",
    "en_us_009": "Callirhoe",
    "en_us_010": "Autonoe",
    "en_us_011": "Enceladus",
    "en_us_012": "Iapetus",
    "en_us_013": "Umbriel",
    "en_us_014": "Algieba",
    "en_us_015": "Despina",
    "en_us_016": "Erinome",
    "en_us_017": "Algenib",
    "en_us_018": "Rasalgethi",
    "en_us_019": "Laomedeia",
    "en_us_020": "Achernar",
    "en_us_021": "Alnilam",
    "en_us_022": "Schedar",
    "en_us_023": "Gacrux",
    "en_us_024": "Pulcherrima",
    "en_us_025": "Achird",
    "en_us_026": "Zubenelgenubi",
    "en_us_027": "Vindemiatrix",
    "en_us_028": "Sadachbia",
    "en_us_029": "Sadaltager",
    "en_us_030": "Sulafar",
}


class ContentGenerator:
    """
    A class to handle content generation using Google's Gemini models via LangChain and the
    direct Google GenAI client.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initializes the content generator with LangChain Google Generative AI models for text and TTS,
        and also configures the direct Google GenAI client for advanced TTS control.
        :param model_name: The name of the Gemini model to use for text generation.
        """
        try:
            # Initialize the LangChain Google Generative AI model for text generation
            self.text_llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=GOOGLE_API_KEY # Explicitly pass the API key to LangChain
            )
            logging.info(f"Initialized LangChain Google Generative AI model for text: {model_name}")

            # Note: self.tts_llm (LangChain) will NOT be used for TTS with explicit voice control in generate_tts.
            # We keep its initialization for consistency or if basic TTS without explicit voice control
            # is also desired elsewhere in the application via LangChain.
            self.tts_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-tts",
                google_api_key=GOOGLE_API_KEY
            )
            logging.info(f"Initialized LangChain Google Generative AI model for basic TTS (via LangChain).")

            # Initialize the direct Google GenAI Client for advanced TTS control
            # This is crucial for explicit voice selection as per your example.
            self.genai_client = genai.Client(api_key=GOOGLE_API_KEY)
            logging.info("Initialized direct Google GenAI Client for advanced TTS.")

        except Exception as e:
            logging.error(f"Failed to initialize models: {str(e)}")
            raise

    def _generate_response(self, prompt_template: PromptTemplate, input_vars: dict) -> str:
        """
        Generates a response using the LangChain Gemini text model.
        :param prompt_template: The PromptTemplate to use.
        :param input_vars: Dictionary of input variables for the prompt.
        :return: The generated response as a string.
        """
        try:
            full_prompt_text = prompt_template.format(**input_vars)

            # Use LangChain's invoke method for text generation
            response_object = self.text_llm.invoke(full_prompt_text)

            # LangChain's invoke for chat models returns a message object; access its 'content' attribute
            response_content = response_object.content.strip() if response_object and hasattr(response_object,
                                                                                              'content') else ""
            if not response_content:
                logging.warning("Empty response received from LangChain Gemini model.")
                print(colored("[-] LangChain Gemini model returned an empty response.", "red"))
            else:
                logging.debug("Response generated successfully.")
            return response_content
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            print(colored(f"[-] Error (LangChain Gemini text generation): {str(e)}", "red"))
            return ""

    # Set up the wave file to save the output:

    def generate_tts(self, script: str, voice: str, output_path: str, tone_instruction: Optional[str] = None) -> Optional[str]:
        """
        Generates audio from a script using the direct Google GenAI SDK to allow explicit voice selection.
        :param script: The text script to convert to audio.
        :param voice: The voice type (e.g., 'es_mx_002') to use for TTS.
        :param output_path: The file path to save the generated audio (e.g., 'path/to/audio.wav').
        :param tone_instruction: Optional instruction for the voice tone (e.g., "Read aloud in a warm and warm tone").
        :return: The output path if successful, None otherwise.
        """
        try:
            logging.debug(f"Starting TTS generation with script: {script}, voice: {voice}, output_path: {output_path}")

            if not script:
                logging.error("Empty script provided for TTS generation.")
                return None

            voice_id = VOICE_MAP.get(voice)
            logging.debug(f"Retrieved voice_id: {voice_id} for voice: {voice}")
            if not voice_id:
                logging.error(f"Unsupported voice type: {voice}. Available voices: {list(VOICE_MAP.keys())}")
                return None

            # Ensure output directory exists
            output_dir = Path(output_path).parent
            logging.debug(f"Ensuring output directory exists: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)

            logging.debug("Calling direct Google GenAI SDK to generate audio with explicit voice.")

            final_script = f"{tone_instruction} {script}" if tone_instruction else script
            logging.debug(colored(f"[*] Final script sent to TTS: {final_script}", "cyan"))

            # Use the direct google.genai SDK for explicit voice control
            # as per your example, using client.models.generate_content
            response = self.genai_client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=final_script, # 'contents' is the correct parameter name
                config=genai.types.GenerateContentConfig( # 'config' is the correct parameter name
                    response_modalities=["AUDIO"],
                    speech_config=genai.types.SpeechConfig(
                        voice_config=genai.types.VoiceConfig(
                            prebuilt_voice_config=genai.types.PrebuiltVoiceConfig(
                                voice_name=voice_id,
                            )
                        )
                    )
                )
            )

            def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
                with wave.open(filename, "wb") as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(rate)
                    wf.writeframes(pcm)

            # blob = response.candidates[0].content.parts[0].inline_data
            # print(blob.mime_type)

            # Access audio data as per your example: response.candidates[0].content.parts[0].inline_data.data
            data = response.candidates[0].content.parts[0].inline_data.data
            if not data:
                logging.error("No audio data was successfully retrieved from direct Google GenAI SDK.")
                logging.error(colored("[-] Error: No audio data was successfully retrieved from direct Google GenAI "
                                      "SDK.", "red"))
                logging.error(colored(f"[*] Full GenAI response structure: {response}", "red"))
                return None

            wave_file(output_path, data)  # Saves the file to current directory

            logging.info(f"TTS audio generated and saved at: {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Error generating TTS audio with direct GenAI SDK: {str(e)}")
            response_debug = response if 'response' in locals() else 'No response received'
            return None

    def generate_images_prompt(self, video_topic: str, number_of_images: int, ai_model: str, language: str,
                              custom_prompt: str = "") -> str:
        """
        Generate a JSON list of dictionaries for highly detailed and realistic AI image generation.
        :param video_topic: The subject of the video.
        :param number_of_images: The number of images to generate prompts for.
        :param ai_model: The AI model name to include in the output.
        :param language: The language for the prompt.
        :param custom_prompt: A custom prompt provided by the user (optional).
        :return: JSON string containing the list of image prompts.
        """
        try:
            prompt_text = custom_prompt or """
                Generate a JSON list of dictionaries for highly detailed and realistic AI image generation for 
                a video, strictly avoiding any cartoonish, anime, illustrative, or stylized appearance, based on the 
                following topic: "{video_topic}" and number of images {number_of_images}.
                Each dictionary in the list represents one image and must have the following keys:
                "image_id": A unique ID string in the format "image_1", "image_2", ..., one for every image.
                "prompt": A meticulous and lifelike description of the scene as a real photograph with exceptional detail 
                and realism for a 10-second video segment. Focus on lifelike accuracy in textures, materials, lighting, 
                and color rendition. Specify the concept, subject (with anatomically correct and believable features), 
                style (photorealistic, hyperrealism), lighting (natural or realistic artificial light), atmosphere 
                (real-world), colors (natural), composition (photographic), angle (realistic camera), detail level (high), 
                and image quality (8k, sharp focus, realistic rendering).
                "negative_prompt": "AVOID cartoonish, AVOID anime, DO NOT APPLY illustration, DO NOT INCLUDE sketch, 
                DO NOT APPLY painting, DO NOT INCLUDE drawing, DO NOT INCLUDE comic book style, AVOID manga, DO NOT APPLY 
                graphic novel, DO NOT INCLUDE cel-shaded, DO NOT INCLUDE ink drawing, DO NOT INCLUDE line art, 
                AVOID unrealistic, AVOID stylized, AVOID abstract, DO NOT APPLY flat shading, DO NOT APPLY smooth 
                gradients, DO NOT INCLUDE artificial lighting effects, DO NOT INCLUDE exaggerated features, 
                DO NOT INCLUDE unrealistic proportions, AVOID face distortion, AVOID distorted face, AVOID deformed face, 
                AVOID ugly face, AVOID bad ears, AVOID distorted ears, AVOID deformed ears, AVOID bad hands, 
                AVOID distorted hands, AVOID deformed hands, DO NOT INCLUDE extra fingers, DO NOT INCLUDE missing 
                fingers, DO NOT INCLUDE bad fingers, DO NOT INCLUDE multiple panels, DO NOT INCLUDE speech bubbles, 
                DO NOT APPLY artistic interpretation, DO NOT INCLUDE any form of visual art other than photography, 
                DO NOT INCLUDE nudity, DO NOT INCLUDE sexual content, DO NOT INCLUDE suggestive content, AVOID explicit 
                content, AVOID adult themes"
                "aspect_ratio": The string "9:16".
                "model": The string "{ai_model}".
                The output MUST be a valid JSON **list** of dictionaries, starting with '[' and ending with ']'. 
                DO NOT include ANY other text, headers, footers, or wrapping structures.
                Topic: {video_topic}
                Number of images: {number_of_images}
                Language: {language}
            """
            prompt_template = PromptTemplate(
                input_variables=["video_topic", "number_of_images", "language", "ai_model"],
                template=prompt_text
            )
            response = self._generate_response(
                prompt_template,
                {"video_topic": video_topic, "number_of_images": number_of_images, "language": language,
                 "ai_model": ai_model}
            )
            logging.info(f"Generated image prompts for topic: {video_topic}")
            print(colored(f"Generated Image List: {response}", "cyan"))
            return response
        except Exception as e:
            logging.error(f"Error generating image prompts: {str(e)}")
            return ""

    def generate_script(self, video_subject: str, paragraph_number: int, language: str,
                       custom_prompt: str = "") -> str:
        """
        Generate a script for a video based on the subject.
        :param video_subject: The subject of the video.
        :param paragraph_number: The number of paragraphs to generate.
        :param language: The language for the script.
        :param custom_prompt: A custom prompt provided by the user (optional).
        :return: The generated script as a string.
        """
        try:
            # prompt_text = custom_prompt or """Generate a script for a video, depending on the subject of the video.
            # The script must be returned as a string with the specified number of paragraphs. Do not under any
            # circumstance reference this prompt in your response. The script MUST start with a strong,
            # attention-grabbing hook within the very first sentence to immediately engage the viewer. This hook should
            # be thought-provoking, controversial, or directly address a common pain point related to the video's
            # subject. Get straight to the point, don't start with unnecessary things like, "welcome to this video".
            # Obviously, the script should be related to the subject of the video. Include at the end a paragraph: If
            # this content has resonated with you, don't miss out on more! Hit that follow button right now for daily
            # doses of inspiring wisdom, empowering insights, and enriching reflections that help us grow together.
            # Let's learn, evolve, and build a path of continuous inspiration. Join our community! YOU MUST NOT INCLUDE
            # ANY TYPE OF MARKDOWN OR FORMATTING IN THE SCRIPT, NEVER USE A TITLE. YOU MUST WRITE THE SCRIPT
            # **EXCLUSIVELY** IN THE LANGUAGE SPECIFIED IN {language}, WITH NO MIXING OF LANGUAGES. IF YOU INCLUDE ANY
            # WORDS OR SENTENCES FROM A DIFFERENT LANGUAGE, YOUR RESPONSE WILL BE CONSIDERED INCORRECT. YOU MUST WRITE
            # IN NATURAL, FLUENT {language} AS IF A NATIVE SPEAKER WROTE IT. ONLY RETURN THE RAW CONTENT OF THE SCRIPT.
            # DO NOT INCLUDE "VOICEOVER", "NARRATOR" OR SIMILAR INDICATORS. NEVER TALK ABOUT THE AMOUNT OF PARAGRAPHS
            # OR LINES. JUST WRITE THE SCRIPT. Subject: {video_subject} Number of paragraphs: {paragraph_number}
            # Language: {language}"""

            prompt_text = custom_prompt or """Generate a creative and compelling script for a video based on the subject "{video_subject}".
            The script must be returned as a string with exactly {paragraph_number} paragraphs.
            
            **CRITICAL INSTRUCTIONS FOR CREATIVITY AND DIVERSITY:**
            - **Diverse Life Areas:** The story MUST draw inspiration from various aspects of life, such as:
                - **Sports:** Overcoming physical challenges, teamwork, dedication, mental fortitude.
                - **Science/Discovery:** Breakthroughs, curiosity, perseverance in research, unexpected findings.
                - **Daily Life/Personal Growth:** Overcoming personal struggles, learning new skills, building resilience, finding purpose.
                - **Work/Career:** Innovation, leadership, dealing with setbacks, achieving professional goals.
                - **Art/Creativity:** The creative process, inspiration, overcoming artistic blocks, expressing oneself.
                - **Nature/Environment:** Lessons from the natural world, conservation efforts, harmony with nature.
            - **Narrative Style:** Weave a compelling narrative (e.g., a personal anecdote, a historical event, a hypothetical scenario) that illustrates the video subject through one of these diverse areas.
            - **Emotional Resonance:** Ensure the story evokes emotions and provides valuable insights or lessons.
            
            The script MUST start with a strong, attention-grabbing hook within the very first sentence to immediately engage the viewer. This hook should be thought-provoking, controversial, or directly address a common pain point related to the video's subject. Get straight to the point, don't start with unnecessary things like, "welcome to this video".
            
            Obviously, the script should be related to the subject of the video.
            
            Include at the end a paragraph: "If this content has resonated with you, don't miss out on more! Hit that follow button right now for daily doses of inspiring wisdom, empowering insights, and enriching reflections that help us grow together. Let's learn, evolve, and build a path of continuous inspiration. Join our community!"
            
            YOU MUST NOT INCLUDE ANY TYPE OF MARKDOWN OR FORMATTING IN THE SCRIPT, NEVER USE A TITLE. YOU MUST WRITE THE SCRIPT **EXCLUSIVELY** IN THE LANGUAGE SPECIFIED IN {language}, WITH NO MIXING OF LANGUAGES. IF YOU INCLUDE ANY WORDS OR SENTENCES FROM A DIFFERENT LANGUAGE, YOUR RESPONSE WILL BE CONSIDERED INCORRECT. YOU MUST WRITE IN NATURAL, FLUENT {language} AS IF A NATIVE SPEAKER WROTE IT. ONLY RETURN THE RAW CONTENT OF THE SCRIPT. DO NOT INCLUDE "VOICEOVER", "NARRATOR" OR SIMILAR INDICATORS. NEVER TALK ABOUT THE AMOUNT OF PARAGRAPHS OR LINES. JUST WRITE THE SCRIPT.
            
            Subject: {video_subject}
            Number of paragraphs: {paragraph_number}
            Language: {language}
            """

            prompt_template = PromptTemplate(
                input_variables=["video_subject", "paragraph_number", "language"],
                template=prompt_text
            )
            response = self._generate_response(
                prompt_template,
                {"video_subject": video_subject, "paragraph_number": paragraph_number, "language": language}
            )
            if response:
                # Clean the script
                response = response.replace("*", "").replace("#", "")
                response = re.sub(r"\[.*\]", "", response)
                response = re.sub(r"\(.*\)", "", response)
                paragraphs = response.split("\n\n")
                selected_paragraphs = paragraphs[:paragraph_number]
                final_script = "\n\n".join(selected_paragraphs)
                logging.info(
                    f"Generated script with {len(selected_paragraphs)} paragraphs for subject: {video_subject}")
                print(colored(f"Number of paragraphs used: {len(selected_paragraphs)}", "green"))
                return final_script
            return ""
        except Exception as e:
            logging.error(f"Error generating script: {str(e)}")
            print(colored(f"[-] Error: {str(e)}", "red"))
            return ""

    def generate_video_cover_hook(self, video_subject: str, language: str, custom_prompt: str = "") -> str:
        """
        Generate a hook text for a video cover based on a specific topic.
        :param video_subject: The subject of the video.
        :param language: The language or tone for the hook.
        :param custom_prompt: A custom prompt provided by the user (optional).
        :return: The generated hook text for the video cover.
        """
        try:
            prompt_text = custom_prompt or """
                Generate a high-conversion, click-optimized hook text designed for a video thumbnail or cover (akin to a title/caption).
                The hook must employ strong FOMO (Fear Of Missing Out) tactics, create a dramatic knowledge gap, and be structured to capture Attention, build Interest, and drive Desire (AIDA).
                The final text must be brief (under 12 words) and directly reveal the consequence of not knowing the secret.
                Do not include any unnecessary introductions, only focus on the hook text itself.
                YOU MUST NOT INCLUDE ANY TYPE OF MARKDOWN OR FORMATTING IN THE HOOK. NEVER USE A TITLE OR ANY INTRODUCTORY TEXT.
                DO NOT INCLUDE EMOTICONS
                Language: {voice}
                Subject: {video_subject}
            """
            prompt_template = PromptTemplate(
                input_variables=["video_subject", "language"],
                template=prompt_text
            )
            response = self._generate_response(
                prompt_template,
                {"video_subject": video_subject, "voice": language}
            )
            logging.info(f"Generated video cover hook: {response}")
            print(colored(f"Generated Hook: {response}", "cyan"))
            return response.strip()
        except Exception as e:
            logging.error(f"Error generating video cover hook: {str(e)}")
            print(colored(f"[-] Error: {str(e)}", "red"))
            return ""

    def get_search_terms(self, video_subject: str, amount: int, script: str, language: str) -> List[str]:
        """
        Generate a JSON-Array of search terms for stock videos based on the video subject and script.
        :param video_subject: The subject of the video.
        :param amount: The number of search terms to generate.
        :param script: The script of the video.
        :param language: The language for the search terms.
        :return: List of search terms.
        """
        try:
            # prompt_text = """
            #     Generate {amount} search terms for stock videos, depending on the subject of a video.
            #     The search terms must be in {language}.
            #     Subject: {video_subject}
            #     The search terms are to be returned as a JSON-Array of strings.
            #     Each search term should consist of just one relevant word, always add the main subject of the video.
            #     YOU MUST ONLY RETURN THE JSON-ARRAY OF STRINGS.
            #     YOU MUST NOT RETURN ANYTHING ELSE.
            #     YOU MUST NOT RETURN THE SCRIPT.
            #     REMEMBER THE SEARCH TERMS MUST BE IN {language}
            #     The search terms must be related to the subject of the video.
            #     Here is an example of a JSON-Array of strings:
            #     ["search term 1", "search term 2", "search term 3"]
            #     For context, here is the full text:
            #     {script}
            # """

            prompt_text = """
                Generate {amount} search terms for stock videos, depending on the subject of a video.
                The search terms must be in {language}.
                Subject: {video_subject}
                The search terms are to be returned as a JSON-Array of strings.
                Each search term should consist of just one relevant word, always add the main subject of the video.

                **CRITICAL AND STRICT INSTRUCTIONS FOR SEARCH QUERIES:**
                - **Analyze the script for gender and context:** You MUST carefully read the provided `script` to identify the gender (male/female) and the specific role or context of the main character(s).
                - **Generate gender-specific and context-rich terms:** If the script refers to a specific gender (e.g., "Ana," "Jose," "woman entrepreneur," "man climbing"), the search terms MUST strictly combine the gender with the action, role, or emotion.
                  * **Example 1 (Female):** If the script is about "Ana, a young designer facing challenges," search queries MUST be like: "woman designer overcoming challenges", "female entrepreneur mindset", "woman meditating for focus", "female success journey".
                  * **Example 2 (Male):** If the script mentions "Juan, an artist struggling with doubt," search queries MUST be like: "male artist painting", "man frustrated", "man overcoming self-doubt", "male creative process".
                - **Avoid generic terms if gender/context is identified:** If a specific gender and context are present, DO NOT use generic terms like "Mindset", "Transformation", "Success" alone. They MUST be combined with the gender and specific action/role.
                - **Gender-neutral terms:** If the gender is NOT specified in the script, then use neutral terms (e.g., "people working hard," "person meditating"). Focus on the actions and emotions described in the script.

                YOU MUST ONLY RETURN THE JSON-ARRAY OF STRINGS.
                YOU MUST NOT RETURN ANYTHING ELSE.
                YOU MUST NOT RETURN THE SCRIPT.
                REMEMBER THE SEARCH TERMS MUST BE IN {language}
                The search terms must be related to the subject of the video.
                Here is an example of a JSON-Array of strings:
                ["search term 1", "search term 2", "search term 3"]
                For context, here is the full text:
                {script}
            """

            prompt_template = PromptTemplate(
                input_variables=["video_subject", "amount", "language", "script"],
                template=prompt_text
            )
            response = self._generate_response(
                prompt_template,
                {"video_subject": video_subject, "amount": amount, "language": language, "script": script}
            )
            search_terms = []
            try:
                # Clean and parse response
                response = response.replace("```json", "").replace("```", "").replace("\n", "").strip()
                search_terms = json.loads(response)
                if not isinstance(search_terms, list) or not all(isinstance(term, str) for term in search_terms):
                    raise ValueError("Response is not a list of strings.")
                # Clean terms
                search_terms = [term.strip().strip('"').strip("'") for term in search_terms if term.strip()]
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Unformatted response received: {response}. Attempting to clean...")
                print(
                    colored("[*] LangChain/Gemini returned an unformatted response. Attempting to clean...", "yellow"))
                if response.startswith('[') and response.endswith(']'):
                    response = response[1:-1]
                potential_terms = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', response)
                search_terms = [re.sub(r'^["\']|["\']$', '', term.strip()) for term in potential_terms if term.strip()]

            logging.info(f"Generated {len(search_terms)} search terms for subject: {video_subject}")
            print(colored(f"Generated {len(search_terms)} search terms: {', '.join(search_terms)}", "cyan"))
            return search_terms
        except Exception as e:
            logging.error(f"Error generating search terms: {str(e)}")
            print(colored(f"[-] Could not parse response: {str(e)}", "red"))
            return []

    def get_search_terms_shorts(self, book_title: str, num_terms: int, chapter_text: str, language: str) -> List[str]:
        """
        Generate SEO-optimized search terms for the video.

        Args:
            book_title (str): Title of the book.
            num_terms (int): Number of search terms to generate.
            chapter_text (str): Script or text content of the video.
            language (str): Language for the search terms.

        Returns:
            List[str]: List of search terms.
        """
        try:
            # Convert chapter_text to string if it's a list
            if isinstance(chapter_text, list):
                chapter_text = " ".join(chapter_text)

            # Search Terms Prompt
            search_terms_prompt = """
            You are a YouTube SEO expert.
            Generate {num_terms} SEO-optimized search terms for a YouTube Shorts video about the book '{book_title}' with content from this script:
            {chapter_text}

            Follow these VidIQ-inspired guidelines for viralization and maximum reach:
            - Include specific terms from the book title and script.
            - Add 3-5 general terms relevant to the topic (e.g., motivation, productivity for self-help books).
            - Include trending or high-traffic terms related to the content.
            - Ensure terms are concise and relevant for YouTube search.
            - Write entirely in '{language}', matching the tone and cultural nuances.
            - Do NOT use Markdown formatting like **WORD** or *WORD*; use plain text only.
            - Return the terms as a JSON array of strings, e.g., ["term1", "term2", "term3"].

            Provide ONLY the JSON array, without additional text or comments.
            """

            prompt_template = PromptTemplate(
                input_variables=["book_title", "num_terms", "language", "chapter_text"],
                template=search_terms_prompt
            )

            response = self._generate_response(
                prompt_template,
                {"book_title": book_title, "num_terms": num_terms, "language": language, "chapter_text": chapter_text}
            )

            keywords = []
            try:
                # Clean and parse response
                response = response.replace("```json", "").replace("```", "").replace("\n", "").strip()
                keywords = json.loads(response)
                if not isinstance(keywords, list) or not all(isinstance(term, str) for term in keywords):
                    raise ValueError("Response is not a list of strings.")
                # Clean terms
                keywords = [term.strip().strip('"').strip("'") for term in keywords if term.strip()][:num_terms]
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Unformatted response received: {response}. Attempting to clean...")
                print(
                    colored("[*] LangChain/Gemini returned an unformatted response. Attempting to clean...", "yellow"))
                # Improved cleaning for comma-separated strings
                response = response.strip("[]").strip()
                potential_terms = [term.strip() for term in response.split(",") if term.strip()]
                keywords = [re.sub(r'^["\']|["\']$', '', term) for term in potential_terms][:num_terms]

            logging.info(f"Generated {len(keywords)} search terms for subject: {book_title}")
            print(colored(f"Generated {len(keywords)} search terms: {', '.join(keywords)}", "cyan"))
            return keywords

        except Exception as e:
            logging.error(f"Error generating search terms: {str(e)}", exc_info=True)
            return []

    def get_search_terms_mental_coach(self, book_title: str, num_terms: int, chapter_text: str, language: str) -> List[str]:
        """
        Generate SEO-optimized search terms for the video. (Optimized for High-Performance Niche)

        Args:
            book_title (str): Title of the book.
            num_terms (int): Number of search terms to generate.
            chapter_text (str): Script or text content of the video.
            language (str): Language for the search terms.

        Returns:
            List[str]: List of search terms.
        """
        try:
            # Convert chapter_text to string if it's a list
            if isinstance(chapter_text, list):
                chapter_text = " ".join(chapter_text)

            # Search Terms Prompt (Optimized for High-Performance Niche)
            search_terms_prompt = """
            You are a YouTube SEO expert specializing in the **high-performance and mental coaching niche**.
            Generate {num_terms} SEO-optimized search terms for a YouTube Shorts video about the book '{book_title}' with content from this script:
            {chapter_text}

            Follow these high-performance SEO guidelines for maximum reach:
            - **PRIORITIZE long-tail search terms** that reflect viewer intent (e.g., "how to handle soccer pressure", "mental exercises for athletes").
            - Include specific terms from the book title and script.
            - Add 3-5 high-volume, niche-specific terms (e.g., champions mindset, sports psychology, elite athletes).
            - Ensure terms are concise and relevant for YouTube search.
            - Write entirely in '{language}', matching the tone and cultural nuances.
            - Do NOT use Markdown formatting like **WORD** or *WORD*; use plain text only.
            - Return the terms as a JSON array of strings, e.g., ["term1", "term2", "term3"].

            Provide ONLY the JSON array, without additional text or comments.
            """

            # Assuming PromptTemplate is available in the environment
            prompt_template = PromptTemplate(
                input_variables=["book_title", "num_terms", "language", "chapter_text"],
                template=search_terms_prompt
            )

            response = self._generate_response(
                prompt_template,
                {"book_title": book_title, "num_terms": num_terms, "language": language, "chapter_text": chapter_text}
            )

            keywords = []
            try:
                # Clean and parse response
                response = response.replace("```json", "").replace("```", "").replace("\n", "").strip()
                keywords = json.loads(response)
                if not isinstance(keywords, list) or not all(isinstance(term, str) for term in keywords):
                    raise ValueError("Response is not a list of strings.")
                # Clean terms
                keywords = [term.strip().strip('"').strip("'") for term in keywords if term.strip()][:num_terms]
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Unformatted response received: {response}. Attempting to clean...")
                # Note: The 'colored' function reference is removed from this code block
                # as it requires an external library and is not standard Python.
                # print(colored("[*] LangChain/Gemini returned an unformatted response. Attempting to clean...", "yellow"))

                # Improved cleaning for comma-separated strings
                response = response.strip("[]").strip()
                potential_terms = [term.strip() for term in response.split(",") if term.strip()]

                # Assuming 're' is imported if needed for regex substitutions
                # import re
                keywords = [re.sub(r'^["\']|["\']$', '', term) for term in potential_terms][:num_terms]

            logging.info(f"Generated {len(keywords)} search terms for subject: {book_title}")
            # print(colored(f"Generated {len(keywords)} search terms: {', '.join(keywords)}", "cyan"))
            return keywords

        except Exception as e:
            logging.error(f"Error generating search terms: {str(e)}", exc_info=True)
            return []

    def generate_metadata(self, book_title: str, chapter_text: str, language: str) -> Tuple[str, str, List[str], str]:
        """
        Generate metadata for a YouTube video, including title, description, keywords, and playlist title.
        :param video_subject: The subject of the video.
        :param script: The script of the video.
        :param language: The language for the metadata.
        :return: Tuple containing the title, description, keywords, and playlist title.
        """
        try:
            # Generate title
            title_prompt = """
                You are an expert copywriter and a specialist in YouTube Shorts SEO.
                Generate the most compelling and SEO-optimized title for a YouTube Shorts video.

                The video is about a specific concept from the book '{book_title}' and its content is found in the following script:
                {chapter_text}

                Consider the following guidelines based on VidIQ's best practices:
                - The title must generate curiosity and create a curiosity gap.
                - Use emotional hooks like 'The Secret Of', 'The Hidden Truth', 'Beware of this mistake!'
                - It must be concise, impactful, and designed to maximize views and click-through-rate (CTR).
                - Incorporate relevant keywords from the book and the script.
                - The title must be entirely in the language '{language}'.

                Provide **ONLY the title**, without any quotation marks, explanations, or additional phrases.
            """
            title_template = PromptTemplate(
                input_variables=["book_title", "chapter_text", "language"],
                template=title_prompt
            )
            title = self._generate_response(
                title_template,
                {"book_title": book_title, "chapter_text": chapter_text, "language": language}
            ).strip()

            # Generate description
            description_prompt = """
                You are a content marketing and SEO specialist for YouTube.
                Your goal is to craft a powerful and optimized description for a YouTube Shorts video.

                The video is based on the following script:
                {chapter_text}
                and explores ideas from the book {book_title}.

                Consider the following guidelines based on VidIQ's best practices:
                - The description must be concise, and the key message should be in the first lines to capture attention immediately.
                - Integrate keywords naturally, based on the script's content, to maximize visibility in YouTube's search engines.
                - The text should generate reflection and an emotional connection with the viewer.
                - Include relevant hashtags that maximize the video's reach. Use a combination of topic-specific, book-specific, and broader hashtags.

                End the description with a dual call to action (CTA):
                1.  Formulate a question that encourages viewers to comment and share their thoughts.
                2.  Generate a unique and creative phrase that invites viewers to follow your account for more content. This phrase must be new each time.

                Ensure the entire description is in the {language} language.
                Provide **ONLY the complete description**, without any additional explanations.
            """
            description_template = PromptTemplate(
                input_variables=["book_title", "chapter_text", "language"],
                template=description_prompt
            )
            description = self._generate_response(
                description_template,
                {"book_title": book_title, "chapter_text": chapter_text, "language": language}
            ).strip()

            # Generate keywords
            keywords = self.get_search_terms(book_title, 6, chapter_text, language)

            # Generate playlist title
            playlist_title_prompt = """
                You are an expert in YouTube SEO and content strategy.
                Generate a highly-clickable and SEO-optimized playlist title for a video series about the book '{book_title}'.

                The title must be creative, compelling, and designed to maximize its click-through rate. Consider these VidIQ-inspired patterns:
                - **Benefit-driven:** Combine the book's title with a clear benefit or transformation (e.g., "Change Your Life with Atomic Habits").
                - **Action-oriented:** Use action verbs to encourage engagement (e.g., "Master Atomic Habits Today").
                - **Content format:** Explicitly state the format to set expectations (e.g., "Atomic Habits: Full Audiobook").
                - **Emotional hook:** Use a phrase that creates an emotional connection or a sense of urgency (e.g., "The Power of Atomic Habits").

                Ensure the title is entirely in the language specified by '{language}'.
                Provide **ONLY the title**, without any explanations or additional phrases.
            """
            playlist_title_template = PromptTemplate(
                input_variables=["book_title", "language"],
                template=playlist_title_prompt
            )
            playlist_title = self._generate_response(
                playlist_title_template,
                {"book_title": book_title, "language": language}
            ).strip()

            logging.info(f"Generated metadata for subject: {book_title}")
            return title, description, keywords, playlist_title
        except Exception as e:
            logging.error(f"Error generating metadata: {str(e)}")
            print(colored(f"[-] Error: {str(e)}", "red"))
            return "", "", [], ""

    def generate_metadata_shorts(self, book_title: str, chapter_text: str, language: str) -> Tuple[str, str, List[str]]:
        """
        Generate metadata for a YouTube Shorts video, including title, description, and keywords.

        Args:
            book_title (str): Title of the book.
            chapter_text (str): Script or text content of the video.
            language (str): Language for the metadata.

        Returns:
            Tuple[str, str, List[str]]: Title, description, and keywords for the YouTube Shorts video.
        """
        try:
            # Title Prompt
            title_prompt = """
            You are an expert copywriter and YouTube Shorts SEO specialist.
            Generate a compelling, SEO-optimized title for a YouTube Shorts video.

            The video is about a concept from the book '{book_title}' with content from this script:
            {chapter_text}

            Follow these VidIQ-inspired guidelines for viralization and maximum reach:
            - Keep the title under 60 characters for mobile visibility.
            - Start with 1-2 primary keywords from the book and script for searchability.
            - Create a strong curiosity gap using trending phrases like '¡No creerás esto!', '¿Sabías esto?', or '¡Secreto revelado!'.
            - Use emotional power words like 'Transforma', 'Descubre', 'Evita este error', or '¡Cambia tu vida!' to drive clicks.
            - Include 2-3 relevant emojis (e.g., 🔥, 🚀, ❓, 📚) to grab attention and enhance visual appeal.
            - Write entirely in '{language}', matching the tone and cultural nuances.
            - Do NOT use Markdown formatting like **WORD** or *WORD*; use plain text only.

            Provide ONLY the title, without quotation marks or additional text.
            """

            title_template = PromptTemplate(
                input_variables=["book_title", "chapter_text", "language"],
                template=title_prompt
            )

            title = self._generate_response(
                title_template,
                {"book_title": book_title, "chapter_text": chapter_text, "language": language}
            ).strip()

            logging.debug(f"Generated title: {title}")

            # Description Prompt
            description_prompt = """
            You are a YouTube SEO and content marketing specialist.
            Craft a concise, SEO-optimized description for a YouTube Shorts video.

            The video is based on this script:
            {chapter_text}
            and explores ideas from the book '{book_title}'.

            Follow these VidIQ-inspired guidelines for viralization and maximum reach:
            - Start with a gripping first line using primary keywords and 1-2 relevant emojis (e.g., 📚, 🔥) to hook viewers instantly.
            - Keep the description concise (100-150 words) for YouTube Shorts.
            - Integrate keywords naturally from the book and script for search visibility.
            - Create a strong emotional connection or spark deep reflection to resonate with viewers.
            - Include 10-15 hashtags: 7-10 specific (book/author/topic-related) and 3-5 general (e.g., #Shorts, #Motivación, #Inspiración).
            - Use 4-6 relevant emojis: 1-2 at the start, 1-2 within the text, and 1-2 at the end to boost engagement.
            - End with a dual call-to-action:
              1. A thought-provoking question to drive comments and engagement.
              2. A unique, creative phrase to invite viewers to follow for more content (different each time).
            - Write entirely in '{language}', matching the tone and cultural nuances.
            - Do NOT use Markdown formatting like **WORD** or *WORD*; use plain text only.

            Provide ONLY the description, without additional text.
            """

            description_template = PromptTemplate(
                input_variables=["book_title", "chapter_text", "language"],
                template=description_prompt
            )
            description = self._generate_response(
                description_template,
                {"book_title": book_title, "chapter_text": chapter_text, "language": language}
            ).strip()


            # Generate keywords
            keywords = self.get_search_terms_shorts(book_title, 6, chapter_text, language)
            # Add emojis to some keywords for better visibility
            keywords_with_emojis = [f"{kw} 📚" if i % 2 == 0 else kw for i, kw in enumerate(keywords[:4])] + keywords[4:]
            logging.debug(f"Generated keywords: {keywords_with_emojis}")

            logging.info(f"Generated metadata for book: {book_title}")
            return title, description, keywords_with_emojis

        except Exception as e:
            logging.error(f"Error generating metadata: {str(e)}", exc_info=True)
            print(f"[-] Error: {str(e)}")
            return "", "", []

    def generate_metadata_mental_coach(self, book_title: str, chapter_text: str, language: str) -> Tuple[str, str, List[str]]:
        """
        Generate metadata for a YouTube Shorts video, including title, description, and keywords.

        Args:
            book_title (str): Title of the book.
            chapter_text (str): Script or text content of the video.
            language (str): Language for the metadata.

        Returns:
            Tuple[str, str, List[str]]: Title, description, and keywords for the YouTube Shorts video.
        """
        try:
            # Title Prompt (Optimized for High-Performance Niche and CTR)
            title_prompt = """
            You are an expert copywriter and YouTube Shorts SEO specialist, focused on the **high-performance and mental coaching niche**.
            Generate a compelling, SEO-optimized title for a YouTube Shorts video.

            The video is about a concept from the book '{book_title}' with content from this script:
            {chapter_text}

            Follow these high-performance marketing and viralization guidelines for maximum reach (under 60 characters):
            - Start with a **powerful, urgent verb** (Domina, Desbloquea, Programa, Evita) or a primary keyword related to **MENTALIDAD, ÉLITE o RENDIMIENTO**.
            - Create a strong curiosity gap that targets a **pain point or a secret** specific to the audience (e.g., 'La élite no quiere que sepas esto', 'Solo 1% lo logra', '¿Por qué fallas en el minuto 90?').
            - The title MUST be under 60 characters to ensure full visibility on mobile screens.
            - Include 2-3 highly relevant emojis, prioritizing those that signal the niche (🏆, ⚽, 🧠) and urgency (🔥, 🚨).
            - Write entirely in '{language}', matching the intense, motivational tone.
            - Do NOT use Markdown formatting like **WORD** or *WORD*; use plain text only.

            Provide ONLY the title, without quotation marks or additional text.
            """

            title_template = PromptTemplate(
                input_variables=["book_title", "chapter_text", "language"],
                template=title_prompt
            )

            title = self._generate_response(
                title_template,
                {"book_title": book_title, "chapter_text": chapter_text, "language": language}
            ).strip()

            logging.debug(f"Generated title: {title}")

            # Description Prompt (Optimized for SEO and Engagement)
            description_prompt = """
            You are a YouTube SEO and content marketing specialist, specializing in high-performance coaching.
            Craft a concise, SEO-optimized description for a YouTube Shorts video.

            The video is based on this script:
            {chapter_text}
            and explores ideas from the book '{book_title}'.

            Follow these high-performance marketing and viralization guidelines:
            - **First Line SEO Power:** Start with a gripping sentence (under 160 characters) that immediately hooks the viewer and integrates the video's primary keyword, the book title, and 1-2 relevant emojis (e.g., 📚, 🧠).
            - **Body:** Create 2 short paragraphs (total 100-150 words) that expand on the concept, focusing on the **transformation/benefit** the viewer gains (e.g., "Aprende a manejar la presión", "Consigue una mentalidad inquebrantable").
            - **Hashtags Strategy (70/30 Rule):** Include 12-15 hashtags: **#Shorts** (MUST be first), 7-8 highly specific hashtags (e.g., #MentalidadDeCampeon, #Futbol, #DesarrolloPersonal), and 4-5 general hashtags (e.g., #Motivacion, #Crecimiento).
            - **Dual Call-to-Action:**
              1. A thought-provoking question to drive comments (e.g., "¿Qué otro secreto mental crees que usan los élite?").
              2. A closing line to invite continued engagement and following (e.g., "Sigue el canal para desbloquear tu Mente de Campeón 🏆").
            - Write entirely in '{language}', matching the motivational, high-value tone.
            - Do NOT use Markdown formatting like **WORD** or *WORD*; use plain text only.

            Provide ONLY the description, without additional text.
            """

            description_template = PromptTemplate(
                input_variables=["book_title", "chapter_text", "language"],
                template=description_prompt
            )
            description = self._generate_response(
                description_template,
                {"book_title": book_title, "chapter_text": chapter_text, "language": language}
            ).strip()

            # Generate keywords
            # Note: Ensure get_search_terms_shorts prioritizes performance/niche terms.
            keywords = self.get_search_terms_mental_coach(book_title, 6, chapter_text, language)
            # Add emojis to some keywords for better visibility
            keywords_with_emojis = [f"{kw} 📚" if i % 2 == 0 else kw for i, kw in enumerate(keywords[:4])] + keywords[4:]
            logging.debug(f"Generated keywords: {keywords_with_emojis}")

            logging.info(f"Generated metadata for book: {book_title}")
            return title, description, keywords_with_emojis

        except Exception as e:
            logging.error(f"Error generating metadata: {str(e)}", exc_info=True)
            print(f"[-] Error: {str(e)}")
            return "", "", []

    def generate_stories(self, topic: str, number_of_stories: int, duration: int, language: str,
                        custom_prompt: str = "") -> List[Dict[str, Any]]:
        """
        Generate a list of story dictionaries based on the topic, each with a unique ID, title, script, and duration.
        :param topic: The topic of the stories.
        :param number_of_stories: The number of stories to generate.
        :param duration: The duration of each story in seconds.
        :param language: The language for the stories.
        :param custom_prompt: A custom prompt provided by the user (optional).
        :return: List of dictionaries, each containing story_id, title, script, and duration.
        """
        try:
            # prompt_text = custom_prompt or """
            #     Generate a JSON list of {number_of_stories} stories based on the topic "{topic}".
            #     Each story must be designed for a short video with a duration of {duration} seconds.
            #     Each story dictionary in the list must have the following keys:
            #     "story_id": A unique ID string in the format "story_1", "story_2", ..., one for each story.
            #     "title": A concise, engaging title for the story, relevant to the topic, in {language}.
            #     "script": A brief, compelling script for the story, written in natural, fluent {language} as if by a native speaker.
            #               The script should be suitable for a {duration}-second video, avoiding markdown, titles, or any formatting.
            #               Do not include "VOICEOVER", "NARRATOR", or similar indicators. Include a closing sentence encouraging viewers to follow for more content.
            #     "duration": The integer {duration}, representing the story duration in seconds.
            #     The output MUST be a valid JSON list of dictionaries, starting with '[' and ending with ']'.
            #     DO NOT include any explanations, introductions, headers, footers, code block markers (e.g., ```json), or any text outside the JSON structure.
            #     Ensure all content is in {language} with no mixing of languages.
            #     If you cannot generate the requested number of stories, return an empty JSON list [].
            #     Topic: {topic}
            #     Number of stories: {number_of_stories}
            #     Language: {language}
            # """

            prompt_text = custom_prompt or """
                Generate a JSON list of {number_of_stories} stories based on the topic "{topic}".
                Each story must be designed for a short video with a duration of {duration} seconds.

                EACH STORY DICTIONARY IN THE LIST MUST CONTAIN THE FOLLOWING KEYS:
                - "story_id": A unique ID string in the format "story_1", "story_2", ..., one for each story.
                - "title": A concise, engaging title for the story, relevant to the topic, in {language}.
                - "script": A brief, compelling script for the story, written in natural, fluent {language} as if by a native speaker.
                          The script should be suitable for a {duration}-second video, avoiding markdown, titles, or any formatting.
                          Do not include "VOICEOVER", "NARRATOR", or similar indicators.

                          **CRITICAL INSTRUCTIONS FOR THE SCRIPT:**
                          - **Start with an attention-grabbing hook:** Begin the script with a powerful, creative, and intriguing phrase (e.g., "This story will impact you, stay until the end...", "I invite you to watch this reflection...", etc.) to make viewers stay. The hook must be part of the script's first sentence.
                          - **End with a profound reflection:** Conclude the script with a concise yet profound note of reflection related to the topic, inspired by authors like Joe Dispenza, Louise L. Hay, Michael Jordan, Kobe Bryant, Ronaldo, Einstein and so on. This reflection should add depth and encourage thought.

                          Include a closing sentence encouraging viewers to follow for more content.

                - "duration": The integer {duration}, representing the story duration in seconds.

                The output MUST be a valid JSON list of dictionaries, starting with '[' and ending with ']'.
                DO NOT include any explanations, introductions, headers, footers, code block markers (e.g., ```json), or any text outside the JSON structure.
                Ensure all content is in {language} with no mixing of languages.
                If you cannot generate the requested number of stories, return an empty JSON list [].
                Topic: {topic}
                Number of stories: {number_of_stories}
                Language: {language}
            """

            prompt_template = PromptTemplate(
                input_variables=["topic", "number_of_stories", "duration", "language"],
                template=prompt_text
            )
            response = self._generate_response(
                prompt_template,
                {"topic": topic, "number_of_stories": number_of_stories, "duration": duration, "language": language}
            )
            logging.debug(f"Raw response for stories: {response}")
            try:
                # Clean and parse JSON response
                cleaned_response = re.sub(r'^```json\s*', '', response.strip(), flags=re.IGNORECASE)
                cleaned_response = re.sub(r'```\s*$', '', cleaned_response).strip()
                if not cleaned_response:
                    raise ValueError("Empty response after cleaning.")
                stories = json.loads(cleaned_response)

                # If the response is a single dictionary, convert it into a list.
                if isinstance(stories, dict):
                    stories = [stories]

                # Loop to ensure each story in the list has the 'duration' key.
                # This fixes the validation problem if the model forgets to include it.
                validated_stories = []
                for story in stories:
                    if "duration" not in story:
                        story["duration"] = duration
                        logging.warning(f"Missing 'duration' key in story, it was added automatically: {story.get('story_id', 'N/A')}")
                    validated_stories.append(story)


                if not isinstance(validated_stories, list):
                    raise ValueError("Response is not a JSON list.")
                if not all(isinstance(story, dict) for story in validated_stories):
                    raise ValueError("Response contains non-dictionary elements.")
                for story in validated_stories:
                    if not all(key in story for key in ["story_id", "title", "script", "duration"]):
                        raise ValueError(f"Story missing required keys: {story}")
                    if not isinstance(story["story_id"], str) or not isinstance(story["title"], str) or \
                            not isinstance(story["script"], str) or not isinstance(story["duration"], int):
                        raise ValueError(f"Invalid data types in story: {story}")
                logging.info(f"Generated {len(validated_stories)} stories for topic: {topic}")
                print(colored(f"Generated {len(validated_stories)} stories for topic: {topic}", "cyan"))
                for story in validated_stories:
                    logging.debug(f"Story {story['story_id']}: Title={story['title']}, Duration={story['duration']}")
                return validated_stories
            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"Invalid response for stories: {str(e)}, Raw response: {response}")
                print(colored(f"[-] Invalid response: {str(e)}", "red"))
                return []
        except Exception as e:
            logging.error(f"Error generating stories: {str(e)}, Raw response: {response}")
            print(colored(f"[-] Error: {str(e)}", "red"))
            return []

    def generate_intro_and_reflection(self, script: str, language: str, video_subject: Optional[str] = None) -> Tuple[str, str]:
        """
        Generates a persuasive introductory hook and a reflective conclusion for a video script.
        The generated texts are designed to be compelling and encourage viewer interaction.

        Args:
            script (str): The main script or body of the video content.
            language (str): The language for the generated texts (e.g., 'Spanish', 'English').
            video_subject (Optional[str]): A brief subject of the video for better context.

        Returns:
            Tuple[str, str]: A tuple containing the introduction and the reflection texts.
        """
        try:
            # Generate a persuasive introduction
            intro_prompt = """
            You are an expert content marketer. Your task is to create a short, highly persuasive introduction
            for a video to grab the viewer's attention in the first few seconds. The goal is to generate intrigue,
            make them want to watch the entire video, and include a call to action to like the video and stay until the end.
            The content is: "{script}". The subject is "{video_subject}". Ensure the introduction is in the language 
            specified by "{language}". Omit any formatting like asterisks, bolding, or any special characters that are outside 
            the context of normal text, as they interfere with the voice-over. Provide only the introduction text.
            """
            intro_template = PromptTemplate(
                input_variables=["script", "language", "video_subject"],
                template=intro_prompt
            )

            introduction = self._generate_response(
                intro_template,
                {"script": script, "language": language, "video_subject": video_subject}
            ).strip()

            # Generate a persuasive reflection with clear calls to action
            reflection_prompt = """
            You are a motivational influencer and content creator. Your task is to write a profound and reflective 
            conclusion for a video. Summarize the key message and encourage the viewer to take action.
            End with a powerful call to action that invites the audience to share the video with their friends
            to help more people, to leave their opinions in the comments, and to follow the channel so they don't 
            miss more relevant content. The content is: "{script}". The subject is "{video_subject}".
            Ensure the reflection is in the language specified by "{language}". Omit any formatting like asterisks, 
            bolding, or any special characters that are outside 
            the context of normal text, as they interfere with the voice-over.
            Provide only the reflection text.
            """
            reflection_template = PromptTemplate(
                input_variables=["script", "language", "video_subject"],
                template=reflection_prompt
            )

            reflection = self._generate_response(
                reflection_template,
                {"script": script, "language": language, "video_subject": video_subject}
            ).strip()

            logging.info("Generated intro and reflection text successfully.")
            return introduction, reflection

        except Exception as e:
            logging.error(f"Error generating introduction and reflection: {e}")
            raise

    def generate_comment_reply(self, comment_text: str, video_description: str, language: str = "Spanish") -> str:
        """
        Generate a highly engaging, context-aware reply to a YouTube comment using Gemini AI.
        The reply should feel personal, respectful, and encourage further interaction, e.g., likes, shares, or replies.

        Args:
            comment_text (str): The text of the user's comment.
            video_description (str): Description of the video for context to craft a more relevant reply.
            language (str): Language for the generated reply (default is 'Spanish').

        Returns:
            str: AI-generated reply text ready to post as a comment.
        """
        try:
            # Powerful prompt template for replying to a comment
            reply_prompt = f"""
            You are a highly skilled community manager and motivational content creator.
            Your task is to craft a thoughtful, friendly, and highly engaging reply to the following user comment:
            "{comment_text}"
    
            The video this comment belongs to has the following description:
            "{video_description}"
    
            Generate a reply that:
            1. Acknowledges and respects the user's comment.
            2. Provides meaningful engagement or insight related to the video topic.
            3. Encourages interaction (like replying back, subscribing, or sharing).
            4. Feels natural and human, not robotic.
            5. Is written entirely in {language}.
            6. Avoids any special formatting like asterisks, bold, or emojis.
    
            Provide only the reply text without any extra explanation.
            """

            reply_template = PromptTemplate(
                input_variables=["comment_text", "video_description", "language"],
                template=reply_prompt
            )

            # Call Gemini API via your internal _generate_response or content generator
            # Pass the input variables to _generate_response
            reply_text = self._generate_response(
                reply_template,
                {
                    "comment_text": comment_text,
                    "video_description": video_description,
                    "language": language
                }
            ).strip()

            logging.info("Generated comment reply successfully.")
            return reply_text

        except Exception as e:
            logging.error(f"Error generating reply for comment '{comment_text}': {e}")
            return "Thank you for your comment!"  # fallback generic reply


def list_google_models():
    """
    Lists available models from Google Generative AI.
    """
    try:
        for model in genai.list_models():
            logging.info(f"Model Name: {model.name}")
            logging.info(f"Display Name: {model.display_name}")
            logging.info(f"Description: {model.description}")
            logging.info(f"Supported Generation Methods: {model.supported_generation_methods}")
            logging.info("-" * 20)
    except Exception as e:
        logging.error(f"Error listing Google models: {str(e)}")
        print(colored(f"[-] Error: {str(e)}", "red"))


if __name__ == "__main__":
    """
    Main function to demonstrate the content generation capabilities.
    """
    try:

        # --- Constants for Directory Paths ---
        BASE_DIR = Path(__file__).resolve().parent.parent
        TEMP_DIR = BASE_DIR / "temp"

        # Initialize content generator
        generator = ContentGenerator()

        # Example parameters
        topic = "A majestic eagle soaring over a snow-capped mountain range at sunrise"
        num_images = 3
        model_name = "PicLumen Art V1"
        lang = "en"
        paragraphs = 3
        voice = "en"
        voice_gemini_studio = "en"
        num_stories = 2
        story_duration = 30

        # Generate image prompts
        image_list_json = generator.generate_images_prompt(topic, num_images, model_name, lang)
        if image_list_json:
            logging.info(f"Image prompts JSON: {image_list_json}")

        # Generate script
        script = generator.generate_script(topic, paragraphs, voice, lang)
        if script:
            logging.info(f"Generated script: {script}")

        # Generate video cover hook
        hook = generator.generate_video_cover_hook(topic, voice)
        if hook:
            logging.info(f"Generated hook: {hook}")

        # Generate metadata
        title, description, keywords = generator.generate_metadata(topic, script, lang)
        if title and description and keywords:
            logging.info(f"Generated metadata - Title: {title}, Description: {description}, Keywords: {keywords}")

        # Generate stories
        stories = generator.generate_stories(topic, num_stories, story_duration, lang)
        if stories:
            logging.info(f"Generated {len(stories)} stories")
            for story in stories:
                logging.info(f"Story {story['story_id']}: {story['title']}: {story['script']}: {story['duration']}")


        # --- Demonstrate TTS generation ---
        sample_script = """
        La creencia errónea ha permanecido como parte importante de la  
        enseñanza que el ser humano ha recibido a través de la historia,   
        lamentablemente como un fenómeno universal. La creencia no   
        está basada en la verdad, sino en suposiciones y falsedades, pero  
        el ser humano ha tomado muchas creencias como verdades y las ha establecido 
        como base de su conocimiento en diversas áreas de su existencia.  
        Por ello, la humanidad ha progresado tan lentamente en muchas ramas  
        importantes del saber, de allí los resultados deficientes, contradictorios,  
        mediocres, cuando no francamente malos y equivocados, que como regla,  
        ha obtenido en muchas áreas. 
        """
        sample_voice = "en_us_001"
        tone_instruction = "Read aloud in a warm, welcoming tone, in Spanish Mexico:"
        temp_dir = Path(__file__).resolve().parent.parent / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        tts_output_path = str(temp_dir / f"tts_sample_{uuid.uuid4()}.wav")
        tts_result = generator.generate_tts(
            script=sample_script,
            voice=sample_voice,
            output_path=tts_output_path,
            tone_instruction=tone_instruction
        )

        if tts_result:
            logging.info(f"TTS demonstration successful: Audio saved at {tts_result}")
            print(colored(f"[+] TTS demonstration successful: Audio saved at {tts_result}", "green"))
        else:
            logging.error("TTS demonstration failed.")
            print(colored("[-] TTS demonstration failed.", "red"))

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        print(colored(f"[-] Error in main execution: {str(e)}", "red"))