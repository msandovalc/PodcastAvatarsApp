import torch
import clip
from transformers import pipeline
import logging
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from .logger_config import setup_logger

# --- Configuration ---
logger = setup_logger(__name__)

# Suppress Transformers warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class ContentAnalyzer:
    """Analyzes video content to score segments for relevance."""

    def __init__(self, use_cuda=True, max_workers=1):
        """
        Initialize the content analyzer with CLIP and sentiment analysis models.

        Args:
            use_cuda (bool): Whether to use CUDA for CLIP processing.
            max_workers (int): Number of threads for parallel frame processing.

        Raises:
            Exception: If model loading fails.
        """
        # Setup logging
        self.logger = logger
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.max_workers = max_workers
        try:
            self.logger.debug("Loading CLIP model on %s", self.device)
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.logger.info("CLIP model loaded on %s", self.device)
            self.logger.debug("Loading sentiment analysis model")
            # self.sentiment_analyzer = pipeline(
            #     "sentiment-analysis",
            #     model="distilbert-base-uncased-finetuned-sst-2-english",
            #     revision="af0f99b",
            #     device=0 if self.device == "cuda" else -1
            # )
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="pysentimiento/robertuito-sentiment-analysis",  # Modelo para español
                device=0 if self.device == "cuda" else -1
            )
            self.logger.info("Sentiment analysis model loaded")
            self.frame_cache = {}
        except Exception as e:
            self.logger.error("Failed to load models: %s", e, exc_info=True)
            raise

    def analyze_segment(self, video_processor, segment, use_ffmpeg_frames=False):
        """
        Analyze a video segment by combining frame and text relevance.

        Args:
            video_processor (VideoProcessor): Video processor instance.
            segment (dict): Segment with start time and text.
            use_ffmpeg_frames (bool): Use FFmpeg for frame extraction instead of OpenCV.

        Returns:
            tuple: (similarity, sentiment_score) or (None, None) if analysis fails.

        Raises:
            Exception: If analysis fails (caught externally).
        """
        try:
            start_time = segment["start"]
            text = segment["text"]

            # Check frame cache
            cache_key = (start_time, use_ffmpeg_frames)
            if cache_key in self.frame_cache:
                frame = self.frame_cache[cache_key]
                self.logger.debug("Using cached frame at %ss", start_time)
            else:
                if use_ffmpeg_frames:
                    frame = video_processor.get_frame_ffmpeg(start_time)
                else:
                    frame = video_processor.get_frame(start_time)
                self.frame_cache[cache_key] = frame

            self.logger.debug("Preprocessing frame at %ss", start_time)
            image = self.preprocess(Image.fromarray(frame)).unsqueeze(0).to(self.device)
            text_tokens = clip.tokenize([text]).to(self.device)

            self.logger.debug("Computing CLIP similarity for segment at %ss", start_time)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)
                text_features = self.clip_model.encode_text(text_tokens)
                similarity = (image_features @ text_features.T).item()

            return similarity, None

        except Exception as e:
            self.logger.error("Error analyzing segment at %ss: %s", start_time, e, exc_info=True)
            return None, None

    def analyze_segments_batch(self, video_processor, segments, use_ffmpeg_frames=False,
                               save_path=os.path.normpath("temp/analyses/analysis_temp.json")):
        """
        Analyze multiple segments in batch for sentiment analysis efficiency.

        Args:
            video_processor (VideoProcessor): Video processor instance.
            segments (list): List of segments with start time and text.
            use_ffmpeg_frames (bool): Use FFmpeg for frame extraction instead of OpenCV.
            save_path (str): Path to save analysis scores.

        Returns:
            list: List of scores for each segment.
        """
        try:
            self.logger.debug("Starting batch sentiment analysis")
            texts = [segment["text"] for segment in segments]
            batch_size = 64
            sentiments = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                self.logger.debug("Processing sentiment batch %s", i // batch_size + 1)
                sentiments.extend(self.sentiment_analyzer(batch))

            scores = []
            self.logger.debug("Starting frame analysis")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self.analyze_segment, video_processor, segment, use_ffmpeg_frames)
                    for segment in segments
                ]
                results = [future.result() for future in tqdm(futures, desc="Analyzing segments", unit="segment")]

            for i, (similarity, _) in enumerate(results):
                try:
                    if similarity is None:
                        scores.append(None)
                        self.logger.debug("Segment at %ss failed analysis", segments[i]["start"])
                        continue

                    sentiment = sentiments[i]
                    sentiment_score = sentiment["score"] if sentiment["label"] == "POSITIVE" else -sentiment["score"]

                    total_score = similarity + sentiment_score
                    scores.append(total_score)
                    self.logger.debug(
                        "Segment at %ss scored: %s (similarity: %s, sentiment: %s)",
                        segments[i]["start"], total_score, similarity, sentiment_score)
                except Exception as e:
                    self.logger.error("Error combining scores for segment at %ss: %s",
                                      segments[i]["start"], e, exc_info=True)
                    scores.append(None)

            self.logger.debug("Saving analysis scores to: %s", save_path)
            analysis_data = {
                "video_path": video_processor.video_path,
                "segments": [
                    {"start": segment["start"], "text": segment["text"], "score": score}
                    for segment, score in zip(segments, scores)
                ]
            }
            Path(save_path).parent.mkdir(exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(analysis_data, f, indent=4)
            self.logger.info("Analysis scores saved to: %s", save_path)

            self.frame_cache.clear()
            return scores
        except Exception as e:
            self.logger.error("Error in batch analysis: %s", e, exc_info=True)
            raise