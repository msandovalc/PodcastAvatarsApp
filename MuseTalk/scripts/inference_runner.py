from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from MuseTalk.scripts.avatar import Avatar
from MuseTalk.scripts.config import MAX_WORKERS, MAX_CLIP_DURATION
from MuseTalk.scripts.exceptions import InferenceError


class MuseTalkRunner:
    """Coordinates avatar inference and concurrent video processing."""

    def __init__(self, consider_duration: bool = False):
        self.consider_duration = consider_duration
        self.results_list = []

    def process_single_video(self, video_path: Path, clip_duration: float = None, max_duration: float = None):
        """Process a single video (dummy logic)."""
        print(f"[INFO] Processing video: {video_path}")
        return {"output_path": str(video_path), "status": "success"}

    def process_all(self, video_paths: list, max_clip_duration: float = MAX_CLIP_DURATION):
        """Process all videos concurrently."""
        futures_with_indices = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for i, video_path in enumerate(video_paths):
                clip_duration_to_process = min(10, max_clip_duration) if self.consider_duration else None
                future = executor.submit(
                    self.process_single_video, video_path, clip_duration_to_process, None
                )
                futures_with_indices.append((i, future))

            for i, future in futures_with_indices:
                try:
                    result = future.result()
                    self.results_list.append(result)
                except Exception as e:
                    raise InferenceError(f"Error processing video {i}: {e}")

        return [r["output_path"] for r in self.results_list]

    def build_avatar_video(self, avatar_id: str):
        """Builds the avatar video from processed frames."""
        avatar = Avatar(avatar_id)
        avatar.process_frames()
        avatar.build_video()
