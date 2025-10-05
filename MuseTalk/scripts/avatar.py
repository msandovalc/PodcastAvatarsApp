# MuseTalk/scripts/avatar.py
"""
Avatar class for MuseTalk project.

Handles:
- Initialization and preparation of avatar assets (video frames, masks, landmarks)
- Loading existing avatars
- Saving avatar information and preprocessed data
"""

import os, sys, glob, copy, pickle, shutil, threading, time, logging
import torch, cv2, numpy as np
from tqdm import tqdm

from .config import CONFIG
from .utils_io import osmakedirs, video2imgs
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import datagen

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@torch.no_grad()
class Avatar:
    """
    Avatar class to handle video/image processing for a single avatar.

    Args:
        args: Namespace object with configuration parameters (expects 'version' attribute)
        avatar_id (str): Unique ID for the avatar
        video_path (str or Path): Path to source video or image folder
        bbox_shift (float): Bounding box shift parameter for landmark extraction
        batch_size (int): Batch size for frame processing
        preparation (bool): If True, prepares avatar materials, else loads existing
    """

    def __init__(self, args, avatar_id, video_path, bbox_shift, batch_size, preparation):
        try:
            self.avatar_id = avatar_id
            self.video_path = video_path
            self.bbox_shift = bbox_shift
            self.args = args
            self.batch_size = batch_size
            self.preparation = preparation
            self.idx = 0

            # Paths
            if args.version == "v15":
                self.base_path = f"{CONFIG.results_dir}/{args.version}/avatars/{avatar_id}"
            else:
                self.base_path = f"{CONFIG.results_dir}/avatars/{avatar_id}"

            self.avatar_path = self.base_path
            self.full_imgs_path = f"{self.avatar_path}/full_imgs"
            self.coords_path = f"{self.avatar_path}/coords.pkl"
            self.latents_out_path = f"{self.avatar_path}/latents.pt"
            self.video_out_path = f"{self.avatar_path}/vid_output/"
            self.mask_out_path = f"{self.avatar_path}/mask"
            self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
            self.avatar_info_path = f"{self.avatar_path}/avatar_info.json"
            self.avatar_info = {
                "avatar_id": avatar_id,
                "video_path": video_path,
                "bbox_shift": bbox_shift,
                "version": args.version
            }

            logging.info(f"[INFO] Initializing avatar: {avatar_id}")
            self.init()

        except Exception as e:
            logging.error(f"[ERROR] Failed to initialize Avatar {avatar_id}: {e}")
            raise

    def init(self):
        """Initialize avatar: prepare materials or load existing data."""
        try:
            if self.preparation:
                if os.path.exists(self.avatar_path):
                    response = input(f"{self.avatar_id} exists. Re-create it? (y/n) ")
                    if response.lower() == "y":
                        shutil.rmtree(self.avatar_path)
                        osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                        self.prepare_material()
                    else:
                        self.load_existing()
                else:
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
            else:
                if not os.path.exists(self.avatar_path):
                    logging.error(f"{self.avatar_id} does not exist. Set preparation=True.")
                    sys.exit()
                self.load_existing()

            logging.info(f"[INFO] Avatar {self.avatar_id} initialized successfully.")

        except Exception as e:
            logging.error(f"[ERROR] Failed during avatar init: {e}")
            raise

    def load_existing(self):
        """Load preprocessed avatar data from disk."""
        try:
            import json
            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)
            self.input_latent_list_cycle = torch.load(self.latents_out_path)

            with open(self.coords_path, 'rb') as f:
                self.coord_list_cycle = pickle.load(f)

            input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')),
                                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.frame_list_cycle = read_imgs(input_img_list)

            with open(self.mask_coords_path, 'rb') as f:
                self.mask_coords_list_cycle = pickle.load(f)

            input_mask_list = sorted(glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]')),
                                     key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.mask_list_cycle = read_imgs(input_mask_list)

            logging.info(f"[INFO] Loaded existing avatar data: {self.avatar_id}")

        except Exception as e:
            logging.error(f"[ERROR] Failed to load existing avatar {self.avatar_id}: {e}")
            raise

    def prepare_material(self):
        """
        Prepare all the materials required for avatar animation.

        This method performs the following steps:
        1. Saves avatar metadata as JSON.
        2. Extracts frames from a video or copies existing PNG images.
        3. Extracts facial landmarks and bounding boxes.
        4. Applies extra margin (for v15 version).
        5. Generates latent representations of cropped faces.
        6. Creates and saves facial masks and bounding boxes.
        7. Saves all intermediate data (coords, masks, latents).

        Raises:
            Exception: If any step fails, the error will be logged and re-raised.
        """
        try:
            logging.info(f"[INFO] Preparing materials for avatar: {self.avatar_id}")

            # --- Step 1: Save avatar metadata ---
            import json
            with open(self.avatar_info_path, "w") as f:
                json.dump(self.avatar_info, f)
            logging.info(f"[OK] Saved avatar info to {self.avatar_info_path}")

            # --- Step 2: Extract or copy images ---
            if os.path.isfile(self.video_path):
                logging.info(f"[INFO] Extracting frames from video: {self.video_path}")
                video2imgs(self.video_path, self.full_imgs_path, ext='png')
            else:
                logging.info(f"[INFO] Copying existing PNG images from folder: {self.video_path}")
                files = sorted([f for f in os.listdir(self.video_path) if f.endswith('.png')])
                for f in files:
                    shutil.copyfile(os.path.join(self.video_path, f),
                                    os.path.join(self.full_imgs_path, f))

            # --- Step 3: Extract landmarks and bounding boxes ---
            logging.info("[INFO] Extracting facial landmarks and bounding boxes...")
            input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)

            # --- Initialize storage lists ---
            input_latent_list = []
            coord_placeholder = (0.0, 0.0, 0.0, 0.0)

            # --- Step 4: Process each frame ---
            idx = -1
            for bbox, frame in zip(coord_list, frame_list):
                idx += 1
                if bbox == coord_placeholder:
                    continue

                x1, y1, x2, y2 = bbox

                # Apply extra margin if version v15
                if hasattr(self.args, "version") and self.args.version == "v15":
                    if hasattr(self.args, "extra_margin"):
                        y2 = y2 + self.args.extra_margin
                        y2 = min(y2, frame.shape[0])
                        coord_list[idx] = [x1, y1, x2, y2]

                # Crop and resize frame
                crop_frame = frame[y1:y2, x1:x2]
                resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)

                # Encode cropped frame to latent space
                latents = vae.get_latents_for_unet(resized_crop_frame)
                input_latent_list.append(latents)

            # --- Step 5: Create cycles for animation ---
            self.frame_list_cycle = frame_list + frame_list[::-1]
            self.coord_list_cycle = coord_list + coord_list[::-1]
            self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
            self.mask_coords_list_cycle = []
            self.mask_list_cycle = []

            logging.info(f"[OK] Processed {len(frame_list)} frames successfully.")

            # --- Step 6: Generate and save masks ---
            logging.info("[INFO] Generating facial masks...")
            for i, frame in enumerate(tqdm(self.frame_list_cycle, desc="Generating masks")):
                cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)

                x1, y1, x2, y2 = self.coord_list_cycle[i]
                if hasattr(self.args, "version") and self.args.version == "v15":
                    mode = getattr(self.args, "parsing_mode", "raw")
                else:
                    mode = "raw"

                mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)

                cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
                self.mask_coords_list_cycle.append(crop_box)
                self.mask_list_cycle.append(mask)

            # --- Step 7: Save data for reuse ---
            logging.info("[INFO] Saving coordinates, masks and latents...")

            with open(self.mask_coords_path, 'wb') as f:
                pickle.dump(self.mask_coords_list_cycle, f)

            with open(self.coords_path, 'wb') as f:
                pickle.dump(self.coord_list_cycle, f)

            torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

            logging.info(f"[SUCCESS] Materials prepared successfully for avatar: {self.avatar_id}")

        except Exception as e:
            logging.critical(f"[CRITICAL] Failed to prepare materials for {self.avatar_id}: {e}", exc_info=True)
            raise

