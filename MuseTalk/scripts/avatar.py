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
        Prepare avatar materials: extract frames, copy images, extract landmarks and bounding boxes.
        """
        try:
            logging.info(f"[INFO] Preparing materials for avatar: {self.avatar_id}")
            import json
            with open(self.avatar_info_path, "w") as f:
                json.dump(self.avatar_info, f)

            # Video/images
            if os.path.isfile(self.video_path):
                video2imgs(self.video_path, self.full_imgs_path)
            else:
                files = sorted([f for f in os.listdir(self.video_path) if f.endswith('.png')])
                for f in files:
                    shutil.copyfile(os.path.join(self.video_path, f), os.path.join(self.full_imgs_path, f))

            # Landmarks
            coord_list, frame_list = get_landmark_and_bbox(
                sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))),
                self.bbox_shift
            )
            self.input_latent_list_cycle = []
            self.frame_list_cycle = frame_list + frame_list[::-1]
            self.coord_list_cycle = coord_list + coord_list[::-1]
            self.mask_list_cycle = []
            self.mask_coords_list_cycle = []

            logging.info(f"[INFO] Materials prepared for avatar: {self.avatar_id}")

        except Exception as e:
            logging.error(f"[ERROR] Failed to prepare materials for {self.avatar_id}: {e}")
            raise
