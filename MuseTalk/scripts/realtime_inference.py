import argparse
import os
from typing import List, Dict, Any

from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
from transformers import WhisperModel

from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor

import shutil
import threading
import queue
import time
import subprocess
from pathlib import Path
import types

import logging
from .avatar import Avatar
from .utils_ffmpeg import FFmpegUtils
from .exceptions import ModelNotFoundError, InferenceError
# from .models import load_all_model, AudioProcessor, WhisperModel
from .config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Get the project root (adjust according to your repo structure)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

FFMPEG_PATH = str(PROJECT_ROOT / "ffmpeg-4.4-amd64-static")
GPU_ID = 0
VAE_TYPE = "sd-vae"
UNET_CONFIG = str(PROJECT_ROOT / "models" / "musetalk" / "musetalk.json")
UNET_MODEL_PATH = str(PROJECT_ROOT / "models" / "musetalk" / "pytorch_model.bin")
WHISPER_DIR = str(PROJECT_ROOT / "models" / "whisper")
INFERENCE_CONFIG_PATH = str(PROJECT_ROOT / "configs" / "inference" / "realtime.yaml")
RESULTS_PATH = str(PROJECT_ROOT / "results")

# --- Initialize Global Objects (Consider managing these within a class) ---
args = None
vae = None
fp = None
audio_processor = None
device = None
weight_dtype = None
whisper = None
pe = None
unet = None
timesteps = None


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break


def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None


@torch.no_grad()
class Avatar:
    def __init__(self,  args, avatar_id, video_path, bbox_shift, batch_size, preparation):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.args = args

        # 根据版本设置不同的基础路径
        if self.args.version == "v15":
            self.base_path = f"./MuseTalk/results/{self.args.version}/avatars/{avatar_id}"
        else:  # v1
            self.base_path = f"./MuseTalk/results/avatars/{avatar_id}"
            
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": self.args.version
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        self.init()

    def init(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                response = input(f"{self.avatar_id} exists, Do you want to re-create it ? (y/n)")
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    self.input_latent_list_cycle = torch.load(self.latents_out_path)
                    with open(self.coords_path, 'rb') as f:
                        self.coord_list_cycle = pickle.load(f)
                    input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.frame_list_cycle = read_imgs(input_img_list)
                    with open(self.mask_coords_path, 'rb') as f:
                        self.mask_coords_list_cycle = pickle.load(f)
                    input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.mask_list_cycle = read_imgs(input_mask_list)
            else:
                print("*********************************")
                print(f"  creating avator: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} does not exist, you should set preparation to True")
                sys.exit()

            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)

            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                response = input(f" 【bbox_shift】 is changed, you need to re-create it ! (c/continue)")
                if response.lower() == "c":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    sys.exit()
            else:
                self.input_latent_list_cycle = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_imgs(input_img_list)
                with open(self.mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_imgs(input_mask_list)

    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if self.args.version == "v15":
                y2 = y2 + self.args.extra_margin
                y2 = min(y2, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]  # 更新coord_list中的bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)

            x1, y1, x2, y2 = self.coord_list_cycle[i]
            if self.args.version == "v15":
                mode = self.args.parsing_mode
            else:
                mode = "raw"
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)

            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)

        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

    def process_frames(self, res_frame_queue, video_len, skip_save_images):
        print(video_len)
        while True:
            if self.idx >= video_len - 1:
                break
            try:
                start = time.time()
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except:
                continue
            mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % (len(self.mask_coords_list_cycle))]
            combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)

            if skip_save_images is False:
                cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png", combine_frame)
            self.idx = self.idx + 1

    @torch.no_grad()
    def inference(self, audio_path, out_vid_name, fps, skip_save_images) -> str:
        os.makedirs(self.avatar_path + '/tmp', exist_ok=True)
        print("start inference")
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        # Extract audio features
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path, weight_dtype=weight_dtype)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            weight_dtype,
            whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )
        print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue()
        self.idx = 0
        # Create a sub-thread and start it
        process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num, skip_save_images))
        process_thread.start()

        gen = datagen(whisper_chunks,
                     self.input_latent_list_cycle,
                     self.batch_size)
        start_time = time.time()
        res_frame_list = []

        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            audio_feature_batch = pe(whisper_batch.to(device))
            latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)

            pred_latents = unet.model(latent_batch,
                                    timesteps,
                                    encoder_hidden_states=audio_feature_batch).sample
            pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        # Close the queue and sub-thread after all tasks are completed
        process_thread.join()

        if args.skip_save_images is True:
            print('Total process time of {} frames without saving images = {}s'.format(
                video_num,
                time.time() - start_time))
        else:
            print('Total process time of {} frames including saving images = {}s'.format(
                video_num,
                time.time() - start_time))

        if out_vid_name is not None and args.skip_save_images is False:

            cmd_img2video = ""
            cmd_combine_audio = ""

            output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")

            print(f"self.video_out_path: {self.video_out_path}")
            print(f"out_vid_name: {out_vid_name}")
            print(f"output_vid: {output_vid}")
            print(f"device: {device}")

            if device == "cuda":
                print(f"[INFO] Using GPU")

                cmd_img2video = (
                    f"ffmpeg -y -v warning -hwaccel cuda "
                    f"-r {fps} -f image2 "
                    f"-i \"{self.avatar_path}/tmp/%08d.png\" "
                    f"-vcodec h264_nvenc -pix_fmt yuv420p -crf 18 "
                    f"\"{self.avatar_path}/temp.mp4\""
                )

                cmd_combine_audio = (
                    f"ffmpeg -y -v warning -hwaccel cuda "
                    f"-i \"{audio_path}\" "
                    f"-i {self.avatar_path}/temp.mp4 "
                    f"-c:v copy -c:a aac -b:a 192k "
                    f"{output_vid}"
                )

            else:
                cmd_img2video = (
                    f"ffmpeg -y -v warning "
                    f"-r {fps} -f image2 "
                    f"-i \"{self.avatar_path}/tmp/%08d.png\" "
                    f"-vcodec libx264 -vf format=yuv420p -crf 18 "
                    f"\"{self.avatar_path}/temp.mp4\""
                )

                cmd_combine_audio = (
                    f"ffmpeg -y -v warning "
                    f"-i \"{audio_path}\" "
                    f"-i \"{self.avatar_path}/temp.mp4\" "
                    f"\"{output_vid}\""
                )

            # optional
            print(cmd_img2video)
            os.system(cmd_img2video)

            print(cmd_combine_audio)
            os.system(cmd_combine_audio)

            os.remove(f"{self.avatar_path}/temp.mp4")
            shutil.rmtree(f"{self.avatar_path}/tmp")
            print(f"result is save to {output_vid}")

            return output_vid

        print("\n")


def musetalk_inference_reloaded(
        version=CONFIG.version,
        gpu_id=CONFIG.gpu_id,
        batch_size=CONFIG.batch_size,
        fps=CONFIG.fps,
        ffmpeg_path=CONFIG.ffmpeg_path,
        unet_config=CONFIG.unet_config,
        unet_model_path=CONFIG.unet_model_path,
        whisper_dir=CONFIG.whisper_dir,
        skip_save_images=CONFIG.skip_save_images
):
    """
    Run MuseTalk inference on all avatars and audio clips defined in the inference configuration.

    Args:
        version (str): MuseTalk version to use (e.g., 'v15').
        gpu_id (int): GPU device ID to use for processing.
        batch_size (int): Batch size for frame processing.
        fps (int): Frames per second for output video.
        ffmpeg_path (str): Path to the FFmpeg executable.
        unet_config (str): Path to UNet configuration file.
        unet_model_path (str): Path to the UNet model weights.
        whisper_dir (str): Path to Whisper feature extractor / model.
        skip_save_images (bool): If True, skip saving intermediate images.

    Returns:
        List[dict]: A list of dictionaries containing avatar_id, output video path, and original audio path.

    Raises:
        InferenceError: If any step of the inference process fails.
    """
    try:

        """Initializes global objects and configurations, checking for required API keys."""
        global args, vae, fp, audio_processor, device, weight_dtype, whisper, pe, unet, timesteps
        args = None
        vae = None
        fp = None
        audio_processor = None
        device = None
        weight_dtype = None
        whisper = None
        pe = None
        unet = None
        timesteps = None

        logging.info("[INFO] Ensuring FFmpeg is available...")
        FFmpegUtils.ensure_ffmpeg_in_path(str(ffmpeg_path))

        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        logging.info(f"[INFO] Using device: {device}")

        # Load UNet and VAE models
        if not os.path.isfile(unet_config) or not os.path.isfile(unet_model_path):
            raise ModelNotFoundError("UNet config or model file not found.")

        logging.info("[INFO] Loading models...")
        vae, unet, pe = load_all_model(
            unet_model_path=unet_model_path,
            vae_type=CONFIG.vae_type,
            unet_config=unet_config,
            device=device
        )
        pe = pe.half().to(device)
        vae.vae = vae.vae.half().to(device)
        unet.model = unet.model.half().to(device)
        timesteps = torch.tensor([0], device=device)

        # Load audio processing models
        audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
        whisper = WhisperModel.from_pretrained(whisper_dir).to(device=device).eval()
        whisper.requires_grad_(False)

        # Load inference configuration
        logging.info("[INFO] Loading inference configuration...")
        inference_config = OmegaConf.load(CONFIG.configs_dir / "inference/realtime.yaml")

        results = []

        # Iterate over avatars
        for avatar_idx, avatar_id in enumerate(inference_config, 1):
            logging.info(f"[INFO] Processing avatar {avatar_idx}/{len(inference_config)}: {avatar_id}")
            avatar_cfg = inference_config[avatar_id]

            avatar = Avatar(
                args=CONFIG,
                avatar_id=avatar_id,
                video_path=avatar_cfg["video_path"],
                bbox_shift=avatar_cfg.get("bbox_shift", 0),
                batch_size=batch_size,
                preparation=avatar_cfg["preparation"]
            )

            # Iterate over audio clips
            for audio_num, audio_path in avatar_cfg["audio_clips"].items():
                logging.info(f"[INFO] Running inference for audio clip: {audio_num}")
                out_vid = avatar.inference(audio_path, audio_num, fps, skip_save_images)
                results.append({
                    "avatar_id": avatar_id,
                    "output_path": out_vid,
                    "original_audio": audio_path
                })
                logging.info(f"[INFO] Finished inference for audio clip: {audio_num}")

        logging.info("[INFO] All inferences completed successfully.")
        return results

    except Exception as e:
        logging.error(f"[ERROR] Inference failed: {e}")
        raise InferenceError(str(e))


def run_musetalk_inference(
        # --- General / Model Configs ---
        version: str = "v15",
        ffmpeg_path: str = FFMPEG_PATH,
        gpu_id: int = 0,
        vae_type: str = "sd-vae",
        # --- Path Configs ---
        unet_config: str = UNET_CONFIG,
        unet_model_path: str = UNET_MODEL_PATH,
        whisper_dir: str = WHISPER_DIR,
        inference_config_path: str = INFERENCE_CONFIG_PATH,
        bbox_shift: int = 0,  # Used 'bbox_shift' from argparse (replacing 'bbox_shift_default')
        extra_margin: int = 10,
        # --- Output / I/O ---
        result_dir: str = RESULTS_PATH,
        output_vid_name: str = None,
        # --- Audio / Time Params ---
        fps: int = 25,
        audio_padding_length_left: int = 2,
        audio_padding_length_right: int = 2,
        # --- Processing / Flags ---
        batch_size: int = 20,
        use_saved_coord: bool = False,
        saved_coord: bool = False,
        skip_save_images: bool = False,
        # --- Face Parsing / Blending ---
        parsing_mode: str = 'jaw',
        left_cheek_width: int = 90,
        right_cheek_width: int = 90
) -> List[Dict[str, Any]]:
    """
    Function to run MuseTalk inference pipeline for avatars.
    All parameters have default values, can be overridden.
    """

    try:

        """Initializes global objects and configurations, checking for required API keys."""
        global args, vae, fp, audio_processor, device, weight_dtype, whisper, pe, unet, timesteps
        args = None
        vae = None
        fp = None
        audio_processor = None
        device = None
        weight_dtype = None
        whisper = None
        pe = None
        unet = None
        timesteps = None

        all_params = locals()

        args = types.SimpleNamespace(**all_params)

        print(f"[INFO] Loading args configuration...{args}")

        print("[INFO] Loading inference configuration...")
        inference_config = OmegaConf.load(inference_config_path)

        # --- Check ffmpeg availability ---
        if not fast_check_ffmpeg():
            print("[INFO] Adding ffmpeg to PATH")
            path_separator = ';' if sys.platform == 'win32' else ':'
            os.environ["PATH"] = f"{ffmpeg_path}{path_separator}{os.environ['PATH']}"
            if not fast_check_ffmpeg():
                print("[WARNING] ffmpeg not found. Make sure it's installed and path is correct.")

        # --- Set computing device ---
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")

        # --- Load UNet and VAE models ---
        print("[INFO] Loading UNet and VAE models...")

        # Check if the config file exists
        if not os.path.isfile(unet_config):
            raise FileNotFoundError(f"Inference - UNet config file not found at: {unet_config}")

        # Check if the model file exists
        if not os.path.isfile(unet_model_path):
            raise FileNotFoundError(f"Inference - UNet model file not found at: {unet_model_path}")

        vae, unet, pe = load_all_model(
            unet_model_path=unet_model_path,
            vae_type=vae_type,
            unet_config=unet_config,
            device=device
        )
        timesteps = torch.tensor([0], device=device)

        pe = pe.half().to(device)
        vae.vae = vae.vae.half().to(device)
        unet.model = unet.model.half().to(device)

        # --- Initialize audio processor and Whisper model ---
        audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
        weight_dtype = unet.model.dtype
        whisper = WhisperModel.from_pretrained(whisper_dir)
        whisper = whisper.to(device=device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)
        print("[INFO] Audio processor and Whisper model initialized.")

        # --- Initialize face parser ---
        if version == "v15":
            fp = FaceParsing(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)
        else:
            fp = FaceParsing()
        print(f"[INFO] Face parser initialized with version {version}.")

        # --- Run inference for each avatar ---
        for avatar_id in inference_config:
            data_preparation = inference_config[avatar_id]["preparation"]
            video_path = inference_config[avatar_id]["video_path"]
            bbox_shift = 0 if version == "v15" else inference_config[avatar_id]["bbox_shift"]

            avatar = Avatar(
                args=args,
                avatar_id=avatar_id,
                video_path=video_path,
                bbox_shift=bbox_shift,
                batch_size=batch_size,
                preparation=data_preparation
            )

            results_list = []

            audio_clips = inference_config[avatar_id]["audio_clips"]
            for audio_num, audio_path in audio_clips.items():
                print(f"[INFO] Inferring avatar {avatar_id} with audio: {audio_path}")
                output_path = avatar.inference(audio_path, audio_num, fps, skip_save_images)

                results_list.append({
                    "avatar_id": avatar_id,
                    "output_path": output_path,
                    "original_audio": audio_path
                })

            print("[SUCCESS] MuseTalk inference finished successfully!")

        return results_list

    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        raise


def inferense_v2()-> List[Dict[str, Any]]:

    from pathlib import Path

    # --- Constants for Directory Paths ---
    BASE_DIR = Path(__file__).resolve().parent.parent
    PODCAST_DIR = BASE_DIR / "podcast_content"
    AUDIO_DIR = PODCAST_DIR / "audio"

    audio_paths = [
        {"speaker": "Speaker1", "audio_path": str(AUDIO_DIR / "segment_1.wav")},
        {"speaker": "Speaker2", "audio_path": str(AUDIO_DIR / "segment_2.wav")},
        {"speaker": "Speaker1", "audio_path": str(AUDIO_DIR / "segment_3.wav")},
        {"speaker": "Speaker2", "audio_path": str(AUDIO_DIR / "segment_4.wav")},
    ]

    return audio_paths


if __name__ == "__main__":
    '''
    This script is used to simulate online chatting and applies necessary pre-processing such as face detection and face parsing in advance. During online chatting, only UNet and the VAE decoder are involved, which makes MuseTalk real-time.
    '''
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Version of MuseTalk: v1 or v15")
    # parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    # parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    # parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    # parser.add_argument("--unet_config", type=str, default="./models/musetalk/musetalk.json", help="Path to UNet configuration file")
    # parser.add_argument("--unet_model_path", type=str, default="./models/musetalk/pytorch_model.bin", help="Path to UNet model weights")
    # parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    # parser.add_argument("--inference_config", type=str, default="configs/inference/realtime.yaml")
    # parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")
    # parser.add_argument("--result_dir", default='./results', help="Directory for output results")
    # parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    # parser.add_argument("--fps", type=int, default=25, help="Video frames per second")
    # parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    # parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    # parser.add_argument("--batch_size", type=int, default=20, help="Batch size for inference")
    # parser.add_argument("--output_vid_name", type=str, default=None, help="Name of output video file")
    # parser.add_argument("--use_saved_coord", action="store_true", help='Use saved coordinates to save time')
    # parser.add_argument("--saved_coord", action="store_true", help='Save coordinates for future use')
    # parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")
    # parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    # parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    # parser.add_argument("--skip_save_images",
    #                    action="store_true",
    #                    help="Whether skip saving images for better generation speed calculation",
    #                    )
    #
    # args = parser.parse_args()
    #
    # # Configure ffmpeg path
    # if not fast_check_ffmpeg():
    #     print("Adding ffmpeg to PATH")
    #     # Choose path separator based on operating system
    #     path_separator = ';' if sys.platform == 'win32' else ':'
    #     os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
    #     if not fast_check_ffmpeg():
    #         print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")
    #
    # # Set computing device
    # device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    #
    # # Load model weights
    # vae, unet, pe = load_all_model(
    #     unet_model_path=args.unet_model_path,
    #     vae_type=args.vae_type,
    #     unet_config=args.unet_config,
    #     device=device
    # )
    # timesteps = torch.tensor([0], device=device)
    #
    # pe = pe.half().to(device)
    # vae.vae = vae.vae.half().to(device)
    # unet.model = unet.model.half().to(device)
    #
    # # Initialize audio processor and Whisper model
    # audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    # weight_dtype = unet.model.dtype
    # whisper = WhisperModel.from_pretrained(args.whisper_dir)
    # whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    # whisper.requires_grad_(False)
    #
    # # Initialize face parser with configurable parameters based on version
    # if args.version == "v15":
    #     fp = FaceParsing(
    #         left_cheek_width=args.left_cheek_width,
    #         right_cheek_width=args.right_cheek_width
    #     )
    # else:  # v1
    #     fp = FaceParsing()
    #
    # inference_config = OmegaConf.load(args.inference_config)
    # print(inference_config)
    #
    # for avatar_id in inference_config:
    #     data_preparation = inference_config[avatar_id]["preparation"]
    #     video_path = inference_config[avatar_id]["video_path"]
    #     if args.version == "v15":
    #         bbox_shift = 0
    #     else:
    #         bbox_shift = inference_config[avatar_id]["bbox_shift"]
    #     avatar = Avatar(
    #         version=args.version,
    #         avatar_id=avatar_id,
    #         video_path=video_path,
    #         bbox_shift=bbox_shift,
    #         batch_size=args.batch_size,
    #         preparation=data_preparation)
    #
    #     audio_clips = inference_config[avatar_id]["audio_clips"]
    #     for audio_num, audio_path in audio_clips.items():
    #         print("Inferring using:", audio_path)
    #         avatar.inference(audio_path,
    #                        audio_num,
    #                        args.fps,
    #                        args.skip_save_images)
