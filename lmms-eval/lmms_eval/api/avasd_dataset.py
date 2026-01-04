import os
import os.path as osp
import json
from torch.utils.data import Dataset
import pandas as pd

import torch
import cv2
import av
import os.path as osp
import json
import numpy as np
import os

from PIL import Image
from decord import VideoReader, cpu
from torch.utils.data import Dataset
# import sys
# sys.path.append('/map-vepfs/weiming/wentao/lmm_cross_attn')

def get_resize_output_image_size(height, width, shortest_edge, longest_edge):
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.

    Args:
        image (`np.ndarray`):
            Image to resize.
        size (`Dict[str, int]`):
            Size of the output image containing the keys "shortest_edge" and "longest_edge".
        input_data_format (`ChannelDimension` or `str`):
            The channel dimension format of the input image.

    Returns:
        The output size of the image after resizing.
    """
    if shortest_edge is None and longest_edge is None:
        return height, width

    min_len = shortest_edge
    max_len = longest_edge
    aspect_ratio = width / height

    if width >= height and width > max_len:
        width = max_len
        height = int(width / aspect_ratio)
    elif height > width and height > max_len:
        height = max_len
        width = int(height * aspect_ratio)
    height = max(height, min_len)
    width = max(width, min_len)
    return height, width

def uniform_indices(num_frames: int, total_frames: int) -> list[int]:
    """Get uniform indices 

    Args:
        num_frames (int): number of frames
        total_frames (int): total number of frames

    Returns:
        list[int]: Output frame indices
    """
    if num_frames < total_frames:
        splits = torch.linspace(0, total_frames, num_frames+1, dtype=int)
        indices = ((splits[:-1] + splits[1:]) // 2).tolist()
    else:
        indices = list(range(total_frames))

    return indices


def fps_indices(input_fps: float, total_frames: int, output_fps: float = None, max_num_frames: int = -1) -> list[int]:
    """Get indices according to the output_fps

    Args:
        input_fps (float): input fps
        total_frames (int): total number of frames
        output_fps (float, optional): output fps. Defaults to None, means output_fps==input_fps.
        max_num_frames (int, optional): max number of frames. Defaults to -1, means no limitation.

    Returns:
        list[int]: Output frame indices
    """
    delta = 1 if output_fps is None else input_fps / output_fps
    indices = torch.arange(0, total_frames, delta).round().to(int)
    indices = [e for e in indices if e < total_frames]
    if 0 < max_num_frames < len(indices):
        indices = indices[:max_num_frames]

    return indices

def load_decord(src_path: str, sample_type: str, **kwargs) -> list[Image.Image]:
    """Load video using decord, optionally load subtitles

    Args:
        src_path (str): video path
        sample_type (str): 'uniform' or 'fps'
        sub_path (str): subtitle path, .srt
        kwargs: for 'uniform', require 'num_frames'; for 'fps', optionally require 'output_fps' and 'max_num_frames'

    Returns:
        list[Image.Image] | tuple[list[Image.Image], str]: frame list, subtitle str (optional)
    """
    vr = VideoReader(src_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    do_resize = kwargs.pop('do_resize', False)
    if sample_type == 'uniform':
        num_frames = kwargs.pop('num_frames')
        width = height = None
        if num_frames == "auto":
            model_patch_size = kwargs['model_patch_size']
            img_shortest_edge = kwargs['img_shortest_edge']
            img_longest_edge = kwargs['img_longest_edge']
            max_img_seq_len = kwargs['max_img_seq_len']
            vid = cv2.VideoCapture(src_path)
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            vid.release()
            height, width = get_resize_output_image_size(height, width, img_shortest_edge, img_longest_edge)
            num_patches = int((height // model_patch_size) * (width // model_patch_size))
            num_frames = int(max_img_seq_len // num_patches)
        if do_resize:
            vid = cv2.VideoCapture(src_path)
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            vid.release()
            img_shortest_edge = kwargs['img_shortest_edge']
            img_longest_edge = kwargs['img_longest_edge']
            height, width = get_resize_output_image_size(height, width, img_shortest_edge, img_longest_edge)
        input_fps = float(vr.get_avg_fps())
        indices = uniform_indices(num_frames, total_frames)
        durations = [idx / input_fps for idx in indices]
        frames = vr.get_batch(indices).asnumpy()        # (T, H, W, C), np.uint8
        frames = [Image.fromarray(frame).resize((int(width), int(height)), resample=3) if width and height else Image.fromarray(frame) for frame in frames]
    elif sample_type == 'fps':
        input_fps = float(vr.get_avg_fps())
        output_fps = kwargs.pop('output_fps', None)
        max_num_frames = kwargs.pop('max_num_frames', -1)
        indices = fps_indices(input_fps, total_frames, output_fps, max_num_frames)
        durations = [idx / input_fps for idx in indices]
        frames = vr.get_batch(indices).asnumpy()        # (T, H, W, C), np.uint8
        frames = [Image.fromarray(frame) for frame in frames]
    else:
        raise ValueError(f'Do not support {sample_type} sample type')

    return frames, durations


def load_pyav(src_path: str, sample_type: str, **kwargs) -> list[Image.Image]:
    """Load video using PyAV

    Args:
        src_path (str): video path
        sample_type (str): 'uniform' or 'fps'
        kwargs: for 'uniform', require 'num_frames'; for 'fps', optionally require 'output_fps' and 'max_num_frames'

    Returns:
        list[Image.Image] | tuple[list[Image.Image], str]: frame list, durations list
    """
    # Open video using PyAV
    container = av.open(src_path)
    stream = container.streams.video[0]
    total_frames = stream.frames
    input_fps = stream.average_rate
    frames = []
    durations = []
    
    if sample_type == 'uniform':
        num_frames = kwargs.pop('num_frames')
        width = height = None
        if num_frames == "auto":
            model_patch_size = kwargs.pop('model_patch_size')
            img_shortest_edge = kwargs.pop('img_shortest_edge')
            img_longest_edge = kwargs.pop('img_longest_edge')
            max_img_seq_len = kwargs.pop('max_img_seq_len')
            # Load video info using OpenCV or any other method
            vid = cv2.VideoCapture(src_path)
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            vid.release()
            height, width = get_resize_output_image_size(height, width, img_shortest_edge, img_longest_edge)
            num_patches = int((height // model_patch_size) * (width // model_patch_size))
            num_frames = int(max_img_seq_len // num_patches)

        indices = uniform_indices(num_frames, total_frames)
        frames = []
        for frame_index, packet in enumerate(container.decode(stream)):
            if frame_index in indices:
                frame = packet.to_image()  # Converts to PIL Image
                if width and height:
                    frame = frame.resize((int(width), int(height)), resample=Image.Resampling.BILINEAR)
                frames.append(frame)
                durations.append(frame_index / input_fps)
                
    elif sample_type == 'fps':
        output_fps = kwargs.pop('output_fps', None)
        max_num_frames = kwargs.pop('max_num_frames', -1)
        indices = fps_indices(float(input_fps), total_frames, output_fps, max_num_frames)
        frames = []
        for frame_index, packet in enumerate(container.decode(stream)):
            if frame_index in indices:
                frame = packet.to_image()  # Converts to PIL Image
                frames.append(frame)
                durations.append(frame_index / input_fps)

    else:
        raise ValueError(f"Do not support {sample_type} sample type")
    
    container.close()
    return frames, durations


def load_folder(src_path: str, sample_type: str, **kwargs) -> list[Image.Image]:
    """Load video using decord, optionally load subtitles

    Args:
        src_path (str): video path
        sample_type (str): 'uniform' or 'fps'
        sub_path (str): subtitle path, .srt
        kwargs: for 'uniform', require 'num_frames'; for 'fps', optionally require 'output_fps' and 'max_num_frames'

    Returns:
        list[Image.Image] | tuple[list[Image.Image], str]: frame list, subtitle str (optional)
    """
    # list all images in the folder
    img_list = sorted([osp.join(src_path, f) for f in os.listdir(src_path) if f.endswith('.jpg')])
    total_frames = len(img_list)
    img_list = [osp.join(src_path, f"frame_{i}.jpg") for i in range(total_frames)]
    do_resize = kwargs.pop('do_resize', False)
    if sample_type == 'uniform':
        num_frames = kwargs.pop('num_frames')
        width = height = None
        if num_frames == "auto":
            model_patch_size = kwargs['model_patch_size']
            img_shortest_edge = kwargs['img_shortest_edge']
            img_longest_edge = kwargs['img_longest_edge']
            max_img_seq_len = kwargs['max_img_seq_len']
            vid = Image.open(img_list[0])
            width, height = vid.size
            height, width = get_resize_output_image_size(height, width, img_shortest_edge, img_longest_edge)
            num_patches = int((height // model_patch_size) * (width // model_patch_size))
            num_frames = int(max_img_seq_len // num_patches)
        if do_resize:
            vid = Image.open(img_list[0])
            width, height = vid.size
            img_shortest_edge = kwargs['img_shortest_edge']
            img_longest_edge = kwargs['img_longest_edge']
            height, width = get_resize_output_image_size(height, width, img_shortest_edge, img_longest_edge)
        input_fps = 25
        indices = uniform_indices(num_frames, total_frames)
        durations = [idx / input_fps for idx in indices]
        frames = [np.array(Image.open(os.path.join(src_path, img_list[idx]))) for idx in indices]
        frames = np.array(frames)        # (T, H, W, C), np.uint8
        frames = [Image.fromarray(frame).resize((int(width), int(height)), resample=3) if width and height else Image.fromarray(frame) for frame in frames]
    else:
        raise ValueError(f'Do not support {sample_type} sample type')

    return frames, durations

class AVASDDataset(Dataset):

    features = {
        "uid": "string",
        "video": "tensor",
        "video_durations": "list",
        "questions": "list"
    }

    def __init__(self, dataset_path: str, csv_path: str, sample_config: dict = None):

        self.dataset_path = dataset_path
        self.data = pd.read_csv(csv_path)
        self.csv_path = csv_path
        self.sample_config = sample_config

        behavior_string = ', '.join(self.data.columns[1:-1])

        self.prompt = (
            "system\n"
            "You are a helpful assistant.\n"
            "user\n"
            f"You are analyzing an AV-ASD (Audio-Visual Autism Spectrum Dataset) video with 128 frames.\n\n"
            f"For each of the following 9 behaviors, indicate 1 if it is present and 0 if not present. The behaviors are:\n"
            f"{behavior_string}\n\n"
            "Output a list of 0 or 1, separated by a comma, one for each behavior. For example, if there are 9 behaviors, and only the first is observed, output 1,0,0,0,0,0,0,0,0\n\n"
            "Output only the list of 0s and 1s separated by commas. There should be 9 behaviors total."
        )

        self.samples = []
        for _, row in self.data.iterrows():
            video_id = row['Video_ID']
            video_path = osp.join(self.dataset_path, f"{video_id}.mp4")
            if osp.exists(video_path):
                self.samples.append({
                    'uid' : video_id,
                    'video_path': video_path,
                    'correct_answers': row[1:-1].tolist()
                })

        self.splits = {"data" : self}
                
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        if isinstance(idx, str):
            try:
                return self.splits[idx]
            except Exception as e:
                print(f"Error processing video {video}: {e}. Split {idx} doesn't exist.")
        
        if isinstance(idx, dict):
            return idx

        item = self.samples[idx]

        video_path = item['video_path']
        if os.path.exists(video_path):
            try:
                frames, durations = load_decord(
                    src_path = video_path,
                    **self.sample_config
                )
            except Exception as e:
                frames, durations = [], []
                print(f"Error processing video {video_path}: {e}")
        
        else:
            frames, durations = [], []
            print(f"Video file {video_path} does not exist.")
        
        return dict(
            uid=item['uid'],
            video=frames,
            video_durations=durations,
            prompt=self.prompt,
            correct_answers=item['correct_answers'],
            video_path=video_path
        )
    
    def copy(self):
        return AVASDDataset(self.dataset_path, self.csv_path, sample_config=self.sample_config.copy())
        
