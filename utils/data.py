'''
Utilities to format each dataset for training with VideoNSA.
Expected format: ["messages", "images", "video_id", "data_source"]
'''

import os
import json
import cv2
from tqdm import tqdm
from PIL import Image


def compress_image(image_path, max_size=224, quality=85):
    """
    Compress and resize an image in place.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum width/height (maintains aspect ratio)
        quality: JPEG quality (1-100, lower = smaller file)
    """
    img = Image.open(image_path)
    
    # Resize maintaining aspect ratio
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Save with compression
    img.save(image_path, quality=quality, optimize=True)


def get_image_frames(
    video_path,
    output_path,
    sample_every_n=1,
    compress=True,
    max_size=224,
    quality=85
):
    """
    Extract frames from a video and save them to the output path.

    Args:
        video_path: Path to the input video file
        output_path: Base directory where frames will be saved
        sample_every_n: Save every nth frame (1 = all frames, 2 = every other frame, etc.)
        compress: Whether to compress images after extraction
        max_size: Maximum dimension for compressed images
        quality: JPEG quality for compression (1-100)

    Returns:
        List of paths to the extracted frame images
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(output_path, video_id)
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_paths = []
    frame_idx = 0
    saved_idx = 0

    with tqdm(total=total_frames, desc=f"Extracting frames from {video_id}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n == 0:
                frame_filename = f"{video_id}_{saved_idx:04d}.jpg"
                frame_path = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                
                # Compress the image after saving
                if compress:
                    compress_image(frame_path, max_size=max_size, quality=quality)
                
                frame_paths.append(frame_path)
                saved_idx += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    return frame_paths


def get_json_entry(
    prompts,
    video_path,
    video_frames_path,
    data_source="",
    sample_every_n=1,
    compress_images=True,
    max_image_size=224,
    image_quality=85
):
    """
    Create a JSON entry in the format expected by VideoNSA.

    Args:
        prompts: List of (question, answer) tuples for the conversation
        video_path: Path to the input video file
        video_frames_path: Base directory where frames will be saved
        data_source: Data source identifier string
        sample_every_n: Save every nth frame (1 = all frames, 2 = every other frame, etc.)
        compress_images: Whether to compress extracted images
        max_image_size: Maximum dimension for compressed images (default: 224)
        image_quality: JPEG quality (1-100, default: 85)

    Returns:
        Dictionary with 'messages', 'images', 'video_id', and 'data_source' keys
    """
    frame_paths = get_image_frames(
        video_path, 
        video_frames_path, 
        sample_every_n,
        compress=compress_images,
        max_size=max_image_size,
        quality=image_quality
    )
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    image_tags = "<image>" * len(frame_paths)

    messages = []
    for i, (question, answer) in enumerate(prompts):
        if i == 0:
            user_content = image_tags + "\n" + question
        else:
            user_content = question

        messages.append({
            "role": "user",
            "content": user_content
        })
        messages.append({
            "role": "assistant",
            "content": answer
        })

    if not prompts:
        messages.append({
            "role": "user",
            "content": image_tags
        })

    entry = {
        "messages": messages,
        "images": frame_paths,
        "video_id": video_id,
        "data_source": data_source
    }

    return entry


if __name__ == "__main__":

    json_entry = get_json_entry(
        prompts=[("What color is the main male character in the video? Choose from exactly one of the following options: Yellow, Red, Green, Blue.", "Yellow")],
        video_path="/home/ixzhu/orcd/scratch/MVLU/videos/MLVU/video/1_plotQA/movie101_66.mp4",
        video_frames_path='/home/ixzhu/orcd/scratch/MVLU/videos/MLVU/frames',
        data_source="",
        sample_every_n=1800,
        compress_images=True,      # Enable compression
        max_image_size=224,        # Resize to 224x224 max
        image_quality=85           # JPEG quality
    )

    # Save as JSONL format (one JSON object per line) - required by ms-swift
    output_jsonl_path = "/home/ixzhu/orcd/scratch/MVLU/data.jsonl"
    with open(output_jsonl_path, 'w') as f:
        f.write(json.dumps(json_entry) + '\n')

    print(f"JSONL saved to {output_jsonl_path}")