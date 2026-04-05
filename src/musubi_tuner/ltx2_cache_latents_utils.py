#!/usr/bin/env python3
"""
LTX-2 Cache Latents Utilities

Utilities for LTX-2 latent caching including:
- AR frame scaling with bucket selector
- Preview MP4 saving for control videos
- Frame extraction modes (head, chunk, slide, uniform, full)
- Control video creation for i2v slider training
"""

from __future__ import annotations

import glob
import logging
import os
from typing import List, Tuple, Optional

import numpy as np
import torch

from musubi_tuner.dataset.image_video_dataset import (
    save_latent_cache_ltx2,
    ItemInfo,
    BucketSelector,
    ARCHITECTURE_LTX2,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Control Args Reading
# ============================================================================

def read_control_args_from_last_config(dataset_config_path: str) -> Optional[List]:
    """Read control_args from last_config.toml for slider training.

    Looks for last_config.toml in the same directory as dataset_config_path,
    reads the [training_strategy] section, and returns control_args if slider mode is enabled.

    Args:
        dataset_config_path: Path to the dataset config TOML file

    Returns:
        List of [control_type, control_num] if slider mode is enabled, None otherwise.
        Example: ["freeze", 0] or ["jump", 1]
    """
    try:
        import toml

        dataset_config_dir = os.path.dirname(dataset_config_path)
        last_config_path = os.path.join(dataset_config_dir, "last_config.toml")

        if not os.path.exists(last_config_path):
            return None

        with open(last_config_path, 'r') as f:
            last_config = toml.load(f)

        training_strategy = last_config.get('training_strategy', {})
        slider_enabled = training_strategy.get('slider', False)
        if not isinstance(slider_enabled, bool):
            slider_enabled = str(slider_enabled).lower() in ['true', '1', 'yes', 'on']

        if not slider_enabled:
            return None

        raw_control_args = training_strategy.get('control_args', None)
        if raw_control_args is None:
            return None

        # Validate and convert control_args
        if not isinstance(raw_control_args, list):
            raise ValueError(f"control_args must be a list, got {type(raw_control_args)}")

        if len(raw_control_args) == 0:
            raise ValueError("control_args cannot be empty")

        control_type = raw_control_args[0]
        valid_types = ["jump", "freeze", "fade", "reverse"]
        if control_type not in valid_types:
            raise ValueError(f"control_args type must be one of {valid_types}, got '{control_type}'")

        # For "reverse" and "freeze", default num to 0 if not provided
        if control_type in ["reverse", "freeze"]:
            if len(raw_control_args) == 1:
                control_num = 0
            elif len(raw_control_args) >= 2:
                try:
                    control_num = int(raw_control_args[1])
                except (ValueError, TypeError):
                    control_num = 0
            else:
                control_num = 0
        else:
            # For "jump" and "fade", require num parameter
            if len(raw_control_args) < 2:
                raise ValueError(f"control_args for '{control_type}' must have 2 elements [type, num]")
            try:
                control_num = int(raw_control_args[1])
                if control_num < 1:
                    raise ValueError(f"control_args num must be >= 1, got {control_num}")
            except (ValueError, TypeError) as e:
                raise ValueError(f"control_args num must be an integer, got '{raw_control_args[1]}'")

        logger.info(f"Slider mode enabled with control_args: [{control_type}, {control_num}]")
        return [control_type, control_num]

    except Exception as e:
        logger.warning(f"Failed to read control_args from last_config.toml: {e}")
        return None


# ============================================================================
# Context Managers
# ============================================================================

def _amp_context(device: torch.device, dtype: torch.dtype):
    """Get autocast context for given device and dtype."""
    if device.type in {"cuda", "xpu"}:
        try:
            from torch.amp import autocast as torch_autocast
            return torch_autocast(device_type=device.type, dtype=dtype)
        except (ImportError, AttributeError):
            from torch.cuda.amp import autocast as torch_autocast
            return torch_autocast(dtype=dtype)
    from contextlib import nullcontext
    return nullcontext()


# ============================================================================
# Video Loading
# ============================================================================

def load_video_frames(video_path: str) -> tuple[np.ndarray, int]:
    """Load video frames from file using imageio.

    Returns:
        (frames, fps) where frames is [F, H, W, C] and fps is the video framerate
    """
    try:
        import imageio.v3 as iio
        # Get FPS first
        props = iio.immeta(video_path)
        fps = props.get('fps', 25)  # Default to 25fps if not found

        frames = iio.imread(video_path)
        if frames.ndim == 3:
            frames = np.expand_dims(frames, axis=0)
        return frames, fps
    except ImportError:
        raise RuntimeError("imageio is required for video loading")


# ============================================================================
# Frame Extraction
# ============================================================================

def extract_frames(
    frames: np.ndarray,
    mode: str,
    target_frames,
    frame_stride: int = 1,
    frame_sample: int = 1,
    max_frames: int = 128,
    vae_frame_stride: int = 8,
) -> List[Tuple[int, int, np.ndarray]]:
    """Extract frame chunks from video based on frame_extraction mode.

    Args:
        frames: Video frames as [F, H, W, C] numpy array
        mode: Frame extraction mode - "head", "chunk", "slide", "uniform", or "full"
        target_frames: Target number of frames per chunk (int or list of ints for frame_buckets)
        frame_stride: Stride for slide mode
        frame_sample: Number of samples for uniform mode
        max_frames: Maximum frames for full mode
        vae_frame_stride: VAE frame stride (used for full mode rounding)

    Returns:
        List of (start_frame, num_frames, chunk_frames) tuples
    """
    frame_count = frames.shape[0]
    chunks = []

    # Handle frame_buckets (list or tuple of target frame counts)
    if isinstance(target_frames, (list, tuple)):
        target_frame_list = list(target_frames)
    else:
        target_frame_list = [target_frames]

    for target_frame in target_frame_list:
        if mode == "head":
            # Take first N frames
            if frame_count >= target_frame:
                chunks.append((0, target_frame, frames[:target_frame]))

        elif mode == "chunk":
            # Split by target_frames
            for i in range(0, frame_count, target_frame):
                if i + target_frame <= frame_count:
                    chunks.append((i, target_frame, frames[i:i + target_frame]))

        elif mode == "slide":
            # Slide window
            if frame_count >= target_frame:
                for i in range(0, frame_count - target_frame + 1, frame_stride):
                    chunks.append((i, target_frame, frames[i:i + target_frame]))

        elif mode == "uniform":
            # Select N frames uniformly (frame_sample determines number of starting positions)
            if frame_count >= target_frame:
                frame_indices = np.linspace(0, frame_count - target_frame, frame_sample, dtype=int)
                for i in frame_indices:
                    chunks.append((int(i), target_frame, frames[int(i):int(i) + target_frame]))

        elif mode == "full":
            # Select all frames (rounded to VAE stride)
            # Only process once for full mode, ignore frame_buckets
            if not chunks:  # Only add once
                target_frame_actual = min(frame_count, max_frames)
                target_frame_actual = (target_frame_actual - 1) // vae_frame_stride * vae_frame_stride + 1
                chunks.append((0, target_frame_actual, frames[:target_frame_actual]))
            break  # Full mode only processes once

        else:
            raise ValueError(f"frame_extraction mode '{mode}' is not supported")

    return chunks


# ============================================================================
# Control Video Creation
# ============================================================================

def create_control_video(frames: np.ndarray) -> np.ndarray:
    """Create a control video by repeating the first frame."""
    first_frame = frames[0:1]
    control_frames = np.repeat(first_frame, frames.shape[0], axis=0)
    return control_frames


def _get_sample_indices(total_frames: int, num: int) -> list[int]:
    """Get frame indices to sample based on num parameter.

    num=1: 3 frames at 0%, 50%, 100% (start, middle, end)
    num=2: 4 frames at 0%, 33%, 66%, 100%
    num=3: 5 frames at 0%, 25%, 50%, 75%, 100%
    num=N: N+2 frames evenly distributed including start and end

    Args:
        total_frames: Total number of frames in the chunk
        num: Sampling parameter

    Returns:
        List of frame indices to sample
    """
    frame_count = total_frames
    if frame_count <= 0:
        return [0]

    # Number of samples = num + 2
    # num=1 -> 3 samples, num=2 -> 4 samples, num=3 -> 5 samples
    num_samples = num + 2

    indices = []
    for i in range(num_samples):
        idx = int(frame_count * i / (num_samples - 1)) if num_samples > 1 else 0
        # Clamp to valid range
        idx = max(0, min(idx, frame_count - 1))
        indices.append(idx)

    return indices


def create_control_video_jump(frames: np.ndarray, num: int, tail_frames: int = 8) -> np.ndarray:
    """Create a control video with jump cuts between sampled frames.

    Reserves tail_frames for the last sample, then evenly distributes
    remaining frames among the other samples.

    For num=1 with 49 frames and tail=8:
    - Work with 41 frames (49 - 8)
    - Samples at 0%, 50%, 100% of 41 = frames 0, 20, 40
    - Frames 0-19: frame 0 (20 frames)
    - Frames 20-39: frame 20 (20 frames)
    - Frames 40-48: frame 40 (9 frames, the tail)

    Args:
        frames: Input frames [F, H, W, C]
        num: Sampling parameter (see _get_sample_indices)
        tail_frames: Number of frames to reserve for the last sample

    Returns:
        Control video frames with jump cuts
    """
    total_frames = frames.shape[0]
    num_samples = num + 2

    # Reserve tail frames for the last sample
    working_frames = total_frames - tail_frames

    # Sample from the working frames only
    sample_indices = _get_sample_indices(working_frames, num)

    # Calculate frames per sample (excluding last)
    num_working_samples = len(sample_indices) - 1  # All except the last
    frames_per_sample = working_frames // num_working_samples
    remainder = working_frames % num_working_samples

    control_frames = []

    # Process all samples except the last
    for i in range(num_working_samples):
        sample_idx = sample_indices[i]
        sampled_frame = frames[sample_idx]
        control_frames.extend([sampleed_frame] * frames_per_sample)

    # Last sample gets tail_frames + remainder
    last_sample_idx = sample_indices[-1]
    last_sample_frame = frames[last_sample_idx]
    control_frames.extend([last_sample_frame] * (tail_frames + remainder))

    return np.array(control_frames, dtype=frames.dtype)


def create_control_video_fade(frames: np.ndarray, num: int, hold_frames: int = 5) -> np.ndarray:
    """Create a control video with smooth fades between sampled frames.

    Samples frames at uniform intervals, holds each for hold_frames,
    then fades to the next one with equal-length transition periods.

    Pattern for num=1 (3 samples): hold+transition+hold+transition+hold
    Pattern for num=2 (4 samples): hold+transition+hold+transition+hold+transition+hold
    Total = hold_frames * num_samples + transition_frames * (num_samples - 1)
    where num_samples = num + 2

    Args:
        frames: Input frames [F, H, W, C]
        num: Sampling parameter (see _get_sample_indices for resulting sample count)
        hold_frames: Number of frames to hold each sample before fading

    Returns:
        Control video frames with holds and fades
    """
    total_frames = frames.shape[0]
    sample_indices = _get_sample_indices(total_frames, num)
    num_samples = len(sample_indices)

    # Calculate transition frames: distribute remaining frames evenly between transitions
    total_hold_frames = hold_frames * num_samples
    remaining_frames = total_frames - total_hold_frames
    num_transitions = num_samples - 1

    if num_transitions > 0:
        transition_frames = remaining_frames // num_transitions
        remainder = remaining_frames % num_transitions
    else:
        transition_frames = 0
        remainder = 0

    control_frames = []
    transition_frame_pool = remainder  # Extra frames to distribute

    for i, sample_idx in enumerate(sample_indices):
        current_sample_frame = frames[sample_idx]

        # Add hold frames
        control_frames.extend([current_sample_frame] * hold_frames)

        # Add transition frames (if not last sample)
        if i < num_samples - 1:
            next_sample_idx = sample_indices[i + 1]
            next_sample_frame = frames[next_sample_idx]

            # Add extra frame to this transition if available
            this_transition = transition_frames + (1 if transition_frame_pool > 0 else 0)
            if transition_frame_pool > 0:
                transition_frame_pool -= 1

            for j in range(this_transition):
                if this_transition == 1:
                    t = 0.0
                else:
                    t = j / (this_transition - 1)

                # Blend frames: (1-t) * current + t * next
                blended = (1.0 - t) * current_sample_frame + t * next_sample_frame
                control_frames.append(blended.astype(frames.dtype))

    # Ensure we have exactly total_frames
    control_frames = control_frames[:total_frames]

    return np.array(control_frames, dtype=frames.dtype)


def create_control_video_reverse(frames: np.ndarray, num: int = 0) -> np.ndarray:
    """Create a control video by reversing the frames.

    Simply reverses the frame order, playing the chunk backwards.
    The num parameter is accepted for API consistency but not used.

    Args:
        frames: Input frames [F, H, W, C]
        num: Unused (kept for API consistency with other control modes)

    Returns:
        Control video frames in reverse order
    """
    # Reverse frames along the first axis (time)
    return frames[::-1].copy()


# ============================================================================
# Preview MP4 Saving
# ============================================================================

def save_control_video_mp4(
    frames: np.ndarray,
    output_path: str,
    fps: int = 25,
) -> None:
    """Save control video frames as MP4 file for visualization.

    Args:
        frames: Video frames as [F, H, W, C] numpy array
        output_path: Path to save the MP4 file
        fps: Frames per second for the output video
    """
    try:
        import imageio.v3 as iio
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save as MP4
        iio.imwrite(output_path, frames, fps=fps, codec="libx264", quality=8)
        logger.info(f"Saved control video MP4 to: {output_path}")
    except ImportError:
        raise RuntimeError("imageio is required for video saving")
    except Exception as e:
        logger.warning(f"Failed to save control video MP4 to {output_path}: {e}")


# ============================================================================
# AR Frame Scaling with Bucket Selector
# ============================================================================

def encode_video_to_latent(
    frames: np.ndarray,
    vae,
    device: torch.device,
    vae_dtype: torch.dtype,
    bucket_selector: BucketSelector = None,
) -> torch.Tensor:
    """Encode video frames to latent space with AR bucketing support.

    Args:
        frames: Video frames as [F, H, W, C] numpy array
        vae: VAE model for encoding
        device: Torch device
        vae_dtype: Data type for VAE
        bucket_selector: Optional BucketSelector for aspect ratio bucketing

    Returns:
        Latent tensor [C, F, H, W]
    """
    # Convert to tensor: [F, H, W, C] -> [1, C, F, H, W]
    content = torch.from_numpy(frames).float()
    content = content.unsqueeze(0)  # [1, F, H, W, C]
    content = content.permute(0, 4, 1, 2, 3).contiguous()  # [1, C, F, H, W]

    # Get current size
    _, c, f, h, w = content.shape
    original_size = (w, h)  # (width, height) for bucket selector

    # Determine target resolution using bucket selector or default to divisible by 32
    if bucket_selector is not None:
        target_size = bucket_selector.get_bucket_resolution(original_size)
        new_w, new_h = target_size
        logger.info(f"Before bucket resize: {original_size} -> bucket: {target_size}")
    else:
        # No bucketing: just make divisible by 32 (LTX-2 spatial downsampling factor)
        new_h = (h // 32) * 32
        new_w = (w // 32) * 32
        logger.info(f"Before resize (no bucketing): {content.shape}, new_h={new_h}, new_w={new_w}")

    # Always resize to ensure divisibility (VGG-style initial conv reduces by 4x first)
    import torchvision.transforms.functional as TF
    # Reshape for resize: [1, C, F, H, W] -> [1*F, C, H, W]
    content = content.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
    content = TF.resize(content, [new_h, new_w], antialias=True)
    # Reshape back: [1*F, C, new_h, new_w] -> [1, C, F, new_h, new_w]
    content = content.reshape(1, f, c, new_h, new_w).permute(0, 2, 1, 3, 4)

    logger.info(f"After resize: {content.shape}")

    content = content.to(device=device, dtype=vae_dtype)
    content = content / 127.5 - 1.0

    # Pad frames if needed (LTX-2 requires (F-1) % 8 == 0)
    frames_count = content.shape[2]
    remainder = (frames_count - 1) % 8
    if remainder != 0:
        pad = 8 - remainder
        last = content[:, :, -1:, :, :].expand(-1, -1, pad, -1, -1)
        content = torch.cat([content, last], dim=2)

    with _amp_context(device, vae_dtype), torch.no_grad():
        latents = vae(content)

    # Return [C, F, H, W]
    latents = latents.squeeze(0).cpu().to(dtype=vae_dtype)
    return latents


def resize_frames_for_mp4(
    frames: np.ndarray,
    bucket_selector: BucketSelector = None,
) -> np.ndarray:
    """Resize frames for MP4 saving using bucket selector (same as latent encoding).

    This ensures the MP4 videos in negative/ folder match the same resolution
    as the encoded latents.

    Args:
        frames: Video frames as [F, H, W, C] numpy array (uint8)
        bucket_selector: Optional BucketSelector for aspect ratio bucketing

    Returns:
        Resized frames as [F, H, W, C] numpy array (uint8)
    """
    import torchvision.transforms.functional as TF

    # Convert to tensor: [F, H, W, C] -> [1, C, F, H, W]
    content = torch.from_numpy(frames).float()
    content = content.unsqueeze(0)  # [1, F, H, W, C]
    content = content.permute(0, 4, 1, 2, 3).contiguous()  # [1, C, F, H, W]

    # Get current size
    _, c, f, h, w = content.shape
    original_size = (w, h)  # (width, height) for bucket selector

    # Determine target resolution using bucket selector or default to divisible by 32
    if bucket_selector is not None:
        target_size = bucket_selector.get_bucket_resolution(original_size)
        new_w, new_h = target_size
    else:
        # No bucketing: just make divisible by 32 (LTX-2 spatial downsampling factor)
        new_h = (h // 32) * 32
        new_w = (w // 32) * 32

    # Reshape for resize: [1, C, F, H, W] -> [1*F, C, H, W]
    content = content.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
    content = TF.resize(content, [new_h, new_w], antialias=True)
    # Reshape back: [1*F, C, new_h, new_w] -> [F, H, W, C]
    content = content.reshape(f, c, new_h, new_w).permute(0, 2, 3, 1)

    # Convert back to numpy as uint8
    return content.clip(0, 255).byte().numpy()


# ============================================================================
# I2V Control Cache Creation (for slider training)
# ============================================================================

def process_single_video_create_negative_mp4(
    video_path: str,
    device: torch.device,
    vae_dtype: torch.dtype,
    frame_extraction: str = "head",
    target_frames = 17,
    frame_stride: int = 1,
    frame_sample: int = 1,
    max_frames: int = 128,
    vae_frame_stride: int = 8,
    control_args: Optional[List[str]] = None,
) -> List[str]:
    """Step 1: Create negative/control MP4 videos from source video.

    Creates control videos based on control_args and saves them to negative/ folder.
    These MP4s will then be processed to create musubi_cache_negative/ latents.

    Args:
        video_path: Path to the source video file
        device: Torch device (not used in this step, but kept for API consistency)
        vae_dtype: Data type for VAE (not used in this step, but kept for API consistency)
        frame_extraction: Frame extraction mode
        target_frames: Target number of frames per chunk (int or list)
        frame_stride: Stride for slide mode
        frame_sample: Number of samples for uniform mode
        max_frames: Maximum frames for full mode
        vae_frame_stride: VAE frame stride
        control_args: Control video creation arguments

    Returns:
        List of paths to the created MP4 files in negative/ folder
    """
    created_mp4s = []
    try:
        # Load video frames (and get original FPS)
        frames, original_fps = load_video_frames(video_path)
        frame_count = frames.shape[0]
        logger.info(f"Loaded video: {video_path}, shape: {frames.shape}, fps: {original_fps}")

        # Create negative directory next to the video
        base_dir = os.path.dirname(video_path)
        negative_dir = os.path.join(base_dir, "negative")
        os.makedirs(negative_dir, exist_ok=True)

        # Extract chunks based on frame_extraction mode
        chunks = extract_frames(
            frames, frame_extraction, target_frames, frame_stride,
            frame_sample, max_frames, vae_frame_stride
        )
        logger.info(f"Extracted {len(chunks)} chunk(s) using mode '{frame_extraction}'")

        # Get basename without extension
        basename = os.path.splitext(os.path.basename(video_path))[0]

        # Process each chunk to create control MP4s
        for start_idx, num_frames, chunk_frames in chunks:
            # Create control video based on control_args
            if control_args is not None:
                control_type = control_args[0]  # "jump", "fade", "reverse", or "freeze"
                control_num = control_args[1]  # Already converted to int

                logger.info(f"Creating control video with type={control_type}, num={control_num}")

                if control_type == "jump":
                    logger.info(f"Using JUMP control with num={control_num}")
                    control_frames = create_control_video_jump(chunk_frames, control_num)
                elif control_type == "freeze":
                    logger.info(f"Using FREEZE control (repeating first frame)")
                    control_frames = create_control_video(chunk_frames)
                elif control_type == "fade":
                    logger.info(f"Using FADE control with num={control_num}")
                    control_frames = create_control_video_fade(chunk_frames, control_num)
                elif control_type == "reverse":
                    logger.info(f"Using REVERSE control (reversing chunk)")
                    control_frames = create_control_video_reverse(chunk_frames, control_num)
                else:
                    logger.error(f"Unknown control type '{control_type}', this should not happen after validation!")
                    raise ValueError(f"Unknown control type '{control_type}'")
            else:
                logger.info("No control_args, using default (repeat first frame)")
                # Default: repeat first frame
                control_frames = create_control_video(chunk_frames)

            # Save control video as MP4 in negative/ folder
            chunk_suffix = f"_{start_idx:05d}-{num_frames:03d}"
            negative_mp4_path = os.path.join(negative_dir, f"{basename}{chunk_suffix}.mp4")
            save_control_video_mp4(control_frames, negative_mp4_path, fps=original_fps)
            created_mp4s.append(negative_mp4_path)

            logger.info(f"Saved negative MP4 to: {negative_mp4_path}")

            # Clean up this chunk
            del chunk_frames, control_frames

        # Clean up
        del frames

    except Exception as e:
        logger.error(f"Failed to process {video_path}: {e}")
        import traceback
        traceback.print_exc()

    return created_mp4s


def process_negative_directory_to_latents(
    video_dir: str,
    vae,
    device: torch.device,
    vae_dtype: torch.dtype,
    bucket_selector: BucketSelector = None,
) -> None:
    """Step 2: Process negative/ MP4 videos to create musubi_cache_negative/ latents.

    Reads MP4 files from {video_dir}/negative/ and encodes them to latents
    saved in {video_dir}/musubi_cache_negative/.

    Args:
        video_dir: Directory containing negative/ subfolder with MP4 files
        vae: VAE model for encoding
        device: Torch device
        vae_dtype: Data type for VAE
        bucket_selector: Optional BucketSelector for aspect ratio bucketing
    """
    try:
        negative_dir = os.path.join(video_dir, "negative")
        neg_cache_dir = os.path.join(video_dir, "musubi_cache_negative")
        os.makedirs(neg_cache_dir, exist_ok=True)

        # Find all MP4 files in negative/ directory
        mp4_files = []
        for ext in ("mp4", "webm", "mov", "avi", "mkv"):
            mp4_files.extend(glob.glob(os.path.join(negative_dir, f"*.{ext}")))
            mp4_files.extend(glob.glob(os.path.join(negative_dir, f"*.{ext.upper()}")))

        if not mp4_files:
            logger.warning(f"No MP4 files found in {negative_dir}")
            return

        logger.info(f"Found {len(mp4_files)} MP4 files in {negative_dir}")

        # Process each MP4 to latents
        for mp4_path in mp4_files:
            logger.info(f"Processing negative MP4: {mp4_path}")
            process_single_video_to_latents(mp4_path, vae, neg_cache_dir, device, vae_dtype, bucket_selector)

    except Exception as e:
        logger.error(f"Failed to process negative directory {video_dir}: {e}")
        import traceback
        traceback.print_exc()


def process_single_video_to_latents(
    video_path: str,
    vae,
    cache_dir: str,
    device: torch.device,
    vae_dtype: torch.dtype,
    bucket_selector: BucketSelector = None,
) -> None:
    """Process a single video to latents (no chunking, full video encoding).

    Args:
        video_path: Path to the video file
        vae: VAE model for encoding
        cache_dir: Directory to save cached latents
        device: Torch device
        vae_dtype: Data type for VAE
        bucket_selector: Optional BucketSelector for aspect ratio bucketing
    """
    try:
        # Load video frames
        frames, original_fps = load_video_frames(video_path)
        logger.info(f"Loaded video: {video_path}, shape: {frames.shape}, fps: {original_fps}")

        # Get basename without extension
        basename = os.path.splitext(os.path.basename(video_path))[0]

        # Encode to latent
        latent = encode_video_to_latent(frames, vae, device, vae_dtype, bucket_selector)

        # Convert to float32 for numpy compatibility
        latent_f32 = latent.float()

        # Create path: {cache_dir}/{basename}_ltx2.safetensors
        latent_path = os.path.join(cache_dir, f"{basename}_ltx2.safetensors")

        # Create ItemInfo and save
        item = ItemInfo(
            item_key=latent_path,
            content=latent_f32.numpy(),
            caption="",
            original_size=frames.shape[1:3],
            latent_cache_path=latent_path,
        )
        save_latent_cache_ltx2(item, latent_f32)

        logger.info(f"Saved latents to: {latent_path}")

        # Clean up
        del frames, latent, latent_f32

    except Exception as e:
        logger.error(f"Failed to process {video_path}: {e}")
        import traceback
        traceback.print_exc()


def process_single_video_i2v(
    video_path: str,
    vae,
    cache_dir: str,
    device: torch.device,
    vae_dtype: torch.dtype,
    frame_extraction: str = "head",
    target_frames = 17,
    frame_stride: int = 1,
    frame_sample: int = 1,
    max_frames: int = 128,
    vae_frame_stride: int = 8,
    control_args: Optional[List[str]] = None,
    bucket_selector: BucketSelector = None,
) -> None:
    """Process a single video to create original + control latent pairs for slider training.

    This is a wrapper that:
    1. Creates negative/ MP4 videos from source (using control_args)
    2. Processes original video to musubi_cache_positive/ latents
    3. Processes negative/ MP4s to musubi_cache_negative/ latents

    Args:
        video_path: Path to the source video file
        vae: VAE model for encoding
        cache_dir: Directory to save cached latents (musubi_cache_positive)
        device: Torch device
        vae_dtype: Data type for VAE
        frame_extraction: Frame extraction mode
        target_frames: Target number of frames per chunk (int or list)
        frame_stride: Stride for slide mode
        frame_sample: Number of samples for uniform mode
        max_frames: Maximum frames for full mode
        vae_frame_stride: VAE frame stride
        control_args: Control video creation arguments
        bucket_selector: Optional BucketSelector for aspect ratio bucketing
    """
    try:
        # Load video frames (and get original FPS)
        frames, original_fps = load_video_frames(video_path)
        frame_count = frames.shape[0]
        logger.info(f"Loaded video: {video_path}, shape: {frames.shape}, fps: {original_fps}")

        # Step 1: Create directories
        base_dir = os.path.dirname(cache_dir)
        negative_dir = os.path.join(base_dir, "negative")
        neg_cache_dir = os.path.join(base_dir, "musubi_cache_negative")
        os.makedirs(negative_dir, exist_ok=True)
        os.makedirs(neg_cache_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        # Step 2: Extract chunks based on frame_extraction mode
        chunks = extract_frames(
            frames, frame_extraction, target_frames, frame_stride,
            frame_sample, max_frames, vae_frame_stride
        )
        logger.info(f"Extracted {len(chunks)} chunk(s) using mode '{frame_extraction}'")

        # Get basename without extension
        basename = os.path.splitext(os.path.basename(video_path))[0]

        # Step 3: Process each chunk
        for start_idx, num_frames, chunk_frames in chunks:
            chunk_suffix = f"_{start_idx:05d}-{num_frames:03d}"

            # 3a. Create control video based on control_args and save as MP4 to negative/
            if control_args is not None:
                control_type = control_args[0]  # "jump", "fade", "reverse", or "freeze"
                control_num = control_args[1]  # Already converted to int

                logger.info(f"Creating control video with type={control_type}, num={control_num}")

                if control_type == "jump":
                    control_frames = create_control_video_jump(chunk_frames, control_num)
                elif control_type == "freeze":
                    control_frames = create_control_video(chunk_frames)
                elif control_type == "fade":
                    control_frames = create_control_video_fade(chunk_frames, control_num)
                elif control_type == "reverse":
                    control_frames = create_control_video_reverse(chunk_frames, control_num)
                else:
                    logger.error(f"Unknown control type '{control_type}'")
                    raise ValueError(f"Unknown control type '{control_type}'")
            else:
                control_frames = create_control_video(chunk_frames)

            # Resize control frames to bucket AR size before saving MP4
            control_frames_resized = resize_frames_for_mp4(control_frames, bucket_selector)
            logger.info(f"Resized control frames from {control_frames.shape} to {control_frames_resized.shape}")

            # Save control MP4 to negative/ folder (with resized frames)
            negative_mp4_path = os.path.join(negative_dir, f"{basename}{chunk_suffix}.mp4")
            save_control_video_mp4(control_frames_resized, negative_mp4_path, fps=original_fps)
            logger.info(f"Saved negative MP4 to: {negative_mp4_path}")

            # 3b. Encode original chunk and save to musubi_cache_positive/
            original_latent = encode_video_to_latent(chunk_frames, vae, device, vae_dtype, bucket_selector)
            original_latent_f32 = original_latent.float()
            original_path = os.path.join(cache_dir, f"{basename}{chunk_suffix}_ltx2.safetensors")
            original_item = ItemInfo(
                item_key=original_path,
                content=original_latent_f32.numpy(),
                caption="",
                original_size=chunk_frames.shape[1:3],
                latent_cache_path=original_path,
            )
            save_latent_cache_ltx2(original_item, original_latent_f32)
            logger.info(f"Saved original latents to: {original_path}")

            # 3c. Encode control video and save to musubi_cache_negative/
            control_latent = encode_video_to_latent(control_frames, vae, device, vae_dtype, bucket_selector)
            control_latent_f32 = control_latent.float()
            control_path = os.path.join(neg_cache_dir, f"{basename}{chunk_suffix}_ltx2.safetensors")
            control_item = ItemInfo(
                item_key=control_path,
                content=control_latent_f32.numpy(),
                caption="",
                original_size=control_frames.shape[1:3],
                latent_cache_path=control_path,
            )
            save_latent_cache_ltx2(control_item, control_latent_f32)
            logger.info(f"Saved control latents to: {control_path}")

            # Clean up this chunk (including resized frames)
            del chunk_frames, control_frames, control_frames_resized
            del original_latent, control_latent
            del original_latent_f32, control_latent_f32
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Clean up
        del frames
        if device.type == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Failed to process {video_path}: {e}")
        import traceback
        traceback.print_exc()


def process_video_directory_i2v(
    video_dir: str,
    vae,
    device: torch.device,
    vae_dtype: torch.dtype,
    frame_extraction: str = "head",
    target_frames = 17,
    frame_stride: int = 1,
    frame_sample: int = 1,
    max_frames: int = 128,
    vae_frame_stride: int = 8,
    control_args: Optional[List[str]] = None,
    bucket_selector: BucketSelector = None,
) -> None:
    """Process all videos in a directory for i2v slider training.

    For slider training mode, creates three directories:
    - negative/ for control MP4 videos (created using control_args)
    - musubi_cache_positive/ for original latents
    - musubi_cache_negative/ for control latents

    Args:
        video_dir: Directory containing video files
        vae: VAE model for encoding
        device: Torch device
        vae_dtype: Data type for VAE
        frame_extraction: Frame extraction mode
        target_frames: Target number of frames per chunk (int or list)
        frame_stride: Stride for slide mode
        frame_sample: Number of samples for uniform mode
        max_frames: Maximum frames for full mode
        vae_frame_stride: VAE frame stride
        control_args: Control video creation arguments
        bucket_selector: Optional BucketSelector for aspect ratio bucketing
    """
    # Find all video files
    video_files = []
    for ext in ("mp4", "webm", "mov", "avi", "mkv"):
        video_files.extend(glob.glob(os.path.join(video_dir, f"*.{ext}")))
        video_files.extend(glob.glob(os.path.join(video_dir, f"*.{ext.upper()}")))

    if not video_files:
        logger.warning(f"No video files found in {video_dir}")
        return

    logger.info(f"Found {len(video_files)} video files in {video_dir}")

    # Create cache directory for musubi_cache_positive (not cache_musubi)
    cache_dir = os.path.join(video_dir, "musubi_cache_positive")
    os.makedirs(cache_dir, exist_ok=True)

    # Process each video
    for video_path in video_files:
        process_single_video_i2v(
            video_path,
            vae,
            cache_dir,
            device,
            vae_dtype,
            frame_extraction,
            target_frames,
            frame_stride,
            frame_sample,
            max_frames,
            vae_frame_stride,
            control_args,
            bucket_selector,
        )


def create_bucket_selector(
    resolution: Tuple[int, int],
    enable_bucket: bool = True,
    no_upscale: bool = False,
    enable_ar_bucket: bool = True,
    min_ar: float = 0.5,
    max_ar: float = 2.0,
    num_ar_buckets: int = 2,
) -> BucketSelector:
    """Create a BucketSelector instance with the given parameters.

    Args:
        resolution: Base resolution (width, height)
        enable_bucket: Whether to enable bucketing
        no_upscale: Whether to disable upscaling in bucketing
        enable_ar_bucket: Whether to enable aspect ratio bucketing
        min_ar: Minimum aspect ratio (width/height)
        max_ar: Maximum aspect ratio (width/height)
        num_ar_buckets: Number of aspect ratio buckets

    Returns:
        Configured BucketSelector instance
    """
    return BucketSelector(
        resolution=resolution,
        enable_bucket=enable_bucket,
        no_upscale=no_upscale,
        architecture=ARCHITECTURE_LTX2,
        enable_ar_bucket=enable_ar_bucket,
        min_ar=min_ar,
        max_ar=max_ar,
        num_ar_buckets=num_ar_buckets,
    )
