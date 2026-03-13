#!/usr/bin/env python3
"""
Cache image encoder outputs (I2V latents) for LTX-2 sampling.

This module handles encoding of start_images to latents using the VAE encoder,
separated from text encoder caching for cleaner modularity.
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import nullcontext

import torch
from PIL import Image
from torchvision import transforms

from musubi_tuner.utils import model_utils
from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
from musubi_tuner.ltx_2.model.video_vae.model_configurator import (
    VideoEncoderConfigurator,
    VAE_ENCODER_COMFY_KEYS_FILTER,
)
from musubi_tuner.ltx_2.model.video_vae.video_vae import VideoEncoder

logger = logging.getLogger(__name__)


def load_vae_encoder(
    vae_path: str,
    device: torch.device,
    vae_dtype: torch.dtype | None = None,
) -> VideoEncoder:
    """
    Load LTX-2 VAE encoder for image-to-latent encoding.

    Args:
        vae_path: Path to LTX-2 checkpoint
        device: Device to load encoder on
        vae_dtype: Data type for encoder (ignored, always uses bfloat16 to match ltx-trainer)

    Returns:
        Loaded VAE encoder in eval mode
    """
    logger.info("Loading VAE encoder from %s for I2V caching", vae_path)
    # CRITICAL: Use bfloat16 for VAE encoder to match ltx-trainer
    actual_dtype = torch.bfloat16
    vae_encoder = SingleGPUModelBuilder(
        model_path=str(vae_path),
        model_class_configurator=VideoEncoderConfigurator,
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=device, dtype=actual_dtype)
    vae_encoder.eval()
    vae_encoder.requires_grad_(False)
    logger.info("Loaded VAE encoder for I2V caching (dtype=%s)", actual_dtype)
    return vae_encoder


def encode_images_to_latents(
    images: list[Image.Image],
    vae_encoder: VideoEncoder,
    target_width: int,
    target_height: int,
    target_num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode PIL images to latents using LTX-2 VAE encoder.

    Following the reference implementation from ltx-trainer/validation_sampler.py:
    - Image input is converted to float32
    - Autocast uses bfloat16 for encoding

    Args:
        images: List of PIL Images in RGB mode
        vae_encoder: Loaded VAE encoder
        target_width: Target video width (must be divisible by 32)
        target_height: Target video height (must be divisible by 32)
        target_num_frames: Target number of frames (must be n*8+1)
        device: Device to encode on

    Returns:
        Latent tensor with shape (1, 128, num_frames, latent_h, latent_w)
    """
    # Convert to tensor and normalize to [0, 1]
    to_tensor = transforms.ToTensor()
    image_tensors = [to_tensor(img) for img in images]
    images_tensor = torch.stack(image_tensors)  # (N, 3, H, W)

    # Scale and center crop to target dimensions
    N, C, orig_h, orig_w = images_tensor.shape

    # Calculate scale factor to cover the target dimensions
    scale_h = target_height / orig_h
    scale_w = target_width / orig_w
    scale = max(scale_h, scale_w)  # Scale to cover

    # Calculate scaled size
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))

    # Resize
    resized_images = torch.nn.functional.interpolate(
        images_tensor,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False
    )  # (N, 3, new_h, new_w)

    # Center crop to exact target dimensions
    top = (new_h - target_height) // 2
    left = (new_w - target_width) // 2
    cropped_images = resized_images[
        :, :,  # N, C
        top:top + target_height,
        left:left + target_width
    ]  # (N, 3, target_height, target_width)

    # Normalize to [-1, 1]
    resized_images = cropped_images * 2.0 - 1.0  # (N, 3, H, W)

    # Reshape to (B, C, F, H, W) format expected by encoder
    if len(images) == 1:
        # Single image: encode as (1, 3, 1, H, W) - 1 frame video
        video_input = resized_images.unsqueeze(2)  # (1, 3, 1, H, W)
        # CRITICAL: Use float32 for VAE input, bfloat16 autocast for encoding (matching ltx-trainer)
        video_input = video_input.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
                encoder_output = vae_encoder(video_input)
            start_images_latents = encoder_output
    else:
        # Multiple images: encode each and concatenate on frame dimension
        latent_list = []
        for i in range(resized_images.shape[0]):
            img = resized_images[i:i+1]  # (1, 3, H, W)
            video_input = img.unsqueeze(2)  # (1, 3, 1, H, W)
            # CRITICAL: Use float32 for VAE input, bfloat16 autocast for encoding (matching ltx-trainer)
            video_input = video_input.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
                    encoder_output = vae_encoder(video_input)
                latent_list.append(encoder_output)
        # Concatenate on frame dimension: (1, 128, N, H_latent, W_latent)
        start_images_latents = torch.cat(latent_list, dim=2)

    # CRITICAL: Apply VAE scaling factor to match LTX-2 expectations
    # For LTX-2, the scaling factor is 1.0 (no scaling needed)
    vae_scaling_factor = 1.0
    start_images_latents = start_images_latents * vae_scaling_factor
    logger.debug("Applied VAE scaling factor %s to start_images_latents", vae_scaling_factor)

    return start_images_latents


def validate_video_dims(raw_width: int, raw_height: int, raw_frames: int) -> tuple[int, int, int]:
    """
    Validate and adjust video dimensions to LTX-2 requirements.

    Requirements:
    - Width and height must be divisible by 32
    - Frames must be in the form n*8+1 (1, 9, 17, 25, 33, 41, 49, 57, 65, ...)

    Args:
        raw_width: Desired width
        raw_height: Desired height
        raw_frames: Desired number of frames

    Returns:
        Tuple of (validated_width, validated_height, validated_frames)
    """
    # Validate and adjust width/height to be divisible by 32
    target_width = round(raw_width / 32) * 32
    target_height = round(raw_height / 32) * 32
    # Ensure at least 32
    target_width = max(32, target_width)
    target_height = max(32, target_height)

    # Validate and adjust frames to be n*8+1 (1, 9, 17, 25, 33, 41, 49, 57, 65, ...)
    target_num_frames = round((raw_frames - 1) / 8) * 8 + 1
    # Ensure at least 1 frame
    target_num_frames = max(1, target_num_frames)

    if (raw_width != target_width or raw_height != target_height or
        raw_frames != target_num_frames):
        logger.info(
            "Adjusted video_dims from (%d, %d, %d) to (%d, %d, %d) "
            "(W,H must be divisible by 32, frames must be n*8+1)",
            raw_width, raw_height, raw_frames,
            target_width, target_height, target_num_frames
        )

    return target_width, target_height, target_num_frames


def parse_start_images_path(start_images_path: str | list) -> list[str]:
    """
    Parse start_images path specification into a list of file paths.

    Args:
        start_images_path: Can be a single string, comma-separated string, or list

    Returns:
        List of image file paths
    """
    if isinstance(start_images_path, str):
        # Handle comma-separated paths or single path
        return [p.strip() for p in start_images_path.split(",") if p.strip()]
    else:
        return [start_images_path]


def load_images(paths: list[str]) -> list[Image.Image]:
    """
    Load images from file paths.

    Args:
        paths: List of image file paths

    Returns:
        List of PIL Images in RGB mode

    Raises:
        FileNotFoundError: If any image file doesn't exist
        Exception: If image loading fails
    """
    images = []
    for img_path in paths:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        images.append(img)
    return images


def encode_prompt_images(
    prompt_dict: dict,
    vae_encoder: VideoEncoder,
    device: torch.device,
    vae_dtype: torch.dtype | None = None,
) -> dict | None:
    """
    Encode start_images for a single prompt dictionary to latents.

    Args:
        prompt_dict: Prompt dictionary containing 'start_images' or 'image_path' and 'video_dims'
        vae_encoder: Loaded VAE encoder
        device: Device to encode on
        vae_dtype: Data type for encoding (ignored, always uses bfloat16 to match ltx-trainer)

    Returns:
        Dictionary with 'start_images_latents' and 'start_images' keys, or None if no images
    """
    # Check both start_images (from TOML config) and image_path (from --i flag in prompts.txt)
    start_images_path = prompt_dict.get("start_images") or prompt_dict.get("image_path", "")

    if not start_images_path or start_images_path == "none":
        return None

    prompt_text = prompt_dict.get("prompt", "")
    logger.info("Prompt '%s' has start_images_path: %s", prompt_text[:30], start_images_path)

    try:
        # Parse start_images path
        start_images_paths = parse_start_images_path(start_images_path)

        if not start_images_paths:
            return None

        # Load images
        images = load_images(start_images_paths)

        # Get target dimensions from prompt and validate
        # Support both video_dims string format ("640, 416, 65") and separate width/height/frames
        video_dims = prompt_dict.get("video_dims", None)
        if video_dims is None:
            # Try to get from separate width, height, frame_count keys
            width = prompt_dict.get("width")
            height = prompt_dict.get("height")
            frame_count = prompt_dict.get("frame_count")
            if width and height:
                # Use frames or default to 65
                frames = frame_count or 65
                video_dims = f"{width}, {height}, {frames}"
            else:
                # Default fallback
                video_dims = "640, 416, 65"

        if isinstance(video_dims, str):
            dims = [int(x.strip()) for x in video_dims.split(",")]
            raw_width, raw_height, raw_frames = dims[0], dims[1], dims[2]
        else:
            raw_width, raw_height, raw_frames = video_dims

        target_width, target_height, target_num_frames = validate_video_dims(
            raw_width, raw_height, raw_frames
        )

        # Encode images to latents
        start_images_latents = encode_images_to_latents(
            images,
            vae_encoder,
            target_width,
            target_height,
            target_num_frames,
            device,
        )

        return {
            "start_images_latents": start_images_latents.cpu(),
            "start_images": start_images_path,
        }

    except Exception as e:
        logger.warning("Failed to cache start_images latents for prompt '%s': %s", prompt_text[:50], e)
        return None


def setup_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """
    Set up argument parser for image encoder caching.

    Args:
        parser: Existing parser to add arguments to, or None to create new parser

    Returns:
        Configured argument parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Cache image encoder outputs (I2V latents) for LTX-2 sampling"
        )

    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="Path to LTX-2 checkpoint for VAE encoder (defaults to --ltx2_checkpoint)",
    )
    parser.add_argument(
        "--ltx2_checkpoint",
        type=str,
        default=None,
        help="Path to LTX-2 checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--vae_dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for VAE encoder",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for encoding",
    )
    parser.add_argument(
        "--sample_prompts",
        type=str,
        required=True,
        help="Sample prompt file containing start_images paths",
    )
    parser.add_argument(
        "--output_cache",
        type=str,
        required=True,
        help="Output path for the image latents cache file (.pt)",
    )

    return parser


def decode_and_verify_latents(
    cache_entries: list[dict],
    vae_path: str,
    vae_dtype: torch.dtype,
    device: torch.device,
    output_dir: str,
) -> None:
    """
    Decode cached latents back to images for verification.

    Args:
        cache_entries: List of cache entries with 'start_images_latents'
        vae_path: Path to VAE checkpoint for decoder
        vae_dtype: Data type for VAE decoder (ignored, always uses bfloat16 to match ltx-trainer)
        device: Device to run decoding on
        output_dir: Directory to save verification images
    """
    if not cache_entries:
        return

    logger.info("Loading VAE decoder for latent verification...")
    try:
        from musubi_tuner.ltx_2.model.video_vae.model_configurator import (
            VideoDecoderConfigurator,
            VAE_DECODER_COMFY_KEYS_FILTER,
        )

        # CRITICAL: Use bfloat16 for VAE decoder to match encoder and ltx-trainer
        actual_dtype = torch.bfloat16
        vae_decoder = SingleGPUModelBuilder(
            model_path=str(vae_path),
            model_class_configurator=VideoDecoderConfigurator,
            model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
        ).build(device=device, dtype=actual_dtype)
        vae_decoder.eval()
        vae_decoder.requires_grad_(False)
        logger.info("Loaded VAE decoder for verification (dtype=%s)", actual_dtype)
    except Exception as e:
        logger.warning("Failed to load VAE decoder for verification: %s", e)
        return

    import torchvision.transforms as T

    for idx, cache_entry in enumerate(cache_entries):
        if cache_entry.get("start_images_latents") is not None:
            try:
                # CRITICAL: Preserve the encoded dtype (bfloat16), don't convert to float16
                latents = cache_entry["start_images_latents"].to(device=device)
                logger.info("Decoding I2V latents for verification (shape: %s, dtype: %s)...", list(latents.shape), latents.dtype)

                with torch.no_grad():
                    # Decode latents to pixel space
                    # The musubi-tuner VAE decoder returns different shapes depending on the input:
                    # - Single frame [1, C, 1, H, W] -> [1, 3, H, W] or [1, 3, 1, H, W]
                    decoded = vae_decoder(latents)
                    logger.info("Decoded output shape: %s, dtype: %s", list(decoded.shape), decoded.dtype)

                # Convert to [0, 1] range
                decoded = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)

                # Handle different output formats from the VAE decoder
                # Expected: [1, 3, H, W] or [1, 3, 1, H, W] -> convert to [3, H, W] for PIL
                if decoded.dim() == 4:
                    # [1, 3, H, W] -> [3, H, W]
                    decoded_image = decoded[0].cpu().float()
                elif decoded.dim() == 5:
                    # [1, 3, 1, H, W] -> [3, H, W]
                    decoded_image = decoded[0, :, 0].cpu().float()
                else:
                    raise ValueError(f"Unexpected decoded shape: {decoded.shape}, expected 4D or 5D tensor")

                # Convert to PIL Image (expects [C, H, W] in float32)
                to_pil = T.ToPILImage()
                pil_image = to_pil(decoded_image)

                # Save verification image
                verify_path = os.path.join(output_dir, f"start_images_decoded_verify_{idx}.png")
                pil_image.save(verify_path)
                logger.info("Saved decoded verification image to %s", verify_path)
            except Exception as e:
                logger.warning("Failed to decode verification image for entry %d: %s", idx, e)

    # Cleanup
    del vae_decoder
    logger.info("Unloaded VAE decoder after verification")


def main() -> None:
    """Main entry point for standalone image encoder caching."""
    logging.basicConfig(level=logging.INFO)

    parser = setup_parser()
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Get VAE dtype from args
    vae_dtype = model_utils.str_to_dtype(args.vae_dtype)

    # Get VAE path from args
    vae_path = args.vae or args.ltx2_checkpoint
    if vae_path is None:
        raise ValueError("--vae or --ltx2_checkpoint is required")

    # Load prompts
    from musubi_tuner.hv_train_network import load_prompts
    prompts = load_prompts(args.sample_prompts)
    if not prompts:
        raise ValueError(f"No prompts found in {args.sample_prompts}")

    # Check if any prompts have start_images
    has_start_images = any(
        (p.get("start_images") and p.get("start_images") != "none") or
        (p.get("image_path") and p.get("image_path") != "none")
        for p in prompts
    )

    if not has_start_images:
        logger.info("No start_images found in prompts, nothing to cache")
        return

    # Load VAE encoder
    vae_encoder = load_vae_encoder(vae_path, device, vae_dtype)

    # Encode images for each prompt
    cache_entries = []
    for prompt_dict in prompts:
        result = encode_prompt_images(prompt_dict, vae_encoder, device, vae_dtype)
        if result is not None:
            cache_entries.append(result)

    # Save cache
    torch.save(cache_entries, args.output_cache)
    logger.info("Saved %d image latent entries to %s", len(cache_entries), args.output_cache)

    # Cleanup encoder
    del vae_encoder
    logger.info("Unloaded VAE encoder")

    # Decode verification: verify encoding by decoding latents back to images
    output_dir = os.path.dirname(args.output_cache)
    decode_and_verify_latents(cache_entries, vae_path, vae_dtype, device, output_dir)


if __name__ == "__main__":
    main()
