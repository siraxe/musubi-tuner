"""LTX-2 Slider LoRA Training Implementation.

Trains a LoRA that shifts model output in a controllable direction
(e.g., "detailed" <-> "blurry") using bidirectional multiplier training.

Supports two modes:
  - text-only: learns direction from positive/negative prompt pairs (no dataset needed)
  - reference: learns direction from paired positive/negative latent examples
"""

import argparse
import glob
import importlib
import os
import re
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import toml
import torch
import torch.nn.functional as F_torch
from accelerate import Accelerator
from accelerate.utils import set_seed
import json
from safetensors import safe_open
from safetensors.torch import load_file
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    read_config_from_file,
    setup_parser_common,
    should_sample_images,
    prepare_accelerator,
    clean_memory_on_device,
)
from musubi_tuner.ltx2_train_network import (
    LTX2NetworkTrainer,
    ltx2_setup_parser,
)
from musubi_tuner.ltx_2.env import apply_ltx2_tweaks
from musubi_tuner.utils import model_utils, train_utils


# ---------------------------------------------------------------------------
# Data classes for slider config
# ---------------------------------------------------------------------------


@dataclass
class SliderTargetConfig:
    positive: str
    negative: str
    target_class: str = ""
    weight: float = 1.0


@dataclass
class SliderConfig:
    mode: str  # "text", "reference", or "i2v"
    targets: List[SliderTargetConfig] = field(default_factory=list)
    guidance_strength: float = 1.0
    frame_rate: int = 25
    sample_slider_range: List[float] = field(default_factory=lambda: [-2.0, -1.0, 0.0, 1.0, 2.0])
    batch_size: int = 1
    # Multiple directories support (required for reference mode)
    pos_cache_dirs: List[str] = field(default_factory=list)
    neg_cache_dirs: List[str] = field(default_factory=list)
    text_cache_dirs: List[str] = field(default_factory=list)
    # i2v mode: video cache directories containing original/control subdirs
    i2v_cache_dirs: List[str] = field(default_factory=list)
    # i2v mode: first frame conditioning probability (typically 1.0 for i2v)
    first_frame_conditioning_p: float = 1.0


def load_slider_config(path: str) -> SliderConfig:
    """Load slider configuration from a TOML file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = toml.load(f)

    mode = raw.get("mode", "text")
    guidance_strength = float(raw.get("guidance_strength", 1.0))
    frame_rate = int(raw.get("frame_rate", 25))
    sample_slider_range = raw.get("sample_slider_range", [-2.0, -1.0, 0.0, 1.0, 2.0])
    batch_size = int(raw.get("batch_size", 1))

    targets = []
    for t in raw.get("targets", []):
        targets.append(
            SliderTargetConfig(
                positive=t["positive"],
                negative=t["negative"],
                target_class=t.get("target_class", ""),
                weight=float(t.get("weight", 1.0)),
            )
        )

    # Load multi-dataset directories
    pos_cache_dirs = raw.get("pos_cache_dirs", [])
    neg_cache_dirs = raw.get("neg_cache_dirs", [])
    text_cache_dirs = raw.get("text_cache_dirs", pos_cache_dirs)  # Default to pos_cache_dirs

    # Load i2v mode directories
    i2v_cache_dirs = raw.get("i2v_cache_dirs", [])
    first_frame_conditioning_p = float(raw.get("first_frame_conditioning_p", 1.0))

    return SliderConfig(
        mode=mode,
        targets=targets,
        guidance_strength=guidance_strength,
        frame_rate=frame_rate,
        sample_slider_range=[float(v) for v in sample_slider_range],
        batch_size=batch_size,
        pos_cache_dirs=pos_cache_dirs,
        neg_cache_dirs=neg_cache_dirs,
        text_cache_dirs=text_cache_dirs,
        i2v_cache_dirs=i2v_cache_dirs,
        first_frame_conditioning_p=first_frame_conditioning_p,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm_like_tensor(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to have the same mean and std as the target tensor.

    Prevents slider targets from shifting the overall output magnitude,
    ensuring the LoRA learns a directional shift only.
    (Ported from ai-toolkit ConceptSliderTrainer.)
    """
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8) * target.std() + target.mean()


def _pad_and_batch(
    items: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad embeddings to max sequence length and stack into a batch.

    Each item is (embed [seq_len, dim], mask [seq_len]).
    Returns (batched_embed [B, max_seq, dim], batched_mask [B, max_seq]).
    """
    max_seq = max(e.shape[0] for e, _ in items)
    dim = items[0][0].shape[-1]

    embeds = []
    masks = []
    for embed, mask in items:
        seq = embed.shape[0]
        if seq < max_seq:
            pad_len = max_seq - seq
            # Gemma uses left-padding, so pad on the left
            embed = torch.cat([torch.zeros(pad_len, dim, dtype=embed.dtype, device=embed.device), embed], dim=0)
            mask = torch.cat([torch.zeros(pad_len, dtype=mask.dtype, device=mask.device), mask], dim=0)
        embeds.append(embed)
        masks.append(mask)

    batched_embed = torch.stack(embeds, dim=0).to(device=device, dtype=dtype)
    batched_mask = torch.stack(masks, dim=0).to(device=device, dtype=torch.int64)
    return batched_embed, batched_mask


# ---------------------------------------------------------------------------
# Cache file helpers
# ---------------------------------------------------------------------------


def _find_latent_tensor(sd: dict) -> torch.Tensor:
    """Extract the latent tensor from a safetensors state dict with dynamic keys."""
    for key, val in sd.items():
        if key.startswith("latents_") and not key.startswith("latents_mean"):
            return val
    raise KeyError(f"No latents key found in {list(sd.keys())}")


def _find_text_tensor(sd: dict) -> torch.Tensor:
    """Extract the text embedding tensor from a safetensors state dict with dynamic keys."""
    for key, val in sd.items():
        if key.startswith("text_") and key != "text_mask":
            return val
    raise KeyError(f"No text key found in {list(sd.keys())}")


def _get_latent_shape_from_file(file_path: str) -> tuple:
    """Get latent tensor shape from safetensors file without loading the full data.

    Reads only the safetensors header (JSON metadata) to get shape information.
    """
    try:
        # Safetensors format:
        # - First 8 bytes: header length (little endian uint64)
        # - Next N bytes: JSON metadata containing tensor names, shapes, dtypes, offsets
        with open(file_path, "rb") as f:
            # Read header length
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                raise ValueError("Invalid safetensors file: too short")
            header_len = int.from_bytes(header_len_bytes, byteorder="little", signed=False)

            # Read header JSON
            header_bytes = f.read(header_len)
            header_json = json.loads(header_bytes.decode("utf-8"))

            # Find the latent tensor (not text_ tensors) and get its shape
            for tensor_name, tensor_info in header_json.items():
                if not tensor_name.startswith("text_"):
                    shape = tensor_info.get("shape")
                    if shape:
                        return tuple(shape)

            raise ValueError(f"No latent tensor found in {file_path}")
    except Exception as e:
        logger.warning(f"Failed to read shape from header of {file_path}: {e}")
        # Last resort: load full file
        sd = load_file(file_path)
        tensor = _find_latent_tensor(sd)
        return tuple(tensor.shape)


# Match both formats:
# 1. With dimensions: {stem}_{W:04d}x{H:04d}_ltx2.safetensors
# 2. Without dimensions (for i2v preprocessed): {stem}_ltx2.safetensors
# 3. Chunk format: {stem}_{start:05d}-{frames:03d}_ltx2.safetensors
_LATENT_BASENAME_RE = re.compile(r"^(.+?)(?:_\d{4}x\d{4})?(?:_\d{5}-\d{3})?_ltx2\.safetensors$")


class PairedSliderDataset(torch.utils.data.Dataset):
    """Loads matched positive/negative latent pairs from multiple cache directories."""

    def __init__(self, pos_cache_dirs: List[str], neg_cache_dirs: List[str],
                 text_cache_dirs: Optional[List[str]] = None):
        if not pos_cache_dirs:
            raise ValueError("pos_cache_dirs cannot be empty")
        if not neg_cache_dirs:
            raise ValueError("neg_cache_dirs cannot be empty")

        self.pos_cache_dirs = pos_cache_dirs
        self.neg_cache_dirs = neg_cache_dirs
        self.text_cache_dirs = text_cache_dirs if text_cache_dirs else pos_cache_dirs

        # Ensure all lists have the same length
        n_dirs = len(self.pos_cache_dirs)
        if len(self.neg_cache_dirs) != n_dirs:
            raise ValueError(f"pos_cache_dirs has {n_dirs} entries but neg_cache_dirs has {len(self.neg_cache_dirs)}")
        if len(self.text_cache_dirs) != n_dirs:
            raise ValueError(f"pos_cache_dirs has {n_dirs} entries but text_cache_dirs has {len(self.text_cache_dirs)}")

        self.pairs = []
        # Store pairs with their shapes for bucket batching: (shape_key, pos_path, neg_path, te_path)
        self.buckets = {}  # shape -> list of indices

        for i, (pos_dir, neg_dir, text_dir) in enumerate(zip(self.pos_cache_dirs, self.neg_cache_dirs, self.text_cache_dirs)):
            # Find latent cache files in pos_cache_dir
            pos_files = sorted(glob.glob(os.path.join(pos_dir, "*_ltx2.safetensors")))
            # Exclude text encoder caches (*_te.safetensors) and audio (*_audio.safetensors)
            pos_files = [f for f in pos_files if not f.endswith("_te.safetensors") and not f.endswith("_audio.safetensors")]

            for pos_path in pos_files:
                basename = os.path.basename(pos_path)
                neg_path = os.path.join(neg_dir, basename)
                if not os.path.exists(neg_path):
                    logger.warning("No negative match for %s in %s, skipping", basename, neg_dir)
                    continue

                # Text cache uses stem without WxH dimensions:
                #   latent: {stem}_{W:04d}x{H:04d}_ltx2.safetensors
                #   text:   {stem}_ltx2_te.safetensors
                m = _LATENT_BASENAME_RE.match(basename)
                if not m:
                    logger.warning("Cannot parse latent filename %s, skipping", basename)
                    continue
                te_basename = f"{m.group(1)}_ltx2_te.safetensors"
                te_path = os.path.join(text_dir, te_basename)
                if not os.path.exists(te_path):
                    logger.warning("No text cache for %s in %s, skipping", basename, text_dir)
                    continue

                # Get the shape of this pair for bucketing (read header only, no data loading)
                shape = _get_latent_shape_from_file(pos_path)
                shape_key = shape  # Use full shape as bucket key

                idx = len(self.pairs)
                self.pairs.append((shape_key, pos_path, neg_path, te_path))

                if shape_key not in self.buckets:
                    self.buckets[shape_key] = []
                self.buckets[shape_key].append(idx)

        if len(self.pairs) == 0:
            raise ValueError(f"No matched pairs found in the provided cache directories")
        logger.info("PairedSliderDataset: found %d matched pairs from %d dataset(s) in %d shape buckets",
                   len(self.pairs), n_dirs, len(self.buckets))
        # Log bucket info
        for shape, indices in sorted(self.buckets.items()):
            # shape is (C, F, H, W) where C=128, F=latent_frames, H=latent_height, W=latent_width
            # Convert latent frames to actual video frames: actual = (latent - 1) * 8 + 1
            # Convert latent resolution to pixels: H*32 x W*32
            c, latent_f, h, w = shape
            actual_f = (latent_f - 1) * 8 + 1
            pixel_h = h * 32
            pixel_w = w * 32
            logger.info("  Bucket shape %s: %d items (frames: %d, resolution: %dx%d)",
                       shape, len(indices), actual_f, pixel_h, pixel_w)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        shape_key, pos_path, neg_path, te_path = self.pairs[idx]

        pos_sd = load_file(pos_path)
        neg_sd = load_file(neg_path)
        te_sd = load_file(te_path)

        pos_latents = _find_latent_tensor(pos_sd)    # [C, F, H, W]
        neg_latents = _find_latent_tensor(neg_sd)     # [C, F, H, W]
        if pos_latents.shape != neg_latents.shape:
            raise ValueError(
                f"Shape mismatch for {os.path.basename(pos_path)}: "
                f"pos {pos_latents.shape} vs neg {neg_latents.shape}"
            )
        text_embeds = _find_text_tensor(te_sd)        # [seq_len, dim]
        text_mask = te_sd.get("text_mask", torch.ones(text_embeds.shape[0]))  # [seq_len]

        return {
            "pos_latents": pos_latents,
            "neg_latents": neg_latents,
            "text_embeds": text_embeds,
            "text_mask": text_mask,
        }


class SliderBucketBatchSampler:
    """Sampler that groups batches by latent shape to enable true batching."""

    def __init__(self, dataset: PairedSliderDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Build batch indices for each bucket
        self.bucket_batches = []  # list of (shape, [indices])

        for shape, indices in dataset.buckets.items():
            # Create batches within this bucket
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                self.bucket_batches.append((shape, batch_indices))

        if shuffle:
            import random
            random.shuffle(self.bucket_batches)

    def __iter__(self):
        for shape, indices in self.bucket_batches:
            yield indices

    def __len__(self):
        return len(self.bucket_batches)


class I2VSliderDataset(torch.utils.data.Dataset):
    """Loads original/control video pairs for i2v slider training.

    Expected file structure for each cache_dir:
        {cache_dir}/{basename}_ltx2.safetensors        # Original video latents
        {cache_dir}/{basename}_control_ltx2.safetensors # Control video latents (repeated first frame)
        {cache_dir}/{basename}_ltx2_te.safetensors      # Text encoder outputs (optional)
    """

    def __init__(self, i2v_cache_dirs: List[str], text_cache_dirs: Optional[List[str]] = None):
        if not i2v_cache_dirs:
            raise ValueError("i2v_cache_dirs cannot be empty")

        self.i2v_cache_dirs = i2v_cache_dirs
        self.text_cache_dirs = text_cache_dirs if text_cache_dirs else i2v_cache_dirs

        # Ensure all lists have the same length
        n_dirs = len(self.i2v_cache_dirs)
        if len(self.text_cache_dirs) != n_dirs:
            raise ValueError(f"i2v_cache_dirs has {n_dirs} entries but text_cache_dirs has {len(self.text_cache_dirs)}")

        self.pairs = []
        self.buckets = {}  # shape -> list of indices

        for i, (i2v_dir, text_dir) in enumerate(zip(self.i2v_cache_dirs, self.text_cache_dirs)):
            if not os.path.exists(i2v_dir):
                logger.warning("Cache directory not found: %s, skipping", i2v_dir)
                continue

            # Find original latent files (not control files)
            original_files = sorted(glob.glob(os.path.join(i2v_dir, "*_ltx2.safetensors")))
            # Filter out control files
            original_files = [f for f in original_files if "_control_" not in os.path.basename(f)]

            for original_path in original_files:
                basename = os.path.basename(original_path)
                # Create control filename by inserting _control before _ltx2
                control_basename = basename.replace("_ltx2.safetensors", "_control_ltx2.safetensors")
                control_path = os.path.join(i2v_dir, control_basename)

                if not os.path.exists(control_path):
                    logger.warning("No control match for %s (expected %s), skipping", basename, control_basename)
                    continue

                # Text cache uses stem without WxH dimensions
                m = _LATENT_BASENAME_RE.match(basename)
                if not m:
                    logger.warning("Cannot parse latent filename %s, skipping", basename)
                    continue

                stem = m.group(1)
                # Try to find matching text encoder cache
                # Text cache can be: {stem}_ltx2_te.safetensors or {stem}_{WxH}_ltx2_te.safetensors
                # For chunked files (e.g., H1_47_00001_00000-049), also try without chunk suffix (H1_47_00001)
                te_path = None
                te_stems_to_try = [stem]

                # For chunked files, also try the base stem without chunk suffix (_XXXXX-XXX)
                chunk_suffix_match = re.match(r"^(.+?)_\d{5}-\d{3}$", stem)
                if chunk_suffix_match:
                    base_stem = chunk_suffix_match.group(1)
                    te_stems_to_try.append(base_stem)

                import glob as glob_module
                for te_stem in te_stems_to_try:
                    # First try without dimensions
                    te_basename = f"{te_stem}_ltx2_te.safetensors"
                    te_candidate = os.path.join(text_dir, te_basename)
                    if os.path.exists(te_candidate):
                        te_path = te_candidate
                        break

                    # Try with dimensions wildcard
                    te_pattern = os.path.join(text_dir, f"{te_stem}_*_ltx2_te.safetensors")
                    te_matches = glob_module.glob(te_pattern)
                    if te_matches:
                        te_path = te_matches[0]
                        break

                if not te_path:
                    logger.warning("No text cache for %s in %s, skipping", basename, text_dir)
                    continue

                # Get the shape of this pair for bucketing (read header only, no data loading)
                shape = _get_latent_shape_from_file(original_path)
                shape_key = shape

                idx = len(self.pairs)
                self.pairs.append((shape_key, original_path, control_path, te_path))

                if shape_key not in self.buckets:
                    self.buckets[shape_key] = []
                self.buckets[shape_key].append(idx)

        if len(self.pairs) == 0:
            raise ValueError(f"No matched pairs found in the provided i2v cache directories")
        logger.info("I2VSliderDataset: found %d matched pairs from %d dataset(s) in %d shape buckets",
                   len(self.pairs), n_dirs, len(self.buckets))
        for shape, indices in sorted(self.buckets.items()):
            # shape is (C, F, H, W) where C=128, F=latent_frames, H=latent_height, W=latent_width
            # Convert latent frames to actual video frames: actual = (latent - 1) * 8 + 1
            # Convert latent resolution to pixels: H*32 x W*32
            c, latent_f, h, w = shape
            actual_f = (latent_f - 1) * 8 + 1
            pixel_h = h * 32
            pixel_w = w * 32
            logger.info("  Bucket shape %s: %d items (frames: %d, resolution: %dx%d)",
                       shape, len(indices), actual_f, pixel_h, pixel_w)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        shape_key, original_path, control_path, te_path = self.pairs[idx]

        original_sd = load_file(original_path)
        control_sd = load_file(control_path)
        te_sd = load_file(te_path)

        original_latents = _find_latent_tensor(original_sd)  # [C, F, H, W]
        control_latents = _find_latent_tensor(control_sd)    # [C, F, H, W]
        if original_latents.shape != control_latents.shape:
            raise ValueError(
                f"Shape mismatch for {os.path.basename(original_path)}: "
                f"original {original_latents.shape} vs control {control_latents.shape}"
            )
        text_embeds = _find_text_tensor(te_sd)  # [seq_len, dim]
        text_mask = te_sd.get("text_mask", torch.ones(text_embeds.shape[0]))  # [seq_len]

        return {
            "pos_latents": original_latents,   # Use original as "positive" (motion)
            "neg_latents": control_latents,    # Use control as "negative" (no motion)
            "text_embeds": text_embeds,
            "text_mask": text_mask,
        }


def _slider_collate_fn(batch):
    """Collate function that stacks tensors from the same shape bucket."""
    # batch is a list of dicts with pos_latents, neg_latents, text_embeds, text_mask
    # All items in the batch have the same latent shape (due to bucketing)

    pos_latents = torch.stack([item["pos_latents"] for item in batch])  # [B, C, F, H, W]
    neg_latents = torch.stack([item["neg_latents"] for item in batch])  # [B, C, F, H, W]

    # Text embeddings may have different seq lengths, so pad them
    text_embeds_list = [item["text_embeds"] for item in batch]
    text_mask_list = [item["text_mask"] for item in batch]

    # Find max seq length
    max_seq_len = max(embed.shape[0] for embed in text_embeds_list)

    # Pad to max length
    padded_embeds = []
    padded_masks = []
    for embed, mask in zip(text_embeds_list, text_mask_list):
        seq_len = embed.shape[0]
        if seq_len < max_seq_len:
            # Pad with zeros
            pad_size = max_seq_len - seq_len
            embed_pad = torch.nn.functional.pad(embed, (0, 0, 0, pad_size))
            mask_pad = torch.nn.functional.pad(mask, (0, pad_size))
            padded_embeds.append(embed_pad)
            padded_masks.append(mask_pad)
        else:
            padded_embeds.append(embed)
            padded_masks.append(mask)

    text_embeds = torch.stack(padded_embeds)  # [B, seq_len, dim]
    text_mask = torch.stack(padded_masks)  # [B, seq_len]

    return {
        "pos_latents": pos_latents,
        "neg_latents": neg_latents,
        "text_embeds": text_embeds,
        "text_mask": text_mask,
    }


# ---------------------------------------------------------------------------
# Main trainer class
# ---------------------------------------------------------------------------


class LTX2SliderTrainer:
    def __init__(self):
        self._net_trainer = LTX2NetworkTrainer()
        self.slider_config: Optional[SliderConfig] = None
        self.cached_embeds: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    # -- Prompt pre-caching --------------------------------------------------

    def _precache_slider_prompts(self, args: argparse.Namespace, accelerator: Accelerator) -> None:
        """Load Gemma, encode all unique slider prompts, unload."""
        te_dtype = self._net_trainer._build_text_encoder(args, accelerator)

        prompts: set = set()
        for target in self.slider_config.targets:
            prompts.add(target.positive)
            prompts.add(target.negative)
            prompts.add(target.target_class)
        prompts.add("")  # neutral / empty prompt

        for prompt_text in prompts:
            embed, mask = self._net_trainer._encode_prompt_text(accelerator, prompt_text, te_dtype)
            self.cached_embeds[prompt_text] = (embed, mask)  # [seq_len, dim], [seq_len] on CPU

        self._net_trainer._cleanup_text_encoder(accelerator)
        clean_memory_on_device(accelerator.device)
        logger.info("Cached %d unique slider prompt embeddings", len(self.cached_embeds))

    # -- Text-only slider step -----------------------------------------------

    def _text_slider_step(
        self,
        transformer,
        network,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_dtype: torch.dtype,
    ) -> float:
        """One training step for text-only slider mode."""
        device = accelerator.device

        target = random.choice(self.slider_config.targets)

        # Synthetic noise latents
        latent_frames = getattr(args, "latent_frames", 1)
        latent_height = getattr(args, "latent_height", 512) // 32
        latent_width = getattr(args, "latent_width", 768) // 32
        noise = torch.randn(1, 128, latent_frames, latent_height, latent_width, device=device, dtype=torch.float32)

        # Sample sigma from shifted logit-normal
        seq_len = latent_frames * latent_height * latent_width
        shift = LTX2NetworkTrainer._shifted_logit_normal_shift_for_sequence_length(seq_len)
        sigma = torch.sigmoid(torch.randn(1, device=device) + shift)
        sigma_exp = sigma.view(1, 1, 1, 1, 1)
        noisy = sigma_exp * noise  # pure noise scaled by sigma

        # Timestep for model: sigma as [B, 1]
        model_ts = sigma.unsqueeze(1)

        # Get cached embeddings
        pos_e, pos_m = self.cached_embeds[target.positive]
        neg_e, neg_m = self.cached_embeds[target.negative]
        neu_e, neu_m = self.cached_embeds[""]
        tgt_e, tgt_m = self.cached_embeds[target.target_class]

        # Pad and batch for 3-pass: [positive, neutral, negative]
        text_3x, mask_3x = _pad_and_batch(
            [(pos_e, pos_m), (neu_e, neu_m), (neg_e, neg_m)], device, dit_dtype
        )

        # No-grad 3-pass forward (LoRA disabled)
        network.set_multiplier(0.0)
        noisy_3x = noisy.expand(3, -1, -1, -1, -1).to(dtype=dit_dtype)
        ts_3x = model_ts.expand(3, -1)

        with torch.no_grad():
            self._net_trainer._ensure_fp8_buffers_on_device(accelerator.unwrap_model(transformer))
            with accelerator.autocast():
                pred_3x = transformer(
                    noisy_3x,
                    timestep=ts_3x,
                    context=text_3x,
                    attention_mask=mask_3x,
                    frame_rate=self.slider_config.frame_rate,
                    transformer_options={},
                )

        pred_pos, pred_neu, pred_neg = pred_3x.chunk(3, dim=0)

        # Compute directional offset
        direction = pred_pos - pred_neg
        gs = self.slider_config.guidance_strength
        target_enhance = _norm_like_tensor(pred_neu + gs * direction, pred_neu).detach()
        target_erase = _norm_like_tensor(pred_neu - gs * direction, pred_neu).detach()

        del pred_3x, noisy_3x, ts_3x, text_3x, mask_3x, pred_pos, pred_neu, pred_neg, direction
        clean_memory_on_device(device)

        # Prepare target_class embeddings for training passes
        tgt_text, tgt_mask = _pad_and_batch([(tgt_e, tgt_m)], device, dit_dtype)
        noisy_dit = noisy.to(dtype=dit_dtype)

        # Store inputs for preservation (blank_preservation, dop)
        if self._net_trainer._preservation_active:
            self._net_trainer._last_dit_inputs = {
                "model_input": noisy_dit,
                "model_timesteps": model_ts,
                "audio_model_timesteps": None,
                "text_embeds": tgt_text,
                "text_mask": tgt_mask,
                "video_conditioning": None,
                "video_conditioning_mask": None,
                "audio_latents": None,
                "video_loss_weight": 1.0,
                "audio_loss_weight": 0.0,
                "frame_rate": self.slider_config.frame_rate,
                "transformer_options": {},
            }

        # Training pass 1: positive direction (multiplier=+1)
        network.set_multiplier(1.0)
        with accelerator.autocast():
            lora_pred_pos = transformer(
                noisy_dit,
                timestep=model_ts,
                context=tgt_text,
                attention_mask=tgt_mask,
                frame_rate=self.slider_config.frame_rate,
                transformer_options={},
            )
        loss_pos = F_torch.mse_loss(lora_pred_pos.float(), target_enhance.float())
        accelerator.backward(loss_pos * target.weight)

        del lora_pred_pos
        clean_memory_on_device(device)

        # Training pass 2: negative direction (multiplier=-1)
        network.set_multiplier(-1.0)
        with accelerator.autocast():
            lora_pred_neg = transformer(
                noisy_dit,
                timestep=model_ts,
                context=tgt_text,
                attention_mask=tgt_mask,
                frame_rate=self.slider_config.frame_rate,
                transformer_options={},
            )
        loss_neg = F_torch.mse_loss(lora_pred_neg.float(), target_erase.float())
        accelerator.backward(loss_neg * target.weight)

        del lora_pred_neg, noisy_dit, tgt_text, tgt_mask
        clean_memory_on_device(device)

        # Restore multiplier
        network.set_multiplier(1.0)
        return (loss_pos.item() + loss_neg.item()) / 2.0

    # -- Reference-based slider step -----------------------------------------

    def _reference_slider_step(
        self,
        transformer,
        network,
        batch: Dict[str, torch.Tensor],
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_dtype: torch.dtype,
    ) -> float:
        """One training step for reference-based slider mode.

        Supports batched training with multiple positive/negative pairs per batch.
        """
        device = accelerator.device

        pos_latents = batch["pos_latents"].to(device=device, dtype=torch.float32)  # [B, 128, F, H, W]
        neg_latents = batch["neg_latents"].to(device=device, dtype=torch.float32)
        text_embeds = batch["text_embeds"].to(device=device, dtype=dit_dtype)
        text_mask = batch["text_mask"].to(device=device, dtype=torch.int64)

        batch_size = pos_latents.shape[0]

        # Handle text embeddings: could be [seq_len, dim] or [B, seq_len, dim]
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        elif text_embeds.shape[0] != batch_size:
            text_embeds = text_embeds.expand(batch_size, -1, -1)

        # Handle text mask similarly
        if text_mask.dim() == 1:
            text_mask = text_mask.unsqueeze(0).expand(batch_size, -1)
        elif text_mask.shape[0] != batch_size:
            text_mask = text_mask.expand(batch_size, -1)

        # Same noise for both, per batch element
        noise = torch.randn_like(pos_latents)

        # Sample sigma per batch element
        seq_len = pos_latents.shape[2] * pos_latents.shape[3] * pos_latents.shape[4]
        shift = LTX2NetworkTrainer._shifted_logit_normal_shift_for_sequence_length(seq_len)
        sigma = torch.sigmoid(torch.randn(batch_size, device=device) + shift)  # [B]
        sigma_exp = sigma.view(batch_size, 1, 1, 1, 1)  # [B, 1, 1, 1, 1]

        # Create noisy versions (flow matching interpolation)
        noisy_pos = ((1.0 - sigma_exp) * pos_latents + sigma_exp * noise).to(dtype=dit_dtype)
        noisy_neg = ((1.0 - sigma_exp) * neg_latents + sigma_exp * noise).to(dtype=dit_dtype)
        model_ts = sigma.unsqueeze(1)  # [B, 1]

        # Flow matching velocity targets
        target_pos = (noise - pos_latents).to(dtype=dit_dtype)
        target_neg = (noise - neg_latents).to(dtype=dit_dtype)

        self._net_trainer._ensure_fp8_buffers_on_device(accelerator.unwrap_model(transformer))

        # Store inputs for preservation (blank_preservation, dop)
        if self._net_trainer._preservation_active:
            self._net_trainer._last_dit_inputs = {
                "model_input": noisy_pos,  # Use positive latents as reference
                "model_timesteps": model_ts,
                "audio_model_timesteps": None,
                "text_embeds": text_embeds,
                "text_mask": text_mask,
                "video_conditioning": None,
                "video_conditioning_mask": None,
                "audio_latents": None,
                "video_loss_weight": 1.0,
                "audio_loss_weight": 0.0,
                "frame_rate": self.slider_config.frame_rate,
                "transformer_options": {},
            }

        # Training pass: positive (multiplier=+1)
        network.set_multiplier(1.0)
        with accelerator.autocast():
            pred_pos = transformer(
                noisy_pos,
                timestep=model_ts,
                context=text_embeds,
                attention_mask=text_mask,
                frame_rate=self.slider_config.frame_rate,
                transformer_options={},
            )
        loss_pos = F_torch.mse_loss(pred_pos.float(), target_pos.float())
        accelerator.backward(loss_pos)

        del pred_pos
        clean_memory_on_device(device)

        # Training pass: negative (multiplier=-1)
        network.set_multiplier(-1.0)
        with accelerator.autocast():
            pred_neg = transformer(
                noisy_neg,
                timestep=model_ts,
                context=text_embeds,
                attention_mask=text_mask,
                frame_rate=self.slider_config.frame_rate,
                transformer_options={},
            )
        loss_neg = F_torch.mse_loss(pred_neg.float(), target_neg.float())
        accelerator.backward(loss_neg)

        del pred_neg
        clean_memory_on_device(device)

        network.set_multiplier(1.0)
        return (loss_pos.item() + loss_neg.item()) / 2.0

    # -- I2V slider step -------------------------------------------------------

    def _i2v_slider_step(
        self,
        transformer,
        network,
        batch: Dict[str, torch.Tensor],
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_dtype: torch.dtype,
    ) -> float:
        """One training step for i2v slider mode.

        Uses first_frame_conditioning to preserve the first frame while training
        the model to control motion amount on subsequent frames.

        pos_latents: Original video with motion (e.g., camera rotation)
        neg_latents: Control video with repeated first frame (no motion)
        """
        device = accelerator.device

        pos_latents = batch["pos_latents"].to(device=device, dtype=torch.float32)  # [B, 128, F, H, W]
        neg_latents = batch["neg_latents"].to(device=device, dtype=torch.float32)
        text_embeds = batch["text_embeds"].to(device=device, dtype=dit_dtype)
        text_mask = batch["text_mask"].to(device=device, dtype=torch.int64)

        batch_size = pos_latents.shape[0]

        # Handle text embeddings: could be [seq_len, dim] or [B, seq_len, dim]
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        elif text_embeds.shape[0] != batch_size:
            text_embeds = text_embeds.expand(batch_size, -1, -1)

        # Handle text mask similarly
        if text_mask.dim() == 1:
            text_mask = text_mask.unsqueeze(0).expand(batch_size, -1)
        elif text_mask.shape[0] != batch_size:
            text_mask = text_mask.expand(batch_size, -1)

        # Same noise for both, per batch element
        noise = torch.randn_like(pos_latents)

        # Sample sigma per batch element
        seq_len = pos_latents.shape[2] * pos_latents.shape[3] * pos_latents.shape[4]
        shift = LTX2NetworkTrainer._shifted_logit_normal_shift_for_sequence_length(seq_len)
        sigma = torch.sigmoid(torch.randn(batch_size, device=device) + shift)  # [B]
        sigma_exp = sigma.view(batch_size, 1, 1, 1, 1)  # [B, 1, 1, 1, 1]

        # Create noisy versions (flow matching interpolation)
        noisy_pos = ((1.0 - sigma_exp) * pos_latents + sigma_exp * noise).to(dtype=dit_dtype)
        noisy_neg = ((1.0 - sigma_exp) * neg_latents + sigma_exp * noise).to(dtype=dit_dtype)

        # Flow matching velocity targets
        target_pos = (noise - pos_latents).to(dtype=dit_dtype)
        target_neg = (noise - neg_latents).to(dtype=dit_dtype)

        # Enable first frame conditioning for i2v mode
        # This preserves the first frame (timestep=0) while training subsequent frames
        first_frame_p = self.slider_config.first_frame_conditioning_p
        num_frames = pos_latents.shape[2]
        video_conditioning_enabled = None
        if first_frame_p > 0.0 and num_frames > 1:
            # Always enable for i2v mode (typically first_frame_p=1.0)
            video_conditioning_enabled = torch.ones((batch_size,), device=device, dtype=torch.bool)

        self._net_trainer._ensure_fp8_buffers_on_device(accelerator.unwrap_model(transformer))

        # Create timestep tensor with first frame conditioning
        model_ts = sigma.unsqueeze(1)  # [B, 1]
        if video_conditioning_enabled is not None:
            # For first frame conditioning, we need to pass the conditioning info
            # to the transformer via transformer_options
            transformer_options = {
                "video_conditioning_enabled": video_conditioning_enabled,
            }
        else:
            transformer_options = {}

        # Store inputs for preservation (blank_preservation, dop)
        if self._net_trainer._preservation_active:
            self._net_trainer._last_dit_inputs = {
                "model_input": noisy_pos,  # Use positive latents as reference
                "model_timesteps": model_ts,
                "audio_model_timesteps": None,
                "text_embeds": text_embeds,
                "text_mask": text_mask,
                "video_conditioning": None,
                "video_conditioning_mask": None,
                "audio_latents": None,
                "video_loss_weight": 1.0,
                "audio_loss_weight": 0.0,
                "frame_rate": self.slider_config.frame_rate,
                "transformer_options": transformer_options,
            }

        # Training pass: positive (multiplier=+1) - original video with motion
        network.set_multiplier(1.0)
        with accelerator.autocast():
            pred_pos = transformer(
                noisy_pos,
                timestep=model_ts,
                context=text_embeds,
                attention_mask=text_mask,
                frame_rate=self.slider_config.frame_rate,
                transformer_options=transformer_options,
            )
        loss_pos = F_torch.mse_loss(pred_pos.float(), target_pos.float())
        accelerator.backward(loss_pos)

        del pred_pos
        clean_memory_on_device(device)

        # Training pass: negative (multiplier=-1) - control video without motion
        network.set_multiplier(-1.0)
        with accelerator.autocast():
            pred_neg = transformer(
                noisy_neg,
                timestep=model_ts,
                context=text_embeds,
                attention_mask=text_mask,
                frame_rate=self.slider_config.frame_rate,
                transformer_options=transformer_options,
            )
        loss_neg = F_torch.mse_loss(pred_neg.float(), target_neg.float())
        accelerator.backward(loss_neg)

        del pred_neg
        clean_memory_on_device(device)

        network.set_multiplier(1.0)
        return (loss_pos.item() + loss_neg.item()) / 2.0

    # -- Sampling at multiple slider strengths --------------------------------

    def _sample_slider(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        transformer,
        vae,
        network,
        sample_parameters,
        dit_dtype: torch.dtype,
        global_step: int,
        epoch: int = 0,
    ) -> None:
        """Generate preview samples at multiple slider strengths."""
        if sample_parameters is None:
            return

        slider_range = self.slider_config.sample_slider_range

        for mult in slider_range:
            network.set_multiplier(mult)
            logger.info("Sampling at slider multiplier=%.1f", mult)

            # Temporarily modify output_name to include multiplier
            original_name = args.output_name
            args.output_name = f"{original_name}_mult{mult:+.1f}"

            self._net_trainer.sample_images(
                accelerator, args, epoch, global_step, vae, transformer, sample_parameters, dit_dtype
            )

            args.output_name = original_name

        # Restore multiplier to training default
        network.set_multiplier(1.0)

    # -- Reference dataset ---------------------------------------------------

    def _build_reference_dataloader(self, args: argparse.Namespace):
        """Build a dataloader for reference-mode slider training.

        Loads matched positive/negative latent pairs from pre-cached directories.
        """
        cfg = self.slider_config
        dataset = PairedSliderDataset(
            pos_cache_dirs=cfg.pos_cache_dirs,
            neg_cache_dirs=cfg.neg_cache_dirs,
            text_cache_dirs=cfg.text_cache_dirs,
        )
        num_workers = min(getattr(args, "max_data_loader_n_workers", 2), os.cpu_count() or 1)

        # Use batch_size from slider config for bucket batching
        # If batch_size > 1, use bucket sampler to group same-shape items together
        batch_size = cfg.batch_size
        use_bucket_batching = batch_size > 1

        if use_bucket_batching:
            # Check if we have enough items per bucket for the requested batch_size
            min_bucket_size = min(len(indices) for indices in dataset.buckets.values())
            if min_bucket_size < batch_size:
                logger.warning(
                    f"Requested batch_size={batch_size} but smallest bucket only has {min_bucket_size} items. "
                    f"Reducing effective batch_size for small buckets. Use batch_size=1 or gradient_accumulation_steps for consistent batching."
                )

            # Use bucket sampler for true batching within each shape group
            batch_sampler = SliderBucketBatchSampler(dataset, batch_size=batch_size, shuffle=True)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=_slider_collate_fn,
                num_workers=num_workers,
                persistent_workers=num_workers > 0 and getattr(args, "persistent_data_loader_workers", False),
            )
        else:
            # batch_size=1: use simple dataloader without bucketing
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=num_workers > 0 and getattr(args, "persistent_data_loader_workers", False),
            )
        return dataloader

    # -- I2V dataset ----------------------------------------------------------

    def _build_i2v_dataloader(self, args: argparse.Namespace):
        """Build a dataloader for i2v-mode slider training.

        Loads original/control video pairs from pre-cached directories.
        """
        cfg = self.slider_config
        dataset = I2VSliderDataset(
            i2v_cache_dirs=cfg.i2v_cache_dirs,
            text_cache_dirs=cfg.text_cache_dirs,
        )
        num_workers = min(getattr(args, "max_data_loader_n_workers", 2), os.cpu_count() or 1)

        # Use batch_size from slider config for bucket batching
        batch_size = cfg.batch_size
        use_bucket_batching = batch_size > 1

        if use_bucket_batching:
            # Check if we have enough items per bucket for the requested batch_size
            min_bucket_size = min(len(indices) for indices in dataset.buckets.values())
            if min_bucket_size < batch_size:
                logger.warning(
                    f"Requested batch_size={batch_size} but smallest bucket only has {min_bucket_size} items. "
                    f"Reducing effective batch_size for small buckets. Use batch_size=1 or gradient_accumulation_steps for consistent batching."
                )

            # Use bucket sampler for true batching within each shape group
            batch_sampler = SliderBucketBatchSampler(dataset, batch_size=batch_size, shuffle=True)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=_slider_collate_fn,
                num_workers=num_workers,
                persistent_workers=num_workers > 0 and getattr(args, "persistent_data_loader_workers", False),
            )
        else:
            # batch_size=1: use simple dataloader without bucketing
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=num_workers > 0 and getattr(args, "persistent_data_loader_workers", False),
            )
        return dataloader

    # -- Main training entry --------------------------------------------------

    def train(self, args: argparse.Namespace) -> None:
        """Main slider training loop."""

        # CUDA settings
        if torch.cuda.is_available():
            if getattr(args, "cuda_memory_fraction", None) is not None:
                torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
            if getattr(args, "cuda_allow_tf32", False):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            if getattr(args, "cuda_cudnn_benchmark", False):
                torch.backends.cudnn.benchmark = True

        # Load slider config
        self.slider_config = load_slider_config(args.slider_config)
        logger.info("Slider mode: %s, targets: %d", self.slider_config.mode, len(self.slider_config.targets))

        # Override from CLI if given
        if getattr(args, "guidance_strength", None) is not None:
            self.slider_config.guidance_strength = args.guidance_strength
        if getattr(args, "sample_slider_range", None) is not None:
            self.slider_config.sample_slider_range = [float(v) for v in args.sample_slider_range.split(",")]
        if getattr(args, "first_frame_conditioning_p", None) is not None:
            self.slider_config.first_frame_conditioning_p = args.first_frame_conditioning_p

        # Validate
        if self.slider_config.mode not in {"text", "reference", "i2v"}:
            raise ValueError(f"Invalid slider mode '{self.slider_config.mode}'. Must be 'text', 'reference', or 'i2v'.")
        if self.slider_config.mode == "text" and len(self.slider_config.targets) == 0:
            raise ValueError("Text-only slider mode requires at least one target in slider config")
        if self.slider_config.mode == "reference":
            # Check for multi-dataset lists
            if not self.slider_config.pos_cache_dirs or not self.slider_config.neg_cache_dirs:
                raise ValueError("Reference slider mode requires pos_cache_dirs and neg_cache_dirs in slider config")
        if self.slider_config.mode == "i2v":
            # Check for i2v dataset lists
            if not self.slider_config.i2v_cache_dirs:
                raise ValueError("I2V slider mode requires i2v_cache_dirs in slider config")

        # Seed
        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        session_id = random.randint(0, 2**32)
        training_started_at = time.time()

        # Model-specific init (sets _ltx_mode, _audio_video, etc.)
        self._net_trainer.handle_model_specific_args(args)

        # Prepare accelerator
        accelerator = prepare_accelerator(args)
        if args.mixed_precision is None:
            args.mixed_precision = accelerator.mixed_precision
        is_main_process = accelerator.is_main_process

        # Precision
        dit_dtype = torch.bfloat16 if args.dit_dtype is None else model_utils.str_to_dtype(args.dit_dtype)
        dit_weight_dtype = (None if getattr(args, "fp8_scaled", False) else torch.float8_e4m3fn) if getattr(args, "fp8_base", False) else dit_dtype

        # -- Sample prompt setup (for preview during training) ----------------
        vae_dtype = torch.float16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
        sample_parameters = None
        vae = None
        if getattr(args, "sample_prompts", None):
            sample_parameters = self._net_trainer.process_sample_prompts(args, accelerator, args.sample_prompts)
            vae = self._net_trainer.load_vae(args, vae_dtype=vae_dtype, vae_path=args.vae)
            vae.requires_grad_(False)
            vae.eval()

        # -- Load transformer -------------------------------------------------
        blocks_to_swap = int(getattr(args, "blocks_to_swap", 0) or 0)
        self._net_trainer.blocks_to_swap = blocks_to_swap
        loading_device = "cpu" if blocks_to_swap > 0 else accelerator.device

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        logger.info("Loading DiT model from %s", args.dit)
        transformer = self._net_trainer.load_transformer(
            accelerator, args, args.dit, "torch", False, loading_device, dit_weight_dtype
        )
        transformer.eval()
        transformer.requires_grad_(False)

        if blocks_to_swap > 0:
            logger.info("Enable block swap: %d blocks", blocks_to_swap)
            transformer.enable_block_swap(
                blocks_to_swap, accelerator.device, supports_backward=True,
                use_pinned_memory=getattr(args, "use_pinned_memory_for_block_swap", False),
                swap_norms=getattr(args, "swap_norms", False),
            )
            transformer.move_to_device_except_swap_blocks(accelerator.device)

        # -- Create LoRA network -----------------------------------------------
        sys.path.append(os.path.dirname(__file__))
        network_module_imported = importlib.import_module(args.network_module)

        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value

        if hasattr(network_module_imported, "create_arch_network"):
            network = network_module_imported.create_arch_network(
                1.0, args.network_dim, args.network_alpha, vae, None, transformer,
                neuron_dropout=args.network_dropout, **net_kwargs,
            )
        else:
            network = network_module_imported.create_network(
                1.0, args.network_dim, args.network_alpha, vae, None, transformer, **net_kwargs,
            )
        if network is None:
            raise RuntimeError("Failed to create LoRA network")

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)

        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)

        if getattr(args, "network_weights", None) is not None:
            info = network.load_weights(args.network_weights)
            logger.info("Loaded network weights from %s: %s", args.network_weights, info)

        # Gradient checkpointing
        if args.gradient_checkpointing:
            blocks_to_ckpt = getattr(args, "blocks_to_checkpoint", -1)
            if getattr(args, "blockwise_checkpointing", False):
                transformer.enable_gradient_checkpointing(
                    args.gradient_checkpointing_cpu_offload,
                    weight_cpu_offloading=True,
                    blocks_to_checkpoint=blocks_to_ckpt,
                )
            else:
                transformer.enable_gradient_checkpointing(
                    args.gradient_checkpointing_cpu_offload,
                    blocks_to_checkpoint=blocks_to_ckpt,
                )
            try:
                network.enable_gradient_checkpointing(
                    args.gradient_checkpointing_cpu_offload,
                    weight_cpu_offloading=bool(getattr(args, "blockwise_checkpointing", False)),
                    blocks_to_checkpoint=blocks_to_ckpt,
                )
            except TypeError:
                network.enable_gradient_checkpointing()

        # -- Pre-cache slider prompt embeddings --------------------------------
        if self.slider_config.mode == "text":
            self._precache_slider_prompts(args, accelerator)

        # -- Pre-cache sample prompt embeddings --------------------------------
        if sample_parameters is not None:
            # Encode sample prompts if they don't already have embeddings
            needs_encoding = any(p.get("prompt_embeds") is None for p in sample_parameters)
            if needs_encoding:
                te_dtype = self._net_trainer._build_text_encoder(args, accelerator)
                for sp in sample_parameters:
                    if sp.get("prompt_embeds") is None:
                        embed, mask = self._net_trainer._encode_prompt_text(accelerator, sp.get("prompt", ""), te_dtype)
                        sp["prompt_embeds"] = embed
                        sp["prompt_attention_mask"] = mask
                        neg = sp.get("negative_prompt")
                        if neg:
                            ne, nm = self._net_trainer._encode_prompt_text(accelerator, neg, te_dtype)
                            sp["negative_prompt_embeds"] = ne
                            sp["negative_prompt_attention_mask"] = nm
                self._net_trainer._cleanup_text_encoder(accelerator)
                clean_memory_on_device(accelerator.device)

        # -- Optimizer & scheduler ---------------------------------------------
        trainable_params, lr_descriptions = network.prepare_optimizer_params(unet_lr=args.learning_rate)
        optimizer_name, optimizer_args_str, optimizer, optimizer_train_fn, optimizer_eval_fn = (
            NetworkTrainer().get_optimizer(args, trainable_params)
        )

        lr_scheduler = NetworkTrainer().get_lr_scheduler(args, optimizer, accelerator.num_processes)

        # -- Prepare with accelerator ------------------------------------------
        if dit_weight_dtype != dit_dtype and dit_weight_dtype is not None:
            transformer.to(dit_weight_dtype)

        if blocks_to_swap > 0:
            transformer = accelerator.prepare(transformer, device_placement=[False])
            accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)
            accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
        else:
            transformer = accelerator.prepare(transformer)

        network, optimizer, lr_scheduler = accelerator.prepare(network, optimizer, lr_scheduler)

        # -- Pre-train hooks (preservation, CREPA, etc.) -------------------------
        # This sets up blank_preservation, dop, prior_divergence, crepa if enabled
        self._net_trainer.pre_train_hook(args, accelerator, transformer=transformer, network=network)

        # -- Register hooks to exclude transformer from state saving ---------------
        # Only save the LoRA network, not the base transformer (13GB+)
        def save_model_hook(models, weights, output_dir):
            """Filter out all models except the LoRA network when saving state."""
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(network))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                models.pop(i)
                weights.pop(i)

        def load_model_hook(models, input_dir):
            """Filter out all models except the LoRA network when loading state."""
            import os
            from safetensors.torch import load_file

            # Check if this is an old state with multiple model files
            # Old states: model.safetensors (transformer) + model_1.safetensors (LoRA)
            # New states: model.safetensors (LoRA only)
            model_1_path = os.path.join(input_dir, "model_1.safetensors")

            if os.path.exists(model_1_path):
                # Old format: keep only model_1 (LoRA), discard model (transformer)
                logger.info("Detected old state format with separate transformer/LoRA files. Loading LoRA only.")
                # Load model_1 into the first model (LoRA network)
                if len(models) > 0:
                    try:
                        state_dict = load_file(model_1_path)
                        models[0].load_state_dict(state_dict, strict=False)
                    except Exception as e:
                        logger.warning(f"Failed to load model_1.safetensors: {e}")
                # Clear all models since we manually loaded
                models.clear()
            else:
                # New format: filter by model type
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, type(accelerator.unwrap_model(network))):
                        remove_indices.append(i)
                for i in reversed(remove_indices):
                    models.pop(i)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        if args.gradient_checkpointing:
            transformer.train()
        else:
            transformer.eval()

        accelerator.unwrap_model(network).prepare_grad_etc(transformer)

        # -- Resume training if specified ----------------------------------------
        global_step = 0
        current_epoch = 0
        if getattr(args, "resume", None):
            logger.info(f"Resuming training from state: {args.resume}")
            self._net_trainer.resume_from_local_or_hf_if_specified(accelerator, args)
            global_step = self._net_trainer._recover_global_step(args.resume) if not getattr(args, "resume_from_huggingface", False) else 0
            logger.info(f"Resumed from global_step: {global_step}")

        # -- Reference dataloader (if reference or i2v mode) -------------------------
        ref_dataloader = None
        if self.slider_config.mode == "reference":
            ref_dataloader = self._build_reference_dataloader(args)
        elif self.slider_config.mode == "i2v":
            ref_dataloader = self._build_i2v_dataloader(args)

        # -- Metadata ----------------------------------------------------------
        metadata = {
            "ss_session_id": session_id,
            "ss_training_started_at": training_started_at,
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_base_model_version": self._net_trainer.architecture_full_name,
            "ss_network_module": args.network_module,
            "ss_network_dim": args.network_dim,
            "ss_network_alpha": args.network_alpha,
            "ss_network_dropout": args.network_dropout,
            "ss_mixed_precision": args.mixed_precision,
            "ss_seed": args.seed,
            "ss_optimizer": optimizer_name + (f"({optimizer_args_str})" if optimizer_args_str else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_fp8_base": bool(getattr(args, "fp8_base", False)),
            "ss_slider_mode": self.slider_config.mode,
            "ss_slider_guidance_strength": self.slider_config.guidance_strength,
        }
        if args.network_args:
            metadata["ss_network_args"] = str(net_kwargs)

        if args.dit is not None:
            sd_model_name = args.dit
            if os.path.exists(sd_model_name):
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        metadata = {k: str(v) for k, v in metadata.items()}

        minimum_metadata = {}
        for key in ["ss_base_model_version", "ss_network_module", "ss_network_dim", "ss_network_alpha", "ss_network_args"]:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        # Init trackers
        if accelerator.is_main_process:
            init_kwargs = {}
            if getattr(args, "wandb_run_name", None):
                init_kwargs["wandb"] = {"name": args.wandb_run_name}
            if getattr(args, "log_tracker_config", None) is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            tracker_name = getattr(args, "log_tracker_name", None) or "slider_train"
            accelerator.init_trackers(
                tracker_name,
                config=train_utils.get_sanitized_config_or_none(args),
                init_kwargs=init_kwargs,
            )

        # -- Save / remove helpers ---------------------------------------------
        save_dtype = dit_dtype

        def save_model(ckpt_name: str, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)
            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata_to_save = minimum_metadata if getattr(args, "no_metadata", False) else metadata
            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)

            # ComfyUI conversion
            self._net_trainer.post_save_checkpoint_hook(args, ckpt_file, ckpt_name, accelerator, force_sync_upload)

            if getattr(args, "huggingface_repo_id", None) is not None:
                from musubi_tuner.utils import huggingface_utils
                huggingface_utils.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        # -- Training loop -----------------------------------------------------
        progress_bar = tqdm(
            range(args.max_train_steps), smoothing=0,
            disable=not accelerator.is_local_main_process, desc="steps",
        )

        # Initialize global_step/current_epoch if not already set by resume
        if not getattr(args, "resume", None):
            global_step = 0
            current_epoch = 0
        else:
            # Update progress bar to show resumed position
            progress_bar.update(global_step)

        loss_recorder = train_utils.LossRecorder()

        clean_memory_on_device(accelerator.device)
        optimizer_train_fn()
        optimizer.zero_grad(set_to_none=True)

        logger.info("Starting slider training")
        logger.info("  mode: %s", self.slider_config.mode)
        logger.info("  max_train_steps: %d", args.max_train_steps)
        logger.info("  learning_rate: %s", args.learning_rate)
        if self.slider_config.mode == "text":
            logger.info("  guidance_strength: %.2f", self.slider_config.guidance_strength)
            logger.info("  latent_frames: %d", getattr(args, "latent_frames", 1))
            logger.info("  latent_height: %d", getattr(args, "latent_height", 512))
            logger.info("  latent_width: %d", getattr(args, "latent_width", 768))
        elif self.slider_config.mode == "reference":
            logger.info("  dataset size: %d", len(ref_dataloader.dataset))
            batch_size = self.slider_config.batch_size
            grad_accum = getattr(args, "gradient_accumulation_steps", 1)
            logger.info("  batch_size: %d (gradient_accumulation_steps: %d, effective_batch_size: %d)",
                       batch_size, grad_accum, batch_size * grad_accum)
            logger.info("  save_every_n_epochs: %s", getattr(args, "save_every_n_epochs", "disabled"))
        elif self.slider_config.mode == "i2v":
            logger.info("  dataset size: %d", len(ref_dataloader.dataset))
            batch_size = self.slider_config.batch_size
            grad_accum = getattr(args, "gradient_accumulation_steps", 1)
            logger.info("  batch_size: %d (gradient_accumulation_steps: %d, effective_batch_size: %d)",
                       batch_size, grad_accum, batch_size * grad_accum)
            logger.info("  save_every_n_epochs: %s", getattr(args, "save_every_n_epochs", "disabled"))

        # Sample at first if requested
        if should_sample_images(args, 0, epoch=current_epoch):
            optimizer_eval_fn()
            self._sample_slider(accelerator, args, transformer, vae, accelerator.unwrap_model(network), sample_parameters, dit_dtype, 0, current_epoch)
            optimizer_train_fn()

        ref_iter = None
        if ref_dataloader is not None:
            ref_iter = iter(ref_dataloader)

        # Track steps in current epoch for reference mode
        steps_in_current_epoch = 0
        dataset_size = len(ref_dataloader.dataset) if ref_dataloader is not None else 0

        # Flag to track if we need to save at epoch boundary
        epoch_to_save = None

        while global_step < args.max_train_steps:
            accelerator.unwrap_model(network).on_step_start()

            # Save at epoch boundary if needed (outside of accumulate block)
            if epoch_to_save is not None:
                optimizer_eval_fn()
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    logger.info("Saving checkpoint at epoch %d", epoch_to_save)
                    ckpt_name = train_utils.get_epoch_ckpt_name(args.output_name, epoch_to_save)
                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch_to_save)

                    if getattr(args, "save_state", False):
                        train_utils.save_and_remove_state_stepwise(args, accelerator, global_step)

                    remove_step_no = train_utils.get_remove_step_no(args, global_step)
                    if remove_step_no is not None:
                        remove_ckpt_name = train_utils.get_step_ckpt_name(args.output_name, remove_step_no)
                        remove_model(remove_ckpt_name)

                optimizer_train_fn()
                epoch_to_save = None

            with accelerator.accumulate(network):
                if self.slider_config.mode == "text":
                    loss = self._text_slider_step(transformer, network, accelerator, args, dit_dtype)
                elif self.slider_config.mode == "reference":
                    # Reference mode: get next batch
                    try:
                        batch = next(ref_iter)
                    except StopIteration:
                        # Completed one epoch
                        epoch_completed = current_epoch
                        current_epoch += 1
                        steps_in_current_epoch = 0
                        logger.info("Completed epoch %d, starting epoch %d", epoch_completed, current_epoch)

                        # Log epoch-level metrics to tensorboard
                        if len(accelerator.trackers) > 0:
                            logs = {"loss/epoch": loss_recorder.moving_average}
                            accelerator.log(logs, step=epoch_completed + 1)

                        # Check if we should save after completing this epoch
                        if (getattr(args, "save_every_n_epochs", None) is not None
                                and epoch_completed > 0
                                and epoch_completed % args.save_every_n_epochs == 0):
                            # Mark for saving after exiting accumulate block
                            epoch_to_save = epoch_completed

                        # Restart iterator for next epoch
                        ref_iter = iter(ref_dataloader)
                        batch = next(ref_iter)

                    loss = self._reference_slider_step(transformer, network, batch, accelerator, args, dit_dtype)
                    steps_in_current_epoch += 1
                else:  # i2v mode
                    # I2V mode: get next batch
                    try:
                        batch = next(ref_iter)
                    except StopIteration:
                        # Completed one epoch
                        epoch_completed = current_epoch
                        current_epoch += 1
                        steps_in_current_epoch = 0
                        logger.info("Completed epoch %d, starting epoch %d", epoch_completed, current_epoch)

                        # Log epoch-level metrics to tensorboard
                        if len(accelerator.trackers) > 0:
                            logs = {"loss/epoch": loss_recorder.moving_average}
                            accelerator.log(logs, step=epoch_completed + 1)

                        # Check if we should save after completing this epoch
                        if (getattr(args, "save_every_n_epochs", None) is not None
                                and epoch_completed > 0
                                and epoch_completed % args.save_every_n_epochs == 0):
                            # Mark for saving after exiting accumulate block
                            epoch_to_save = epoch_completed

                        # Restart iterator for next epoch
                        ref_iter = iter(ref_dataloader)
                        batch = next(ref_iter)

                    loss = self._i2v_slider_step(transformer, network, batch, accelerator, args, dit_dtype)
                    steps_in_current_epoch += 1

                # Gradient clipping
                if accelerator.sync_gradients and getattr(args, "max_grad_norm", 0.0) != 0.0:
                    params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                # -- Preservation / Regularization (blank_preservation, dop) -----------
                # Run preservation backward passes for blank and DOP
                # Note: prior_divergence is skipped for slider training since it requires
                # a single video_pred which isn't available in the multi-pass slider step
                pres_losses = {}
                if self._net_trainer._preservation_active and self._net_trainer._preservation_helper is not None:
                    pres_losses = self._net_trainer.preservation_backward(
                        args, accelerator, transformer, network, dit_dtype,
                    )

                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue

            if global_step == 0:
                progress_bar.reset()
            progress_bar.update(1)
            global_step += 1

            loss_recorder.add(epoch=current_epoch, step=global_step - 1, loss=loss)
            avr_loss = loss_recorder.moving_average
            progress_bar.set_postfix(avr_loss=f"{avr_loss:.4f}", loss=f"{loss:.4f}")

            if len(accelerator.trackers) > 0:
                # Get LR - handle automagic optimizer specially
                if args.optimizer_type.lower() == "automagic":
                    # Automagic has per-parameter LRs, get them directly
                    actual_optimizer = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
                    if hasattr(actual_optimizer, "get_avg_learning_rate"):
                        lr_value = actual_optimizer.get_avg_learning_rate()
                        lr_tensor = actual_optimizer.get_lr_tensor()
                        logs = {
                            "loss/current": loss,
                            "loss/average": avr_loss,
                            "lr/unet": lr_value,
                        }
                        # Log additional automagic metrics if available
                        if lr_tensor is not None and len(lr_tensor) > 1:
                            logs["lr/automagic_min"] = float(lr_tensor.min())
                            logs["lr/automagic_max"] = float(lr_tensor.max())
                            logs["lr/automagic_std"] = float(lr_tensor.std())
                    else:
                        lr_value = 0.0
                        logs = {
                            "loss/current": loss,
                            "loss/average": avr_loss,
                            "lr/unet": lr_value,
                        }
                else:
                    # Standard optimizer/scheduler
                    lrs = lr_scheduler.get_last_lr()
                    logs = {
                        "loss/current": loss,
                        "loss/average": avr_loss,
                        "lr/unet": float(lrs[0]) if lrs else 0.0,
                    }

                # Add preservation losses to logs (blank_pres, dop)
                if pres_losses:
                    logs.update(pres_losses)

                accelerator.log(logs, step=global_step)

            # Sampling
            should_sampling = should_sample_images(args, global_step, epoch=current_epoch)
            should_saving_steps = (
                getattr(args, "save_every_n_steps", None) is not None
                and global_step % args.save_every_n_steps == 0
            )
            # Note: epoch-based saving is handled inside the training loop when dataloader is exhausted

            if should_sampling or should_saving_steps:
                optimizer_eval_fn()

                if should_sampling:
                    self._sample_slider(
                        accelerator, args, transformer, vae,
                        accelerator.unwrap_model(network), sample_parameters, dit_dtype, global_step, current_epoch,
                    )

                if should_saving_steps:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        ckpt_name = train_utils.get_step_ckpt_name(args.output_name, global_step)
                        save_model(ckpt_name, accelerator.unwrap_model(network), global_step, current_epoch)

                        if getattr(args, "save_state", False):
                            train_utils.save_and_remove_state_stepwise(args, accelerator, global_step)

                        remove_step_no = train_utils.get_remove_step_no(args, global_step)
                        if remove_step_no is not None:
                            remove_ckpt_name = train_utils.get_step_ckpt_name(args.output_name, remove_step_no)
                            remove_model(remove_ckpt_name)

                optimizer_train_fn()

        # -- End of training ---------------------------------------------------
        accelerator.end_training()
        optimizer_eval_fn()

        if is_main_process and (getattr(args, "save_state", False) or getattr(args, "save_state_on_train_end", False)):
            train_utils.save_state_on_train_end(args, accelerator)

        if is_main_process:
            ckpt_name = train_utils.get_last_ckpt_name(args.output_name)
            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, current_epoch, force_sync_upload=True)
            logger.info("Slider training complete. Model saved.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def slider_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add slider-specific arguments."""
    parser.add_argument(
        "--slider_config", type=str, required=True,
        help="Path to slider targets TOML file",
    )
    parser.add_argument(
        "--latent_frames", type=int, default=1,
        help="Number of latent frames for text-only mode (1=image, >1=video)",
    )
    parser.add_argument(
        "--latent_height", type=int, default=512,
        help="Pixel height for synthetic latents (text-only mode)",
    )
    parser.add_argument(
        "--latent_width", type=int, default=768,
        help="Pixel width for synthetic latents (text-only mode)",
    )
    parser.add_argument(
        "--guidance_strength", type=float, default=None,
        help="Override guidance strength from slider config",
    )
    parser.add_argument(
        "--sample_slider_range", type=str, default=None,
        help="Comma-separated multiplier values for preview, e.g. '-2,-1,0,1,2'",
    )
    parser.add_argument(
        "--first_frame_conditioning_p", type=float, default=None,
        help="Override first frame conditioning probability from slider config (typically 1.0 for i2v mode)",
    )
    return parser


def main() -> None:
    parser = setup_parser_common()
    parser = ltx2_setup_parser(parser)
    parser = slider_setup_parser(parser)

    # Make dataset_config optional for text-only mode
    for action in parser._actions:
        if hasattr(action, "dest") and action.dest == "dataset_config":
            action.required = False

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    # LTX mode normalization
    if hasattr(args, "ltx_mode"):
        short_map = {"v": "video", "a": "audio", "va": "av"}
        if args.ltx_mode in short_map:
            args.ltx_mode = short_map[args.ltx_mode]

    apply_ltx2_tweaks(args)

    # Default network_module for sliders
    if getattr(args, "network_module", None) is None:
        args.network_module = "networks.lora_ltx2"

    # Point dit/vae to ltx2_checkpoint
    if getattr(args, "dit", None) is None or (
        getattr(args, "ltx2_checkpoint", None) is not None and args.dit != args.ltx2_checkpoint
    ):
        args.dit = args.ltx2_checkpoint
    if getattr(args, "vae", None) is None or (
        getattr(args, "ltx2_checkpoint", None) is not None and args.vae != args.ltx2_checkpoint
    ):
        args.vae = args.ltx2_checkpoint

    args.weighting_scheme = "none"

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    # Inject lora_target_preset
    lora_target_preset = getattr(args, "lora_target_preset", None)
    if lora_target_preset is not None:
        if args.network_args is None:
            args.network_args = []
        if not any(arg.startswith("lora_target_preset=") for arg in args.network_args):
            args.network_args.append(f"lora_target_preset={lora_target_preset}")

    # Block swap env var
    blocks_to_swap = int(getattr(args, "blocks_to_swap", 0) or 0)
    if blocks_to_swap > 0:
        os.environ["LTX2_SWAP_TRAIN_FULL"] = "1"

    trainer = LTX2SliderTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
