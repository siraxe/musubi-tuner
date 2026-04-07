#!/usr/bin/env python3
"""
Strip audio-specific LoRA keys from a checkpoint for video-only fine-tuning.

When fine-tuning an AV-trained checkpoint for video generation, the audio_
prefixed keys cause loading errors. This script removes them.
"""

import argparse
import torch
from pathlib import Path

try:
    from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
    HAS_SAFE_TENSORS = True
except ImportError:
    HAS_SAFE_TENSORS = False


def main():
    parser = argparse.ArgumentParser(
        description="Strip audio LoRA keys from checkpoint for video fine-tuning"
    )
    parser.add_argument("input", type=str, help="Input checkpoint path")
    parser.add_argument("output", type=str, help="Output checkpoint path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading checkpoint: {input_path}")

    # Load safetensors file
    if not HAS_SAFE_TENSORS:
        raise RuntimeError("safetensors library required. Install with: pip install safetensors")
    state_dict = load_safetensors(input_path)

    original_keys = set(state_dict.keys())
    data_dict = state_dict

    print(f"Original checkpoint has {len(original_keys)} keys")

    # Keep only non-audio, non-ff_net keys for video fine-tuning
    def should_keep_key(key: str) -> bool:
        # Remove audio-specific keys
        if "audio_" in key:
            return False
        # Remove ff_net feed-forward keys (not supported in this mode)
        if "ff_net" in key:
            return False
        return True

    filtered_keys = [k for k in original_keys if should_keep_key(k)]
    removed_count = len(original_keys) - len(filtered_keys)

    print(f"Removing {removed_count} audio_ keys")
    print(f"Keeping {len(filtered_keys)} non-audio keys")

    # Create filtered state dict
    filtered_state_dict = {k: data_dict[k] for k in filtered_keys}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors(filtered_state_dict, output_path)

    print(f"Saved cleaned checkpoint: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
