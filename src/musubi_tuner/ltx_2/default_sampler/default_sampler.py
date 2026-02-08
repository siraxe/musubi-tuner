"""
Default Sampler - Main sampler interface for LTX-2.

This module provides the main sampler class for sampling from LTX-2 models,
using the Rectified Flow scheduler from LTX-2.

Usage:
    from musubi_tuner.ltx_2.default_sampler import DefaultSampler

    sampler = DefaultSampler(
        transformer=transformer_model,
        vae=vae_model,
        num_steps=20,
        guidance_scale=4.5,
    )

    result = sampler(
        prompt="A video of a cat",
        height=512,
        width=512,
        num_frames=41,
    )
"""

from typing import Optional, Union, List, Dict, Any
import logging
import torch
from diffusers.image_processor import VaeImageProcessor

logger = logging.getLogger(__name__)


class DefaultSampler:
    """
    Default Sampler for LTX-2 using Rectified Flow scheduler.

    This sampler is completely independent from musubi-tuner's sampling implementation.
    It uses the Rectified Flow scheduler directly from LTX-2.

    Args:
        transformer: The transformer model for denoising
        vae: The VAE model for encoding/decoding
        tokenizer: The text tokenizer
        text_encoder: The text encoder model
        scheduler: The Rectified Flow scheduler (optional, will use default if not provided)
        patchifier: The patchifier for latent processing (optional)
    """

    def __init__(
        self,
        transformer,
        vae,
        tokenizer=None,
        text_encoder=None,
        scheduler=None,
        patchifier=None,
    ):
        self.transformer = transformer
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.patchifier = patchifier

        # Default scheduler if not provided
        if scheduler is None:
            from .schedulers.rf import RectifiedFlowScheduler
            self.scheduler = RectifiedFlowScheduler()
        else:
            self.scheduler = scheduler

        # Get VAE scale factors
        if hasattr(vae, 'spatial_downscale_factor'):
            self.vae_scale_factor = vae.spatial_downscale_factor
            self.video_scale_factor = vae.temporal_downscale_factor
        elif hasattr(vae, 'spatial_downsample_factor'):
            # _LTX2VideoVAE uses "downsample" instead of "downscale"
            self.vae_scale_factor = vae.spatial_downsample_factor
            self.video_scale_factor = vae.temporal_downsample_factor
        else:
            self.vae_scale_factor = 4
            self.video_scale_factor = 1

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _get_vae_spatial_scale_factor(self) -> int:
        """Get VAE spatial downsampling factor, handling different attribute names."""
        # Try different attribute names that various VAE implementations use
        for attr_name in ('spatial_downscale_factor', 'spatial_downsample_factor'):
            if hasattr(self.vae, attr_name):
                return getattr(self.vae, attr_name)
        return 4  # Default for most video VAEs

    def _get_vae_temporal_scale_factor(self) -> int:
        """Get VAE temporal downsampling factor, handling different attribute names."""
        for attr_name in ('temporal_downscale_factor', 'temporal_downsample_factor'):
            if hasattr(self.vae, attr_name):
                return getattr(self.vae, attr_name)
        return 1  # Default for most video VAEs

    def _vae_has_temporal_downsampling(self) -> bool:
        """Check if VAE has temporal downsampling capability."""
        return hasattr(self.vae, 'temporal_downscale_factor') or hasattr(self.vae, 'temporal_downsample_factor')

    @torch.inference_mode()
    def sample(
        self,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        latent_input: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        frame_rate: float = 25.0,
        first_image_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the sampling loop.

        Args:
            prompt_embeds: Text embeddings for the prompt
            prompt_attention_mask: Attention mask for the prompt embeddings
            height: Height of the output video
            width: Width of the output video
            num_frames: Number of frames in the output video
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG guidance scale
            negative_prompt_embeds: Text embeddings for negative prompt
            negative_prompt_attention_mask: Attention mask for negative prompt
            latent_input: Optional input latents for img2img/video2video
            device: Device to run on
            dtype: Data type to use
            frame_rate: Frame rate of the output video
            first_image_latents: Optional first frame latents for I2V mode [B, C, 1, H, W]

        Returns:
            Generated video tensor of shape (B, C, F, H, W)
        """
        if device is None:
            device = next(self.transformer.parameters()).device
        if dtype is None:
            dtype = next(self.transformer.parameters()).dtype

        batch_size = prompt_embeds.shape[0]

        # Compute latent dimensions
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        latent_num_frames = num_frames // self.video_scale_factor

        # For CausalVideoAutoencoder or _LTX2VideoVAE, add 1 frame
        if self._vae_has_temporal_downsampling():
            latent_num_frames += 1

        # Get in_channels from transformer (handle LTX2Wrapper which has .model.config)
        if hasattr(self.transformer, 'config'):
            in_channels = self.transformer.config.in_channels
        elif hasattr(self.transformer, 'model') and hasattr(self.transformer.model, 'config'):
            in_channels = self.transformer.model.config.in_channels
        elif hasattr(self.transformer, 'in_channels'):
            in_channels = self.transformer.in_channels
        else:
            in_channels = 128  # Default for LTX-2

        latent_shape = (
            batch_size,
            in_channels,
            latent_num_frames,
            latent_height,
            latent_width,
        )

        # Set up timesteps
        self.scheduler.set_timesteps(
            num_inference_steps,
            samples_shape=latent_shape,
            device=device,
        )
        timesteps = self.scheduler.timesteps

        # Prepare initial latents
        if latent_input is None:
            latents = torch.randn(
                latent_shape,
                device=device,
                dtype=dtype,
                generator=None,
            )
        else:
            latents = latent_input.to(device=device, dtype=dtype)

        # Import required types for musubi-tuner LTXModel API
        from musubi_tuner.ltx_2.types import VideoLatentShape, SpatioTemporalScaleFactors
        from musubi_tuner.ltx_2.model.transformer.modality import Modality
        from musubi_tuner.ltx_2.components.patchifiers import get_pixel_coords
        from musubi_tuner.ltx_2.guidance.perturbations import BatchedPerturbationConfig

        # Prepare prompt embeddings for CFG
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance and negative_prompt_embeds is not None:
            prompt_embeds_batch = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask_batch = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )
        else:
            prompt_embeds_batch = prompt_embeds
            prompt_attention_mask_batch = prompt_attention_mask

        # Patchify latents and prepare modality objects
        if self.patchifier is not None:
            video_tokens = self.patchifier.patchify(latents=latents)
            video_seq_len = video_tokens.shape[1]

            # Get latent coordinates
            latent_coords = self.patchifier.get_patch_grid_bounds(
                output_shape=VideoLatentShape(
                    batch=batch_size,
                    channels=in_channels,
                    frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                ),
                device=device,
            )

            video_positions = get_pixel_coords(
                latent_coords=latent_coords,
                scale_factors=SpatioTemporalScaleFactors.default(),
                causal_fix=True,
            ).to(dtype=latents.dtype)
            video_positions[:, 0, ...] = video_positions[:, 0, ...] / float(frame_rate)

            # I2V Mode: Calculate first frame token count for per-token timesteps
            first_frame_token_count = None
            denoise_mask = None  # [B, seq_len] - 1.0 for tokens to denoise, 0.0 for conditioned tokens
            clean_first_frame_tokens = None  # Store clean first frame tokens for restoration
            if first_image_latents is not None:
                # Patchify the first image latents to get token count
                first_image_tokens = self.patchifier.patchify(latents=first_image_latents)
                first_frame_token_count = first_image_tokens.shape[1]

                # Store clean first frame tokens for restoration after each denoising step
                clean_first_frame_tokens = first_image_tokens.to(device)

                # Replace first frame tokens with cached latents (initial state)
                video_tokens[:, :first_frame_token_count] = clean_first_frame_tokens

                # Create denoise_mask: 0.0 for first frame (conditioned), 1.0 for other frames (to be denoised)
                denoise_mask = torch.ones(1, video_seq_len, device=device, dtype=latents.dtype)
                denoise_mask[:, :first_frame_token_count] = 0.0

                logger.info("I2V mode: First frame has %d tokens, total tokens: %d",
                           first_frame_token_count, video_seq_len)
        else:
            # No patchifier - can't work with musubi-tuner LTXModel
            raise ValueError("patchifier is required for musubi-tuner LTXModel")

        # Denoising loop
        from tqdm import tqdm
        for i, t in enumerate(tqdm(timesteps, desc="Sampling", leave=False, disable=None)):
            # Broadcast tokens for CFG
            video_tokens_input = (
                torch.cat([video_tokens] * 2) if do_classifier_free_guidance else video_tokens
            )

            # Prepare timestep
            current_timestep = t
            if not torch.is_tensor(current_timestep):
                current_timestep = torch.tensor([current_timestep], dtype=torch.float32, device=device)
            elif len(current_timestep.shape) == 0:
                current_timestep = current_timestep[None].to(device)

            # Extract scalar timestep value
            if current_timestep.dim() > 0:
                sigma = current_timestep[0]
            else:
                sigma = current_timestep

            # Expand timestep to sequence length
            bsz = video_tokens_input.shape[0]
            sigma_batch = sigma.expand(bsz)

            # CRITICAL for I2V: Use per-token timesteps via denoise_mask
            # First frame tokens have timesteps=0 (clean), other tokens have timesteps=sigma (noisy)
            if denoise_mask is not None:
                # Per-token timesteps: sigma * denoise_mask (first frame gets 0, others get sigma)
                denoise_mask_expanded = denoise_mask.expand(bsz, -1)  # [B, seq_len]
                video_timesteps = sigma_batch.view(bsz, 1) * denoise_mask_expanded  # [B, seq_len]
            else:
                # T2V mode: All tokens get the same timestep
                video_timesteps = sigma_batch.view(bsz, 1).expand(bsz, video_tokens_input.shape[1])

            # Create video modality object (musubi-tuner API)
            video_modality = Modality(
                enabled=True,
                latent=video_tokens_input,
                timesteps=video_timesteps,
                positions=video_positions.expand(bsz, -1, -1, -1),
                context=prompt_embeds_batch.to(dtype),
                context_mask=None,  # Pass None to avoid FlashAttention2 mask shape issues with CFG
            )

            # Create perturbations and call transformer (musubi-tuner API)
            perturbations = BatchedPerturbationConfig.empty(bsz)
            video_pred_tokens, _ = self.transformer(video_modality, None, perturbations)

            # Apply CFG if needed
            if do_classifier_free_guidance:
                video_pred_uncond, video_pred_text = video_pred_tokens.chunk(2)
                video_pred = video_pred_uncond + guidance_scale * (video_pred_text - video_pred_uncond)
            else:
                video_pred = video_pred_tokens

            # Compute previous sample using scheduler
            video_tokens = self.scheduler.step(
                video_pred,
                t,
                video_tokens,
                return_dict=False,
            )[0]

            # CRITICAL for I2V: Restore first frame tokens to their initial state after each step
            # This matches ltx-trainer's approach where conditioned tokens are restored from clean_latent
            # We don't blend - we restore exactly to avoid contrast mismatches
            if clean_first_frame_tokens is not None and first_frame_token_count is not None:
                video_tokens[:, :first_frame_token_count] = clean_first_frame_tokens

        # Unpatchify
        latents = self.patchifier.unpatchify(
            latents=video_tokens,
            output_shape=VideoLatentShape(
                batch=batch_size,
                channels=in_channels,
                frames=latent_num_frames,
                height=latent_height,
                width=latent_width,
            ),
        )

        # I2V Mode: First frame tokens are already restored after each denoising step
        # No additional restoration needed here - the first frame is preserved throughout sampling
        if first_frame_token_count is not None and first_image_latents is not None:
            logger.info("I2V mode: First frame preserved throughout sampling (exact restoration)")

        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to video.

        Args:
            latents: Latent tensor of shape (B, C, F, H, W)

        Returns:
            Decoded video tensor of shape (B, C, F, H, W)
        """
        # Check if VAE has a decode method (e.g., _LTX2VideoVAE wrapper from training)
        # If so, use it directly. Otherwise use vae_decode for raw VAE models.
        if hasattr(self.vae, 'decode') and callable(self.vae.decode):
            # Use the VAE's decode method directly (for wrapped VAEs like _LTX2VideoVAE)
            # The decode method expects a list of latents
            result = self.vae.decode([latents.squeeze(0)])
            if isinstance(result, list) and result:
                video = result[0]
            else:
                video = result
        else:
            # Use vae_decode for raw VAE models
            from .models.autoencoders.vae_encode import vae_decode
            is_video = latents.dim() == 5
            video = vae_decode(
                latents,
                self.vae,
                is_video=is_video,
                vae_per_channel_normalize=True,
                timestep=None,
            )
        return video

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latents.

        Args:
            video: Video tensor of shape (B, C, F, H, W)

        Returns:
            Encoded latents of shape (B, C, F, H, W)
        """
        from .models.autoencoders.vae_encode import vae_encode

        latents = vae_encode(
            video,
            self.vae,
            vae_per_channel_normalize=True,
        )
        return latents

    @torch.inference_mode()
    def __call__(
        self,
        prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_frames: int = 41,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        latent_input: Optional[torch.Tensor] = None,
        output_type: str = "latent",
        device: Optional[Union[str, torch.device]] = None,
        return_dict: bool = True,
        frame_rate: float = 25.0,
        first_image_latents: Optional[torch.Tensor] = None,
    ):
        """
        Main sampling method.

        Args:
            prompt: Text prompt for generation
            prompt_embeds: Pre-computed prompt embeddings
            prompt_attention_mask: Attention mask for prompt embeddings
            negative_prompt: Negative prompt for CFG
            height: Height of the output video
            width: Width of the output video
            num_frames: Number of frames in the output video
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG guidance scale
            negative_prompt_embeds: Pre-computed negative prompt embeddings
            negative_prompt_attention_mask: Attention mask for negative prompt embeddings
            latent_input: Optional input latents
            output_type: Output type ("latent" or "pil")
            device: Device to run on
            return_dict: Whether to return a dict or tuple
            frame_rate: Frame rate of the output video
            first_image_latents: Optional first frame latents for I2V mode [B, C, 1, H, W]

        Returns:
            Generated video or latents
        """
        if device is None:
            device = next(self.transformer.parameters()).device

        # Encode prompt if needed
        if prompt_embeds is None and prompt is not None:
            assert self.tokenizer is not None, "Tokenizer required if prompt_embeds not provided"
            assert self.text_encoder is not None, "Text encoder required if prompt_embeds not provided"

            text_inputs = self.tokenizer(
                [prompt] if isinstance(prompt, str) else prompt,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            prompt_attention_mask = text_inputs.attention_mask.to(device)

            prompt_embeds = self.text_encoder(
                text_input_ids, attention_mask=prompt_attention_mask
            )[0]

        # Encode negative prompt if needed
        if negative_prompt_embeds is None and negative_prompt != "":
            assert self.tokenizer is not None, "Tokenizer required for negative prompt"
            assert self.text_encoder is not None, "Text encoder required for negative prompt"

            uncond_input = self.tokenizer(
                [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt,
                padding="max_length",
                max_length=prompt_embeds.shape[1],
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_attention_mask = uncond_input.attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=negative_prompt_attention_mask,
            )[0]

        # Run sampling
        latents = self.sample(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            latent_input=latent_input,
            device=device,
            frame_rate=frame_rate,
            first_image_latents=first_image_latents,
        )

        # Decode if needed
        if output_type == "pil":
            video = self.decode_latents(latents)
            video = self.image_processor.postprocess(video, output_type="pil")
            return video if not return_dict else {"images": video}
        elif output_type == "pt":
            video = self.decode_latents(latents)
            return video if not return_dict else {"images": video}
        else:
            return latents if not return_dict else {"latents": latents}

    @torch.inference_mode()
    def generate_video(
        self,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        cfg_scale: Optional[float] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        frame_rate: float = 25.0,
        first_image_latents: Optional[torch.Tensor] = None,
        first_image_path: Optional[str] = None,
        offload_model_for_decode: bool = False,
    ) -> torch.Tensor:
        """Generate video with full pipeline (sampling + decode).

        This is a convenience method that combines sampling and decoding into one call.
        It supports both text-to-video and image-to-video modes.

        Args:
            prompt_embeds: Text embeddings for the prompt [B, L, D]
            prompt_attention_mask: Attention mask for the prompt embeddings
            height: Height of the output video
            width: Width of the output video
            num_frames: Number of frames in the output video
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG guidance scale
            cfg_scale: Optional CFG scale (overrides guidance_scale if provided)
            negative_prompt_embeds: Text embeddings for negative prompt
            negative_prompt_attention_mask: Attention mask for negative prompt
            seed: Random seed for generation
            device: Device to run on (defaults to transformer device)
            dtype: Data type to use (defaults to transformer dtype)
            frame_rate: Frame rate of the output video
            first_image_latents: Optional pre-encoded first frame latents for I2V [B, C, 1, H, W]
            first_image_path: Optional image path for I2V on-the-fly encoding
            offload_model_for_decode: Whether to offload transformer to CPU before decoding

        Returns:
            video: Tensor of shape [1, C, T, H, W] in [0, 1] range
        """
        if device is None:
            device = next(self.transformer.parameters()).device
        if dtype is None:
            dtype = next(self.transformer.parameters()).dtype

        # Handle I2V on-the-fly encoding
        if first_image_latents is None and first_image_path:
            logger.info("I2V mode: Encoding image on-the-fly from %s", first_image_path)
            vae_encoder = None
            try:
                # Get VAE path from the VAE model (assuming it has a checkpoint_path attribute)
                # This is a limitation - the VAE doesn't expose its path
                # For now, we'll raise an error asking the caller to provide pre-encoded latents
                # or use generate_from_sample_param which has access to the VAE path
                raise ValueError(
                    "on-the-fly I2V encoding requires VAE path. "
                    "Use generate_from_sample_param() or provide first_image_latents directly."
                )
            finally:
                if vae_encoder is not None:
                    del vae_encoder

        # Calculate latent dimensions
        vae_scale_factor_temporal = self._get_vae_temporal_scale_factor()
        vae_scale_factor_spatial = self._get_vae_spatial_scale_factor()
        latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
        latent_height = height // vae_scale_factor_spatial
        latent_width = width // vae_scale_factor_spatial

        # Get in_channels from transformer
        if hasattr(self.transformer, 'config'):
            in_channels = self.transformer.config.in_channels
        elif hasattr(self.transformer, 'model') and hasattr(self.transformer.model, 'config'):
            in_channels = self.transformer.model.config.in_channels
        elif hasattr(self.transformer, 'in_channels'):
            in_channels = self.transformer.in_channels
        else:
            in_channels = 128

        # Create random noise latents
        generator = torch.Generator(device=device) if seed is not None else None
        if generator is not None:
            generator.manual_seed(seed)

        latents = torch.randn(
            (1, int(in_channels), latent_frames, latent_height, latent_width),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        # Run sampling
        actual_guidance_scale = cfg_scale if cfg_scale is not None else guidance_scale
        sampled_latents = self.sample(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=actual_guidance_scale,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            latent_input=latents,
            device=device,
            dtype=dtype,
            frame_rate=frame_rate,
            first_image_latents=first_image_latents,
        )

        # Offload transformer before decode if requested
        if offload_model_for_decode:
            self.transformer.to("cpu")
            logger.info("generate_video: moved transformer to CPU for VAE decode")
            torch.cuda.empty_cache() if device.type == "cuda" else None

        # Decode latents
        video = self.decode_latents(sampled_latents)
        if video.dim() == 4:  # [C, T, H, W]
            video = video.unsqueeze(0)  # [1, C, T, H, W]

        # Normalize to [0, 1]
        video = (video / 2 + 0.5).clamp(0, 1).to(torch.float32).to("cpu")

        return video

    @torch.inference_mode()
    def generate_from_sample_param(
        self,
        sample_parameter: Dict[str, Any],
        vae_path: str,
        width: int,
        height: int,
        frame_count: int,
        sample_steps: int,
        guidance_scale: float,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
        offload_model_for_decode: bool = False,
    ) -> torch.Tensor:
        """Generate video from training's sample_parameter dict format.

        This is a convenience method for compatibility with the training pipeline.
        It extracts embeddings and I2V data from the sample_parameter dict.

        Args:
            sample_parameter: Dict containing:
                - prompt_embeds: Text embeddings
                - prompt_attention_mask: Attention mask
                - negative_prompt_embeds: Optional negative prompt embeddings
                - negative_prompt_attention_mask: Optional negative attention mask
                - start_images_latents: Optional pre-encoded I2V latents
                - start_images: Optional I2V image path (TOML config format)
                - image_path: Optional I2V image path (prompts.txt format)
                - frame_rate: Optional frame rate
            vae_path: Path to VAE checkpoint (for on-the-fly I2V encoding)
            width: Output video width
            height: Output video height
            frame_count: Number of frames
            sample_steps: Number of denoising steps
            guidance_scale: CFG guidance scale
            cfg_scale: Optional CFG scale
            seed: Random seed
            offload_model_for_decode: Whether to offload transformer before decode

        Returns:
            video: Tensor of shape [1, C, T, H, W] in [0, 1] range
        """
        device = next(self.transformer.parameters()).device
        dtype = next(self.transformer.parameters()).dtype

        # Extract embeddings
        prompt_embeds = sample_parameter.get("prompt_embeds")
        if prompt_embeds is None:
            raise ValueError("sample_parameter missing prompt_embeds")
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

        prompt_mask = sample_parameter.get("prompt_attention_mask")
        if prompt_mask is not None:
            if prompt_mask.dim() == 1:
                prompt_mask = prompt_mask.unsqueeze(0)
            prompt_mask = prompt_mask.to(device=device, dtype=torch.int64)

        negative_prompt_embeds = sample_parameter.get("negative_prompt_embeds")
        negative_prompt_mask = sample_parameter.get("negative_prompt_attention_mask")
        if negative_prompt_embeds is not None:
            if negative_prompt_embeds.dim() == 2:
                negative_prompt_embeds = negative_prompt_embeds.unsqueeze(0)
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
        if negative_prompt_mask is not None:
            if negative_prompt_mask.dim() == 1:
                negative_prompt_mask = negative_prompt_mask.unsqueeze(0)
            negative_prompt_mask = negative_prompt_mask.to(device=device, dtype=torch.int64)

        # Handle I2V
        first_image_latents = sample_parameter.get("start_images_latents")

        # On-the-fly I2V encoding fallback
        if first_image_latents is None:
            image_path = sample_parameter.get("start_images") or sample_parameter.get("image_path", "")
            if image_path and str(image_path).strip().lower() not in ("none", ""):
                logger.info("I2V mode: No cached latents, encoding on-the-fly from %s", image_path)
                vae_encoder = None
                try:
                    vae_encoder = load_vae_encoder(vae_path, device)
                    first_image_latents = encode_i2v_image_on_the_fly(
                        image_path=str(image_path),
                        target_width=width,
                        target_height=height,
                        target_num_frames=frame_count,
                        encoder=vae_encoder,
                        device=device,
                    )
                    logger.info("I2V mode: Encoded image to latents with shape %s", list(first_image_latents.shape))
                except Exception as e:
                    logger.warning("I2V mode: Failed to encode image: %s. Falling back to T2V.", e)
                    first_image_latents = None
                finally:
                    if vae_encoder is not None:
                        del vae_encoder

        # Get frame rate
        frame_rate = sample_parameter.get("frame_rate", 25.0)

        # Call generate_video
        return self.generate_video(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_mask,
            height=height,
            width=width,
            num_frames=frame_count,
            num_inference_steps=sample_steps,
            guidance_scale=guidance_scale,
            cfg_scale=cfg_scale,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_mask,
            seed=seed,
            device=device,
            dtype=dtype,
            frame_rate=frame_rate,
            first_image_latents=first_image_latents,
            offload_model_for_decode=offload_model_for_decode,
        )


def create_default_sampler(
    transformer,
    vae,
    tokenizer=None,
    text_encoder=None,
    **kwargs
) -> DefaultSampler:
    """
    Helper function to create a DefaultSampler.

    Args:
        transformer: The transformer model
        vae: The VAE model
        tokenizer: The text tokenizer (optional)
        text_encoder: The text encoder (optional)
        **kwargs: Additional arguments passed to DefaultSampler

    Returns:
        Configured DefaultSampler instance
    """
    return DefaultSampler(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        **kwargs
    )


def load_vae_encoder(vae_path: str, device: torch.device) -> torch.nn.Module:
    """Load VAE encoder for I2V on-the-fly encoding.

    Args:
        vae_path: Path to the VAE checkpoint
        device: Device to load the encoder on

    Returns:
        Loaded VAE encoder model

    This loads the encoder component of the VAE for encoding images to latents
    in Image-to-Video (I2V) mode. Uses bfloat16 precision to match ltx-trainer.
    """
    from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from musubi_tuner.ltx_2.model.video_vae.model_configurator import (
        VideoEncoderConfigurator,
        VAE_ENCODER_COMFY_KEYS_FILTER,
    )

    logger.info(f"Loading VAE encoder from {vae_path} for I2V encoding")

    # CRITICAL: Use bfloat16 for VAE encoder to match ltx-trainer
    actual_dtype = torch.bfloat16
    encoder = SingleGPUModelBuilder(
        model_path=str(vae_path),
        model_class_configurator=VideoEncoderConfigurator,
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=device, dtype=actual_dtype)
    encoder.eval()
    encoder.requires_grad_(False)
    logger.info("Loaded VAE encoder for I2V on-the-fly encoding")
    return encoder


def encode_i2v_image_on_the_fly(
    image_path: str,
    target_width: int,
    target_height: int,
    target_num_frames: int,
    encoder: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Encode an image to latents on-the-fly for I2V sampling.

    This follows the reference implementation from ltx-trainer/validation_sampler.py.

    Args:
        image_path: Path to the input image
        target_width: Target width for the output video (must be divisible by 32)
        target_height: Target height for the output video (must be divisible by 32)
        target_num_frames: Target number of frames (unused, kept for API compatibility)
        encoder: VAE encoder model
        device: Device to run encoding on

    Returns:
        Encoded latents tensor of shape [1, C, 1, H, W]

    Raises:
        FileNotFoundError: If the image file doesn't exist

    The function handles:
    - Loading and converting the image to RGB
    - Resizing while maintaining aspect ratio (cover strategy)
    - Center cropping to exact target dimensions
    - Normalizing to [-1, 1] range
    - Encoding with VAE using bfloat16 autocast
    """
    import os
    from PIL import Image
    from torchvision import transforms

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load image
    img = Image.open(image_path).convert("RGB")

    # Convert to tensor and normalize to [0, 1]
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(img)  # (C, H, W)

    # Resize maintaining aspect ratio (cover target, then center crop)
    # Following the reference implementation from ltx-trainer
    current_height, current_width = image_tensor.shape[1:]

    # Validate and adjust dimensions to be divisible by 32
    target_width = round(target_width / 32) * 32
    target_height = round(target_height / 32) * 32
    target_width = max(32, target_width)
    target_height = max(32, target_height)

    if current_height != target_height or current_width != target_width:
        aspect_ratio = current_width / current_height
        target_aspect_ratio = target_width / target_height

        if aspect_ratio > target_aspect_ratio:
            # Image is wider - resize to match height, crop width
            resize_height = target_height
            resize_width = int(target_height * aspect_ratio)
        else:
            # Image is taller - resize to match width, crop height
            resize_height = int(target_width / aspect_ratio)
            resize_width = target_width

        image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
        image_tensor = torch.nn.functional.interpolate(
            image_tensor, size=(resize_height, resize_width), mode="bilinear", align_corners=False
        )

        # Center crop to target dimensions
        h_start = (resize_height - target_height) // 2
        w_start = (resize_width - target_width) // 2
        image_tensor = image_tensor[:, :, h_start:h_start + target_height, w_start:w_start + target_width]
    else:
        image_tensor = image_tensor.unsqueeze(0)

    # Add frame dimension and convert to [-1, 1]
    image_tensor = image_tensor.unsqueeze(2)  # (1, C, 1, H, W)
    image_tensor = (image_tensor * 2.0 - 1.0).to(device=device, dtype=torch.float32)

    # Encode with VAE
    # CRITICAL: Use bfloat16 autocast for encoding to match ltx-trainer
    with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
        encoded = encoder(image_tensor)

    # LTX-2 uses scaling_factor=1.0 (no scaling needed)
    logger.info(f"Encoded I2V image {image_path} to latents with shape {encoded.shape}")
    return encoded
