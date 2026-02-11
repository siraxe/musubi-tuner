"""
Aspect ratio bucket utilities for dataset preprocessing.

Based on diffusion-pipe implementation, this module provides functionality to
generate and manage aspect ratio buckets with configurable min/max AR and bucket count.
"""

import logging
import math
import numpy as np
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


def generate_ar_buckets(
    resolution: Union[int, Tuple[int, int]],
    min_ar: float = 0.5,
    max_ar: float = 2.0,
    num_ar_buckets: int = 2,
    reso_steps: int = 32,
) -> list[Tuple[int, int]]:
    """
    Generate aspect ratio buckets using geometric spacing between min and max AR.

    Args:
        resolution: Target resolution (scalar or width,height tuple). If scalar, uses square resolution.
        min_ar: Minimum aspect ratio (width/height)
        max_ar: Maximum aspect ratio (width/height)
        num_ar_buckets: Number of aspect ratio buckets to create
        reso_steps: Resolution steps (must divide evenly into bucket dimensions)

    Returns:
        List of (width, height) tuples for each bucket
    """
    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    bucket_area = resolution[0] * resolution[1]

    # Generate AR buckets using geometric spacing (like diffusion-pipe)
    ars = np.geomspace(min_ar, max_ar, num=num_ar_buckets)

    logger.info(f"Generating AR buckets: resolution={resolution}, min_ar={min_ar}, max_ar={max_ar}, num_ar_buckets={num_ar_buckets}")
    logger.info(f"AR values: {ars}")

    bucket_resolutions = []
    for ar in ars:
        # Calculate width and height from aspect ratio and area
        w = math.sqrt(bucket_area * ar)
        h = bucket_area / w

        # Round to nearest multiple of reso_steps
        w = round_to_nearest_multiple(w, reso_steps)
        h = round_to_nearest_multiple(h, reso_steps)

        # Ensure minimum size
        w = max(w, reso_steps)
        h = max(h, reso_steps)

        logger.info(f"AR={ar:.4f} -> ({w}, {h})")
        bucket_resolutions.append((w, h))

    # Remove duplicates and sort
    bucket_resolutions = list(set(bucket_resolutions))
    bucket_resolutions.sort()

    logger.info(f"Final buckets ({len(bucket_resolutions)}): {bucket_resolutions}")
    return bucket_resolutions


def round_to_nearest_multiple(value: float, multiple: int) -> int:
    """Round a value to the nearest multiple."""
    return int(round(value / multiple) * multiple)


def divisible_by(value: int, divisor: int) -> int:
    """Round down to nearest multiple of divisor."""
    return value // divisor * divisor


class ARBucketSelector:
    """
    Aspect ratio bucket selector that finds the closest bucket for a given image size.

    Similar to BucketSelector but uses AR-based bucketing instead of iterating
    over all possible widths.
    """

    def __init__(
        self,
        resolution: Tuple[int, int],
        min_ar: float = 0.5,
        max_ar: float = 2.0,
        num_ar_buckets: int = 2,
        reso_steps: int = 32,
        no_upscale: bool = False,
    ):
        """
        Initialize AR bucket selector.

        Args:
            resolution: Base (width, height) resolution
            min_ar: Minimum aspect ratio (width/height)
            max_ar: Maximum aspect ratio (width/height)
            num_ar_buckets: Number of aspect ratio buckets
            reso_steps: Resolution must be divisible by this
            no_upscale: If True, don't upscale smaller images
        """
        self.resolution = resolution
        self.min_ar = min_ar
        self.max_ar = max_ar
        self.num_ar_buckets = num_ar_buckets
        self.reso_steps = reso_steps
        self.no_upscale = no_upscale
        self.bucket_area = resolution[0] * resolution[1]

        # Generate bucket resolutions
        self.bucket_resolutions = generate_ar_buckets(
            resolution=resolution,
            min_ar=min_ar,
            max_ar=max_ar,
            num_ar_buckets=num_ar_buckets,
            reso_steps=reso_steps,
        )

        # Calculate aspect ratios for fast lookup
        self.aspect_ratios = np.array([w / h for w, h in self.bucket_resolutions])

    def get_bucket_resolution(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Get the bucket resolution for a given image size.

        Args:
            image_size: (width, height) of the image

        Returns:
            (bucket_width, bucket_height)
        """
        area = image_size[0] * image_size[1]

        # If no_upscale and image is smaller, just round to steps
        if self.no_upscale and area <= self.bucket_area:
            w, h = image_size
            w = divisible_by(w, self.reso_steps)
            h = divisible_by(h, self.reso_steps)
            return (max(w, self.reso_steps), max(h, self.reso_steps))

        # Find closest aspect ratio bucket
        aspect_ratio = image_size[0] / image_size[1]
        ar_errors = self.aspect_ratios - aspect_ratio
        bucket_id = np.abs(ar_errors).argmin()
        return self.bucket_resolutions[bucket_id]

    @classmethod
    def calculate_bucket_resolution(
        cls,
        image_size: Tuple[int, int],
        resolution: Tuple[int, int],
        min_ar: float = 0.5,
        max_ar: float = 2.0,
        num_ar_buckets: int = 2,
        reso_steps: int = 32,
    ) -> Tuple[int, int]:
        """
        Calculate the best bucket resolution for a given image size.

        This is a convenience method that creates a temporary selector
        and returns the bucket resolution.

        Args:
            image_size: (width, height) of the image
            resolution: Base (width, height) resolution
            min_ar: Minimum aspect ratio
            max_ar: Maximum aspect ratio
            num_ar_buckets: Number of AR buckets
            reso_steps: Resolution steps

        Returns:
            (width, height) bucket resolution
        """
        selector = cls(resolution, min_ar, max_ar, num_ar_buckets, reso_steps)
        return selector.get_bucket_resolution(image_size)


def find_closest_ar_bucket(log_ar: float, ars: np.ndarray, log_ars: np.ndarray) -> int:
    """
    Find the closest aspect ratio bucket index.

    Args:
        log_ar: Log of the image aspect ratio
        ars: Array of aspect ratio bucket values
        log_ars: Pre-computed log of ars

    Returns:
        Index of the closest bucket
    """
    return np.argmin(np.abs(log_ar - log_ars))
