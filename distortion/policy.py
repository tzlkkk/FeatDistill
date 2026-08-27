"""Configurable degradation policy for optional robustness experiments."""

from __future__ import annotations

import random
import warnings

import torch

from .pipeline_distortions import distort_images_extra
from .utils_data import distort_images


DEFAULT_AUGMENT_PROBABILITY = 0.8

# Position-wise feature comparisons require token i to retain its spatial area.
_MISALIGNING_BASIC = {"jitter", "colorshift"}
_MISALIGNING_EXTRA = {
    "pixel_shuffle",
    "digital_glitch",
    "ghosting",
    "color_bleed",
    "water_drops",
}


def _check_image(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be a torch.Tensor, got {type(image)!r}")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected image shape [3,H,W], got {tuple(image.shape)}")
    if image.device.type != "cpu":
        raise ValueError("legacy distortions must run on CPU before moving a batch to GPU")
    image = image.float()
    if not torch.isfinite(image).all():
        raise ValueError("image contains NaN or Inf")
    if image.min().item() < 0.0 or image.max().item() > 1.0:
        raise ValueError("distortions must run before normalization on values in [0,1]")
    return image


def apply_training_distortion(
    image: torch.Tensor,
    probability: float = DEFAULT_AUGMENT_PROBABILITY,
    extra_probability: float = 0.5,
    max_distortions: int = 3,
    basic_num_levels: int = 5,
    extra_num_levels: int = 15,
    strict: bool = False,
    preserve_spatial_alignment: bool = False,
):
    """Possibly degrade one unnormalized image.

    Returns ``(image, operations, levels_or_values)`` for compatibility with
    the historical ``distort_images`` and ``distort_images_extra`` APIs.
    ``probability=0.8`` reproduces the original experimental sampling policy.
    """

    image = _check_image(image)
    probability = min(max(float(probability), 0.0), 1.0)
    extra_probability = min(max(float(extra_probability), 0.0), 1.0)
    if max_distortions < 1:
        raise ValueError("max_distortions must be at least 1")
    if random.random() >= probability:
        return image, [], []

    original = image.clone()
    use_extra = random.random() < extra_probability
    try:
        if use_extra:
            if max_distortions > 19 or extra_num_levels < 1:
                raise ValueError("invalid extra distortion configuration")
            result = distort_images_extra(
                image,
                max_distortions=max_distortions,
                num_levels=extra_num_levels,
                p=1.0,
                excluded_ops=_MISALIGNING_EXTRA if preserve_spatial_alignment else None,
            )
        else:
            if max_distortions > 7 or not 1 <= basic_num_levels <= 5:
                raise ValueError("invalid basic distortion configuration")
            result = distort_images(
                image,
                max_distortions=max_distortions,
                num_levels=basic_num_levels,
                excluded_names=_MISALIGNING_BASIC if preserve_spatial_alignment else None,
            )

        output, operations, levels = result
        if output.shape != original.shape or not torch.isfinite(output).all():
            raise RuntimeError("distortion returned an invalid tensor")
        if not use_extra:
            operations = [getattr(op, "__name__", str(op)) for op in operations]
        return output.float().clamp(0.0, 1.0), operations, levels
    except Exception as exc:
        if strict:
            raise
        warnings.warn(f"distortion failed; using the original image: {exc}", RuntimeWarning)
        return original, ["augmentation_failed"], []
