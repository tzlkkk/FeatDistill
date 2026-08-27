
import random

import numpy as np
import torch

from .distortions import *

distortion_groups = {
    "blur": ["gausblur", "lensblur"],
    "color_distortion": ["colorshift", "colorsat"],
    "jpeg": ["jpeg"],
    "noise": ["whitenoise", "impulsenoise"],
    "brightness_change": ["brighten", "darken"],
    "spatial_distortion": ["jitter",  "quantization"],
    "sharpness_contrast": ["lincontrchange"],
}

distortion_groups_mapping = {
    "gausblur": "blur",
    "lensblur": "blur",
    "colorshift": "color_distortion",
    "colorsat": "color_distortion",
    "jpeg": "jpeg",
    "whitenoise": "noise",
    "impulsenoise": "noise",
    "brighten": "brightness_change",
    "darken": "brightness_change",
    "jitter": "spatial_distortion",
    "quantization": "spatial_distortion",
    "lincontrchange": "sharpness_contrast",
}

distortion_range = {
    "gausblur": [0.1, 0.5, 1, 2, 5],
    "lensblur": [1, 2, 4, 6, 8],
    "colorshift": [1, 3, 6, 8, 12],
    "colorsat": [0.4, 0.2, 0.1, 0, -0.4],
    "jpeg": [43, 36, 24, 7, 4],
    "whitenoise": [0.001, 0.002, 0.003, 0.005, 0.01],
    "impulsenoise": [0.001, 0.005, 0.01, 0.02, 0.03],
    "brighten": [0.1, 0.2, 0.4, 0.7, 1.1],
    "darken": [0.05, 0.1, 0.2, 0.4, 0.8],
    "jitter": [0.05, 0.1, 0.2, 0.5, 1],
    "quantization": [20, 16, 13, 10, 7],
    "lincontrchange": [0., 0.15, -0.4, 0.3, -0.6],
}

distortion_functions = {
    "gausblur": gaussian_blur,
    "lensblur": lens_blur,
    "colorshift": color_shift,
    "colorsat": color_saturation,
    "jpeg": jpeg,
    "whitenoise": white_noise,
    "impulsenoise": impulse_noise,
    "brighten": brighten,
    "darken": darken,
    "jitter": jitter,
    "quantization": quantization,
    "lincontrchange": linear_contrast_change
}


def distort_images(image: torch.Tensor, distort_functions: list = None, distort_values: list = None,
                   max_distortions: int = 3, num_levels: int = 5,
                   excluded_names: set[str] | None = None) -> torch.Tensor:
    """
    Args:
        image (Tensor of size [3,H,W] and value range [0,1]): image to distort
        distort_functions (list): list of the distortion functions to apply to the image. If None, the functions are randomly chosen.
        distort_values (list): list of the values of the distortion functions to apply to the image. If None, the values are randomly chosen.
        max_distortions (int): maximum number of distortions to apply to the image
        num_levels (int): number of levels of distortion that can be applied to the image

    Returns:
        image (Tensor): distorted image
        distort_functions (list): list of the distortion functions applied to the image
        distort_values (list): list of the values of the distortion functions applied to the image
    """
    if not 1 <= max_distortions <= len(distortion_groups):
        raise ValueError(f"max_distortions must be in [1, {len(distortion_groups)}]")
    if not 1 <= num_levels <= 5:
        raise ValueError("the basic distortion policy supports num_levels in [1, 5]")
    image_loc = image.clone()
    if distort_functions is None or distort_values is None:
        distort_functions, distort_values = get_distortions_composition(
            max_distortions, num_levels, excluded_names=excluded_names
        )

    for distortion, value in zip(distort_functions, distort_values):
        image_loc = distortion(image_loc, value)
        image_loc = image_loc.to(torch.float32)
        image_loc = torch.clip(image_loc, 0, 1)

    return image_loc, distort_functions, distort_values


def get_distortions_composition(
    max_distortions: int = 3,
    num_levels: int = 5,
    excluded_names: set[str] | None = None,
):
    """
    Args:
        max_distortions (int): maximum number of distortions to apply to the image
        num_levels (int): number of levels of distortion that can be applied to the image

    Returns:
        distort_functions (list): list of the distortion functions to apply to the image
        distort_values (list): list of the values of the distortion functions to apply to the image
    """
    MEAN = 0
    STD = 2.5

    excluded_names = excluded_names or set()
    eligible_groups = {
        group: [name for name in names if name not in excluded_names]
        for group, names in distortion_groups.items()
    }
    eligible_groups = {group: names for group, names in eligible_groups.items() if names}
    if not eligible_groups:
        raise ValueError("all basic distortions were excluded")
    num_distortions = random.randint(1, min(max_distortions, len(eligible_groups)))
    groups = random.sample(list(eligible_groups.keys()), num_distortions)
    distortions = [random.choice(eligible_groups[group]) for group in groups]
    distort_functions = [distortion_functions[dist] for dist in distortions]

    probabilities = [1 / (STD * np.sqrt(2 * np.pi)) * np.exp(-((i - MEAN) ** 2) / (2 * STD ** 2))
                     for i in range(num_levels)]  # probabilities according to a gaussian distribution
    normalized_probabilities = [prob / sum(probabilities)
                                for prob in probabilities]  # normalize probabilities
    distort_values = [np.random.choice(distortion_range[dist][:num_levels], p=normalized_probabilities) for dist
                      in distortions]

    return distort_functions, distort_values
