"""Optional image degradations retained for robustness analysis.

All public functions expect a floating point ``[3, H, W]`` tensor in
``[0, 1]``.  Apply model-specific normalization only after degradation.
The default FeatDistill inference path never imports or calls this package.
"""

from .pipeline_distortions import distort_images_extra
from .policy import DEFAULT_AUGMENT_PROBABILITY, apply_training_distortion
from .utils_data import distort_images

__all__ = [
    "DEFAULT_AUGMENT_PROBABILITY",
    "apply_training_distortion",
    "distort_images",
    "distort_images_extra",
]
