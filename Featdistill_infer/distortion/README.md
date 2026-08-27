# Distortion Utilities

This package preserves the image degradation operators used during the development of the FeatDistill method.

Install the optional dependencies from the repository root:

```bash
python -m pip install -e ".[distortion]"
```

Use the public package interface:

```python
from distortion import apply_training_distortion

degraded, operations, levels = apply_training_distortion(
    image,
    probability=0.8,
)
```

Input contract:

- one CPU `float32` RGB tensor with shape `[3,H,W]`;
- finite, unnormalized values in the `[0,1]` range;
- apply model-specific normalization only after degradation.

Random degradation changes the input and may also change the score. Do not silently enable random degradation during inference, and do not compare degraded and clean predictions as if they used the same protocol.
