"""Authoritative inference architecture for the four released experts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import (
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    SiglipVisionConfig,
    SiglipVisionModel,
)


CHECKPOINT_NAMES = (
    "Expert_1_clip.pth",
    "Expert_2_clip.pth",
    "Expert_3_siglip.pth",
    "Expert_4_siglip.pth",
)
MANIFEST_NAME = "manifest.json"


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(weights_dir: Path) -> dict:
    path = weights_dir / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"weight manifest not found: {path}; download the release manifest "
            "together with the four checkpoints"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid weight manifest: {path}") from exc
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), dict):
        raise ValueError(f"unsupported weight manifest schema: {path}")
    return payload


def checkpoint_identity_digest(
    weights_dir: str | Path,
    *,
    use_manifest: bool = True,
) -> str:
    """Hash only immutable checkpoint identities, ignoring mutable download URLs."""

    root = Path(weights_dir).expanduser().resolve()
    manifest_path = root / MANIFEST_NAME
    if use_manifest and manifest_path.is_file():
        manifest = _load_manifest(root)
        identity = {
            name: {
                "size_bytes": manifest["files"].get(name, {}).get("size_bytes"),
                "sha256": manifest["files"].get(name, {}).get("sha256"),
            }
            for name in CHECKPOINT_NAMES
        }
    else:
        identity = {
            name: {
                "size_bytes": (root / name).stat().st_size,
                "modified_time_ns": (root / name).stat().st_mtime_ns,
            }
            for name in CHECKPOINT_NAMES
        }
    serialized = json.dumps(
        identity,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def verify_checkpoint_files(
    weights_dir: str | Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Path]:
    """Validate all required files before any checkpoint is deserialized."""

    root = Path(weights_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"weights directory not found: {root}")

    paths = {name: root / name for name in CHECKPOINT_NAMES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing expert checkpoint(s):\n  " + "\n  ".join(missing))
    if not verify_hashes:
        return paths

    manifest = _load_manifest(root)
    entries = manifest["files"]
    for name, path in paths.items():
        entry = entries.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"manifest has no valid entry for {name}")
        expected_size = entry.get("size_bytes")
        expected_sha = str(entry.get("sha256", "")).lower()
        if not isinstance(expected_size, int) or expected_size <= 0:
            raise ValueError(f"manifest has an invalid size for {name}")
        if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
            raise ValueError(f"manifest has an invalid SHA-256 for {name}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"checkpoint size mismatch for {name}: {actual_size} != {expected_size}"
            )
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"checkpoint SHA-256 mismatch for {name}: {actual_sha} != {expected_sha}"
            )
    return paths


def _clip_config() -> CLIPVisionConfig:
    # Exact vision architecture of openai/clip-vit-large-patch14. The released
    # expert checkpoints contain every encoder and classifier parameter, so no
    # separate base-model weights are needed.
    return CLIPVisionConfig(
        hidden_size=1024,
        intermediate_size=4096,
        projection_dim=768,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
    )


def _siglip_config() -> SiglipVisionConfig:
    # Exact vision architecture of google/siglip-so400m-patch14-384.
    return SiglipVisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=384,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
    )


def _load_raw_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"expected a non-empty state_dict in {path}")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError(f"checkpoint is not a tensor-only state_dict: {path}")
    return state


class CLIPExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = CLIPVisionModelWithProjection(_clip_config())
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        features = self.vision_encoder(pixel_values=pixels).image_embeds
        return self.classifier(features)


class SigLIPExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = SiglipVisionModel(_siglip_config())
        self.classifier = nn.Sequential(
            nn.Linear(1152, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        features = self.vision_encoder(pixel_values=pixels).pooler_output
        return self.classifier(features)


def _restore_expert(
    model: nn.Module,
    checkpoint: Path,
    device: torch.device,
) -> nn.Module:
    state = _load_raw_state_dict(checkpoint)
    try:
        model.load_state_dict(state, strict=True, assign=True)
    except RuntimeError as exc:
        raise RuntimeError(f"checkpoint architecture mismatch: {checkpoint}") from exc
    del state
    return model.to(device).eval()


class FeatDistillEnsemble(nn.Module):
    """Two CLIP experts plus two SigLIP experts with probability averaging."""

    def __init__(
        self,
        weights_dir: str | Path,
        device: str | torch.device,
        *,
        verify_hashes: bool = True,
    ) -> None:
        super().__init__()
        target = torch.device(device)
        paths = verify_checkpoint_files(weights_dir, verify_hashes=verify_hashes)
        self.expert1_clip = _restore_expert(CLIPExpert(), paths["Expert_1_clip.pth"], target)
        self.expert2_clip = _restore_expert(CLIPExpert(), paths["Expert_2_clip.pth"], target)
        self.expert3_siglip = _restore_expert(
            SigLIPExpert(), paths["Expert_3_siglip.pth"], target
        )
        self.expert4_siglip = _restore_expert(
            SigLIPExpert(), paths["Expert_4_siglip.pth"], target
        )

    def forward(self, pixels_384: torch.Tensor, pixels_224: torch.Tensor) -> torch.Tensor:
        prob_clip1 = F.softmax(self.expert1_clip(pixels_224).float(), dim=1)[:, 1]
        prob_clip2 = F.softmax(self.expert2_clip(pixels_224).float(), dim=1)[:, 1]
        prob_siglip3 = F.softmax(self.expert3_siglip(pixels_384).float(), dim=1)[:, 1]
        prob_siglip4 = F.softmax(self.expert4_siglip(pixels_384).float(), dim=1)[:, 1]
        # Preserve the arithmetic order of the original released inference code.
        return (prob_siglip3 + prob_siglip4 + prob_clip1 + prob_clip2) / 4.0


def resolve_device(device: str | torch.device) -> torch.device:
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


class Model:
    """Public inference wrapper retained for challenge-style integrations."""

    def __init__(
        self,
        device: str | torch.device = "auto",
        model_data_dir: str | Path = "weights",
        *,
        verify_hashes: bool = True,
        use_amp: bool = True,
    ) -> None:
        self.device = resolve_device(device)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.model = FeatDistillEnsemble(
            model_data_dir,
            self.device,
            verify_hashes=verify_hashes,
        ).eval()
        self.transform_384 = T.Compose(
            [
                T.Resize(384, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(384),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        self.transform_224 = T.Compose(
            [
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Predict a `[B,3,H,W]` uint8 or float tensor (float range `[0,1]`)."""

        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be a torch tensor with shape [B,3,H,W]")
        if images.shape[0] == 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        if images.is_floating_point():
            if not torch.isfinite(images).all():
                raise ValueError("images contain NaN or infinity")
            minimum = float(images.min())
            maximum = float(images.max())
            if minimum < 0.0 or maximum > 1.0:
                raise ValueError("floating-point images must be in [0,1]")
        elif images.dtype != torch.uint8:
            raise ValueError("integer images must use torch.uint8")
        pil_images = [T.ToPILImage()(image.cpu()) for image in images]
        return self.predict_pil(pil_images)

    @torch.inference_mode()
    def predict_pil(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """Predict RGB-converted PIL images; mixed source sizes are supported."""

        if not images:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        batch_384 = torch.stack(
            [self.transform_384(image.convert("RGB")) for image in images]
        ).to(self.device)
        batch_224 = torch.stack(
            [self.transform_224(image.convert("RGB")) for image in images]
        ).to(self.device)
        amp_context = (
            torch.amp.autocast("cuda", enabled=True)
            if self.use_amp
            else nullcontext()
        )
        with amp_context:
            scores = self.model(batch_384, batch_224)
        if scores.ndim != 1 or len(scores) != len(images) or not torch.isfinite(scores).all():
            raise RuntimeError("model produced invalid probabilities")
        return scores
