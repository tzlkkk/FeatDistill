"""Command-line inference with resumable CSV output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from contextlib import contextmanager
from importlib.metadata import version as package_version
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm

from . import __version__
from .model import (
    Model,
    checkpoint_identity_digest,
    resolve_device,
    verify_checkpoint_files,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CSV_FIELDS = ["image_name", "score"]


def _list_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"image directory not found: {root}")
    images = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"no supported images found under: {root}")
    return images


def _load_processed(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(f"unexpected CSV header in {csv_path}: {reader.fieldnames}")
        names: set[str] = set()
        for row in reader:
            if None in row:
                raise ValueError(f"row has unexpected extra columns in {csv_path}: {row!r}")
            name = row.get("image_name", "")
            if not name or name in names:
                raise ValueError(f"invalid or duplicate image_name in {csv_path}: {name!r}")
            try:
                score = float(row.get("score", ""))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid score for {name!r} in {csv_path}") from exc
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"invalid score for {name!r} in {csv_path}: {score!r}")
            names.add(name)
        return names


def _append_rows(csv_path: Path, rows: list[tuple[str, float]]) -> None:
    if not rows:
        return
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(CSV_FIELDS)
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"failed to decode image: {path}") from exc


def _save_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


@contextmanager
def _exclusive_output_lock(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".inference.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"another process is using this output directory: {lock_path}; "
            "if a previous process crashed, verify that it stopped before removing the lock"
        ) from exc
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def _prepare_output(
    *,
    out_dir: Path,
    overwrite: bool,
    metadata: dict,
) -> tuple[Path, Path, set[str]]:
    csv_path = out_dir / "predictions.csv"
    metadata_path = out_dir / "predictions.meta.json"
    if overwrite:
        csv_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid output metadata: {metadata_path}") from exc
        if existing != metadata:
            raise ValueError(
                f"output metadata does not match this run: {metadata_path}; "
                "choose another --out-dir or pass --overwrite"
            )
    elif csv_path.exists():
        raise ValueError(
            f"found {csv_path} without matching metadata; pass --overwrite to replace it"
        )
    return csv_path, metadata_path, _load_processed(csv_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the four-expert FeatDistill inference ensemble."
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--no-amp", action="store_true", help="disable CUDA autocast")
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="skip SHA-256 verification (only for trusted local checkpoints)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace predictions.csv instead of resuming it",
    )
    return parser


def _inference_metadata(
    *,
    images_dir: Path,
    weights_dir: Path,
    device: torch.device,
    use_amp: bool,
    hashes_verified: bool,
    batch_size: int,
) -> dict:
    return {
        "format_version": 2,
        "inference_protocol": "two-clip-two-siglip-probability-mean-v1",
        "package_version": __version__,
        "images_dir": str(images_dir),
        "checkpoint_identity_sha256": checkpoint_identity_digest(
            weights_dir,
            use_manifest=hashes_verified,
        ),
        "checkpoint_hashes_verified": hashes_verified,
        "device": str(device),
        "cuda_autocast": use_amp,
        "batch_size": batch_size,
        "runtime": {
            "torch": torch.__version__,
            "torchvision": package_version("torchvision"),
            "transformers": package_version("transformers"),
            "pillow": package_version("pillow"),
        },
        "preprocessing": "bicubic-short-edge-resize-center-crop-v1",
        "score": "arithmetic mean of four P(fake), class index 1",
    }


def _run(args: argparse.Namespace) -> None:
    images_dir = args.images_dir.expanduser().resolve()
    weights_dir = args.weights_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    device = resolve_device(args.device)
    use_amp = bool(not args.no_amp and device.type == "cuda")
    image_paths = _list_images(images_dir)
    verify_checkpoint_files(weights_dir, verify_hashes=not args.skip_hash_check)
    metadata = _inference_metadata(
        images_dir=images_dir,
        weights_dir=weights_dir,
        device=device,
        use_amp=use_amp,
        hashes_verified=not args.skip_hash_check,
        batch_size=args.batch_size,
    )

    with _exclusive_output_lock(out_dir):
        csv_path, metadata_path, processed = _prepare_output(
            out_dir=out_dir,
            overwrite=args.overwrite,
            metadata=metadata,
        )
        remaining = [
            path
            for path in image_paths
            if path.relative_to(images_dir).as_posix() not in processed
        ]
        if not remaining:
            print(f"All {len(image_paths)} images are already present in {csv_path}")
            return

        print(f"Images: {len(image_paths)} total, {len(processed)} already processed")
        model = Model(
            device=device,
            model_data_dir=weights_dir,
            # The CLI completed the requested hash preflight immediately above;
            # avoid reading all 5.86 GB a second time during construction.
            verify_hashes=False,
            use_amp=use_amp,
        )
        if not metadata_path.exists():
            _save_json_atomic(metadata_path, metadata)
        print(f"Device: {model.device}; CUDA autocast: {model.use_amp}")

        buffered_rows: list[tuple[str, float]] = []
        with tqdm(
            total=len(remaining),
            desc="Inference",
            unit="img",
            dynamic_ncols=True,
        ) as progress:
            for start in range(0, len(remaining), args.batch_size):
                batch_paths = remaining[start : start + args.batch_size]
                batch_images = [_load_image(path) for path in batch_paths]
                scores = model.predict_pil(batch_images).detach().cpu()
                if len(scores) != len(batch_paths) or not torch.isfinite(scores).all():
                    raise RuntimeError("invalid inference result")
                buffered_rows.extend(
                    (
                        path.relative_to(images_dir).as_posix(),
                        float(score),
                    )
                    for path, score in zip(batch_paths, scores, strict=True)
                )
                progress.update(len(batch_paths))
                if len(buffered_rows) >= args.save_every:
                    _append_rows(csv_path, buffered_rows)
                    buffered_rows.clear()

        _append_rows(csv_path, buffered_rows)
        print(f"Finished: {csv_path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.save_every <= 0:
        parser.error("--save-every must be positive")

    _run(args)


if __name__ == "__main__":
    main()
