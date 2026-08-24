"""Package a validated Roboflow dataset as an upload-ready ZIP archive."""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

from crop_dataset import ANNOTATION_FILENAME, DEFAULT_SPLITS, CropDatasetError


def default_zip_path(dataset_root: str | Path) -> Path:
    dataset_root = Path(dataset_root)
    return dataset_root.parent / f"{dataset_root.name}.zip"


def create_dataset_zip(
    dataset_root: str | Path,
    zip_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Create an atomic ZIP with split folders directly at its root."""

    dataset_root = Path(dataset_root).resolve()
    zip_path = (
        Path(zip_path).resolve()
        if zip_path is not None
        else default_zip_path(dataset_root).resolve()
    )
    if not dataset_root.is_dir():
        raise CropDatasetError(f"Dataset directory not found: {dataset_root}")
    if zip_path.is_dir():
        raise CropDatasetError(f"ZIP destination is a directory: {zip_path}")
    if zip_path.exists() and not overwrite:
        raise CropDatasetError(
            f"ZIP archive already exists: {zip_path}. Use --overwrite to replace it."
        )

    splits = [
        split
        for split in DEFAULT_SPLITS
        if (dataset_root / split / ANNOTATION_FILENAME).is_file()
    ]
    if not splits:
        raise CropDatasetError(f"No COCO split folders found in {dataset_root}")

    files = sorted(path for path in dataset_root.rglob("*") if path.is_file())
    if not files:
        raise CropDatasetError(f"Dataset contains no files: {dataset_root}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{zip_path.name}.", suffix=".tmp", dir=zip_path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for file_path in files:
                if file_path.resolve() == zip_path:
                    continue
                archive.write(file_path, file_path.relative_to(dataset_root).as_posix())
        temporary_path.replace(zip_path)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise CropDatasetError(f"Could not create ZIP archive {zip_path}: {exc}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return zip_path
