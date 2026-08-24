"""Command-line entry point for the Roboflow COCO cropper."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from crop_dataset import (
    ANNOTATION_FILENAME,
    CropDatasetError,
    DatasetCropReport,
    crop_dataset,
    load_coco,
)
from package_dataset import create_dataset_zip, default_zip_path
from validate_dataset import DatasetValidationReport, validate_dataset


DEFAULT_PADDING = 10


def non_negative_integer(value: str) -> int:
    """Argparse converter for crop padding."""

    try:
        converted = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if converted < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return converted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crop each image in a Roboflow COCO export to the union of its "
            "annotations and rewrite the annotation coordinates."
        )
    )
    parser.add_argument(
        "input_dataset",
        type=Path,
        help="root of the extracted Roboflow COCO dataset",
    )
    parser.add_argument(
        "output_dataset",
        type=Path,
        help="new directory for cropped split folders and COCO JSON files",
    )
    parser.add_argument(
        "--padding",
        type=non_negative_integer,
        default=DEFAULT_PADDING,
        metavar="PIXELS",
        help=f"context to retain around annotations (default: {DEFAULT_PADDING})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output directory",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="write the validated dataset folders without creating a ZIP archive",
    )
    return parser


def validate_dataset_paths(
    input_dataset: str | Path, output_dataset: str | Path
) -> tuple[Path, Path]:
    """Resolve paths and reject configurations that could modify the source."""

    input_path = Path(input_dataset).expanduser().resolve()
    output_path = Path(output_dataset).expanduser().resolve()

    if not input_path.is_dir():
        raise CropDatasetError(f"Input dataset directory not found: {input_path}")
    if input_path == output_path:
        raise CropDatasetError("Input and output dataset directories must be different")
    if output_path.is_relative_to(input_path):
        raise CropDatasetError(
            "Output dataset cannot be inside the input dataset; choose a separate directory"
        )
    if input_path.is_relative_to(output_path):
        raise CropDatasetError(
            "Output dataset cannot contain the input dataset; choose a separate directory"
        )
    if output_path == Path(output_path.anchor):
        raise CropDatasetError("The filesystem root cannot be used as the output dataset")

    return input_path, output_path


def prepare_output_directory(output_path: Path, overwrite: bool) -> None:
    """Ensure output is absent, removing it only when explicitly authorized."""

    if output_path.is_symlink():
        raise CropDatasetError(
            f"Output path cannot be a symbolic link: {output_path}"
        )
    if not output_path.exists():
        return
    if not output_path.is_dir():
        raise CropDatasetError(f"Output path is not a directory: {output_path}")
    if not overwrite:
        raise CropDatasetError(
            f"Output directory already exists: {output_path}. "
            "Use --overwrite to replace it."
        )

    try:
        shutil.rmtree(output_path)
    except OSError as exc:
        raise CropDatasetError(
            f"Could not remove existing output directory {output_path}: {exc}"
        ) from exc


def check_archive_destination(zip_path: Path, overwrite: bool) -> None:
    """Reject archive conflicts before starting the crop operation."""

    if zip_path.is_symlink() or zip_path.is_dir():
        raise CropDatasetError(f"ZIP destination must be a regular file: {zip_path}")
    if zip_path.exists() and not overwrite:
        raise CropDatasetError(
            f"ZIP archive already exists: {zip_path}. Use --overwrite to replace it."
        )


def build_processing_report(
    input_path: Path,
    output_path: Path,
    padding: int,
    crop_report: DatasetCropReport,
    validation_report: DatasetValidationReport,
    archive_path: Path | None,
) -> dict:
    """Build a JSON-serializable audit report for one processing run."""

    split_results: dict[str, dict] = {}
    for split_name, split_report in crop_report.splits.items():
        input_coco = load_coco(input_path / split_name / ANNOTATION_FILENAME)
        output_coco = load_coco(output_path / split_name / ANNOTATION_FILENAME)
        validation = validation_report.splits.get(split_name)
        split_results[split_name] = {
            "input_images": len(input_coco["images"]),
            "output_images": len(output_coco["images"]),
            "input_annotations": len(input_coco["annotations"]),
            "output_annotations": len(output_coco["annotations"]),
            "skipped_empty_images": split_report.skipped_empty_images,
            "crops": split_report.to_dict()["cropped_images"],
            "validation": validation.to_dict() if validation is not None else None,
        }

    warnings = list(validation_report.warnings)
    for split_report in validation_report.splits.values():
        warnings.extend(split_report.warnings)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "valid" if validation_report.valid else "invalid",
        "input_dataset": str(input_path),
        "output_dataset": str(output_path),
        "zip_archive": str(archive_path) if archive_path is not None else None,
        "padding": padding,
        "totals": {
            "cropped_images": crop_report.cropped_count,
            "skipped_empty_images": crop_report.skipped_count,
            "validation_errors": validation_report.error_count,
        },
        "warnings": warnings,
        "validation": validation_report.to_dict(),
        "splits": split_results,
    }


def write_processing_report(report: dict, output_path: Path) -> Path:
    report_path = output_path / "processing_report.json"
    try:
        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, ensure_ascii=False)
            report_file.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise CropDatasetError(
            f"Could not write processing report {report_path}: {exc}"
        ) from exc
    return report_path


def print_report(
    report: DatasetCropReport,
    output_path: Path,
    report_path: Path,
    archive_path: Path | None,
) -> None:
    """Print a compact, human-readable completion summary."""

    print("Cropping complete.")
    for split_name, split_report in report.splits.items():
        print(
            f"  {split_name}: {split_report.cropped_count} cropped, "
            f"{split_report.skipped_count} empty images skipped"
        )
    print(
        f"Total: {report.cropped_count} cropped, "
        f"{report.skipped_count} empty images skipped"
    )
    print("Validation: passed")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    if archive_path is not None:
        print(f"ZIP: {archive_path}")


def run(args: argparse.Namespace) -> DatasetCropReport:
    """Execute a parsed crop command and return its structured report."""

    input_path, output_path = validate_dataset_paths(
        args.input_dataset, args.output_dataset
    )
    archive_path = None if args.no_zip else default_zip_path(output_path).resolve()
    if archive_path is not None:
        check_archive_destination(archive_path, overwrite=args.overwrite)
    prepare_output_directory(output_path, overwrite=args.overwrite)
    crop_report = crop_dataset(
        source_root=input_path,
        output_root=output_path,
        padding=args.padding,
    )
    validation_report = validate_dataset(output_path, source_root=input_path)
    processing_report = build_processing_report(
        input_path=input_path,
        output_path=output_path,
        padding=args.padding,
        crop_report=crop_report,
        validation_report=validation_report,
        archive_path=archive_path,
    )
    report_path = write_processing_report(processing_report, output_path)
    if not validation_report.valid:
        raise CropDatasetError(
            f"Output validation failed with {validation_report.error_count} error(s); "
            f"see {report_path}"
        )

    if archive_path is not None:
        create_dataset_zip(
            dataset_root=output_path,
            zip_path=archive_path,
            overwrite=args.overwrite,
        )
    print_report(crop_report, output_path, report_path, archive_path)
    return crop_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (CropDatasetError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
