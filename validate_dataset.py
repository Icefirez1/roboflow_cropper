"""Validate a cropped Roboflow COCO dataset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from crop_dataset import ANNOTATION_FILENAME, DEFAULT_SPLITS, CropDatasetError, load_coco


BOUNDARY_TOLERANCE = 0.51
AREA_TOLERANCE = 1e-9


@dataclass
class SplitValidationReport:
    split: str
    image_count: int = 0
    annotation_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["valid"] = self.valid
        return result


@dataclass
class DatasetValidationReport:
    splits: dict[str, SplitValidationReport] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors and all(report.valid for report in self.splits.values())

    @property
    def error_count(self) -> int:
        return len(self.errors) + sum(len(report.errors) for report in self.splits.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "error_count": self.error_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "splits": {
                split: report.to_dict() for split, report in self.splits.items()
            },
        }


def _safe_image_path(split_path: Path, file_name: Any) -> Path:
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("missing or invalid file_name")
    relative_path = Path(file_name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"unsafe file_name: {file_name}")
    return split_path / relative_path


def _numeric_coordinates(values: Any, expected_length: int | None = None) -> bool:
    if not isinstance(values, list):
        return False
    if expected_length is not None and len(values) != expected_length:
        return False
    return all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in values
    )


def _polygon_area(polygon: list[int | float]) -> float:
    points = list(zip(polygon[0::2], polygon[1::2]))
    doubled_area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        doubled_area += x1 * y2 - x2 * y1
    return abs(doubled_area) / 2.0


def _normalise_polygons(segmentation: Any) -> list[list[int | float]]:
    if segmentation in (None, []):
        return []
    if isinstance(segmentation, dict):
        raise ValueError("RLE segmentation is unsupported")
    if not isinstance(segmentation, list):
        raise ValueError("segmentation must be a list of polygons")
    if _numeric_coordinates(segmentation):
        polygons = [segmentation]
    else:
        polygons = segmentation
    for polygon in polygons:
        if not _numeric_coordinates(polygon) or len(polygon) < 6 or len(polygon) % 2:
            raise ValueError("polygon must contain at least three finite x/y points")
    return polygons


def _inside(value: float, upper_bound: int, tolerance: float = BOUNDARY_TOLERANCE) -> bool:
    return -tolerance <= value <= upper_bound + tolerance


def _annotations_by_image(coco: Mapping[str, Any]) -> dict[int | str, list[Mapping[str, Any]]]:
    grouped: dict[int | str, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        if isinstance(annotation, dict) and "image_id" in annotation:
            grouped[annotation["image_id"]].append(annotation)
    return dict(grouped)


def validate_split(
    split_path: str | Path,
    source_split_path: str | Path | None = None,
) -> SplitValidationReport:
    """Validate one output split, optionally comparing it with its source."""

    split_path = Path(split_path)
    report = SplitValidationReport(split=split_path.name)
    try:
        coco = load_coco(split_path / ANNOTATION_FILENAME)
    except CropDatasetError as exc:
        report.errors.append(str(exc))
        return report

    images = coco["images"]
    annotations = coco["annotations"]
    report.image_count = len(images)
    report.annotation_count = len(annotations)

    categories = coco.get("categories")
    if not isinstance(categories, list):
        report.errors.append("COCO field 'categories' must be a list")
        category_ids: set[int | str] = set()
    else:
        category_ids = {
            category["id"]
            for category in categories
            if isinstance(category, dict) and "id" in category
        }
        if len(category_ids) != len(categories):
            report.errors.append("Categories must be objects with unique IDs")

    image_dimensions: dict[int | str, tuple[int, int]] = {}
    image_file_names: dict[int | str, str] = {}
    for image in images:
        if not isinstance(image, dict) or "id" not in image:
            report.errors.append("Every image must be an object with an ID")
            continue
        image_id = image["id"]
        if image_id in image_dimensions:
            report.errors.append(f"Duplicate image ID: {image_id}")
            continue

        width = image.get("width")
        height = image.get("height")
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or not isinstance(height, int)
            or isinstance(height, bool)
            or width <= 0
            or height <= 0
        ):
            report.errors.append(f"Image {image_id} has invalid width or height")
            continue
        image_dimensions[image_id] = (width, height)
        image_file_names[image_id] = str(image.get("file_name", ""))

        try:
            image_path = _safe_image_path(split_path, image.get("file_name"))
            with Image.open(image_path) as image_file:
                actual_size = image_file.size
                image_file.verify()
            if actual_size != (width, height):
                report.errors.append(
                    f"Image {image_id} dimensions are {actual_size[0]}x{actual_size[1]}, "
                    f"but COCO declares {width}x{height}"
                )
        except (OSError, ValueError) as exc:
            report.errors.append(f"Image {image_id} is missing or unreadable: {exc}")

    seen_annotation_ids: set[int | str] = set()
    rounded_boundary_annotations: set[int | str] = set()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            report.errors.append("Every annotation must be an object")
            continue
        annotation_id = annotation.get("id", "<unknown>")
        if annotation_id in seen_annotation_ids:
            report.errors.append(f"Duplicate annotation ID: {annotation_id}")
        seen_annotation_ids.add(annotation_id)

        image_id = annotation.get("image_id")
        if image_id not in image_dimensions:
            report.errors.append(
                f"Annotation {annotation_id} references missing or invalid image {image_id}"
            )
            continue
        if annotation.get("category_id") not in category_ids:
            report.errors.append(
                f"Annotation {annotation_id} references invalid category "
                f"{annotation.get('category_id')}"
            )

        width, height = image_dimensions[image_id]
        bbox = annotation.get("bbox")
        if not _numeric_coordinates(bbox, expected_length=4):
            report.errors.append(f"Annotation {annotation_id} has an invalid bbox")
        else:
            x, y, box_width, box_height = bbox
            if box_width <= 0 or box_height <= 0:
                report.errors.append(f"Annotation {annotation_id} has a non-positive bbox")
            elif not (
                _inside(x, width)
                and _inside(y, height)
                and _inside(x + box_width, width)
                and _inside(y + box_height, height)
            ):
                report.errors.append(
                    f"Annotation {annotation_id} bbox lies outside image {image_id}"
                )
            elif not (
                _inside(x, width, tolerance=0)
                and _inside(y, height, tolerance=0)
                and _inside(x + box_width, width, tolerance=0)
                and _inside(y + box_height, height, tolerance=0)
            ):
                rounded_boundary_annotations.add(annotation_id)

        try:
            polygons = _normalise_polygons(annotation.get("segmentation"))
            for polygon in polygons:
                if _polygon_area(polygon) <= AREA_TOLERANCE:
                    report.errors.append(
                        f"Annotation {annotation_id} has a zero-area polygon"
                    )
                if any(
                    not _inside(x, width) or not _inside(y, height)
                    for x, y in zip(polygon[0::2], polygon[1::2])
                ):
                    report.errors.append(
                        f"Annotation {annotation_id} polygon lies outside image {image_id}"
                    )
                elif any(
                    not _inside(x, width, tolerance=0)
                    or not _inside(y, height, tolerance=0)
                    for x, y in zip(polygon[0::2], polygon[1::2])
                ):
                    rounded_boundary_annotations.add(annotation_id)
        except ValueError as exc:
            report.errors.append(f"Annotation {annotation_id} has invalid segmentation: {exc}")

    if rounded_boundary_annotations:
        report.warnings.append(
            f"{len(rounded_boundary_annotations)} annotation(s) extend no more than "
            f"{BOUNDARY_TOLERANCE} pixels beyond an image edge due to coordinate rounding"
        )

    if source_split_path is None:
        report.warnings.append(
            "No source split supplied; annotation-retention comparison was not performed"
        )
    else:
        try:
            source_coco = load_coco(Path(source_split_path) / ANNOTATION_FILENAME)
            source_images = {
                image["id"]: image
                for image in source_coco["images"]
                if isinstance(image, dict) and "id" in image
            }
            source_annotations = _annotations_by_image(source_coco)
            output_annotations = _annotations_by_image(coco)
            for image_id in image_dimensions:
                if image_id not in source_images:
                    report.errors.append(
                        f"Output image {image_id} does not exist in the source split"
                    )
                    continue
                source_ids = {ann.get("id") for ann in source_annotations.get(image_id, [])}
                output_ids = {ann.get("id") for ann in output_annotations.get(image_id, [])}
                if source_ids != output_ids:
                    missing = len(source_ids - output_ids)
                    added = len(output_ids - source_ids)
                    report.errors.append(
                        f"Image {image_id} annotation IDs changed: {missing} missing, "
                        f"{added} added"
                    )
        except CropDatasetError as exc:
            report.errors.append(f"Could not compare source annotations: {exc}")

    return report


def validate_dataset(
    dataset_root: str | Path,
    source_root: str | Path | None = None,
) -> DatasetValidationReport:
    """Validate all supported COCO split folders in a dataset."""

    dataset_root = Path(dataset_root)
    source_root_path = Path(source_root) if source_root is not None else None
    report = DatasetValidationReport()
    discovered = [
        split
        for split in DEFAULT_SPLITS
        if (dataset_root / split / ANNOTATION_FILENAME).is_file()
    ]
    if not discovered:
        report.errors.append(
            f"No COCO splits found under {dataset_root}; expected {ANNOTATION_FILENAME}"
        )
        return report

    for split in discovered:
        source_split = None
        if source_root_path is not None:
            candidate = source_root_path / split
            if candidate.is_dir():
                source_split = candidate
            elif split == "val" and (source_root_path / "valid").is_dir():
                source_split = source_root_path / "valid"
            elif split == "valid" and (source_root_path / "val").is_dir():
                source_split = source_root_path / "val"
            else:
                report.errors.append(f"Source split not found for output split: {split}")
        report.splits[split] = validate_split(dataset_root / split, source_split)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a Roboflow COCO dataset")
    parser.add_argument("dataset", type=Path, help="dataset root to validate")
    parser.add_argument(
        "--source",
        type=Path,
        help="optional source dataset for annotation-retention checks",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="print the validation report as JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_dataset(args.dataset, args.source)
    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        for split, split_report in report.splits.items():
            status = "PASS" if split_report.valid else "FAIL"
            print(
                f"{split}: {status} ({split_report.image_count} images, "
                f"{split_report.annotation_count} annotations)"
            )
            for error in split_report.errors:
                print(f"  error: {error}")
            for warning in split_report.warnings:
                print(f"  warning: {warning}")
        for error in report.errors:
            print(f"error: {error}")
        print(f"Validation {'passed' if report.valid else 'failed'}.")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
