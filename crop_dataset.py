"""Crop Roboflow COCO images and translate their annotations."""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


ANNOTATION_FILENAME = "_annotations.coco.json"
DEFAULT_SPLITS = ("train", "valid", "val", "test")


class CropDatasetError(ValueError):
    """Raised when a dataset cannot be cropped safely."""


@dataclass(frozen=True)
class CropBounds:
    """Integer Pillow crop bounds, where right and bottom are exclusive."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_pillow_box(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


@dataclass(frozen=True)
class CroppedImage:
    image_id: int | str
    file_name: str
    source_width: int
    source_height: int
    crop: CropBounds
    annotation_count: int


@dataclass
class SplitCropReport:
    split: str
    cropped_images: list[CroppedImage] = field(default_factory=list)
    skipped_empty_images: list[str] = field(default_factory=list)

    @property
    def cropped_count(self) -> int:
        return len(self.cropped_images)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_empty_images)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetCropReport:
    splits: dict[str, SplitCropReport]

    @property
    def cropped_count(self) -> int:
        return sum(report.cropped_count for report in self.splits.values())

    @property
    def skipped_count(self) -> int:
        return sum(report.skipped_count for report in self.splits.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "cropped_count": self.cropped_count,
            "skipped_count": self.skipped_count,
            "splits": {
                name: report.to_dict() for name, report in self.splits.items()
            },
        }


def load_coco(annotation_path: str | Path) -> dict[str, Any]:
    """Load a COCO JSON file and check the fields needed for cropping."""

    annotation_path = Path(annotation_path)
    try:
        with annotation_path.open("r", encoding="utf-8") as annotation_file:
            coco = json.load(annotation_file)
    except FileNotFoundError as exc:
        raise CropDatasetError(f"Annotation file not found: {annotation_path}") from exc
    except json.JSONDecodeError as exc:
        raise CropDatasetError(
            f"Invalid JSON in {annotation_path}: line {exc.lineno}, column {exc.colno}"
        ) from exc

    if not isinstance(coco, dict):
        raise CropDatasetError(f"COCO root must be an object: {annotation_path}")
    for key in ("images", "annotations"):
        if not isinstance(coco.get(key), list):
            raise CropDatasetError(
                f"COCO field '{key}' must be a list: {annotation_path}"
            )
    return coco


def group_annotations_by_image(
    annotations: Iterable[Mapping[str, Any]],
) -> dict[int | str, list[Mapping[str, Any]]]:
    """Index annotations by their COCO image ID."""

    grouped: dict[int | str, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        if "image_id" not in annotation:
            raise CropDatasetError("Every annotation must contain an 'image_id'")
        grouped[annotation["image_id"]].append(annotation)
    return dict(grouped)


def _bbox_extrema(annotation: Mapping[str, Any]) -> tuple[float, float, float, float]:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        raise CropDatasetError(
            f"Annotation {annotation.get('id', '<unknown>')} has an invalid bbox"
        )

    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in bbox
    ):
        raise CropDatasetError(
            f"Annotation {annotation.get('id', '<unknown>')} has a non-numeric bbox"
        )
    x, y, width, height = bbox

    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise CropDatasetError(
            f"Annotation {annotation.get('id', '<unknown>')} has a non-finite bbox"
        )
    if width <= 0 or height <= 0:
        raise CropDatasetError(
            f"Annotation {annotation.get('id', '<unknown>')} has a non-positive bbox"
        )
    return x, y, x + width, y + height


def _polygon_points(annotation: Mapping[str, Any]) -> Iterable[tuple[float, float]]:
    """Yield points from a COCO polygon segmentation."""

    segmentation = annotation.get("segmentation")
    if segmentation is None:
        return
    if isinstance(segmentation, dict):
        raise CropDatasetError(
            f"Annotation {annotation.get('id', '<unknown>')} uses unsupported COCO RLE "
            "segmentation; only bounding boxes and polygons are supported"
        )
    if not isinstance(segmentation, list):
        raise CropDatasetError(
            f"Annotation {annotation.get('id', '<unknown>')} has an invalid segmentation"
        )

    # Standard COCO polygons are a list of flat coordinate lists. Supporting a
    # single flat list as well makes the cropper tolerant of common exporters.
    polygons: list[Sequence[Any]]
    if segmentation and all(isinstance(value, (int, float)) for value in segmentation):
        polygons = [segmentation]
    else:
        if any(not isinstance(polygon, list) for polygon in segmentation):
            raise CropDatasetError(
                f"Annotation {annotation.get('id', '<unknown>')} has an invalid polygon"
            )
        polygons = segmentation

    for polygon in polygons:
        if len(polygon) < 6 or len(polygon) % 2:
            raise CropDatasetError(
                f"Annotation {annotation.get('id', '<unknown>')} has an invalid polygon"
            )
        for index in range(0, len(polygon), 2):
            x = polygon[index]
            y = polygon[index + 1]
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in (x, y)
            ):
                raise CropDatasetError(
                    f"Annotation {annotation.get('id', '<unknown>')} has a non-numeric polygon"
                )
            if not math.isfinite(x) or not math.isfinite(y):
                raise CropDatasetError(
                    f"Annotation {annotation.get('id', '<unknown>')} has a non-finite polygon"
                )
            yield x, y


def calculate_crop_bounds(
    annotations: Sequence[Mapping[str, Any]],
    image_width: int,
    image_height: int,
    padding: int = 10,
) -> CropBounds:
    """Calculate a padded crop containing every bbox and polygon point."""

    if not annotations:
        raise CropDatasetError("Cannot calculate crop bounds without annotations")
    if image_width <= 0 or image_height <= 0:
        raise CropDatasetError("Image dimensions must be positive")
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise CropDatasetError("Padding must be a non-negative integer")

    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf

    for annotation in annotations:
        left, top, right, bottom = _bbox_extrema(annotation)
        min_x = min(min_x, left)
        min_y = min(min_y, top)
        max_x = max(max_x, right)
        max_y = max(max_y, bottom)

        for x, y in _polygon_points(annotation):
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    crop = CropBounds(
        left=max(0, math.floor(min_x - padding)),
        top=max(0, math.floor(min_y - padding)),
        right=min(image_width, math.ceil(max_x + padding)),
        bottom=min(image_height, math.ceil(max_y + padding)),
    )
    if crop.width <= 0 or crop.height <= 0:
        raise CropDatasetError("Annotations do not overlap the image bounds")
    return crop


def _safe_relative_image_path(file_name: Any) -> Path:
    if not isinstance(file_name, str) or not file_name.strip():
        raise CropDatasetError("Every COCO image must have a non-empty 'file_name'")
    relative_path = Path(file_name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise CropDatasetError(f"Unsafe image file name: {file_name}")
    return relative_path


def preflight_coco(coco: Mapping[str, Any]) -> None:
    """Validate references and geometry before writing any cropped images."""

    image_ids: set[int | str] = set()
    for image_record in coco["images"]:
        if not isinstance(image_record, dict) or "id" not in image_record:
            raise CropDatasetError("Every COCO image must be an object with an 'id'")
        image_id = image_record["id"]
        if image_id in image_ids:
            raise CropDatasetError(f"Duplicate COCO image id: {image_id}")
        image_ids.add(image_id)
        _safe_relative_image_path(image_record.get("file_name"))

    for annotation in coco["annotations"]:
        if not isinstance(annotation, dict):
            raise CropDatasetError("Every COCO annotation must be an object")
        if "image_id" not in annotation:
            raise CropDatasetError("Every annotation must contain an 'image_id'")
        if annotation["image_id"] not in image_ids:
            raise CropDatasetError(
                f"Annotation {annotation.get('id', '<unknown>')} references missing "
                f"image id {annotation['image_id']}"
            )
        _bbox_extrema(annotation)
        # Exhaust the generator so invalid polygons and RLE masks fail now,
        # before any output image is written.
        list(_polygon_points(annotation))


def _translate_segmentation(
    annotation: Mapping[str, Any], crop: CropBounds
) -> list[Any] | None:
    segmentation = annotation.get("segmentation")
    if segmentation is None:
        return None

    # Geometry has already passed _polygon_points during preflight.
    is_flat_polygon = bool(segmentation) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in segmentation
    )
    polygons = [segmentation] if is_flat_polygon else segmentation
    translated_polygons: list[list[int | float]] = []
    for polygon in polygons:
        translated_polygon: list[int | float] = []
        for index, coordinate in enumerate(polygon):
            offset = crop.left if index % 2 == 0 else crop.top
            translated_polygon.append(coordinate - offset)
        translated_polygons.append(translated_polygon)

    return translated_polygons[0] if is_flat_polygon else translated_polygons


def translate_annotation(
    annotation: Mapping[str, Any], crop: CropBounds
) -> dict[str, Any]:
    """Return an annotation translated into a cropped image's coordinates."""

    translated = copy.deepcopy(dict(annotation))
    bbox = annotation["bbox"]
    translated["bbox"] = [
        bbox[0] - crop.left,
        bbox[1] - crop.top,
        bbox[2],
        bbox[3],
    ]
    if "segmentation" in annotation:
        translated["segmentation"] = _translate_segmentation(annotation, crop)
    return translated


def rewrite_coco_for_crops(
    coco: Mapping[str, Any], report: SplitCropReport
) -> dict[str, Any]:
    """Create a COCO document containing only retained, translated images."""

    crops_by_image_id = {
        cropped.image_id: cropped for cropped in report.cropped_images
    }
    rewritten = copy.deepcopy(dict(coco))

    rewritten_images: list[dict[str, Any]] = []
    for image_record in coco["images"]:
        cropped = crops_by_image_id.get(image_record["id"])
        if cropped is None:
            continue
        rewritten_image = copy.deepcopy(image_record)
        rewritten_image["width"] = cropped.crop.width
        rewritten_image["height"] = cropped.crop.height
        rewritten_images.append(rewritten_image)

    rewritten_annotations = [
        translate_annotation(annotation, crops_by_image_id[annotation["image_id"]].crop)
        for annotation in coco["annotations"]
        if annotation["image_id"] in crops_by_image_id
    ]

    rewritten["images"] = rewritten_images
    rewritten["annotations"] = rewritten_annotations
    return rewritten


def write_coco(coco: Mapping[str, Any], annotation_path: str | Path) -> None:
    """Write a COCO document as readable UTF-8 JSON."""

    annotation_path = Path(annotation_path)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with annotation_path.open("w", encoding="utf-8") as annotation_file:
            json.dump(coco, annotation_file, indent=2, ensure_ascii=False)
            annotation_file.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise CropDatasetError(
            f"Could not write annotation file {annotation_path}: {exc}"
        ) from exc


def crop_split(
    source_split: str | Path,
    output_split: str | Path,
    padding: int = 10,
) -> SplitCropReport:
    """Crop all annotated images in one Roboflow COCO split."""

    source_split = Path(source_split).resolve()
    output_split = Path(output_split).resolve()
    if source_split == output_split:
        raise CropDatasetError("Source and output split directories must be different")

    coco = load_coco(source_split / ANNOTATION_FILENAME)
    preflight_coco(coco)
    annotations_by_image = group_annotations_by_image(coco["annotations"])
    report = SplitCropReport(split=source_split.name)

    for image_record in coco["images"]:
        image_id = image_record["id"]
        relative_path = _safe_relative_image_path(image_record.get("file_name"))
        annotations = annotations_by_image.get(image_id, [])
        if not annotations:
            report.skipped_empty_images.append(relative_path.as_posix())
            continue

        source_image_path = source_split / relative_path
        if not source_image_path.is_file():
            raise CropDatasetError(f"Image file not found: {source_image_path}")

        output_image_path = output_split / relative_path
        output_image_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(source_image_path) as source_image:
                source_width, source_height = source_image.size
                crop = calculate_crop_bounds(
                    annotations,
                    image_width=source_width,
                    image_height=source_height,
                    padding=padding,
                )
                cropped_image = source_image.crop(crop.as_pillow_box())
                cropped_image.save(output_image_path)
        except CropDatasetError:
            raise
        except (OSError, ValueError) as exc:
            raise CropDatasetError(
                f"Could not crop image {source_image_path}: {exc}"
            ) from exc

        report.cropped_images.append(
            CroppedImage(
                image_id=image_id,
                file_name=relative_path.as_posix(),
                source_width=source_width,
                source_height=source_height,
                crop=crop,
                annotation_count=len(annotations),
            )
        )

    rewritten_coco = rewrite_coco_for_crops(coco, report)
    write_coco(rewritten_coco, output_split / ANNOTATION_FILENAME)
    return report


def discover_splits(dataset_root: str | Path) -> list[Path]:
    """Find supported Roboflow split directories in stable order."""

    dataset_root = Path(dataset_root)
    discovered = [
        dataset_root / split
        for split in DEFAULT_SPLITS
        if (dataset_root / split / ANNOTATION_FILENAME).is_file()
    ]
    if not discovered:
        raise CropDatasetError(
            f"No COCO splits found under {dataset_root}; expected {ANNOTATION_FILENAME} "
            f"inside one of: {', '.join(DEFAULT_SPLITS)}"
        )
    return discovered


def crop_dataset(
    source_root: str | Path,
    output_root: str | Path,
    padding: int = 10,
) -> DatasetCropReport:
    """Crop every discovered split in a Roboflow COCO dataset."""

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if source_root == output_root:
        raise CropDatasetError("Source and output dataset directories must be different")

    reports: dict[str, SplitCropReport] = {}
    for source_split in discover_splits(source_root):
        reports[source_split.name] = crop_split(
            source_split=source_split,
            output_split=output_root / source_split.name,
            padding=padding,
        )
    return DatasetCropReport(splits=reports)
