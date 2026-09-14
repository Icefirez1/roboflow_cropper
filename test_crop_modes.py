import copy
import json
import zipfile

import pytest
from PIL import Image

from crop_dataset import (
    ANNOTATION_FILENAME, CropDatasetError, CropMode, crop_dataset,
    cookie_cutter_file_name, load_coco, write_coco,
)
from main import main
from validate_dataset import validate_dataset, main as validate_main


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    split = root / "train"
    split.mkdir(parents=True)
    Image.new("RGB", (24, 20), (80, 120, 160)).save(split / "sample.jpg")
    coco = {
        "info": {"description": "keep me"},
        "categories": [{"id": 1, "name": "object"}],
        "images": [
            {"id": 1, "file_name": "sample.jpg", "width": 24, "height": 20, "extra": "keep"},
            {"id": 2, "file_name": "empty.jpg", "width": 24, "height": 20},
        ],
        "annotations": [
            {"id": 11, "image_id": 1, "category_id": 1, "bbox": [3, 3, 4, 4],
             "segmentation": [[3, 3, 7, 3, 7, 7, 3, 7]], "area": 16, "extra": [1, 2]},
            {"id": 12, "image_id": 1, "category_id": 1, "bbox": [14, 10, 4, 4],
             "segmentation": [[14, 10, 18, 10, 18, 14, 14, 14]], "area": 16},
        ],
    }
    write_coco(coco, split / ANNOTATION_FILENAME)
    return root


def edit_source(source, edit):
    path = source / "train" / ANNOTATION_FILENAME
    coco = load_coco(path)
    edit(coco)
    write_coco(coco, path)


@pytest.mark.parametrize("mode", list(CropMode))
def test_cli_reports_validation_and_zip(source, tmp_path, mode, capsys):
    output = tmp_path / mode.value
    args = [str(source), str(output), "--padding", "2"]
    if mode == CropMode.COOKIE_CUTTER:
        args += ["--crop-mode", mode.value]
    assert main(args) == 0
    report = json.loads((output / "processing_report.json").read_text())
    assert report["crop_mode"] == mode.value
    assert report["totals"]["cropped_images"] == 1
    assert report["totals"]["skipped_empty_images"] == 1
    coco = load_coco(output / "train" / ANNOTATION_FILENAME)
    original = load_coco(source / "train" / ANNOTATION_FILENAME)
    assert coco["info"] == original["info"]
    assert coco["categories"] == original["categories"]
    assert coco["images"][0]["extra"] == "keep"
    for old, new in zip(original["annotations"], coco["annotations"]):
        expected = copy.deepcopy(old)
        expected["bbox"][:2] = [old["bbox"][0] - 1, old["bbox"][1] - 1]
        expected["segmentation"] = [[v - 1 for v in old["segmentation"][0]]]
        assert new == expected
    name = coco["images"][0]["file_name"]
    record = report["splits"]["train"]["crops"][0]
    assert record["source_file_name"] == "sample.jpg"
    assert record["output_file_name"] == name
    with Image.open(output / "train" / name) as image:
        assert image.size == (19, 15)
        if mode == CropMode.BOUNDS:
            assert name == "sample.jpg"
            assert image.format == "JPEG" and image.mode == "RGB"
            assert "roboflow_cropper" not in coco
        else:
            assert image.format == "PNG" and image.mode == "RGBA"
            alpha = image.getchannel("A")
            assert alpha.getpixel((3, 3)) == 255
            assert alpha.getpixel((14, 10)) == 255
            assert alpha.getpixel((10, 8)) == 0
            assert alpha.getpixel((0, 0)) == 0
    assert validate_main([str(output), "--source", str(source), "--json"]) == 0
    with zipfile.ZipFile(output.with_suffix(".zip")) as archive:
        assert archive.testzip() is None
        assert f"train/{name}" in archive.namelist()
        assert f"train/{ANNOTATION_FILENAME}" in archive.namelist()
        assert "processing_report.json" in archive.namelist()


def test_multiple_polygons_overlap_and_flat_polygon(source, tmp_path):
    def edit(coco):
        coco["annotations"][0]["segmentation"].append([9, 3, 11, 3, 11, 5, 9, 5])
        coco["annotations"][1]["segmentation"] = [5, 5, 9, 5, 9, 9, 5, 9]
    edit_source(source, edit)
    output = tmp_path / "out"
    crop_dataset(source, output, padding=1, crop_mode=CropMode.COOKIE_CUTTER)
    coco = load_coco(output / "train" / ANNOTATION_FILENAME)
    with Image.open(output / "train" / coco["images"][0]["file_name"]) as image:
        alpha = image.getchannel("A")
        for point in [(2, 2), (8, 2), (4, 4), (6, 6)]:
            assert alpha.getpixel(point) == 255
        assert alpha.getpixel((6, 1)) == 0
    assert validate_dataset(output, source).valid


@pytest.mark.parametrize("segmentation", [None, []])
@pytest.mark.parametrize("padding", [0, 3])
def test_box_fallback_and_border(source, tmp_path, segmentation, padding):
    def edit(coco):
        coco["annotations"] = [dict(coco["annotations"][0], bbox=[0, 0, 4, 4], segmentation=segmentation)]
    edit_source(source, edit)
    output = tmp_path / "out"
    assert main([str(source), str(output), "--crop-mode", "cookie-cutter", "--padding", str(padding)]) == 0
    coco = load_coco(output / "train" / ANNOTATION_FILENAME)
    assert coco["roboflow_cropper"]["fallback_annotation_ids"] == [11]
    report = json.loads((output / "processing_report.json").read_text())
    assert report["totals"]["fallback_count"] == 1
    assert report["splits"]["train"]["fallback_annotation_ids"] == [11]
    assert report["warnings"]
    with Image.open(output / "train" / coco["images"][0]["file_name"]) as image:
        assert image.size == (4 + padding, 4 + padding)
        alpha = image.getchannel("A")
        assert alpha.getpixel((3, 3)) == 255
        if padding:
            assert alpha.getpixel((4, 3)) == 0


def test_deterministic_names_with_same_stem(source, tmp_path):
    def edit(coco):
        coco["images"][1] = dict(coco["images"][0], id="1", file_name="sample.png")
        coco["annotations"].append(dict(coco["annotations"][0], id=13, image_id="1"))
    edit_source(source, edit)
    Image.new("RGBA", (24, 20), (1, 2, 3, 0)).save(source / "train" / "sample.png")
    names = []
    for directory in ["first", "second"]:
        output = tmp_path / directory
        crop_dataset(source, output, crop_mode="cookie-cutter")
        coco = load_coco(output / "train" / ANNOTATION_FILENAME)
        names.append([image["file_name"] for image in coco["images"]])
        assert validate_dataset(output, source).valid
    assert names[0] == names[1]
    assert len(set(names[0])) == 2
    assert names[0][0] == cookie_cutter_file_name("sample.jpg", 1)


@pytest.mark.parametrize("point,value", [((0, 0), 255), ((4, 4), 0), ((4, 4), 128)])
def test_reject_tampered_alpha(source, tmp_path, point, value):
    output = tmp_path / "out"
    crop_dataset(source, output, padding=3, crop_mode="cookie-cutter")
    coco = load_coco(output / "train" / ANNOTATION_FILENAME)
    path = output / "train" / coco["images"][0]["file_name"]
    with Image.open(path) as image:
        changed = image.copy()
    changed.putpixel(point, (80, 120, 160, value))
    changed.save(path)
    report = validate_dataset(output)
    assert not report.valid
    assert any("alpha channel does not match" in error for error in report.splits["train"].errors)


@pytest.mark.parametrize("tamper", ["fallback", "collision", "rgb", "jpeg", "missing_image", "representation"])
def test_reject_invalid_cookie_dataset(source, tmp_path, tamper):
    output = tmp_path / "out"
    crop_dataset(source, output, crop_mode="cookie-cutter")
    path = output / "train" / ANNOTATION_FILENAME
    coco = load_coco(path)
    image_path = output / "train" / coco["images"][0]["file_name"]
    if tamper == "fallback":
        coco["roboflow_cropper"]["fallback_annotation_ids"] = [11]
    elif tamper == "collision":
        coco["images"].append(dict(coco["images"][0], id=3))
    elif tamper in ("rgb", "jpeg"):
        with Image.open(image_path) as image:
            changed = image.convert("RGB")
        changed.save(image_path, format="PNG" if tamper == "rgb" else "JPEG")
    elif tamper == "missing_image":
        coco["images"] = []
        coco["annotations"] = []
    else:
        coco["roboflow_cropper"]["alpha_mask_representation"] = "unknown"
    write_coco(coco, path)
    assert not validate_dataset(output, source).valid


def test_rle_rejected_before_image_output(source, tmp_path):
    edit_source(source, lambda coco: coco["annotations"][0].update(segmentation={"counts": [], "size": [20, 24]}))
    output = tmp_path / "out"
    with pytest.raises(CropDatasetError, match="RLE"):
        crop_dataset(source, output, crop_mode="cookie-cutter")
    assert not output.exists()


def test_generated_collision_rejected_before_writing(source, tmp_path, monkeypatch):
    import crop_dataset as crop_module

    def edit(coco):
        coco["annotations"].append(dict(coco["annotations"][0], id=13, image_id=2))
    edit_source(source, edit)
    monkeypatch.setattr(crop_module, "cookie_cutter_file_name", lambda *args: "collision.png")
    output = tmp_path / "out"
    with pytest.raises(CropDatasetError, match="filename collision"):
        crop_dataset(source, output, crop_mode="cookie-cutter")
    assert not output.exists()


def test_invalid_mask_prevents_packaging(source, tmp_path, monkeypatch):
    import main as cli

    def corrupt_after_crop(*args, **kwargs):
        report = crop_dataset(*args, **kwargs)
        output = kwargs["output_root"]
        coco = load_coco(output / "train" / ANNOTATION_FILENAME)
        path = output / "train" / coco["images"][0]["file_name"]
        with Image.open(path) as image:
            changed = image.copy()
        changed.putalpha(255)
        changed.save(path)
        return report

    monkeypatch.setattr(cli, "crop_dataset", corrupt_after_crop)
    output = tmp_path / "out"
    assert cli.main([str(source), str(output), "--crop-mode", "cookie-cutter"]) == 1
    assert not output.with_suffix(".zip").exists()
    report = json.loads((output / "processing_report.json").read_text())
    assert report["status"] == "invalid"
