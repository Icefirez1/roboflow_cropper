# Roboflow COCO Cropper

Crop unused borders from a Roboflow COCO dataset without losing its object
annotations. For each image, the tool finds the rectangle containing all COCO
bounding boxes and polygon segmentations, adds optional context padding, crops
the image, and translates the annotations into the new coordinate system.

The source dataset is never modified. Successful runs produce a validated COCO
dataset, a detailed processing report, and (by default) a ZIP ready for dataset
import.

## Requirements

- Python 3.10 or newer
- A Roboflow dataset exported as COCO JSON or COCO Segmentation

From this directory, create an environment and install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Pillow is the only runtime dependency. Pytest is included for project testing.

## Downloading the input from Roboflow

1. Open the desired dataset version in Roboflow.
2. Select **Download Dataset**.
3. Select **COCO JSON** for bounding boxes or **COCO Segmentation** when polygon
   annotations must be retained.
4. Download and extract the archive locally.
5. Pass the extracted dataset root—not an individual split—to `main.py`.

The cropper automatically discovers `train`, `valid`, `val`, and `test` split
folders. Each discovered split must contain `_annotations.coco.json` beside its
images:

```text
downloaded_dataset/
├── train/
│   ├── image_001.jpg
│   └── _annotations.coco.json
├── valid/                 # "val" is also supported
│   ├── image_002.jpg
│   └── _annotations.coco.json
└── test/
    ├── image_003.jpg
    └── _annotations.coco.json
```

## Running the cropper

Basic usage:

```powershell
python main.py INPUT_DATASET OUTPUT_DATASET
```

Example with paths inside this project:

```powershell
python main.py .\data\downloaded_dataset .\data\cropped_dataset --padding 10
```

Options:

- `--crop-mode {bounds,cookie-cutter}` selects the crop representation. The
  default `bounds` keeps the established rectangular crop behavior, filenames,
  image formats, and annotation coordinates.
- `--padding PIXELS` retains that many pixels around the combined annotation
  boundary. It defaults to `10`, must be an integer, and may be `0` for a tight
  crop.
- `--overwrite` replaces an existing output directory and, when ZIP packaging
  is enabled, its matching ZIP. Without this flag, existing output is left
  untouched and the command fails.
- `--no-zip` creates and validates the output folders without packaging them.

The input and output must be separate, non-overlapping directories. A normal
successful run exits with code `0`; processing or validation errors exit with
code `1`; an interrupted run exits with code `130`.

## Output

For an output argument of `data/cropped_dataset`, the default result is:

```text
data/
├── cropped_dataset/
│   ├── train/
│   │   ├── image_001.jpg
│   │   └── _annotations.coco.json
│   ├── valid/
│   │   ├── image_002.jpg
│   │   └── _annotations.coco.json
│   ├── test/
│   │   ├── image_003.jpg
│   │   └── _annotations.coco.json
│   └── processing_report.json
└── cropped_dataset.zip
```

The ZIP contains the split folders directly at its root rather than wrapping
them in another `cropped_dataset` directory.

### Cropping behavior

- One crop is calculated per image from the union of every bounding box and
  polygon point associated with that image.
- Padding is clipped at the original image boundaries.
- Bounding-box `x`/`y` and every polygon point are translated by the crop
  origin. Bounding-box sizes, annotation areas, IDs, categories, and unknown
  Roboflow metadata are preserved.
- COCO image width and height are updated to match the cropped file.
- Images without annotations are omitted and listed in the report.
- Split names are preserved. Bounds mode preserves image filenames; cookie-cutter
  mode generates PNG filenames. IDs are not renumbered.
- Images are cropped but never resized.

### Cookie-cutter mode

```powershell
python main.py INPUT_DATASET OUTPUT_DATASET --crop-mode cookie-cutter --padding 10
```

This mode produces one transparent RGBA PNG per annotated source image. All
polygons are filled into one opaque mask, including multiple polygons per
annotation and overlapping objects. Pixels outside that union, including gaps
between objects and padding, are fully transparent. The combined crop rectangle
and annotation translations are the same as in bounds mode. Padding still adds
space around that rectangle and is clipped at source image borders.

Annotations without polygons (missing, null, or empty segmentation) use their
bounding boxes as rectangular masks. Fallback annotation IDs, counts, and an
aggregate warning per split appear in the processing report. Polygon edges use
Pillow's hard, inclusive rasterization without antialiasing; fallback boxes use
floor/ceil pixel coverage with exclusive right and bottom bounds. The generated
mask replaces any source alpha channel, making annotated regions fully opaque.

PNG filenames combine the original stem with a stable SHA-256 hash of the COCO
image ID; `images[].file_name` and report source/output mappings identify each
result. Image and annotation IDs, areas, categories, and other metadata are
preserved. Every source image and annotation is retained except the established
skipping of images without annotations. Transparent PNG archives may be larger
than the corresponding JPEG datasets.

Cookie-cutter COCO files include `roboflow_cropper` metadata recording
`crop_mode`, `padding`, `alpha_mask_representation`, and
`fallback_annotation_ids`. Both Python entry points accept
`crop_mode=CropMode.COOKIE_CUTTER` (import `CropMode` from `crop_dataset`);
their default remains `CropMode.BOUNDS`.

### Validation and report

Packaging occurs only after all output splits pass validation. Checks include:

- Image existence, readability, and agreement with declared dimensions
- Valid image and category references
- Positive bounding-box dimensions and polygon areas
- Bounding boxes and polygon coordinates remaining within image boundaries
- Preservation of annotation IDs for every retained source image
- Retention of all annotated source images when a source dataset is supplied
- Unique image filenames
- For cookie-cutter metadata: PNG format with an alpha channel, matching fallback
  IDs, and an exact pixel comparison against the reconstructed annotation mask

Roboflow may round a boundary box up to half a pixel beyond an image edge. The
validator accepts at most `0.51` pixels for this known representation artifact
and records an aggregate warning in `processing_report.json`.

The report includes input/output counts, skipped filenames, every crop rectangle
and source dimension, selected mode and padding, source/output filenames,
fallback counts and IDs, warnings, and full validation results.
If validation fails, the report is still written for diagnosis, but no ZIP is
created.

## Standalone validation

Validate any compatible output dataset:

```powershell
python validate_dataset.py .\data\cropped_dataset
```

Supply the original dataset to additionally verify that retained annotation IDs
were not lost or added:

```powershell
python validate_dataset.py .\data\cropped_dataset `
  --source .\data\downloaded_dataset
```

Use `--json` to print a machine-readable validation result. The validator exits
with code `0` when valid and `1` when any error is found.

Run the regression suite with `python -m pytest -q`. It covers both modes,
mask geometry and tampering, fallback reporting, and ZIP integrity and layout.

## Uploading to Roboflow

After a successful run, import `cropped_dataset.zip` into the intended Roboflow
project. If the selected Roboflow upload flow requests individual files rather
than an archive, extract the ZIP and upload the split folders with their images
and `_annotations.coco.json` files. Review the image, annotation, and category
counts shown by Roboflow before completing the import.

The scripts intentionally do not store API keys or download/upload data through
the Roboflow SDK.

## Current limitations

- COCO bounding boxes and polygon segmentations are supported.
- COCO compressed and uncompressed RLE masks are rejected during preflight;
  they are not decoded, cropped, or re-encoded.
- Invalid or missing images, malformed geometry, missing references, and unsafe
  image paths stop processing rather than being silently repaired.
- Bounding boxes and polygons are translated, not clipped. Source coordinates
  more than `0.51` pixels outside their image will fail output validation.
- Existing output is removed only when `--overwrite` is explicitly supplied.
