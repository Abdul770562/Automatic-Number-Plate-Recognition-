# Automatic Number Plate Recognition (ANPR)

A Python-based automatic number plate recognition prototype that detects vehicle license plates in an image, prepares each detected plate for recognition, and reads the plate text with a vision-language OCR model. The project combines a custom Ultralytics YOLO plate detector with Microsoft's TrOCR (`microsoft/trocr-large-printed`).

> **Project status:** This repository contains a single interactive Python script and a trained detector checkpoint. It is a research/learning prototype, not a production-ready traffic or enforcement system. Recognition accuracy depends on image quality, plate style, and the detector/OCR models.

## Features

- Detects license plate regions using the included `license_plate_detector.pt` YOLO model.
- Accepts either an existing image file or a still image captured from the default webcam.
- Applies contrast enhancement and noise reduction to plate crops before OCR.
- Estimates whether a plate is single-line or two-line and adjusts the crop accordingly.
- Uses TrOCR to generate text from each detected plate.
- Selects the recognized text associated with the highest-confidence detection.
- Can save intermediate images for inspection and writes the final result and timestamp to `ocr_result.txt`.

## How it works

1. `ANPR.py` loads the YOLO detector and the TrOCR processor/model.
2. The program prompts you to use an existing image or capture one from the webcam.
3. The image is resized to 640 × 640 and passed to YOLO for plate detection.
4. Each detected plate crop is enhanced with grayscale conversion, CLAHE contrast equalization, and bilateral filtering.
5. A horizontal projection heuristic estimates whether the plate has one or two text lines. Two-line crops are split and concatenated horizontally for OCR.
6. TrOCR generates text for each crop. The result associated with the highest YOLO detection confidence is returned.
7. The result is printed, saved to `ocr_result.txt`, and debug images are written when debug output is enabled.

## Repository contents

| Path | Description |
| --- | --- |
| `ANPR.py` | Interactive detection, preprocessing, OCR, and output script. |
| `license_plate_detector.pt` | YOLO license plate detection weights loaded by the script. |
| `requirements.txt` | Python package dependencies. |
| `WhatsApp Image ... .jpg` | Example image files included in the repository; filenames contain spaces. |

## Requirements

- Python 3.9 or later is recommended. The dependency list does not pin a Python version; use a version supported by the selected PyTorch release.
- A webcam is optional. Existing image files can be used instead.
- Internet access is needed on first run to download the TrOCR model from Hugging Face, unless it is already cached locally.
- PyTorch can run on a CPU; a compatible CUDA setup can accelerate inference. Install the PyTorch packages appropriate for your operating system and hardware from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Setup

1. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/Abdul770562/Automatic-Number-Plate-Recognition-.git
   cd Automatic-Number-Plate-Recognition-
   ```

2. Create and activate a virtual environment (recommended):

   **Windows (PowerShell):**

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   **macOS/Linux:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install PyTorch. Choose a CPU or CUDA command for your platform from the official PyTorch selector. The repository's `requirements.txt` also lists `torch`, `torchvision`, and `torchaudio`; install the selected compatible versions before continuing.

4. Install the remaining project dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. Configure the detector checkpoint path in `ANPR.py`. The script currently uses an author-machine-specific absolute path:

   ```python
   yolo_model = YOLO("C:/Users/Abdul/Downloads/anpr/license_plate_detector.pt")
   ```

   Replace it with the path to `license_plate_detector.pt` on your machine. For example, when running the script from the repository root, use:

   ```python
   yolo_model = YOLO("license_plate_detector.pt")
   ```

   The checkpoint is included in this repository. TrOCR is loaded from `microsoft/trocr-large-printed` and is downloaded/cached automatically by Transformers when needed.

## Run

From the repository root, start the interactive program:

```bash
python ANPR.py
```

When prompted:

- Enter `y` to provide the path to an existing image, then enter its path. Relative paths are resolved from the current working directory.
- Enter `n` to capture an image from the default webcam. In the webcam window, press **c** to save a frame or **q** to cancel.

Example image path (use the actual filename present in your checkout):

```text
WhatsApp Image 2025-06-24 at 11.15.33_61c3539f.jpg
```

The program prints the detection and OCR information to the terminal. The result is written to `ocr_result.txt` in the current working directory, with a timestamp and the recognized plate text.

## Outputs and debugging

The main script calls `detect_and_read_plate(img_path, save_debug=True)`, so its normal interactive run saves intermediate files in the current working directory:

- `plate_crop_<n>.jpg` — raw crop for a detected plate.
- `enhanced_crop_<n>.jpg` — contrast-enhanced crop passed into plate-layout handling.
- `aligned_plate_<n>.jpg` — crop arranged for OCR.
- `ocr_result.txt` — final selected text and run timestamp.

Existing files with these names may be overwritten on subsequent runs. The script displays aligned plate images with Matplotlib when those debug files are found.

The processing function can also be imported and called with debug output disabled:

```python
from ANPR import detect_and_read_plate

text = detect_and_read_plate("path/to/vehicle-image.jpg", save_debug=False)
print(text)
```

Note that importing `ANPR.py` loads both models immediately, so the detector checkpoint path must be configured and the required model dependencies must be available first.

## Configuration and implementation notes

- **Detector weights:** The YOLO weights path is currently hard-coded near the top of `ANPR.py`; update it for your environment.
- **OCR model:** The script uses `microsoft/trocr-large-printed`. Model loading and inference may require substantial memory and can be slow on CPU.
- **Input resizing:** The full image is resized to 640 × 640 before inference. This can change the image's aspect ratio and may affect detection on some inputs.
- **Plate layout:** Single-line/two-line classification is heuristic-based. It may not work consistently across plate formats or image conditions.
- **Confidence:** The printed confidence is YOLO's detection confidence, not a calibrated measure of OCR text correctness. The script chooses the text belonging to the highest-confidence detected box.
- **Debug artifacts:** The script writes debug images to the current directory when `save_debug=True`.

## Troubleshooting

### The detector checkpoint cannot be found

Update the `YOLO(...)` path in `ANPR.py` to point to the included `license_plate_detector.pt`, or to the correct location of your own checkpoint. Run the program from the repository root if using a relative path such as `license_plate_detector.pt`.

### Webcam cannot be opened

Check that a camera is connected and available to the operating system, close other applications using it, and verify that OpenCV has camera permission. The script uses camera index `0`; systems with multiple cameras may require changing the index in `capture_image_from_webcam`.

### No license plate is detected

Try a clearer, higher-resolution image in which the plate is visible and unobstructed. Ensure the checkpoint is the expected detector model. The included model may not generalize to every country, plate design, angle, or lighting condition.

### OCR output is inaccurate or empty

OCR quality depends on plate size, sharpness, exposure, perspective, and plate format. Use a closer, well-lit image with the plate in focus. Detection confidence does not verify that the recognized characters are correct. Review the saved crop and aligned debug image when available.

### PyTorch/CUDA installation errors

Install a PyTorch build compatible with your Python version, operating system, and CUDA driver using the official PyTorch installation selector. CPU inference is an alternative if no compatible GPU setup is available.

### Hugging Face model download errors

Check network access and try running again. Transformers caches downloaded files locally; on restricted networks, arrange access to the model through your organization's approved process.

## Privacy, safety, and responsible use

License plate images and recognized plate text may be sensitive or personally identifiable data depending on jurisdiction and use. Obtain appropriate authorization, provide required notices, and follow applicable privacy, data-protection, and surveillance laws. Minimize retention and access, secure image/result files, and do not use model output as the sole basis for consequential decisions. This prototype does not provide identity verification, legal compliance, or guaranteed recognition accuracy.

## Limitations

- Designed as a simple interactive prototype, rather than a continuously running video-stream or multi-camera service.
- Uses a fixed webcam index and fixed image-resize dimensions.
- Contains an environment-specific model path that must be changed before running elsewhere.
- Does not validate OCR text against regional plate formats or report a separate OCR confidence score.
- Does not include automated tests, a command-line argument interface, or a formal deployment configuration.

## License

No standard open-source license is specified in the repository metadata. Unless a license is added, reuse and redistribution may be restricted by copyright. The included model weights and the referenced TrOCR model may also have separate terms; review their respective licenses and usage conditions before use.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection.
- [Microsoft TrOCR](https://huggingface.co/microsoft/trocr-large-printed) for printed-text recognition.
- [OpenCV](https://opencv.org/) for image processing and webcam capture.
