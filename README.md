# Pedestrian Detection Dashboard

A FastAPI web application for detecting pedestrians in uploaded images and comparing three trained computer-vision models. The project includes a browser dashboard, sample images, pre-trained Keras/TensorFlow model files, training histories, and a standalone Python evaluation script.

## Features

- Upload PNG, JPG, or JPEG images from the dashboard.
- Select and test the sample images stored in `testImages/`.
- Display the original image beside an annotated image with bounding boxes.
- Return the pedestrian count and bounding-box coordinates as JSON.
- Show model accuracy, precision, recall, and F1-score comparisons.
- Plot training accuracy and loss histories.
- Support the included Faster R-CNN, YOLOv5-style attention, and YOLOv6-style models.

## How Detection Works

The web API chooses the detector based on the uploaded image:

1. Color images are processed with OpenCV's built-in HOG people detector.
2. Grayscale or thermal-style images are resized to `64 x 64` and processed with the included custom YOLO model (`model/yolo_weights.hdf5`).

The API returns an annotated PNG encoded as a data URL. The dashboard uses this response to display the result.

> The dashboard's color-image path does not use the saved Keras models. The custom YOLO model is used for images detected as grayscale/thermal.

## Project Structure

```text
.
├── app.py                     # FastAPI server and prediction API
├── Attention.py               # Custom attention layer used by the YOLO model
├── Pedestriansdetection.py    # Dataset processing, training, evaluation, and plotting script
├── requirements.txt            # Python dependencies
├── start.bat                  # Windows menu for launching the project
├── model/                     # Datasets, trained models, and history files
├── static/
│   ├── index.html              # Dashboard markup
│   ├── script.js               # Dashboard behavior and API calls
│   └── style.css               # Dashboard styling
└── testImages/                # Sample images for dashboard testing
```

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer recommended
- A machine with enough disk space for TensorFlow and the model files
- Approximately 520 MB for the tracked datasets and model artifacts

## Installation

From the project directory, create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the server directly with `venv\\Scripts\\python.exe`, or use Command Prompt:

```bat
venv\Scripts\activate.bat
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Web Dashboard

With the virtual environment activated:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser.

On Windows, `start.bat` provides a menu. Option 2 starts the dashboard and opens the browser automatically:

```bat
start.bat
```

The application loads the saved model files and dataset arrays during startup. This can take time and uses substantial memory. The first request should be made only after the server has finished loading.

## API Reference

### `GET /api/metrics`

Returns cached metrics for the three models. If dynamic metric calculation fails, the application returns fallback values.

### `GET /api/history`

Returns training accuracy and loss history data for the dashboard charts.

### `GET /api/samples`

Returns the names of image files in `testImages/`.

### `GET /api/samples/{filename}`

Returns a sample image. The filename must refer to an existing image in `testImages/`.

### `POST /api/predict`

Accepts an image upload as multipart form data using the field name `file`.

Example with PowerShell:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/predict -F "file=@testImages/1.png"
```

Successful responses have this shape:

```json
{
	"image": "data:image/png;base64,...",
	"pedestrians_count": 1,
	"detections": [
		{"box": [10, 20, 100, 220]}
	],
	"class_label": "Pedestrian Detected"
}
```

## Included Models and Data

| File | Purpose |
| --- | --- |
| `model/frcnn.hdf5` | Saved Faster R-CNN-style Keras model |
| `model/yolo_weights.hdf5` | Saved custom YOLOv5-style model with attention |
| `model/yolov6.hdf5` | Saved YOLOv6-style Keras model |
| `model/X.txt.npy` | Preprocessed image data |
| `model/Y.txt.npy` | Labels for the preprocessed data |
| `model/bb.txt.npy` | Normalized bounding-box data |
| `model/*history.pckl` and `model/yolov6.pckl` | Training history files |

The `.hdf5` files are tracked with Git LFS. After cloning, install Git LFS and fetch the model objects:

```bash
git lfs install
git lfs pull
```

Without the real LFS objects, model loading will fail because pointer files are not valid Keras models.

## Standalone Training and Evaluation Script

`Pedestriansdetection.py` contains the original dataset preparation, training, metric calculation, and plotting workflow. It expects the original CVC-09 dataset at:

```text
Dataset/
├── Annotations/
└── FramesPos/
```

That dataset is not included in this repository. If the processed arrays exist in `model/`, the script loads them; otherwise it attempts to build them from `Dataset/`. The script may train missing models, calculate metrics, open Matplotlib windows, and run a sample prediction. Run it with:

```bash
python Pedestriansdetection.py
```

On Windows, this is option 1 in `start.bat`.

## Model Metrics Shown by the Dashboard

The dashboard is configured with these fallback values when it cannot calculate metrics at startup:

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Faster R-CNN | 78.86% | 78.86% | 78.86% | 78.86% |
| YOLOv5 + Coordinated Attention | 94.55% | 94.55% | 94.55% | 94.55% |
| YOLOv6 | 100.00% | 100.00% | 100.00% | 100.00% |

When all required data and models load successfully, the API attempts to calculate metrics dynamically from the test split instead.

## Troubleshooting

### Model files fail to load

Confirm that the `.hdf5` files are present and are not Git LFS pointer files:

```bash
git lfs pull
```

### `ModuleNotFoundError`

Activate the virtual environment and reinstall dependencies:

```bash
python -m pip install -r requirements.txt
```

### Port 8000 is already in use

Run Uvicorn on another port:

```bash
uvicorn app:app --host 127.0.0.1 --port 8001
```

Then open `http://127.0.0.1:8001`.

### Dashboard loads but model metrics are fallback values

Check the server terminal for model, dataset, or history loading errors. The dashboard intentionally falls back to stored values so that it can still render when dynamic evaluation is unavailable.

## GitHub and Git LFS

This repository excludes local environments, Python caches, and VS Code settings through `.gitignore`. Large HDF5 files use Git LFS. Contributors should install Git LFS before cloning or pulling the repository.

## License and Dataset Note

No license file is currently included. Add an appropriate license before redistributing the project. The original CVC-09 dataset is not included; follow its terms and obtain it from its legitimate source before training.