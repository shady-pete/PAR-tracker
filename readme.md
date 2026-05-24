# Artificial Vision Group 1 - Person Tracking and Attribute Recognition

This project implements a system for tracking people in video streams, detecting line crossings, and recognizing attributes such as gender, bag presence, and hat presence. It utilizes YOLO for object detection and a custom Multi-Task CNN for attribute recognition (PAR).

## Description

The `tracker.py` script processes video files to:
1.  **Track People**: Uses YOLOv8 (or YOLOv8-seg) and BoT-SORT for robust person tracking.
2.  **Detect Line Crossings**: Counts people crossing defined lines in specific directions.
3.  **Recognize Attributes**: Classifies detected people for:
    *   **Gender**: Male / Female
    *   **Bag**: Yes / No
    *   **Hat**: Yes / No

## Libraries Used

The project relies on the following key libraries:

*   **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)**: For object detection and tracking.
*   **[PyTorch](https://pytorch.org/)**: Deep learning framework for the PAR model.
*   **[OpenCV (cv2)](https://opencv.org/)**: Video processing and drawing visualizations.
*   **[NumPy](https://numpy.org/)**: Numerical operations.
*   **[Shapely](https://shapely.readthedocs.io/)**: Geometric operations for polygon containment and line intersections.
*   **[Pillow (PIL)](https://python-pillow.org/)**: Image manipulation.

## Usage

To run the tracker on a video file:

```bash
python tracker.py --input <path_to_video> --output <path_to_output> --config <path_to_config>
```

### Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--input` | `../video.mp4` | Path to the input video file. |
| `--output` | `./output/output.mp4` | Path where the processed video will be saved. |
| `--config` | `./config/camera.json` | Path to the camera configuration file (defines lines and zones). |

### Example

```bash
python tracker.py --input data/my_video.mp4 --output results/tracked_video.mp4
```

Alternatively, you can run the script with the default arguments:

```bash
python tracker.py
```

## Authors

*   [Pietro Martano](https://github.com/shady-pete)
*   [Angelo Molinario](https://github.com/amolinario3)
*   [Massimiliano Ranauro](https://github.com/MassimilianoRanauro)
*   [Antonio Sessa](https://github.com/Antuke)
