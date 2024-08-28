
# Palm Detection and Distance Estimation

## Description

This project estimates the distance of a hand palm from a webcam in real-time. The project involves calibrating a camera using a chessboard pattern and then using OpenCV and MediaPipe for palm detection. The distance is calculated using the triangular similarity method, and the results are displayed live on the screen.

## Features

- Real-time palm detection and distance estimation.
- Camera calibration using a chessboard pattern.
- Saves and loads calibration data for accurate distance measurements.

## Installation

### Prerequisites

- Python 3.x
- OpenCV
- MediaPipe
- NumPy

### Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/shirin6150/palm_detection
    cd palm-estimation
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Step 1: Capture Chessboard Images

To calibrate the camera, you first need to capture around 60-80 images of a chessboard pattern.

1. **Run the `capture_image.py` script:**

    ```bash
    python capture_image.py
    ```

2. **Controls:**
    - Press `s` to capture and save an image.
    - Press `q` to quit the capture.

3. **Specify the directory path** where the captured images will be saved when prompted.

### Step 2: Calibrate the Camera

Use the captured chessboard images to calibrate the camera.

1. **Run the `calibration.py` script:**

    ```bash
    python calibration.py
    ```

2. **Specify the path** where the captured images are located.
3. **Specify the path** where the calibrated data will be saved (e.g., `MultiMatrix_2.npz`).

### Step 3: Palm Distance Estimation

After calibration, you can estimate the distance of a hand palm in real-time.

1. **Run the `palm_distance.py` script:**

    ```bash
    python palm_distance.py
    ```

2. The script will use the calibrated data and display the estimated distance of the palm live.

## Configuration

- **Image Directory:** When running `capture_image.py`, specify the directory where you want the images to be saved.
- **Calibration Data Path:** When running `calibration.py`, specify the paths for both input images and output calibration data.
- **Calibrated Data:** Ensure that `MultiMatrix_2.npz` is correctly loaded in the `palm_distance.py` script for accurate distance estimation.

## Files

- `capture_image.py`: Script to capture and save chessboard pattern images for camera calibration.
- `calibration.py`: Script to calibrate the camera using the captured images and save the calibration data.
- `palm_distance.py`: Script to detect the palm and estimate its distance from the camera in real-time.


- Project Link: [https://github.com/shirin6150/palm_detection](https://github.com/shirin6150/palm_detection)
