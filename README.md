# Real Time Attention Monitoring System

A real time computer vision application that monitors eye movements and detects potential drowsiness using webcam input. Built with **Python, OpenCV, MediaPipe Face Mesh, and NumPy**, the system analyzes facial landmarks to estimate eye openness, count blinks, and classify individuals as Attentive or Drowsy.

The application also provides a live monitoring dashboard with attention statistics. It supports monitoring up to six faces simultaneously.

## Features

- **Real-Time Face Detection:** Detects and tracks facial landmarks using MediaPipe Face Mesh.
- **Multi-Face Monitoring:** Supports up to six faces simultaneously.
- **Eye Aspect Ratio (EAR):** Measures eye openness using facial landmark coordinates.
- **Drowsiness Detection:** Identifies prolonged eye closure using a consecutive-frame threshold.
- **Blink Detection:** Tracks individual blink counts based on eye closure and reopening.
- **Live Attention Dashboard:** Displays real-time monitoring results and session statistics.
- **Visual Alerts:** Highlights detected drowsiness with color-coded bounding boxes and alerts.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| OpenCV | Webcam access, image processing, and visualization |
| MediaPipe Face Mesh | Facial landmark detection |
| NumPy | Numerical operations and EAR calculation |


## Installation and Setup

### Prerequisites

- Python installed on your system.
- A working webcam.
- Windows, Linux, or macOS desktop environment.

### 1. Clone the Repository

```bash
git clone https://github.com/NoorFatima-14/Real-Time-Attention-Monitor.git

cd Real-Time-Attention-Monitor
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv attention
attention\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv attention
source attention/bin/activate
```

### 3. Install Dependencies

```bash
pip install opencv-python mediapipe numpy
```

### 4. Run the Application

```bash
python attention_monitor.py
```

The application will access your webcam and launch the **Classroom Attention Monitor** dashboard.

## Keyboard Controls

| Key | Function |
|---|---|
| `SPACE` | Pause / Resume monitoring |
| `S` | Capture dashboard screenshot |
| `Q` | Quit application |
| `ESC` | Exit application |

### Screenshot Storage

Screenshots are automatically saved in the `screenshots/` directory.

Example:

```text
screenshots/
├── attention_001.png
├── attention_002.png
└── attention_003.png
```

## Configuration

You can customize the monitoring behavior by modifying the configuration parameters in `attention_monitor.py`.

```python
EAR_THRESH = 0.22
EAR_CONSEC = 15

DISPLAY_W = 1000
DISPLAY_H = 620
```

Adjusting the EAR threshold and consecutive-frame requirement changes the sensitivity of drowsiness detection.

## Project Structure

```text
Real-Time-Attention-Monitor/
│
├── attention_monitor.py
├── screenshots/
└── README.md
```

- `attention_monitor.py`: Main application containing face detection, EAR calculation, blink counting, drowsiness classification, and dashboard visualization.
- `screenshots/`: Automatically created directory for captured dashboard screenshots.
- `README.md`: Project documentation.

## Limitations

- Detection accuracy depends on lighting conditions, camera quality, and face visibility.
- Glasses, occlusions, and extreme head angles may affect landmark detection.
- The system uses a fixed EAR threshold rather than individual calibration.
- Face indices are assigned during frame processing and do not represent persistent identity tracking.

## Future Improvements

- Individual EAR calibration.
- Head-pose estimation and gaze tracking.
- More robust face tracking across frames.
- Session report generation and data export.
- Configurable detection thresholds.

## Author

**Noor Fatima**

[GitHub Profile](https://github.com/NoorFatima-14)

[Project Repository](https://github.com/NoorFatima-14/Real-Time-Attention-Monitor)
