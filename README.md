# 👁️ VisionMate

> **An AI-powered assistive computer vision and voice navigation assistant designed to empower visually impaired and low-vision individuals with real-time spatial awareness, obstacle warning, and hands-free object retrieval.**

---

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/Computer%20Vision-YOLO11%20%2F%20YOLO26-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![Tracking](https://img.shields.io/badge/Tracking-ByteTrack%20%2B%20Recovery-orange.svg)](#multi-object-tracking--temporal-stabilization)
[![TTS / STT](https://img.shields.io/badge/Voice-SAPI5%20%2B%20Google%20STT-purple.svg)](#voice-assistant--speech-interaction)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#prerequisites)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Tech Stack](#-tech-stack)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Quick Start](#-quick-start)
- [How to Interact](#-how-to-interact)
  - [Voice Commands](#voice-commands)
  - [Keyboard Shortcuts](#keyboard-shortcuts)
  - [Terminal Fallback](#terminal-fallback)
- [Under the Hood](#-under-the-hood)
  - [Multi-Object Tracking & Custom Recovery](#1-multi-object-tracking--custom-recovery)
  - [Temporal Detection Stabilizer](#2-temporal-detection-stabilizer)
  - [Spatial & Proximity Engine](#3-spatial--proximity-engine)
  - [Priority Speech Engine](#4-priority-speech-engine)
- [Model Evaluation](#-model-evaluation)
  - [Confusion Matrices](#confusion-matrices)
  - [Scoring Formula](#scoring-formula)
  - [MSE and MAE](#mse-and-mae)
  - [Error Percentage](#error-percentage)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🌟 Overview

Navigating dynamic everyday environments poses significant challenges for individuals with visual impairments. Traditional navigation aids such as white canes detect physical obstacles at ground level but cannot identify *what* the object is, anticipate hazards at chest level, or help locate misplaced personal belongings.

**VisionMate** solves this by turning a camera and microphone into an intelligent perception companion:
1. **Detects and classifies** common objects, furniture, people, and navigation obstacles in real time.
2. **Estimates 2D spatial positioning** (e.g., *upper left*, *directly in front*, *far right*) and relative proximity (*far*, *medium*, *close*, *very close*).
3. **Issues safety obstacle alerts** when navigation hazards pose an immediate collision risk.
4. **Listens and responds to verbal questions** (e.g., *"Where is my bottle?"*, *"Find my phone"*) even if the item was briefly hidden or moved out of the frame.
5. **Ensures smooth continuity** using a custom track recovery manager and temporal stabilization to avoid track-dropping and label flicker common with head and camera movements.

---

## ✨ Key Features

- **⚡ Real-Time Object Detection**: Powered by lightweight YOLO nano architectures (`yolo11n.pt` / `yolo26n.pt`) running at fluid video framerates on consumer hardware.
- **🙂 Live Face Detection**: OpenCV's bundled Haar cascade marks visible frontal faces in the camera HUD. This locates faces only; it does not identify or recognize people.
- **🎯 Occlusion-Resistant Track Recovery**: A specialized motion- and trajectory-aware recovery mechanism restores dropped ByteTrack IDs during sudden camera rotations and brief occlusions without the overhead of heavy neural Re-ID models.
- **🛡️ Flicker-Free Detection Stabilization**: Dual-layer temporal hysteresis holds missing tracks across brief gaps (up to 8 frames) and dampens transient class switches (requiring 3 consecutive consistent frames).
- **🧭 2D Spatial Localization & Proximity Estimation**: Maps detections into a spatial 3x3 grid with fine-grained verbal directions (*"slightly right"*, *"upper center"*) and relative bounding-box area proximity scores (0.0 to 1.0).
- **⚠️ Active Obstacle & Collision Alerts**: Flags hazardous objects (*chairs, tables, people, vehicles, backpacks*) and triggers prioritized auditory warnings when close.
- **🎙️ Hands-Free Natural Language Voice Assistant**: Listens continuously via microphone energy thresholding, parses natural speech queries with extensive alias resolution (e.g., `phone` / `mobile` -> `cell phone`), and speaks back clear answers.
- **🧠 30-Frame Query Memory**: Clusters spatial observations over time so the assistant accurately locates items even if they momentarily step out of view.
- **🔊 Dual-Tier Priority Speech Engine**: High-priority assistant responses preempt low-priority ambient narration. Built with multi-threaded COM isolation to prevent audio engine hangs.
- **🔄 Robust Hardware Probing & Fallbacks**: Automatically probes camera indices (0 to 3) and gracefully falls back to interactive terminal input if the microphone is unavailable.

---

## 🏗️ System Architecture

VisionMate uses a multi-threaded, non-blocking pipeline ensuring that computer vision processing, audio capture, and speech synthesis operate concurrently without dropping video frames.

```
+-----------------------------------------------------------------------------------+
|                                 VISIONMATE SYSTEM                                 |
+-----------------------------------------------------------------------------------+
                                          |
    +-------------------------------------+-----------------------------------+
    |                                     |                                   |
    v                                     v                                   v
[ Main Video Loop ]             [ Voice Listener Thread ]           [ Speech Announcer ]
  - Camera Capture (cv2)          - sounddevice (RMS > 300)           - pyttsx3 / SAPI5
  - YOLO11n / YOLO26n Inference   - SpeechRecognition (Google STT)    - Priority Queue:
  - ByteTrack Tracker             - Natural Language Parser             * P0: Voice Answers
  - Track Recovery Engine         - Command Queue Push                  * P1: Ambient Chatter
  - Temporal Stabilizer                   |                           - Non-blocking Worker
  - Spatial & Proximity Engine            |                                   ^
  - Obstacle Collision Monitor            v                                   |
  - Visual HUD & Bounding Boxes  ===> [ Assistant Controller ] ---------------+
  - Query Memory Snapshot             - 30-Frame Memory Cluster
                                      - Spatially-Aware Response Synthesis
```

---

## 📁 Repository Structure

```
VisionMate/
├── app.py                         # Application entrypoint & multi-threaded orchestration
├── requirements.txt               # Python package dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore rules
├── yolo11n.pt                     # YOLO11 nano pre-trained weights
├── yolo26n.pt                     # YOLO26 nano pre-trained weights
│
├── detection/                     # Vision & Tracking Subsystem
│   ├── __init__.py                # Package exports
│   ├── detector.py                # YOLO ObjectDetector inference wrapper
│   ├── face_detector.py           # OpenCV Haar-cascade face bounding boxes (no identity recognition)
│   ├── tracker.py                 # ByteTrack wrapper integration
│   ├── recovery.py                # TrackRecoveryManager (velocity & IoU trajectory matching)
│   └── stability.py               # TemporalDetectionStabilizer (hysteresis & smoothing)
│
├── spatial/                       # Spatial & Navigation Subsystem
│   ├── __init__.py                # Package exports
│   ├── position.py                # ObjectPositionAnalyzer (2D directional grid)
│   ├── distance.py                # DistanceEstimator (relative bounding-box area proximity)
│   └── obstacle.py                # ObstacleAnalyzer (navigation hazards & proximity alerts)
│
├── assistant/                     # Voice & Natural Language Subsystem
│   ├── __init__.py                # Package exports
│   ├── controller.py              # AssistantController (query resolution & orchestration)
│   ├── commands.py                # ObjectCommandParser (intents & alias mapping)
│   ├── query_memory.py            # ObjectQueryMemory (temporal object clustering)
│   ├── responses.py               # ObjectResponseGenerator (natural language synthesis)
│   └── speech.py                  # SpeechAnnouncer (threaded priority TTS with pyttsx3)
│
├── models/                        # Configuration & Weights
│   └── visionmate_bytetrack.yaml  # Tuned ByteTrack association parameters
│
└── datasets/                      # Directory for evaluation datasets and benchmarks
```

---

## 💻 Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Language** | Python 3.8+ / 3.10+ | Core application runtime |
| **Object Detection** | [Ultralytics YOLO](https://docs.ultralytics.com/) | Real-time edge inference on 80 COCO classes (`yolo11n` / `yolo26n`) |
| **Face Detection** | OpenCV Haar cascade | Frontal face localization and HUD boxes; no face identification |
| **Vision & GUI** | OpenCV (`cv2`) | Video stream capture, HUD rendering, high-contrast bounding boxes |
| **Multi-Object Tracking** | ByteTrack | Two-stage association algorithm for robust trajectory tracking |
| **Numeric Operations** | NumPy | Bounding box spatial computations, RMS audio energy filtering |
| **Speech-to-Text (STT)**| `SpeechRecognition` + `sounddevice` | Live microphone capture with energy thresholding & STT API |
| **Text-to-Speech (TTS)**| `pyttsx3` + `pywin32` | Offline, zero-latency speech synthesis using Windows SAPI5 / system TTS |
| **Concurrency** | `threading`, `queue.PriorityQueue` | Non-blocking camera, listener, and announcer threads |

---

## ⚙️ Prerequisites & Installation

### 1. Prerequisites
- **Operating System**: Windows 10/11 (fully supported with native SAPI5 TTS), Linux, or macOS.
- **Python**: Version 3.8 to 3.11 recommended.
- **Hardware**:
  - Webcam / USB camera.
  - Microphone for voice input.
  - Speakers or headphones for voice feedback.

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/VisionMate.git
cd VisionMate
```

### 3. Set Up a Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note on Windows Dependencies**: The `pywin32` package is included in `requirements.txt` to enable COM library initialization (`pythoncom`), preventing Windows SAPI5 audio deadlocks across threads.

---

## 🚀 Quick Start

Ensure your webcam and microphone are connected, then run:

```bash
python app.py
```

1. **Camera Initialization**: VisionMate automatically scans camera indices `0`, `1`, `2`, and `3` and connects to the first operational video device.
2. **Audio Setup**: The speech announcer greets you: *"VisionMate is ready."*
3. **Live HUD**: A camera window will open displaying detected objects, frontal-face boxes, proximity indicators, recovery flags, and system status. Face boxes are not identity labels.

---

## 🗣️ How to Interact

### Voice Commands
Speak naturally into your microphone at any time. VisionMate recognizes conversational phrasing and synonyms:

| Intent | Example Utterances | System Action |
|---|---|---|
| **Find Object** | *"Find my bottle"*<br>*"Where is my phone?"*<br>*"Locate my laptop"*<br>*"Can you see my keys?"* | Searches current scene and 30-frame query memory; responds with spatial position & distance. |
| **Synonym Aliases** | *"Where is my smartphone?"* -> `cell phone`<br>*"Find my mug"* -> `cup`<br>*"Where is the couch?"* -> `couch` / `chair` | Resolves common speech variants to standard object categories automatically. |

### Keyboard Shortcuts
When the video feed window is active:
- **`F`**: Toggle Fullscreen Mode on/off.
- **`Q`** or **`ESC`**: Gracefully terminate and stop all background threads.

### Terminal Fallback
If no microphone is detected or speech recognition is offline, VisionMate automatically provides a console prompt in your terminal:
```text
> Find my cup
```
Type any query and press `Enter` to receive immediate spoken and visual feedback.

---

## 🔬 Under the Hood

### 1. Multi-Object Tracking & Custom Recovery
Traditional tracking algorithms frequently reassign or drop tracking IDs when an object is momentarily occluded by a hand, passes behind furniture, or when the user turns their head.

VisionMate implements a **`TrackRecoveryManager`** (`detection/recovery.py`) that monitors lost tracks. When an unassigned detection appears, the recovery manager evaluates candidate tracks based on:
$$\text{Score} = w_{\text{dist}} \cdot S_{\text{dist}} + w_{\text{vel}} \cdot S_{\text{vel}} + w_{\text{area}} \cdot S_{\text{area}} + w_{\text{traj}} \cdot S_{\text{traj}}$$
- **Normalized Center Distance**: Penalizes sudden spatial leaps.
- **Velocity Extrapolation**: Predicts where the object should be based on previous movement vectors.
- **Bounding Box Area Ratio**: Verifies scale consistency.
- **Trajectory Cosine Similarity**: Compares historical motion direction against candidate displacement.

### 2. Temporal Detection Stabilizer
Transient detector noise can cause bounding boxes to flicker or labels to alternate rapidly (e.g., oscillating between `laptop` and `keyboard`).

The **`TemporalDetectionStabilizer`** (`detection/stability.py`) applies dual hysteresis:
- **Missing Frame Hold**: Retains dropped tracks for up to 8 frames, marking them as `is_stale`. If the object reappears within 8 frames, its identity is seamlessly preserved.
- **Class Label Smoothing**: Requires a new class classification to be observed over 3 consecutive frames before changing the confirmed label.

### 3. Spatial & Proximity Engine
- **Position Analyzer (`spatial/position.py`)**: Breaks the video frame into a 3x3 directional matrix:
  ```
  +-------------------+--------------------+--------------------+
  |    Upper Left     |    Upper Center    |    Upper Right     |
  +-------------------+--------------------+--------------------+
  |    Left           |    Center          |    Right           |
  +-------------------+--------------------+--------------------+
  |    Lower Left     |    Lower Center    |    Lower Right     |
  +-------------------+--------------------+--------------------+
  ```
- **Proximity Estimator (`spatial/distance.py`)**: Calculates the ratio of the object bounding box area relative to the total camera frame area, mapping it into:
  - `far` ($\text{ratio} < 0.05$)
  - `medium` ($0.05 \le \text{ratio} < 0.15$)
  - `close` ($0.15 \le \text{ratio} < 0.35$)
  - `very_close` ($\text{ratio} \ge 0.35$)
- **Obstacle Analyzer (`spatial/obstacle.py`)**: Continuously evaluates navigation hazards (e.g., *person, chair, table, couch, car, bus*). If a hazard enters `close` or `very_close` range, an urgent collision warning is dispatched.

### 4. Priority Speech Engine
To prevent speech overlap and ensure user safety:
- **Priority 0 (High Priority)**: Direct assistant answers to user queries and critical obstacle collision alerts. Interrupts any active low-priority speech.
- **Priority 1 (Normal Priority)**: Ambient narration of newly detected objects, rate-limited by configurable cooldown timers.
- **Thread Safety**: On Windows, every utterance runs within an initialized COM apartment (`pythoncom.CoInitialize()`), eliminating the SAPI5 concurrency crashes typical in multi-threaded Python apps.

---

## 📊 Model Evaluation

The YOLO models were evaluated on `datasets/coco_experiment.yaml`. The reported
values below are the baseline results before project fine-tuning.

| Model | Precision | Recall | F1-score | mAP@0.50 | mAP@0.50:0.95 | FPS |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n baseline | 64.56% | 62.34% | 63.43% | 69.49% | 54.87% | 23.3 |
| YOLO26n baseline | **84.00%*** | 58.00%* | 68.63%* | 72.00%* | 55.00%* | 42.7 |

### Confusion Matrices

For multiclass object detection, each matrix is organized by the actual COCO
class on the vertical axis and the predicted class on the horizontal axis.
The normalized matrices show the proportion of predictions for each actual
class. Darker diagonal values indicate better correct-class classification;
off-diagonal values indicate confusion between classes.

| Model | Confusion Matrix | Normalized Confusion Matrix |
|---|---|---|
| YOLO11n baseline | [![YOLO11n confusion matrix](runs/research/eval/yolo11n_baseline/confusion_matrix.png)](runs/research/eval/yolo11n_baseline/confusion_matrix.png) | [![YOLO11n normalized confusion matrix](runs/research/eval/yolo11n_baseline/confusion_matrix_normalized.png)](runs/research/eval/yolo11n_baseline/confusion_matrix_normalized.png) |
| YOLO26n baseline | [![YOLO26n confusion matrix](runs/research/eval/yolo26n_baseline/confusion_matrix.png)](runs/research/eval/yolo26n_baseline/confusion_matrix.png) | [![YOLO26n normalized confusion matrix](runs/research/eval/yolo26n_baseline/confusion_matrix_normalized.png)](runs/research/eval/yolo26n_baseline/confusion_matrix_normalized.png) |

Additional matrices from pilot and fine-tuned experiments are available in:

- [`runs/research/pilot_yolo11n/`](runs/research/pilot_yolo11n/)
- [`runs/research/pilot_yolo26n/`](runs/research/pilot_yolo26n/)
- [`runs/research/eval/yolo11n_finetuned/`](runs/research/eval/yolo11n_finetuned/)
- [`runs/research/eval/yolo26n_finetuned/`](runs/research/eval/yolo26n_finetuned/)

### Scoring Formula

The basic binary classification terms used in object-detection evaluation are:

- **TP**: Correct object detected.
- **FP**: Object/background falsely detected.
- **FN**: Real object missed by the detector.
- **TN**: Background correctly identified as background.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

$$
\text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}
{\text{Precision} + \text{Recall}}
$$

The overall model score is also reported using mean average precision:

$$
\text{mAP} = \frac{1}{N}\sum_{i=1}^{N} AP_i
$$

where $N$ is the number of evaluated COCO classes and $AP_i$ is the average
precision for class $i$. `mAP@0.50` uses an IoU threshold of 0.50, while
`mAP@0.50:0.95` averages IoU thresholds from 0.50 through 0.95.

### MSE and MAE

Mean Squared Error (MSE) and Mean Absolute Error (MAE) can be used when
VisionMate is evaluated with numerical prediction values and corresponding
ground-truth values:

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

Here, $y_i$ is the expected value and $\hat{y}_i$ is the model-predicted value.
For YOLO object detection, these formulas are mainly appropriate for
regression-style outputs such as bounding-box coordinates or distance estimates.
MSE penalizes large errors more strongly because the squared differences give
greater weight to outliers.

Using the baseline-to-fine-tuned changes in mAP@0.50, mAP@0.50:0.95,
precision, and recall as four numerical observations:

| Model | MAE | MSE |
|---|---:|---:|
| YOLO11n | 0.0111 (1.11 percentage points) | 0.000210 (2.10 percentage-points squared) |
| YOLO26n | 0.0347 (3.47 percentage points) | 0.002410 (24.10 percentage-points squared) |

These MSE/MAE values summarize the aggregate metric changes reported by the
fine-tuning experiments. They are **not YOLO bounding-box or per-sample
detection errors**, because the raw continuous target/prediction arrays are
not stored in the generated result summary.

### Error Percentage

Evaluation errors can be derived from the complementary score percentages:

$$
\text{Precision Error} = (1 - \text{Precision}) \times 100
$$

$$
\text{Recall/Missed-Detection Error} = (1 - \text{Recall}) \times 100
$$

$$
\text{F1 Error} = (1 - \text{F1-score}) \times 100
$$

Using the baseline values:

| Model | False-Positive/Precision Error | Missed-Detection/Recall Error | F1 Error |
|---|---:|---:|---:|
| YOLO11n baseline | 35.44% | 37.66% | 36.57% |
| YOLO26n baseline | **16.00%** | 42.00%* | 29.00%* |

These are **metric-derived error complements**, not literal counts of all
incorrect predictions. Total TP, FP, FN, and TN counts may be obtained from
raw YOLO prediction records and an externally produced confusion-matrix CSV.

---

## 🔧 Configuration

### ByteTrack Tuning (`models/visionmate_bytetrack.yaml`)
You can fine-tune association thresholds directly in the YAML config:
```yaml
tracker_type: bytetrack
track_high_thresh: 0.35    # High confidence detection threshold
track_low_thresh: 0.1      # Low confidence detection threshold
new_track_thresh: 0.4      # Threshold to initiate a new track
track_buffer: 30           # Number of frames to retain lost tracks
match_thresh: 0.8          # IoU matching threshold
```

### Switching YOLO Models
VisionMate ships with both `yolo11n.pt` and `yolo26n.pt`. In `app.py`, you can select your preferred model:
```python
# app.py
detector = ObjectDetector(model_path="yolo11n.pt", confidence=0.35)
```

---

## ❓ Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| **Camera not opening** | Device busy or incorrect index | Close other apps using the webcam. VisionMate auto-checks indices 0-3; verify your webcam shows up in Windows Device Manager. |
| **No audio output** | Missing SAPI5 voices or audio device mute | Verify system audio output is enabled. On Linux, ensure `espeak` and `alsa-utils` are installed (`sudo apt-get install espeak`). |
| **Microphone not transcribing** | Ambient noise or low input level | Speak clearly into the microphone. Check Windows Microphone Privacy Settings to ensure Python has access. |
| **High CPU usage** | Frame processing load | Use the default `yolo11n.pt` nano model or reduce camera resolution in `app.py`. |

---

## 🗺️ Roadmap

- [ ] **Depth Sensor Integration**: Support for Intel RealSense and stereoscopic depth cameras for millimeter-accurate 3D spatial mapping.
- [ ] **OCR & Document Reading**: Integrate text reading models for signage, product labels, and medication bottles.
- [ ] **Haptic Feedback**: Support for wearable BLE haptic vibration bands to indicate obstacle direction without sound.
- [ ] **Edge Device Porting**: Optimized deployment pipelines for Raspberry Pi 5 and NVIDIA Jetson Orin Nano.
- [ ] **Offline Speech Recognition**: Local on-device Whisper (whisper.cpp) integration for complete offline privacy.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — free for educational, personal, and commercial assistive technology applications.

---

<p align="center">
  <b>Developed with ❤️ to make the world more accessible for everyone.</b>
</p>