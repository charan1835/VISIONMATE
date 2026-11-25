# 📋 VisionMate Capstone Project Planner

A structured engineering roadmap and work breakdown structure for the **VisionMate Capstone Team**: **Charan**, **Shreya**, and **Chiru**.

---

## 👥 1. Team Roster & Ownership Overview

```
+----------------------------------------------------------------------------------------------------+
|                                    VISIONMATE CAPSTONE TEAM                                        |
+------------------------------------+-----------------------------------+---------------------------+
| CHARAN (Member 1 / You)            | SHREYA (Member 2)                 | CHIRU (Member 3)          |
| Interaction & Recognition Lead     | Data Collection & Fine-Tuning Lead| CV Detection & Pipeline   |
+------------------------------------+-----------------------------------+---------------------------+
| • Object Recognition & Recovery    | • Assistive Data Collection       | • Object Detection Core   |
| • 2D Spatial & Proximity Engine    | • Annotation Guidelines & Labeling| • Automated Data Pipeline |
| • Speech-to-Text (STT) Commands    | • Transfer Learning & Fine-Tuning | • Preprocessing & Augment |
| • Text-to-Speech (TTS) Narration   | • Hyperparameter Optimization     | • Train / Val / Test Split|
| • Assistant Controller & Memory    | • Validation Loss & mAP Curves    | • ByteTrack Configuration |
| • System Orchestration (app.py)    | • Hard-Example Mining             | • Quantization & Export   |
+------------------------------------+-----------------------------------+---------------------------+
```

---

## 🎯 2. Detailed Member Responsibilities

### 👤 **Charan — Object Recognition & Speech Interaction Lead**
* **Primary Scope**: What the visually impaired user experiences — seeing, continuous object identification, spatial understanding, and hands-free voice dialogue.
* **Key Tasks**:
  1. **Object Recognition & Identity Recovery**:
     - Maintain `TrackRecoveryManager` (`detection/recovery.py`): continuous ID recovery across sudden camera rotations, arm movements, and occlusions using motion vectors, velocity, and scale ratios.
     - Implement `TemporalDetectionStabilizer` (`detection/stability.py`): 8-frame missing track hold and 3-frame class label smoothing to eliminate visual flicker.
  2. **Spatial Perception & Proximity Engine**:
     - Fine-tune the 3x3 2D spatial grid (`spatial/position.py`) for natural verbal directions (*"upper left"*, *"center"*).
     - Calculate relative bounding-box area proximity scores (`spatial/distance.py`) and classify distances (`far`, `medium`, `close`, `very_close`).
     - Configure the obstacle hazard warning engine (`spatial/obstacle.py`) with categorized warning severities.
  3. **Voice Interaction & Speech Engine**:
     - Microphone audio streaming with RMS energy gating using `sounddevice` and `SpeechRecognition`.
     - Natural language command parser with synonym alias mapping (`assistant/commands.py`).
     - 30-frame temporal query memory clustering (`assistant/query_memory.py`) to locate items recently seen.
     - Multi-threaded, prioritized SAPI5/system TTS in `assistant/speech.py`.
  4. **System Orchestration & HUD**:
     - Maintain the main runtime loop in `app.py`, multi-threading controls, and visual OpenCV HUD display.

---

### 👤 **Shreya — Data Collection & Model Fine-Tuning Lead**
* **Primary Scope**: Sourcing real-world assistive visual data and training the deep learning model to accurately identify objects in low-vision scenarios.
* **Key Tasks**:
  1. **Assistive Scenario Data Collection**:
     - Source and compile indoor/outdoor assistive datasets (chairs, tables, doors, curbs, steps, white canes, pedestrians, dropped personal items like keys/phones).
     - Record custom video clips under realistic conditions: handheld and body-worn camera movement, low light, motion blur, and partial occlusions.
  2. **Dataset Annotation & Quality Control**:
     - Establish labeling guidelines in standard YOLO format (`<class_id> <x_center> <y_center> <width> <height>`).
     - Label custom assistive video frames and perform QA to ensure tight, accurate bounding boxes.
  3. **Model Fine-Tuning & Transfer Learning**:
     - Fine-tune YOLO models (`yolo11n`, `yolo26n`) using PyTorch and Ultralytics on the curated dataset provided by Chiru's pipeline.
     - Optimize training hyperparameters: learning-rate schedule, weight decay, and loss balance (box, class, and DFL where supported by the selected model).
     - Prevent overfitting with frozen backbone layers and targeted fine-tuning on custom assistive classes.
  4. **Validation & Training Performance Tracking**:
     - Monitor training/validation loss curves, mAP@0.5, and mAP@0.5:0.95 across training epochs.
     - Perform hard-example mining: identify misclassified instances to guide targeted data retraining.

---

### 👤 **Chiru — Computer Vision Detection, Data Pipeline & Optimization Lead**
* **Primary Scope**: Core detection architectures, automated data transformation pipelines, tracking integration, and edge model optimization.
* **Key Tasks**:
  1. **Object Detection Core & Baseline Benchmarking**:
     - Set up and benchmark baseline YOLO architectures (`yolo11n`, `yolo26n`) on inference latency (CPU vs. GPU) and memory usage.
  2. **Automated Data Pipeline & Augmentations**:
     - Build automated Python ingestion scripts (`datasets/scripts/prepare_data.py`) to process Shreya's raw collected datasets.
     - Implement assistive-specific data augmentations (simulated camera shake, illumination shifts, motion blur, partial synthetic occlusions) using Albumentations or OpenCV.
     - Generate verified train/val/test splits (70% train, 20% val, 10% test) by source video or capture session, preventing sequential-frame leakage across partitions.
  3. **Tracking Architecture Integration**:
     - Benchmark and configure ByteTrack parameters in `models/visionmate_bytetrack.yaml` (`track_high_thresh`, `track_low_thresh`, `match_thresh`, `track_buffer`).
     - Coordinate with Charan and add an adapter, if needed, so tracker outputs are normalized to the recovery manager schema (`track_id`, `class_name`, `bbox`, `confidence`).
  4. **Model Optimization & Edge Export**:
     - Take Shreya's fine-tuned model weights and export them to ONNX / OpenVINO / TensorRT for maximum inference FPS.
     - Evaluate INT8 / FP16 post-training quantization trade-offs for smooth execution on laptops or portable devices.

---

## 🔄 3. Team Interface Contracts & Hand-offs

```
[ SHREYA: Raw Data Collection & Annotation ]
                   │
                   │ Raw Images & YOLO Labels
                   ▼
[ CHIRU: Data Pipeline, Augmentations & Splits ]
                   │
                   │ Partitioned Dataset (train/val/test + data.yaml)
                   ▼
[ SHREYA: Transfer Learning & Model Fine-Tuning ]
                   │
                   │ Trained Model Weights (best.pt)
                   ▼
[ CHIRU: Quantization, Edge Export & ByteTrack Tuning ]
                   │
                   │ Optimized Weights (best.onnx / best.pt) + visionmate_bytetrack.yaml
                   ▼
[ CHARAN: Object Recognition, Recovery & Speech Assistant ]
                   │
                   ▼
[ Complete Integrated Application (app.py) ]
```

| Step | Producer | Consumer | Hand-off Artifact | Specification |
|:---:|---|---|---|---|
| **1** | **Shreya** | **Chiru** | Raw Collected & Labeled Data | Annotated images and YOLO `.txt` labels. |
| **2** | **Chiru** | **Shreya** | Partitioned Dataset & Pipeline | Clean augmented dataset (`train/`, `val/`, `test/`) with `data.yaml`. |
| **3** | **Shreya** | **Chiru** | Fine-Tuned Model Weights | `best.pt` with validation loss logs and mAP evaluations. |
| **4** | **Chiru** | **Charan** | Optimized Model & Tracker Config | `best.onnx` / `best.pt` + tuned `visionmate_bytetrack.yaml`. |
| **5** | **Charan** | **All** | Main App Integration | Executable `app.py` for live testing, HUD validation, and voice interaction. |

---

## 📅 4. 8-Week Milestone Schedule

### **Phase 1: Foundation & Baseline (Weeks 1–2)**
- **Charan**: Benchmark existing `app.py` end-to-end pipeline; stress-test STT/TTS on different microphones.
- **Shreya**: Define class ontology (15–20 high-priority assistive classes); collect initial raw video clips and public assistive datasets.
- **Chiru**: Build the data pipeline skeleton (`datasets/scripts/prepare_data.py`); benchmark baseline `yolo11n.pt` and `yolo26n.pt` on local hardware.
> **Milestone 1**: Class list frozen, baseline inference latency measured, data pipeline ingestion tested.

---

### **Phase 2: Data Processing & Initial Fine-Tuning (Weeks 3–4)**
- **Charan**: Implement `TrackRecoveryManager` scoring algorithms and 30-frame temporal query memory in `assistant/query_memory.py`.
- **Chiru**: Run automated pipeline on Shreya's initial dataset; generate augmented train/val/test splits; configure ByteTrack parameters.
- **Shreya**: Execute initial fine-tuning runs using transfer learning on Chiru's dataset splits; track validation loss and baseline mAP scores.
> **Milestone 2**: Dataset v1.0 pipeline operational; first fine-tuned model produced by Shreya; recovery & query memory working in `app.py`.

---

### **Phase 3: Fine-Tuning Optimization & Assistant Integration (Weeks 5–6)**
- **Shreya**: Retrain on targeted hard examples (low light, motion blur); perform hyperparameter sweeps; deliver finalized `best.pt`.
- **Chiru**: Export Shreya's model to ONNX / OpenVINO; evaluate quantization (FP16/INT8); benchmark FPS and latency gains.
- **Charan**: Connect optimized model into the voice assistant; expand natural language alias mapping and collision alert thresholds.
> **Milestone 3**: Fine-tuned and optimized model running smoothly inside the voice-guided application at a measured target of >25 FPS on the agreed hardware, input resolution, and deployment backend.

---

### **Phase 4: Full System Integration, Testing & Capstone Presentation (Weeks 7–8)**
- **Charan**: Conduct simulated low-vision navigation tests; refine HUD overlay and audio error handling.
- **Shreya**: Prepare training progression curves (loss, mAP@0.5, precision-recall) and error analysis for report/presentation.
- **Chiru**: Compile data pipeline architecture diagram, augmentation ablation studies, and edge latency benchmarks.
- **All (Charan, Shreya, Chiru)**: Final report writing, presentation rehearsals, and live demo verification.
> **Final Milestone**: Live project demonstration, final capstone report, and code repository submission.

---

## 📊 5. RACI Matrix

| Task / Deliverable | Charan | Shreya | Chiru |
|---|:---:|:---:|:---:|
| **Assistive Data Collection & Raw Footage** | I | **A / R** | C |
| **Annotation Guidelines & Bounding Box QA** | C | **A / R** | C |
| **Automated Ingestion & Augmentation Pipeline** | I | C | **A / R** |
| **Train / Val / Test Dataset Splits** | I | C | **A / R** |
| **YOLO Model Fine-Tuning & Hyperparameters** | C | **A / R** | C |
| **Training Loss & mAP Validation** | C | **A / R** | C |
| **Model Quantization & Inference Speed (ONNX)** | C | C | **A / R** |
| **ByteTrack Parameter Configuration** | C | I | **A / R** |
| **Track Recovery & Temporal Stabilization** | **A / R** | I | C |
| **Spatial 2D Position & Proximity Engine** | **A / R** | I | I |
| **Voice Command Parsing & Query Memory** | **A / R** | I | I |
| **Priority Speech Narration Engine (TTS)** | **A / R** | I | I |
| **Main App Orchestration (`app.py`)** | **A / R** | C | C |
| **Final Capstone Report & Presentation** | **R** | **R** | **R** |

*(**R**: Responsible, **A**: Accountable, **C**: Consulted, **I**: Informed)*

---

## 🎯 6. Member Deliverables Checklist

```markdown
- [ ] Charan's Deliverables:
  - [ ] Multi-threaded speech pipeline with priority queue
  - [ ] Track recovery and temporal hysteresis algorithms with continuity metrics
  - [ ] Conversational query engine with 30-frame temporal memory
  - [ ] Clean, executable `app.py` with camera auto-probing and HUD

- [ ] Shreya's Deliverables:
  - [ ] Curated and annotated assistive navigation dataset
  - [ ] Fine-tuned YOLO model weights (`best.pt`)
  - [ ] Training loss, mAP validation curves, and error analysis report

- [ ] Chiru's Deliverables:
  - [ ] Automated data pipeline & augmentation script (`prepare_data.py`)
  - [ ] Verified train/val/test dataset partitions with balance statistics
  - [ ] Quantized/optimized model exports (`best.onnx`) and latency benchmarks
  - [ ] Tuned ByteTrack configuration file
```
