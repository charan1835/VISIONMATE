# 📋 VisionMate Capstone Project Planner

A structured engineering roadmap and work breakdown structure for the **VisionMate Capstone Team**: **Charan**, **Shreya**, and **Chiru**.

---

## 👥 1. Team Roster & Ownership Overview

```
+----------------------------------------------------------------------------------------------------+
|                                    VISIONMATE CAPSTONE TEAM                                        |
+------------------------------------+-----------------------------------+---------------------------+
| CHARAN (Member 1 / You)            | SHREYA (Member 2)                 | CHIRU (Member 3)          |
| Interaction & Recognition Lead     | Data Engineering & Pipeline Lead  | CV Detection & Tuning Lead|
+------------------------------------+-----------------------------------+---------------------------+
| • Object Recognition & Recovery    | • Assistive Data Collection       | • YOLO Object Detection   |
| • 2D Spatial & Proximity Engine    | • Annotation & Dataset Labeling   | • Custom Model Fine-Tuning|
| • Speech-to-Text (STT) Commands    | • Data Augmentation Pipeline      | • Model Export & Quantize |
| • Text-to-Speech (TTS) Narration   | • Preprocessing & Split Scripts   | • ByteTrack Integration   |
| • Assistant Controller & Memory    | • Data QA & Version Control       | • Latency & mAP Benchmarks|
+------------------------------------+-----------------------------------+---------------------------+
```

---

## 🎯 2. Detailed Member Responsibilities

### 👤 **Charan — Object Recognition & Speech Interaction Lead**
* **Primary Scope**: What the visually impaired user experiences — seeing, continuous object identification, spatial understanding, and hands-free voice dialogue.
* **Key Tasks**:
  1. **Object Recognition & Identity Recovery**:
     - Lead `TrackRecoveryManager` (`detection/recovery.py`): continuous ID recovery across sudden camera rotations, arm movements, and occlusions using motion vectors and scale ratios.
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

### 👤 **Shreya — Data Engineering & Dataset Pipeline Lead**
* **Primary Scope**: Everything that powers and fuels the AI models — data gathering, labeling, cleaning, augmentations, and automated pipelines.
* **Key Tasks**:
  1. **Assistive Scenario Data Collection**:
     - Source and compile indoor/outdoor assistive datasets (e.g., chairs, tables, doors, curbs, steps, white canes, pedestrians, dropped personal items like keys/phones).
     - Record custom video clips under realistic conditions: handheld and body-worn camera movement, low light, motion blur, and partial occlusions.
  2. **Data Annotation & Standardization**:
     - Define standardized labeling conventions in YOLO format (`<class_id> <x_center> <y_center> <width> <height>`).
     - Standardize class naming ontology across public datasets and custom recordings.
     - Conduct annotation quality control (fixing tight bounding boxes, handling missing tags).
  3. **Automated Data Pipeline & Augmentations**:
     - Build automated Python ingestion scripts (`datasets/scripts/prepare_data.py`).
     - Implement assistive-specific data augmentations (simulated camera shake, illumination shifts, partial synthetic occlusions) using Albumentations / OpenCV.
     - Generate verified train/val/test splits (70% train, 20% val, 10% test) ensuring no video-frame leakage across splits.
  4. **Dataset Documentation & Reporting**:
     - Document dataset statistics, class balance histograms, and annotation guidelines for the capstone final report.

---

### 👤 **Chiru — Computer Vision Detection & Model Fine-Tuning Lead**
* **Primary Scope**: The vision core — model selection, transfer learning, fine-tuning, tracking integration, and edge inference optimization.
* **Key Tasks**:
  1. **Model Selection & Baseline Benchmarking**:
     - Benchmark pre-trained YOLO architectures (`yolo11n`, `yolo11s`, `yolo26n`) on baseline metrics (FPS, inference latency on CPU vs. GPU, mAP@0.5, mAP@0.5:0.95).
  2. **Model Fine-Tuning & Transfer Learning**:
     - Set up the PyTorch / Ultralytics fine-tuning pipeline using Shreya's prepared dataset.
     - Optimize training hyperparameters: learning rate schedulers, weight decay, anchor optimization, loss weights (box loss, class loss, DFL loss).
     - Mitigate overfitting on smaller custom classes using frozen backbone layers and targeted fine-tuning.
  3. **Tracking Architecture Integration**:
     - Benchmark and configure ByteTrack parameters in `models/visionmate_bytetrack.yaml` (`track_high_thresh`, `track_low_thresh`, `match_thresh`, `track_buffer`).
     - Ensure tracker output data contracts match Charan's recovery manager schema (`track_id`, `class_name`, `bbox`, `confidence`).
  4. **Model Optimization & Edge Export**:
     - Export trained models to ONNX / OpenVINO / TensorRT for maximum runtime FPS.
     - Evaluate INT8 / FP16 post-training quantization trade-offs for edge/laptop execution.

---

## 🔄 3. Team Interface Contracts & Hand-offs

```
[ SHREYA: Data Pipeline ]
       │
       │  Structured YOLO Dataset (images + labels + splits)
       ▼
[ CHIRU: Detection & Fine-Tuning ]
       │
       │  Trained Weights (.pt / .onnx) + Tuned Tracker Config (.yaml)
       ▼
[ CHARAN: Recognition & Speech Integration ]
       │
       ▼
[ End-to-End System (app.py) ]
```

| Producer | Consumer | Hand-off Artifact | Specification |
|---|---|---|---|
| **Shreya** | **Chiru** | Curated Dataset | YOLO format (`images/`, `labels/`, `data.yaml`) with balanced splits. |
| **Chiru** | **Charan** | Fine-tuned weights & ByteTrack config | `best.pt` / `best.onnx` compatible with `ObjectDetector` + `visionmate_bytetrack.yaml`. |
| **Charan** | **All** | Main App Harness | Standalone runtime (`app.py`) for live validation, HUD testing, and audio feedback. |

---

## 📅 4. 8-Week Milestone Schedule

### **Phase 1: Foundation & Baseline (Weeks 1–2)**
- **Charan**: Benchmark existing `app.py` end-to-end pipeline; stress-test STT/TTS on different microphones.
- **Shreya**: Finalize class ontology (15–20 high-priority assistive classes); collect initial raw video clips and public datasets.
- **Chiru**: Set up training environment (CUDA, PyTorch, Ultralytics); benchmark baseline `yolo11n.pt` and `yolo26n.pt` on local hardware.
> **Milestone 1**: Class list frozen, baseline inference latency measured, repository workflow aligned.

---

### **Phase 2: Data Pipeline & Perception Modules (Weeks 3–4)**
- **Charan**: Implement `TrackRecoveryManager` scoring algorithms and 30-frame temporal query memory in `assistant/query_memory.py`.
- **Shreya**: Complete annotation of initial dataset (1,500–3,000 labeled instances); build automated data augmentation script.
- **Chiru**: Run first fine-tuning experiments; tune ByteTrack parameters (`track_buffer`, `match_thresh`) for indoor movement.
> **Milestone 2**: Dataset v1.0 delivered to Chiru; first fine-tuned weights handed to Charan; recovery & memory operational.

---

### **Phase 3: Fine-Tuning Optimization & Assistant Integration (Weeks 5–6)**
- **Charan**: Connect Chiru's fine-tuned model into the voice assistant; expand natural language alias mapping and collision alert thresholds.
- **Shreya**: Analyze false-positives/false-negatives from Chiru's model; record targeted edge-case data (low light, motion blur); deliver Dataset v2.0.
- **Chiru**: Retrain on Dataset v2.0 with hyperparameter tuning; export model to optimized ONNX/OpenVINO format; benchmark mAP and FPS.
> **Milestone 3**: Fine-tuned model running smoothly inside the voice-guided application at >25 FPS.

---

### **Phase 4: Full System Integration, Testing & Capstone Presentation (Weeks 7–8)**
- **Charan**: Conduct simulated low-vision navigation tests; refine HUD overlay and audio error handling.
- **Shreya**: Generate dataset distribution graphs, data documentation, and demo video footage.
- **Chiru**: Compile performance benchmark tables (latency, memory footprint, mAP curves) and model architecture slides.
- **All (Charan, Shreya, Chiru)**: Final report writing, presentation rehearsals, and live demo verification.
> **Final Milestone**: Live project demonstration, final capstone report, and code repository submission.

---

## 📊 5. RACI Matrix

| Task / Deliverable | Charan | Shreya | Chiru |
|---|:---:|:---:|:---:|
| **Assistive Data Collection & Cleansing** | I | **A / R** | C |
| **Annotation Guidelines & Split Pipeline** | C | **A / R** | C |
| **YOLO Model Fine-Tuning & Hyperparameters** | C | C | **A / R** |
| **Model Quantization & Inference Speed** | C | I | **A / R** |
| **ByteTrack Parameter Tuning** | C | I | **A / R** |
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
  - [ ] Curated assistive navigation dataset (labeled in YOLO format)
  - [ ] Automated data ingestion and augmentation pipeline script
  - [ ] Dataset statistics report (class distributions, splits, sample gallery)

- [ ] Chiru's Deliverables:
  - [ ] Fine-tuned YOLO model weights (`best.pt` / `best.onnx`)
  - [ ] Quantitative benchmarking suite (mAP@0.5, precision, recall, latency)
  - [ ] Tuned ByteTrack configuration file
```
