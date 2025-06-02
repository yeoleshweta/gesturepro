# GesturePro

GesturePro is a **real-time** hand-gesture recognition pipeline designed to translate live hand movements into actionable commands or text output with minimal latency. Built with UX and accessibility at its core, this project demonstrates how machine learning can bridge communication gaps—whether for sign language translation, hands-free controls in AR/VR, or assistive interfaces. Researchers, designers, and developers can collaborate on every stage: data collection, preprocessing, model training, and real-time deployment.

---
## Table of Contents

1. [Project Overview](#project-overview)  
2. [Human-Centered Approach](#human-centered-approach)  
   - [User Profiles & Personas](#user-profiles--personas)  
   - [Empathy & Accessibility](#empathy--accessibility)  
   - [Iterative Research & Testing](#iterative-research--testing)  
3. [Features](#features)  
4. [Repository Structure](#repository-structure)  
5. [Installation & Setup](#installation--setup)  
6. [Data Collection & Preprocessing](#data-collection--preprocessing)  
   - [Dataset Schema](#dataset-schema)  
   - [Labeling Guidelines](#labeling-guidelines)  
   - [Preprocessing Pipeline](#preprocessing-pipeline)  
7. [Model Training](#model-training)  
   - [Supported Architectures](#supported-architectures)  
   - [Training on Google Colab (TPU)](#training-on-google-colab-tpu)  
   - [Hyperparameter Configuration](#hyperparameter-configuration)  
8. [Evaluation & Metrics](#evaluation--metrics)  
9. [Usage](#usage)  
   - [Real-Time Inference with Webcam](#real-time-inference-with-webcam)  
   - [Batch Prediction](#batch-prediction)  
   - [Integrating into Applications](#integrating-into-applications)  
10. [Human-Centered Validation](#human-centered-validation)  
11. [Future Work & Roadmap](#future-work--roadmap)  
12. [Contributing Guidelines](#contributing-guidelines)  
13. [Acknowledgments & References](#acknowledgments--references)  
14. [License](#license)  

---

## Project Overview

**GesturePro** aims to create an end-to-end, **real-time** gesture recognition system that:  
1. **Streams live video frames** from a webcam or camera feed and detects hand landmarks on the fly.  
2. **Runs a lightweight deep‐learning model** that classifies each frame (or short sequence) into predefined gestures.  
3. **Outputs predictions instantly** (on-screen overlay or via an API) with confidence scores, enabling immediate feedback.  
4. **Supports integration** into assistive tools, AR/VR interfaces, IoT devices, and other hands-free applications.

Key goals:  
- **Accessibility & Inclusion**: Provide sign-language interpreters or assistive communication tools capable of translating gestures to text/speech in near–real time.  
- **Low Latency & Robustness**: Maintain recognition accuracy (≥95%) even under varied lighting, backgrounds, or hand orientations.  
- **User-Friendly Pipeline**: Offer a documented workflow—spanning data collection, model training, and real-time inference—so anyone can test and deploy quickly.  

---

## Human-Centered Approach

GesturePro’s design philosophy is grounded in human-centered research methods. At every stage, we prioritize real-user needs, iterate rapidly on feedback, and ensure the real-time demo feels intuitive and reliable.

### User Profiles & Personas

1. **Deaf or Hard-of-Hearing Individuals**  
   - _Use Case_: Live sign-language translation, enabling on-the-fly conversion to text or speech during conversations.  
   - _Pain Points_: Existing solutions often have high latency or require specialized hardware; fatigue when waiting for delayed translations.  
   - _Key Requirements_: Sub-second response time, high accuracy across lighting conditions and varied signing speeds.

2. **Caregivers & Educators in Special Education**  
   - _Use Case_: Assistive interface in classrooms or therapy sessions to translate simple gestures (e.g., “help,” “yes,” “no”) into text for non-verbal individuals.  
   - _Pain Points_: Difficulty capturing fast-changing gestures; unreliable recognition can confuse students.  
   - _Key Requirements_: Intuitive setup, real-time feedback, and easy customization of gesture sets.

3. **Hands-Free Interaction Enthusiasts (AR/VR, Smart Homes)**  
   - _Use Case_: Control smart devices or navigate VR menus using natural hand motions.  
   - _Pain Points_: Many gesture-control systems require calibration or fail under different backgrounds.  
   - _Key Requirements_: Rapid recognition with minimal setup, robust to background clutter, and low CPU/GPU usage for real-time smoothness.

### Empathy & Accessibility

- **Diverse Data Collection**: Volunteers with varied skin tones, hand sizes, and environmental settings ensure the real-time model generalizes.  
- **Inclusive Labeling**: Define gestures (including both static and dynamic signs) with clear instructions for angle variations and comfort.  
- **Rapid Feedback Loops**: Conduct weekly “guerilla tests” in busy hallways or labs—observe how quickly users detect and correct misclassifications, then refine preprocessing or model parameters to reduce false positives in real-time.

### Iterative Research & Testing

- **Stage 1: Initial Contextual Inquiries**  
  - Interview target users (e.g., deaf students, VR developers) about real-time needs and pain points.  
  - Synthesize findings via affinity mapping in Dovetail—prioritize gestures that matter most for conversational flow (e.g., “thank you,” “help”).  

- **Stage 2: Prototype & Lightning Guerilla Tests**  
  - Build a minimal Python prototype using MediaPipe to stream landmarks; display live predictions on a screen.  
  - Recruit 10 participants for 15-minute sessions, capturing time-to-recognition and perceived delay (Likert scale).  

- **Stage 3: Remote Longitudinal Usability Study**  
  - Share a packaged demo (with simple UI) to 20 remote testers. They use it over one week, logging instances when recognition lag disrupted communication.  
  - Weekly Qualtrics survey gathers subjective satisfaction (ease of use, trust in real-time accuracy).

- **Stage 4: Real-Time Model Refinement & Validation**  
  - Train and deploy the model on a TPU in Colab. Gather live feedback: record which frames had low confidence to identify patterns of failure (e.g., rapid hand rotation).  
  - Iterate on architecture (e.g., temporal smoothing via short LSTM windows) until latency is under 150 ms and accuracy exceeds 95% on critical gestures.

---

## Features

- **End-to-End Real-Time Pipeline**  
  1. **Live Video Input**: Captures webcam feed at 30 FPS using OpenCV or MediaPipe.  
  2. **Landmark Extraction**: Runs fast hand-landmark detection on each frame.  
  3. **On-Device Inference**: Passes normalized landmarks through a lightweight CNN or LSTM model optimized for low latency.  
  4. **Instant Overlay & Output**: Overlays bounding box, predicted label, and confidence on the video; can also send predictions to other applications via API or web socket.

- **Human-Centered Validation Tools**  
  - Interactive Jupyter notebook (`notebooks/4_user_usability_demo.ipynb`) for live demos—captures user metrics (recognition time, misfires).  
  - Auto-generated confusion matrices and per-gesture accuracy reports during live sessions.  
  - Customizable UI skins for high-contrast or large-font overlays to assist users with visual impairments.

- **Extensibility & Custom Gesture Sets**  
  - Easily add new gestures by updating `data/labels/gesture_map.yaml` and supplying new training samples—model retrains in under an hour on Colab TPU.  
  - Toggle between single-hand and two-hand gestures (e.g., full ASL alphabet) via a configuration flag.

---

## Repository Structure

gesturepro/
├── README.md                # ← You are here
├── docs/
│   ├── field_study_notes.md
│   ├── guerilla_testing_report.md
│   ├── user_personas.md
│   └── system_architecture_diagram.png
├── data/
│   ├── raw/
│   │   ├── videos/          # Live or prerecorded sessions (.mp4, .avi)
│   │   └── annotations/     # CSVs with timestamped landmark coordinates + labels
│   ├── processed/
│   │   ├── train/           # Preprocessed NumPy arrays for training
│   │   ├── val/             # Preprocessed arrays for validation
│   │   └── test/            # Preprocessed arrays for testing
│   └── labels/              # YAML/JSON label maps (e.g., gesture → integer)
├── src/
│   ├── data_loader.py       # Functions to load and batch streaming or recorded data
│   ├── preprocess.py        # Landmark normalization, temporal smoothing, augmentation
│   ├── models/
│   │   ├── cnn_model.py      # Low-latency CNN for frame-wise classification
│   │   ├── lstm_model.py     # LSTM for short temporal sequences (e.g., dynamic gestures)
│   │   └── utils.py          # Helper functions (saving/loading models, metrics)
│   ├── train.py             # Training loop supporting CPU/GPU/TPU, with live TensorBoard logging
│   ├── evaluate.py          # Compute accuracy, confusion matrices, and latency metrics
│   ├── infer.py             # Real-Time inference module (Webcam/Video stream) with overlay
│   └── config.py            # Central hyperparameters, paths, and real-time flags
├── notebooks/
│   ├── 1_data_exploration.ipynb    # Visualize landmark distributions and class balance
│   ├── 2_model_prototyping.ipynb    # Quick prototyping for real-time suitability
│   ├── 3_episode_evaluation.ipynb   # Metrics logging, confusion matrix, and latency plots
│   └── 4_user_usability_demo.ipynb  # Remote moderated real-time usability sessions
├── requirements.txt         # Dependencies (e.g., mediapipe, tensorflow, torch, dovetail-sdk)
├── setup.py                 # If using as an installable package (pip install -e .)
└── LICENSE                  # MIT License

---

## Installation & Setup

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yeoleshweta/gesturepro.git
   cd gesturepro

	2.	Create & Activate a Virtual Environment

python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
.\.venv\Scripts\activate        # Windows


	3.	Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt


	4.	Download or Record Data
	•	For existing datasets: place raw videos under data/raw/videos/ and annotations under data/raw/annotations/.
	•	To gather new examples in real-time: follow notebooks/1_data_exploration.ipynb to capture webcam streams and save landmark CSVs.
	5.	Configure Hyperparameters
	•	Open src/config.py and set:

DATA_PATH = "data/processed/"
LABEL_MAP_PATH = "data/labels/gesture_map.yaml"
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 20
USE_TPU = True           # Set False for local GPU/CPU
REALTIME_THRESHOLD_MS = 150  # Target end-to-end latency per frame
AUGMENT_ROTATION = True
AUGMENT_SCALE = True
AUGMENT_NOISE = True



⸻

Data Collection & Preprocessing

Dataset Schema
	•	Video Files (.mp4, .avi)
	•	Resolution: Minimum 640×480 recommended.
	•	Framerate: 30 FPS for smoother real-time landmark extraction.
	•	Filename convention: gestureName_subjectID_sessionID.mp4.
	•	Annotation CSVs
	•	Columns: timestamp, x1, y1, z1, x2, y2, z2, …, x21, y21, z21, label
	•	(xN, yN, zN) are 3D hand-landmark coordinates detected via MediaPipe or similar.
	•	label is an integer index matching data/labels/gesture_map.yaml.

Labeling Guidelines
	•	Edit data/labels/gesture_map.yaml:

thumbs_up: 0
thumbs_down: 1
peace_sign: 2
thank_you: 3
# Add more gestures as needed


	•	Ensure consistency: For two-hand gestures, prefix with left_ or right_, or combine if both hands move together (e.g., hello_two_hands: 4).

Preprocessing Pipeline

Run:

python src/preprocess.py \
  --raw_video_dir data/raw/videos/ \
  --annotation_dir data/raw/annotations/ \
  --output_dir data/processed/ \
  --img_size 224 \
  --normalize landmarks \
  --temporal_smoothing True

This script will:
	1.	Extract frames at 30 FPS (configurable).
	2.	Detect 21 hand landmarks per frame (MediaPipe/OpenCV).
	3.	Normalize and center landmarks—ensuring invariance to hand size/position.
	4.	Apply optional augmentations (rotation ±10°, scaling ±10%, Gaussian noise σ=0.01).
	5.	Smooth temporal landmark sequences to reduce jitter in real-time.
	6.	Split data into train/, val/, test/ (80/10/10, random seed for reproducibility).

After running, you’ll see:

data/processed/
├── train/
│   ├── thumbs_up.npy
│   ├── peace_sign.npy
│   └── …
├── val/
│   └── …
└── test/
    └── …


⸻

Model Training

Supported Architectures
	•	CNN Frame-Wise Classifier (src/models/cnn_model.py)
	•	Input: 21 landmarks × 3 coordinates per frame.
	•	Architecture:
	•	Conv → BatchNorm → ReLU → MaxPool (×3 layers)
	•	Fully connected → Softmax.
	•	Use Case: Static gestures in real-time with latency < 100 ms per frame.
	•	LSTM Sequence Classifier (src/models/lstm_model.py)
	•	Input: Sequence of consecutive frames (e.g., 15–30 frames).
	•	Architecture:
	•	TimeDistributed Dense → 2 LSTM layers → Dropout → Dense → Softmax.
	•	Use Case: Dynamic gestures (e.g., multi-frame signs) with end-to-end latency ~150 ms.

Training on Google Colab (TPU)

Open notebooks/2_model_prototyping.ipynb, switch to a TPU runtime:
	1.	Mount Google Drive (for data and checkpoints).
	2.	Install required packages:

!pip install mediapipe tensorflow torch dovetail-sdk


	3.	Set DATA_DIR, LABEL_MAP_PATH, and OUTPUT_DIR.
	4.	Run preprocessing if needed.
	5.	Train models:
	•	CNN:

from src.train import train_cnn
train_cnn(epochs=20, batch_size=32, use_tpu=True)


	•	LSTM:

from src.train import train_lstm
train_lstm(epochs=15, batch_size=16, use_tpu=True)


	6.	Checkpoints and TensorBoard logs appear in your Drive.

Hyperparameter Configuration

Edit src/config.py:

# Data paths
DATA_PATH = "data/processed/"
LABEL_MAP_PATH = "data/labels/gesture_map.yaml"
# Model hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 20
USE_TPU = True
REALTIME_THRESHOLD_MS = 150  # Target latency
# Augmentation flags
AUGMENT_ROTATION = True
AUGMENT_SCALE = True
AUGMENT_NOISE = True
# Temporal smoothing for real-time stability
TEMPORAL_SMOOTHING = True


⸻

Evaluation & Metrics

Use src/evaluate.py to measure both accuracy and latency:

python src/evaluate.py \
  --model_path checkpoints/cnn_best.h5 \
  --test_data_dir data/processed/test/ \
  --label_map data/labels/gesture_map.yaml \
  --output_dir eval_reports/ \
  --measure_latency True

Outputs include:
	•	Overall Accuracy (e.g., 96.8%)
	•	Per-Class Accuracy (e.g., “thumbs_up: 97%”, “peace_sign: 95%”)
	•	Confusion Matrix (eval_reports/confusion_matrix.png)
	•	Precision / Recall / F1-Score (eval_reports/metrics.csv)
	•	Latency Statistics (mean, median, max per frame in ms) saved as eval_reports/latency.txt

⸻

Usage

Real-Time Inference with Webcam
	1.	Activate Environment

source .venv/bin/activate


	2.	Install Additional Dependencies

pip install opencv-python mediapipe


	3.	Run the Real-Time Inference Script

python src/infer.py \
  --model_path checkpoints/cnn_best.h5 \
  --label_map data/labels/gesture_map.yaml \
  --use_webcam True \
  --show_confidence True \
  --target_fps 30

	•	A window opens showing live webcam feed with:
	•	Detected hand bounding box
	•	Overlayed landmarks
	•	Predicted gesture label
	•	Confidence percentage
	•	Frame latency (ms) to ensure real-time performance
	•	Press q to exit.

	4.	Adjusting Real-Time Settings
	•	In src/config.py, tweak REALTIME_THRESHOLD_MS or TARGET_FPS to match your hardware capabilities.
	•	Use --model_path to switch between CNN (low latency) and LSTM (higher accuracy for dynamic gestures).

Batch Prediction (Offline)

python src/infer.py \
  --model_path checkpoints/lstm_best.h5 \
  --label_map data/labels/gesture_map.yaml \
  --input_videos data/raw/videos/ \
  --output_csv predictions/results.csv \
  --batch_mode True

	•	Processes videos offline.
	•	Outputs a CSV: filename, start_time, end_time, predicted_label, confidence.

Integrating into Other Applications
	•	Python Package Import
After pip install -e ., import and use the recognizer:

from gesturepro.infer import RealTimeGestureRecognizer

recognizer = RealTimeGestureRecognizer(
    model_path="checkpoints/cnn_best.h5",
    label_map="data/labels/gesture_map.yaml",
    target_fps=30
)
label, confidence, latency_ms = recognizer.predict_frame(frame)


	•	REST API Example
A src/api/ folder includes a FastAPI skeleton. Run:

cd src/api
uvicorn main:app --reload

Send a POST request with base64-encoded frames → Receive JSON { "label": "thank_you", "confidence": 0.92, "latency_ms": 120 }.

⸻

Human-Centered Validation

To ensure the real-time system truly meets user needs:
	1.	Remote Moderated Usability Testing
	•	Use Lookback or Zoom to observe participants interacting with the live inference demo.
	•	Capture quantitative metrics (recognition time, false positives per minute) and qualitative feedback (ease, comfort with speed).
	2.	In-Lab Usability Sessions
	•	Host sessions with 5–7 target users (e.g., deaf or hard-of-hearing volunteers).
	•	Ask them to perform conversational tasks (e.g., “Spell your name live”) and measure end-to-end latency and success rate.
	•	Post-task SUS survey aiming for score ≥ 80.
	3.	Field Deployment & Diary Study
	•	Provide a desktop or mobile demo to 15 beta testers for a week.
	•	Participants record daily usage logs: “Did I rely on GesturePro today? Was response too slow? Any misfires during conversation?”
	•	Analyze open-ended feedback via thematic coding in Dovetail to identify persistent real-time challenges.
	4.	Accessibility & Ethical Audit
	•	Verify the on-screen overlay uses high-contrast colors, large fonts, and audio cues for visually impaired users.
	•	Confirm no personal video frames are stored without explicit consent—adhere to ethical guidelines.

⸻

Future Work & Roadmap
	•	Expanded Real-Time Vocabulary
	•	Add full ASL alphabet and common phrases, ensuring each new gesture still meets sub-150 ms latency.
	•	Lightweight On-Device Models
	•	Convert the best CNN model to TensorFlow Lite or PyTorch Mobile so inference can run on smartphones or edge devices, maintaining real-time performance.
	•	Multimodal Fusion
	•	Incorporate depth data (e.g., from a depth camera) to reduce false positives in cluttered backgrounds, while still maintaining low latency.
	•	Explainable AI in Real-Time
	•	Overlay saliency heatmaps showing which landmarks influenced each prediction, helping users trust the system’s decisions.
	•	Cross-Cultural & Domain Expansion
	•	Collaborate with global deaf communities to validate dynamic conversational gestures across dialects.
	•	Explore applications in industrial settings (e.g., hands-free machinery control) where real-time responsiveness is critical.

⸻

Contributing Guidelines

We welcome contributions that align with our human-centered, real-time ethos. To propose changes:
	1.	Fork & Create a Branch

git checkout -b feature/<feature-name>


	2.	Document Your Change
	•	For new real-time features, include performance benchmarks (latency stats under various lighting).
	•	If adding gestures, supply both data samples and notes on how you tested in real-time conditions.
	•	Provide a brief user-testing summary (in docs/) showing how real users interacted with the new feature.
	3.	Submit a Pull Request (PR)
	•	Include a clear description of changes, screenshots or video demonstrations, and any user-testing evidence.
	•	Ensure any new code maintains real-time performance (latency under target threshold).
	4.	Review & Merge
	•	Maintainers will evaluate code quality, performance benchmarks, and alignment with human-centered real-time goals.

Please run existing tests and add unit or integration tests for new real-time scenarios.

⸻

Acknowledgments & References
	•	MediaPipe Hands: Google’s real-time hand-pose detection framework—critical for sub-100 ms landmark extraction.
	•	TensorFlow & PyTorch: Open-source deep learning libraries used for model development and quantization.
	•	Dovetail: Platform for qualitative data analysis—used extensively to code user feedback during real-time demos.
	•	UserTesting.com & Lookback: Tools for remote moderated usability testing—essential for collecting real-time user insights.
	•	OpenCV: Computer vision library enabling webcam capture, landmark overlays, and latency measurement.
	•	FastAPI & Uvicorn: Lightweight frameworks for building a real-time inference API.

Special thanks to all beta testers who volunteered for live testing and provided invaluable feedback on real-time responsiveness.

⸻

“Putting real users at the center of a real-time system demands not just accuracy, but sub-second responsiveness and continual iteration based on immediate feedback.”

