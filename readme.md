# Multimodal Elderly Fall Detection & Intelligent Emergency Response

An end-to-end multimodal AI system for detecting possible elderly falls from **prerecorded video and audio** uploads and coordinating an auditable emergency response workflow—without requiring wearables, live webcams, microphones, or IoT sensors.

## Supported Scenarios (Audio+Video, Video-Only, Audio-Only)

All three modality combinations are supported end-to-end by `fusion/multimodal_fusion.py` and `backend/incident_service.py`:

| Scenario | Classification | Alert Level | What Happens | 
| ----- | ----- | ----- | ----- | 
| **Audio + Video** (fall + impact/distress) | `POSSIBLE_MULTIMODAL_FALL_CANDIDATE` | `HIGH` | Combines pose transition, floor inactivity, and acoustic impact/distress; triggers 3-tier escalation, caregiver verification, and VLM timeline. | 
| **Video Only** (silent fall with `>= 3.0s` inactivity & Video `>= 75`) | `VIDEO_INACTIVITY_FALL_CANDIDATE` | `HIGH` | Flags silent floor falls using pose + inactivity without requiring audio; creates a `HIGH` alert for caregiver verification. | 
| **Video Only** (posture candidate without prolonged inactivity) | `VIDEO_ONLY_CANDIDATE` | `REVIEW` | Logs an incident for caregiver review (`REVIEW_AUDIO_CONTEXT`) without over-escalating normal movement. | 
| **Audio Only** (e.g., user shouts *"help"* with no fall in video) | `AUDIO_ONLY_CANDIDATE` | `HIGH` | Even when `video_score = 0.00`, `speech_distress.py` + `timestamp_alignment.py` flag an off-camera distress event, create a `HIGH` severity `OPEN` alert with the transcript, and require caregiver verification. | 

## Architecture Overview

The pipeline processes uploaded `.mp4`/`.avi` video (and optional separate `.wav`/`.mp3` audio) through five stages:

1. **Video Analysis (`video/`)**

   * Uses **OpenCV** and **MediaPipe Pose Landmarker Lite** (33 body landmarks) to track posture transitions, torso angles, hip-drop movement, rapid fall-and-recovery, and post-fall floor inactivity.

   * Applies duplicate-pose suppression and temporal gap merging to avoid fragmented events.

2. **Audio Analysis (`audio/`)**

   * Extracts 16 kHz mono WAV audio via **FFmpeg** (or ingests a separate uploaded audio file).

   * Uses **TensorFlow Hub YAMNet** to detect impact-like acoustic events (thuds, slams, crashes) and **Speech-to-Text** (`Groq Whisper` with `Google Web Speech` fallback) to flag distress phrases such as *"help"*, *"help me"*, or *"I have fallen"*.

3. **Timestamp Alignment, Fusion & VLM Context (`fusion/`)**

   * Aligns video and audio events on a shared timeline (including off-camera audio-only distress).

   * Computes dynamic modality scores and classifies events across the 3-scenario taxonomy above.

   * Uses **Google Gemini 2.5 Flash VLM** (with a deterministic heuristic fallback) as an additional contextual step to generate a human-readable activity summary and `[BEFORE] / [DURING] / [AFTER]` timeline.

4. **Backend & Emergency Orchestration (`backend/`, `app/`, `orchestration/`)**

   * **FastAPI** REST APIs + **SQLAlchemy / SQLite** (`data/incidents/incidents.db`) persist incidents, alerts, caregiver verifications, and follow-up tasks.

   * Implements a **3-tier escalation ladder** (`Assigned Caregiver` → `Senior Caregiver` → `Nurse / ERT`), enforces caregiver verification (`TRUE_FALL`, `FALSE_ALERT`, `UNCERTAIN`) to prevent silent alert closure, and creates post-fall Nurse assessment tasks.

5. **Evaluation & Dashboard (`evaluation/`, `dashboard/`)**

   * Interactive HTML/CSS/JavaScript dashboard for uploading media, inspecting VLM timelines, managing alerts, and viewing **Sub-Case 3** evaluation metrics (Video-only vs. Audio-only vs. Multimodal Precision, Recall, F1-Score, and False Positive Rate).

## Technology Stack & Key Parameters

| Layer | Technologies & Models | Key Thresholds / Settings | 
| ----- | ----- | ----- | 
| **Video AI** | OpenCV, MediaPipe Pose Landmarker Lite | `MAX_VIDEO_GAP_SECONDS = 1.00s`, `MIN_TRANSITION_HIP_DROP = 0.05`, `INACTIVITY_DISPLACEMENT_THRESHOLD = 0.035`, `PROLONGED_INACTIVITY_SECONDS = 2.0s` | 
| **Audio AI** | FFmpeg (16 kHz mono), TensorFlow Hub YAMNet, Groq Whisper / Google Web Speech | Impact energy scoring + distress keyword matching (`fallen`, `help`, `help me`, `i have fallen`, `emergency`, `can't get up`) | 
| **Fusion & VLM** | Timestamp Alignment, Dynamic Evidence Scoring, Gemini 2.5 Flash | `POSSIBLE_MULTIMODAL_FALL_CANDIDATE` (Video >= 60 + Audio support), `VIDEO_INACTIVITY_FALL_CANDIDATE` (Video >= 75 + >= 3.0s inactivity), `AUDIO_ONLY_CANDIDATE` (distress/impact) | 
| **Backend & DB** | Python 3.11, FastAPI, Uvicorn, SQLAlchemy, SQLite | `data/incidents/incidents.db` | 
| **Frontend** | Vanilla HTML, CSS, JavaScript | Served at `/dashboard/` | 

## Project Structure

```
multimodal_fall_detection_project/
├── app/
│   └── main.py                         # FastAPI application & API routes
├── audio/
│   ├── speech_distress.py              # STT transcription & distress keyword detection
│   └── yamnet_classifier.py            # YAMNet acoustic impact classification
├── backend/
│   ├── alert_service.py                # 3-tier escalation, acknowledgement & verification
│   ├── database.py                     # SQLite / SQLAlchemy engine & session setup
│   ├── incident_service.py             # Incident import & Nurse follow-up tasks
│   └── models.py                       # ORM models (Incident, Alert, FollowUpTask)
├── dashboard/
│   ├── index.html                      # Web UI dashboard
│   ├── script.js                       # Frontend API integration & rendering
│   └── style.css                       # Dashboard styling
├── evaluation/
│   ├── benchmark_dataset.json          # Labeled benchmark test cases
│   └── evaluator.py                    # Unimodal vs. Multimodal metric computation
├── fusion/
│   ├── evidence_scoring.py             # Modality confidence scoring
│   ├── multimodal_fusion.py            # Classification taxonomy & fusion logic
│   ├── timestamp_alignment.py          # Cross-modal timeline alignment
│   └── vlm_analyzer.py                 # Gemini 2.5 Flash contextual summary & timeline
├── orchestration/
│   └── run_pipeline.py                 # End-to-end CLI pipeline runner
├── video/
│   ├── duplicate_pose_suppression.py   # Duplicate detection filtering
│   └── temporal_pose_analyzer.py       # MediaPipe pose & temporal inactivity analysis
├── .env.example                        # Full environment template (keys left blank)
├── requirements.txt                    # Python dependencies
└── README.md

```

## Quick Start

### 1. Prerequisites

* Python **3.11**

* **FFmpeg** installed and available on your system `PATH`

### 2. Install Dependencies

```
pip install -r requirements.txt

```

### 3. Configure `.env` (Full `.env.example` Template)

Copy `.env.example` to `.env` and add your free-tier API keys if you want live Gemini VLM and Groq Whisper STT (deterministic fallbacks run automatically if keys are left blank):

```
# --- API Keys (Leave Blank in Git / Fill Locally in .env) ---
GEMINI_API_KEY=
GROQ_API_KEY=

# --- Model Configuration (Non-Secret Defaults) ---
GEMINI_MODEL=gemini-2.5-flash
GROQ_STT_MODEL=whisper-large-v3-turbo
AUDIO_SAMPLE_RATE=16000

# --- Server & Database Configuration ---
APP_HOST=127.0.0.1
APP_PORT=8000
DATABASE_URL=sqlite:///data/incidents/incidents.db

# --- Video & Inactivity Detection Thresholds ---
MAX_VIDEO_GAP_SECONDS=1.00
MIN_TRANSITION_HIP_DROP=0.05
INACTIVITY_DISPLACEMENT_THRESHOLD=0.035
PROLONGED_INACTIVITY_SECONDS=2.0

```

### 4. Start the FastAPI Server & Dashboard

```
uvicorn app.main:app --reload

```

Open `http://127.0.0.1:8000/dashboard/` in your browser to upload videos, inspect VLM timelines, manage alerts, and view evaluation results.

### 5. Run via CLI (Optional)

```
python -m orchestration.run_pipeline --video data/uploads/le2i_home_01_fall.mp4 --resident "Mr. John" --room "Room 204"

```

You can also pass a separate audio file using `--audio path/to/audio.wav`.

## Current Limitations & Future Improvements

* **Prerecorded Media Scope:** Currently processes uploaded video/audio files rather than live RTSP/CCTV camera streams.

* **Simulated Alert Escalation:** Tiered escalation is tracked inside SQLite and the dashboard UI; external SMS, email, or call notifications (e.g., Twilio/FCM) are planned for future integration.

* **Benchmark Scale:** The evaluation module is a working prototype; expanding to a larger, multi-resident labeled dataset is needed for clinical-grade validation.

## Repository

* **GitLab Internship Repository:** [Siam Computing Internship Tasks](https://gitlab.siamcomputing.com/siamcomputing-training/cet/machine-learning/ml-interns/2026/nirmal-kumar/internship-tasks)