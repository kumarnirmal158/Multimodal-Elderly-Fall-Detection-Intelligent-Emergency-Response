import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from backend.database import SessionLocal, create_tables
from backend.incident_service import import_fusion_events
from backend.alert_service import generate_alerts
from audio.speech_distress import analyze_speech_distress
from fusion.timestamp_alignment import run_alignment
from fusion.evidence_scoring import run_scoring
from fusion.multimodal_fusion import run_fusion

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_project_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def run_command(command, description):
    print("\n" + "=" * 65)
    print(description)
    print("=" * 65)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def extract_or_convert_audio(source_media_path: Path, output_wav_path: Path):
    """Extract 16kHz mono PCM WAV from MP4/AVI or convert separate WAV/MP3."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError("FFmpeg is unavailable. Install ffmpeg or imageio-ffmpeg.") from exc

    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg, "-y",
        "-i", str(source_media_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(output_wav_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not output_wav_path.exists() or output_wav_path.stat().st_size == 0:
        raise RuntimeError("Audio extraction failed: output WAV is missing or empty.")
    print(f"Audio prepared successfully: {output_wav_path}")


def run_pipeline(
    video_path,
    audio_path=None,
    source_id=None,
    resident_name="Mr. John",
    room_location="Room 204",
):
    video_path = resolve_project_path(video_path)
    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    separate_audio = resolve_project_path(audio_path) if audio_path else None
    if separate_audio is not None and not separate_audio.exists():
        raise FileNotFoundError(f"Input audio not found: {separate_audio}")

    if source_id is None:
        path_hash = hashlib.sha1(str(video_path).lower().encode("utf-8")).hexdigest()[:8]
        source_id = f"{video_path.stem}-{path_hash}"

    pipeline_dir = PROJECT_ROOT / "data" / "annotations" / "pipeline" / source_id
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    video_csv = pipeline_dir / "video_pose_features.csv"
    prepared_wav = PROJECT_ROOT / "data" / "audio" / "processed" / f"{source_id}.wav"
    audio_csv = pipeline_dir / "yamnet_features.csv"
    speech_json = pipeline_dir / "speech_distress.json"
    aligned_csv = pipeline_dir / "aligned_events.csv"
    scored_csv = pipeline_dir / "scored_events.csv"
    fusion_dir = PROJECT_ROOT / "outputs" / "pipeline" / source_id
    final_events_csv = fusion_dir / "final_fusion_events.csv"

    # 1. Video Pose + Inactivity Extraction
    run_command(
        [sys.executable, "-m", "video.temporal_pose_analyzer", "--video", str(video_path), "--output", str(video_csv)],
        "STAGE 1 — VIDEO POSE & INACTIVITY ANALYSIS",
    )

    # 2. Audio Extraction / Conversion (supports prerecorded video audio OR separate WAV/MP3)
    extract_or_convert_audio(separate_audio if separate_audio else video_path, prepared_wav)

    # 3A. YAMNet Classification
    run_command(
        [sys.executable, "-m", "audio.yamnet_classifier", "--audio", str(prepared_wav), "--output", str(audio_csv)],
        "STAGE 3A — YAMNET AUDIO CLASSIFICATION",
    )

    # 3B. Speech-to-Text & Distress Keywords
    print("\n" + "=" * 65)
    print("STAGE 3B — SPEECH-TO-TEXT & DISTRESS DETECTION")
    print("=" * 65)
    speech_result = analyze_speech_distress(prepared_wav, output_json=speech_json)
    print(f"Transcript: {speech_result.get('transcript') or '(none)'} | Distress: {speech_result.get('distress_detected')}")

    # 4. Timestamp Alignment
    run_alignment(video_csv=video_csv, audio_csv=audio_csv, output_csv=aligned_csv, speech_json=speech_json)

    # 5. Evidence Scoring
    run_scoring(input_csv=aligned_csv, output_csv=scored_csv)

    # 6. Multimodal Fusion + VLM Interpretation
    run_fusion(input_csv=scored_csv, output_dir=fusion_dir, video_path=video_path)

    # 7. Database Import & Caretaker Alert Generation
    create_tables()
    with SessionLocal() as db:
        import_result = import_fusion_events(
            db,
            csv_path=final_events_csv,
            source_video=str(video_path),
            source_id=source_id,
            resident_name=resident_name,
            room_location=room_location,
        )
        alert_result = generate_alerts(db)

    return {
        "source_id": source_id,
        "video_csv": str(video_csv),
        "audio_csv": str(audio_csv),
        "speech_json": str(speech_json),
        "aligned_csv": str(aligned_csv),
        "scored_csv": str(scored_csv),
        "final_events_csv": str(final_events_csv),
        "fusion_dir": str(fusion_dir),
        "import": import_result,
        "alerts": alert_result,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the complete elderly fall detection backend.")
    parser.add_argument("--video", required=True, help="Path to prerecorded MP4/AVI video.")
    parser.add_argument("--audio", default=None, help="Optional separate WAV/MP3 audio file.")
    parser.add_argument("--source-id", default=None, help="Optional identifier for this video source.")
    parser.add_argument("--resident", default="Mr. John", help="Resident name.")
    parser.add_argument("--room", default="Room 204", help="Resident room location.")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        audio_path=args.audio,
        source_id=args.source_id,
        resident_name=args.resident,
        room_location=args.room,
    )


if __name__ == "__main__":
    main()
