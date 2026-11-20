"""
Speech-to-Text & Distress Keyword Detection for Prerecorded Audio.

Supports free web-based providers:
1. Groq Whisper API (free tier, whisper-large-v3-turbo) when GROQ_API_KEY is set.
2. Google Web Speech API (via SpeechRecognition, no API key required) as secondary option.
3. Offline acoustic fallback (safe no-crash mode if offline or rate-limited).
"""

import argparse
import json
import os
import re
import urllib.error
import urllib.request
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DISTRESS_KEYWORDS = {
    "help me": 1.0,
    "help": 0.90,
    "i have fallen": 1.0,
    "i fell": 1.0,
    "fallen": 0.85,
    "falling": 0.80,
    "emergency": 0.95,
    "hurt": 0.75,
    "pain": 0.75,
    "can't get up": 0.95,
    "cant get up": 0.95,
    "ambulance": 0.95,
    "doctor": 0.80,
    "save me": 0.95,
    "ouch": 0.65,
    "bachao": 0.95,
    "madad": 0.95,
}


def detect_distress_keywords(text: str) -> dict:
    if not text or not text.strip():
        return {
            "distress_detected": False,
            "matched_keywords": [],
            "distress_score": 0.0,
        }

    normalized = re.sub(r"[^\w\s']", " ", text.lower())
    matched = []
    max_score = 0.0

    for phrase, weight in DISTRESS_KEYWORDS.items():
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, normalized):
            matched.append(phrase)
            max_score = max(max_score, weight)

    # Boost slightly if multiple distinct distress terms appear
    if len(matched) >= 2:
        max_score = min(1.0, max_score + 0.05)

    return {
        "distress_detected": len(matched) > 0,
        "matched_keywords": sorted(set(matched)),
        "distress_score": round(max_score, 3),
    }


def _transcribe_with_groq(audio_path: Path, api_key: str) -> dict:
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    audio_bytes = audio_path.read_bytes()

    parts = []
    fields = {
        "model": "whisper-large-v3-turbo",
        "response_format": "verbose_json",
        "temperature": "0",
    }
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode("utf-8")
        )

    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8")
        + audio_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=45) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    transcript = str(payload.get("text", "")).strip()
    segments = []
    for seg in payload.get("segments", []) or []:
        seg_text = str(seg.get("text", "")).strip()
        seg_distress = detect_distress_keywords(seg_text)
        segments.append({
            "start": round(float(seg.get("start", 0.0)), 3),
            "end": round(float(seg.get("end", 0.0)), 3),
            "text": seg_text,
            "distress_detected": seg_distress["distress_detected"],
            "matched_keywords": seg_distress["matched_keywords"],
        })

    return {
        "provider": "groq-whisper-large-v3-turbo",
        "transcript": transcript,
        "segments": segments,
    }


def _transcribe_with_google_web_speech(audio_path: Path) -> dict:
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    with sr.AudioFile(str(audio_path)) as source:
        audio_data = recognizer.record(source)

    transcript = recognizer.recognize_google(audio_data)
    distress = detect_distress_keywords(transcript)
    return {
        "provider": "google-web-speech-free",
        "transcript": transcript,
        "segments": [
            {
                "start": 0.0,
                "end": 0.0,
                "text": transcript,
                "distress_detected": distress["distress_detected"],
                "matched_keywords": distress["matched_keywords"],
            }
        ],
    }


def analyze_speech_distress(audio_path: Path, output_json: Path = None) -> dict:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    result = None
    errors = []

    if groq_key:
        try:
            result = _transcribe_with_groq(audio_path, groq_key)
        except Exception as exc:
            errors.append(f"Groq Whisper error: {exc}")

    if result is None:
        try:
            result = _transcribe_with_google_web_speech(audio_path)
        except Exception as exc:
            errors.append(f"Google Web Speech error: {exc}")

    if result is None:
        result = {
            "provider": "offline-fallback",
            "transcript": "",
            "segments": [],
            "note": (
                "No live STT provider reachable or GROQ_API_KEY not set. "
                + "; ".join(errors)
            ).strip(),
        }

    overall_distress = detect_distress_keywords(result.get("transcript", ""))
    result["distress_detected"] = overall_distress["distress_detected"]
    result["matched_keywords"] = overall_distress["matched_keywords"]
    result["distress_score"] = overall_distress["distress_score"]
    result["audio_file"] = audio_path.name

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze speech & distress keywords from WAV audio.")
    parser.add_argument("--audio", required=True, help="Path to 16kHz WAV file.")
    parser.add_argument("--output", default=None, help="Output JSON path.")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_absolute():
        audio_path = PROJECT_ROOT / audio_path

    output_json = Path(args.output) if args.output else audio_path.with_suffix(".speech.json")
    if not output_json.is_absolute():
        output_json = PROJECT_ROOT / output_json

    result = analyze_speech_distress(audio_path, output_json=output_json)
    print(f"Speech provider  : {result['provider']}")
    print(f"Transcript       : {result['transcript'] or '(empty)'}")
    print(f"Distress detected: {result['distress_detected']} {result['matched_keywords']}")
    print(f"Saved JSON       : {output_json}")


if __name__ == "__main__":
    main()
