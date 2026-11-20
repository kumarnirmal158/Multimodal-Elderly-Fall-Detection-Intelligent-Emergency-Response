"""
Vision-Language Model (VLM) Contextual Event Interpreter.
"""

import base64
import json
import os
import urllib.request
from pathlib import Path

import cv2


def extract_event_keyframes(video_path: Path, start_time: float, end_time: float) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total_frames / fps) if fps > 0 and total_frames > 0 else max(end_time + 1.0, 1.0)

    mid_time = (start_time + end_time) / 2.0
    sample_points = [
        ("before", max(0.0, start_time - 0.8)),
        ("during", min(duration - 0.05, max(0.0, mid_time))),
        ("after", min(duration - 0.05, end_time + 0.8)),
    ]

    frames_b64 = []
    try:
        for label, ts in sample_points:
            frame_idx = max(0, int(ts * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if max(h, w) > 640:
                scale = 640.0 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok_enc:
                frames_b64.append({
                    "label": label,
                    "timestamp": round(ts, 2),
                    "b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                })
    finally:
        cap.release()

    return frames_b64


def _call_gemini_vlm(keyframes: list[dict], context: dict, api_key: str) -> dict:
    prompt = (
        "You are an elderly-care safety AI reviewing 3 sequential video frames "
        "(BEFORE, DURING, AFTER) of a candidate movement event.\n"
        f"Sensor context: {json.dumps(context)}\n\n"
        "Distinguish an actual fall from normal activities (bending down or kneeling to pick up an object, "
        "sitting on a chair, push-ups/exercise, or intentionally lying down on a bed/mat).\n"
        "Return ONLY valid JSON with these exact keys:\n"
        "{\n"
        '  "activity_type": "fall | rapid_fall_recovery | bending | sitting | lying_intentionally | walking | ambiguous",\n'
        '  "fall_likely": true or false,\n'
        '  "confidence": 0.0 to 1.0,\n'
        '  "post_fall_state": "remains_on_floor | recovered_upright | seated_or_standing | unclear",\n'
        '  "environmental_context": "short description of surroundings and any object/impact clues",\n'
        '  "incident_summary": "2-sentence clinical/caregiver summary of before, during, and after."\n'
        "}"
    )

    parts = [{"text": prompt}]
    for kf in keyframes:
        parts.append({"text": f"Frame ({kf['label']} at {kf['timestamp']}s):"})
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": kf["b64"],
            }
        })

    body = json.dumps({
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }).encode("utf-8")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=45) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    raw_text = payload["candidates"][0]["content"]["parts"][0]["text"]
    parsed = json.loads(raw_text)
    parsed["vlm_provider"] = "gemini-2.5-flash"
    return parsed


def _heuristic_vlm_fallback(context: dict, reason: str) -> dict:
    video_score = float(context.get("video_score", 0.0))
    video_duration = float(context.get("video_duration", 0.0))
    audio_support = bool(context.get("audio_support_present", False))
    distress = bool(context.get("distress_detected", False))
    inactivity = float(context.get("post_fall_inactivity_seconds", 0.0))
    rapid_recovery = bool(context.get("rapid_fall_recovery", False)) and (video_duration <= 2.5)

    if rapid_recovery:
        activity = "rapid_fall_recovery"
        fall_likely = True
        post_state = "recovered_upright"
        conf = 0.72 if audio_support else 0.60
    elif video_score >= 60 and (audio_support or distress):
        activity = "fall"
        fall_likely = True
        post_state = "remains_on_floor" if inactivity >= 2.0 else "unclear"
        conf = 0.85
    elif video_score >= 75 and inactivity >= 3.0:
        activity = "fall"
        fall_likely = True
        post_state = "remains_on_floor"
        conf = 0.78
    elif video_score >= 60:
        activity = "ambiguous"
        fall_likely = False
        post_state = "unclear"
        conf = 0.50
    else:
        activity = "bending" if video_score > 0 else "walking"
        fall_likely = False
        post_state = "seated_or_standing"
        conf = 0.70

    summary = (
        f"Pose analysis recorded a {video_duration:.2f}s candidate window "
        f"(video score {video_score:.1f}/100, post-event inactivity {inactivity:.1f}s). "
        f"Audio impact support={audio_support}, speech distress={distress}."
    )

    return {
        "vlm_provider": f"heuristic-fallback ({reason})",
        "activity_type": activity,
        "fall_likely": fall_likely,
        "confidence": conf,
        "post_fall_state": post_state,
        "environmental_context": "Evaluated from multimodal pose and audio telemetry.",
        "incident_summary": summary,
    }


def interpret_event_with_vlm(video_path: Path, event_context: dict) -> dict:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    start_t = float(event_context.get("video_start_time", 0.0))
    end_t = float(event_context.get("video_end_time", start_t + 1.0))

    if not api_key:
        return _heuristic_vlm_fallback(event_context, "GEMINI_API_KEY not set")

    keyframes = extract_event_keyframes(Path(video_path), start_t, end_t)
    if not keyframes:
        return _heuristic_vlm_fallback(event_context, "could not extract keyframes")

    try:
        return _call_gemini_vlm(keyframes, event_context, api_key)
    except Exception as exc:
        return _heuristic_vlm_fallback(event_context, f"API error: {exc}")
