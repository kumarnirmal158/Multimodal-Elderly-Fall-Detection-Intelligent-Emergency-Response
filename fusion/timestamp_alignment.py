import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_VIDEO_FPS = 30.0
# Allow up to 1.0s gap (e.g. brief occlusion or floor-level landmark dropout)
# so one continuous fall + lying sequence stays a single event.
MAX_VIDEO_GAP_SECONDS = 1.00

ALIGNED_EVENT_COLUMNS = [
    "event_id",
    "track_id",
    "video_start_time",
    "video_end_time",
    "video_duration",
    "video_observation_count",
    "max_torso_angle",
    "mean_torso_angle",
    "transition_observation_count",
    "fall_like_observation_count",
    "horizontal_observation_count",
    "rapid_fall_recovery",
    "post_fall_inactivity_seconds",
    "prolonged_inactivity",
    "overlapping_audio_frame_count",
    "audio_impact_candidate_count",
    "audio_high_energy_count",
    "max_audio_impact_score",
    "max_vocal_distress_score",
    "max_audio_rms",
    "audio_frame_indices",
    "audio_impact_frame_indices",
    "audio_high_energy_frame_indices",
    "audio_categories",
    "audio_impact_categories",
    "audio_support_present",
    "distress_detected",
    "distress_score",
    "distress_keywords",
    "distress_transcript",
]


def to_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def estimate_frame_duration(video_df):
    timestamps = (
        pd.to_numeric(video_df["timestamp"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    differences = timestamps.diff().dropna()
    if not differences.empty:
        median_difference = differences.median()
        if median_difference > 0:
            return float(median_difference)
    return 1.0 / DEFAULT_VIDEO_FPS


def validate_columns(df, required_columns, filename):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{filename} is missing required columns: {missing}")


def prepare_video_candidates(video_df):
    if video_df.empty:
        return video_df.copy()

    video_df = video_df.copy()
    for col in ["timestamp", "frame", "torso_angle"]:
        video_df[col] = pd.to_numeric(video_df[col], errors="coerce")

    for col in ["transition_candidate", "fall_like_event", "horizontal_posture"]:
        video_df[col] = video_df[col].apply(to_bool)

    if "rapid_fall_recovery_candidate" in video_df.columns:
        video_df["rapid_fall_recovery_candidate"] = video_df["rapid_fall_recovery_candidate"].apply(to_bool)
    else:
        video_df["rapid_fall_recovery_candidate"] = False

    if "prolonged_inactivity" in video_df.columns:
        video_df["prolonged_inactivity"] = video_df["prolonged_inactivity"].apply(to_bool)
    else:
        video_df["prolonged_inactivity"] = False

    if "post_fall_inactivity_seconds" in video_df.columns:
        video_df["post_fall_inactivity_seconds"] = pd.to_numeric(
            video_df["post_fall_inactivity_seconds"], errors="coerce"
        ).fillna(0.0)
    else:
        video_df["post_fall_inactivity_seconds"] = 0.0

    candidates = video_df[
        video_df["transition_candidate"] | video_df["horizontal_posture"]
    ].copy()
    candidates = candidates.dropna(subset=["timestamp", "track_id"])
    return candidates.sort_values(["track_id", "timestamp"])


def create_video_event(rows, track_id, frame_duration):
    event_df = pd.DataFrame(rows)
    start_time = float(event_df["timestamp"].min())
    end_time = float(event_df["timestamp"].max()) + frame_duration
    duration = end_time - start_time

    # Sustained horizontal duration also reflects post-fall inactivity if person stays down
    reported_inactivity = float(event_df["post_fall_inactivity_seconds"].max())
    horizontal_span = 0.0
    horiz_df = event_df[event_df["horizontal_posture"]]
    if len(horiz_df) >= 2:
        horizontal_span = float(horiz_df["timestamp"].max() - horiz_df["timestamp"].min())

    inactivity_seconds = round(max(reported_inactivity, horizontal_span if not bool(event_df["rapid_fall_recovery_candidate"].any()) else 0.0), 3)
    prolonged_inactivity = bool(event_df["prolonged_inactivity"].any()) or (inactivity_seconds >= 2.0)

    return {
        "track_id": int(track_id),
        "video_start_time": round(start_time, 3),
        "video_end_time": round(end_time, 3),
        "video_duration": round(duration, 3),
        "video_observation_count": len(event_df),
        "max_torso_angle": float(event_df["torso_angle"].max()),
        "mean_torso_angle": float(event_df["torso_angle"].mean()),
        "transition_observation_count": int(event_df["transition_candidate"].sum()),
        "fall_like_observation_count": int(event_df["fall_like_event"].sum()),
        "horizontal_observation_count": int(event_df["horizontal_posture"].sum()),
        "rapid_fall_recovery": bool(event_df["rapid_fall_recovery_candidate"].any()),
        "post_fall_inactivity_seconds": inactivity_seconds,
        "prolonged_inactivity": prolonged_inactivity,
    }


def group_video_events(video_candidates, frame_duration):
    if video_candidates.empty:
        return []

    events = []
    for track_id, track_df in video_candidates.groupby("track_id", sort=True):
        track_df = track_df.sort_values("timestamp")
        current_group = []
        previous_timestamp = None

        for _, row in track_df.iterrows():
            timestamp = float(row["timestamp"])
            if previous_timestamp is not None and (timestamp - previous_timestamp) > MAX_VIDEO_GAP_SECONDS:
                events.append(create_video_event(current_group, track_id, frame_duration))
                current_group = []
            current_group.append(row)
            previous_timestamp = timestamp

        if current_group:
            events.append(create_video_event(current_group, track_id, frame_duration))
    return events


def prepare_audio_windows(audio_df):
    if audio_df.empty:
        return audio_df.copy()

    audio_df = audio_df.copy()
    for col in ["start_time", "end_time", "impact_score", "frame_index"]:
        audio_df[col] = pd.to_numeric(audio_df[col], errors="coerce")

    if "vocal_distress_score" in audio_df.columns:
        audio_df["vocal_distress_score"] = pd.to_numeric(
            audio_df["vocal_distress_score"], errors="coerce"
        ).fillna(0.0)
    else:
        audio_df["vocal_distress_score"] = 0.0

    audio_df["impact_candidate"] = audio_df["impact_candidate"].apply(to_bool)
    audio_df["high_energy"] = audio_df["high_energy"].apply(to_bool)
    audio_df = audio_df.dropna(subset=["start_time", "end_time"])
    return audio_df.sort_values("start_time")


def align_video_with_audio(video_events, audio_df, speech_data=None):
    speech_data = speech_data or {}
    distress_detected = bool(speech_data.get("distress_detected", False))
    distress_score = float(speech_data.get("distress_score", 0.0))
    distress_keywords = ", ".join(speech_data.get("matched_keywords", [])) or "None"
    distress_transcript = str(speech_data.get("transcript", "")).strip() or "None"

    aligned_events = []

    for event_number, video_event in enumerate(video_events, start=1):
        video_start = video_event["video_start_time"]
        video_end = video_event["video_end_time"]

        overlapping_audio = audio_df[
            (audio_df["start_time"] < video_end + 1.0)
            & (audio_df["end_time"] > max(0.0, video_start - 0.5))
        ].copy()

        if overlapping_audio.empty:
            audio_candidate_count = 0
            high_energy_count = 0
            max_impact_score = 0.0
            max_vocal_distress = 0.0
            max_audio_rms = 0.0
            audio_categories = "None"
            audio_impact_categories = "None"
            audio_frame_indices = "None"
            audio_impact_frame_indices = "None"
            audio_high_energy_frame_indices = "None"
        else:
            all_frame_indices = sorted(set(overlapping_audio["frame_index"].dropna().astype(int)))
            audio_frame_indices = ", ".join(map(str, all_frame_indices))

            impact_rows = overlapping_audio[overlapping_audio["impact_candidate"]].copy()
            audio_candidate_count = len(impact_rows)
            impact_frame_indices = sorted(set(impact_rows["frame_index"].dropna().astype(int)))
            audio_impact_frame_indices = (
                ", ".join(map(str, impact_frame_indices)) if impact_frame_indices else "None"
            )

            high_energy_rows = overlapping_audio[overlapping_audio["high_energy"]].copy()
            high_energy_count = len(high_energy_rows)
            high_energy_frame_indices = sorted(set(high_energy_rows["frame_index"].dropna().astype(int)))
            audio_high_energy_frame_indices = (
                ", ".join(map(str, high_energy_frame_indices)) if high_energy_frame_indices else "None"
            )

            max_impact_score = float(overlapping_audio["impact_score"].max())
            max_vocal_distress = float(overlapping_audio["vocal_distress_score"].max())

            if "rms" in overlapping_audio.columns:
                rms_values = pd.to_numeric(overlapping_audio["rms"], errors="coerce").dropna()
                max_audio_rms = float(rms_values.max()) if not rms_values.empty else 0.0
            else:
                max_audio_rms = 0.0

            audio_categories = " | ".join(
                overlapping_audio["top_categories"].dropna().astype(str).unique()
            )
            audio_impact_categories = (
                " | ".join(impact_rows["top_categories"].dropna().astype(str).unique())
                if not impact_rows.empty else "None"
            )

        aligned_events.append({
            "event_id": event_number,
            **video_event,
            "overlapping_audio_frame_count": len(overlapping_audio),
            "audio_impact_candidate_count": audio_candidate_count,
            "audio_high_energy_count": high_energy_count,
            "max_audio_impact_score": max_impact_score,
            "max_vocal_distress_score": max_vocal_distress,
            "max_audio_rms": max_audio_rms,
            "audio_frame_indices": audio_frame_indices,
            "audio_impact_frame_indices": audio_impact_frame_indices,
            "audio_high_energy_frame_indices": audio_high_energy_frame_indices,
            "audio_categories": audio_categories,
            "audio_impact_categories": audio_impact_categories,
            "audio_support_present": audio_candidate_count > 0,
            "distress_detected": distress_detected,
            "distress_score": distress_score,
            "distress_keywords": distress_keywords,
            "distress_transcript": distress_transcript,
        })

    if not aligned_events and (distress_detected or (not audio_df.empty and audio_df["impact_candidate"].any())):
        impact_rows = audio_df[audio_df["impact_candidate"]]
        start_t = float(impact_rows["start_time"].min()) if not impact_rows.empty else 0.0
        end_t = float(impact_rows["end_time"].max()) if not impact_rows.empty else float(audio_df["end_time"].max() if not audio_df.empty else 1.0)
        aligned_events.append({
            "event_id": 1,
            "track_id": 0,
            "video_start_time": round(start_t, 3),
            "video_end_time": round(end_t, 3),
            "video_duration": round(max(0.0, end_t - start_t), 3),
            "video_observation_count": 0,
            "max_torso_angle": 0.0,
            "mean_torso_angle": 0.0,
            "transition_observation_count": 0,
            "fall_like_observation_count": 0,
            "horizontal_observation_count": 0,
            "rapid_fall_recovery": False,
            "post_fall_inactivity_seconds": 0.0,
            "prolonged_inactivity": False,
            "overlapping_audio_frame_count": len(audio_df),
            "audio_impact_candidate_count": len(impact_rows),
            "audio_high_energy_count": int(audio_df["high_energy"].sum()) if not audio_df.empty else 0,
            "max_audio_impact_score": float(audio_df["impact_score"].max()) if not audio_df.empty else 0.0,
            "max_vocal_distress_score": float(audio_df["vocal_distress_score"].max()) if not audio_df.empty else 0.0,
            "max_audio_rms": float(audio_df["rms"].max()) if not audio_df.empty and "rms" in audio_df.columns else 0.0,
            "audio_frame_indices": ", ".join(map(str, impact_rows["frame_index"].astype(int).tolist())) if not impact_rows.empty else "None",
            "audio_impact_frame_indices": ", ".join(map(str, impact_rows["frame_index"].astype(int).tolist())) if not impact_rows.empty else "None",
            "audio_high_energy_frame_indices": "None",
            "audio_categories": " | ".join(impact_rows["top_categories"].dropna().astype(str).unique()) if not impact_rows.empty else "None",
            "audio_impact_categories": " | ".join(impact_rows["top_categories"].dropna().astype(str).unique()) if not impact_rows.empty else "None",
            "audio_support_present": len(impact_rows) > 0,
            "distress_detected": distress_detected,
            "distress_score": distress_score,
            "distress_keywords": distress_keywords,
            "distress_transcript": distress_transcript,
        })

    return aligned_events


def run_alignment(video_csv: Path, audio_csv: Path, output_csv: Path, speech_json: Path = None) -> pd.DataFrame:
    video_df = pd.read_csv(video_csv) if Path(video_csv).exists() else pd.DataFrame()
    audio_df = pd.read_csv(audio_csv) if Path(audio_csv).exists() else pd.DataFrame()

    if not video_df.empty:
        validate_columns(
            video_df,
            ["frame", "timestamp", "track_id", "transition_candidate", "fall_like_event", "horizontal_posture", "torso_angle"],
            Path(video_csv).name,
        )
    if not audio_df.empty:
        validate_columns(
            audio_df,
            ["frame_index", "start_time", "end_time", "impact_score", "impact_candidate", "high_energy", "top_categories"],
            Path(audio_csv).name,
        )

    speech_data = {}
    if speech_json is not None and Path(speech_json).exists():
        speech_data = json.loads(Path(speech_json).read_text(encoding="utf-8"))

    frame_duration = estimate_frame_duration(video_df) if not video_df.empty else (1.0 / DEFAULT_VIDEO_FPS)
    video_candidates = prepare_video_candidates(video_df)
    video_events = group_video_events(video_candidates, frame_duration)
    audio_windows = prepare_audio_windows(audio_df)
    aligned_events = align_video_with_audio(video_events, audio_windows, speech_data=speech_data)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(aligned_events, columns=ALIGNED_EVENT_COLUMNS)
    result_df.to_csv(output_csv, index=False)
    print(f"Timestamp alignment complete ({len(result_df)} events) -> {output_csv}")
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Align video pose events with audio & speech windows.")
    parser.add_argument("--video-csv", required=True)
    parser.add_argument("--audio-csv", required=True)
    parser.add_argument("--speech-json", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_alignment(
        video_csv=Path(args.video_csv),
        audio_csv=Path(args.audio_csv),
        output_csv=Path(args.output),
        speech_json=Path(args.speech_json) if args.speech_json else None,
    )


if __name__ == "__main__":
    main()
