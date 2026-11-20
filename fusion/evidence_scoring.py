import argparse
from pathlib import Path
import pandas as pd

VIDEO_WEIGHT = 0.65
AUDIO_WEIGHT = 0.35

MIN_VIDEO_OBSERVATIONS = 10
MIN_VIDEO_DURATION = 0.40
IMPACT_REFERENCE = 0.30
SUSTAINED_INACTIVITY_THRESHOLD_SECONDS = 3.0


def safe_number(value, default=0.0):
    try:
        number = float(value)
        return default if pd.isna(number) else number
    except (TypeError, ValueError):
        return default


def safe_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def safe_ratio(numerator, denominator):
    return (numerator / denominator) if denominator > 0 else 0.0


def calculate_video_score(row):
    duration = safe_number(row.get("video_duration"))
    observation_count = safe_number(row.get("video_observation_count"))
    transition_count = safe_number(row.get("transition_observation_count"))
    horizontal_count = safe_number(row.get("horizontal_observation_count"))
    prolonged_inactivity = safe_bool(row.get("prolonged_inactivity", False))
    rapid_recovery = safe_bool(row.get("rapid_fall_recovery", False))

    if observation_count == 0:
        return 0.0

    duration_score = min(duration / 0.75, 1.0) * 25.0
    observation_score = min(observation_count / 15.0, 1.0) * 20.0
    transition_score = safe_ratio(transition_count, observation_count) * 20.0
    horizontal_score = safe_ratio(horizontal_count, observation_count) * 25.0
    inactivity_bonus = 10.0 if prolonged_inactivity else 0.0

    score = duration_score + observation_score + transition_score + horizontal_score + inactivity_bonus

    if not rapid_recovery and (observation_count < MIN_VIDEO_OBSERVATIONS or duration < MIN_VIDEO_DURATION):
        score = min(score, 25.0)
    elif rapid_recovery:
        score = max(score, 62.0)

    return round(min(score, 100.0), 2)


def calculate_audio_score(row):
    impact_score = safe_number(row.get("max_audio_impact_score"))
    vocal_distress = safe_number(row.get("max_vocal_distress_score"))
    impact_candidate = safe_bool(row.get("audio_support_present", False))
    high_energy_count = safe_number(row.get("audio_high_energy_count"))
    distress_detected = safe_bool(row.get("distress_detected", False))
    distress_score = safe_number(row.get("distress_score"))

    impact_component = min(impact_score / IMPACT_REFERENCE, 1.0) * 30.0
    candidate_component = 25.0 if impact_candidate else 0.0
    energy_component = 15.0 if high_energy_count > 0 else 0.0
    speech_distress_component = (30.0 * distress_score) if distress_detected else (15.0 * min(vocal_distress / 0.2, 1.0))

    score = impact_component + candidate_component + energy_component + speech_distress_component
    return round(max(0.0, min(score, 100.0)), 2)


def classify_event(row):
    video_score = safe_number(row["video_score"])
    audio_score = safe_number(row["audio_score"])
    video_duration = safe_number(row.get("video_duration"))
    video_observations = safe_number(row.get("video_observation_count"))
    inactivity_seconds = safe_number(row.get("post_fall_inactivity_seconds"))
    rapid_recovery = safe_bool(row.get("rapid_fall_recovery", False))
    audio_support = safe_bool(row.get("audio_support_present", False))
    distress_detected = safe_bool(row.get("distress_detected", False))
    has_audio_evidence = audio_support or distress_detected or audio_score >= 40.0

    if video_observations == 0 and has_audio_evidence:
        return "AUDIO_ONLY_CANDIDATE"

    if not rapid_recovery and (video_observations < MIN_VIDEO_OBSERVATIONS or video_duration < MIN_VIDEO_DURATION):
        if has_audio_evidence:
            return "INSUFFICIENT_VIDEO_EVIDENCE"
        return "INSUFFICIENT_EVIDENCE"

    # Strictly requires BOTH Video AND Audio evidence
    if video_score >= 60.0 and has_audio_evidence:
        return "POSSIBLE_MULTIMODAL_FALL_CANDIDATE"

    # Silent floor fall: Video pose + >=3.0s motionless floor inactivity (no audio)
    if video_score >= 75.0 and inactivity_seconds >= SUSTAINED_INACTIVITY_THRESHOLD_SECONDS:
        return "VIDEO_INACTIVITY_FALL_CANDIDATE"

    if rapid_recovery and video_score >= 60.0:
        return "VIDEO_ONLY_CANDIDATE"

    if video_score >= 60.0 and not has_audio_evidence and inactivity_seconds < SUSTAINED_INACTIVITY_THRESHOLD_SECONDS:
        return "NO_FALL_EVIDENCE"

    if video_score >= 60.0:
        return "VIDEO_ONLY_CANDIDATE"

    if has_audio_evidence:
        return "AUDIO_ONLY_CANDIDATE"

    return "NO_FALL_EVIDENCE"


def run_scoring(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    if not Path(input_csv).exists():
        raise FileNotFoundError(f"Aligned CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        for col, dtype in [
            ("video_score", "float64"),
            ("audio_score", "float64"),
            ("combined_score", "float64"),
            ("classification", "object"),
            ("video_only_fall", "bool"),
            ("audio_only_fall", "bool"),
            ("multimodal_fall", "bool"),
        ]:
            df[col] = pd.Series(dtype=dtype)
        df.to_csv(output_csv, index=False)
        return df

    df["video_score"] = df.apply(calculate_video_score, axis=1)
    df["audio_score"] = df.apply(calculate_audio_score, axis=1)
    df["combined_score"] = (df["video_score"] * VIDEO_WEIGHT + df["audio_score"] * AUDIO_WEIGHT).round(2)
    df["classification"] = df.apply(classify_event, axis=1)

    df["video_only_fall"] = df["video_score"] >= 60.0
    df["audio_only_fall"] = df.apply(
        lambda r: safe_bool(r.get("audio_support_present", False)) or safe_bool(r.get("distress_detected", False)),
        axis=1,
    )
    df["multimodal_fall"] = (
        df["classification"].isin({
            "POSSIBLE_MULTIMODAL_FALL_CANDIDATE",
            "VIDEO_INACTIVITY_FALL_CANDIDATE",
            "AUDIO_ONLY_CANDIDATE",
        })
        & (df["combined_score"] >= 50.0)
    )

    df.to_csv(output_csv, index=False)
    print(f"Evidence scoring complete ({len(df)} events) -> {output_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Score aligned video and audio events.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_scoring(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
