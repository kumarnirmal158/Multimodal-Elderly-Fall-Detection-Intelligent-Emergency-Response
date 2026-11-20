import argparse
import json
from pathlib import Path
import pandas as pd

from fusion.vlm_analyzer import interpret_event_with_vlm

PRIMARY_CANDIDATE_CLASSES = {
    "POSSIBLE_MULTIMODAL_FALL_CANDIDATE",
    "VIDEO_INACTIVITY_FALL_CANDIDATE",
    "VIDEO_ONLY_CANDIDATE",
    "VIDEO_CANDIDATE_AUDIO_AMBIGUOUS",
    "AUDIO_ONLY_CANDIDATE",
}


def parse_audio_frame_indices(value):
    if pd.isna(value):
        return set()
    frame_indices = set()
    for item in str(value).split(","):
        item = item.strip()
        if not item or item.lower() in {"none", "nan", "null"}:
            continue
        numeric_value = float(item)
        if not numeric_value.is_integer():
            raise ValueError(f"Frame index must be a whole number: {item}")
        frame_indices.add(int(numeric_value))
    return frame_indices


def find_shared_frames(frame_column):
    frame_sets = frame_column.apply(parse_audio_frame_indices)
    frame_event_counts = {}
    for frame_set in frame_sets:
        for frame_index in frame_set:
            frame_event_counts[frame_index] = frame_event_counts.get(frame_index, 0) + 1
    return {idx for idx, count in frame_event_counts.items() if count > 1}


def add_shared_audio_information(df):
    df = df.copy()
    df["_audio_frame_set"] = df["audio_frame_indices"].apply(parse_audio_frame_indices)
    if "audio_impact_frame_indices" in df.columns:
        df["_audio_impact_frame_set"] = df["audio_impact_frame_indices"].apply(parse_audio_frame_indices)
    else:
        df["_audio_impact_frame_set"] = [set() for _ in range(len(df))]

    shared_audio_frames = find_shared_frames(df["audio_frame_indices"])
    shared_impact_frames = find_shared_frames(
        df["audio_impact_frame_indices"]
        if "audio_impact_frame_indices" in df.columns
        else pd.Series(["None"] * len(df), index=df.index)
    )

    df["shared_audio_frame_indices"] = df["_audio_frame_set"].apply(
        lambda fs: ", ".join(str(i) for i in sorted(fs & shared_audio_frames))
    )
    df["shared_audio_evidence"] = df["shared_audio_frame_indices"].apply(lambda v: bool(v.strip()))
    df["shared_audio_impact_frame_indices"] = df["_audio_impact_frame_set"].apply(
        lambda fs: ", ".join(str(i) for i in sorted(fs & shared_impact_frames))
    )
    df["shared_audio_impact_evidence"] = df["shared_audio_impact_frame_indices"].apply(lambda v: bool(v.strip()))

    return df.drop(columns=["_audio_frame_set", "_audio_impact_frame_set"])


def build_temporal_timeline(row) -> list[dict]:
    start_t = float(row.get("video_start_time") or 0.0)
    end_t = float(row.get("video_end_time") or start_t)
    inactivity = float(row.get("post_fall_inactivity_seconds") or 0.0)
    audio_support = str(row.get("audio_support_present", False)).lower() == "true"
    distress = str(row.get("distress_detected", False)).lower() == "true"
    keywords = str(row.get("distress_keywords") or "None")

    timeline = [
        {
            "phase": "BEFORE",
            "timestamp": round(max(0.0, start_t - 0.5), 2),
            "observation": "Upright/pre-event baseline movement observed.",
        },
        {
            "phase": "DURING",
            "timestamp": round(start_t, 2),
            "observation": (
                f"Sudden posture transition detected (max torso angle {float(row.get('max_torso_angle') or 0.0):.1f}°)."
                if int(row.get("video_observation_count") or 0) > 0
                else "Off-camera acoustic event detected."
            ),
        },
    ]
    if audio_support:
        timeline.append({
            "phase": "DURING_AUDIO",
            "timestamp": round((start_t + end_t) / 2.0, 2),
            "observation": f"Impact-like acoustic energy detected ({row.get('audio_impact_categories', 'impact')}).",
        })
    if distress:
        timeline.append({
            "phase": "POST_DISTRESS",
            "timestamp": round(end_t, 2),
            "observation": f"Distress speech detected (keywords: {keywords}).",
        })
    timeline.append({
        "phase": "AFTER",
        "timestamp": round(end_t, 2),
        "observation": (
            f"Prolonged floor-level inactivity ({inactivity:.1f}s)."
            if inactivity >= 2.0
            else (
                "Rapid recovery to upright posture."
                if str(row.get("rapid_fall_recovery", False)).lower() == "true"
                else f"Post-event observation window ended (inactivity: {inactivity:.1f}s)."
            )
        ),
    })
    return timeline


def assign_review_priority(row):
    classification = row["classification"]
    if classification in {"POSSIBLE_MULTIMODAL_FALL_CANDIDATE", "VIDEO_INACTIVITY_FALL_CANDIDATE", "AUDIO_ONLY_CANDIDATE"}:
        return "HIGH"
    if classification in {"VIDEO_ONLY_CANDIDATE", "VIDEO_CANDIDATE_AUDIO_AMBIGUOUS", "INSUFFICIENT_VIDEO_EVIDENCE"}:
        return "REVIEW_AUDIO_CONTEXT"
    return "LOW"


def run_fusion(input_csv: Path, output_dir: Path, video_path: Path = None) -> dict:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    final_events = add_shared_audio_information(df)
    final_events["review_priority"] = final_events.apply(assign_review_priority, axis=1)
    final_events["final_event_status"] = final_events["classification"].apply(
        lambda c: "CANDIDATE_REQUIRES_REVIEW" if c in PRIMARY_CANDIDATE_CLASSES else "INSUFFICIENT_EVIDENCE"
    )

    vlm_summaries, vlm_activities, vlm_providers, timelines = [], [], [], []

    for _, row in final_events.iterrows():
        timeline = build_temporal_timeline(row)
        timelines.append(json.dumps(timeline))
        vlm_res = interpret_event_with_vlm(Path(video_path) if video_path else input_csv, row.to_dict())
        vlm_summaries.append(vlm_res.get("incident_summary", ""))
        vlm_activities.append(vlm_res.get("activity_type", "ambiguous"))
        vlm_providers.append(vlm_res.get("vlm_provider", "heuristic-fallback"))

    final_events["vlm_activity_type"] = vlm_activities
    final_events["vlm_summary"] = vlm_summaries
    final_events["vlm_provider"] = vlm_providers
    final_events["temporal_timeline"] = timelines

    event_output_path = output_dir / "final_fusion_events.csv"
    summary_output_path = output_dir / "incident_summary.json"
    report_output_path = output_dir / "incident_report.txt"

    final_events.to_csv(event_output_path, index=False)

    summary = {
        "source_file": input_csv.name,
        "total_video_candidate_events": int(len(final_events)),
        "primary_candidate_count": int(final_events["classification"].isin(PRIMARY_CANDIDATE_CLASSES).sum()) if not final_events.empty else 0,
        "overall_assessment": (
            "NO_FALL_CANDIDATE_DETECTED"
            if final_events.empty
            else (
                "POSSIBLE_MULTIMODAL_FALL_CANDIDATE"
                if "POSSIBLE_MULTIMODAL_FALL_CANDIDATE" in set(final_events["classification"])
                else str(final_events["classification"].iloc[0])
            )
        ),
        "events": final_events.to_dict(orient="records"),
    }
    summary_output_path.write_text(json.dumps(summary, indent=4, default=str), encoding="utf-8")

    lines = [
        "ELDERLY FALL DETECTION - MULTIMODAL INCIDENT REPORT",
        "=" * 58,
        f"Overall assessment: {summary['overall_assessment']}",
        f"Candidate events  : {summary['total_video_candidate_events']}",
        "",
    ]
    for ev in summary["events"]:
        lines.extend([
            f"Event {ev.get('event_id')} | {ev.get('classification')} | Priority: {ev.get('review_priority')}",
            f"  Time: {ev.get('video_start_time')}s - {ev.get('video_end_time')}s | Inactivity: {ev.get('post_fall_inactivity_seconds')}s",
            f"  Scores -> Video: {ev.get('video_score')} | Audio: {ev.get('audio_score')} | Combined: {ev.get('combined_score')}",
            f"  VLM ({ev.get('vlm_activity_type')}): {ev.get('vlm_summary')}",
            "",
        ])
    report_output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Multimodal fusion complete -> {event_output_path}")
    return {
        "event_output_path": event_output_path,
        "summary_output_path": summary_output_path,
        "report_output_path": report_output_path,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Run multimodal fusion + VLM interpretation.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video", default=None)
    args = parser.parse_args()
    run_fusion(Path(args.input), Path(args.output_dir), video_path=Path(args.video) if args.video else None)


if __name__ == "__main__":
    main()
