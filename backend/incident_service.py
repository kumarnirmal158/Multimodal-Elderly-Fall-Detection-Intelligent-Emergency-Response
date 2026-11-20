from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from backend.models import FollowUpTask, Incident

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HIGH_PRIORITY_CLASSES = {
    "POSSIBLE_MULTIMODAL_FALL_CANDIDATE",
    "VIDEO_INACTIVITY_FALL_CANDIDATE",
    "AUDIO_ONLY_CANDIDATE",
}

REVIEW_CLASSES = {
    "VIDEO_ONLY_CANDIDATE",
    "VIDEO_CANDIDATE_AUDIO_AMBIGUOUS",
}

IGNORED_NOISE_CLASSES = {
    "INSUFFICIENT_VIDEO_EVIDENCE",
    "INSUFFICIENT_EVIDENCE",
    "NO_FALL_EVIDENCE",
}

VALID_VERIFICATION_STATUSES = {"TRUE_FALL", "FALSE_ALERT", "UNCERTAIN"}


def current_utc_time():
    return datetime.now(timezone.utc)


def safe_value(value):
    if pd.isna(value):
        return None
    return value


def safe_int(value):
    value = safe_value(value)
    return int(value) if value is not None else None


def safe_float(value):
    value = safe_value(value)
    return float(value) if value is not None else None


def safe_bool(value):
    value = safe_value(value)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def determine_alert_level(classification):
    if classification in HIGH_PRIORITY_CLASSES:
        return "HIGH"
    if classification in REVIEW_CLASSES:
        return "REVIEW"
    return "NONE"


def import_fusion_events(
    db: Session,
    csv_path,
    source_video=None,
    source_id=None,
    resident_name="Mr. John",
    room_location="Room 204",
):
    csv_path = Path(csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Fusion CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        return {"total_events": 0, "inserted": 0, "updated": 0, "skipped_noise": 0}

    missing_columns = {"event_id", "classification"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Fusion CSV is missing required columns: {sorted(missing_columns)}")

    if source_id is None:
        source_id = csv_path.parent.name

    inserted = 0
    updated = 0
    skipped_noise = 0

    try:
        for _, row in df.iterrows():
            event_id = safe_int(row["event_id"])
            classification_value = safe_value(row["classification"])
            if event_id is None or classification_value is None:
                continue

            classification = str(classification_value).strip()
            if classification in IGNORED_NOISE_CLASSES:
                skipped_noise += 1
                continue

            event_key = f"{source_id}:{event_id}"

            incident_data = {
                "resident_name": resident_name,
                "room_location": room_location,
                "source_id": str(source_id),
                "source_video": str(source_video) if source_video else None,
                "source_csv": str(csv_path),
                "event_id": event_id,
                "track_id": safe_int(row.get("track_id")),
                "video_start_time": safe_float(row.get("video_start_time")),
                "video_end_time": safe_float(row.get("video_end_time")),
                "video_duration": safe_float(row.get("video_duration")),
                "post_fall_inactivity_seconds": safe_float(row.get("post_fall_inactivity_seconds")) or 0.0,
                "prolonged_inactivity": safe_bool(row.get("prolonged_inactivity")),
                "video_score": safe_float(row.get("video_score")),
                "audio_score": safe_float(row.get("audio_score")),
                "combined_score": safe_float(row.get("combined_score")),
                "video_only_fall": safe_bool(row.get("video_only_fall")),
                "audio_only_fall": safe_bool(row.get("audio_only_fall")),
                "multimodal_fall": safe_bool(row.get("multimodal_fall")),
                "distress_detected": safe_bool(row.get("distress_detected")),
                "distress_transcript": (
                    str(row["distress_transcript"])
                    if safe_value(row.get("distress_transcript")) is not None
                    else None
                ),
                "vlm_activity_type": (
                    str(row["vlm_activity_type"])
                    if safe_value(row.get("vlm_activity_type")) is not None
                    else None
                ),
                "vlm_summary": (
                    str(row["vlm_summary"])
                    if safe_value(row.get("vlm_summary")) is not None
                    else None
                ),
                "temporal_timeline": (
                    str(row["temporal_timeline"])
                    if safe_value(row.get("temporal_timeline")) is not None
                    else None
                ),
                "classification": classification,
                "review_priority": (
                    str(row["review_priority"])
                    if safe_value(row.get("review_priority")) is not None
                    else None
                ),
                "alert_level": determine_alert_level(classification),
            }

            existing = db.query(Incident).filter(Incident.event_key == event_key).first()
            if existing:
                for key, value in incident_data.items():
                    setattr(existing, key, value)
                updated += 1
            else:
                incident = Incident(
                    event_key=event_key,
                    incident_status="NEW",
                    verification_status="PENDING",
                    **incident_data,
                )
                db.add(incident)
                inserted += 1

        db.commit()
    except Exception:
        db.rollback()
        raise

    return {
        "total_events": len(df),
        "inserted": inserted,
        "updated": updated,
        "skipped_noise": skipped_noise,
    }


def verify_incident(
    db: Session,
    incident_id: int,
    verification_status: str,
    verified_by: str = "Caregiver",
    action_taken: str = None,
    notes: str = None,
    medical_assistance_required: bool = False,
):
    verification_status = verification_status.strip().upper()
    if verification_status not in VALID_VERIFICATION_STATUSES:
        raise ValueError(
            f"Invalid verification_status. Must be one of {sorted(VALID_VERIFICATION_STATUSES)}"
        )

    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if incident is None:
        return None

    try:
        incident.verification_status = verification_status
        incident.verified_by = verified_by
        incident.verified_at = current_utc_time()
        incident.medical_assistance_required = bool(medical_assistance_required)
        if action_taken is not None:
            incident.action_taken = action_taken
        if notes is not None:
            incident.notes = notes
        incident.incident_status = "VERIFIED"

        if verification_status == "TRUE_FALL" or medical_assistance_required:
            existing_task = (
                db.query(FollowUpTask)
                .filter(FollowUpTask.incident_id == incident.id)
                .first()
            )
            if existing_task is None:
                db.add(FollowUpTask(
                    incident_id=incident.id,
                    title=f"Post-Fall Resident Assessment ({incident.resident_name}, {incident.room_location})",
                    assigned_to="Registered Nurse",
                    priority="HIGH",
                    status="PENDING",
                    notes=notes or "Assess resident vitals, mobility, and document follow-up.",
                ))

        db.commit()
        db.refresh(incident)
        return incident
    except Exception:
        db.rollback()
        raise


def get_all_incidents(db: Session):
    return db.query(Incident).order_by(Incident.created_at.desc()).all()


def get_incident_by_id(db: Session, incident_id: int):
    return db.query(Incident).filter(Incident.id == incident_id).first()


def get_all_tasks(db: Session):
    return db.query(FollowUpTask).order_by(FollowUpTask.created_at.desc()).all()


def complete_task(db: Session, task_id: int, notes: str = None):
    task = db.query(FollowUpTask).filter(FollowUpTask.id == task_id).first()
    if task is None:
        return None
    try:
        task.status = "COMPLETED"
        task.completed_at = current_utc_time()
        if notes:
            task.notes = notes
        db.commit()
        db.refresh(task)
        return task
    except Exception:
        db.rollback()
        raise
