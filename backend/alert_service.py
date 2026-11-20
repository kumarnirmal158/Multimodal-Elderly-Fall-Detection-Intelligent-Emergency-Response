import json
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from backend.models import Alert, Incident

ESCALATION_LADDER = {
    1: ("Assigned Caregiver", "Caregiver on Duty"),
    2: ("Senior Caregiver", "Senior Caregiver Lead"),
    3: ("Nurse / Emergency Response Team", "Registered Nurse / ERT"),
}


def current_utc_time():
    return datetime.now(timezone.utc)


def build_alert_message(incident: Incident) -> str:
    prefix = (
        f"Urgent: Possible fall detected for {incident.resident_name} in {incident.room_location}."
        if incident.alert_level == "HIGH"
        else f"Review requested for {incident.resident_name} in {incident.room_location}."
    )
    details = []
    if incident.source_video:
        details.append(f"Source: {incident.source_video}")
    if incident.video_start_time is not None:
        details.append(f"Time: {incident.video_start_time:.2f}s")
    details.append(f"Classification: {incident.classification}")
    return f"{prefix} {' | '.join(details)}"


def generate_alerts(db: Session) -> dict:
    eligible_incidents = (
        db.query(Incident)
        .filter(Incident.alert_level.in_(["HIGH", "REVIEW"]))
        .all()
    )

    created = 0
    updated = 0
    already_exists = 0

    try:
        for incident in eligible_incidents:
            existing_alert = db.query(Alert).filter(Alert.incident_id == incident.id).first()
            severity = "HIGH" if incident.alert_level == "HIGH" else "REVIEW"
            message = build_alert_message(incident)

            if existing_alert:
                already_exists += 1
                if existing_alert.status == "OPEN":
                    existing_alert.severity = severity
                    existing_alert.message = message
                    existing_alert.alert_type = "POSSIBLE_FALL_REVIEW"
                    updated += 1
                continue

            role, assignee = ESCALATION_LADDER[1]
            initial_history = [
                {
                    "level": 1,
                    "role": role,
                    "assigned_to": assignee,
                    "timestamp": current_utc_time().isoformat(),
                    "note": "Initial alert dispatched (simulated notification).",
                }
            ]
            alert = Alert(
                incident_id=incident.id,
                alert_type="POSSIBLE_FALL_REVIEW",
                severity=severity,
                message=message,
                status="OPEN",
                escalation_level=1,
                assigned_role=role,
                assigned_to=assignee,
                escalation_history=json.dumps(initial_history),
            )
            db.add(alert)
            created += 1

        db.commit()
    except Exception:
        db.rollback()
        raise

    return {
        "eligible_incidents": len(eligible_incidents),
        "created": created,
        "updated": updated,
        "already_exists": already_exists,
    }


def run_escalation_check(
    db: Session,
    level2_threshold_seconds: int = 30,
    level3_threshold_seconds: int = 60,
    force_step: bool = False,
) -> dict:
    """
    Escalate unacknowledged OPEN alerts through the 3-level ladder:
    Level 1 (Assigned Caregiver) -> Level 2 (Senior Caregiver) -> Level 3 (Nurse / ERT).
    """
    now = current_utc_time()
    open_alerts = db.query(Alert).filter(Alert.status == "OPEN").all()
    escalated = []

    try:
        for alert in open_alerts:
            created = alert.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            elapsed = (now - created).total_seconds()

            target_level = alert.escalation_level
            if force_step:
                target_level = min(3, alert.escalation_level + 1)
            else:
                if elapsed >= level3_threshold_seconds:
                    target_level = max(target_level, 3)
                elif elapsed >= level2_threshold_seconds:
                    target_level = max(target_level, 2)

            if target_level > alert.escalation_level:
                role, assignee = ESCALATION_LADDER[target_level]
                history = json.loads(alert.escalation_history or "[]")
                history.append({
                    "level": target_level,
                    "role": role,
                    "assigned_to": assignee,
                    "timestamp": now.isoformat(),
                    "note": (
                        "Simulated escalation triggered manually for demo."
                        if force_step
                        else f"Unacknowledged after {int(elapsed)}s; escalated to {role}."
                    ),
                })
                alert.escalation_level = target_level
                alert.assigned_role = role
                alert.assigned_to = assignee
                alert.escalated_at = now
                alert.escalation_history = json.dumps(history)
                escalated.append({"alert_id": alert.id, "new_level": target_level, "assigned_role": role})

        db.commit()
    except Exception:
        db.rollback()
        raise

    return {"checked": len(open_alerts), "escalated_count": len(escalated), "escalated": escalated}


def get_all_alerts(db: Session):
    return db.query(Alert).order_by(Alert.created_at.desc()).all()


def get_open_alerts(db: Session):
    return db.query(Alert).filter(Alert.status != "RESOLVED").order_by(Alert.created_at.desc()).all()


def acknowledge_alert(db: Session, alert_id: int):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if alert is None:
        return None
    if alert.status != "OPEN":
        return alert

    try:
        alert.status = "ACKNOWLEDGED"
        alert.acknowledged_at = current_utc_time()
        alert.incident.incident_status = "ACKNOWLEDGED"
        db.commit()
        db.refresh(alert)
        return alert
    except Exception:
        db.rollback()
        raise


def resolve_alert(db: Session, alert_id: int):
    """
    Resolve an alert ONLY if caregiver verification has been recorded.
    Prevents silent closure without acknowledgement & verification (Sub-Case 2).
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if alert is None:
        return None
    if alert.status == "RESOLVED":
        return alert

    if alert.incident.verification_status == "PENDING":
        raise ValueError(
            "Cannot silently close alert: caregiver verification "
            "(TRUE_FALL, FALSE_ALERT, or UNCERTAIN) is required first."
        )

    try:
        if alert.acknowledged_at is None:
            alert.acknowledged_at = current_utc_time()
        alert.status = "RESOLVED"
        alert.resolved_at = current_utc_time()
        alert.incident.incident_status = "RESOLVED"
        db.commit()
        db.refresh(alert)
        return alert
    except Exception:
        db.rollback()
        raise
