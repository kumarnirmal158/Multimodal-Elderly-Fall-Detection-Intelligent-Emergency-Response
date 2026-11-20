import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.alert_service import (
    acknowledge_alert,
    generate_alerts,
    get_all_alerts,
    get_open_alerts,
    resolve_alert,
    run_escalation_check,
)
from backend.database import PROJECT_ROOT, create_tables, get_db, reset_database
from backend.incident_service import (
    complete_task,
    get_incident_by_id,
    get_all_tasks,
    import_fusion_events,
    verify_incident,
)
from backend.models import Alert, FollowUpTask, Incident
from evaluation.evaluator import evaluate_system
from orchestration.run_pipeline import run_pipeline

app = FastAPI(
    title="Multimodal Elderly Fall Detection & Emergency Response API",
    description=(
        "Prerecorded video/audio fall detection, VLM contextual reasoning, "
        "3-tier caregiver escalation, verification, follow-up tasks, and evaluation."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")


class VerifyIncidentRequest(BaseModel):
    verification_status: str
    verified_by: str = "Caregiver"
    action_taken: Optional[str] = None
    notes: Optional[str] = None
    medical_assistance_required: bool = False


class CompleteTaskRequest(BaseModel):
    notes: Optional[str] = None


@app.on_event("startup")
def startup_event():
    create_tables()


def serialize_incident(incident: Incident) -> dict:
    timeline = []
    if incident.temporal_timeline:
        try:
            timeline = json.loads(incident.temporal_timeline)
        except Exception:
            timeline = []

    return {
        "id": incident.id,
        "event_key": incident.event_key,
        "resident_name": incident.resident_name,
        "room_location": incident.room_location,
        "source_id": incident.source_id,
        "source_video": incident.source_video,
        "source_csv": incident.source_csv,
        "event_id": incident.event_id,
        "track_id": incident.track_id,
        "video_start_time": incident.video_start_time,
        "video_end_time": incident.video_end_time,
        "video_duration": incident.video_duration,
        "post_fall_inactivity_seconds": incident.post_fall_inactivity_seconds,
        "prolonged_inactivity": incident.prolonged_inactivity,
        "video_score": incident.video_score,
        "audio_score": incident.audio_score,
        "combined_score": incident.combined_score,
        "video_only_fall": incident.video_only_fall,
        "audio_only_fall": incident.audio_only_fall,
        "multimodal_fall": incident.multimodal_fall,
        "distress_detected": incident.distress_detected,
        "distress_transcript": incident.distress_transcript,
        "vlm_activity_type": incident.vlm_activity_type,
        "vlm_summary": incident.vlm_summary,
        "temporal_timeline": timeline,
        "classification": incident.classification,
        "review_priority": incident.review_priority,
        "alert_level": incident.alert_level,
        "incident_status": incident.incident_status,
        "verification_status": incident.verification_status,
        "medical_assistance_required": incident.medical_assistance_required,
        "verified_by": incident.verified_by,
        "verified_at": incident.verified_at.isoformat() if incident.verified_at else None,
        "action_taken": incident.action_taken,
        "notes": incident.notes,
        "created_at": incident.created_at.isoformat() if incident.created_at else None,
        "updated_at": incident.updated_at.isoformat() if incident.updated_at else None,
    }


def serialize_alert(alert: Alert) -> dict:
    history = []
    if alert.escalation_history:
        try:
            history = json.loads(alert.escalation_history)
        except Exception:
            history = []

    return {
        "id": alert.id,
        "incident_id": alert.incident_id,
        "alert_type": alert.alert_type,
        "severity": alert.severity,
        "message": alert.message,
        "status": alert.status,
        "escalation_level": alert.escalation_level,
        "assigned_role": alert.assigned_role,
        "assigned_to": alert.assigned_to,
        "escalation_history": history,
        "escalated_at": alert.escalated_at.isoformat() if alert.escalated_at else None,
        "created_at": alert.created_at.isoformat() if alert.created_at else None,
        "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
        "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
    }


def serialize_task(task: FollowUpTask) -> dict:
    return {
        "id": task.id,
        "incident_id": task.incident_id,
        "title": task.title,
        "assigned_to": task.assigned_to,
        "priority": task.priority,
        "status": task.status,
        "notes": task.notes,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }


@app.get("/")
def root():
    return {
        "message": "Multimodal Elderly Fall Detection Backend is running",
        "documentation": "/docs",
        "dashboard": "/dashboard",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "elderly-fall-detection-backend",
        "timestamp": datetime.now().astimezone().isoformat(),
    }


@app.post("/api/system/reset")
def reset_system_database():
    """Reset SQLite database tables for a clean-slate evaluation."""
    reset_database()
    return {"message": "Database reset successfully."}


# ---------------------------------------------------------
# PRERECORDED MEDIA UPLOAD & PIPELINE EXECUTION
# ---------------------------------------------------------

@app.post("/api/pipeline/upload")
async def upload_and_run_pipeline(
    video: UploadFile = File(...),
    audio: Optional[UploadFile] = File(default=None),
    resident_name: str = Form(default="Mr. John"),
    room_location: str = Form(default="Room 204"),
):
    uploads_dir = PROJECT_ROOT / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    video_dest = uploads_dir / Path(video.filename).name
    with video_dest.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    audio_dest = None
    if audio is not None and audio.filename:
        audio_dest = uploads_dir / Path(audio.filename).name
        with audio_dest.open("wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

    try:
        result = run_pipeline(
            video_path=video_dest,
            audio_path=audio_dest,
            resident_name=resident_name,
            room_location=room_location,
        )
        return {"message": "Pipeline completed successfully", "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------
# INCIDENT ENDPOINTS
# ---------------------------------------------------------

@app.get("/api/incidents")
def list_incidents(
    classification: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(Incident)
    if classification:
        query = query.filter(Incident.classification == classification)
    if status:
        query = query.filter(Incident.incident_status == status)
    incidents = query.order_by(Incident.created_at.desc()).all()
    return {"count": len(incidents), "incidents": [serialize_incident(i) for i in incidents]}


@app.post("/api/incidents/import")
def import_latest_fusion_events(
    csv_path: str,
    source_id: Optional[str] = None,
    source_video: Optional[str] = None,
    resident_name: str = "Mr. John",
    room_location: str = "Room 204",
    db: Session = Depends(get_db),
):
    try:
        import_result = import_fusion_events(
            db,
            csv_path=csv_path,
            source_video=source_video,
            source_id=source_id,
            resident_name=resident_name,
            room_location=room_location,
        )
        alert_result = generate_alerts(db)
        return {"message": "Fusion events imported", "import": import_result, "alerts": alert_result}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/incidents/{incident_id}")
def read_incident(incident_id: int, db: Session = Depends(get_db)):
    incident = get_incident_by_id(db, incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return serialize_incident(incident)


@app.post("/api/incidents/{incident_id}/verify")
def verify_incident_endpoint(
    incident_id: int,
    payload: VerifyIncidentRequest,
    db: Session = Depends(get_db),
):
    try:
        incident = verify_incident(
            db,
            incident_id=incident_id,
            verification_status=payload.verification_status,
            verified_by=payload.verified_by,
            action_taken=payload.action_taken,
            notes=payload.notes,
            medical_assistance_required=payload.medical_assistance_required,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")

    return {"message": "Incident verified", "incident": serialize_incident(incident)}


# ---------------------------------------------------------
# ALERT & ESCALATION ENDPOINTS
# ---------------------------------------------------------

@app.get("/api/alerts")
def list_alerts(db: Session = Depends(get_db)):
    alerts = get_all_alerts(db)
    return {"count": len(alerts), "alerts": [serialize_alert(a) for a in alerts]}


@app.get("/api/alerts/open")
def list_open_alerts(db: Session = Depends(get_db)):
    alerts = get_open_alerts(db)
    return {"count": len(alerts), "alerts": [serialize_alert(a) for a in alerts]}


@app.post("/api/alerts/escalate")
def escalate_alerts_endpoint(
    force_step: bool = Query(default=False),
    level2_seconds: int = Query(default=30),
    level3_seconds: int = Query(default=60),
    db: Session = Depends(get_db),
):
    result = run_escalation_check(
        db,
        level2_threshold_seconds=level2_seconds,
        level3_threshold_seconds=level3_seconds,
        force_step=force_step,
    )
    return {"message": "Escalation check completed", **result}


@app.post("/api/alerts/{alert_id}/acknowledge")
def acknowledge_alert_endpoint(alert_id: int, db: Session = Depends(get_db)):
    alert = acknowledge_alert(db, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert acknowledged", "alert": serialize_alert(alert)}


@app.post("/api/alerts/{alert_id}/resolve")
def resolve_alert_endpoint(alert_id: int, db: Session = Depends(get_db)):
    try:
        alert = resolve_alert(db, alert_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert resolved", "alert": serialize_alert(alert)}


# ---------------------------------------------------------
# FOLLOW-UP TASK & EVALUATION ENDPOINTS
# ---------------------------------------------------------

@app.get("/api/tasks")
def list_tasks(db: Session = Depends(get_db)):
    tasks = get_all_tasks(db)
    return {"count": len(tasks), "tasks": [serialize_task(t) for t in tasks]}


@app.post("/api/tasks/{task_id}/complete")
def complete_task_endpoint(
    task_id: int,
    payload: Optional[CompleteTaskRequest] = None,
    db: Session = Depends(get_db),
):
    task = complete_task(db, task_id, notes=payload.notes if payload else None)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"message": "Task completed", "task": serialize_task(task)}


@app.get("/api/evaluation")
def get_evaluation_metrics(db: Session = Depends(get_db)):
    return evaluate_system(db)
