from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from backend.database import Base


def current_utc_time():
    return datetime.now(timezone.utc)


class Incident(Base):
    """Stores each event produced by the multimodal fusion pipeline."""

    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    event_key = Column(String(255), unique=True, nullable=False, index=True)

    # Resident & source context
    resident_name = Column(String(120), nullable=False, default="Mr. John")
    room_location = Column(String(120), nullable=False, default="Room 204")
    source_id = Column(String(255), nullable=False, index=True)
    source_video = Column(String(500), nullable=True)
    source_csv = Column(String(500), nullable=False)

    # Event information
    event_id = Column(Integer, nullable=False)
    track_id = Column(Integer, nullable=True)
    video_start_time = Column(Float, nullable=True)
    video_end_time = Column(Float, nullable=True)
    video_duration = Column(Float, nullable=True)
    post_fall_inactivity_seconds = Column(Float, nullable=True, default=0.0)
    prolonged_inactivity = Column(Boolean, nullable=False, default=False)

    # Evidence scores & modality flags
    video_score = Column(Float, nullable=True)
    audio_score = Column(Float, nullable=True)
    combined_score = Column(Float, nullable=True)
    video_only_fall = Column(Boolean, nullable=False, default=False)
    audio_only_fall = Column(Boolean, nullable=False, default=False)
    multimodal_fall = Column(Boolean, nullable=False, default=False)

    # Audio STT & VLM reasoning
    distress_detected = Column(Boolean, nullable=False, default=False)
    distress_transcript = Column(Text, nullable=True)
    vlm_activity_type = Column(String(100), nullable=True)
    vlm_summary = Column(Text, nullable=True)
    temporal_timeline = Column(Text, nullable=True)

    # Fusion result & caregiver verification
    classification = Column(String(100), nullable=False)
    review_priority = Column(String(100), nullable=True)
    alert_level = Column(String(30), nullable=False, default="NONE")
    incident_status = Column(String(30), nullable=False, default="NEW")

    verification_status = Column(String(40), nullable=False, default="PENDING")
    medical_assistance_required = Column(Boolean, nullable=False, default=False)
    verified_by = Column(String(120), nullable=True)
    verified_at = Column(DateTime(timezone=True), nullable=True)
    action_taken = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=current_utc_time, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=current_utc_time, onupdate=current_utc_time, nullable=False)

    alert = relationship("Alert", back_populates="incident", uselist=False, cascade="all, delete-orphan")
    tasks = relationship("FollowUpTask", back_populates="incident", cascade="all, delete-orphan")


class Alert(Base):
    """Stores caregiver alerts and multi-tier escalation state."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, ForeignKey("incidents.id", ondelete="CASCADE"), unique=True, nullable=False)

    alert_type = Column(String(50), nullable=False)
    severity = Column(String(30), nullable=False)
    message = Column(Text, nullable=False)
    status = Column(String(30), nullable=False, default="OPEN")

    # 3-Tier Escalation Ladder (1: Assigned Caregiver, 2: Senior Caregiver, 3: Registered Nurse / ERT)
    escalation_level = Column(Integer, nullable=False, default=1)
    assigned_role = Column(String(80), nullable=False, default="Assigned Caregiver")
    assigned_to = Column(String(120), nullable=False, default="Caregiver on Duty")
    escalation_history = Column(Text, nullable=True)
    escalated_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=current_utc_time, nullable=False)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    incident = relationship("Incident", back_populates="alert")


class FollowUpTask(Base):
    """Stores post-fall clinical/caregiver follow-up tasks."""

    __tablename__ = "follow_up_tasks"

    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False)

    title = Column(String(200), nullable=False)
    assigned_to = Column(String(120), nullable=False, default="Registered Nurse")
    priority = Column(String(30), nullable=False, default="HIGH")
    status = Column(String(30), nullable=False, default="PENDING")
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=current_utc_time, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    incident = relationship("Incident", back_populates="tasks")


Index("ix_incidents_classification", Incident.classification)
Index("ix_incidents_status", Incident.incident_status)
Index("ix_alerts_status", Alert.status)
Index("ix_tasks_status", FollowUpTask.status)
