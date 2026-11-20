import json
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from backend.models import Alert, Incident

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_FILE = PROJECT_ROOT / "evaluation" / "benchmark_dataset.json"
PIPELINE_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "pipeline"


def _compute_binary_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)

    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "total_samples": total,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
    }


def evaluate_system(db: Optional[Session] = None) -> dict:
    """
    Evaluate Video-Only, Audio-Only, and Multimodal performance.
    Uses caregiver-verified DB incidents when available, and also summarizes
    pipeline outputs matched against evaluation/benchmark_dataset.json.
    """
    benchmark = []
    if BENCHMARK_FILE.exists():
        benchmark = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))

    # 1. Live Database Operational & Verification Metrics
    live_summary = {
        "total_incidents": 0,
        "verified_incidents": 0,
        "true_falls_confirmed": 0,
        "false_alerts_confirmed": 0,
        "uncertain_incidents": 0,
        "avg_acknowledgement_seconds": None,
        "escalated_alerts_count": 0,
    }
    verified_comparison = None

    if db is not None:
        incidents = db.query(Incident).all()
        alerts = db.query(Alert).all()
        live_summary["total_incidents"] = len(incidents)
        live_summary["escalated_alerts_count"] = sum(
            1 for a in alerts if (a.escalation_level or 1) > 1
        )

        ack_times = []
        for a in alerts:
            if a.acknowledged_at and a.created_at:
                ack_times.append(abs((a.acknowledged_at - a.created_at).total_seconds()))
        if ack_times:
            live_summary["avg_acknowledgement_seconds"] = round(sum(ack_times) / len(ack_times), 2)

        verified = [i for i in incidents if i.verification_status in {"TRUE_FALL", "FALSE_ALERT"}]
        live_summary["verified_incidents"] = len(verified)
        live_summary["true_falls_confirmed"] = sum(1 for i in incidents if i.verification_status == "TRUE_FALL")
        live_summary["false_alerts_confirmed"] = sum(1 for i in incidents if i.verification_status == "FALSE_ALERT")
        live_summary["uncertain_incidents"] = sum(1 for i in incidents if i.verification_status == "UNCERTAIN")

        if verified:
            y_true = [i.verification_status == "TRUE_FALL" for i in verified]
            verified_comparison = {
                "video_only": _compute_binary_metrics(y_true, [bool(i.video_only_fall) for i in verified]),
                "audio_only": _compute_binary_metrics(y_true, [bool(i.audio_only_fall) for i in verified]),
                "multimodal": _compute_binary_metrics(y_true, [bool(i.multimodal_fall) for i in verified]),
            }

    # 2. Benchmark Dataset Evaluation (from pipeline outputs if run)
    dataset_rows = []
    if PIPELINE_OUTPUTS_DIR.exists() and benchmark:
        for item in benchmark:
            stem = Path(item["video_name"]).stem
            matching_dirs = sorted(PIPELINE_OUTPUTS_DIR.glob(f"{stem}*"))
            if not matching_dirs:
                continue
            csv_path = matching_dirs[-1] / "final_fusion_events.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            v_pred = bool(df["video_only_fall"].astype(bool).any()) if not df.empty and "video_only_fall" in df.columns else False
            a_pred = bool(df["audio_only_fall"].astype(bool).any()) if not df.empty and "audio_only_fall" in df.columns else False
            m_pred = bool(df["multimodal_fall"].astype(bool).any()) if not df.empty and "multimodal_fall" in df.columns else False
            dataset_rows.append({
                "video_name": item["video_name"],
                "ground_truth_fall": bool(item["ground_truth_fall"]),
                "video_only": v_pred,
                "audio_only": a_pred,
                "multimodal": m_pred,
            })

    benchmark_comparison = None
    if dataset_rows:
        y_true_b = [r["ground_truth_fall"] for r in dataset_rows]
        benchmark_comparison = {
            "evaluated_videos": len(dataset_rows),
            "video_only": _compute_binary_metrics(y_true_b, [r["video_only"] for r in dataset_rows]),
            "audio_only": _compute_binary_metrics(y_true_b, [r["audio_only"] for r in dataset_rows]),
            "multimodal": _compute_binary_metrics(y_true_b, [r["multimodal"] for r in dataset_rows]),
            "per_video": dataset_rows,
        }

    return {
        "live_summary": live_summary,
        "verified_comparison": verified_comparison,
        "benchmark_comparison": benchmark_comparison,
        "benchmark_dataset_count": len(benchmark),
    }
