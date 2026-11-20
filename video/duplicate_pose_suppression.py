from pathlib import Path
import argparse
from itertools import combinations

import numpy as np
import pandas as pd


# Conservative duplicate-detection thresholds.
HIP_DISTANCE_THRESHOLD = 0.04
JOINT_DISTANCE_THRESHOLD = 0.06
IOU_THRESHOLD = 0.10
MIN_COMMON_LANDMARKS = 10
VISIBILITY_THRESHOLD = 0.5

AUDIT_COLUMNS = [
    "frame",
    "kept_track_id",
    "removed_track_id",
    "hip_distance",
    "joint_distance",
    "iou",
    "common_landmarks",
]

TRACK_AUDIT_COLUMNS = [
    "track_a",
    "track_b",
    "track_a_first_seen",
    "track_b_first_seen",
    "matching_frame_count",
    "matching_frames",
    "first_matching_frame",
    "last_matching_frame",
    "median_hip_distance",
    "median_joint_distance",
    "median_iou",
    "assessment",
]


def calculate_iou(a, b):
    """Calculate IoU using normalized bounding-box coordinates."""

    x1 = max(a["bbox_x_min"], b["bbox_x_min"])
    y1 = max(a["bbox_y_min"], b["bbox_y_min"])
    x2 = min(a["bbox_x_max"], b["bbox_x_max"])
    y2 = min(a["bbox_y_max"], b["bbox_y_max"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (
        (a["bbox_x_max"] - a["bbox_x_min"])
        * (a["bbox_y_max"] - a["bbox_y_min"])
    )

    area_b = (
        (b["bbox_x_max"] - b["bbox_x_min"])
        * (b["bbox_y_max"] - b["bbox_y_min"])
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def calculate_hip_distance(a, b):
    """Calculate distance between visible hip centres."""

    hip_points = []

    for row in (a, b):
        visible_hips = []

        for side in ("left", "right"):
            x_col = f"{side}_hip_x"
            y_col = f"{side}_hip_y"
            visibility_col = f"{side}_hip_visibility"

            if (
                x_col in row.index
                and y_col in row.index
                and visibility_col in row.index
                and row[visibility_col] >= VISIBILITY_THRESHOLD
            ):
                visible_hips.append([
                    row[x_col],
                    row[y_col]
                ])

        if not visible_hips:
            return float("inf")

        hip_points.append(np.mean(visible_hips, axis=0))

    return float(np.linalg.norm(hip_points[0] - hip_points[1]))


def calculate_joint_similarity(a, b, landmark_columns):
    """Compare corresponding landmarks visible in both detections."""

    distances = []

    for x_col, y_col, visibility_col in landmark_columns:
        if (
            a[visibility_col] >= VISIBILITY_THRESHOLD
            and b[visibility_col] >= VISIBILITY_THRESHOLD
        ):
            point_a = np.array([a[x_col], a[y_col]], dtype=float)
            point_b = np.array([b[x_col], b[y_col]], dtype=float)

            if np.all(np.isfinite(point_a)) and np.all(np.isfinite(point_b)):
                distances.append(np.linalg.norm(point_a - point_b))

    if len(distances) < MIN_COMMON_LANDMARKS:
        return float("inf"), len(distances)

    return float(np.mean(distances)), len(distances)


def get_landmark_columns(df):
    """Find landmark x/y/visibility columns automatically."""

    columns = []

    for x_col in df.columns:
        if not x_col.endswith("_x"):
            continue

        y_col = x_col[:-1] + "y"
        visibility_col = x_col[:-1] + "visibility"

        if y_col in df.columns and visibility_col in df.columns:
            columns.append((x_col, y_col, visibility_col))

    return columns


def get_track_quality(row):
    """Fallback quality score when track ages are equal."""

    return (
        float(row["visible_landmark_count"]),
        float(row["mean_visibility"]),
        float(row["bbox_width"] * row["bbox_height"])
    )


def suppress_duplicates(df):
    """Remove only highly similar simultaneous pose detections."""

    df = df.copy().reset_index(drop=True)

    required = [
        "frame", "timestamp", "track_id",
        "bbox_x_min", "bbox_x_max",
        "bbox_y_min", "bbox_y_max",
        "bbox_width", "bbox_height",
        "visible_landmark_count",
        "left_hip_x", "left_hip_y",
        "right_hip_x", "right_hip_y"
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    visibility_columns = [
        column for column in df.columns
        if column.endswith("_visibility")
    ]

    if not visibility_columns:
        raise ValueError("No landmark visibility columns found.")

    df["mean_visibility"] = df[visibility_columns].mean(axis=1)

    landmark_columns = get_landmark_columns(df)

    if not landmark_columns:
        raise ValueError("No landmark coordinate columns found.")

    first_seen = df.groupby("track_id")["timestamp"].min().to_dict()

    removed_indices = set()
    audit = []

    for frame, group in df.groupby("frame", sort=True):
        if len(group) < 2:
            continue

        for index_a, index_b in combinations(group.index.tolist(), 2):
            if index_a in removed_indices or index_b in removed_indices:
                continue

            a = df.loc[index_a]
            b = df.loc[index_b]

            if a["track_id"] == b["track_id"]:
                continue

            hip_distance = calculate_hip_distance(a, b)
            joint_distance, common_landmarks = (
                calculate_joint_similarity(a, b, landmark_columns)
            )
            iou = calculate_iou(a, b)

            is_duplicate = (
                hip_distance <= HIP_DISTANCE_THRESHOLD
                and joint_distance <= JOINT_DISTANCE_THRESHOLD
                and iou >= IOU_THRESHOLD
                and common_landmarks >= MIN_COMMON_LANDMARKS
            )

            if not is_duplicate:
                continue

            track_a = a["track_id"]
            track_b = b["track_id"]

            age_a = first_seen[track_a]
            age_b = first_seen[track_b]

            if age_a < age_b:
                keep_index, remove_index = index_a, index_b
            elif age_b < age_a:
                keep_index, remove_index = index_b, index_a
            elif get_track_quality(a) >= get_track_quality(b):
                keep_index, remove_index = index_a, index_b
            else:
                keep_index, remove_index = index_b, index_a

            removed_indices.add(remove_index)

            audit.append({
                "frame": int(frame),
                "kept_track_id": int(df.loc[keep_index, "track_id"]),
                "removed_track_id": int(df.loc[remove_index, "track_id"]),
                "hip_distance": hip_distance,
                "joint_distance": joint_distance,
                "iou": iou,
                "common_landmarks": common_landmarks
            })

    filtered = df.drop(index=list(removed_indices)).copy()
    filtered = filtered.drop(columns=["mean_visibility"])

    audit_df = pd.DataFrame(audit, columns=AUDIT_COLUMNS)

    return filtered, audit_df


def audit_track_duplicates(df):
    """Identify possible duplicate track pairs using simultaneous pose observations."""

    required = ["frame", "timestamp", "track_id"]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    landmark_columns = get_landmark_columns(df)

    first_seen = (
        df.groupby("track_id")["timestamp"]
        .min()
        .to_dict()
    )

    pair_evidence = {}

    for frame, group in df.groupby("frame", sort=True):
        if group["track_id"].nunique() < 2:
            continue

        for index_a, index_b in combinations(group.index.tolist(), 2):
            a = df.loc[index_a]
            b = df.loc[index_b]

            track_a = a["track_id"]
            track_b = b["track_id"]

            if track_a == track_b:
                continue

            hip_distance = calculate_hip_distance(a, b)
            joint_distance, common_landmarks = (
                calculate_joint_similarity(a, b, landmark_columns)
            )
            iou = calculate_iou(a, b)

            is_duplicate = (
                hip_distance <= HIP_DISTANCE_THRESHOLD
                and joint_distance <= JOINT_DISTANCE_THRESHOLD
                and iou >= IOU_THRESHOLD
                and common_landmarks >= MIN_COMMON_LANDMARKS
            )

            if not is_duplicate:
                continue

            pair = tuple(sorted((track_a, track_b)))

            pair_evidence.setdefault(pair, []).append({
                "frame": int(frame),
                "timestamp": float(a["timestamp"]),
                "hip_distance": hip_distance,
                "joint_distance": joint_distance,
                "iou": iou,
                "common_landmarks": common_landmarks
            })

    audit_rows = []

    for (track_a, track_b), matches in pair_evidence.items():
        matches = sorted(matches, key=lambda item: item["frame"])
        matching_frames = [item["frame"] for item in matches]

        audit_rows.append({
            "track_a": track_a,
            "track_b": track_b,
            "track_a_first_seen": first_seen[track_a],
            "track_b_first_seen": first_seen[track_b],
            "matching_frame_count": len(matches),
            "matching_frames": ", ".join(map(str, matching_frames)),
            "first_matching_frame": matching_frames[0],
            "last_matching_frame": matching_frames[-1],
            "median_hip_distance": float(np.median(
                [item["hip_distance"] for item in matches]
            )),
            "median_joint_distance": float(np.median(
                [item["joint_distance"] for item in matches]
            )),
            "median_iou": float(np.median(
                [item["iou"] for item in matches]
            )),
            "assessment": (
                "REVIEW_POSSIBLE_DUPLICATE_TRACK"
                if len(matches) >= 2
                else "INSUFFICIENT_REPEATED_EVIDENCE"
            )
        })

    return pd.DataFrame(audit_rows, columns=TRACK_AUDIT_COLUMNS)


def main():
    parser = argparse.ArgumentParser(
        description="Conservative duplicate-pose suppression"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    filtered, audit = suppress_duplicates(df)
    track_audit = audit_track_duplicates(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path = output_path.with_name(output_path.stem + "_audit.csv")
    track_audit_path = output_path.with_name(output_path.stem + "_track_audit.csv")

    filtered.to_csv(output_path, index=False)
    audit.to_csv(audit_path, index=False)
    track_audit.to_csv(track_audit_path, index=False)

    print("\nDUPLICATE POSE SUPPRESSION RESULTS")
    print("----------------------------------")
    print("Original observations:", len(df))
    print("Suppressed observations:", len(df) - len(filtered))
    print("Remaining observations:", len(filtered))


if __name__ == "__main__":
    main()
