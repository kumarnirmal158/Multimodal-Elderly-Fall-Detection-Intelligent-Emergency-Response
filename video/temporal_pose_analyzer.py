from pathlib import Path
import argparse
import math
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from video.duplicate_pose_suppression import suppress_duplicates


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/"
    "pose_landmarker_lite.task"
)

MAX_POSES = 3
VISIBILITY_THRESHOLD = 0.5
MAX_COMPARISON_GAP_SECONDS = 1.0

MAX_TRACK_DISTANCE = 0.20
MAX_TRACK_AGE_SECONDS = 1.0

UPRIGHT_ANGLE_THRESHOLD = 55.0
HORIZONTAL_ANGLE_THRESHOLD = 65.0

MIN_TRANSITION_ANGLE_CHANGE = 15.0
MAX_TRANSITION_GAP_SECONDS = 1.0
# Require a downward hip/center drop (normalized height) to separate
# standing-to-floor falls from floor-sitting/plank exercises
MIN_TRANSITION_HIP_DROP = 0.05

MIN_HORIZONTAL_DURATION_SECONDS = 0.5
MAX_HORIZONTAL_GAP_SECONDS = 0.25
MIN_HORIZONTAL_OBSERVATIONS = 10

MAX_RAPID_RECOVERY_SECONDS = 1.0
MIN_RAPID_HORIZONTAL_OBSERVATIONS = 2
MIN_RECOVERY_OBSERVATIONS = 2

INACTIVITY_DISPLACEMENT_THRESHOLD = 0.035
PROLONGED_INACTIVITY_SECONDS = 2.0

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def ensure_model_exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        print(f"Using existing model: {MODEL_PATH}")
        return

    print("Downloading MediaPipe Pose Landmarker Lite model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as error:
        raise RuntimeError(
            "Could not download the MediaPipe model. "
            "Check your internet connection and try again."
        ) from error
    print(f"Model downloaded: {MODEL_PATH}")


def resolve_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    current_location = Path.cwd() / path
    if current_location.exists():
        return current_location
    return PROJECT_ROOT / path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract MediaPipe pose features from a video.")
    parser.add_argument("--video", default="data/videos/fall_activity.mp4", help="Input video path.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    return parser.parse_args()


def get_landmark_arrays(pose_landmarks):
    coordinates, visibility, presence = [], [], []
    for landmark in pose_landmarks:
        coordinates.append([landmark.x, landmark.y, landmark.z])
        visibility.append(float(getattr(landmark, "visibility", 1.0)))
        presence_value = getattr(landmark, "presence", None)
        presence.append(np.nan if presence_value is None else float(presence_value))

    return (
        np.asarray(coordinates, dtype=float),
        np.asarray(visibility, dtype=float),
        np.asarray(presence, dtype=float),
    )


def calculate_torso_angle(coordinates, visibility, width, height):
    shoulder_indices = [i for i in [11, 12] if visibility[i] >= VISIBILITY_THRESHOLD]
    hip_indices = [i for i in [23, 24] if visibility[i] >= VISIBILITY_THRESHOLD]
    if not shoulder_indices or not hip_indices:
        return np.nan

    shoulder_center = np.mean(coordinates[shoulder_indices, :2], axis=0)
    hip_center = np.mean(coordinates[hip_indices, :2], axis=0)
    dx = (shoulder_center[0] - hip_center[0]) * width
    dy = (shoulder_center[1] - hip_center[1]) * height
    if math.hypot(dx, dy) < 1e-6:
        return np.nan

    return math.degrees(math.atan2(abs(dx), abs(dy)))


def calculate_person_center(coordinates, visibility):
    hip_indices = [23, 24]
    visible_hips = [i for i in hip_indices if visibility[i] >= VISIBILITY_THRESHOLD]
    if visible_hips:
        return np.mean(coordinates[visible_hips, :2], axis=0)

    visible_points = coordinates[visibility >= VISIBILITY_THRESHOLD, :2]
    if len(visible_points) == 0:
        return np.mean(coordinates[:, :2], axis=0)
    return np.mean(visible_points, axis=0)


def assign_track_ids(current_centers, active_tracks, next_track_id, timestamp):
    assignments = [None] * len(current_centers)
    available_tracks = {
        tid: track
        for tid, track in active_tracks.items()
        if (timestamp - track["timestamp"]) <= MAX_TRACK_AGE_SECONDS
    }

    possible_matches = []
    for person_index, center in enumerate(current_centers):
        for track_id, track in available_tracks.items():
            distance = float(np.linalg.norm(center - track["center"]))
            if distance <= MAX_TRACK_DISTANCE:
                possible_matches.append((distance, person_index, track_id))

    possible_matches.sort(key=lambda item: item[0])
    used_people, used_tracks = set(), set()

    for _, person_index, track_id in possible_matches:
        if person_index in used_people or track_id in used_tracks:
            continue
        assignments[person_index] = track_id
        used_people.add(person_index)
        used_tracks.add(track_id)

    for person_index in range(len(current_centers)):
        if assignments[person_index] is None:
            assignments[person_index] = next_track_id
            next_track_id += 1

    for person_index, track_id in enumerate(assignments):
        active_tracks[track_id] = {
            "center": current_centers[person_index],
            "timestamp": timestamp,
        }

    expired_tracks = [
        tid for tid, track in active_tracks.items()
        if timestamp - track["timestamp"] > MAX_TRACK_AGE_SECONDS
    ]
    for tid in expired_tracks:
        del active_tracks[tid]

    return assignments, next_track_id


def extract_person_features(coordinates, visibility, presence, width, height):
    angle = calculate_torso_angle(coordinates, visibility, width, height)
    visible_indices = np.where(visibility >= VISIBILITY_THRESHOLD)[0]

    if len(visible_indices) > 0:
        visible_points = coordinates[visible_indices, :2]
        x_min, x_max = float(np.min(visible_points[:, 0])), float(np.max(visible_points[:, 0]))
        y_min, y_max = float(np.min(visible_points[:, 1])), float(np.max(visible_points[:, 1]))
        box_width = max((x_max - x_min) * width, 0.0)
        box_height = max((y_max - y_min) * height, 0.0)
    else:
        x_min = x_max = y_min = y_max = box_width = box_height = np.nan

    bbox_ratio = (box_width / box_height) if (pd.notna(box_height) and box_height > 0) else np.nan
    center = calculate_person_center(coordinates, visibility)

    row = {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "bbox_x_min": x_min,
        "bbox_x_max": x_max,
        "bbox_y_min": y_min,
        "bbox_y_max": y_max,
        "bbox_width": box_width,
        "bbox_height": box_height,
        "bbox_ratio": bbox_ratio,
        "torso_angle": angle,
        "visible_landmark_count": len(visible_indices),
    }

    for index, name in enumerate(LANDMARK_NAMES):
        row[f"{name}_x"] = coordinates[index, 0]
        row[f"{name}_y"] = coordinates[index, 1]
        row[f"{name}_z"] = coordinates[index, 2]
        row[f"{name}_visibility"] = visibility[index]
        row[f"{name}_presence"] = presence[index]

    return row


def detect_temporal_fall_events(dataframe):
    dataframe = dataframe.copy()

    dataframe["angle_change"] = 0.0
    dataframe["ratio_change"] = 0.0
    dataframe["center_displacement"] = 0.0
    dataframe["posture_candidate"] = False
    dataframe["transition_candidate"] = False
    dataframe["horizontal_posture"] = False
    dataframe["fall_like_event"] = False
    dataframe["rapid_fall_recovery_candidate"] = False
    dataframe["post_fall_inactivity_seconds"] = 0.0
    dataframe["prolonged_inactivity"] = False
    dataframe["event_id"] = pd.NA

    next_event_id = 1

    for _, group in dataframe.groupby("track_id"):
        group = group.sort_values("timestamp")
        indices = group.index.tolist()
        if len(indices) < 2:
            continue

        transitions = []

        for position in range(1, len(indices)):
            previous_index = indices[position - 1]
            current_index = indices[position]
            time_gap = dataframe.at[current_index, "timestamp"] - dataframe.at[previous_index, "timestamp"]
            if time_gap <= 0:
                continue

            previous_angle = dataframe.at[previous_index, "torso_angle"]
            current_angle = dataframe.at[current_index, "torso_angle"]
            previous_ratio = dataframe.at[previous_index, "bbox_ratio"]
            current_ratio = dataframe.at[current_index, "bbox_ratio"]

            if time_gap <= MAX_COMPARISON_GAP_SECONDS:
                if pd.notna(previous_angle) and pd.notna(current_angle):
                    dataframe.at[current_index, "angle_change"] = current_angle - previous_angle
                if pd.notna(previous_ratio) and pd.notna(current_ratio):
                    dataframe.at[current_index, "ratio_change"] = current_ratio - previous_ratio

                prev_cx, prev_cy = dataframe.at[previous_index, "center_x"], dataframe.at[previous_index, "center_y"]
                curr_cx, curr_cy = dataframe.at[current_index, "center_x"], dataframe.at[current_index, "center_y"]
                if pd.notna(prev_cx) and pd.notna(curr_cx):
                    dataframe.at[current_index, "center_displacement"] = math.hypot(curr_cx - prev_cx, curr_cy - prev_cy)

            if pd.notna(current_angle) and current_angle >= HORIZONTAL_ANGLE_THRESHOLD:
                dataframe.at[current_index, "horizontal_posture"] = True
                dataframe.at[current_index, "posture_candidate"] = True

        for position, current_index in enumerate(indices):
            current_angle = dataframe.at[current_index, "torso_angle"]
            current_time = dataframe.at[current_index, "timestamp"]
            current_cy = dataframe.at[current_index, "center_y"]

            if pd.isna(current_angle) or current_angle < HORIZONTAL_ANGLE_THRESHOLD:
                continue

            for previous_position in range(position - 1, -1, -1):
                previous_index = indices[previous_position]
                previous_time = dataframe.at[previous_index, "timestamp"]
                previous_angle = dataframe.at[previous_index, "torso_angle"]
                previous_cy = dataframe.at[previous_index, "center_y"]

                if (current_time - previous_time) > MAX_TRANSITION_GAP_SECONDS:
                    break
                if pd.isna(previous_angle):
                    continue

                hip_drop = (current_cy - previous_cy) if (pd.notna(current_cy) and pd.notna(previous_cy)) else 0.0
                if (
                    previous_angle <= UPRIGHT_ANGLE_THRESHOLD
                    and (current_angle - previous_angle) >= MIN_TRANSITION_ANGLE_CHANGE
                    and hip_drop >= MIN_TRANSITION_HIP_DROP
                ):
                    transitions.append(current_index)
                    break

                if previous_angle >= HORIZONTAL_ANGLE_THRESHOLD:
                    break

        if transitions:
            dataframe.loc[transitions, "transition_candidate"] = True

        # 1. Sustained-horizontal fall detection + post-fall inactivity measurement
        for transition_index in transitions:
            start_position = indices.index(transition_index)

            horizontal_positions = []
            for position in range(start_position, len(indices)):
                current_index = indices[position]
                current_time = dataframe.at[current_index, "timestamp"]
                angle = dataframe.at[current_index, "torso_angle"]
                if pd.isna(angle):
                    continue

                if angle >= HORIZONTAL_ANGLE_THRESHOLD:
                    if horizontal_positions:
                        previous_index = indices[horizontal_positions[-1]]
                        if (current_time - dataframe.at[previous_index, "timestamp"]) > MAX_HORIZONTAL_GAP_SECONDS:
                            break
                    horizontal_positions.append(position)
                else:
                    break

            if len(horizontal_positions) < MIN_HORIZONTAL_OBSERVATIONS:
                continue

            first_horizontal = indices[horizontal_positions[0]]
            last_horizontal = indices[horizontal_positions[-1]]
            duration = dataframe.at[last_horizontal, "timestamp"] - dataframe.at[first_horizontal, "timestamp"]

            if duration >= MIN_HORIZONTAL_DURATION_SECONDS:
                event_indices = indices[start_position : horizontal_positions[-1] + 1]
                dataframe.loc[event_indices, "fall_like_event"] = True
                dataframe.loc[event_indices, "event_id"] = next_event_id

                inactivity_start_time = None
                max_inactivity = 0.0
                anchor_x = dataframe.at[first_horizontal, "center_x"]
                anchor_y = dataframe.at[first_horizontal, "center_y"]

                for pos in horizontal_positions:
                    idx = indices[pos]
                    cx, cy = dataframe.at[idx, "center_x"], dataframe.at[idx, "center_y"]
                    ts = dataframe.at[idx, "timestamp"]
                    if pd.notna(cx) and pd.notna(anchor_x):
                        if math.hypot(cx - anchor_x, cy - anchor_y) <= INACTIVITY_DISPLACEMENT_THRESHOLD:
                            if inactivity_start_time is None:
                                inactivity_start_time = ts
                            max_inactivity = max(max_inactivity, ts - inactivity_start_time)
                        else:
                            anchor_x, anchor_y = cx, cy
                            inactivity_start_time = ts

                dataframe.loc[event_indices, "post_fall_inactivity_seconds"] = round(max_inactivity, 3)
                if max_inactivity >= PROLONGED_INACTIVITY_SECONDS:
                    dataframe.loc[event_indices, "prolonged_inactivity"] = True

                next_event_id += 1

        # 2. Rapid fall-and-recovery candidate detection
        for transition_index in transitions:
            start_position = indices.index(transition_index)
            if dataframe.at[transition_index, "fall_like_event"]:
                continue

            start_time = dataframe.at[transition_index, "timestamp"]
            horizontal_positions = []
            recovery_positions = []
            recovery_position = None
            previous_observation_time = start_time

            for position in range(start_position + 1, len(indices)):
                current_index = indices[position]
                current_time = dataframe.at[current_index, "timestamp"]

                if (current_time - start_time) > MAX_RAPID_RECOVERY_SECONDS:
                    break
                if (current_time - previous_observation_time) > MAX_COMPARISON_GAP_SECONDS:
                    break

                angle = dataframe.at[current_index, "torso_angle"]
                if pd.isna(angle):
                    continue

                if angle >= HORIZONTAL_ANGLE_THRESHOLD:
                    horizontal_positions.append(position)
                    recovery_positions = []
                elif angle <= UPRIGHT_ANGLE_THRESHOLD:
                    if len(horizontal_positions) >= MIN_RAPID_HORIZONTAL_OBSERVATIONS:
                        if recovery_positions:
                            prev_rec_idx = indices[recovery_positions[-1]]
                            if (current_time - dataframe.at[prev_rec_idx, "timestamp"]) > MAX_HORIZONTAL_GAP_SECONDS:
                                recovery_positions = []
                        recovery_positions.append(position)
                        if len(recovery_positions) >= MIN_RECOVERY_OBSERVATIONS:
                            recovery_position = position
                            break
                else:
                    recovery_positions = []

                previous_observation_time = current_time

            if recovery_position is None:
                continue

            event_indices = indices[start_position : recovery_position + 1]
            dataframe.loc[event_indices, "fall_like_event"] = True
            dataframe.loc[event_indices, "rapid_fall_recovery_candidate"] = True
            dataframe.loc[event_indices, "event_id"] = next_event_id
            next_event_id += 1

    return dataframe


def main():
    args = parse_arguments()
    video_path = resolve_path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path = resolve_path(args.output) if args.output else (
        PROJECT_ROOT / "data" / "annotations" / f"{video_path.stem}_mediapipe_features.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError("Invalid video FPS.")

    ensure_model_exists()

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_poses=MAX_POSES,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    rows, active_tracks, next_track_id, frame_index = [], {}, 1, 0

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                height, width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(frame_index * 1000.0 / fps)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                frame_poses = result.pose_landmarks

                if frame_poses:
                    pose_data = []
                    for pose_landmarks in frame_poses:
                        coordinates, visibility, presence = get_landmark_arrays(pose_landmarks)
                        center = calculate_person_center(coordinates, visibility)
                        pose_data.append({
                            "coordinates": coordinates,
                            "visibility": visibility,
                            "presence": presence,
                            "center": center,
                        })

                    timestamp = frame_index / fps
                    current_centers = [item["center"] for item in pose_data]
                    track_ids, next_track_id = assign_track_ids(
                        current_centers, active_tracks, next_track_id, timestamp
                    )

                    for person_index, item in enumerate(pose_data):
                        person_features = extract_person_features(
                            item["coordinates"], item["visibility"], item["presence"], width, height
                        )
                        row = {
                            "frame": frame_index,
                            "timestamp": timestamp,
                            "track_id": track_ids[person_index],
                            "pose_index": person_index,
                        }
                        row.update(person_features)
                        rows.append(row)

                frame_index += 1
    finally:
        cap.release()

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        dataframe.to_csv(output_path, index=False)
        print("No poses were detected in this video.")
        return

    dataframe = dataframe.sort_values(["track_id", "frame"]).reset_index(drop=True)
    dataframe = detect_temporal_fall_events(dataframe)
    dataframe, duplicate_audit = suppress_duplicates(dataframe)

    audit_path = output_path.with_name(f"{output_path.stem}_duplicate_audit.csv")
    duplicate_audit.to_csv(audit_path, index=False)
    dataframe.to_csv(output_path, index=False)

    print(f"Video pose analysis complete -> {output_path}")
    print(f"Fall-like events detected: {int(dataframe['event_id'].nunique())}")


if __name__ == "__main__":
    main()
