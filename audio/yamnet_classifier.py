import argparse
from pathlib import Path
import wave
import csv

import numpy as np
import resampy
import tensorflow_hub as hub


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_FOLDER = PROJECT_ROOT / "data" / "audio" / "test_samples"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "annotations"
OUTPUT_CSV = OUTPUT_FOLDER / "yamnet_audio_features.csv"

TARGET_SAMPLE_RATE = 16000
MODEL_URL = "https://tfhub.dev/google/yamnet/1"
FRAME_DURATION = 0.96
FRAME_HOP = 0.48
TOP_N = 5

IMPORTANT_CLASSES = {
    "crying_score": ["Crying, sobbing", "Baby cry, infant cry"],
    "cough_score": ["Cough"],
    "breathing_score": ["Breathing"],
    "scream_score": ["Screaming", "Shout"],
    "footsteps_score": ["Walk, footsteps"],
    "knock_score": ["Knock"],
    "glass_score": ["Glass", "Breaking", "Shatter"],
    "thud_score": ["Thump, thud"],
    "crash_score": ["Smash, crash", "Bang"],
    "speech_score": ["Speech"],
}

_YAMNET_MODEL = None
_CLASS_NAMES = None
_CLASS_INDICES = None


def load_yamnet():
    global _YAMNET_MODEL, _CLASS_NAMES, _CLASS_INDICES
    if _YAMNET_MODEL is not None:
        return _YAMNET_MODEL, _CLASS_NAMES, _CLASS_INDICES

    print("\nLoading YAMNet model...")
    _YAMNET_MODEL = hub.load(MODEL_URL)
    class_map_path = _YAMNET_MODEL.class_map_path().numpy().decode("utf-8")

    with open(class_map_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        _CLASS_NAMES = [row["display_name"] for row in reader]

    _CLASS_INDICES = {name: index for index, name in enumerate(_CLASS_NAMES)}
    print("YAMNet loaded successfully!")
    return _YAMNET_MODEL, _CLASS_NAMES, _CLASS_INDICES


def load_audio(audio_path):
    with wave.open(str(audio_path), "rb") as audio_file:
        sample_rate = audio_file.getframerate()
        channels = audio_file.getnchannels()
        sample_width = audio_file.getsampwidth()
        audio_bytes = audio_file.readframes(audio_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM audio: {audio_path.name}")

    waveform = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    if channels > 1:
        waveform = waveform.reshape(-1, channels).mean(axis=1)
    waveform = waveform / 32768.0

    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = resampy.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)

    return waveform.astype(np.float32)


def calculate_audio_features(segment):
    rms = float(np.sqrt(np.mean(np.square(segment))))
    zero_crossings = np.count_nonzero(np.diff(np.signbit(segment)))
    zcr = float(zero_crossings / max(len(segment) - 1, 1))
    return rms, zcr


def get_important_scores(predictions, class_indices):
    result = {}
    for output_name, category_names in IMPORTANT_CLASSES.items():
        matching_indices = [
            class_indices[name] for name in category_names if name in class_indices
        ]
        if matching_indices:
            result[output_name] = float(np.max(predictions[matching_indices]))
        else:
            result[output_name] = 0.0
    return result


def process_audio(audio_path):
    yamnet_model, class_names, class_indices = load_yamnet()
    waveform = load_audio(audio_path)
    duration = len(waveform) / TARGET_SAMPLE_RATE

    scores, embeddings, spectrogram = yamnet_model(waveform)
    frame_scores = scores.numpy()

    frame_length = int(FRAME_DURATION * TARGET_SAMPLE_RATE)
    frame_hop = int(FRAME_HOP * TARGET_SAMPLE_RATE)

    records = []
    rms_values = []

    for frame_index, predictions in enumerate(frame_scores):
        start_sample = frame_index * frame_hop
        end_sample = start_sample + frame_length
        segment = waveform[start_sample:end_sample]
        if len(segment) < frame_length:
            segment = np.pad(segment, (0, frame_length - len(segment)))

        rms, zcr = calculate_audio_features(segment)
        rms_values.append(rms)

        top_indices = np.argsort(predictions)[::-1][:TOP_N]
        top_categories = [
            f"{class_names[index]} ({predictions[index]:.3f})"
            for index in top_indices
        ]
        important_scores = get_important_scores(predictions, class_indices)

        records.append({
            "filename": audio_path.name,
            "frame_index": frame_index,
            "start_time": round(frame_index * FRAME_HOP, 3),
            "end_time": round(min(frame_index * FRAME_HOP + FRAME_DURATION, duration), 3),
            "top_categories": " | ".join(top_categories),
            "top_score": float(np.max(predictions)),
            "rms": rms,
            "zcr": zcr,
            **important_scores,
        })

    rms_values = np.array(rms_values) if rms_values else np.array([0.0])
    mean_rms = float(np.mean(rms_values))
    std_rms = float(np.std(rms_values))
    energy_threshold = max(0.03, mean_rms + (1.5 * std_rms))

    for index, record in enumerate(records):
        previous_rms = rms_values[index - 1] if index > 0 else 0.0
        energy_ratio = (record["rms"] / previous_rms) if previous_rms > 1e-6 else 0.0
        high_energy = record["rms"] >= energy_threshold
        impact_score = max(
            record["glass_score"],
            record["thud_score"],
            record["crash_score"],
            record["knock_score"],
        )
        vocal_distress_score = max(
            record["scream_score"],
            record["crying_score"],
        )
        energy_anomaly = high_energy and energy_ratio >= 1.5
        impact_candidate = high_energy and impact_score >= 0.05

        record["energy_ratio"] = round(energy_ratio, 4)
        record["high_energy"] = high_energy
        record["impact_score"] = impact_score
        record["vocal_distress_score"] = vocal_distress_score
        record["energy_anomaly"] = energy_anomaly
        record["impact_candidate"] = impact_candidate

    return records


def main():
    parser = argparse.ArgumentParser(description="YAMNet audio feature extraction")
    parser.add_argument("--audio", default=None, help="Optional path to one WAV file.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.is_absolute():
            audio_path = PROJECT_ROOT / audio_path
        if not audio_path.exists():
            parser.error(f"Audio file not found: {audio_path}")
        audio_files = [audio_path]
    else:
        audio_files = sorted(AUDIO_FOLDER.glob("*.wav"))

    output_csv = Path(args.output) if args.output else OUTPUT_CSV
    if not output_csv.is_absolute():
        output_csv = PROJECT_ROOT / output_csv

    if not audio_files:
        print("No WAV files found.")
        return

    all_records = []
    for audio_path in audio_files:
        all_records.extend(process_audio(audio_path))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)

    print(f"YAMNet audio features saved -> {output_csv}")


if __name__ == "__main__":
    main()
