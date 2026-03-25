import os
import pandas as pd
import argparse
from pathlib import Path

# Keep EXACTLY the same activity map style as the HR script
ACTIVITY_MAP = {
    'adaptation': 1,
    'baseline': 2,
    'tsst-prep': 3,
    'tsst-speech': 4,
    'meditation-1': 5,
    'screen-reading': 6,
    'ssst-prep': 7,
    'ssst-sing': 8,
    'meditation-2': 9,
    'keyboard-typing': 10,
    'rest-1': 11,
    'mobile-typing': 12,
    'rest-2': 13,
    'standing': 14,
    'rest-3': 15,
    'walking': 16,
    'rest-4': 17,
    'jogging': 18,
    'rest-5': 19,
    'running': 20
}

def build_event_intervals(events_df):
    """
    Convert ENTER/EXIT events into labeled time intervals.
    Keeps raw session names exactly like the HR script.
    """
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)
    intervals = []
    active_sessions = {}

    for _, row in events_df.iterrows():
        ts = row["timestamp"]
        session = row["session"]
        status = row["status"]

        if status == "ENTER":
            active_sessions[session] = ts
        elif status == "EXIT":
            if session in active_sessions:
                start_ts = active_sessions.pop(session)
                intervals.append({
                    "session": session,
                    "start_ts": start_ts,
                    "end_ts": ts
                })

    return pd.DataFrame(intervals)

def label_data(df, intervals_df):
    """
    Assign session labels and integer mappings to samples.
    """
    df["label"] = "None"

    for _, row in intervals_df.iterrows():
        # E4 data uses microsecond timestamps in this paper's dataset
        mask = (
            (df["timestamp"] >= row["start_ts"]) &
            (df["timestamp"] <= row["end_ts"])
        )
        df.loc[mask, "label"] = row["session"]

    df["activity_int_merged"] = (
        df["label"].map(ACTIVITY_MAP).fillna(-1).astype(int)
    )

    return df

def process_participant(participant_path, output_dir, sensor_type):

    participant_id = os.path.basename(participant_path)
    event_file = os.path.join(participant_path, "Event.csv")
    
    # Map sensor type to filename
    sensor_filename = f"{sensor_type.upper()}.csv"
    # User's actual folder is "E4"
    sensor_file = os.path.join(participant_path, "E4", sensor_filename)

    if not os.path.exists(event_file) or not os.path.exists(sensor_file):
        print(f"⚠️ {participant_id}: Missing {sensor_filename} or Event.csv — skipping.")
        return

    print(f"→ Processing {participant_id} ({sensor_type})")

    events_df = pd.read_csv(event_file)
    sensor_df = pd.read_csv(sensor_file)

    # Convert timestamps to numeric
    events_df["timestamp"] = pd.to_numeric(events_df["timestamp"], errors="coerce")
    sensor_df["timestamp"] = pd.to_numeric(sensor_df["timestamp"], errors="coerce")

    # Normalize Event timestamps to match E4's microsecond precision if needed
    # Event.csv is usually ms (13 digits), E4 is us (16 digits)
    if not events_df.empty and not sensor_df.empty:
        event_ts_sample = events_df["timestamp"].dropna().iloc[0]
        sensor_ts_sample = sensor_df["timestamp"].dropna().iloc[0]
        
        # If event is ms and sensor is us, scale event by 1000
        if event_ts_sample < 1e14 and sensor_ts_sample > 1e15:
            print(f"  ℹ Normalizing Event.csv timestamps (ms -> us) to match E4")
            events_df["timestamp"] = events_df["timestamp"] * 1000

    # Drop bad rows
    events_df.dropna(subset=["timestamp"], inplace=True)
    sensor_df.dropna(subset=["timestamp"], inplace=True)

    # Ensure required event columns exist
    required_event_cols = {"timestamp", "session", "status"}
    if not required_event_cols.issubset(events_df.columns):
        print(f"⚠️ {participant_id}: Event.csv missing required columns {required_event_cols} — skipping.")
        return

    # Check for timestamp column in sensor data
    if "timestamp" not in sensor_df.columns:
        print(f"⚠️ {participant_id}: {sensor_filename} missing 'timestamp' column — skipping.")
        return

    intervals_df = build_event_intervals(events_df)
    if intervals_df.empty:
        print(f"⚠️ {participant_id}: No valid ENTER/EXIT pairs found.")
        return

    sensor_labeled = label_data(sensor_df, intervals_df)

    # E4 timestamps are microseconds in this dataset (according to the paper)
    sensor_labeled["datetime"] = pd.to_datetime(
        sensor_labeled["timestamp"], unit="us", errors="coerce"
    )

    # Define sensor-specific output directory
    sensor_out_dir = os.path.join(output_dir, sensor_type.upper())
    os.makedirs(sensor_out_dir, exist_ok=True)
    
    output_file = os.path.join(
        sensor_out_dir,
        f"{participant_id}_EmpaticaE4_{sensor_type.upper()}_merged_with_events.csv"
    )
    sensor_labeled.to_csv(output_file, index=False)

    print(f"  ✓ Saved → {sensor_type.upper()}/{os.path.basename(output_file)}")

def main():
    parser = argparse.ArgumentParser(
        description="Merge Empatica E4 data with Event labels and Int Mappings."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset folder (containing P01, P02...)"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Directory to save merged E4 files"
    )
    parser.add_argument(
        "--sensor",
        default="HR",
        choices=["ACC", "HR", "BVP", "TEMP", "IBI"],
        help="Sensor type to process (default: HR)"
    )
    parser.add_argument(
        "--pids",
        nargs="+",
        default=None,
        help="Optional participant IDs, e.g. --pids P02 P03"
    )

    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)

    if not data_dir.exists():
        raise ValueError("Dataset root directory does not exist.")

    if args.pids:
        participant_folders = [data_dir / pid for pid in args.pids]
    else:
        participant_folders = sorted([
            p for p in data_dir.iterdir()
            if p.is_dir() and p.name.startswith("P")
        ])

    for participant_path in participant_folders:
        if participant_path.exists():
            process_participant(str(participant_path), str(out_dir), args.sensor)
        else:
            print(f"⚠️ {participant_path.name}: folder not found — skipping.")

    print(f"\n🎉 Empatica E4 {args.sensor} processing complete!")

if __name__ == "__main__":
    main()
