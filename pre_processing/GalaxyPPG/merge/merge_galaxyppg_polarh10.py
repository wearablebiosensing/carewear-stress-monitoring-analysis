import os
import pandas as pd
import argparse
from pathlib import Path

# Mapping dictionary for GalaxyPPG activities
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

def label_data(df, intervals_df, ts_col):
    """
    Assign session labels and integer mappings to samples.
    """
    df["label"] = "None"
    
    for _, row in intervals_df.iterrows():
        mask = (
            (df[ts_col] >= row["start_ts"]) &
            (df[ts_col] <= row["end_ts"])
        )
        df.loc[mask, "label"] = row["session"]

    # Map the labels to integers; fill unmapped/None with -1
    df["activity_int_merged"] = df["label"].map(ACTIVITY_MAP).fillna(-1).astype(int)

    return df

def process_participant(participant_path, output_dir, sensor_type):
    participant_id = os.path.basename(participant_path)
    event_file = os.path.join(participant_path, "Event.csv")
    
    # PolarH10 data is in a subfolder "PolarH10"
    sensor_filename = f"{sensor_type.upper()}.csv"
    sensor_file = os.path.join(participant_path, "PolarH10", sensor_filename)

    if not os.path.exists(event_file) or not os.path.exists(sensor_file):
        print(f"⚠️ {participant_id}: Missing {sensor_filename} or Event.csv — skipping.")
        return

    print(f"→ Processing {participant_id} ({sensor_type})")

    events_df = pd.read_csv(event_file)
    sensor_df = pd.read_csv(sensor_file)

    # Convert timestamps to numeric
    events_df["timestamp"] = pd.to_numeric(events_df["timestamp"], errors="coerce")
    
    # PolarH10 uses 'phoneTimestamp' or 'timestamp'
    ts_col = None
    for col in ["phoneTimestamp", "timestamp"]:
        if col in sensor_df.columns:
            ts_col = col
            break
            
    if ts_col is None:
        print(f"⚠️ {participant_id}: No timestamp column found in {sensor_filename} — skipping.")
        return

    sensor_df[ts_col] = pd.to_numeric(sensor_df[ts_col], errors="coerce")
    
    # Correction: PolarH10 is in UTC+0900, Event.csv is in UTC.
    # Subtract 9 hours (32,400,000 ms) to align.
    print(f"  ℹ Correcting {ts_col} (UTC+0900 -> UTC) with -9 hour offset")
    sensor_df[ts_col] = sensor_df[ts_col] - (9 * 3600 * 1000)
    
    events_df.dropna(subset=["timestamp"], inplace=True)
    sensor_df.dropna(subset=[ts_col], inplace=True)

    # Build intervals and apply labels
    intervals_df = build_event_intervals(events_df)
    if intervals_df.empty:
        print(f"⚠️ {participant_id}: No valid ENTER/EXIT pairs found.")
        return

    sensor_labeled = label_data(sensor_df, intervals_df, ts_col)

    # Human-readable datetime (assuming milliseconds like Event.csv)
    sensor_labeled["datetime"] = pd.to_datetime(sensor_labeled[ts_col], unit="ms", errors="coerce")

    # Define sensor-specific output directory
    sensor_out_dir = os.path.join(output_dir, "PolarH10", sensor_type.upper())
    os.makedirs(sensor_out_dir, exist_ok=True)
    
    output_file = os.path.join(
        sensor_out_dir, 
        f"{participant_id}_PolarH10_{sensor_type.upper()}_merged_with_events.csv"
    )
    sensor_labeled.to_csv(output_file, index=False)
    print(f"  ✓ Saved → PolarH10/{sensor_type.upper()}/{os.path.basename(output_file)}")

def main():
    parser = argparse.ArgumentParser(description="Merge PolarH10 belt data with Event labels and Int Mappings.")
    parser.add_argument("--data", required=True, help="Path to GalaxyPPG/Dataset folder")
    parser.add_argument("--out", required=True, help="Directory to save merged PolarH10 files")
    parser.add_argument("--sensor", default="HR", choices=["ACC", "HR", "ECG", "IBI"], help="Sensor type (default: HR)")
    parser.add_argument("--pids", nargs="+", default=None, help="Optional participant IDs, e.g. --pids P02 P03")

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

    print(f"\n🎉 PolarH10 {args.sensor} processing complete!")

if __name__ == "__main__":
    main()
