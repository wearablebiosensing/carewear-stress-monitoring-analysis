import os
import pandas as pd
import numpy as np
import argparse

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

def label_hr_data(hr_df, intervals_df):
    """
    Assign session labels and integer mappings to HR samples.
    """
    # Initialize with 'None' for string labels and -1 for integer labels
    hr_df["label"] = "None"
    
    for _, row in intervals_df.iterrows():
        mask = (
            (hr_df["timestamp"] >= row["start_ts"]) &
            (hr_df["timestamp"] <= row["end_ts"])
        )
        hr_df.loc[mask, "label"] = row["session"]

    # Map the labels to integers; fill unmapped/None with -1
    hr_df["activity_int_merged"] = hr_df["label"].map(ACTIVITY_MAP).fillna(-1).astype(int)

    return hr_df

def process_participant(participant_path, output_dir):
    participant_id = os.path.basename(participant_path)
    event_file = os.path.join(participant_path, "Event.csv")
    hr_file = os.path.join(participant_path, "GalaxyWatch", "HR.csv")

    if not os.path.exists(event_file) or not os.path.exists(hr_file):
        print(f"⚠️ {participant_id}: Missing files — skipping.")
        return

    print(f"→ Processing {participant_id}")

    events_df = pd.read_csv(event_file)
    hr_df = pd.read_csv(hr_file)

    # Convert to numeric timestamps
    events_df["timestamp"] = pd.to_numeric(events_df["timestamp"], errors="coerce")
    hr_df["timestamp"] = pd.to_numeric(hr_df["timestamp"], errors="coerce")
    events_df.dropna(subset=["timestamp"], inplace=True)
    hr_df.dropna(subset=["timestamp"], inplace=True)

    # Build intervals and apply labels
    intervals_df = build_event_intervals(events_df)
    if intervals_df.empty:
        print(f"⚠️ {participant_id}: No valid ENTER/EXIT pairs found.")
        return

    hr_labeled = label_hr_data(hr_df, intervals_df)

    # Human-readable datetime
    hr_labeled["datetime"] = pd.to_datetime(hr_labeled["timestamp"], unit="ms", errors="coerce")

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{participant_id}_GalaxyWatch_HR_merged_with_events.csv")
    hr_labeled.to_csv(output_file, index=False)
    print(f"  ✓ Saved → {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Merge GalaxyPPG HR data with Event labels and Int Mappings.")
    parser.add_argument("--dataset_root", required=True, help="Path to GalaxyPPG/Dataset folder")
    parser.add_argument("--output_dir", required=True, help="Directory to save merged HR files")

    args = parser.parse_args()
    if not os.path.exists(args.dataset_root):
        raise ValueError("Dataset root directory does not exist.")

    participant_folders = sorted([
        os.path.join(args.dataset_root, d)
        for d in os.listdir(args.dataset_root)
        if d.startswith("P") and os.path.isdir(os.path.join(args.dataset_root, d))
    ])

    for participant_path in participant_folders:
        process_participant(participant_path, args.output_dir)

    print("\n🎉 Processing complete!")

if __name__ == "__main__":
    main()