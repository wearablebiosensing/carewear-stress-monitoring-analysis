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

def label_ppg_data(ppg_df, intervals_df):
    """
    Assign session labels and integer mappings to PPG samples.
    Same logic as HR labeling, just applied to PPG rows.
    """
    ppg_df["label"] = "None"

    for _, row in intervals_df.iterrows():
        mask = (
            (ppg_df["timestamp"] >= row["start_ts"]) &
            (ppg_df["timestamp"] <= row["end_ts"])
        )
        ppg_df.loc[mask, "label"] = row["session"]

    ppg_df["activity_int_merged"] = (
        ppg_df["label"].map(ACTIVITY_MAP).fillna(-1).astype(int)
    )

    return ppg_df

def process_participant(participant_path, output_dir):
    participant_id = os.path.basename(participant_path)
    event_file = os.path.join(participant_path, "Event.csv")
    ppg_file = os.path.join(participant_path, "GalaxyWatch", "PPG.csv")

    if not os.path.exists(event_file) or not os.path.exists(ppg_file):
        print(f"⚠️ {participant_id}: Missing files — skipping.")
        return

    print(f"→ Processing {participant_id}")

    events_df = pd.read_csv(event_file)
    ppg_df = pd.read_csv(ppg_file)

    # Convert timestamps to numeric
    events_df["timestamp"] = pd.to_numeric(events_df["timestamp"], errors="coerce")
    ppg_df["timestamp"] = pd.to_numeric(ppg_df["timestamp"], errors="coerce")

    # Drop bad rows
    events_df.dropna(subset=["timestamp"], inplace=True)
    ppg_df.dropna(subset=["timestamp"], inplace=True)

    # Ensure required event columns exist
    required_event_cols = {"timestamp", "session", "status"}
    if not required_event_cols.issubset(events_df.columns):
        print(f"⚠️ {participant_id}: Event.csv missing required columns {required_event_cols} — skipping.")
        return

    # Ensure PPG columns exist
    required_ppg_cols = {"timestamp", "ppg"}
    if not required_ppg_cols.issubset(ppg_df.columns):
        print(f"⚠️ {participant_id}: PPG.csv missing required columns {required_ppg_cols} — skipping.")
        return

    intervals_df = build_event_intervals(events_df)
    if intervals_df.empty:
        print(f"⚠️ {participant_id}: No valid ENTER/EXIT pairs found.")
        return

    ppg_labeled = label_ppg_data(ppg_df, intervals_df)

    ppg_labeled["datetime"] = pd.to_datetime(
        ppg_labeled["timestamp"], unit="ms", errors="coerce"
    )

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"{participant_id}_GalaxyWatch_PPG_merged_with_events.csv"
    )
    ppg_labeled.to_csv(output_file, index=False)

    print(f"  ✓ Saved → {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Merge GalaxyPPG PPG data with Event labels and Int Mappings."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to GalaxyPPG/Dataset folder"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Directory to save merged PPG files"
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
            process_participant(str(participant_path), str(out_dir))
        else:
            print(f"⚠️ {participant_path.name}: folder not found — skipping.")

    print("\n🎉 PPG processing complete!")

if __name__ == "__main__":
    main()