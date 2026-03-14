import pandas as pd
import glob
import os
import argparse

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
root = "/Volumes/ss/Project_CareWear/DATASET/ss_drive"
INPUT_DIR = os.path.join(root, "2_Concat_File/hr")
OUTPUT_DIR = os.path.join(root, "4_merged_lables")
MANUAL_LABELS_DIR = os.path.join(root, "3_task_timeline/Task_Time_Line_Manual")
BELT_LABELS_DIR = os.path.join(root, "3_task_timeline/Task_Time_Line_Belt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# CANONICAL ACTIVITY NORMALIZATION
# -------------------------------------------------
def normalize_activity(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    mapping = {
        "Stationary_Bike_Legs": "stationary_Bike1",
        "Stationary_Bike_Hand": "stationary_Bike2",
        "Prepare_Speech": "prepare speech",
        "Give_Speech": "give speech",
        "Mental_Math": "mental math",
        "Rest_1": "rest1",
        "Rest_2": "rest2",
        "Rest_3": "rest3"
    }
    return mapping.get(x, x)

reverse_mapping = {
    "stationary_Bike1": "Stationary_Bike_Legs",
    "stationary_Bike2": "Stationary_Bike_Hand",
    "prepare speech": "Prepare_Speech",
    "give speech": "Give_Speech",
    "mental math": "Mental_Math",
    "rest1": "Rest_1",
    "rest2": "Rest_2",
    "rest3": "Rest_3"
}

activity_int_map = {
    "rest1": 1,
    "prepare speech": 2,
    "give speech": 3,
    "rest2": 4,
    "mental math": 5,
    "rest3": 6,
    "stationary_Bike1": 7,
    "stationary_Bike2": 8
}

# -------------------------------------------------
# MERGE FUNCTION
# -------------------------------------------------
def merge_labels(df_raw, labels_df):
    df = df_raw.copy()

    # Preserve existing data or initialize
    if "activity_merged" not in df.columns:
        df["activity_merged"] = df.get("activity", None)
    if "activity_int_merged" not in df.columns:
        df["activity_int_merged"] = df.get("activity_int", -1)

    # FIX: Explicitly cast to object to avoid FutureWarning when inserting strings
    df["activity_merged"] = df["activity_merged"].astype(object)

    # Ensure Timestamp_pd is datetime
    df["Timestamp_pd"] = pd.to_datetime(df["Timestamp_pd"], errors="coerce")
    
    valid_timestamps = df["Timestamp_pd"].dropna()
    if valid_timestamps.empty:
        print("Warning: No valid timestamps found in HR data.")
        return df
    
    base_date = valid_timestamps.min().date()

    labels_df = labels_df.copy()
    labels_df["start_dt"] = labels_df["start_time"].apply(
        lambda t: pd.Timestamp.combine(base_date, pd.to_datetime(t).time())
    )
    labels_df["end_dt"] = labels_df["end_time"].apply(
        lambda t: pd.Timestamp.combine(base_date, pd.to_datetime(t).time())
    )

    labels_df["norm_act"] = labels_df["manual_labels_activity"].apply(normalize_activity)

    for _, row in labels_df.iterrows():
        start = row["start_dt"]
        end = row["end_dt"]
        act = row["norm_act"] 
        raw_act_string = reverse_mapping.get(act, act)

        mask = (
            (df["Timestamp_pd"] >= start) &
            (df["Timestamp_pd"] < end) &
            (df["activity_int_merged"] == -1)
        )

        df.loc[mask, "activity_merged"] = raw_act_string
        df.loc[mask, "activity_int_merged"] = activity_int_map.get(act, -1)

    return df

# -------------------------------------------------
# MAIN SCRIPT
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_type", default="heart_rate", choices=["biopac","heart_rate","acc","gry"])
    parser.add_argument("--labels_dir", default="manual", choices=["manual","belt"])
    args = parser.parse_args()

    pattern = os.path.join(INPUT_DIR, f"{args.file_type}_P*.csv")
    files = sorted(glob.glob(pattern))

    for file_path in files:
        fname = os.path.basename(file_path)
        pid = fname.split("_P")[-1].split(".csv")[0]
        
        print(f"\n--- Processing P{pid} ---")

        df_raw = pd.read_csv(file_path, low_memory=False)
        df_raw["Timestamp_pd"] = pd.to_datetime(df_raw["Timestamp_pd"], errors="coerce")

        if args.labels_dir == "manual":
            labels_path = os.path.join(MANUAL_LABELS_DIR, f"manual_task_timeline_P{pid}.csv")
        else:
            labels_path = os.path.join(BELT_LABELS_DIR, f"belt_task_timeline_P{pid}.csv")

        if not os.path.exists(labels_path):
            print(f"Skipping: Label file not found at {labels_path}")
            continue

        labels_df = pd.read_csv(labels_path)

        try:
            label_start_hour = pd.to_datetime(labels_df["start_time"].iloc[0]).hour
            hr_min_hour = df_raw["Timestamp_pd"].dt.hour.min()

            if hr_min_hour < 12 and label_start_hour >= 12:
                print(f"!!! Detected AM/PM mismatch. Applying 12-hour correction !!!")
                df_raw["Timestamp_pd"] = df_raw["Timestamp_pd"] + pd.Timedelta(hours=12)
        except Exception as e:
            print(f"Note: Could not perform auto-time-correction check: {e}")

        df_merged = merge_labels(df_raw, labels_df)
        df_merged = df_merged.drop(columns=["Unnamed: 3"], errors="ignore")

        out_file = os.path.join(
            OUTPUT_DIR,
            f"{args.file_type}_{pid}_merged_with_{args.labels_dir}_labels.csv"
        )
        df_merged.to_csv(out_file, index=False)
        
        # Statistics
        total_rows = len(df_merged)
        num_labeled = (df_merged["activity_int_merged"] != -1).sum()
        num_missing = total_rows - num_labeled
        
        # Check for the 8 unique labels
        unique_labels_found = df_merged[df_merged["activity_int_merged"] != -1]["activity_int_merged"].nunique()

        print(f"Saved: {out_file}")
        print(f"Summary: {num_labeled} Labeled | {num_missing} Missing | {total_rows} Total")
        print(f"Unique Labels Found: {unique_labels_found} / 8")
        
        if unique_labels_found < 8:
            found_list = sorted(df_merged[df_merged["activity_int_merged"] != -1]["activity_int_merged"].unique().tolist())
            print(f"Missing Label IDs: {set(range(1, 9)) - set(found_list)}")