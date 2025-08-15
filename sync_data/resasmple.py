import pandas as pd
from pathlib import Path

# === Input/Output folders ===
input_folder = Path(
    "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/"
    "Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/other/merged_lables/"
)
output_folder = Path(
    "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/"
    "Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/other/resampled_HR"
)
output_folder.mkdir(exist_ok=True)

# === Resample helper ===
def resample_by_activity(df, time_col, activity_col, freq="1s"):
    """
    Resample dataframe by given frequency grouped by activity.
    - Drops invalid datetime rows before resampling
    - Converts heart rate columns to numeric
    - Only numeric columns are averaged
    """
    resampled_list = []
    for activity, group in df.groupby(activity_col):
        # Drop invalid or missing time values
        group = group[group[time_col].notna()]
        if group.empty:
            continue

        group = group.set_index(time_col)

        # Convert heart rate(s) to numeric so they are preserved
        for hr_col in ["Heart_Rate", "HeartRate"]:
            if hr_col in group.columns:
                group[hr_col] = pd.to_numeric(group[hr_col], errors="coerce")

        # Resample numeric columns
        resampled = group.resample(freq).mean(numeric_only=True)

        # Add the activity back
        resampled[activity_col] = activity

        resampled_list.append(resampled.reset_index())

    if resampled_list:
        return pd.concat(resampled_list, ignore_index=True)
    else:
        return pd.DataFrame()  # Empty result if no data groups

# === Process Biopac files ===
for file in input_folder.glob("*_biopac.csv"):
    df = pd.read_csv(file)

    # Create absolute datetime from milliSec offset
    start_time = pd.Timestamp("2025-01-01")
    df["time"] = start_time + pd.to_timedelta(df["milliSec"], unit="ms")

    # Force Heart_Rate numeric if present
    if "Heart_Rate" in df.columns:
        df["Heart_Rate"] = pd.to_numeric(df["Heart_Rate"], errors="coerce")

    # Drop invalid times
    df = df[df["time"].notna()]

    resampled_df = resample_by_activity(df, "time", "activity")

    # Preserve filename if present
    if "filename" in df.columns:
        resampled_df["filename"] = file.name

    # Save
    out_path = output_folder / file.name
    resampled_df.to_csv(out_path, index=False)
    print(f"✅ Saved Biopac resampled: {out_path}")

# === Process Smartwatch Heart Rate files ===
for file in input_folder.glob("heart_rate_*_merged_labels.csv"):
    df = pd.read_csv(file)

    # Ensure datetime
    df["Timestamp_pd"] = pd.to_datetime(df["Timestamp_pd"], errors="coerce")

    # Force HeartRate numeric if present
    if "HeartRate" in df.columns:
        df["HeartRate"] = pd.to_numeric(df["HeartRate"], errors="coerce")

    # Drop invalid datetimes
    df = df[df["Timestamp_pd"].notna()]

    resampled_df = resample_by_activity(df, "Timestamp_pd", "manual_labels_activity")

    # Merge back original Timestamp if available (drop null keys before merge)
    if "Timestamp" in df.columns and not resampled_df.empty:
        right_df = (
            df[["Timestamp_pd", "Timestamp"]]
            .dropna(subset=["Timestamp_pd"])
            .sort_values("Timestamp_pd")
        )
        left_df = resampled_df.dropna(subset=["Timestamp_pd"]).sort_values(
            "Timestamp_pd"
        )

        if not right_df.empty and not left_df.empty:
            resampled_df = pd.merge_asof(left_df, right_df, on="Timestamp_pd")

    # Save
    out_path = output_folder / file.name
    resampled_df.to_csv(out_path, index=False)
    print(f"✅ Saved Heart Rate resampled: {out_path}")
