import pandas as pd
import numpy as np
import os
import glob
import re


print("\n==============================")
print("GalaxyPPG ACC Chunking Pipeline")
print("==============================\n")


# ==========================================================
# SELECT INPUT FOLDER
# ==========================================================

input_folder = input("Enter GalaxyPPG merged ACC folder path: ").strip().strip("'").strip('"')
output_folder = input("Enter Output folder path for chunks: ").strip().strip("'").strip('"')

os.makedirs(output_folder, exist_ok=True)


# ==========================================================
# FIND FILES
# ==========================================================

file_list = glob.glob(os.path.join(input_folder, "*.csv"))

print("Files detected:", len(file_list))


# ==========================================================
# GLOBAL LABEL MAPPING
# ==========================================================

global_label_map = {}
label_counter = 1


# ==========================================================
# PROCESS FILES
# ==========================================================

for file_path in file_list:

    try:

        file_name = os.path.basename(file_path)

        print("\nProcessing:", file_name)

        df = pd.read_csv(file_path)

        # --------------------------------------------------
        # Ensure label column exists
        # --------------------------------------------------

        # Check which string label column exists
        string_label_col = None
        if "activity_merged" in df.columns:
            string_label_col = "activity_merged"
        elif "label" in df.columns:
            string_label_col = "label"
            
        if not string_label_col or "activity_int_merged" not in df.columns:
            print("Skipping file, missing required label columns:", file_name)
            continue


        # --------------------------------------------------
        # Create consistent string label column
        # --------------------------------------------------

        df["manual_labels_activity"] = df[string_label_col]
        df["activity_int"] = df["activity_int_merged"]


        # --------------------------------------------------
        # Segment when label changes
        # --------------------------------------------------

        df["segment_change"] = (df[string_label_col] != df[string_label_col].shift()).astype(int)

        df["segment_id"] = df["segment_change"].cumsum()


        # --------------------------------------------------
        # Extract participant ID
        # --------------------------------------------------

        # Try GalaxyPPG format first (P02_), then fallback to CareWear format (acc_2_)
        pid_match = re.search(r'P(\d+)', file_name)
        if not pid_match:
            pid_match = re.search(r'acc_(\d+)_merged', file_name)

        if pid_match:
            participant_id = pid_match.group(1)
        else:
            participant_id = "unknown"


        # --------------------------------------------------
        # Create chunks (same naming style as HR)
        # --------------------------------------------------

        for seg_id, seg_df in df.groupby("segment_id"):

            label_string = seg_df["manual_labels_activity"].iloc[0]
            label_int = seg_df["activity_int"].iloc[0]

            if pd.isna(label_string):
                continue

            output_name = f"activity_id_{label_int}_acc_{participant_id}_merged_labels.csv"

            output_path = os.path.join(output_folder, output_name)

            seg_df.to_csv(output_path, index=False)

        print("Chunks created:", df["segment_id"].nunique())


    except Exception as e:

        print("Error processing", file_name, e)


# ==========================================================
# SAVE LABEL MAP
# ==========================================================

map_df = pd.DataFrame(
    list(global_label_map.items()),
    columns=["label_string", "label_int"]
)

map_df.to_csv(os.path.join(output_folder, "label_mapping.csv"), index=False)


print("\n==============================")
print("Chunking complete")
print("Unique labels:", len(global_label_map))
print("Chunks folder:", output_folder)
print("==============================")