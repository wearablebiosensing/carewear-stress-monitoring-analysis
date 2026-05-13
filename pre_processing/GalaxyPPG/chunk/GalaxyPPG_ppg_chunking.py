import pandas as pd
import numpy as np
import os
import glob
import re


print("\n==============================")
print("GalaxyPPG PPG Chunking Pipeline")
print("==============================\n")


# ==========================================================
# SELECT INPUT FOLDER
# ==========================================================

input_folder = input("Enter GalaxyPPG merged PPG folder path: ").strip()

output_folder = os.path.join(input_folder, "ppg_chunks")
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

        if "label" not in df.columns:
            print("Skipping file, no label column:", file_name)
            continue


        # --------------------------------------------------
        # Create consistent string label column
        # --------------------------------------------------

        df["manual_labels_activity"] = df["label"]


        # --------------------------------------------------
        # Build global label mapping
        # --------------------------------------------------

        for label in df["label"].dropna().unique():

            if label not in global_label_map:
                global_label_map[label] = label_counter
                label_counter += 1


        # --------------------------------------------------
        # Apply integer encoding
        # --------------------------------------------------

        df["activity_int"] = df["label"].map(global_label_map)


        # --------------------------------------------------
        # Segment when label changes
        # --------------------------------------------------

        df["segment_change"] = (df["label"] != df["label"].shift()).astype(int)

        df["segment_id"] = df["segment_change"].cumsum()


        # --------------------------------------------------
        # Extract participant ID
        # --------------------------------------------------

        pid_match = re.search(r'P(\d+)', file_name)

        if pid_match:
            participant_id = pid_match.group(1)
        else:
            participant_id = "unknown"


        # --------------------------------------------------
        # Create chunks
        # --------------------------------------------------

        for seg_id, seg_df in df.groupby("segment_id"):

            label_string = seg_df["manual_labels_activity"].iloc[0]
            label_int = seg_df["activity_int"].iloc[0]

            if pd.isna(label_string):
                continue

            output_name = f"activity_id_{label_int}_P{participant_id}_GalaxyWatch_PPG_chunk.csv"

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