import pandas as pd
import numpy as np
import os
import glob
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description="PolarH10 Chunking Pipeline")
    parser.add_argument("--input", required=True, help="Path to PolarH10 merged sensor folder (e.g. .../PolarH10/HR)")
    parser.add_argument("--output", required=True, help="Folder to save the chunks")
    parser.add_argument("--sensor", help="Sensor name (e.g. ACC, HR, ECG, IBI). If not provided, will try to detect from input path.")

    args = parser.parse_args()

    input_folder = args.input.strip()
    output_folder = args.output.strip()
    
    # Detect sensor from path if not provided
    if args.sensor:
        sensor = args.sensor.upper()
    else:
        sensor_match = re.search(r'PolarH10[\\/](\w+)', input_folder, re.IGNORECASE)
        if sensor_match:
            sensor = sensor_match.group(1).upper()
        else:
            sensor = "SENSOR"

    print("\n==============================")
    print(f"PolarH10 {sensor} Chunking Pipeline")
    print("==============================\n")
    print(f"Input:  {input_folder}")
    print(f"Output: {output_folder}")

    os.makedirs(output_folder, exist_ok=True)

    # ==========================================================
    # FIND FILES
    # ==========================================================

    file_list = glob.glob(os.path.join(input_folder, "*.csv"))
    print("Files detected:", len(file_list))

    # ==========================================================
    # PROCESS FILES
    # ==========================================================

    for file_path in file_list:
        try:
            file_name = os.path.basename(file_path)
            print("\nProcessing:", file_name)

            df = pd.read_csv(file_path)

            # --------------------------------------------------
            # Ensure activity_int_merged column exists
            # --------------------------------------------------
            if "activity_int_merged" not in df.columns:
                print("Skipping file, no activity_int_merged column:", file_name)
                continue

            # --------------------------------------------------
            # Extract participant ID
            # --------------------------------------------------
            pid_match = re.search(r'P(\d+)', file_name)
            if pid_match:
                participant_id = pid_match.group(1)
            else:
                participant_id = "unknown"

            # --------------------------------------------------
            # Segment when activity changes
            # --------------------------------------------------
            # Fill NaN in activity_int_merged with -1 to handle transition from/to None
            df["activity_int_merged"] = df["activity_int_merged"].fillna(-1).astype(int)
            
            df["segment_change"] = (df["activity_int_merged"] != df["activity_int_merged"].shift()).astype(int)
            df["segment_id"] = df["segment_change"].cumsum()

            # --------------------------------------------------
            # Create chunks
            # --------------------------------------------------
            for seg_id, seg_df in df.groupby("segment_id"):
                label_int = int(seg_df["activity_int_merged"].iloc[0])

                # Skip unassigned samples (-1)
                if label_int == -1:
                    continue

                output_name = f"activity_id_{label_int}_P{participant_id}_PolarH10_{sensor}_chunk.csv"
                output_path = os.path.join(output_folder, output_name)

                # Keep only original columns in chunks (avoid segment_change/segment_id if not needed)
                # But existing script kept them, so I'll keep them too for now.
                seg_df.to_csv(output_path, index=False)

            print("Chunks created:", df[df["activity_int_merged"] != -1]["segment_id"].nunique())

        except Exception as e:
            print("Error processing", file_name, e)

    print("\n==============================")
    print("Chunking complete")
    print("Chunks folder:", output_folder)
    print("==============================")

if __name__ == "__main__":
    main()
