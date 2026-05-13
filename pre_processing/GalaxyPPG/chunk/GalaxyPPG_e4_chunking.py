import pandas as pd
import numpy as np
import os
import glob
import re
from pathlib import Path

print("\n==============================")
print("GalaxyPPG E4 Chunking Pipeline")
print("==============================\n")

# ==========================================================
# SELECT INPUT FOLDER
# ==========================================================
# Expected: /.../4_merged_lables/EmpaticaE4
input_root = input("Enter GalaxyPPG merged EmpaticaE4 folder path: ").strip()
input_root = Path(input_root)

if not input_root.exists():
    print(f"❌ Error: Folder not found: {input_root}")
    exit(1)

# Prompt for separate output folder
output_root = input("Enter separate Output folder path: ").strip()
output_root = Path(output_root)
os.makedirs(output_root, exist_ok=True)

print(f"📂 Output Root: {output_root}")

# ==========================================================
# PROCESS EACH SENSOR SUBFOLDER
# ==========================================================
sensors = ["ACC", "HR", "BVP", "TEMP", "IBI"]

for sensor in sensors:
    sensor_input_dir = input_root / sensor
    if not sensor_input_dir.exists():
        print(f"⏩ Sensor {sensor} folder not found, skipping.")
        continue

    print(f"\n📂 Processing Sensor: {sensor}")
    
    # Create sensor output subfolder
    sensor_output_dir = output_root / sensor
    os.makedirs(sensor_output_dir, exist_ok=True)

    # Find merged CSV files
    file_list = glob.glob(str(sensor_input_dir / "*.csv"))
    print(f"   Files detected: {len(file_list)}")

    for file_path in file_list:
        try:
            file_name = os.path.basename(file_path)
            # print(f"   → Chunking: {file_name}")

            df = pd.read_csv(file_path)

            if "activity_int_merged" not in df.columns:
                print(f"   ⚠️ Skipping {file_name}: 'activity_int_merged' not found.")
                continue

            # --------------------------------------------------
            # Segment when label changes
            # --------------------------------------------------
            # We use 'label' or 'activity_int_merged' to detect transitions
            df["segment_change"] = (df["activity_int_merged"] != df["activity_int_merged"].shift()).astype(int)
            df["segment_id"] = df["segment_change"].cumsum()

            # --------------------------------------------------
            # Extract participant ID (e.g., P02)
            # --------------------------------------------------
            pid_match = re.search(r'P(\d+)', file_name)
            participant_id = f"P{pid_match.group(1)}" if pid_match else "unknown"

            # --------------------------------------------------
            # Create chunks
            # --------------------------------------------------
            chunk_count = 0
            for seg_id, seg_df in df.groupby("segment_id"):
                label_int = seg_df["activity_int_merged"].iloc[0]

                # Skip unlabeled data (-1)
                if label_int == -1:
                    continue

                # Pattern: activity_id_1.0_P03_EmpaticaE4_HR_chunk.csv
                # We use float formatting for label_int to match user request "1.0"
                output_name = f"activity_id_{float(label_int)}_{participant_id}_EmpaticaE4_{sensor}_chunk.csv"
                output_path = sensor_output_dir / output_name

                seg_df.to_csv(output_path, index=False)
                chunk_count += 1

            # print(f" ✅ Created {chunk_count} chunks.")

        except Exception as e:
            print(f"   ❌ Error processing {file_name}: {e}")

print("\n==============================")
print("Chunking complete!")
print(f"Chunks saved to: {output_root}")
print("==============================")
