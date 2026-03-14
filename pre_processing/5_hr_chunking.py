import pandas as pd
import os
import re
from pathlib import Path

# --- SELECTION SETTINGS ---
# Edit this list to choose what to process. 
# Example: SELECTED_SENSORS = ["hr"] or ["acc", "gyr"]
SELECTED_SENSORS = ["hr"] 

SOURCE_DIR = "/Volumes/ss/Project_CareWear/DATASET/ss_drive/4_merged_lables"
OUTPUT_BASE_DIR = "/Volumes/ss/Project_CareWear/DATASET/ss_drive/5_activity_chunks/hr_chunks"
 


SENSOR_MAP = {
    "hr": "heart_rate",
    "acc": "accelerometer",
    "gyr": "gyroscope"
}

def chunk_data():
    src_path = Path(SOURCE_DIR)
    if not src_path.exists():
        print(f"ERROR: Cannot find directory: {SOURCE_DIR}")
        return

    for sensor in SELECTED_SENSORS:
        os.makedirs(os.path.join(OUTPUT_BASE_DIR, sensor), exist_ok=True)

    file_count = 0
    for file_path in src_path.glob("*.csv"):
        if file_path.name.startswith("._"):
            continue
            
        file_name = file_path.name.lower()
        active_sensor = next((key for key, val in SENSOR_MAP.items() if val in file_name), None)
        
        if active_sensor:
            print(f"Processing: {file_path.name}")
            
            # Extract PID from filename (e.g., heart_rate_10_merged -> 10)
            # This looks for the digits immediately following 'heart_rate_'
            pid_match = re.search(r'(?:heart_rate|accelerometer|gyroscope)_(\d+)', file_name)
            
            if not pid_match:
                print(f"  ! Skipping: Could not find PID in filename {file_path.name}")
                continue
                
            pid = pid_match.group(1)

            try:
                df = pd.read_csv(file_path)
                # Clean column names
                df.columns = [c.strip() for c in df.columns]

                if 'activity_int' in df.columns:
                    file_count += 1
                    # Group by activity_int since PID is constant for this whole file
                    grouped = df.groupby('activity_int')

                    for activity, group_data in grouped:
                        # Final Naming: activity_id_1.0_hr_(PID - 1)_merged_labels.csv
                        new_filename = f"activity_id_{float(activity)}_{active_sensor}_{pid}_merged_labels.csv"
                        output_path = os.path.join(OUTPUT_BASE_DIR, active_sensor, new_filename)
                        group_data.to_csv(output_path, index=False)
                else:
                    print(f"  ! Skipping: Missing 'activity_int'. Found: {list(df.columns)}")
            
            except Exception as e:
                print(f"  ! Error reading {file_path.name}: {e}")
        
    print(f"\n--- DONE: Processed {file_count} files successfully. ---")

if __name__ == "__main__":
    chunk_data()