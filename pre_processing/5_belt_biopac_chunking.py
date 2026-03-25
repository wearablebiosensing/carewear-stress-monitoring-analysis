import pandas as pd
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

# --- SELECTION SETTINGS ---
# The sensors to scan for in the filenames
SENSOR_MAP = {
    "belt": "belt",
    "biopac": "biopac"
}

def chunk_data(source_dir, output_base_dir):
    src_path = Path(source_dir)
    if not src_path.exists():
        print(f"[ERROR] Cannot find directory: {source_dir}")
        return

    # Create subdirectories for the sensors
    for sensor in SENSOR_MAP.keys():
        os.makedirs(os.path.join(output_base_dir, sensor), exist_ok=True)

    file_count = 0
    # Use rglob to search recursively so it easily finds your belt/ and biopac/ nested folders!
    for file_path in src_path.rglob("*.csv"):
        if file_path.name.startswith("._"):
            continue
            
        file_name = file_path.name.lower()
        active_sensor = next((key for key, val in SENSOR_MAP.items() if val in file_name), None)
        
        if active_sensor:
            print(f"Processing: {file_path.name}")
            
            # Extract PID from filename (e.g., belt_10_merged_with_manual_labels.csv -> 10)
            pid_match = re.search(r'(?:belt|biopac)_(\d+)', file_name)
            
            if not pid_match:
                print(f"  ! Skipping: Could not find PID in filename {file_path.name}")
                continue
                
            pid = pid_match.group(1)

            try:
                # low_memory=False to prevent DtypeWarnings if files are huge
                df = pd.read_csv(file_path, low_memory=False)
                
                # Clean column names just in case
                df.columns = [c.strip() for c in df.columns]

                if 'activity_int_merged' in df.columns:
                    file_count += 1
                    
                    # Group by the activity mapping integer (1 through 8)
                    grouped = df.groupby('activity_int_merged')

                    for activity, group_data in grouped:
                        # Skip if activity_int_merged is missing or empty
                        if pd.isna(activity):
                            continue
                            
                        # Final Naming Example: activity_id_1.0_belt_1_merged_labels.csv
                        new_filename = f"activity_id_{float(activity)}_{active_sensor}_{pid}_merged_labels.csv"
                        output_path = os.path.join(output_base_dir, active_sensor, new_filename)
                        group_data.to_csv(output_path, index=False)
                        print(f"  -> Saved chunk: {new_filename}")
                else:
                    print(f"  ! Skipping: Missing 'activity_int_merged'. Found cols: {list(df.columns)}")
            
            except Exception as e:
                print(f"  ! Error reading {file_path.name}: {e}")
        
    print(f"\\n--- DONE: Processed {file_count} files successfully. ---")
    messagebox.showinfo("Complete", f"Successfully chunked {file_count} merged files into separate tasks!")

def main():
    root = tk.Tk()
    root.withdraw()
    
    messagebox.showinfo(
        "Select DIRECTORY", 
        "Please select the INPUT DIRECTORY.\\n(The folder containing the merged 'belt_X_merged...' and 'biopac_X_merged...' CSVs)"
    )
    source_dir = filedialog.askdirectory(title="Select Merged Input Directory")
    if not source_dir:
        print("Cancelled.")
        return
        
    messagebox.showinfo(
        "Select DIRECTORY", 
        "Please select the CHUNKING OUTPUT DIRECTORY.\\n(Where the split activity files will be saved)"
    )
    output_dir = filedialog.askdirectory(title="Select Chunking Output Base Directory")
    if not output_dir:
        print("Cancelled.")
        return
        
    print(f"INPUT_DIR: {source_dir}")
    print(f"OUTPUT_DIR: {output_dir}\\n")
    
    chunk_data(source_dir, output_dir)

if __name__ == "__main__":
    main()
