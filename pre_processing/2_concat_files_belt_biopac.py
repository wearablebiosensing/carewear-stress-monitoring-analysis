import os
import glob
import pandas as pd
import bioread
import tkinter as tk
from tkinter import filedialog, messagebox

# Activity Mapping to integers 1-8
ACTIVITY_MAPPING = {
    'rest1': 1, '1_rest': 1, 'rest 1': 1,
    'prepare speech': 2, 'prepare_speech': 2, 'preparespeech': 2,
    'give speech': 3, 'give_speech': 3, 'givespeech': 3,
    'rest2': 4, '4_rest2': 4, 'rest 2': 4,
    'mental math': 5, 'mental_math': 5, 'mentalmath': 5,
    'rest3': 6, '6_rest3': 6, 'rest 3': 6,
    'stationary bike1': 7, 'stationary bike 1': 7, 'stationarybike1': 7,
    'stationary bike2': 8, 'stationary bike 2': 8, 'stationarybike2': 8,
    'test_1': 7, 'test_2': 8
}

def extract_activity_int(activity_str):
    """Maps the extracted string activity to standard integers 1-8."""
    activity_lower = activity_str.strip().lower()
    for key, val in ACTIVITY_MAPPING.items():
        if key in activity_lower:
            return val
    return -1

def extract_activity_string(file_name):
    """Extracts 'rest1', 'prepare speech' etc from 'P1-rest1-belt-09-11-2024.csv'."""
    parts = file_name.split('-')
    if len(parts) >= 2:
        return parts[1].strip()
    return "Unknown"

def process_belt(input_dir, output_dir):
    p_folders = [f for f in os.listdir(input_dir) if f.startswith('P') and os.path.isdir(os.path.join(input_dir, f))]
    
    for p_name in p_folders:
        belt_dir = os.path.join(input_dir, p_name, 'BELT')
        if not os.path.isdir(belt_dir):
            continue
        
        participant_id = ''.join(filter(str.isdigit, p_name))  # Extracts '1' from 'P-1'
        print(f"\\n[BELT] Processing Participant {participant_id}...")
        
        df_list = []
        for root, dirs, files in os.walk(belt_dir):
            for file in files:
                if file.startswith('._') or file == '.DS_Store':
                    continue
                
                # Check if it has 'belt' in the name
                if 'belt' not in file.lower():
                    continue
                    
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(
                        file_path, delimiter=",", 
                        names=["timestamp", "Respiration1", "Respiration2", "Respiration3", "ECG"], 
                        skiprows=1
                    )
                    
                    act_str = extract_activity_string(file)
                    df['activity_merged'] = act_str
                    df['activity_int_merged'] = extract_activity_int(act_str)
                    
                    df_list.append(df)
                    print(f"  -> Added {file} [Activity: {act_str}]")
                except Exception as e:
                    print(f"  -> [ERROR] Failed reading {file}: {e}")
                    
        if df_list:
            combined_df = pd.concat(df_list, ignore_index=True)
            belt_out_dir = os.path.join(output_dir, "BELT")
            os.makedirs(belt_out_dir, exist_ok=True)
            out_name = f"belt_{participant_id}_merged_with_manual_labels.csv"
            out_path = os.path.join(belt_out_dir, out_name)
            combined_df.to_csv(out_path, index=False)
            print(f"[SUCCESS] Saved BELT data -> {os.path.basename(belt_out_dir)}/{out_name}")
        else:
            print(f"[WARN] No BELT files found for {p_name}")

def process_biopac(input_dir, output_dir):
    p_folders = [f for f in os.listdir(input_dir) if f.startswith('P') and os.path.isdir(os.path.join(input_dir, f))]
    
    for p_name in p_folders:
        biopac_dir = os.path.join(input_dir, p_name, 'BIOPAC')
        if not os.path.isdir(biopac_dir):
            continue
        
        participant_id = ''.join(filter(str.isdigit, p_name))
        print(f"\\n[BIOPAC] Processing Participant {participant_id}...")
        
        df_list = []
        for root, dirs, files in os.walk(biopac_dir):
            for file in files:
                if file.startswith('._') or file == '.DS_Store':
                    continue
                
                # Process strictly the .acq biopac files
                if file.endswith('.acq'):
                    file_path = os.path.join(root, file)
                    try:
                        data = bioread.read_file(file_path)
                        df = pd.DataFrame()
                        for channel in data.channels:
                            df[channel.name] = channel.data
                            
                        df['sample_number'] = range(len(df))
                        
                        act_str = extract_activity_string(file)
                        df['activity_merged'] = act_str
                        df['activity_int_merged'] = extract_activity_int(act_str)
                        
                        df_list.append(df)
                        print(f"  -> Added {file} [Activity: {act_str}]")
                    except Exception as e:
                        print(f"  -> [ERROR] Failed reading {file}: {e}")
                        
        if df_list:
            combined_df = pd.concat(df_list, ignore_index=True)
            biopac_out_dir = os.path.join(output_dir, "BIOPAC")
            os.makedirs(biopac_out_dir, exist_ok=True)
            out_name = f"biopac_{participant_id}_merged_with_manual_labels.csv"
            out_path = os.path.join(biopac_out_dir, out_name)
            combined_df.to_csv(out_path, index=False)
            print(f"[SUCCESS] Saved BIOPAC data -> {os.path.basename(biopac_out_dir)}/{out_name}")
        else:
            print(f"[WARN] No BIOPAC .acq files found for {p_name}")

def main():
    root = tk.Tk()
    root.withdraw()
    
    messagebox.showinfo("Select DIRECTORY", "Please select the INPUT ROOT DIRECTORY.\\n(This folder should contain your 'P-1', 'P-2' folders)")
    input_dir = filedialog.askdirectory(title="Select Input Root Directory")
    if not input_dir:
        print("Cancelled.")
        return
        
    messagebox.showinfo("Select DIRECTORY", "Please select the OUTPUT DIRECTORY.\\n(Where concatenated CSVs will be saved)")
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    if not output_dir:
        print("Cancelled.")
        return
        
    print(f"INPUT_DIR: {input_dir}")
    print(f"OUTPUT_DIR: {output_dir}\\n")
    
    process_belt(input_dir, output_dir)
    process_biopac(input_dir, output_dir)
    print("\\nAll operations completed!")

if __name__ == "__main__":
    main()
