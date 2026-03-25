import pandas as pd
import os
import glob
import tkinter as tk
from tkinter import filedialog

def generate_hr_summary(folder_path):
    # List of features to extract
    features = [
        'HR Range: 0–40 bpm', 'HR Range: 40–60 bpm', 'HR Range: 60–80 bpm', 
        'HR Range: 80–100 bpm', 'HR Range: 100–120 bpm', 'HR Range: 120–140 bpm', 
        'HR Range: 140–160 bpm', 'HR Range: 160–180 bpm', 'HR Range: 180–200 bpm', 
        'HR Range: >200 bpm'
    ]

    # Mapping file patterns to cleaned device names
    mapping = {
        'CareWear_biopac': 'CareWear Biopac',
        'CareWear_heart_rate': 'CareWear Galaxy Watch',
        'CareWear_belt': 'CareWear Belt',
        'GalaxyPPG_E4': 'GalaxyPPG Empatica E4',
        'GalaxyPPG_Polar': 'GalaxyPPG Polar H10',
        'PolarH10': 'Polar H10',
        'EmpaticaE4': 'Empatica E4',
        'GalaxyWatch': 'Galaxy Watch',
        'biopac': 'Biopac'
    }

    results = {}
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        # Skip output files
        if 'HR_Range_Summary' in filename or 'automated' in filename.lower(): 
            continue
            
        try:
            df = pd.read_csv(file_path)
            
            # Determine device mapping
            device_target = "Unknown"
            for pattern, name in mapping.items():
                if pattern.lower() in filename.lower():
                    device_target = name
                    break
            
            if device_target == "Unknown":
                # Fallback: remove suffix and use as device name
                device_target = filename.replace('_heart_rate_quality_features.csv', '').replace('.csv', '')
                
            file_stats = {}
            # --- NEW: Total Rows ---
            file_stats['Total Rows'] = len(df)
            
            for feat in features:
                if feat in df.columns:
                    total_count = df[feat].sum()
                    std_val = df[feat].std()
                    # Display count and standard deviation
                    file_stats[feat] = f"{int(total_count)} ± {std_val:.2f}"
                else:
                    file_stats[feat] = ""
            
            results[device_target] = {"filename": filename, "stats": file_stats}
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not results:
        print(f"No relevant CSV files found in: {folder_path}")
        return

    # Standard columns to check first
    standard_cols = [
        'CareWear Galaxy Watch', 'CareWear Belt', 'CareWear Biopac',
        'GalaxyPPG Galaxy Watch', 'GalaxyPPG Empatica E4', 'GalaxyPPG Polar H10'
    ]
    
    # Sort devices for consistent columns
    all_devices = sorted(list(set(standard_cols) | set(results.keys())))
    column_names = [c for c in all_devices if c in results]

    # Organize data rows
    filenames_row = {col: results[col]["filename"] for col in column_names}
    total_rows_row = {col: results[col]["stats"]["Total Rows"] for col in column_names}
    data_rows = {feat: {col: results[col]["stats"][feat] for col in column_names} for feat in features}

    # Create summary DataFrame
    summary_df = pd.DataFrame(data_rows).T
    
    # Prepend filename and Total Rows info
    summary_df.loc['source_filename'] = pd.Series(filenames_row)
    summary_df.loc['Total Rows'] = pd.Series(total_rows_row)
    
    # Reorder index to have source_filename and Total Rows at top
    summary_df = summary_df.reindex(['source_filename', 'Total Rows'] + features)
    summary_df.index.name = 'filename'
    
    # OUTPUT: Write to the SAME folder provided in folder_path
    output_filename = 'HR_Range_Summary_Automated.csv'
    output_path = os.path.join(folder_path, output_filename)
    summary_df.to_csv(output_path)
    
    print(f"\n✅ SUCCESS: Results written to chosen folder:")
    print(f"  📂 Folder: {folder_path}")
    print(f"  📄 File:   {output_filename}")
    
    return summary_df

# Launch Tkinter Directory Picker
root = tk.Tk()
root.withdraw()
root.lift()
root.attributes('-topmost', True)
root.update()

try:
    input_folder = filedialog.askdirectory(title="Select Folder containing Quality Check CSVs")
except Exception as e:
    print(f"Dialog error: {e}")
    input_folder = None

root.destroy()

# Fallback to manual entry if dialog is cancelled or fails
if not input_folder:
    print("No folder selected via popup.")
    input_folder = input("Please manually enter the folder path (or leave blank to cancel): ").strip()

if input_folder and os.path.isdir(input_folder):
    summary = generate_hr_summary(input_folder)
    if 'display' in globals():
        display(summary)
    else:
        print(summary)
elif input_folder:
    print(f"Error: '{input_folder}' is not a valid directory.")
else:
    print("Operation cancelled.")