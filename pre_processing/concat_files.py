import pandas as pd
import numpy as np
import os
import glob
import re
import bioread
import pytz
import argparse
from datetime import datetime
from pathlib import Path
from pandas.errors import EmptyDataError
from scipy.signal import resample

root_data_set = Path("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/data/")
WRITE_FILE = Path("/Volumes/CW_2024/Concat_File")
def process_and_save_data(csv_files, participant_folder, data_type, columns=None):
    """
    Helper function to process and save a specific type of data.
    """
    df_list = []
    found_files = False
    
    for csv_file in csv_files:
        if data_type in csv_file.name: 
            print(f"Processing {data_type} for {participant_folder}...")
            found_files = True
            try:
                if columns:
                    df = pd.read_csv(csv_file, on_bad_lines='skip', names=columns)
                else:
                    df = pd.read_csv(csv_file, on_bad_lines='skip')
                df_list.append(df)
            except EmptyDataError:
                print(f"EmptyDataError: {csv_file} is empty.")
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
    
    if found_files and df_list:
        try:
            combined_df = pd.concat(df_list, ignore_index=True)
            
            # --- START: Modified code to include 'acc' data type ---
            if data_type in ['heart_rate', 'acc']:
                # Define the mapping dictionary
                activity_mapping = {
                    'Rest_1': 1,
                    'Prepare_Speech': 2,
                    'Give_Speech': 3,
                    'Rest_2': 4,
                    'Mental_Math': 5,
                    'Rest_3': 6,
                    'Stationary_Bike_Legs': 7,
                    'Stationary_Bike_Leegs': 7,
                    'Stationary_Biketationary_Bike_Legs': 7,
                    'SStationary_Bike_Legs': 7,
                    'Stationaregs': 7,
                    'Sy_Bike_Legs': 7,
                    'Statiotationary_Bike_Legs': 7,
                    'Stationary_Bike_Lege_Legs': 7,
                    'Statioke_Legs': 7,
                    'Stationary_nary_Bike_Legs': 7,
                    'SBike_Legs': 7,
                    'St_Legs': 7,
                    'Stationary_Bike_Hand': 8,
                    'Stationary_Bike_Hantationary_Bike_Hand': 8,
                    'Statationary_Bike_Hand': 8,
                    'Statiand': 8,
                    'Stationar_Hand': 8,
                    'Stationary_By_Bike_Hand': 8,
                    'Stationary_Bikeke_Hand': 8,
                    'Statind': 8,
                }
                
                # Apply the mapping and fill NaN with -1
                combined_df['activity_int'] = combined_df['activity'].map(activity_mapping).fillna(-1)
                print(f"Created new 'activity_int' column for {data_type} data.")
            # --- END: Modified code ---

            if 'Timestamp' in combined_df.columns:
                combined_df["Timestamp_pd"] = pd.to_datetime(combined_df["Timestamp"], errors='coerce')
            
            output_path = WRITE_FILE / f"{data_type}_{participant_folder}.csv"
            combined_df.to_csv(output_path, index=False)
            print(f"Successfully saved {data_type} data to {output_path}")
        except ValueError as e:
            print(f"ValueError during concatenation for {data_type}: {e}")
        except TypeError as e:
            print(f"TypeError during processing for {data_type}: {e}")
    elif found_files:
        print(f"No valid data to concatenate for {data_type} from {participant_folder}.")
    else:
        print(f"No {data_type} files found for {participant_folder}.")
        
def process_watch_data(root_data_set):
    participant_folders = [folder.name for folder in root_data_set.iterdir() if folder.is_dir() and folder.name.startswith('P')]
    sorted_participants = sorted(participant_folders, key=lambda x: int(x[1:]))
    
    for participant_folder in sorted_participants:
        print(f"\nProcessing PID: {participant_folder}")
        p_folder = root_data_set / participant_folder / "SMARTWATCH"
        
        if not p_folder.is_dir():
            print(f"SMARTWATCH folder not found for {participant_folder}. Skipping.")
            continue
            
        dated_folders = [name for name in os.listdir(p_folder) if os.path.isdir(os.path.join(p_folder, name))]
        if not dated_folders:
            print(f"No dated folders found for {participant_folder}. Skipping.")
            continue
        
        smart_watch_base_folder = p_folder / dated_folders[0]
        csv_files = list(smart_watch_base_folder.glob('*.csv'))
        
        if not csv_files:
            print(f"No CSV files found for {participant_folder}. Skipping.")
            continue
        
        process_and_save_data(csv_files, participant_folder, 'heart_rate')
        process_and_save_data(csv_files, participant_folder, 'acc', columns=["x", "y", "z", "unix_timesamp", "date_time", "activity"])
        process_and_save_data(csv_files, participant_folder, 'gry')

def process_belt_data(root_data_set):
    participant_folders = [folder.name for folder in root_data_set.iterdir() if folder.is_dir() and folder.name.startswith('P')]
    sorted_participants = sorted(participant_folders, key=lambda x: int(x[1:]))
    
    for pid in sorted_participants:
        output_file_path = WRITE_FILE / f"{pid}_belt.csv"
        if output_file_path.exists():
            print(f"File exists, skipping processing for {pid} belt data.")
            continue
        
        print(f"\nProcessing PID: {pid}")
        main_folder_path = root_data_set.parent / pid / 'BELT' 
        
        if not main_folder_path.is_dir():
            print(f"BELT folder not found for {pid}. Skipping.")
            continue
            
        dfs = []
        for folder_name in os.listdir(main_folder_path):
            folder_path = main_folder_path / folder_name
            if folder_path.is_dir():
                for file_name in os.listdir(folder_path):
                    file_path = folder_path / file_name
                    if file_path.is_file() and file_name != ".DS_Store":
                        try:
                            df_belt = pd.read_csv(file_path, delimiter=",", names=["timestamp", "Respiration1", "Respiration2", "Respiration3", "ECG"], skiprows=1)
                            df_belt["activity"] = re.sub(r'[^a-zA-Z0-9_]', '', file_name.split("-")[1])
                            dfs.append(df_belt)
                        except (pd.errors.EmptyDataError, UnicodeDecodeError):
                            print(f"Could not read or empty file: {file_path}")
                        except IndexError:
                            print(f"Skipping malformed filename: {file_name}")

        if dfs:
            df_combined_concat = pd.concat(dfs, ignore_index=True)
            df_combined_concat.to_csv(output_file_path, index=False)
            print(f"Successfully saved combined belt data to {output_file_path}")
        else:
            print(f"No valid belt data found for {pid}.")
            
def process_biopac_data(root_data_set):
    participant_folders = [folder.name for folder in root_data_set.iterdir() if folder.is_dir() and folder.name.startswith('P')]
    sorted_participants = sorted(participant_folders, key=lambda x: int(x[1:]))

    for pid in sorted_participants:
        output_file_path = WRITE_FILE / f"{pid}_biopac.csv"
        if output_file_path.exists():
            print(f"File exists, skipping processing for {pid} biopac data.")
            continue
            
        print(f"\nProcessing PID: {pid}")
        biopac_folder = root_data_set.parent / pid / 'BIOPAC'
        
        if not biopac_folder.is_dir():
            print(f"BIOPAC folder not found for {pid}. Skipping.")
            continue
        
        biopac_files = list(biopac_folder.glob('*.acq'))
        if not biopac_files:
            print(f"No Biopac files found for {pid}. Skipping.")
            continue

        dfs = []
        for file_path in biopac_files:
            try:
                data = bioread.read_file(file_path)
                df = pd.DataFrame()
                for channel in data.channels:
                    df[channel.name] = channel.data
                
                df['sample_number'] = range(len(df))
                
                dfs.append(df)
            except Exception as e:
                print(f"Error reading Biopac file {file_path}: {e}")

        if dfs:
            combined_biopac_df = pd.concat(dfs, ignore_index=True)
            combined_biopac_df.to_csv(output_file_path, index=False)
            print(f"Successfully saved combined Biopac data to {output_file_path}")
        else:
            print(f"No valid Biopac data found for {pid}.")

# ----------------- Main function to handle arguments -----------------
def main(data_type):
    if data_type == 'watch':
        process_watch_data(root_data_set)
    elif data_type == 'belt':
        process_belt_data(root_data_set)
    elif data_type == 'biopac':
        process_biopac_data(root_data_set)
    elif data_type == 'all':
        print("Processing all data types...")
        process_watch_data(root_data_set)
        process_belt_data(root_data_set)
        process_biopac_data(root_data_set)
    else:
        print("Invalid data type specified. Please choose 'watch', 'belt', 'biopac', or 'all'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process different types of study data.")
    parser.add_argument("--data_type", required=True, choices=['watch', 'belt', 'biopac', 'all'], help="The type of data to process.")
    
    args = parser.parse_args()
    main(args.data_type)