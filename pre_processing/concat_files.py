
import pandas as pd 
import numpy as np 
import os 
import glob
import plotly.express as px
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import bioread
from scipy.signal import resample

import pytz

from datetime import datetime
from pathlib import Path
import glob
import ipywidgets as widgets
from IPython.display import display
from pandas.errors import EmptyDataError
import dqm
# from smarwatch_modules.smartwatch_processing_module import *


root_folder_str = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/data/"
root_data_set = Path("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/data/")
WRITE_FILE = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/Concat_File"

def process_and_save_data(csv_files, participant_folder, data_type, columns=None):
    """
    Helper function to process and save a specific type of data.
    """
    df_list = []
    found_files = False
    
    for csv_file in csv_files:
        # Corrected line: Use .name to get the filename string
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
            if 'Timestamp' in combined_df.columns:
                combined_df["Timestamp_pd"] = pd.to_datetime(combined_df["Timestamp"], errors='coerce')
            
            output_path = Path(WRITE_FILE) / f"{data_type}_{participant_folder}.csv"
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

def read_watch_data(root_folder_str, root_data_set):
    participant_folders = [folder.name for folder in root_data_set.iterdir() if folder.is_dir() and folder.name.startswith('P')]
    sorted_participants = sorted(participant_folders, key=lambda x: int(x[1:]))
    
    for participant_folder in sorted_participants:
        print(f"\nProcessing PID: {participant_folder}")
        p_folder = Path(root_folder_str) / participant_folder / "SMARTWATCH"
        
        if not p_folder.is_dir():
            print(f"SMARTWATCH folder not found for {participant_folder}. Skipping.")
            continue
            
        dated_folders = [name for name in os.listdir(p_folder) if os.path.isdir(os.path.join(p_folder, name))]
        if not dated_folders:
            print(f"No dated folders found for {participant_folder}. Skipping.")
            continue
        
        smart_watch_base_folder = p_folder / dated_folders[0]
        csv_files = list(smart_watch_base_folder.glob('*.csv'))
        
        # Check if any file exists before processing to avoid unnecessary loops
        if not csv_files:
            print(f"No CSV files found for {participant_folder}. Skipping.")
            continue
        
        # Process each modality separately
        process_and_save_data(csv_files, participant_folder, 'heart_rate')
        process_and_save_data(csv_files, participant_folder, 'acc', columns=["x", "y", "z", "unix_timesamp", "date_time", "activity"])
        process_and_save_data(csv_files, participant_folder, 'gry')
   
# Path to the main folder
def read_belt_data(main_folder_path):
    # List to store individual DataFrames.
    dfs = []
    # Iterate through each folder.
    for folder_name in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder_name)
        # Check if it's a directory.
        if os.path.isdir(folder_path):
            # Iterate through each file in the folder.
            for file_name in os.listdir(folder_path):
                if file_name != ".DS_Store":
                    print("file_name: ",file_name)
                    if "TEST" in file_name and file_name != ".DS_Store":
                        folder_path_test = os.path.join(folder_path, file_name)
                        print("TEST file_path_test: ",folder_path_test)
                        for filename in  os.listdir(folder_path_test):
                            if  filename != ".DS_Store":
                                print("TEST: filename: ",filename)
                                try:
                                    df_belt = pd.read_csv(os.path.join(folder_path_test, filename), delimiter=",")  # Change delimiter if needed
                                    df_belt.columns = ["timestamp","Respiration1","Respiration2","Respiration3","ECG"]
                                    activity_name1 = filename.split(" ")[0].split("-")[1]  #"stationary"
                                    activity_name2 =  filename.split(" ")[1].split("-")[0] # "Bike1"
                                    print("TEST Activity ID: ",activity_name1 + "_"+activity_name2)
                                    #print("TEST Activity ID2: ",filename.split(" ")[1].split("-")[0])
                                    df_belt["activity"]= activity_name1 + "_" + activity_name2 #file_name.split("-")[1]
                                    print(df_belt.head())
                                    dfs.append(df_belt)
                                except EmptyDataError:
                                    df_belt = None
                                    #print("PASS")
                    else:
                        try:
                            file_path = os.path.join(folder_path, file_name)
                            df_belt = pd.read_csv(file_path, delimiter=",")  # Change delimiter if needed.
                            df_belt.columns = ["timestamp","Respiration1","Respiration2","Respiration3","ECG"]
                            df_belt["activity"]=file_name.split("-")[1]
                            print(df_belt.head())
                        except UnicodeDecodeError:
                            df_belt =None
                            print("PASS UnicodeDecodeError")
                        
                        # Read the .ACQ file and append to list.
                        dfs.append(df_belt)
    df_combined_concat = pd.concat(dfs)
    print("df_combined_concat: ",df_combined_concat["activity"].unique())
    return  df_combined_concat

# Get start and end time from belt data
# Input parameters: df_belt
def getLables(df_belt):
    # Convert the millisecond Unix timestamp to human-readable datetime in EST
    df_combined_concat['datetime_est'] = pd.to_datetime(df_combined_concat['timestamp'], unit='ms')  # Convert to UTC datetime
    df_combined_concat['datetime_est'] = df_combined_concat['datetime_est'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')  # Convert to EST timezone
    # Group by activity and find start and end times
    activity_times = df_combined_concat.groupby('activity')['datetime_est'].agg(['min', 'max']).reset_index()
    activity_times.rename(columns={'min': 'start_time', 'max': 'end_time',"activity":"Belt_Activity_Labels"}, inplace=True)
    return activity_times


# read_watch_data(root_folder_str,root_data_set)

if __name__ == "__main__":

    print("######################################## SMARTWATCH DATA ########################################")

    # root_folder = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/2025-Test/CareWear_V2Test/V2/stress_protocol_minder/06-30-25"

    read_watch_data(root_folder_str,root_data_set)

    # participant_folders = [folder.name for folder in root_data_set.iterdir() if folder.is_dir() and folder.name.startswith('P')]
    # print("######################################## BIOPAC DATA ########################################")
    # print("participant_folders: ",participant_folders)
    # for pid in participant_folders:
    #     ## Check if participant ID has already been processed:
    #     if os.path.exists(WRITE_FILE +"/"+ pid + "_belt.csv"): # eg: P22_belt.csv
    #         print("File exists PARTICIPANT DATA already processed !! ======================")
    #     else:
    #         print("File does not exist")
    #         print(" pid ================================================= ",pid)
    #         main_folder_path = '/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/' + pid + '/BELT'
    #         read_bio_pac_data(main_folder_path)
    #         # df_combined_concat = read_belt_data(main_folder_path)
    #         # print("df_combined_concat: ",df_combined_concat["activity"].unique())
    #         # activity_times = getLables(df_combined_concat)
    #         activity_times.to_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/Task_Time_Line_Belt/" + "belt_task_timeline_" + pid +".csv")
    #         print("Filepath of the written file: ")
    #         print(WRITE_FILE + "/BELT/" + pid + "_belt.csv")
    #         df_combined_concat.to_csv(WRITE_FILE + "/" + pid + "_belt.csv")
    #         print("df_combined_concat concat file written ===========================")
    #         print(df_combined_concat.head())


    # participant_folders = [folder.name for folder in root_data_set.iterdir() if folder.is_dir() and folder.name.startswith('P')]
    # ######################################## BIOPAC DATA ########################################
    # print("######################################## BIOPAC DATA ########################################")
    # for pid in participant_folders:
    #     if os.path.exists(WRITE_FILE +"/"+ pid + "_biopac.csv"): # eg: P13_biopac.csv
    #         print("File exists PARTICIPANT DATA already processed !! ======================")
    #     else:
    #         print(" pid ================================================= ",pid)
    #         main_folder_path = '/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/' + pid + '/BIOPAC'
    #         df_combined_concat = read_bio_pac_data(main_folder_path)
    #         WRITE_FILE = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/Concat_File"

    #         df_combined_concat.to_csv(WRITE_FILE + "/" + pid + "_biopac.csv")
