
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


root_folder_str = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/"
root_data_set= Path("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/")
WRITE_FILE = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/Concat_File"


def write_concat_files(root_folder_str,root_data_set):
    participant_folders = [folder.name for folder in root_data_set.iterdir() if folder.is_dir() and folder.name.startswith('P')]
    sorted_participants = sorted(participant_folders, key=lambda x: int(x[1:]))
    print("sorted_participants: ",sorted_participants)
    for participant_folder in sorted_participants:
        if os.path.exists(WRITE_FILE +"/"+ "heart_rate_"+participant_folder + ".csv"): # eg: heart_rate_P10.csv
            print("File exists PARTICIPANT DATA already processed !! ======================")
        else:

            print("PID==  ",participant_folder)
            p_folder = root_folder_str + participant_folder +"/SMARTWATCH"
            print("participant_folder ========",participant_folder)
            dated_folders = [name for name in os.listdir(p_folder) if os.path.isdir(os.path.join(p_folder, name))]
            smart_watch_base_folder = p_folder +"/"+ dated_folders[0]
            print("dated_folders: ",dated_folders)
            file_type = 'csv'  # Change to 'xlsx', 'json', etc., as needed.
            file_pattern = os.path.join(smart_watch_base_folder, f'*.{file_type}')
            print("file_pattern: ",file_pattern)
            csv_files = glob.glob(file_pattern)
            print("csv_files: ",csv_files)
            df_list = []
            df_list_acc = []
            df_list_gyr = []
            for csv_file in csv_files:
                print("csv_file: ",csv_file)
                if "heart_rate" in csv_file:
                    print("heart_rate , ",participant_folder)
                    df = pd.read_csv(csv_file,on_bad_lines='skip')
                    print("",df.head())
                    df_list.append(df)
                if "acc" in csv_file:
                    print("acc , ",participant_folder)
                    try:
                        df_acc = pd.read_csv(csv_file,on_bad_lines='skip')
                        df_list_acc.append(df_acc)
                    except EmptyDataError:
                        print("EmptyDataError")
                if "gry"  in csv_file:
                    print("gry , ",participant_folder)
                    try:
                        df_gyr = pd.read_csv(csv_file,on_bad_lines='skip')
                        df_list_gyr.append(df_gyr)
                    except EmptyDataError:
                        print("EmptyDataError") 
            try:
                combined_df = pd.concat(df_list, ignore_index=True)
                combined_df_acc = pd.concat(df_list_acc, ignore_index=True)
                combined_df_gyr = pd.concat(df_list_gyr, ignore_index=True)
                combined_df["Timestamp_pd"] = pd.to_datetime(combined_df["Timestamp"], errors='coerce')
            except ValueError as e:
                print("ValueError: ")

            try:
                combined_df_sort = combined_df.sort_values(by="Timestamp_pd")
                WRITE_FOLDER = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/Concat_File/"
                combined_df.to_csv(WRITE_FOLDER + "heart_rate_" + participant_folder + ".csv")
                combined_df_acc.to_csv(WRITE_FOLDER + "acc_" + participant_folder + ".csv")
                combined_df_gyr.to_csv(WRITE_FOLDER + "gry_" + participant_folder + ".csv")
            except TypeError as e:
                print("combined_df_sort: ")
# Path to the main folder
def read_bio_pac_data(main_folder_path):
    # List to store individual DataFrames
    dfs = []
    # Iterate through each folder
    for folder_name in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder_name)
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Iterate through each file in the folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.acq'):
                    file_path = os.path.join(folder_path, file_name)
                    print("BIOPAC file_name: ",file_name.split("/")[-1])
                    biopac_df = bioread.read_file(file_path)
                    # Convert biopac data to a data frame.
                    json_biopac_data = {}
                    for channel in biopac_df.channels:
                        #print("channel: ",channel)
                        json_biopac_data[channel] = channel.data
                    biopac_dataframe = pd.DataFrame.from_dict(json_biopac_data)
                    biopac_dataframe.columns = ['RSP-R_Y', 'ECG_X','FIR_ECG','ECG_R-R','Heart_Rate','ECG_R_wave','Respiration_Rate']
                    #print(biopac_dataframe.head())
                    biopac_dataframe["activity"]=file_name.split("-")[1]
                    # Read the .ACQ file and append to list
                    dfs.append(biopac_dataframe)
                elif "TEST" in file_name and file_name != ".DS_Store":
                    folder_path_test = os.path.join(folder_path, file_name)
                    print("BIOPAC: TEST file_path_test: ",folder_path_test)
                    for filename in  os.listdir(folder_path_test):
                        if  filename != ".DS_Store" and filename.endswith('.acq'):
                            print("TEST: filename: ",filename)
                            biopac_df_bike = bioread.read_file(os.path.join(folder_path_test, filename))
                            # Convert biopac data to a data frame.
                            json_biopac_data = {}
                            for channel in biopac_df_bike.channels:
                                #print("channel: ",channel)
                                json_biopac_data[channel] = channel.data
                            biopac_df_bike = pd.DataFrame.from_dict(json_biopac_data)
                            biopac_df_bike.columns = ['RSP-R_Y', 'ECG_X','FIR_ECG','ECG_R-R','Heart_Rate','ECG_R_wave','Respiration_Rate']
                            #print(biopac_dataframe.head())
                            biopac_df_bike["activity"]=filename.split("-")[1]
                            dfs.append(biopac_df_bike)


    df_combined_concat = pd.concat(dfs)
    return  df_combined_concat
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
print("######################################## SMARTWATCH DATA ########################################")


write_concat_files(root_folder_str,root_data_set)

participant_folders = [folder.name for folder in root_data_set.iterdir() if folder.is_dir() and folder.name.startswith('P')]
print("######################################## BIOPAC DATA ########################################")
print("participant_folders: ",participant_folders)
for pid in participant_folders:
    ## Check if participant ID has already been processed:
    if os.path.exists(WRITE_FILE +"/"+ pid + "_belt.csv"): # eg: P22_belt.csv
        print("File exists PARTICIPANT DATA already processed !! ======================")
    else:
        print("File does not exist")
        print(" pid ================================================= ",pid)
        main_folder_path = '/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/' + pid + '/BELT'
        df_combined_concat = read_belt_data(main_folder_path)
        print("df_combined_concat: ",df_combined_concat["activity"].unique())
            # Get start and end time,
        # Convert the millisecond Unix timestamp to human-readable datetime in EST
        df_combined_concat['datetime_est'] = pd.to_datetime(df_combined_concat['timestamp'], unit='ms')  # Convert to UTC datetime
        df_combined_concat['datetime_est'] = df_combined_concat['datetime_est'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')  # Convert to EST timezone
        # Group by activity and find start and end times
        activity_times = df_combined_concat.groupby('activity')['datetime_est'].agg(['min', 'max']).reset_index()
        activity_times.rename(columns={'min': 'start_time', 'max': 'end_time'}, inplace=True)
        # activity_times.to_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/Task_Time_Line_Belt/" + "belt_task_timeline_" + pid +".csv")
        print("Filepath of the written file: ")
        print(WRITE_FILE + "/BELT/" + pid + "_belt.csv")
        df_combined_concat.to_csv(WRITE_FILE + "/" + pid + "_belt.csv")
        print("df_combined_concat concat file written ===========================")
        print(df_combined_concat.head())


participant_folders = [folder.name for folder in root_data_set.iterdir() if folder.is_dir() and folder.name.startswith('P')]
######################################## BIOPAC DATA ########################################
print("######################################## BIOPAC DATA ########################################")
for pid in participant_folders:
    if os.path.exists(WRITE_FILE +"/"+ pid + "_biopac.csv"): # eg: P13_biopac.csv
        print("File exists PARTICIPANT DATA already processed !! ======================")
    else:
        print(" pid ================================================= ",pid)
        main_folder_path = '/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/' + pid + '/BIOPAC'
        df_combined_concat = read_bio_pac_data(main_folder_path)
        WRITE_FILE = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/Concat_File"

        df_combined_concat.to_csv(WRITE_FILE + "/" + pid + "_biopac.csv")
