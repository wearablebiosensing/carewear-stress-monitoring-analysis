
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

root_folder_str = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/data/"
root_data_set= Path("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/data/")
WRITE_FILE = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/Concat_File"

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
                if file_name.endswith('.txt'):
                    file_path = os.path.join(folder_path, file_name)
                    print("BIOPAC file_name: ",file_name.split("/")[-1])
                    # biopac_df = bioread.read_file(file_path)
                    # First, detect where the data table starts (at 'milliSec...')
                    with open(file_path, 'r') as f:
                        lines = f.readlines()

                    # Find the line number where the column headers (data) begin
                    for i, line in enumerate(lines):
                        if line.strip().startswith("milliSec"):
                            data_start = i
                            break

                    # Now read the data from that line forward
                    df = pd.read_csv(file_path, sep="\t", skiprows=data_start)

                    # Drop empty or unnamed columns if needed
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed|^\s*$', regex=True)]
                    # df.columns = ["milliSec", 'RSP-R_Y', 'ECG_X','FIR_ECG','ECG_R-R','Heart_Rate','ECG_R_wave','Respiration_Rate']
                    df = df.sort_values(by='milliSec', ascending=True)
                    df.columns = ["milliSec",'RSP-R_Y', 'ECG_X','FIR_ECG','ECG_R-R','Heart_Rate','ECG_R_wave','Respiration_Rate']
                    print("if -> file_name", file_name)
                    df["activity"]=file_name.split("-")[1] # P1-prepare speech-biopac-09-05-2025.txt
                    df.to_csv(WRITE_FILE +"/" +file_name.split("-")[0] +"_"+file_name.split("-")[1].split(" ")[0]+"_"+"biopac" +".csv") # P1-prepare speech-biopac-09-05-2025.csv
                    print("Biopac file written =================================" + file_name.split(".")[0] + "_biopac"+".csv")
                    # Convert biopac data to a data frame.
                    # json_biopac_data = {}
                    # for channel in biopac_df.channels:
                    #     #print("channel: ",channel)
                    #     json_biopac_data[channel] = channel.data
                    # biopac_dataframe = pd.DataFrame.from_dict(json_biopac_data)
                    # biopac_dataframe.columns = ['RSP-R_Y', 'ECG_X','FIR_ECG','ECG_R-R','Heart_Rate','ECG_R_wave','Respiration_Rate']
                    # #print(biopac_dataframe.head())
                    # biopac_dataframe["activity"]=file_name.split("-")[1]
                    # Read the .ACQ file and append to list
                    dfs.append(df)
                elif "TEST" in file_name and file_name != ".DS_Store":
                    folder_path_test = os.path.join(folder_path, file_name)
                    print("BIOPAC: TEST file_path_test: ",folder_path_test)
                    for filename in  os.listdir(folder_path_test):
                        if  filename != ".DS_Store" and filename.endswith('.txt'):
                            file_path = os.path.join(folder_path,  file_name + "/"+ filename)

                    # First, detect where the data table starts (at 'milliSec...')
                            with open(file_path, 'r') as f:
                                lines = f.readlines()

                            # Find the line number where the column headers (data) begin
                            for i, line in enumerate(lines):
                                if line.strip().startswith("milliSec"):
                                    data_start = i
                                    break
                            if os.path.exists(file_path):
        
                                # Now read the data from that line forward
                                df = pd.read_csv(file_path, sep="\t", skiprows=data_start)

                                # Drop empty or unnamed columns if needed
                                df = df.loc[:, ~df.columns.str.contains('^Unnamed|^\s*$', regex=True)]
                                # df.columns = ["milliSec", 'RSP-R_Y', 'ECG_X','FIR_ECG','ECG_R-R','Heart_Rate','ECG_R_wave','Respiration_Rate']
                                df = df.sort_values(by='milliSec', ascending=True)
                                df.columns = ["milliSec",'RSP-R_Y', 'ECG_X','FIR_ECG','ECG_R-R','Heart_Rate','ECG_R_wave','Respiration_Rate']
                                print("elseif -> filename", filename)
                                df["activity"]=filename.split("-")[1] # P1-prepare speech-biopac-09-05-2025.txt
                                df.to_csv(WRITE_FILE +"/"+ filename.split("-")[0] +"_"+filename.split("-")[1].split(" ")[0]+"_"+"biopac" + ".csv") # P1-prepare speech-biopac-09-05-2025.csv
                                # print("TEST: filename: ",filename)
                                # biopac_df_bike = bioread.read_file(os.path.join(folder_path_test, filename))
                                # # Convert biopac data to a data frame.
                                # json_biopac_data = {}
                                # for channel in biopac_df_bike.channels:
                                #     #print("channel: ",channel)
                                #     json_biopac_data[channel] = channel.data
                                # biopac_df_bike = pd.DataFrame.from_dict(json_biopac_data)
                                # biopac_df_bike.columns = ['RSP-R_Y', 'ECG_X','FIR_ECG','ECG_R-R','Heart_Rate','ECG_R_wave','Respiration_Rate']
                                # #print(biopac_dataframe.head())
                                # read_bio_pac_data.to_csv(WRITE_FILE + "")
                                # dfs.append(biopac_df_bike)
                            else:
                                print("File does not exist")
    # df_combined_concat = pd.concat(dfs)
if __name__ == "__main__":

    # # Pattern to match any file containing 'biopac.csv' in the name
    # pattern = os.path.join(WRITE_FILE, '*biopac.csv*')

    # # Find matching files
    # files_to_delete = glob.glob(pattern)

    # # Delete each file
    # for file_path in files_to_delete:
    #     try:
    #         os.remove(file_path)
    #         print(f"Deleted: {file_path}")
    #     except Exception as e:
    #         print(f"Error deleting {file_path}: {e}")


    participant_folders = [folder.name for folder in root_data_set.iterdir() if folder.is_dir() and folder.name.startswith('P')]
    print("######################################## BIOPAC DATA ########################################")
    print("participant_folders: ",participant_folders)
    for pid in participant_folders:
        ## Check if participant ID has already been processed:
        if os.path.exists(WRITE_FILE +"/"+ pid + "_biopac.csv"): # eg: P22_belt.csv
            print("File exists PARTICIPANT DATA already processed !! ======================")
        else:
            print("File does not exist")
            print(" pid ================================================= ",pid)
            main_folder_path = '/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/data/' + pid + '/BIOPAC'
            read_bio_pac_data(main_folder_path)
