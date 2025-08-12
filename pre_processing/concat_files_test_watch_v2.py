
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
watch_data_folder = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/2025-Test/CareWear_V2Test/V2/07-03-25"
WRITE_FOLDER = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/2025-Test/CareWear_V2Test/V2/Concat_Files/"

def concat_files(watch_data_folder):
    smart_watch_base_folder =watch_data_folder
    file_type = 'csv'  # Change to 'xlsx', 'json', etc., as needed.
    file_pattern = os.path.join(smart_watch_base_folder, f'*.{file_type}')
    print("file_pattern: ",file_pattern)
    csv_files = glob.glob(file_pattern)
    print("csv_files: ",csv_files)
    df_list = []
    df_list_acc = []
    df_list_gyr = []
    for csv_file in csv_files:
        # print("csv_file ===== ",csv_file)
        if "heart_rate" in csv_file:
            print("Heart Rate Only===== \n",csv_file)
            df = pd.read_csv(csv_file,on_bad_lines='skip')
            print("",df.head())
            df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df["Timestamp_pd"] = pd.to_datetime(combined_df["Timestamp"], errors='coerce')
    pid = "T3"
    print("============ combined_df ============ \n")
    print(combined_df.head())
    combined_df.to_csv(WRITE_FOLDER + "heart_rate_" + pid + ".csv")

        # if "acc" in csv_file:
        #     print("acc files==== \n",csv_file)
        #     try:
        #         df_acc = pd.read_csv(csv_file,on_bad_lines='skip')
        #         df_list_acc.append(df_acc)
        #     except EmptyDataError:
        #                 print("EmptyDataError")
        # if "gry" in csv_file:
        #     print("gry files==== \n",csv_file)
        #     try:
        #         df_gyr = pd.read_csv(csv_file,on_bad_lines='skip')
        #         df_list_gyr.append(df_gyr)
        #     except EmptyDataError:
        #         print("EmptyDataError") 
    
        # try:
        # combined_df = pd.concat(df_list, ignore_index=True)
        # print("combined_df==== \n")
        # print(combined_df.head())
        #     combined_df_acc = pd.concat(df_list_acc, ignore_index=True)
        #     combined_df_gyr = pd.concat(df_list_gyr, ignore_index=True)
        #     combined_df["Timestamp_pd"] = pd.to_datetime(combined_df["Timestamp"], errors='coerce')

        #     WRITE_FOLDER = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/Concat_File/"
        #     combined_df.to_csv(WRITE_FOLDER + "heart_rate_" + "T2" + ".csv")
        #     combined_df_acc.to_csv(WRITE_FOLDER + "acc_" + "T2" + ".csv")
        #     combined_df_gyr.to_csv(WRITE_FOLDER + "gry_" + "T2" + ".csv")
        # except TypeError as e:
        #      print("combined_df_sort: ")
concat_files(watch_data_folder)