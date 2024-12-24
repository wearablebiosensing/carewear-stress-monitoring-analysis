
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

#####################################################################################################################################
# HELPER FUNCTION FOR WATCH DATA PROCESSING
#####################################################################################################################################

# INPUT: date_and_time:  2024-08-21 14:01:10.895
# OUTPUT: '14:01:10.895'
def get_time(date_and_time): # Code that might throw an error.
    try:
        #print("date_and_time: ",date_and_time)
        return date_and_time.split(" ")[1]
    # Handle the error or pass to continue
    except Exception as e: 
        print(f"An error occurred: {e}")
        pass  # Continue to the next line.
def get_time_seconds(date_and_time):
    try:
        return date_and_time.split(".")[0]
    except Exception as e:
        print(f"An error occurred: {e}")
        pass  # Continue to the next line.
def get_date(date_and_time):
    try:
        return date_and_time.split(" ")[0]
    except Exception as e:
        print(f"An error occurred: {e}")
        pass  # Continue to the next line.
    
def get_filename_date_and_time(date_and_time): # Code that might throw an error.
    try:
        # print("get_filename_time():/ date_and_time: ",date_and_time)
        pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}\.\d{3}\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}\.\d{3})'
        match = re.search(pattern, date_and_time)
        if match:
            result = match.group(1)
            print("result: ",result)
            return result
    # Handle the error or pass to continue
    except Exception as e: 
        print(f"An error occurred: {e}")
        pass  # Continue to the next line.
    
def get_fimename_time(date_and_time): # Code that might throw an error.
    try:
        return date_and_time.split(".")[1]
    except Exception as e: 
        print(f"An error occurred: {e}")
        pass  # Continue to the next line.
    
######################################################################################################################################################################
# ACCELEROMETER DATA PROCESSING
######################################################################################################################################################################

# Concatinates a folder consisting of CSV files into one pandas df.
# Input files_list: ['/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/StudyData_Drive_2024/P7/SMARTWATCH/10-05-2024/acc_2024-12-05_14_42_59.472.csv',
 # '/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/StudyData_Drive_2024/P7/SMARTWATCH/10-05-2024/gry_2024-12-05_13_41_11.695.csv',
 # '/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/StudyData_Drive_2024/P7/SMARTWATCH/10-05-2024/acc_2024-12-05_14_32_32.657.csv',
 # '/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/StudyData_Drive_2024/P7/SMARTWATCH/10-05-2024/acc_2024-12-05_14_48_16.472.csv',
 # '/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/StudyData_Drive_2024/P7/SMARTWATCH/10-05-2024/gry_2024-12-05_15_04_12.530.csv']
# OUTPUT: pandas df with raw acceleromter data 
def create_dataframe_acc(files_list):
    print("create_dataframe(): files:",files_list)
    df_list_acc = []
    for full_file_path in files_list:
        print("create_dataframe(): full_file_path:", full_file_path)
        if "acc" in full_file_path:
            try:
                column_names = ['X', 'Y', 'Z','system_timestamp',"timestamp","activity"]
                df = pd.read_csv(full_file_path,names=column_names)
                print("df.shape: ",df.shape)
                print("Filename = =",full_file_path[-1])
                df_list_acc.append(df)
                print("full_file_path: ",full_file_path.split("/")[-1])
                if df.shape[1] != 6:
                    print("Shape not matching 6==================",)
            except Exception as e:
                print("full_file_path: ERROR======== ",full_file_path)
                print(f"An error occurred: {e}")
                pass  # Continue to the next line.
    return pd.concat(df_list_acc, axis=0, ignore_index=True)

def acc_plots(taskID,acc_df):
    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(y=acc_df[acc_df["activity"]==taskID]["X"], name='Acc X',
                             line=dict(color='red', width=4)))
    fig.add_trace(go.Scatter(y=acc_df[acc_df["activity"]==taskID]["Y"], name = 'Acc Y',
                             line=dict(color='royalblue', width=4)))
    fig.add_trace(go.Scatter(y=acc_df[acc_df["activity"]==taskID]["Z"], name='Acc Z',
                             line=dict(color='green', width=4) # dash options include 'dash', 'dot', and 'dashdot'
    ))

    # Edit the layout
    fig.update_layout(title=taskID,
                       xaxis_title='Number of Samples',
                       yaxis_title='Accelerometer Value')
    return fig

def plot_acc_raw(acc_df):
    fig1 = acc_plots("Rest_1",acc_df[acc_df["activity"]=="Rest_1"])
    fig2 = acc_plots("Prepare_Speech",acc_df[acc_df["activity"]=="Prepare_Speech"])
    fig3 = acc_plots("Give_Speech",acc_df[acc_df["activity"]=="Give_Speech"])
    fig4 = acc_plots("Stationary_Bike",acc_df[acc_df["activity"]=="Stationary_Bike"])
    fig5 = acc_plots("Rest_2",acc_df[acc_df["activity"]=="Rest_2"])
    fig6 = acc_plots("Mental_Math",acc_df[acc_df["activity"]=="Mental_Math"])
    fig7 = acc_plots("Rest_3",acc_df[acc_df["activity"]=="Rest_3"])
    
    # Organize into figures.
    # Create a 3x2 subplot grid
    fig_combined = make_subplots(rows=3, cols=2, subplot_titles=("Rest 1", "Prepare_Speech", "Give_Speech", "Stationary Bike", " Rest 2", "Rest 3"))

    # Add each individual plot to the subplot grid
    fig_combined.add_trace(fig1['data'][0], row=1, col=1)
    fig_combined.add_trace(fig1['data'][1], row=1, col=1)
    fig_combined.add_trace(fig1['data'][2], row=1, col=1)

    fig_combined.add_trace(fig2['data'][0], row=1, col=2)
    fig_combined.add_trace(fig2['data'][1], row=1, col=2)
    fig_combined.add_trace(fig2['data'][2], row=1, col=2)

    fig_combined.add_trace(fig3['data'][0], row=2, col=1)
    fig_combined.add_trace(fig3['data'][1], row=2, col=1)
    fig_combined.add_trace(fig3['data'][2], row=2, col=1)

    fig_combined.add_trace(fig4['data'][0], row=2, col=2)
    fig_combined.add_trace(fig4['data'][1], row=2, col=2)
    fig_combined.add_trace(fig4['data'][2], row=2, col=2)

    fig_combined.add_trace(fig5['data'][0], row=3, col=1)
    fig_combined.add_trace(fig5['data'][1], row=3, col=1)
    fig_combined.add_trace(fig5['data'][2], row=3, col=1)

    fig_combined.add_trace(fig6['data'][0], row=3, col=2)
    fig_combined.add_trace(fig6['data'][1], row=3, col=2)
    fig_combined.add_trace(fig6['data'][2], row=3, col=2)

    # Update layout for the combined figure
    fig_combined.update_layout(
        height=900, width=800,  # Adjust size if needed
        title_text="Combined accelerometer data",
        showlegend=False  # Hide legend if you don't need it
    )
    return fig_combined
    # # Show the combined figure.
    # fig_combined.show()

######################################################################################################################################################################
# HEART RATE DATA PROCESSING
######################################################################################################################################################################
def process_heart_rate(files):
    error_files =[]
    task_id_list = []
    file_name_list = []
    pid_list = []
    number_of_samples = []
    number_rows_list =[]
    df_list = []
    for full_file_path in files:
        if "heart_rate" in full_file_path:
            print("full_file_path: ",full_file_path)
            try:
                file_name = full_file_path.split("/")[-1]
                # print("file_name: ",file_name)
                # print("Task ID: ",file_name.split("_")[1])
                df = pd.read_csv(full_file_path,error_bad_lines=False)
                df["filename"] = full_file_path.split("/")[-1]
                df['time'] = df['Timestamp'].apply(get_time)
                df['date'] = df['Timestamp'].apply(get_date)
                df["seconds_timestamp"] = df['time'].apply(get_time_seconds)
                number_of_samples.append(df.shape[0])
                # print("average_sample_rate: ",average_sample_rate)
                # print("number of rows: ", df.shape)
                task_id_list.append(file_name.split("_")[1])
                file_name_list.append(file_name)
                pid_list.append(file_name.split("_")[0])
                number_rows_list.append(df.shape[0]) 
                df_list.append(df)
            except Exception as e:
                print("full_file_path: ERROR======== ",full_file_path)
                error_files.append(full_file_path)
                print(f"An error occurred: {e}")
                pass  # Continue to the next line.
    df_concat_full = pd.concat(df_list)
    # print("df_concat_full: ",df_concat_full.columns)
    # Convert the values in the column to float
    df_concat_full['heart_rate_float'] = pd.to_numeric(df_concat_full['HeartRate'], errors='coerce') 
    subset_df = df_concat_full[df_concat_full['activity'].isin(['Give_Speech', 'Stationary_Bike_Hand','Stationary_Bike_Legs','Mental_Math','Rest_1', 'Rest_3', 'Rest_2', 'Prepare_Speech'])]
    return subset_df

def plot_acc_raw(acc_df):
    fig1 = acc_plots("Rest_1",acc_df[acc_df["activity"]=="Rest_1"])
    fig2 = acc_plots("Prepare_Speech",acc_df[acc_df["activity"]=="Prepare_Speech"])
    fig3 = acc_plots("Give_Speech",acc_df[acc_df["activity"]=="Give_Speech"])
    fig4 = acc_plots("Stationary_Bike",acc_df[acc_df["activity"]=="Stationary_Bike"])
    fig5 = acc_plots("Rest_2",acc_df[acc_df["activity"]=="Rest_2"])
    fig6 = acc_plots("Mental_Math",acc_df[acc_df["activity"]=="Mental_Math"])
    fig7 = acc_plots("Rest_3",acc_df[acc_df["activity"]=="Rest_3"])
    
    # Organize into figures.
    # Create a 3x2 subplot grid
    fig_combined = make_subplots(rows=3, cols=2, subplot_titles=("Rest 1", "Prepare_Speech", "Give_Speech", "Stationary Bike", " Rest 2", "Rest 3"))

    # Add each individual plot to the subplot grid
    fig_combined.add_trace(fig1['data'][0], row=1, col=1)
    fig_combined.add_trace(fig1['data'][1], row=1, col=1)
    fig_combined.add_trace(fig1['data'][2], row=1, col=1)

    fig_combined.add_trace(fig2['data'][0], row=1, col=2)
    fig_combined.add_trace(fig2['data'][1], row=1, col=2)
    fig_combined.add_trace(fig2['data'][2], row=1, col=2)

    fig_combined.add_trace(fig3['data'][0], row=2, col=1)
    fig_combined.add_trace(fig3['data'][1], row=2, col=1)
    fig_combined.add_trace(fig3['data'][2], row=2, col=1)

    fig_combined.add_trace(fig4['data'][0], row=2, col=2)
    fig_combined.add_trace(fig4['data'][1], row=2, col=2)
    fig_combined.add_trace(fig4['data'][2], row=2, col=2)

    fig_combined.add_trace(fig5['data'][0], row=3, col=1)
    fig_combined.add_trace(fig5['data'][1], row=3, col=1)
    fig_combined.add_trace(fig5['data'][2], row=3, col=1)

    fig_combined.add_trace(fig6['data'][0], row=3, col=2)
    fig_combined.add_trace(fig6['data'][1], row=3, col=2)
    fig_combined.add_trace(fig6['data'][2], row=3, col=2)

    # Update layout for the combined figure
    fig_combined.update_layout(
        height=900, width=800,  # Adjust size if needed
        title_text="Combined accelerometer data",
        showlegend=False  # Hide legend if you don't need it
    )
    return fig_combined
    # # Show the combined figure.
    # fig_combined.show()

# INPUT: activity_id_str - "Prepare_Speech"
# The function calculates sample rate based on the seconds timestamp "'14:01:10"
def calculate_sample_rate(df_sorted_clean,activity_id_str):
    average_sample_rate = df_sorted_clean[df_sorted_clean["activity"]==activity_id_str]["seconds_timestamp"].value_counts().mean()
    return average_sample_rate

# Companion run funtion to get sample rate of each activity.
# Input: A pandas df with activty column.
# Returns a dictioanry with keys as sctivity and value as sample rate.
'''
for exmpale: 
{'Stationary_Bike_Legs': 20.11,
 'Rest_3': 18.09,
 'Rest_1': 2.01,
 'Rest_2': 10.02,
 'Give_Speech': 6.97,
 'Prepare_Speech': 5.05,
 'Stationary_Bike_Hand': 24.05,
 'Mental_Math': 14.03}
'''
def get_activity_based_sr(hr_df):
    activity_list = ['Stationary_Bike_Legs', 'Rest_3',
           'Rest_1', 'Rest_2', 'Give_Speech', 'Prepare_Speech',
           'Stationary_Bike_Hand', 'Mental_Math']
    sample_rate_dict = {}
    for activity in activity_list:
        avg_acctivity_sample_rate = calculate_sample_rate(hr_df,activity)
        rounded_num = round(avg_acctivity_sample_rate, 2)
        sample_rate_dict[activity] = rounded_num
    return sample_rate_dict

####### Plot sample rate metrics.
# returns a figure object.
# takes in a sample rate dictionary : sample_rate_dict
def plot_sample_rate(sample_rate_dict,pid):
    # Extracting keys and values for plotting
    activities = list(sample_rate_dict.keys())
    durations = list(sample_rate_dict.values())
        # Extracting keys and values
    activities = list(sample_rate_dict.keys())
    durations = list(sample_rate_dict.values())
    custom_order = ["Rest_1","Prepare_Speech","Give_Speech","Rest_2","Mental_Math","Rest_3","Stationary_Bike_Legs","Stationary_Bike_Hand"]

    # Reorder based on custom_order if provided
    if custom_order:
        ordered_activities = [activity for activity in custom_order if activity in activities]
        ordered_durations = [sample_rate_dict[activity] for activity in ordered_activities]
    else:
        # Default to the original order if no custom order is provided
        ordered_activities = activities
        ordered_durations = durations
    # Creating the bar plot
    fig = go.Figure([
        go.Bar(x=ordered_activities, y=ordered_durations, marker_color='blue',text=durations,textposition='inside')
    ])

    # Adding titles and improving aesthetics
    fig.update_layout(
            title={
            'text': pid + ": Activity based sample rate (Hz).",
            'x': 0.5,  # Centers the title
            'xanchor': 'center',
            'yanchor': 'top'
        },
        # title="Activity based sample rate.",
        xaxis_title="Activities",
        yaxis_title="Sample Rate (Hz)",
        xaxis_tickangle=45,
        template="plotly_white",
        height=400,  # Set the height of the plot
        width=800    # Set the width of the plot

    )
    return fig
    # Displaying the plot
    # fig.show()
    
# Calcuate missing data.
def missing_hr_data(df):
    # Create DataFrame
    df = pd.DataFrame(df)
    # Convert watch_timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # Generate a complete time index.
    start_time = df.iloc[0]["Timestamp"] #df['watch_timestamp'].min()
    end_time = df.iloc[-1]["Timestamp"] #df['watch_timestamp'].max()
    
    print("start_time,end_time: ",start_time,end_time)
    full_time_index = pd.date_range(start=start_time, end=end_time, freq='1S')
    print("df: ",df.shape[0],"len(full_time_index): ",len(full_time_index))
    expected_number_of_samples = len(full_time_index)
    recieved_number_of_samples = df.shape[0]
    percentage_missing=  ((expected_number_of_samples-recieved_number_of_samples)/expected_number_of_samples)*100
    print("Percentage of missing data: ", percentage_missing)
    return percentage_missing

# Get missing data percentage based on timestamp. 
# missing_data_percentage = missing_hr_data(hr_df)
def get_missing_data_percentage(hr_df):
    activity_list = ['Stationary_Bike_Legs', 'Rest_3',
           'Rest_1', 'Rest_2', 'Give_Speech', 'Prepare_Speech',
           'Stationary_Bike_Hand', 'Mental_Math']
    missing_data_percentage_dict = {}
    for activity in activity_list:
        missing_data_percentage = missing_hr_data(hr_df[hr_df["activity"]==activity])
        
        rounded_num = round(missing_data_percentage, 2)
        missing_data_percentage_dict[activity] = rounded_num
    return missing_data_percentage_dict


###################################################
# Analyze missing number of samples in the HR file.
###################################################


def find_missing_samples(df, timestamp_column):
    """
    Checks whether there is data for every consecutive second in the given DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing a column with timestamps.
        timestamp_column (str): Name of the timestamp column.
    
    Returns:
        bool: True if all consecutive seconds are present, False otherwise.
        list: Missing seconds if any.
    """
    print("debug timestamp error: ",df[timestamp_column])
    # Convert the column to datetime
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
    
    # Sort by timestamp to ensure order
    df = df.sort_values(by=timestamp_column)
    
    # Generate the complete range of seconds between the minimum and maximum timestamps
    start_time = df[timestamp_column].min()
    end_time = df[timestamp_column].max()
    full_range = pd.date_range(start=start_time, end=end_time, freq='1S')
    # print("full_range",len(full_range))
    # Extract the unique seconds from the existing timestamps
    existing_seconds = df[timestamp_column].dt.floor('1S').unique()
    # print("existing_seconds: ",len(existing_seconds))
    # Find missing seconds
    missing_seconds = [time for time in full_range if time not in existing_seconds]
    
    return len(full_range),df.shape[0]#len(existing_seconds)
# Companion run function to get sample rate of each activity.
# Input: A pandas df with activty column.
# Returns a dictioanry with keys as sctivity and value as sample rate.
'''
For example:
here -- [300, 298] => [expected_samples, recieved_samples].
{'Stationary_Bike_Legs': [300, 298],
 'Rest_3': [300, 298],
 'Rest_1': [300, 298],
 'Rest_2': [300, 298],
 'Give_Speech': [300, 298],
 'Prepare_Speech': [300, 298],
 'Stationary_Bike_Hand': [300, 298],
 'Mental_Math': [300, 298]}
'''
def get_expected_recieved_samples(hr_df):
    activity_list = hr_df["activity"].unique()
    # ['Stationary_Bike_Legs', 'Rest_3', 'Rest_1', 'Rest_2',
    #    'Give_Speech', 'Prepare_Speech', 'Stationary_Bike_Hand',
    #    'Mental_Math']
    sample_rate_actual = get_activity_based_sr(hr_df)
    missing_samples_dict = {}
    for activity in activity_list:
        print("get_expected_recieved_samples()/: activity:=========",activity)
        # Get start and end times of each activity.
#         start_time = hr_df[hr_df["activity"]==activity].iloc[0]["Timestamp"] #df['watch_timestamp'].min()
#         print("get_expected_recieved_samples(): hr_df \n",hr_df[hr_df["activity"]==activity].tail())
#         end_time = hr_df[hr_df["activity"]==activity].iloc[-1]["Timestamp"] #df['watch_timestamp'].max()
        
#         print("get_expected_recieved_samples(): start_time,end_time: ",start_time,end_time)
#         print("get_expected_recieved_samples(): sample Rate activity",sample_rate_actual[activity])
#         # Calculate the difference in seconds
#         elapsed_seconds = (end_time - start_time).total_seconds()
#         expected_num_samples = int(elapsed_seconds*sample_rate_actual[activity])
#         print("get_expected_recieved_samples(): EXPECTED SAMPLES =  #Seconds * SR",expected_num_samples)
    
        #print("activity ===================== ",activity)
        #print("df activity",hr_df[hr_df["activity"]==activity])
        len_expected_samples,len_recieved_samples = find_missing_samples(hr_df[hr_df["activity"]==activity], "Timestamp") #find_missing_samples(hr_df, activity) #calculate_sample_rate(hr_df,activity)
        missing_samples_dict[activity] = [len_expected_samples,len_recieved_samples]
        
    return missing_samples_dict
# Input : missing_samples_dict in the following format.
'''
{'Stationary_Bike_Legs': [300, 298],
 'Rest_3': [300, 298],
 'Rest_1': [268, 267],
 'Rest_2': [300, 299],
 'Give_Speech': [301, 301],
 'Prepare_Speech': [300, 297],
 'Stationary_Bike_Hand': [301, 299],
 'Mental_Math': [300, 299]}

'''
def plot_expected_recieved_samples(missing_samples_dict,pid):
    # Create the grouped bar plot using Plotly
    fig = go.Figure()
    
    # Extract keys and values
    keys = list(missing_samples_dict.keys())
    values = np.array(list(missing_samples_dict.values()))
    custom_order = ["Rest_1","Prepare_Speech","Give_Speech","Rest_2","Mental_Math","Rest_3","Stationary_Bike_Legs","Stationary_Bike_Hand"]


#     # Reorder keys and values based on custom_order if provided
#     if custom_order:
#         ordered_keys = [key for key in custom_order if key in keys]
#         ordered_values = [missing_samples_dict[key] for key in ordered_keys]
        
#     else:
#         # Default to the original order if no custom order is provided
#         ordered_keys = keys
#         ordered_values = values
#         # Ensure all activities in custom_order are present in the dictionary, with defaults of [0, 0]
#     complete_samples_dict = {key: missing_samples_dict.get(key, [0, 0]) for key in custom_order}

#     # Separate the values for red and blue bars
#     ordered_values = np.array(ordered_values)
    # Ensure all activities in custom_order are present in the dictionary, with defaults of [0, 0]
    complete_samples_dict = {key: missing_samples_dict.get(key, [0, 0]) for key in custom_order}

    # Extract ordered keys and values
    ordered_keys = list(complete_samples_dict.keys())
    ordered_values = np.array(list(complete_samples_dict.values()))

    red_values = ordered_values[:, 0]
    blue_values = ordered_values[:, 1]

    # Separate the values for red and blue bars
    red_values = values[:, 0]
    blue_values = values[:, 1]
    # Add red bars for index 0 values
    fig.add_trace(go.Bar(
        x=ordered_keys,
        y=red_values,
        text=red_values,textposition='inside',
        name='Expected number of samples',
        marker_color='#3c914a'
    ))

    # Add blue bars for index 1 values
    fig.add_trace(go.Bar(
        x=ordered_keys,
        y=blue_values,
        text=blue_values,textposition='inside',
        name='Recieved number of samples',
        marker_color='#dbaa21'
    ))

    # Update layout for better visualization
    fig.update_layout(
            title={
                    'text': pid + ": Number of samples",
                    'x': 0.4,  # Centers the title
                    'xanchor': 'center',
                    'yanchor': 'top'
                },    xaxis_title='Activities',
        yaxis_title='Number of samples',
        barmode='group',
        xaxis_tickangle=45,
                height=400,  # Set the height of the plot
            width=800    # Set the width of the plot
    )
    return fig
    # Show plot
    # fig.show()
def participant_based_fig(hr_df_pid,participant_folder):
    print("participant_based_fig: ",participant_folder)
    missing_samples_dict = get_expected_recieved_samples(hr_df_pid)
    sample_rate_dict = get_activity_based_sr(hr_df_pid)
    
    fig1 = plot_expected_recieved_samples(missing_samples_dict,participant_folder)
    fig2 = plot_sample_rate(sample_rate_dict,participant_folder)
    
    # Create a subplot layout with one row and two columns
    combined_fig = make_subplots(rows=1, cols=2, subplot_titles=(participant_folder + ": Expected vs Received Samples", participant_folder+ ": Sample Rate"))

    # Add traces from fig1 to the first subplot
    for trace in fig1.data:
        combined_fig.add_trace(trace, row=1, col=1)

    # Add traces from fig2 to the second subplot
    for trace in fig2.data:
        combined_fig.add_trace(trace, row=1, col=2)

    # Update layout for aesthetics
    combined_fig.update_layout(
        title="Participant-Based Analysis",
        showlegend=True,
        height=600,  # Adjust height as needed
        width=1500,  # Adjust width as needed
    )

    # Return the combined figure
    return combined_fig

    
# Plot raw heart rate data with activity annotations.
# Input - Particiapnt based heart rate df with all the actiivties.
def plot_hr_raw_data(df,pid):
    # Get the first ouxxurence of the task
    rest_1_first_occurrence_index_df = (df['activity'] == 'Rest_1').idxmax() #df[df['activity'] == 'Rest_1'].index[0]
    prepare_speech_first_occurrence_index_df = (df['activity'] == 'Prepare_Speech').idxmax()
    give_speech_first_occurrence_index_df = (df['activity'] == 'Give_Speech').idxmax()
    rest_2_first_occurrence_index_df = (df['activity'] == 'Rest_2').idxmax()  #df[df['activity'] == 'Rest_3'].index[0]

    mental_math_first_occurrence_index_df = (df['activity'] == 'Mental_Math').idxmax()
    rest_3_first_occurrence_index_df = (df['activity'] == 'Rest_3').idxmax()  #df[df['activity'] == 'Prepare_Speech'].index[0]
    stationary_bike_hand_first_occurrence_index_df = (df['activity'] == 'Stationary_Bike_Hand').idxmax()
    bike_legs_first_occurrence_index_df = (df['activity'] == 'Stationary_Bike_Legs').idxmax() #df[df['activity'] == 'Stationary_Bike_Legs'].index[0]
    print("bike_legs_first_occurrence_index_df: ",bike_legs_first_occurrence_index_df)
    print("stationary_bike_hand_first_occurrence_index_df: ",stationary_bike_hand_first_occurrence_index_df)
    fig1 = px.line(df, y="heart_rate_float",line_shape="linear", color_discrete_sequence=['#0072BD'],template="plotly_white")

    fig1.add_vline(x=rest_1_first_occurrence_index_df,  annotation_text="Rest1", line_width=3, line_dash="solid", line_color="grey")
    fig1.add_vline(x=prepare_speech_first_occurrence_index_df, annotation_text="PS",line_width=3, line_dash="solid", line_color="grey")
    fig1.add_vline(x=give_speech_first_occurrence_index_df,annotation_text="Give Speech", line_width=3, line_dash="solid", line_color="grey")
    fig1.add_vline(x=rest_2_first_occurrence_index_df,annotation_text="Rest 2", line_width=3,line_dash="solid", line_color="grey")
    fig1.add_vline(x=mental_math_first_occurrence_index_df, annotation_text="Mental Math",line_width=3,line_dash="solid", line_color="grey")
    fig1.add_vline(x=rest_3_first_occurrence_index_df,annotation_text="Rest 3", line_width=3, line_dash="solid", line_color="grey")
    fig1.add_vline(x=stationary_bike_hand_first_occurrence_index_df,annotation_text="Bike Hand", line_width=3, line_dash="solid", line_color="grey")
    fig1.add_vline(x=bike_legs_first_occurrence_index_df, annotation_text="Bike Legs",line_width=3, line_dash="solid", line_color="grey")
    # fig1.update_yaxes(title_text='Heart Rate (bpm)', range=[40, 180], tickmode='linear', dtick=10)
    fig1.update_yaxes(title_text='Heart Rate (bpm)', range=[-10, 180], tickmode='linear', dtick=10)

    # Update layout to adjust title position and distance from plot
    fig1.update_layout(
            xaxis=dict(
            title="Time in seconds",               # Title for the x-axis
            showline=True,              # Show x-axis line
            linecolor='black',          # Color of x-axis line
            linewidth=2,                # Width of x-axis line
           # mirror=True                 # Mirror the axis line (to extend it across the top of the plot)
        ),
        yaxis=dict(
            title="Heart Rate (bpm)",    # Title for the y-axis
            showline=True,               # Show y-axis line
            linecolor='black',           # Color of y-axis line
            linewidth=2,                 # Width of y-axis line
           # mirror=True                  # Mirror the axis line (to extend it across the right side of the plot)
        ),
        title={
            'text':pid + " : Smart watch HR data in bpm",
            'x': 0.5,               # Center the title horizontally
            'xanchor': 'center',
            'y': 0.95,              # Adjust vertical position (closer to plot)
            'yanchor': 'top',        # Set anchor point to top
            'pad': {'b': 10}         # Adjust padding (decrease distance)
        }
    )
    # Show the plot
    # fig1.show()
    return fig1