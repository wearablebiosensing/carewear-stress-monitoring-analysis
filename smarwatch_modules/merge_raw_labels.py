###############################################################
# Appends experimenter data labels to  
###############################################################
TAG = "TASK TIMELINE MERGE"

import pandas as pd
import os
import glob
error_pid_list = []
def merge_labels(hr_data,labels,pid):
    # Convert timestamps to datetime objects
    hr_data['Timestamp_pd'] = pd.to_datetime(hr_data['Timestamp_pd'])

    # Convert labels timestamps to datetime objects with the same date as the earliest heart rate data
    base_date = hr_data['Timestamp_pd'].min().date()
    labels['start'] = pd.to_datetime(labels['start'], format='%I:%M %p').apply(lambda x: x.replace(year=base_date.year, month=base_date.month, day=base_date.day))
    labels['stop'] = pd.to_datetime(labels['stop'], format='%I:%M %p').apply(lambda x: x.replace(year=base_date.year, month=base_date.month, day=base_date.day))

    # Add a new column for activity labels initialized with None
    hr_data['Activity_Lables'] = "None"

    # Assign activity labels based on start and stop times
    for _, row in labels.iterrows():
        start_time = row['start']
        stop_time = row['stop']
        activity_label = row['activity']
        
        if pd.isnull(start_time) or pd.isnull(stop_time):
            continue
        
        # Find rows within the start and stop time range
        mask = (hr_data['Timestamp_pd'] >= start_time) & (hr_data['Timestamp_pd'] <= stop_time)
        
        # If no exact match exists, find nearest timestamps
        if not mask.any():
            nearest_start_idx = (hr_data['Timestamp_pd'] - start_time).abs().idxmin()
            nearest_stop_idx = (hr_data['Timestamp_pd'] - stop_time).abs().idxmin()
            
            # Assign activity label to rows between nearest start and stop times
            mask = (hr_data.index >= nearest_start_idx) & (hr_data.index <= nearest_stop_idx)
        
        # Assign activity label to matching rows
        hr_data.loc[mask, 'Activity_Lables'] = activity_label
    print("Unique lables \n",hr_data["Activity_Lables"].unique())
    if len(hr_data["Activity_Lables"].unique()) == 9:
        # Save the updated heart rate data with activity labels to a new file
        output_file = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/merged_lables/heart_rate_"+pid+"_merged_lables.csv"
        hr_data.to_csv(output_file, index=False)
        print(f"Updated file saved as {output_file}")
    else:
        error_pid_list.append(pid)
        print("Error in PID",pid)   

        print("All lables did not get appended")
        print("Check")

raw_data_root = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/Concat_File/"
labels_root = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/"
belt_lables_root = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/Task_Time_Line_Belt/"
file_type = 'csv'  # Change to 'xlsx', 'json', etc., as needed.
file_pattern = os.path.join(raw_data_root, f'*.{file_type}')
files_concat = glob.glob(file_pattern)

file_pattern_lables = os.path.join(belt_lables_root, f'*.{file_type}')
files_concat_lables = glob.glob(file_pattern_lables)
print("files_concat_lables: ",files_concat_lables)
# print("files_concat",files_concat)
for filepath in files_concat:
    if "heart" in filepath :
        pid = filepath.split("/")[-1].split(".")[0].split("_")[-1]
        print("pid ============ ",pid)
        # Load the raw heart rate data
        hr_data = pd.read_csv(filepath)
        # Load the labels file
        labels = pd.read_csv(belt_lables_root + "labels.csv")

        # labels_subset = labels[labels["PID"]==pid]
        # files_concat
        merge_labels(hr_data,labels_subset,pid)
with open(labels_root + "error_log_merge.txt", "w") as file:
    file.write("\n".join(error_pid_list))     