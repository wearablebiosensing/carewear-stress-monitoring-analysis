###############################################################
# Appends experimenter data labels to  
###############################################################
TAG = "TASK TIMELINE MERGE"
#!/usr/bin/env python3
import pandas as pd
import glob
import os
import plotly.express as px
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np 

# Constants - 
INPUT_DIR = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/Concat_File"
INPUT_DIR_merged = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/Task_Time_Line_Belt"
OUTPUT_DIR = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/merged_lables"
error_pids = []
expected_lables_list = ['rest1', 'rest3', "stationary_Bike1",'stationary_Bike2', 'prepare speech', 'rest2'
 ,'give speech' ,'mental math']

# Define the mapping dictionary
activity_mapping = {
    'Stationary_Bike_Legs': 'stationary_Bike1',
    'Stationary_Bike_Hand': 'stationary_Bike2',
    'Prepare_Speech': 'prepare speech',
    'Rest_1': 'rest1',
    'Rest_2': 'rest2',
    'Rest_3': 'rest3',
    'Give_Speech': 'give speech',
    'Mental_Math': 'mental math',
    'activity': 'activity',
    'None': 'None',
    "Stationary_Bike_ke_Hand":"stationary_Bike2",
    "Stationary_BiHand":"stationary_Bike2",
    np.nan: np.nan  # If needed
}

# Improved mapping function for single values
def map_activity_value(x):
    return activity_mapping.get(x, x)

# Function to map the list of activities (if needed)
def map_activities(activity_list):
    if isinstance(activity_list, list):
        return [activity_mapping.get(act, act) for act in activity_list]
    return activity_list  # Leave unchanged if not a list

def count_class_samples(df, class_column):
    """
    Returns the count of samples in each unique class in the specified column,
    excluding the class 'None' (as a string or actual None/NaN).
    """
    if class_column not in df.columns:
        raise ValueError(f"Column '{class_column}' not found in DataFrame.")
    
    # Drop rows with NaN or string 'None' in class column
    filtered = df[~df[class_column].isin(['None']) & df[class_column].notna()]
    return filtered[class_column].value_counts()

# Input : hr_df: Raw HR df, labels_df: with start and end timestamps.
# Output:
def add_activity_labels(hr_df, labels_df, base_date, label_col_name, activity_col):
    """Assigns activity labels to hr_df based on start/end times from labels_df."""
    # Convert and align label times
    for col in ('start_time', 'end_time'):
        labels_df[col] = pd.to_datetime(labels_df[col],utc=True)
        labels_df[col] = labels_df[col].dt.tz_localize(None)
        labels_df[col] = labels_df[col].apply(lambda ts: ts.replace(
            year=base_date.year,
            month=base_date.month,
            day=base_date.day
        ) if pd.notnull(ts) else ts)

    # Add column if not present
    if label_col_name not in hr_df.columns:
        hr_df[label_col_name] = "None"

    # Assign labels
    for _, row in labels_df.iterrows():
        start, stop, activity = row['start_time'], row['end_time'], row[activity_col]
        if pd.isna(start) or pd.isna(stop):
            continue
        mask = (hr_df['Timestamp_pd'] >= start) & (hr_df['Timestamp_pd'] <= stop)
        if not mask.any():
            i0 = (hr_df['Timestamp_pd'] - start).abs().idxmin()
            i1 = (hr_df['Timestamp_pd'] - stop ).abs().idxmin()
            mask = (hr_df.index >= i0) & (hr_df.index <= i1)
        hr_df.loc[mask, label_col_name] = activity

# num_labels_json = {}
num_labels_json = {str(i): 0 for i in range(1, 26)}

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pattern = os.path.join(INPUT_DIR, "heart_rate_P*.csv")
    activity_records = []  # For plotting
    per_participant_activity_json = {}  # For saving JSON
    for hr_path in sorted(glob.glob(pattern)):
        fname = os.path.basename(hr_path)
        pid = fname.split('_P')[-1].split('.csv')[0]
        print(f"\n→ Processing P{pid}")
        hr_df = pd.read_csv(hr_path)
        # Map activity column right after loading (if present)
        if 'activity' in hr_df.columns:
            hr_df['activity'] = hr_df['activity'].apply(map_activity_value)

        hr_df['Timestamp_pd'] = pd.to_datetime(hr_df['Timestamp_pd'], utc=False)
        base_date = hr_df['Timestamp_pd'].min()
        print("hr_df unique() activity ============================== ",hr_df["activity"].unique())
        # Add Belt labels
        belt_labels_path = os.path.join(
            "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/Task_Time_Line_Belt",
            f"belt_task_timeline_P{pid}.csv")
        if os.path.exists(belt_labels_path):
            belt_labels_df = pd.read_csv(belt_labels_path)
            add_activity_labels(hr_df, belt_labels_df, base_date, "Belt_Activity_Labels", activity_col='Belt_Activity_Labels')
            # Mapping will be done after all label assignments

            print("Unique belt activity merged = ",hr_df["Belt_Activity_Labels"].unique())
        else:
            print(f"[P{pid}] ⚠️ missing belt_task_timeline_P{pid}.csv")

        # Add Manual labels
        manual_labels_path = os.path.join(
            "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/Task_Time_Line_Manual",
            f"manual_task_timeline_P{pid}.csv")
        if os.path.exists(manual_labels_path):
            manual_labels_df = pd.read_csv(manual_labels_path)
            add_activity_labels(hr_df, manual_labels_df, base_date, "manual_labels_activity", activity_col='manual_labels_activity')
            # Mapping will be done after all label assignments

            print("Unique watch activity merged = ",hr_df["manual_labels_activity"].unique())
            print("Number of activites matched ======",len(hr_df["manual_labels_activity"].unique()))
            num_labels_json[int(pid)]=len(hr_df["manual_labels_activity"].unique())
            print("hr_df.column")
            print(hr_df.columns)
            num_samples_each_class = count_class_samples(hr_df, "manual_labels_activity")

            print("Number of samples in each class ====")
            print(num_samples_each_class)
            per_participant_activity_json[f"P{pid}"] = num_samples_each_class.to_dict()

            # Collect data for plotting
            for activity, count in num_samples_each_class.items():
                activity_records.append({
                    "Participant": f"P{pid}",
                    "Activity": activity,
                    "Samples": count
                })
            # Save full JSON with sample counts per activity
            json_output_path = os.path.join(OUTPUT_DIR, "samples_per_activity_per_participant.json")
            with open(json_output_path, 'w') as f:
                json.dump(per_participant_activity_json, f, indent=4)
            print(f"✅ Saved activity sample counts JSON → {json_output_path}")
        else:
            # same as activity recorded.
            hr_df["manual_labels_activity"] = hr_df["activity"]
            num_labels_json[int(pid)]=len(hr_df["manual_labels_activity"].unique())

            num_samples_each_class = count_class_samples(hr_df, "manual_labels_activity")
            # Collect data for plotting
            for activity, count in num_samples_each_class.items():
                activity_records.append({
                    "Participant": f"P{pid}",
                    "Activity": activity,
                    "Samples": count
                })
            print(hr_df.head())
            print(f"[P{pid}] ⚠️ missing manual_task_timeline_P{pid}.csv")
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >>>>>>>> CONSISTENT ACTIVITY MAPPING FOR ALL RELEVANT COLUMNS <<<<<<<<
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        for col in ['activity', 'manual_labels_activity', 'Belt_Activity_Labels']:
            if col in hr_df.columns:
                hr_df[col] = hr_df[col].apply(map_activity_value)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # Save output
        out_csv = os.path.join(OUTPUT_DIR, f"heart_rate_{pid}_merged_labels.csv")    
        hr_df.to_csv(out_csv, index=False)
        print(f"[P{pid}] saved → {out_csv}")

print(num_labels_json)
num_labels_json_sorted = dict(sorted(num_labels_json.items(), key=lambda item: int(item[0])))

# Convert dictionary to lists for plotting
x = list(num_labels_json_sorted.keys())
y = list(num_labels_json_sorted.values())

# First plot: Quality Score per Participant
trace1 = go.Bar(
    x=x,
    y=y,
    text=y,
    textposition='outside',
    marker_color='rgba (7, 55, 99, 100)',  # Hex code for teal
    name="number of labels"
)

# ========================== Create Grouped Bar Plot (Activity as Group, Sorted) ==========================
if activity_records:
    df_plot = pd.DataFrame(activity_records)

    # Remove participant P10
    df_plot = df_plot[df_plot['Participant'] != 'P10']

    # Extract numeric part of participant ID for sorting
    df_plot['Participant_num'] = df_plot['Participant'].str.replace('P', '').astype(int)

    # Sort participants by numeric ID (ascending)
    sorted_participants = (
        df_plot[['Participant', 'Participant_num']]
        .drop_duplicates()
        .sort_values('Participant_num')
        ['Participant']
        .tolist()
    )

    # Set 'Participant' as a categorical variable with the correct order
    df_plot['Participant'] = pd.Categorical(df_plot['Participant'], categories=sorted_participants, ordered=True)

    # Sort DataFrame by participant order (optional, but helps with debugging)
    df_plot = df_plot.sort_values('Participant')

    # Create the plot
    fig2 = px.bar(
        df_plot,
        x="Participant",
        y="Samples",
        color="Activity",
        barmode="group",
        category_orders={"Participant": sorted_participants},
        title="Sample Count per Participant (Grouped by Activity)",
        height=600,
        width=1200
    )

    fig2.update_layout(
        plot_bgcolor='white',
        title_font=dict(size=22),
        legend_title="Activity",
        font=dict(size=16),
        xaxis_title="Participant ID",
        yaxis_title="Sample Count"
    )

    fig2.show()# Create subplots
fig = make_subplots(
    rows=1, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.15,
    subplot_titles=("Number of merged labels")
)
fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=22),row=1, col=1, title="")
fig.update_xaxes(title="Participant ID", title_font=dict(size=20), tickfont=dict(size=22),showticklabels=True, row=1, col=1)

# Add traces
fig.add_trace(trace1, row=1, col=1)

# Update layout for clarity
fig.update_layout(
    height=500,
    width = 1200,
    showlegend=True,
    barmode="group",
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    title_font=dict(size=20), 
    plot_bgcolor='lightgray',      # Plotting area background
    # paper_bgcolor='lightgray'  # Whole figure background
)
# Show the plot
fig.show()
