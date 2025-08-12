import os
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import re
import numpy as np 
from plotly.subplots import make_subplots
from ecg_biopac_neurokit import get_ecg_hr_neurokit

# === CONFIGURATION ===
DATA_FOLDER = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/other/merged_lables/"
CONCAT_FOLDER = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/Concat_File"

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
    np.nan: np.nan
}

activity_mapping_num = {
    "None":-1,
    'rest1': 1,
    'prepare speech': 2,
    'give speech': 3,
    'rest2': 4,
    'mental math': 5,
    'rest3': 6,
    'stationary_Bike1': 7,
    'stationary_Bike2': 8,
    np.nan: np.nan
}

def map_activities(activity_list):
    if isinstance(activity_list, list):
        return [activity_mapping.get(act, act) for act in activity_list]
    return activity_list

def load_all_participant_data(folder):
    data = {}
    for file in os.listdir(folder):
        # read only - heart_rate_19_merged_labels.csv
        if "heart" in file.lower() and file.endswith(".csv"):
            participant_id = os.path.splitext(file)[0]
            print("Manual activity lables",os.path.join(folder, file))
            df = pd.read_csv(os.path.join(folder, file))
            expected_lables_list = ['rest1', 'rest3', "stationary_Bike1",'stationary_Bike2', 'prepare speech', 'rest2','give speech' ,'mental math']
            df = df[df['manual_labels_activity'].isin(expected_lables_list)]
            df['manual_labels_numeric'] = df['manual_labels_activity'].map(activity_mapping_num)
            df["HeartRate"] = pd.to_numeric(df["HeartRate"], errors="coerce")
            df["Timestamp_pd"] = pd.to_datetime(df["Timestamp_pd"])
            df = df.sort_values("Timestamp_pd")
            data[participant_id] = df
    return data

def load_ecg_biopac(folder):
    data = {}
    for file in os.listdir(folder):
        if "biopac" in file:
            participant_id = os.path.splitext(file)[0]
            df = pd.read_csv(os.path.join(folder, file))
            data[participant_id] = df
    return data

def extract_participant_number(pid):
    match = re.search(r'(\d+)', pid)
    return int(match.group(1)) if match else float('inf')

ecg_biopac = load_ecg_biopac(CONCAT_FOLDER)
participant_ids_ecg = sorted(ecg_biopac.keys(), key=extract_participant_number)

participant_data = load_all_participant_data(DATA_FOLDER)
participant_ids = sorted(participant_data.keys(), key=extract_participant_number)

app = Dash(__name__)
app.layout = html.Div([
    html.H2("Heart Rate Timeline Dashboard"),
    dcc.Dropdown(
        id='participant-dropdown',
        options=[{'label': pid, 'value': pid} for pid in participant_ids],
        value=participant_ids[0]
    ),
    html.Div([
        dcc.Checklist(
            id='outlier-toggle',
            options=[{'label': 'Remove HR Outliers (<30 or >220 bpm)', 'value': 'remove'}],
            value=[],
            style={'marginTop': '10px'}
        )
    ]),
    dcc.Dropdown(
        id='participant-dropdown-ecg',
        options=[{'label': pid, 'value': pid} for pid in participant_ids_ecg],
        value=participant_ids_ecg[0]
    ),
    html.Br(), html.Br(), html.Br(), html.Br(),
    dcc.Graph(id='hr-graph')
])

@app.callback(
    Output('hr-graph', 'figure'),
    Input('participant-dropdown', 'value'),
    Input('participant-dropdown-ecg', 'value'),
    Input('outlier-toggle', 'value')
)
def update_graph(participant_id, participant_id_ecg, outlier_toggle):
    df = participant_data[participant_id].copy()
    df_ecg = ecg_biopac[participant_id_ecg].copy()

    df_ecg = df_ecg[~df_ecg.astype(str).apply(lambda row: row.str.contains("samples", case=False, na=False)).any(axis=1)]
    df_ecg["ECG_X"] = df_ecg["ECG_X"].astype(float)
    df_ecg["Heart_Rate"] = df_ecg["Heart_Rate"].astype(float)
    df_ecg["milliSec"] = df_ecg["milliSec"].astype(float)

    resampled_hr, peak_indices, num_rr_intervals_short = get_ecg_hr_neurokit(df_ecg)
    peak_values = df_ecg["ECG_X"].iloc[peak_indices].values

    if 'remove' in outlier_toggle:
        df = df[(df["HeartRate"] >= 30) & (df["HeartRate"] <= 220)]
        resampled_hr = resampled_hr[(resampled_hr >= 30) & (resampled_hr <= 220)]

    fig = make_subplots(rows=4, cols=1,
                        subplot_titles=("Heart rate plot from Galaxy watch", "ECG in mV from biopac",
                                        "HR form ECG biopac calculated (ours)" + " Num RR <250 ms =" + str(num_rr_intervals_short),
                                        "HR form ECG provided from biopac"))
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=28)

    fig.add_trace(go.Scatter(x=df["Timestamp_pd"], y=df["HeartRate"], mode='lines', name='Heart Rate', line=dict(color='blue')), row=1, col=1)

    for label, group in df.groupby("manual_labels_numeric"):
        if pd.isna(label):
            continue
        start_time = group["Timestamp_pd"].min()
        end_time = group["Timestamp_pd"].max()
        mid_time = start_time + (end_time - start_time) / 2

        fig.add_vline(x=start_time, line=dict(color='green', dash='dot'), row=1, col=1)
        fig.add_vline(x=end_time, line=dict(color='red', dash='dot'), row=1, col=1)
        fig.add_annotation(x=mid_time, y=max(df["HeartRate"]) + 5 if not df["HeartRate"].empty else 0,
                           text=str(label), showarrow=False, yanchor='bottom', font=dict(size=34),
                           bgcolor="white", opacity=0.8, xref="x1", yref="y1")

    fig.add_trace(go.Scatter(y=df_ecg["ECG_X"], mode='lines', name='ECG', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=peak_indices, y=peak_values, mode='markers', name='R-peaks',
                             marker=dict(color='red', size=8, symbol='x')), row=2, col=1)
    fig.add_trace(go.Scatter(y=resampled_hr, mode='lines', name='ECG-HR in bpm', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_ecg.index, y=df_ecg["Heart_Rate"], mode='lines', name='Heart Rate',
                             line=dict(color='blue')), row=4, col=1)

    activity_boundaries = (df_ecg.reset_index()
        .groupby("activity")["index"]
        .agg(["first", "last"])
        .rename(columns={"first": "start_idx", "last": "end_idx"})
        .reset_index())

    for _, row_act in activity_boundaries.iterrows():
        activity = row_act["activity"]
        start_idx = row_act["start_idx"]
        end_idx = row_act["end_idx"]
        mid_idx = start_idx + (end_idx - start_idx) // 2

        fig.add_vline(x=start_idx, line=dict(color='green', dash='dot'), row=4, col=1)
        fig.add_vline(x=end_idx, line=dict(color='red', dash='dot'), row=4, col=1)
        fig.add_annotation(x=mid_idx, y=max(df_ecg["Heart_Rate"]) + 5 if not df_ecg["Heart_Rate"].empty else 0,
                           text=str(activity), showarrow=False, yanchor='bottom', font=dict(size=14),
                           bgcolor="white", opacity=0.8, xref="x4", yref="y4")

    fig.update_layout(height=1400, width=2000, showlegend=True, template="plotly_white")

    axis_titles = ["Time", "Samples", "Samples", "Samples"]
    y_titles = ["Heart Rate (BPM)", "ECG (mV)", "Heart Rate in (bpm)", "Heart Rate in (bpm)"]

    for i in range(1, 5):
        fig.update_xaxes(showline=True, title_text=axis_titles[i-1], title_font=dict(size=28), tickfont=dict(size=22), row=i, col=1)
        fig.update_yaxes(showline=True, title_text=y_titles[i-1], title_font=dict(size=28), tickfont=dict(size=22), row=i, col=1)

    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8051)

# import os
# import pandas as pd
# import plotly.graph_objs as go
# from dash import Dash, dcc, html, Input, Output
# import re
# import numpy as np 
# from plotly.subplots import make_subplots
# # from smartwatch_modules.ecg_biopac_hrpy import *
# # from smartwatch_modules.ecg_biopac_neurokit import *
# from ecg_biopac_neurokit import *

# # === CONFIGURATION ===
# DATA_FOLDER = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/merged_lables/"
# CONCAT_FOLDER = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/Concat_File"
# # Define the mapping dictionary
# activity_mapping = {
#     'Stationary_Bike_Legs': 'stationary_Bike1',
#     'Stationary_Bike_Hand': 'stationary_Bike2',
#     'Prepare_Speech': 'prepare speech',
#     'Rest_1': 'rest1',
#     'Rest_2': 'rest2',
#     'Rest_3': 'rest3',
#     'Give_Speech': 'give speech',
#     'Mental_Math': 'mental math',
#     'activity': 'activity',
#     'None': 'None',
#     "Stationary_Bike_ke_Hand":"stationary_Bike2",
#     "Stationary_BiHand":"stationary_Bike2",
#     np.nan: np.nan  # If needed
# }

# activity_mapping_num = {
#     "None":-1,
#     'rest1': 1,
#     'prepare speech': 2,
#     'give speech': 3,
#     'rest2': 4,
#     'mental math': 5,
#     'rest3': 6,
#     'stationary_Bike1': 7,
#     'stationary_Bike2': 8,
#     np.nan: np.nan  # If needed
# }
# # Function to map the list of activities
# def map_activities(activity_list):
#     if isinstance(activity_list, list):
#         return [activity_mapping.get(act, act) for act in activity_list]
#     return activity_list  # Leave unchanged if not a list

# # === LOAD DATA ===
# def load_all_participant_data(folder):
#     data = {}
#     for file in os.listdir(folder):
#         if file.endswith(".csv"):
#             participant_id = os.path.splitext(file)[0]
#             df = pd.read_csv(os.path.join(folder, file))
#             # print("PID = ",participant_id)
#             # print("================== manual_labels_activity ==================")
#             # print(df["manual_labels_activity"].unique())
#             expected_lables_list = ['rest1', 'rest3', "stationary_Bike1",'stationary_Bike2', 'prepare speech', 'rest2'
#             ,'give speech' ,'mental math']
#             df = df[df['manual_labels_activity'].isin(expected_lables_list)]
#             # df["manual_labels_num"] = activity_mapping_num
#             # Suppose your column is named 'manual_labels_activity'
#             df['manual_labels_numeric'] = df['manual_labels_activity'].map(activity_mapping_num)
#             df["HeartRate"] = pd.to_numeric(df["HeartRate"], errors="coerce")
#             df["Timestamp_pd"] = pd.to_datetime(df["Timestamp_pd"])
#             df = df.sort_values("Timestamp_pd")
#             data[participant_id] = df
#     return data

# def load_ecg_biopac(folder):
#     data = {}
#     for file in os.listdir(folder):
#         if "biopac" in file:
#             participant_id = os.path.splitext(file)[0]
#             df = pd.read_csv(os.path.join(folder, file))
#             data[participant_id] = df
#     return data

# def extract_participant_number(pid):
#     match = re.search(r'(\d+)', pid)
#     return int(match.group(1)) if match else float('inf')
# #"ECG_X"
# ecg_biopac = load_ecg_biopac(CONCAT_FOLDER)
# participant_ids_ecg = sorted(ecg_biopac.keys(), key=extract_participant_number)

# participant_data = load_all_participant_data(DATA_FOLDER)
# participant_ids = sorted(participant_data.keys(), key=extract_participant_number)

# # === DASH APP ===
# app = Dash(__name__)
# app.layout = html.Div([
#     html.H2("Heart Rate Timeline Dashboard"),
#     dcc.Dropdown(
#         id='participant-dropdown',
#         options=[{'label': pid, 'value': pid} for pid in participant_ids],
#         value=participant_ids[0]
#     ),
#     html.Div([
#         dcc.Checklist(
#             id='outlier-toggle',
#             options=[{'label': 'Remove HR Outliers (<30 or >220 bpm)', 'value': 'remove'}],
#             value=[],
#             style={'marginTop': '10px'}
#         )
#     ]),
#     dcc.Dropdown(
#         id='participant-dropdown-ecg',
#         options=[{'label': pid, 'value': pid} for pid in participant_ids_ecg],
#         value=participant_ids_ecg[0]
#     ),
#     html.Br(),
#     html.Br(),
#     html.Br(),
#     html.Br(),

#     dcc.Graph(id='hr-graph')
# ])

# # === CALLBACK ===
# @app.callback(
#     Output('hr-graph', 'figure'),
#     Input('participant-dropdown', 'value'),
#     Input('participant-dropdown-ecg', 'value'),
#     Input('outlier-toggle', 'value')
# )
# def update_graph(participant_id, participant_id_ecg,outlier_toggle):
#     ecg_pid = participant_id_ecg.split("_")[0]
    
#     df = participant_data[participant_id].copy()
#     df_ecg = ecg_biopac[participant_id_ecg].copy()

#     # Filter out rows that contain 'samples' in any column
#     df_ecg = df_ecg[~df_ecg.astype(str).apply(lambda row: row.str.contains("samples", case=False, na=False)).any(axis=1)]

#     # Filter out rows that contain "samples"
#     # df_ecg = df[~df['column_name'].str.contains("samples", case=False, na=False)]


#     df_ecg["ECG_X"] = df_ecg["ECG_X"].astype(float)
#     df_ecg["Heart_Rate"] = df_ecg["Heart_Rate"].astype(float)
#     df_ecg["milliSec"] = df_ecg["milliSec"].astype(float)
#     # resampled_hr, peak_indices,num_rr_intervals_short = get_ecg_hr(df_ecg)
#     resampled_hr, peak_indices,num_rr_intervals_short = get_ecg_hr_neurokit(df_ecg)
#     peak_values = df_ecg["ECG_X"].iloc[peak_indices].values
    
#     # df_raw = data_raw[participant_id].copy()
#     # Remove outliers if checkbox is selected
#     # fig = make_subplots(rows=1, cols=3, subplot_titles=('Plot 1', 'Plot 2', 'Plot 3'))

#     if 'remove' in outlier_toggle:
#         df = df[(df["HeartRate"] >= 30) & (df["HeartRate"] <= 220)]# Create 1 row and 2 column subplot
#         resampled_hr = resampled_hr[(resampled_hr >= 30) & (resampled_hr <= 220)]

#     fig = make_subplots(rows=4, cols=1,
#                         subplot_titles=("Heart rate plot from Galaxy watch", "ECG in mV from biopac","HR form ECG biopac calculated (ours)" + " Num RR <250 ms =" + str(num_rr_intervals_short),"HR form ECG provided from biopac"),                                                 
#                         # vertical_spacing=0.23,
#                         # row_heights=[0.25, 0.25, 0.25, 0.25]   # adjust the proportion of space each subplot takes

#                         )  # default is 0.02; increase for more spacing
#     for annotation in fig['layout']['annotations']:
#         annotation['font'] = dict(size=28)  # or 20, 24, etc.


#     # Main HR Line in first subplot
#     fig.add_trace(
#         go.Scatter(
#             x=df["Timestamp_pd"],
#             y=df["HeartRate"],
#             mode='lines',
#             name='Heart Rate',

#             line=dict(color='blue')
#         ),
#         row=1, col=1
#     )
#     # Add vertical lines and annotations in first subplot
#     for label, group in df.groupby("manual_labels_numeric"):
#         if pd.isna(label):
#             continue
#         start_time = group["Timestamp_pd"].min()
#         end_time = group["Timestamp_pd"].max()
#         mid_time = start_time + (end_time - start_time) / 2

#         fig.add_vline(
#             x=start_time,
#             line=dict(color='green', dash='dot'),
#             row=1, col=1
#         )
#         fig.add_vline(
#             x=end_time,
#             line=dict(color='red', dash='dot'),
#             row=1, col=1
#         )
#         fig.add_annotation(
#             x=mid_time,
#             y=max(df["HeartRate"]) + 5 if not df["HeartRate"].empty else 0,
#             text=str(label),
#             showarrow=False,
#             yanchor='bottom',
#             font=dict(size=34),
#             bgcolor="white",
#             opacity=0.8,
#             xref="x1",
#             yref="y1"
#         )
#     fig.add_trace(
#         go.Scatter(
#             # x = df_ecg["milliSec"],
#             y=df_ecg["ECG_X"],
#             mode='lines',
#             name='ECG',
#             line=dict(color='blue')
#         ),
#         row=2, col=1
#     )
#     # --- Add R-peak markers ---
#     fig.add_trace(
#         go.Scatter(
#             x=peak_indices,
#             y=peak_values,
#             mode='markers',
#             name='R-peaks',
#             marker=dict(color='red', size=8, symbol='x')
#         ),
        
#         row=2, col=1
#     )
#     fig.add_trace(
#         go.Scatter(
#             y=resampled_hr,
#             mode='lines',
#             name='ECG-HR in bpm',
#             line=dict(color='blue')
#         ),
#         row=3, col=1
#     )
#     # Add Heart Rate Line to Row 4
#     fig.add_trace(
#         go.Scatter(
#             x=df_ecg.index,  # sample indices
#             y=df_ecg["Heart_Rate"],
#             mode='lines',
#             name='Heart Rate',
#             line=dict(color='blue')
#         ),
#         row=4, col=1
#     )

#     # Optional: Clean out invalid HR values if needed
#     # df_ecg = df_ecg[(df_ecg["Heart_Rate"] >= 30) & (df_ecg["Heart_Rate"] <= 220)]

#     # Compute first and last index of each activity
#     activity_boundaries = (
#         df_ecg.reset_index()
#             .groupby("activity")["index"]
#             .agg(["first", "last"])
#             .rename(columns={"first": "start_idx", "last": "end_idx"})
#             .reset_index()
#     )

#     # Add vertical lines and activity annotations
#     for _, row_act in activity_boundaries.iterrows():
#         activity = row_act["activity"]
#         start_idx = row_act["start_idx"]
#         end_idx = row_act["end_idx"]
#         mid_idx = start_idx + (end_idx - start_idx) // 2

#         fig.add_vline(
#             x=start_idx,
#             line=dict(color='green', dash='dot'),
#             row=4, col=1
#         )
        
#         fig.add_vline(
#             x=end_idx,
#             line=dict(color='red', dash='dot'),
#             row=4, col=1
#         )

#         fig.add_annotation(
#             x=mid_idx,
#             y=max(df_ecg["Heart_Rate"]) + 5 if not df_ecg["Heart_Rate"].empty else 0,
#             text=str(activity),
#             showarrow=False,
#             yanchor='bottom',
#             font=dict(size=14),
#             bgcolor="white",
#             opacity=0.8,
#             xref="x4",  # reference x-axis of row 4
#             yref="y4"
#         )



#     # Update layout
#     fig.update_layout(
#         height=1400,  # increased from 600
#         width=2000,
#         title_text="",
#         showlegend=True,
#         # font=dict(size=34),
#         # margin=dict(l=40, r=40, t=40, b=40),
#         template="plotly_white",

#     )

#     fig.update_xaxes(
#         showline=True,
#         title_text="Time",
#         title_font=dict(size=28),  # 👈 Font size for X-axis title
#         tickfont=dict(size=22),

#         row=1, col=1
#     )
#     fig.update_yaxes(
#         showline=True,
#         title_text="Heart Rate (BPM)",
#         title_font=dict(size=28),  # 👈 Font size for Y-axis title
#         tickfont=dict(size=22),
#         row=1, col=1
#     )

#     fig.update_xaxes(
#         showline=True,
#         title_text="Samples",
#         title_font=dict(size=28),
#         tickfont=dict(size=22),
#         row=2, col=1
#     )
#     fig.update_yaxes(
#         showline=True,
#         title_text="ECG (mV)",
#         title_font=dict(size=28),
#         tickfont=dict(size=22),
#         row=2, col=1
#     )
#     fig.update_xaxes(
#             showline=True,
#             title_text="Samples",
#             title_font=dict(size=28),
#             tickfont=dict(size=22),
#             row=3, col=1
#         )
#     fig.update_yaxes(
#             showline=True,
#             title_text="Heart Rate in (bpm)",
#             title_font=dict(size=28),
#             tickfont=dict(size=22),
#             row=3, col=1
#     )
#     fig.update_xaxes(
#             showline=True,
#             title_text="Samples",
#             title_font=dict(size=28),
#             tickfont=dict(size=22),
#             row=4, col=1
#         )
#     fig.update_yaxes(
#             showline=True,
#             title_text="Heart Rate in (bpm)",
#             title_font=dict(size=28),
#             tickfont=dict(size=22),
#             row=4, col=1
#     )

#     return fig

# # === RUN ===
# if __name__ == '__main__':
#     app.run(debug=True, port=8051)

# #    app.run_server(debug=True, port=8051)
