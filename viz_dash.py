
import os
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import re
import numpy as np
from plotly.subplots import make_subplots
from biopac_modules.ecg_biopac_neurokit import get_ecg_hr_neurokit

# === CONFIGURATION ===
DATA_FOLDER = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/other/merged_lables/"
CONCAT_FOLDER = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/Concat_File"
RESAMPLED_FOLDER = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/other/resampled_HR"

activity_mapping_num = {
    "None": -1,
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


# === Helper functions ===
def extract_participant_number(pid):
    match = re.search(r'(\d+)', pid)
    return int(match.group(1)) if match else float('inf')


def load_all_participant_data(folder):
    data = {}
    expected_lables_list = [
        'rest1', 'rest3', "stationary_Bike1", 'stationary_Bike2',
        'prepare speech', 'rest2', 'give speech', 'mental math'
    ]
    for file in os.listdir(folder):
        if "heart" in file.lower() and file.endswith(".csv"):
            participant_id = os.path.splitext(file)[0]
            df = pd.read_csv(os.path.join(folder, file))
            df = df[df['manual_labels_activity'].isin(expected_lables_list)]
            df['manual_labels_numeric'] = df['manual_labels_activity'].map(activity_mapping_num)
            df["HeartRate"] = pd.to_numeric(df["HeartRate"], errors="coerce").fillna(-1)
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


# === Load base data ===
ecg_biopac = load_ecg_biopac(CONCAT_FOLDER)
participant_ids_ecg = sorted(ecg_biopac.keys(), key=extract_participant_number)


participant_data = load_all_participant_data(DATA_FOLDER)
participant_ids = sorted(participant_data.keys(), key=extract_participant_number)


# === Load available resampled files ===
resampled_biopac_files = sorted(
    [f for f in os.listdir(RESAMPLED_FOLDER) if f.endswith("_biopac.csv")]
)
resampled_watch_files = sorted(
    [f for f in os.listdir(RESAMPLED_FOLDER) if f.startswith("heart_rate_") and f.endswith("_merged_labels.csv")]
)


# === Dash App ===
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

    html.Br(),

    html.Div([
        html.Label("Select Resampled Biopac File:"),
        dcc.Dropdown(
            id='resampled-biopac-dropdown',
            options=[{'label': f, 'value': f} for f in resampled_biopac_files],
            value=resampled_biopac_files[0] if resampled_biopac_files else None
        )
    ]),

    html.Div([
        html.Label("Select Resampled Smartwatch File:"),
        dcc.Dropdown(
            id='resampled-watch-dropdown',
            options=[{'label': f, 'value': f} for f in resampled_watch_files],
            value=resampled_watch_files[0] if resampled_watch_files else None
        )
    ]),

    html.Br(),

    dcc.Graph(id='hr-graph')
])


# === Callback ===
@app.callback(
    Output('hr-graph', 'figure'),
    [
        Input('participant-dropdown', 'value'),
        Input('participant-dropdown-ecg', 'value'),
        Input('outlier-toggle', 'value'),
        Input('resampled-biopac-dropdown', 'value'),
        Input('resampled-watch-dropdown', 'value')
    ]
)
def update_graph(participant_id, participant_id_ecg, outlier_toggle, selected_biopac_file, selected_watch_file):
    # Load smartwatch data and labels
    df = participant_data[participant_id].copy()

    # Load raw biopac ECG
    df_ecg = ecg_biopac[participant_id_ecg].copy()
    df_ecg = df_ecg[~df_ecg.astype(str).apply(lambda row: row.str.contains("samples", case=False, na=False)).any(axis=1)]

    # Convert types
    df_ecg["ECG_X"] = df_ecg["ECG_X"].astype(float)
    df_ecg["Heart_Rate"] = df_ecg["Heart_Rate"].astype(float)
    df_ecg["milliSec"] = df_ecg["milliSec"].astype(float)

    # Calculate ECG-HR
    resampled_hr, peak_indices, num_rr_intervals_short = get_ecg_hr_neurokit(df_ecg)
    peak_values = df_ecg["ECG_X"].iloc[peak_indices].values

    # Remove outliers if checkbox is selected
    if 'remove' in outlier_toggle:
        df = df[(df["HeartRate"] >= 30) & (df["HeartRate"] <= 220)]
        resampled_hr = resampled_hr[(resampled_hr >= 30) & (resampled_hr <= 220)]

    # Load selected resampled biopac HR
    df_resampled_biopac = pd.DataFrame()
    if selected_biopac_file:
        path_bio = os.path.join(RESAMPLED_FOLDER, selected_biopac_file)
        if os.path.exists(path_bio):
            df_resampled_biopac = pd.read_csv(path_bio)
            if "time" in df_resampled_biopac.columns:
                df_resampled_biopac["time"] = pd.to_datetime(df_resampled_biopac["time"], errors='coerce')
            # Sort by milliSec if available
            if "milliSec" in df_resampled_biopac.columns:
                df_resampled_biopac = df_resampled_biopac.sort_values("milliSec")

    # Load selected resampled watch HR
    df_resampled_watch = pd.DataFrame()
    if selected_watch_file:
        path_watch = os.path.join(RESAMPLED_FOLDER, selected_watch_file)
        if os.path.exists(path_watch) and os.path.getsize(path_watch) > 0:
            df_resampled_watch = pd.read_csv(path_watch)
            df_resampled_watch["Timestamp_pd"] = pd.to_datetime(df_resampled_watch["Timestamp_pd"], errors='coerce')
            df_resampled_watch["HeartRate"] = pd.to_numeric(df_resampled_watch["HeartRate"], errors='coerce').fillna(-1)
            # Sort by Timestamp_pd
            df_resampled_watch = df_resampled_watch.sort_values("Timestamp_pd")

    # === Build subplots (6 rows) ===
    fig = make_subplots(
        rows=6, cols=1,
        subplot_titles=(
            "Heart rate plot from Galaxy watch",
            "ECG in mV from biopac",
            f"HR from ECG biopac calculated (ours) | Num RR <250 ms = {num_rr_intervals_short}",
            "HR from ECG provided from biopac",
            "Resampled Heart Rate - Biopac",
            "Resampled Heart Rate - Smartwatch"
        ),
        vertical_spacing=0.05
    )

    # Row 1: Galaxy watch HR
    fig.add_trace(go.Scatter(
        x=df["Timestamp_pd"], y=df["HeartRate"], mode='lines', name='Watch Heart Rate', line=dict(color='blue')
    ), row=1, col=1)

    # Restore activity annotations on smartwatch HR plot
    for label, group in df.groupby("manual_labels_numeric"):
        if pd.isna(label):
            continue
        start_time = group["Timestamp_pd"].min()
        end_time = group["Timestamp_pd"].max()
        mid_time = start_time + (end_time - start_time) / 2

        # Vertical boundaries
        fig.add_vline(x=start_time, line=dict(color='green', dash='dot'), row=1, col=1)
        fig.add_vline(x=end_time, line=dict(color='red', dash='dot'), row=1, col=1)

        # Numeric annotation above plot
        y_max = max(df["HeartRate"]) if not df["HeartRate"].empty else 0
        fig.add_annotation(
            x=mid_time,
            y=y_max + 5,
            text=str(int(label)),
            showarrow=False,
            yanchor='bottom',
            font=dict(size=34),
            bgcolor="white",
            opacity=0.8,
            xref="x1",
            yref="y1"
        )

    # Row 2: ECG waveform and R-peaks
    fig.add_trace(go.Scatter(y=df_ecg["ECG_X"], mode='lines', name='ECG', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=peak_indices, y=peak_values, mode='markers', name='R-peaks',
                             marker=dict(color='red', size=8, symbol='x')), row=2, col=1)

    # Row 3: ECG-HR from Neurokit
    fig.add_trace(go.Scatter(y=resampled_hr, mode='lines', name='ECG-HR (Neurokit)', line=dict(color='blue')), row=3, col=1)

    # Row 4: Biopac-provided HR
    fig.add_trace(go.Scatter(x=df_ecg.index, y=df_ecg["Heart_Rate"], mode='lines', name='Biopac Heart Rate',
                             line=dict(color='blue')), row=4, col=1)

    # Row 5: Resampled Biopac HR from output folder
    if not df_resampled_biopac.empty and "Heart_Rate" in df_resampled_biopac.columns:
        # Use 'time' column for x-axis if available
        x_data_biopac = df_resampled_biopac["time"] if "time" in df_resampled_biopac.columns else df_resampled_biopac.index
        fig.add_trace(go.Scatter(
            x=x_data_biopac, y=df_resampled_biopac["Heart_Rate"], mode='lines',
            name='Resampled Biopac Heart Rate', line=dict(color='purple')
        ), row=5, col=1)
        
        # Add activity vertical lines for biopac resampled data
        if "activity" in df_resampled_biopac.columns:
            for activity, group in df_resampled_biopac.groupby("activity"):
                if pd.isna(activity):
                    continue
                if "time" in df_resampled_biopac.columns:
                    start_time = group["time"].min()
                    end_time = group["time"].max()
                else:
                    start_time = group.index.min()
                    end_time = group.index.max()
                
                # Add vertical dashed lines for activity start and end
                fig.add_vline(x=start_time, line=dict(color='green', dash='dash'), row=5, col=1)
                fig.add_vline(x=end_time, line=dict(color='red', dash='dash'), row=5, col=1)
    else:
        fig.add_annotation(text="No Biopac Resampled HR Data Available",
                           xref="x5", yref="y5", showarrow=False, row=5, col=1)

    # Row 6: Resampled Watch HR from output folder
    if not df_resampled_watch.empty and "HeartRate" in df_resampled_watch.columns:
        fig.add_trace(go.Scatter(
            x=df_resampled_watch["Timestamp_pd"], y=df_resampled_watch["HeartRate"], mode='lines',
            name='Resampled Watch Heart Rate', line=dict(color='orange')
        ), row=6, col=1)
        
        # Add activity vertical lines for smartwatch resampled data
        if "manual_labels_activity" in df_resampled_watch.columns:
            for activity, group in df_resampled_watch.groupby("manual_labels_activity"):
                if pd.isna(activity):
                    continue
                start_time = group["Timestamp_pd"].min()
                end_time = group["Timestamp_pd"].max()
                
                # Add vertical dashed lines for activity start and end
                fig.add_vline(x=start_time, line=dict(color='green', dash='dash'), row=6, col=1)
                fig.add_vline(x=end_time, line=dict(color='red', dash='dash'), row=6, col=1)
    else:
        fig.add_annotation(text="No Smartwatch Resampled HR Data Available",
                           xref="x6", yref="y6", showarrow=False, row=6, col=1)

    # Layout
    fig.update_layout(height=2000, width=2000, template="plotly_white", showlegend=True)

    # Axes labels
    for r in range(1, 7):
        fig.update_xaxes(title_text="Time", row=r, col=1, showline=True, tickfont=dict(size=12))
        fig.update_yaxes(title_text="Heart Rate (bpm)" if r != 2 else "ECG (mV)", row=r, col=1, showline=True, tickfont=dict(size=12))

    return fig


# === Run app ===
if __name__ == '__main__':
    app.run(debug=True, port=8051)
