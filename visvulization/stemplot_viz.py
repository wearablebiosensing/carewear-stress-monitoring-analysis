import os
import glob
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# --- Tkinter folder picker ---
import tkinter as tk
from tkinter import filedialog

def pick_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory(title="Select Folder Containing heart_rate_P*.csv Files")
    root.destroy()
    return folder_selected

DATA_FOLDER = pick_folder()
if not DATA_FOLDER:
    print("No folder selected. Exiting.")
    exit()

# --- Dash App ---
def get_participant_options(data_folder):
    csv_files = glob.glob(os.path.join(data_folder, "heart_rate_P*.csv"))
    options = [
        {
            "label": os.path.basename(f),
            "value": f
        }
        for f in sorted(csv_files)
    ]
    return options

participant_options = get_participant_options(DATA_FOLDER)
default_file = participant_options[0]["value"] if participant_options else None

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Heart Rate Discrete-Time Signal Viewer"),
    html.P(f"Data folder: {DATA_FOLDER}"),
    dcc.Dropdown(
        id='participant-dropdown',
        options=participant_options,
        value=default_file,
        clearable=False,
        placeholder="Select a participant file"
    ),
    dcc.Graph(id='heart-rate-graph')
])

@app.callback(
    Output('heart-rate-graph', 'figure'),
    Input('participant-dropdown', 'value')
)
def update_graph(selected_file):
    if not selected_file or not os.path.isfile(selected_file):
        return go.Figure()
    df = pd.read_csv(selected_file)
    # Adjust column names if necessary
    x = df['Timestamp']
    y = df['HeartRate']

    fig = go.Figure()
    for xi, yi in zip(x, y):
        fig.add_trace(go.Scatter(x=[xi, xi], y=[0, yi], mode='lines', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='blue', size=8)))
    fig.update_layout(
        title=f'Heart Rate Signal: {os.path.basename(selected_file)}',
        xaxis_title='Timestamp',
        yaxis_title='HeartRate',
        showlegend=False
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True,port=8055)
